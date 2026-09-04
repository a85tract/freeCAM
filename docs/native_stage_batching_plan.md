# FreeCAM Native Stage 批量执行与可替换 Kernel 分段运行计划

分支：`native-stage-batching`（自 `standalone-physics-function` 建立，2026-09-04）。
本文件是任务的计划书，逐条对应实现与验证；历史性能证据见
`validation/performance_overhead.md`，不得被覆盖。

## Summary

保留现有 Python class 和 kernel 替换接口，但停止使用 Python 逐个调度几百个 Fortran helper/kernel。

新增三种内部执行模式：

- **native-whole**：没有 kernel 被替换，Python 只调用一次原始 Fortran stage。
- **segmented**：存在 Python/AI kernel，native runner 运行到替换点后返回 Python，Python 执行模型，
  native runner 从断点继续。
- **legacy-python**：保留当前逐调用实现，仅用于调试、BFB 对照和性能回归。

严格遵守已选择的限制：

- 不允许 Fortran 回调 Python。
- 不使用 c_funptr 或 ctypes.CFUNCTYPE。
- 所有执行都由 Python 主动调用 start() 或 resume()。
- 只有实际被替换的 kernel 会让 native runner 返回 Python。

第一批完整覆盖当前四个可替换边界：`mmacro_pcond`、`micro_mg_tend`、`rad_rrtmg_sw`、`rad_rrtmg_lw`。

UI 工作继续暂停，不修改未提交的 `workflow_builder/`。

## Public API 与运行语义

保持现有用户接口不变：

```python
stage = fc.CloudMacroMicrophysics()
stage.kernels["mmacro_pcond"] = None      # None 代表使用原始 Fortran kernel
stage.kernels["mmacro_pcond"] = model     # 替换成 Python 或 AI 模型
stage.attach(driver)
stage.kernel = model                      # 单 kernel 的简写仍可用
stage.kernels["micro_mg_tend"] = model
```

内部新增只读诊断信息：

```python
stage.execution.mode        # "native-whole" / "segmented" / "legacy-python"
stage.execution.describe()  # mode、active replacements、native segment calls、
                            # Python model calls、Python/Fortran crossings
```

执行规则：

- 所有 kernel 均为 None 时，class 仍存在，但 run() 直接执行原始完整 Fortran stage。
- 某个 kernel 被替换时，执行计划只在该 kernel 前停止。
- 未替换的相邻 kernel、handle 调用、copy 和 history 写入在 native runner 内连续执行。
- Python 模型继续接收与当前接口相同的字段名称、shape 和输出契约。
- replacement 输入只包含有效 ncol，padding lane 不交给模型；结果复制回原存储后再恢复 native runner。
- kernel 赋值、reload 或移除只能在完整 action 边界生效；当前 stage 正在暂停时禁止修改执行计划。
- 模型异常或输出契约错误时销毁暂停 context 并将当前模型标记为 tainted；因为该 stage 是非事务性的，
  不伪装成可以回滚。

## Implementation Changes

### 1. 先恢复无替换场景的性能

- 扩展 NativeAccess，允许 Python stage 直接按原始 action ID 调用 backend native primitive，
  不能再次经过 workflow 分派，避免递归。
- NativeStage 每次运行前根据四个 kernel slot 生成轻量 fingerprint。
- fingerprint 中没有 replacement 时选择 native-whole：不进入 tend_chunk()，不创建 kernel 字典、
  scratch copy 或逐字段 view，不执行数百次 ctypes 调用，直接调用未修改的原始 Fortran stage。
- 当前 Python 逐语句实现保留为 legacy-python，用于比较，不能再作为默认路径。

### 2. 建立可暂停的 native segment runner

统一 ABI：

```
stage_context_create(stage_id, config, status) -> context_id
stage_run(context_id, replacement_mask, dynamic_scalars, event, status)
stage_resume(context_id, completed_kernel_id, event, status)
stage_kernel_frame(context_id, kernel_id, pointers, ndims, shapes, intents, status)
stage_context_reset(context_id)
stage_context_destroy(context_id)
```

event 只能返回 `DONE`、`NEEDS_PYTHON_KERNEL`、`ERROR`。

运行流程：Python 调用 stage_run() → native runner 连续执行原生操作 → 遇到未替换 kernel 直接调用并继续 →
遇到 replacement 保存 program counter、chunk 和 substep 并返回 NEEDS_PYTHON_KERNEL → Python 取得
kernel frame 并运行模型 → Python 写回输出并调用 stage_resume() → native runner 继续执行。

每个 MPI rank 拥有独立的 rank-local context；不通过 MPI 传输模型数组。

### 3. 用声明式 StageProgram 替代热路径中的 Python 语句调度

新增内部 StageProgramSpec，支持 `NativeCall`、`Copy`、`Fill`、`HistoryWrite`、`Loop`、
`ConstantBranch`、`ModelBoundary`。

- 当前 SEQUENCE 继续作为顺序审计依据，但扩展为包含参数绑定的完整 program spec。
- 配置相关分支在初始化后解析一次，生成线性执行计划。
- dt、nstep、lchnk 和 ncol 作为每步动态标量传入。
- 将 Python stage 里剩余的 NumPy 浮点运算提升为原生 direct kernel，native runner 本身不重新实现科学公式。
- Python class 只负责声明计划和 replacement，不再在每一步构造字典或执行数值语句。
- program 在每个 rank 初始化一次，并按 library hash + stage type + configuration digest 缓存。

### 4. 统一 Fortran dispatch 与内存绑定

- 为 direct kernel、host service 和 history service 生成统一的 bind(C) dispatcher，以整数 operation ID
  调用现有 Fortran 实现。
- runner 中的 select case(operation_id) 由现有 descriptor 和新的 stage program manifest 自动生成，
  禁止手写重复参数表。
- 每个 stage/chunk 开始时通过一次 bulk resolver 取得需要的 Fortran 存储地址、shape 和 dtype；
  不再逐字段调用 _deref()。
- scratch、pbuf、physics state 和 cam_in/cam_out 使用整数 slot ID 绑定；地址改变时只更新 slot table。
- kernel 参数表、shape 表、字符串和 history 字段名全部预绑定，只在存储地址或配置改变时重建。
- runner 内部连续完成 copy、native calls 和 outfld，不为每个操作返回 Python。
- 原始完整 Fortran stage 保持不修改，专门服务 native-whole 路径。

### 5. Replacement frame 与生命周期

- ModelBoundary 保存 kernel ID、program counter、chunk/substep index、输入输出 slot、ncol、
  dtype、shape 和 intent。
- Python 根据 frame 生成当前接口所需的 batch mapping。
- 默认复制有效输入给用户模型，避免模型保留 native 临时指针；模型输出通过 np.copyto 写回 context 输出 slot。
- shape、dtype、必需输出和有限生命周期在 resume() 前验证。
- 一个 stage 允许依次经过多个 replacement；每次只保存一个活动 boundary。
- stage 成功完成后 context 回到 idle；checkpoint、history flush、workflow 修改和 finalize 只允许在
  idle 状态执行。
- finalize 或异常必须释放所有 native context 和临时数组。
- KernelSlots 替换当前普通字典实现，但保持 `stage.kernels[name] = value` 语法，并维护 generation 计数
  用于计划失效。

### 6. 兼容与交付

测试专用选择：`--stage-execution=auto|native-whole|segmented|legacy-python`，默认 auto
（无 replacement → native-whole；有 replacement → segmented）。

现有 `--cloud-macro-micro-python` 继续可用，但改为启用 Python class 接口，不再强制逐 kernel Python 调度。
保留现有 whole-driver 和 legacy 验证入口，等新路径全部通过后再标记 deprecated，不立即删除。

提交机器可读指标：

```json
{"execution_mode": "native-whole", "native_stage_calls": 1488, "native_segment_calls": 0,
 "python_model_calls": 0, "python_fortran_crossings_per_step": 1}
```

更新性能文档时同时保留旧的 475.53 秒结果，不能覆盖历史证据。

## Test Plan

### 单元和 ABI 测试

- 无 replacement 时只发生一次完整 native stage 调用，tend_chunk() 调用数必须为零。
- replacement fingerprint 变化后正确切换为 segmented；移除 replacement 后恢复 native-whole。
- runner 只在指定 kernel 返回，其他 kernel 连续执行。
- 多 replacement 顺序、chunk 循环和 substep 循环的 program counter 正确。
- kernel frame 的名称、shape、dtype、intent 和有效 ncol 与现有契约一致。
- 模型缺字段、返回错误 shape/dtype、抛异常和 resume 顺序错误均能清理 context。
- storage 地址变化时 slot table 更新；地址不变时不重新绑定。
- 静态测试禁止 Fortran→Python callback 符号和 callback 注册路径。
- fake dispatcher 测试可在 GitHub Actions 运行，不依赖 Derecho、MPI 或真实 CAM 库。

### BFB 验证

1. 单 rank synthetic ABI：legacy、native-whole 和 segmented-original 逐数组完全一致。
2. 512-rank、50-step：class 启用但无 replacement；mmacro_pcond 在 segmented 边界调用原始 kernel；
   micro_mg_tend 同样验证；两个 RRTMG kernel 分别验证；四个边界组合验证。
3. 每个测试比较全部 StatePool 字段、CAM history 和 restart，不使用容差。
4. 一个月 1488 步验证全部 18 个 history/restart 文件 BFB。
5. 月度通过后运行一年，比较全部 180 个文件 BFB。

replacement 调用原始 kernel 仍不 BFB 时，不允许以"模型本来就会改变结果"为理由跳过；这代表 segment
前后状态没有正确保存。

### 性能验收

使用与已有结果相同的 512 ranks、4 节点、输入、编译器、rank placement 和无 profiler 配置，至少运行三次并报告中位数：

| 路径 | 当前基线 | 验收目标 |
| --- | ---: | --- |
| 普通 FreeCAM 一月 | 402.51 s | 对照 |
| 当前细粒度 Python stage | 475.53 s | 历史对照 |
| 新 native-whole | — | 不超过普通 FreeCAM 5% |
| 原始 Fortran stage | 37.68 ms/step/rank | 对照 |
| 当前 Python stage | 92.13 ms/step/rank | 历史对照 |
| 新 native-whole stage | — | 不超过 48.6 ms，消除至少 80% 的额外 stage 时间 |
| segmented-original | — | 相对当前 Python stage 至少消除 50% 的额外 stage 时间 |

同时报告：native segment 调用数、Python model 调用数、bulk pointer resolve 次数、copy 字节数、
每步 Python/Fortran crossings、MPI 最慢 rank 与平均 rank 时间、BFB 结果。
性能比较只使用模型推进区间，不把 PBS 排队、启动和文件准备时间混入结果。

## Assumptions and Defaults

- 从当前 standalone-physics-function 分支建立独立 native-stage-batching 分支。
- 未提交的 UI 目录和 UI 实验不进入本任务。
- Python 继续拥有 workflow 和 replacement 选择；native runner 只执行 Python 已经编译好的 stage program。
- 原始无 replacement 路径优先调用未修改的完整 Fortran stage，而不是强迫它经过 segment state machine。
- Python/AI replacement 允许比原始 kernel 慢，但框架调度成本必须与模型计算成本分别报告。
- 第一版只保证当前 PI-CAM 配置和四个现有 SWAPPABLE 边界；其他配置必须明确报 unsupported。
- 不改变科学公式、浮点操作顺序、MPI communicator 或 rank placement。
- 只有全部 50-step、月度 BFB 和性能 gate 通过后，auto 才切换为新默认路径。
