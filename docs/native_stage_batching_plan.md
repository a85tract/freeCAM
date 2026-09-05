# FreeCAM Python Process 性能重构计划（v2）

分支：`native-stage-batching`（自 `standalone-physics-function`，2026-09-04）。本文件是任务的计划书；
历史性能证据见 `validation/performance_overhead.md`，不得被覆盖。v2 取代了同名 v1：分段执行改为
给原始 Fortran 调用点生成 start/resume 接口，不再把 Python class 翻译回 Fortran。

## Summary

保留现有 Python process class 和 kernel 替换接口，但改变生产执行方式：

- Python class 不再被重新生成或编译成 Fortran。
- 没有 kernel 替换时，直接一次调用原始完整 Fortran process。
- 有 Python/AI kernel 替换时，原始 Fortran 运行到替换点后返回 Python；Python 执行替代 kernel，
  再调用 resume() 让 Fortran 从原位置继续。
- 不允许 Fortran 回调 Python，不使用 c_funptr 或 ctypes.CFUNCTYPE。
- 当前逐句翻译的 Python 实现继续保留，作为可读参考、调试工具和 BFB oracle，不再作为默认热路径。
- UI 和未提交的 workflow_builder/ 不属于本任务。

状态：native-whole 已实现并通过一个月、18 个 history/restart 文件 BFB；424.30 s，比普通 FreeCAM
的 402.51 s 慢约 5.4%。剩余核心任务是 segmented replacement 和最后的 wrapper 开销优化。

状态更新（2026-09-04）：trusted native 路径下 native-whole 三次月运行中位数 404.61 s，与普通
FreeCAM 持平（+0.5%）。segmented-original 已通过 512 rank、50 step gate（作业 7322256）：
mmacro_pcond 被替换为「原始 kernel 经 Python 边界执行」，runner 每步暂停 2 次（每 chunk 一次），
50 步共 150 次 segment 调用、100 次 Python 模型调用，4 个 history/restart 文件全部 BFB
（`validation/pi_cam_stage7_segmented_original_50step.json` 及其 `_vs_oracle_50step_bfb.json`）。

状态更新（2026-09-04，晚）：交付顺序第 8 条已落地——auto 策略在「有替换且镜像的 runner 覆盖全部被替换 kernel」
时选 segmented，runner 不覆盖的替换仍走 legacy-python。同时修正一个静默回退：按路径配置的 surrogate 原先在
首次调用时才加载，选路径时槽位为空，一次 surrogate 月因此以原始 Fortran 跑完并报告 BFB（作业 7322838）。
现在槽位在构造时就由未加载的 `PendingSurrogate` 占住，`tend` 对「被告知替换但槽位未体现」的情况直接报错。

## 1. 执行语义

保持现有接口：

```python
stage.kernels["mmacro_pcond"] = None
stage.kernels["mmacro_pcond"] = model
stage.kernels["micro_mg_tend"] = model
stage.kernels["rad_rrtmg_sw"] = model
stage.kernels["rad_rrtmg_lw"] = model
```

内部三种模式：

- **native-whole**：所有 replacement 都是 None；Python class 每步只调用一次原始完整 Fortran process。
- **segmented**：至少有一个 kernel 被替换；Fortran 只在实际替换点暂停并返回 Python；未替换的 helper、
  kernel、循环和数值操作继续在原始 Fortran 中连续执行。
- **legacy-python**：当前逐语句 Python 翻译；只用于调试、调用顺序检查和性能对照。

默认 auto：没有 replacement → native-whole；存在 replacement → segmented。

Python 仍控制 workflow 顺序、process 启用状态以及 replacement 选择；process 内部未被替换的高频循环和
helper 调度留在 Fortran，避免数万次 Python/Fortran 跨界。

## 2. 实现方案

### 2.1 完成 native-whole 快速路径

优化内置 NativeStage 经过 Python process registry 时的额外开销：

- 给内置 stage process 增加可信原生标记，和普通 Notebook Python process 分开处理。
- native-whole 不创建 PythonFieldView，不执行字段 snapshot。
- 每步不扫描全部 StatePool 指针；指针稳定性检查改到初始化、字段注册变化和 debug 模式。
- 合并重复的 MPI 错误收集，只保留一次 collective 状态检查。
- native.run_action() 直接调用 backend primitive，不重新进入 workflow dispatcher，避免递归。

### 2.2 为原始 Fortran 增加 start/resume 接口

从固定版本的原始 Fortran call site 生成并人工审核分段代码，不从 Python class 反向生成 Fortran。

统一内部 ABI：

```
stage_context_create(stage_id) -> context_id
stage_start(context_id, replacement_mask) -> event
stage_frame(context_id) -> kernel arguments
stage_resume(context_id, completed_kernel_id) -> event
stage_context_destroy(context_id)
```

事件只有 `DONE`、`NEEDS_PYTHON_KERNEL`、`ERROR`。

执行过程：Python stage.run() → stage_start() → Fortran 连续运行 → 遇到被替换的 kernel → 保存当前位置和
live state → 返回 NEEDS_PYTHON_KERNEL → Python 执行 model → 检查并写回结果 → stage_resume() →
Fortran 从原位置继续。

### 2.3 保存 Fortran 暂停状态

普通 Fortran 局部变量在 subroutine 返回后不会保留，因此每个 MPI rank 建立独立的 rank-local context，保存：
program counter；当前 chunk、lchnk、ncol；substep 和 kernel 调用序号；replacement mask；需要跨暂停点存活
的标量；自动数组或临时 tendency；指向 CAM module/derived-type storage 的稳定 handle。

生成工具对原始调用点做 live-variable 分析并生成 scaffold；每个边界的 live-state 清单必须人工审核。
原始源码 hash 或调用点 anchor 变化时构建直接失败，不能静默使用旧 adapter。

### 2.4 支持四个替换边界

第一版完整支持 `mmacro_pcond`、`micro_mg_tend`、`rad_rrtmg_sw`、`rad_rrtmg_lw`。

处理嵌套调用：cloud stage context 保存外层 tphysbc 和 substep 状态；macrophysics/microphysics 子 context
暂停在对应核心前；子 context 返回 replacement event 时，外层 context 同时保存位置并返回 Python；
resume() 先恢复子过程，再恢复外层过程；radiation 使用同一个 context 依次支持 SW 和 LW 两个暂停点。

每次原始 kernel 调用最多产生一次 pause/resume。第一版不跨 chunk 或 substep 重新排序调用，
避免改变浮点和 history 顺序。

### 2.5 Python replacement frame

stage_frame() 返回：kernel_id、call_index、lchnk、ncol、substep、argument names、pointers、shape、
dtype、intent。

Python 沿用当前模型接口：只把有效的 ncol 交给模型，不暴露 padding lane；输入按照现有契约生成 batch
mapping；输出必须包含所有 required fields；shape、dtype 和字段名称必须完全匹配；使用
`np.copyto(..., casting="no")` 写回原生 context；写回完成后才允许调用 resume()。

模型异常或返回非法结果时：销毁当前 context；将模型标记为 tainted；禁止继续 step、checkpoint 或
finalize；不宣称能够回滚已经执行过的非事务性 Fortran 操作。

### 2.6 生命周期限制

只有 context 为 idle 时才允许：更换或移除 replacement；修改 workflow；checkpoint/restart；history flush；
finalize。每个 context 带 generation 和调用 token，拒绝重复 resume、错误 kernel resume 或旧 frame 写回。

## 3. 测试与验证

### 单元测试

- 无 replacement 时每步只调用一次原始 stage，tend_chunk() 调用次数为零。
- 增加 replacement 后 auto 切换到 segmented；移除后恢复 native-whole。
- runner 只在被替换的 kernel 前返回；未替换 kernel 不返回 Python。
- 多 replacement、chunk 和 substep 的暂停顺序正确。
- nested context 能正确向外传播事件并从原位置恢复。
- frame 的名称、shape、dtype、intent 和 ncol 与当前接口一致。
- 非法输出、异常、重复 resume 和 stale token 能安全清理。
- 静态检查确认不存在 Fortran→Python callback。
- GitHub Actions 使用 fake backend 验证状态机，不依赖 Derecho。

### BFB 验证（依次执行）

1. 单 rank synthetic：native-whole、legacy-python、segmented-original 全数组 bitwise identical。
2. 512 ranks、50 steps：mmacro_pcond 分段后仍调用原始 kernel；micro_mg_tend；rad_rrtmg_sw；
   rad_rrtmg_lw；四个边界同时启用。
3. 比较全部 StatePool、history 和 restart 数值，不使用容差。
4. 一个月 1488 steps，全部 18 个文件 BFB。
5. 月度通过后运行一年，全部 180 个文件 BFB。

"segmented 后仍调用原始 kernel"必须 BFB；如果不 BFB，说明暂停状态、调用顺序或写回位置有错误，
不能归因于模型替换。

### 性能 gates

相同编译器、512 ranks、4 节点、rank placement 和输入，各运行三次取中位数：

- native-whole：相对普通 FreeCAM 不超过 5%。
- native-whole stage：不超过 48.6 ms/step/rank。
- segmented-original：相对当前 92.13 ms 至少消除一半额外开销，目标不超过约 65 ms/step/rank。
- 分别报告：Python/Fortran crossings；pause/resume 次数；Python model 调用次数；pointer resolve 次数；
  copy 字节数；平均 rank 和最慢 rank 时间。

真实模型的推理时间和框架 pause/resume 开销必须分开报告。

## 4. 交付顺序

在当前 native-stage-batching 分支继续：

1. 精简 native-whole registry 和 MPI 检查，达到 5% gate。
2. 实现通用 context、event、frame ABI 和 fake runner。
3. 接入 mmacro_pcond，完成 50-step BFB。
4. 接入 micro_mg_tend，完成嵌套 context BFB。
5. 接入两个 RRTMG kernel。
6. 完成四边界组合、月度和年度 BFB。
7. 更新性能文档和机器可读 validation JSON。
8. 通过所有 gates 后才把 auto + replacement → segmented 设为默认。

每个阶段独立提交。保留当前历史性能结果，不覆盖旧数据；不提交或修改未跟踪的 workflow_builder/ UI 工作。

## Assumptions

- 原始 Fortran 是科学公式和浮点顺序的唯一 native source of truth。
- Python class 是公开 process 对象、replacement dispatcher，以及可执行的参考/debug 实现。
- 不把 Python class 翻译回 Fortran。
- 不要求每个内部 helper-level if 和循环都由解释执行的 Python 负责；否则当前 30% 左右的损耗无法根本消除。
- 第一版只覆盖当前 PI-CAM 配置和四个已声明的 swappable kernels；其他配置明确报 unsupported。
- 不改变 MPI communicator、rank placement、数值公式或 reduction 顺序。
