# FreeCAM 性能优化计划：保持现有 UI 和 BFB，超过原始 Fortran 至少 5%

分支：`native-stage-batching`（2026-09-04 起）。本文件是任务的计划书（v3，取代
`native_stage_batching_plan.md` 作为总目标；后者的 native-whole / segmented / legacy-python 三模式
与 runner 设计仍然有效，是本计划第 3 阶段的内容）。历史性能证据见 `validation/performance_overhead.md`，
不得被覆盖。

## 1. 目标与验收标准

保留当前 Python class、workflow 编辑、kernel 替换、动态变量、动态 Python 函数、reload、逐步运行和绘图
接口。所有 Python 函数继续由 Python 主动执行，Fortran 通过返回状态交还控制权。

最终目标：同样资源、同样 online 耦合、同样数值结果和输出，FreeCAM 的完整模型推进时间比原始 Fortran
至少减少 5%。

验收分别覆盖：一个月（1488 coupling steps）；一年（17520 coupling steps）；当前 PI-atm 配置，512 MPI
ranks、4 节点；全部活动 component 正常计算，使用真实 online x2a/a2x；history/restart 的变量、频率、精度
和压缩设置相同；原始 kernel 路径保持 BFB；保留全部现有用户功能，不通过减少科学计算或输出达到目标。

主要性能门槛：`median(FreeCAM 推进耗时 / 原始 Fortran 推进耗时) ≤ 0.95`。同时要求完整生命周期耗时不出现
回退，并报告峰值内存。

现有的"相对普通 FreeCAM 额外开销 ≤5%"仅作为历史阶段指标，不再代表最终任务完成。native-whole 的月度结果
已经验证；分段 runner 正在开发，需要接续现有工作。

## 2. 建立可信基线与性能预算

### 三组对照

| 组别 | 控制层 | 原生实现 | 用途 |
| --- | --- | --- | --- |
| A | 原始 Fortran driver | 固定版本原始实现 | 最终目标基线 |
| B | 原始 Fortran driver | 优化后的原生实现 | 测量原生优化收益 |
| C | FreeCAM Python driver/class | 与 B 相同的原生实现 | 测量完整 FreeCAM 性能 |

最终比较 C/A；同时报告 B/A 和 C/B，区分原生优化收益与 Python 框架成本。可独立使用的原生优化必须同时用于
B 和 C。

### 固定运行条件

- 固定源码、编译器、优化选项、MPI 库、输入文件、namelist 和 native library hash。
- 固定 rank-to-node、CPU affinity、OpenMP/数学库线程数。
- 核对现有 PBS 请求与实际 CPU 绑定，避免把资源布局变化误算成代码收益。
- 路径、账户和队列从现有 site 配置读取，新增脚本不写入个人路径或账户。
- 性能运行关闭细粒度 profiler；诊断运行单独执行。
- 同一组配对测试尽量在同一个 allocation 中顺序运行，交替 A/C 顺序，使用独立输出目录。

计时使用相同的完整 coupling-loop 边界，包含 CAM、其他 components、coupler、正常通信及循环内输出。初始化、
最终输出和清理另报生命周期时间。不能用 FreeCAM online 耗时与原始 Fortran 的单独 ATM timer 比较。

### 找出足够的收益

先做 50-step 正确性运行和 300-step 性能诊断，覆盖：Python workflow、adapter、参数绑定和 trace；CAM
physics、dynamics；online provider、coupler 和其他 components；MPI 状态通信及等待；原生
allocation/deallocation、数组复制和临时数组；history/restart 和目录切换。同时记录平均 rank 与最慢 rank。
优化优先级按完整推进时间中的可回收耗时排序，不能简单相加重叠 timer。

以历史一年结果估算，从约 5510 秒降到原始约 5000 秒的 95%，需要节省约 760 秒。该数字仅用于说明工作量，
正式预算以重新测得的 A/C 为准。

## 3. 实现路线

### A. 缓存 workflow 和 ABI 调用准备

增加内部执行缓存，保持公开 UI 不变。

- workflow 变更时建立执行列表，提前分类 native、Python callback、边界、时钟和 I/O。
- 将稳定 action 从通用 adapter.call() 改为预绑定调用，复用函数入口、指针表、shape、dtype 和错误缓冲。
- StatePool、workflow、kernel registry 分别维护变更版本。
- 数组数值的原地修改不触发重建；字段增删、地址变化、workflow 调整和 reload 使相关缓存失效。
- 缓存保留数组所有者引用，动态移除字段时同步释放相关绑定。
- 普通用户 callback 保留字段权限、返回值检查和事务性语义。

缓存只优化准备过程，不缓存随时间变化的计算结果，也不能跳过用户覆盖的 Python 方法。

### B. 完成原始过程与分段 runner

保留三种内部执行模式：无 replacement → native-whole；有 replacement → segmented；调试与对照 →
legacy-python。完成 mmacro_pcond、micro_mg_tend、rad_rrtmg_sw、rad_rrtmg_lw 四个边界。

- 从固定原始 Fortran 调用点切分控制，复用已有句柄和参数描述。
- 数值 kernel 保留原始实现；adapter 不复制数值算法。
- runner 保存跨暂停点存活的变量、chunk、substep 和执行位置。
- Python 使用 start/frame/resume 驱动；没有 Fortran→Python callback。
- 只在被替换的 kernel 前暂停，其余原生操作连续执行。
- 不同 rank 可以有不同的本地调用次数，因此不在每个暂停点增加 MPI collective。
- replacement 出错后禁止继续推进；允许显式关闭并释放资源。
- context 活跃时禁止更换绑定、checkpoint 或移动过程；完成后恢复正常操作。

原始 kernel 通过 replacement 边界执行也必须 BFB，作为暂停/恢复正确性的独立证明。

### C. 合并相邻原生 action

在 stage 分段通过验证后，将相同原则应用于 workflow：

- Python 根据当前 workflow 建立连续 native action 列表。
- native runner 按列表执行，并返回实际完成的 action、状态和必要计时。
- Python callback、运行时 Python 条件、字段观察、时钟和不能合并的 I/O 边界结束当前批次。
- 不重排 kernel，不跨越副作用或通信依赖。
- 用户单独调用某个 process 时仍只执行该 process。
- 修改 workflow 或 reload 后，下一 action 边界采用新计划。
- 每步采样和暂停语义保持一致，不跨用户要求观察的 step 批量推进。

这一步执行的是 Python 已确定的调用列表，不是把 Python class 重新翻译成 Fortran。

### D. 优化 MPI 控制通信

- 正常路径用预分配整数 buffer 做 Allreduce 检查错误标志。
- 只有发现错误时才收集字符串和 traceback。
- 删除重复验证之前，先证明相关操作具有相同 communicator 和一致执行顺序。
- 所有 rank 必须在进入下一项需要集体参与的计算前确认错误状态。
- 不随意删除同步点，不改变数值 reduction 的顺序和算法。
- 配置、payload hash 和 workflow 一致性检查移到安装或修改边界；每步只检查需要动态验证的状态。

分别测量消息数量、序列化成本和同步等待，确认收益出现在完整 step，而不是把等待移到其他 timer。

### E. 优化 Fortran 内部内存与重复工作

这是超越原始实现的重要阶段，首先处理源码中已经确认的候选：

**微物理 packed workspace。** micro_mg_cam 每次调用申请和释放大量 packed 数组。改为 model/rank 所有的
可复用 workspace：首次申请，容量不足时扩展；每次调用仍恢复原始初始化值；保持实际 mgncol、shape、leading
dimension 和有效列范围；不依赖上一次调用留下的值；model 关闭时统一释放。

**物理状态与 tendency 临时存储。** 针对 physics_state_copy、physics_ptend_init 及对应释放路径：分离存储
准备与数值初始化；复用分配好的存储，同时保留原始逐次初始化语义；审核 allocation 状态是否被其他逻辑读取，
不能直接让本来应失效的对象继续表现为有效；只删除经过依赖分析证明多余的复制；对必须隔离更新的 state_loc
保留私有存储。

**数据搬运和静态查询。** 缓存生命周期内稳定的字段索引、metadata 和映射查询；保留 MCT component/coupler
的布局差异，只有布局和所有权都已证明一致时才省略中转复制；无额外复制的 view 必须有明确寿命，不能让 Python
callback 保留即将复用的 scratch；online provider 的目录切换按连续调用区域合并，并确保各 component 的 I/O
仍写到正确目录。

每项优化使用独立源码补丁，同时构建 A/B/C 对照。原始源码和 oracle 输出保持只读。

如果上述收益不足，按实测最慢路径继续处理 dynamics、radiation 或 coupler 中的分配、复制、索引和可证明不变
的重复工作。浮点表达式、求和顺序及数学库调用保持不变；引入表达式调整的候选必须独立验证，失败即撤销。

### F. 压缩观测开销，保留结果

- trace 使用紧凑记录和批量转换，保留现有顺序、数量及保留范围。
- 图表数据按照用户注册的变量、统计和 step 采样，展示时再创建 UI 对象。
- 已要求的逐步采样、history 和 restart 不能省略。
- 性能门槛采用原始相同输出配置；额外 Notebook 观测的成本单独量化。

## 4. 验证与性能验收

### 单元和接口测试

覆盖预绑定失效、数组生命周期、workflow 重排、enable/disable、动态字段增删、callback 安装与 reload、单独
运行过程、逐步采样和分段异常清理。重点验证：修改后不会继续调用旧函数或旧数组地址；native 批次与逐 action
路径的实际执行顺序一致；callback 的读写权限和回滚保持原语义；单 rank 出错不会使其他 rank 卡在后续
collective；runner 成功、失败和关闭都能释放资源。

### 数值 gates（按顺序）

1. 真实 kernel 输入 capture/replay，逐数组字节比较。
2. 覆盖 ncol < pcols、不同 chunk 数、变化的 mgncol、多 substep、workspace 扩展以及 restart。
3. 每项会影响数值或调用顺序的修改执行 512-rank、50-step gate。
4. 四个 replacement 边界分别及组合执行原始 kernel，证明实际经过暂停路径并保持 BFB。
5. 一个月、再一年完整 online 验证。
6. 对比 CAM 全部 history/restart，并核对其他活动 component 输出和 coupling 边界；比较所有有定义的
   StatePool 数值，明确排除未初始化 padding。

每项记录实际执行次数、source/library hash、首个差异及覆盖范围。模型替换实验的科学结果单独评价，不用 AI
近似结果替代 BFB gate。

### 性能测试

- 月度至少五组 A/C 配对；年度至少三组 A/C 配对。
- B 组同步测量原生优化收益，并报告 C/B 框架开销。
- 不挑选最快一次，报告每次结果、配对比值中位数及置信区间。
- 月度、年度都要求中位比值 ≤0.95，且配对比值的 95% 置信区间上界 <1。
- 如果波动导致结论不清楚，按预定规则增加两组配对，不因某次结果不好而任意剔除样本。
- 只有调度故障、节点故障或输入配置不一致等可核查原因才能排除运行，并保留记录。

最终性能 gate 使用原始 kernels、Python class 接口启用的默认科学路径。任意用户 Python/AI 模型的运行速度取决
于其自身实现，不承诺所有 replacement 都快于原始 kernel。

内存同时验收：重复运行后 workspace 达到稳定容量，不随 step 持续增长；峰值不超过优化前对应 FreeCAM 配置
的 105%。

## 5. 执行顺序与交付

在当前 native-stage-batching 工作基础上推进，保留正在开发的 runner 和未提交 UI 内容。首先记录工作区与
已有验证结果，再按下列阶段提交：

1. online 配对基准、计时边界和性能预算。
2. workflow/ABI 缓存和 MPI 状态通信。
3. 四个 kernel 分段边界及原始 kernel BFB。
4. 相邻 native action 批量执行。
5. packed workspace、state/tendency 存储复用和数据搬运优化。
6. 完整月度、年度性能与 BFB 验收。
7. 默认执行策略、文档和维护脚本更新。

内部优化默认由框架选择，现有 Notebook 不需要新增性能参数。保留诊断路径以复现优化前行为。

新增 `validation/pi_cam_faster_than_fortran.json`，保存基线、构建信息、每次 PBS 运行、A/B/C 时间、内存、
BFB 和最终目标是否达成。更新 `validation/performance_overhead.md`，保留历史记录。

最终完成条件是月度和年度均达到至少 5% 的耗时下降，并通过科学和 UI 验证。如果尚未达到，继续依据剩余耗时
推进可验证优化；如果可行候选耗尽，则交付准确的收益分解和限制，明确标记目标未达成，不降低验收线或把阶段
结果当作完成。

## 6. 结果与状态（2026-09-04）

**目标未达成；已达到与原始 Fortran 持平，并按用户决定在此停止。**

### 配对测量（月度，online 耦合，同一 allocation，512 ranks）

| 作业 | 代码 | 顺序 | A 原始 Fortran | C freeCAM | C/A | BFB |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 7322199 | 改动前 | AC | 457.59 s | 491.14 s | 1.0733 | 是 |
| 7322349 | 改动前 | CA | 437.88 s | 476.80 s | 1.0889 | 是 |
| 7322467 | 改动后 `ed685d5` | AC | 437.47 s | 440.07 s | **1.0060** | 是 |
| 7322553 | 改动后 `f565b67` | CA | 438.30 s | 439.86 s | **1.0036** | 是 |

年度（17520 步，同一 allocation，A 先 C 后）：

| 作业 | 代码 | 顺序 | A 原始 Fortran | C freeCAM | C/A | BFB |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 7322841 | 改动后 `958bd60` | AC | 5067.39 s | 5130.82 s | **1.0125** | 是（180 个文件） |

记录在 `validation/pi_cam_faster_than_fortran.json`（每对含生命周期时间、内存、BFB、provider 的
collective 次数与代码提交）。B 组未运行：没有进入 Fortran 侧的原生优化阶段。年度配对一对（用户决定持平即可之后补测）。

### 收益分解

- **CAM 本身不慢。** 改动前 C 的 CAM action 平均 376 s，A 的 `CPL:ATM_RUN` 平均 382 s；Python 控制层
  每步约 2.6 ms（≈1%，见 native-whole 月 7314664 的计时表）。第 3.A、3.F 条最多回收 1%，未做。
- **差距全在边界路径。** 改动前 C 的 import 71 s、export 44 s（A 对应约 48 s 与 27 s）。原因：online
  provider 在每个 coupler action 之后做一次 pickle allreduce（每步约 30 次）。原始驱动连续调用、各 rank
  跳过不属于自己的组件，LND（rank 0–255）、ICE（256–383）、OCN（384–415）本是并行的，被逐动作同步串行化，
  并叠加轮换 straggler 的等待。
- **第 3.D 条已实施**（提交 `3384292`）：step-begin 组、ATM iteration（含完成投票）、closing 组各一次
  2 整数 `Allreduce`；driver 的边界 collective 改为整数标志、仅出错时收集 traceback；import 与 schedule
  合并。每步 5 次 reduction。协议状态错误由全 rank 一致报告；组内 rank 本地失败按 `shr_sys_abort` 语义
  abort，不让其他 rank 卡在后续 collective。512 rank 在线 50 步 gate 7322441 BFB。
- **效果**：边界路径 115 s → 61 s，C/A 中位数 1.081 → 1.005（改动后两对：1.0060、1.0036）。

### 剩余预算（步内 perf，作业 7322501，跳过初始化）

| 份额 | rank 100 | rank 400 |
| --- | ---: | ---: |
| CAM（`libfreecam_pi_cam.so`） | 67.7% | 66.3% |
| MPI（libmpi + libfabric，含等待） | 9.4% | 12.3% |
| Intel 数学库 | 7.4% | 7.8% |
| libc（主要是 progress engine 的 `sched_yield`） | 6.8% | 8.1% |
| Python | 4.2% | 4.1% |
| CAM 内 `memcpy` / `memset` | 4.2% / 3.7% | 4.1% / 3.2% |

页错误中位数每 rank 每步约 52 次，`malloc`/`free` 不在前列：分配本身不是成本，第 3.E 条的可回收部分只有
复制与置零（约 7%）以及边界路径剩余的约 16 s/月（3.5%）。dwarf 调用链无法穿过固定地址镜像展开，复制的
调用者未能定位。要再快 5.5% 需要同时拿到这两部分的大半，在用户决定持平即可之后未再推进。

### 交付物

- 分段 runner 的原始 kernel BFB gate（7322256）、grouped collectives gate（7322441）、配对记录、perf 记录。
- segmented 路径的实测成本：原始 kernel 经 Python 回答时 stage 65 ms/step/rank（gate 7322256），native-whole 39 ms，legacy-python walk 92 ms。真模型（`mmacro_pcond_soft_gated.pt`）经 runner 运行时
  与 legacy walk 一样在首次写 history 时被 PIO 拒绝（作业 7324422，约 100 步，285 GB），模型本身不成立。
- 诊断工具：`validation/jobs/pi_cam_perf_online_50step.pbs`、`tools/perf_rank_wrapper.sh`、
  `tools/report_pi_cam_perf.py`；配对作业支持 `PYCAM_PAIR_DURATION=1year`。
