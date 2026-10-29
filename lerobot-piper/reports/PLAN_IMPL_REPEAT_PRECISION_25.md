# 实施计划 —— 重复精度封顶验证（25 次）的专用脚本

**状态：用户 2026-10-05 认可方向，并补两处订正（已并入第 3 节）；按 Native 方式零机时实施。不连接机械臂、不启动采集、不训练网络或 π。`motion_authorized: false` 保持。**
- 依据：冻结的实验设计 `reports/PLAN_REPEAT_PRECISION_CAPPED.md`（76785f5）。本计划只实现它，不新增实验问题、条件或判据。
- 范围仅限：采集入口、单次复放执行器、分析脚本、测试。pilot 的文件不改；pilot 产物只读。

## 1 文件

| 文件 | 作用 |
|---|---|
| `sim/run_repeat_precision25.py` | 采集入口：冻结输入、账本、双上限、失败锁停、逐条完整性核验、A0 对齐 |
| `sim/replay_repeat_precision25.py` | 单次复放执行器（第 3 节） |
| `sim/analyze_repeat_precision25.py` | 分析与判定 |
| `sim/test_*_repeat_precision25.py` | 入口、执行器、分析、端到端四组测试（第 5 节） |
| `outputs/repeat_precision25/` | 从 pilot 复制并按哈希核对的冻结输入（A / P± / Q± 序列、发送计数、标定参考、audit）、manifest、账本、记录 |
| `reports/RELEASE_REPEAT_PRECISION_25.json` | 冻结时写入代码指纹，`motion_authorized: false`（第 6 节） |

## 2 采集入口

- **命令**：`--prepare`（离线复制并锁定输入、写 manifest）、`--dry-run`、`--status`、`--fingerprints`、`--start-session --operator-ready`、`--run-next --operator-ready`。
  - 没有 `--force`，没有重试、跳过或补跑。
  - 一次调用只执行固定顺序里的下一条。
- **顺序**：写死为方案第 4 节的 A0 + 4 个区组，共 25 条。每条的条件和符号取自 manifest，不从名字推断。
- **预算封顶**：
  - 每次尝试在连接设备**之前**记账，初始化失败也消耗一次。第 26 次一律拒绝。
  - `--start-session` 开始 50 分钟现场时钟，复位与等待都计入。
  - 剩余不足 60 秒时，拒绝开始下一条。
  - 复放进行中到「截止前 15 秒」即停止采集，转入受截止约束的归位（第 3 节），并记为失败。
  - 重启机器、删除或重建账本，都不能重置预算。
- **失败锁停**：以下任一情况都标记 `halted`。此后所有 `--run-next` 都拒绝，按方案记为「无法判定」。
  - 退出码非 0，或出现异常；
  - 预移动或稳定门禁不过；
  - 归位受阻，或归位途中到达截止时间；
  - 完整性核验不过。
- **记录完整性**：每条结束立即核验，不过即失败锁停；分析时再独立复核一遍。
  1. 记录哈希写入账本。
  2. 正式指令与冻结波形逐帧完全一致；换算出的发送计数与计划计数一致。
  3. 100 Hz 设备时间戳单调，最大间隔 ≤ 15 ms。
  4. 每个 20 Hz 样本都能在 100 Hz 流中找到相同的值。
  5. 写总线间隔 ≤ 100 ms。
  6. 另查：
     - 身份字段、帧数与有限值；
     - 前置保持记录齐全，且与正式帧分开；
     - 处理链的取样点按实际时间戳**全部**落在 100 Hz 覆盖范围内。不外推，不足即无效。
- **A0**：只用 A0 和冻结的标定参考确定对齐（与 pilot 同法）。对齐冻结之前，拒绝任何探针条。A0 不进任何指标。

## 3 执行器（用户已同意另写；两处订正已并入）

- `replay_probe.py` 的放行钩子写死为 `run_pilot_response14`，所以另写执行器。照搬 pilot 已实测 14 次的执行路径，只把授权核验换成本次入口。
  - 照搬的部分：预移动、位置带与到位门禁、采集循环，以及「清理时不额外发归位或禁能」。
  - 只从 `replay_probe` 导入**不发运动指令**的部分，并锁定其哈希：
    - `safety_checks`、`pilot_check_imu_freshness`；
    - 常量 `SAFE_DISABLE`、`RETURN_STAGES`、`RET_*`、`FPS`、`DT`、预留时间。
- **订正 1：归位也受截止时间约束。**
  - `staged_return` 会发送动作，且内部没有截止检查，不能当作纯函数导入。
  - 执行器自写受截止约束的分段归位：分段、速度、到位容差都与原函数相同；每次下发动作之前，都检查会话截止时间。
  - 到期即不再下发任何新运动指令，记录 `return_deadline` 与失败原因，入口锁停。不为赶时间加速归位。
  - 预采集阶段异常时走的归位路径，同样受此约束。
- **订正 2：0.2 秒前置保持。**
  - 作用：打开 100 Hz 记录后，以 20 Hz 发送 4 帧与稳定段**相同的保持目标**（目标不变），再开始第一个正式帧。
  - 它只增加覆盖余量，不保证覆盖。覆盖仍按实际时间戳检查，不足就拒绝，即判该条失败。
  - 前置保持单独记录在 `prehold` 字段（时刻、写总线时刻、指令、目标），不混入正式的 `cmd` / `t_send`。冻结波形和正式帧数（587）都不变。
  - 所有条件，包括 A0，都统一执行。
  - 依据：pilot 14 条记录里，首帧取样点之前的余量最小只有 3.2 ms，中位约 20 ms。记录队列无上限。

## 4 分析脚本

- **先核验**：
  - 代码指纹与 release 一致，不符即拒绝运行；
  - 25 条全部完成；记录哈希与账本一致；逐条有效性复核。
  - 后两项任一不满足 ⇒ 「无法判定」，只报告执行现状，不计算主指标。
- **主分析**：`precision_chain.frame_locked` → `P(·)`；A0 不参与。
  - 散布：σ_c（每个条件 df 3，合并 df 12）、σ_bare（df 7）。
  - 复现度：R_c 取 {1,3}｜{2,4} 和 {1,4}｜{2,3} 两种分法的平均。
- **分半基线不共享**：两半直接由区组表构造。断言两半复放集合不相交；每半恰为 2 次同条件 + 4 次裸复放；裸复放只来自本半的区组。
- **判定**：纯函数，只接收「执行是否完整」和四个 R_c。三种出口：
  - 四个都 ≥ 0.6 → 支持另行设计正式采集；
  - 平均 < 0.3 → 停止推进；
  - 其余 → 无法判定。任一 R_c 非有限也记为无法判定。
- **辅助报告**（方案第 6 节）：单独一节，在判定之后计算，不能流入判定。
  - κ、正负关系；
  - 分半尖峰匹配、门控 / 计数台阶同位；
  - 按实际时间戳的漂移与时刻差；
  - 旧 20 Hz 口径、帧锁定偏移 {0, 10, 20, 30, 40} ms、与 pilot 的跨日对比。
- **产物**：`outputs/repeat_precision25_analysis/analysis.json`。只含相对名与哈希，不含绝对路径；打印 sha16。

## 5 测试（全部离线；任何测试都不得到达真实设备工厂；产物只写临时目录）

- **入口**：
  - 移植 pilot 入口测试并改为 25 条；
  - 新增：每类完整性失败都锁停、第 26 次拒绝、截止的两道闸、CLI 无 `--force`、未授权时拒绝开始。
- **执行器**：
  - 移植 pilot 执行器的作用域核验与 11 种故障路径；
  - 新增：
    - **归位途中到期**：到期后零新指令、记录失败；预采集归位同样受约束；
    - 前置保持的目标、帧数、与正式帧分开、A0 同样执行；
    - 覆盖不足即拒绝。
- **分析**：
  - 分半不共享：集合断言；只扰动区组 2、4 的裸复放时，{1,3} 半不变；
  - 判定边界：恰为 0.6、恰为 0.3、非有限、不完整；
  - 篡改哈希或任一有效性失败 ⇒ 无法判定；
  - 辅助不改判定。
- **端到端**：
  - 假设备：100 Hz 陀螺带设备毫秒、主机抖动和逐次相位；20 Hz 取最新样本；提供写总线时刻；各条件的响应可设。配合假时钟，经入口 → 执行器 → 记录 → 分析跑满 25 条。
  - 场景：
    - 可复现响应 → 支持；
    - 无响应 → 停止；
    - P 有、Q 无 → 无法判定；
    - 第 k 条注入故障 → 锁停 → 无法判定；
    - 时钟触顶 → 拒绝下一条 → 无法判定。
  - 完整场景各跑两遍，分析产物哈希一致。

## 6 冻结与授权顺序

1. 实现完成，全部测试通过，两遍一致。
2. 离线执行 `--prepare`、`--dry-run`、`--status`：不连接设备，不开始现场时钟。
3. 写 `RELEASE_REPEAT_PRECISION_25.json`（`motion_authorized: false`），提交后即锁定代码指纹，交用户核验。
   - 指纹覆盖第 8 节 Task 1 列出的文件。
   - 入口和分析每次运行都核对指纹，不符即拒绝。
4. 用户单独确认这 25 次的运动授权后，再单独提交一次：把 `motion_authorized` 改为 true，并记录授权来源。现场手册也在这一步写。
   在此之前，入口拒绝执行 `--start-session`。

## 7 不做

- 不连接机械臂、不启动采集、不训练网络或 π。
- 不改 pilot 文件；不改方案的次数、预算、处理链和判据。
- 判定之外不加任何自动决策；结果报告在数据整理后另写。

## 8 实施任务（TDD：每步先写测试、看它失败，再实现）

通用约定：
- 测试命令都在 `lerobot-piper/` 下运行：`$PY -m pytest -q <测试文件>`，其中 `PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python`。
- 测试模块开头 `gc.disable()`（mujoco + scipy 的 GC 段错误）。
- 提交在 develop 上，只写明确路径，不加 Co-Authored-By。

### Task 1: 入口骨架——顺序、冻结输入、manifest、release 绑定

- 文件：`sim/run_repeat_precision25.py`；`sim/test_run_repeat_precision25.py`。
- 接口：
  - `expected_runs() -> list[dict]`，字段：`run_id`、`block`（A0 为 0）、`group`（A/P/Q）、`sign`、`condition`（A0 / bare / P+ / P- / Q+ / Q-）、`sequence`、`use_probe`。
  - `FROZEN_INPUTS`：9 个文件到 sha256 的映射，取自 pilot manifest，写死在代码里。
  - `CODE_FILES`：需要锁指纹的文件，包括：
    - 本次的三个脚本；
    - `precision_chain`、`q1_windows`、`q1_data`；
    - `analyze_pilot_response14`、`diag_pilot_response14_spikes`；
    - `replay_probe`、`analyze_probe_3a`、`make_probe_b2`、`compare_ideal_methods`、`ideal_imu_kinematic`、`piper_model.xml`；
    - `can_hs_recorder`、`b_runtime`、两个 lerobot piper 文件、`tsnd151_reader`。
  - 函数：`code_fingerprints()`、`prepare()`、`load_frozen(require_motion=True)`。
- RED 测试：
  - 顺序：25 条；4 个区组，每区组每个条件各一次；每个条件 4 次、裸 8 次；探针平均序号 14，裸平均序号 12.5。与方案第 4 节的表逐项相同。
  - `prepare()`：复制后哈希与 `FROZEN_INPUTS` 一致；源文件被篡改即拒绝；已存在不同的 manifest 时拒绝覆盖。
  - `load_frozen()`：拒绝以下情况——字节被改、路径逃逸、代码指纹不符；`require_motion=True` 而 `motion_authorized` 不为 true；序列不合法（负号序列 ≠ −正号序列、裸序列含探针、j4 非零、超过 10 mrad）。
- Expected：先 FAIL（模块不存在），实现后全部 PASS。
- 提交：`重复精度 25 次：入口骨架（顺序、冻结输入、manifest、release 绑定）与测试`。

### Task 2: 入口会话——预算、锁停、A0、放行核验

- 接口：`start_session(operator_ready)`、`read_session()`、`validate_replay_context(context, seqfile, run_id, cmd7, da, use_probe)`、`freeze_a0()`、`run_next(operator_ready, replay_fn=None)`、`main()`。
  - 其中 `replay_fn(run=, seqfile=, outdir=, context=) -> int`。
- RED 测试：移植 pilot 入口测试并改为 25 条，再加以下新测试。
  - 未授权（`motion_authorized: false`）时，`--start-session` 拒绝且不建账本；`--dry-run` 通过并报告未授权。
  - 第 26 次拒绝。
  - 剩余 < 60 秒时，不调用后端，也不消耗尝试。
  - 复放结束时已过截止 ⇒ 失败锁停。
  - 有记录时不能新建会话。
  - argparse 里不存在 `--force`。
- Expected：先 FAIL，实现后 PASS。
- 提交：`重复精度 25 次：入口会话、预算封顶、失败锁停、A0 对齐与测试`。

### Task 3: 记录完整性核验

- 接口：`check_record(run, manifest) -> dict`，返回 `sha256`、覆盖首尾余量、时间基准拟合残差、最大间隔等；不过即抛 `ValueError`。`run_next` 在复放成功后调用它。
- RED 测试：从一条合法的合成记录出发，逐项破坏，每项都必须被拒绝：
  - `stopped`、`return_blocked`、`return_deadline`；
  - 帧数、非有限值、时间戳不递增；
  - 身份字段；
  - 正式指令与冻结波形不一致、发送计数不一致；
  - 100 Hz 间隔 > 15 ms，或设备毫秒不单调；
  - 20 Hz 样本在 100 Hz 中找不到；
  - 写总线间隔 > 100 ms；
  - 覆盖不足（首帧取样点早于 100 Hz 起点）；
  - 前置保持缺失、帧数不是 4、目标不是稳定段目标、时刻不早于第一个正式帧。
- Expected：先 FAIL，实现后 PASS。
- 提交：`重复精度 25 次：逐条记录完整性核验与测试`。

### Task 4: 执行器——照搬 pilot 路径、受截止约束的归位、前置保持

- 文件：`sim/replay_repeat_precision25.py`；`sim/test_replay_repeat_precision25.py`。
- 接口：
  - `run(run, seqfile, outdir, context, robot_type="piper", canhs=True) -> int`；
  - `staged_return_guarded(robot, torch, grip, deadline) -> "ok" | "blocked" | "deadline"`；
  - `PREHOLD_FRAMES = 4`。
- 记录：字段同 pilot，另加 `prehold`、`return_deadline`、`scope`（身份）。`sequence` 只记文件名与哈希，不记绝对路径。
- RED 测试：
  - 移植 pilot 执行器的作用域核验与 11 种故障路径。
  - **归位途中到期**：到期之后零新 `send_action`；记录失败且返回非 0；不加速（每段帧数不变）。预采集异常时的归位同样受约束。
  - 前置保持：恰好 4 帧、指令 = 稳定段目标、全部早于第一个正式帧、正式 `cmd` 与冻结波形一致且为 587 帧；裸条和 A0 同样执行。
  - 100 Hz 起点早于第一个正式取样点至少 0.2 秒（假设备下）。
- Expected：先 FAIL，实现后 PASS。
- 提交：`重复精度 25 次：专用执行器（受截止约束的归位、0.2 秒前置保持单独记录）与测试`。

### Task 5: 分析主路径与判定

- 文件：`sim/analyze_repeat_precision25.py`；`sim/test_analyze_repeat_precision25.py`。
- 接口：
  - `SPLITS = (((1, 3), (2, 4)), ((1, 4), (2, 3)))`；
  - `halves(runs, blocks)`、`split_R(ra, rb)`、`main_metrics(X, runs)`、`verdict(complete, R_c)`、`verify(root, release)`、`analyze(root, release)`。
- RED 测试：
  - 分半不共享：集合断言；只扰动区组 2、4 的裸复放时，{1,3} 半的 r 不变。
  - 判定边界：四个恰为 0.6 → 支持；平均恰为 0.3 → 无法判定；平均 0.2999 → 停止；非有限 → 无法判定；不完整 → 无法判定。
  - 篡改记录哈希 ⇒ 无法判定、不算主指标；代码指纹不符 ⇒ 拒绝运行。
  - 合成数据：σ_c 与 R_c 的数值与手算一致。
- Expected：先 FAIL，实现后 PASS。
- 提交：`重复精度 25 次：分析主路径（σ、分半 R、判定）与测试`。

### Task 6: 辅助报告

- 接口：`aux_report(X, recs, runs, manifest, ...)`，在 `verdict` 之后调用，输出写到 `aux` 节。
- RED 测试：
  - 辅助函数抛异常或返回任意值，判定都不变；
  - κ 在裸噪声为零时不超过 1.5（方案第三节的代数耦合）；
  - 正负关系、分半尖峰匹配、门控 / 计数台阶同位、漂移与实际时刻差、旧 20 Hz 口径、偏移 {0, 10, 20, 30, 40} ms 在合成数据上给出预期值；
  - 与 pilot 的跨日对比只读 pilot 记录。
- Expected：先 FAIL，实现后 PASS。
- 提交：`重复精度 25 次：辅助报告（不进判定）与测试`。

### Task 7: 假设备端到端回归

- 文件：`sim/test_e2e_repeat_precision25.py`。
- 设置：
  - 假设备与假时钟；用真实冻结输入（复制到临时目录）；测试专用 release 放在临时目录。
  - 真实 `safety_checks` 每个序列只算一次，之后缓存。
  - 设备工厂只返回假设备；CAN 记录器为假。
- RED / GREEN：
  - 可复现响应 → 支持；
  - 无响应 → 停止；
  - P 有、Q 无 → 无法判定；
  - 第 3 条 IMU 陈旧 → 锁停 → 无法判定；
  - 时钟触顶 → 拒绝下一条且不消耗尝试 → 无法判定；
  - 前三个完整场景各跑两遍，`analysis.json` 哈希一致。
- Expected：全部 PASS。
- 提交：`重复精度 25 次：假设备端到端回归`。

### Task 8: 离线准备与锁定指纹

- 步骤：
  - 离线执行 `--prepare`、`--dry-run`、`--status`，结果应为：通过、报告未授权、未开始。
  - 写 `reports/RELEASE_REPEAT_PRECISION_25.json`（`motion_authorized: false`、`waive_missing_rate_limit_only: false`、manifest 哈希、`code_sha256`）。
  - 再跑 `--dry-run`，核对指纹一致。
- 测试：全部四组测试再跑一遍。
- 提交：`重复精度 25 次：离线准备完成、锁定代码指纹（motion_authorized: false）`。
