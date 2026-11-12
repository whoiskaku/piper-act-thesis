# 重新生成网络 ② 实施计划（由 ACT 指令和实测，从零生成指令序列）

> **给执行者：** 必须用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐个任务实现。步骤用 `- [ ]` 勾选跟踪。

**Goal：** 按设计 v0.2 训练网络 ②。离线过门槛后，用新的在线回放程序在验证集上和 Shift 做真机比较。网络 ② 的要点：教师指令当标签，输入含实测，Transformer，输出绝对指令。

**Architecture：**
- **共用的量放在 `lerobot/common/policies/act/b4_online.py`**：实测窗口、输入单元、指令切换、逐帧检查、网络类。训练和在线回放调用同一批函数，保证逐位一致。
- **离线部分放在新文件 `sim/b4_regen.py`**：建数据、训练、选种子、对照、离线检查、冻结，输出到 `outputs/b4/regen/`。
- **在线部分**：回放程序是新文件 `sim/b4_regen_replay.py`，不改 `sim/replay_probe.py`；评价集 `valregen` 加在 `sim/b4_eval.py`。
- **分两个阶段：**
  - 阶段一（任务 1–5）只用已有数据，不动机械臂。
  - 阶段二（任务 6–7）只在离线门槛通过、并且用户同意之后才开始。

**Tech Stack：** Python（conda 环境 `lerobot`）、numpy、PyTorch（`nn.TransformerEncoder`）、pytest。

**Spec：** `lerobot-piper/reports/PLAN_b4_regen.md`（v0.2，用户 11-11 批准）。旧实现参照：`reports/PLAN_IMPL_b4_seq.md`、`sim/b4_net.py`、`sim/b4_eval.py`。执行者两份都要读。

**执行方式（CLAUDE.md）：**
- 每个任务开头，按 `reports/ADVISOR_STANDARD.md` 逐条回答第 1–6 条。
- 每个任务做完：
  - 说明改了什么、为什么这样改、关键代码怎么工作；
  - 给出运行命令和预期结果，**测试由用户运行**；
  - 等用户反馈后再做下一个。
- 只在用户要求时提交：在 develop 分支上，提交信息不加 `Co-Authored-By`，只 `git add` 本任务的文件。
- 不改 `sim/replay_probe.py`；不碰用户未提交的删除。
- 写代码遵守四条准则：防过度防御、入口单次校验、拒绝提前抽象、控制复杂度。

## Global Constraints

- **帧与对齐：**
  - `DT = 0.05` s；保持段 `HOLD = 20` 帧（`repeat_reversal.HOLD`）。
  - 回放帧 k 对应录制帧 k − HOLD。已核对：`seq_base` 的 `a7[20:] == a_pi`、`s7[20:] == shift`，保持段的值等于第 20 帧的值。
- **实测截止时刻：**
  - 离线：`t_send[k] − 0.010` s；在线：本帧开始时刻 − 0.010 s。旧回放的 `t_send` 紧跟在帧开始之后，两者相差约 1 ms。
  - 只用到达时刻早于截止时刻的采样。
- **编码器窗口：**
  - 数据来自 `can_q200`，每行是 `[t_realtime, 组号(1/3/5), q_a, q_b]`，q 的单位是 0.001°。组号与关节的对应：1 → j1、j2；3 → j3、j4；5 → j5、j6。
  - 到达时刻 = `t_realtime − can_hs_meta.clock.offset_realtime_minus_monotonic_s`。
  - 网格：截止时刻 − 0.400 + 0.005·i，i = 1…80。
- **IMU 窗口：**
  - 数据来自 `imu100`，每行是 `[到达时刻, 设备 ms, 加速度 3 维, 陀螺 3 维]`。
  - 坐标变换：ω = Rᵀ(陀螺 − gyro_bias)，f = Rᵀ(加速度 − b)，R、b、gyro_bias 取该会话的标定（`b4_teacher.session_ctx(sess).calib`）。
  - 网格：截止时刻 − 0.400 + 0.010·i，i = 1…40；每行是 [ω, f]。
- **两个窗口的插值：** 线性插值。比第一个可用采样早的网格点取第一个采样的值，比最后一个可用采样晚的网格点取最后一个采样的值。不用截止时刻之后的采样。
- **输入单元：**
  - 过去 8 帧：单元 i 对应第 k−8+i 帧（i = 0…7），共 102 维，按顺序是：
    - `a7` 的 6 个关节；
    - 已发出指令的 6 个关节；
    - 编码器窗口第 10i…10i+9 行（60 维）；
    - IMU 窗口第 5i…5i+4 行（30 维）。
  - 帧号小于 0 时取第 0 帧。
  - 之后 10 帧：单元 j 是 `predicted(a_pi, ens[kn], kn, k−HOLD+j)` 的 6 个关节（j = 0…9），其中 kn = max(k − HOLD, 0)。
- **标签与权重：**
  - Y[k, j] = 教师 `cmd7[min(k+j, n−1), :6]`。
  - 权重 W[k, j]：k+j ≤ n−1 时，五个会动关节各自取 `w[k+j] > 0`，j4 取 `w[k+j].min() > 0`；k+j > n−1 时为 0。
  - 只保留 W[k] 不全为 0 的帧。
- **数据：**
  - 教师取 `outputs/b4/teacher/summary.json` 里 `passed` 为真的；`split` 为 train、spare 的算训练，val 的算验证，test 不用。
  - 不用会话 `20261104b`、`20261104c`、`20261104d`。
  - 回放有效的条件：`stopped` 为空，`t_send` 长度等于 n，`can_q200` 和 `imu100` 非空，且回放序列文件里的 `cmd_hash` 和记录里的一致。
  - 预期规模：训练 26 条轨迹、468 遍，验证 7 条、126 遍；实际数以运行时打印的为准。
- **网络 `RegenNet`：**
  - 结构：d = 128，4 层，4 头，前馈 512，dropout 0（用户 11-11 改，原 0.1；训练时由 `ARCH` 传入），`batch_first`、`norm_first`；18 个位置的可学位置编码。
  - 先经 `RegenNet.prep` 改成相对量（用户 11-11 选 B）：r = 之后单元的第一帧；过去单元的 ACT、已发出的指令减 r，编码器各点减同一帧已发出的指令，
    IMU 不变；之后单元为 [ACT − r, r]（12 维）。数据文件和在线回放仍给原始单元，相对化只在网络里做。
  - 过去单元和之后单元各用一个线性映射到 d；在之后 10 个单元上用同一个线性头输出 6 维。
  - 输出 = mu_y + sd_y × 头的输出，是绝对角度；**网络里没有把 ACT 或 Shift 指令加回输出的通路。**
  - 标准化：均值、标准差在训练集上、按相对化之后的通道算；sd 的下限，输入取 1e-6、输出取 1e-3（j4 的标签恒为 0，sd 用 1e-3）。
  - 对照用的输入组：`sent` = 第 6–11 列，`enc` = 第 12–71 列，`imu` = 第 72–101 列；去掉某组 = 标准化之后把这几列置 0。
- **训练：**
  - 损失是加权均方误差，按 mrad 计、各关节等权（用户 11-11 改，原为标准化后的），第一帧的权重 × 9。
  - AdamW，lr 3e-4 在 100 轮里按余弦降到 0（用户 11-11 加），weight_decay 0.01，批 1024，最多 100 轮，
    不早停：跑满 100 轮（耐心 = 100，用户 11-11；原 10 轮，见账本），留验证集第一帧加权损失最小的那一轮。
  - 种子 0、1、2，按验证集第一帧损失选种子。对照只训种子 0。
  - 以上超参数是默认值，执行者可以在同量级内调整，调整要记进账本。
- **使用时：**
  - 每帧把网络第一帧输出按关节切换：会动的关节 `w[k] > 0` 时用网络、否则用 Shift；j4 在 `w[k].min() > 0` 时用网络；夹爪用 `s7[k, 6]`。
  - 已发出的指令就是自己前几帧最终实发的。
- **逐帧检查**（停机条件，不是限幅）：
  - 位置：在 [已发过的最小值 − 0.05, 最大值 + 0.05] 与 [模型限位 − 0.035 + 0.2°, 模型限位 + 0.035 − 0.2°] 的交集内。
  - 每帧变化：j1…j6 依次不超过 0.060、0.045、0.045、0.005、0.025、0.045 rad。
  - 与 `a7[k]` 的差：j1…j6 依次不超过 0.100、0.100、0.100、0.010、0.050、0.100 rad。
  - 非有限值。
  - 实测中断：编码器任一组的最新采样早于截止时刻 30 ms 以上，或 IMU 早于 50 ms 以上。
- **离线门槛：**
  - 门槛 1：验证集 7 条轨迹 Shift 回放（`b01`、`b02`、`as01`、`as02`、`as03`）的状态上，网络第一帧减 `s7`，对教师补正 `cmd7 − s7` 的 R²（逐关节，按 `w` 加权，同 `b4_net._r2`）。五个会动关节都要 ≥ 冻结的一帧版 ① 在同样 7 条轨迹上重算的 R² − 0.10。
  - 门槛 2：10 条验证轨迹按使用方式预演（实测取 `sessions/20261106q/replay_<轨迹>_qshift1_r1.json`，已发出的指令用自己的输出），逐帧检查一次都不触发。
- **文件与编号：**
  - 只写 `outputs/b4/regen/` 和 `outputs/b4/eval/valregen/`。
  - 评价回放编号 `{轨迹}_g{条件}{轮}`；上机试跑编号 `{轨迹}_t{模式}{轮}`，放在单独的会话目录里。
  - 不写 ① 的目录，不复用 `e`、`q` 前缀。
- **测试命令**（在 `lerobot-piper/` 下）：`OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_regen.py -q`。阶段二另加 `sim/test_b4_regen_replay.py`。每个任务结束时连同 `sim/test_b4_net.py sim/test_b4_eval.py` 一起跑。

## Review Focus

1. **截止时刻附近的采样，以及三组编码器到达时刻不齐。** 期望：截止时刻及之后到达的采样一律不用；某组最新采样之后的网格点保持该组最后的值；离线和在线算出的窗口逐位相同。测试在任务 1、6。
2. **坏回放和缺标定。** 回放中途停了、没有编码器、哈希对不上，或会话的标定文件找不到。期望：跳过并打印原因，计入跳过数；不报错，也不悄悄混进数据。测试在任务 2。
3. **开头和结尾的帧**（k < 8、k < HOLD、k + j ≥ n）。期望：
   - 帧号小于 0 时用第 0 帧补齐（任务 1）；
   - 保留下来的第一帧是 HOLD − 9 = 11，因为这是输出窗口第一次碰到有权重的帧（任务 2）；
   - 超出末尾的输出帧权重为 0，第一帧照常（任务 2）。
4. **j4 的标签恒为 0。** 期望：标准化不出现除零或 NaN；j4 的输出有限，并且在逐帧检查的界内。测试在任务 3。
5. **在线一帧计算超时，或实测中断。** 期望：在发出这一帧之前停机，记录原因，不发陈旧的指令。测试在任务 6。

## 用户运行的节点（每次先给命令和预期结果）

| 在哪个任务之后 | 做什么 |
|---|---|
| 2 | `python -u sim/b4_regen.py build`：看各划分的轨迹数、遍数、样本数和跳过清单 |
| 3 | `python -u sim/b4_regen.py train`（3 个种子并选种子）；`python -u sim/b4_regen.py ablate`（3 个对照） |
| 4 | `python -u sim/b4_regen.py bounds`：看已发过的范围和最终的界 |
| 5 | `python -u sim/b4_regen.py check`：**门槛不过就停下讨论**；通过后再 `freeze` |
| 6 | 先 `dry` 核对 p44 的逐位一致和计算耗时；再上机（用户在场）：p23、p44 各跑一遍影子，然后各跑一遍真正发网络指令 |
| 7 | 标定后跑 `b4_eval.py replay --set valregen`，分两批，每批结束后做收尾核对；然后 `analyze --set valregen` |

---

## 阶段一：只用已有数据，不动机械臂

### Task 1: 共用的实测窗口与输入单元

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`：文件末尾加一节「重新生成网络 ②」，文件头的说明同步更新。
- Create: `lerobot-piper/sim/test_b4_regen.py`

**Interfaces:**
- Consumes: `predicted`、`te_states`。
- Produces（都在 `b4_online.py`）：
  - 常量：`REGEN_P = 8`、`REGEN_N = 10`、`CUT_S = 0.010`、`WIN_S = 0.400`、`ENC_DT = 0.005`、`IMU_DT = 0.010`、`CAN_GROUPS = ((1, (0, 1)), (3, (2, 3)), (5, (4, 5)))`。
  - `enc_window(q200: np.ndarray, offset: float, cutoff: float) -> tuple[np.ndarray, float]`：返回 (80, 6) 的弧度窗口和最旧那一组的最新采样距截止时刻的秒数；某组没有可用采样时，该组两个关节为 NaN，秒数为 inf。
  - `imu_window(imu100: np.ndarray, calib: dict, cutoff: float) -> tuple[np.ndarray, float]`：返回 (40, 6) 的窗口和最新采样距截止时刻的秒数。
  - `regen_act(a7: np.ndarray, a_pi: np.ndarray, ens: np.ndarray, k: int, hold: int) -> tuple[np.ndarray, np.ndarray]`：返回过去 8 帧的 ACT 指令 (8, 6) 和之后 10 帧的 ACT 指令 (10, 6)。
  - `regen_tokens(act_past, sent_past, enc, imu, act_seg) -> tuple[np.ndarray, np.ndarray]`：返回过去单元 (8, 102) float32 和之后单元 (10, 6) float32。

- [ ] **Step 1: 写测试**（在测试文件里写合成数据的工具函数：三组 200 Hz 的编码器、100 Hz 的 IMU，时间从 0 开始）

```python
def test_enc_window_ignores_rows_at_or_after_cutoff():
    rows = q200_ramp(t_end=1.2)
    a, _ = O.enc_window(rows[rows[:, 0] < 1.0], 0.0, 1.0)
    b, _ = O.enc_window(rows, 0.0, 1.0)
    assert np.array_equal(a, b)

def test_enc_window_converts_millidegrees_to_rad():
    w, _ = O.enc_window(q200_const(1000, t_end=1.0), 0.0, 1.0)
    assert w.shape == (80, 6) and np.allclose(w, np.radians(1.0))

def test_enc_window_holds_last_value_and_reports_age():          # Review Focus 1
    rows = q200_ramp(t_end=0.95)
    w, age = O.enc_window(rows, 0.0, 1.0)
    grid = 1.0 - O.WIN_S + O.ENC_DT * np.arange(1, 81)
    assert np.allclose(w[grid > rows[:, 0].max()], w[grid > rows[:, 0].max()][0]) and age >= 0.05 - 1e-9

def test_enc_window_missing_group_is_nan_and_stale():
    rows = q200_ramp(t_end=1.0)
    w, age = O.enc_window(rows[rows[:, 1] != 3], 0.0, 1.0)
    assert np.isnan(w[:, 2:4]).all() and np.isfinite(w[:, [0, 1, 4, 5]]).all() and age == np.inf

def test_imu_window_rotates_and_removes_bias():
    R = rot_z(90)
    calib = {"R": R.tolist(), "gyro_bias": [0.1, 0, 0], "b": [0, 0, 0.2]}
    rows = imu_rows(gyro=R @ [1, 2, 3] + [0.1, 0, 0], acc=R @ [4, 5, 6] + [0, 0, 0.2], t_end=1.0)
    w, _ = O.imu_window(rows, calib, 1.0)
    assert np.allclose(w, [1, 2, 3, 4, 5, 6])

def test_regen_act_uses_only_chunks_up_to_k():
    A, B_ = act_case(seed=0), act_case(seed=0, change_after=31)    # 各返回 (a7, a_pi, ens)；B_ 从录制帧 31 起的 chunk 不同
    for k in (25, 50):                                  # 回放帧；HOLD = 20 ⇒ 录制帧 5、30
        assert all(np.array_equal(x, y) for x, y in zip(O.regen_act(*A, k, 20), O.regen_act(*B_, k, 20)))

def test_regen_act_pads_with_frame_zero():                      # Review Focus 3
    a7, a_pi, ens = act_case(seed=0)
    past, seg = O.regen_act(a7, a_pi, ens, 3, 20)
    assert np.array_equal(past[:5], np.repeat(a7[:1, :6], 5, 0)) and np.array_equal(seg[0], a_pi[0, :6])

def test_regen_tokens_layout():
    past, fut = O.regen_tokens(*layout_inputs())
    assert past.shape == (8, 102) and fut.shape == (10, 6)
    assert np.array_equal(past[3, 12:72], layout_inputs()[2][30:40].ravel())
```

- [ ] **Step 2: 用户运行，确认失败**（`AttributeError: module ... has no attribute 'enc_window'`）
- [ ] **Step 3: 实现上述函数**
  - 两个窗口都对每组（或 IMU）只取到达时刻早于截止时刻的行，用 `np.interp` 插值到网格上。
  - 比最后一个可用采样晚的网格点，`np.interp` 已经会保持最后的值；不要用截止时刻之后的行参与插值。
  - `regen_act`：过去部分是 `a7[np.clip(np.arange(k - 8, k), 0, None), :6]`；之后部分见 Global Constraints。
- [ ] **Step 4: 用户运行，全部通过**；旧测试 `sim/test_b4_net.py sim/test_b4_eval.py` 也全部通过。
- [ ] **Step 5: 用户要求时提交**（`b4_online.py`、`sim/test_b4_regen.py`）

### Task 2: 数据

**Files:**
- Create: `lerobot-piper/sim/b4_regen.py`：包含文件头用法、`REGEN`、`BAD_SESSIONS`、`SHIFT_RIDS`、命令行 `build`。
- Test: `lerobot-piper/sim/test_b4_regen.py`：加夹具 `case`，是一条合成的完整轨迹，见下。

**Interfaces:**
- Consumes：任务 1 的全部函数；`b4_teacher.load_seq`、`session_ctx`、`_sessions`；`b4_eval._log`；`b4_positions.REC`。
- Produces：
  - `REGEN = C.OUT / "regen"`；`BAD_SESSIONS = ("20261104b", "20261104c", "20261104d")`；`SHIFT_RIDS = ("b01", "b02", "as01", "as02", "as03")`。
  - `cases(rows: dict | None = None) -> list[tuple[str, str, str]]`：返回 (轨迹, 划分, 会话名)，划分是 "train" 或 "val"；被排除的打印原因。
  - `valid_replays(sess: Path, tid: str, n: int) -> list[tuple[str, dict]]`：返回 (rid, 回放记录)，按 rid 排序；无效的打印原因并跳过。
  - `assemble_regen(h: dict, base: dict, log: dict, teacher7: np.ndarray, calib: dict) -> dict`：返回 `past` (m, 8, 102)、`fut` (m, 10, 6)、`Y` (m, 10, 6)、`W` (m, 10, 6)，都是 float32；另有 `k` (m,)。
  - `build_regen(rows: dict | None = None, out: Path = REGEN, loader=None) -> Path`：写 `out / "data.npz"`，内容为上面各数组加 `traj_id`、`rid`、`split`、`data_sha`（past 与 Y 字节的 sha256 前 16 位）；逐划分打印轨迹数、遍数、样本数。
    - `loader(tid) -> (base, log, teacher7, calib, [(rid, h), …])`；默认从磁盘读，测试时注入。

夹具 `case`（测试文件里的工具函数和夹具都由执行者写在测试文件中）：
- n = 80 帧（含 HOLD = 20 帧），t_send = 0.05·k。
- 编码器 200 Hz、IMU 100 Hz，到达时刻带几毫秒的随机抖动。
- a7 是平滑的随机游走，每帧变化不超过 5 mrad；s7[k] = a7[min(k+1, n−1)]，夹爪列相同。这样切换处的每帧变化都不超过 10 mrad，不会误触发逐帧检查。
- `w` 在保持段和第 45–52 帧为 0，其余为 1，各关节相同；第 30 帧上 j3 单独为 0。
- teacher7 = s7 + w·δ（float32），|δ| ≤ 3 mrad；j4 和夹爪的 δ 为 0。
- `replays[0]` 是 Shift 回放（`cmd` = s7）；`replays[1]` 的已发出指令和它不同。

- [ ] **Step 1: 写测试**

```python
def test_labels_are_the_teacher_commands_for_every_replay(case):
    for h in case.replays:
        d = R.assemble_regen(h, case.base, case.log, case.teacher7, case.calib)
        for j in (0, 4, 9):
            kk = np.minimum(d["k"] + j, case.n - 1)
            assert np.array_equal(d["Y"][:, j], case.teacher7[kk, :6])

def test_weights_follow_window_and_stop_at_the_end(case):     # Review Focus 3
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    i = list(d["k"]).index(case.n - 3)
    assert d["W"][i, 3:].sum() == 0 and d["W"][i, 0].sum() > 0
    i = list(d["k"]).index(30)
    assert d["W"][i, 0, 2] == 0 and d["W"][i, 0, 3] == 0 and d["W"][i, 0, [0, 1, 4, 5]].all()   # j3 窗为 0 ⇒ j4 也为 0

def test_frames_without_any_weight_are_dropped(case):
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    assert d["W"].reshape(len(d["k"]), -1).any(1).all() and 0 not in d["k"]

def test_first_kept_frame_is_hold_minus_nine(case):           # Review Focus 3
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    assert d["k"].min() == 20 - (O.REGEN_N - 1)

def test_inputs_use_only_samples_before_cutoff(case):          # Review Focus 1
    h = case.replays[0]
    a = R.assemble_regen(h, case.base, case.log, case.teacher7, case.calib)
    b = R.assemble_regen(with_rows_after(h, frame=40, seconds=0.010), case.base, case.log, case.teacher7, case.calib)
    i = list(a["k"]).index(40)
    assert np.array_equal(a["past"][i], b["past"][i])

def test_cases_skip_bad_sessions_failed_teachers_and_test_split():
    rows = {"a": {"split": "train", "passed": True, "session": "20261104e"},
            "b": {"split": "spare", "passed": True, "session": "20261105"},
            "c": {"split": "val", "passed": True, "session": "20261104b"},
            "d": {"split": "train", "passed": False, "session": "20261105"},
            "e": {"split": "test", "passed": True, "session": "20261105"}}
    assert R.cases(rows) == [("a", "train", "20261104e"), ("b", "train", "20261105")]

def test_invalid_replays_are_skipped_with_a_reason(tmp_path, capsys):   # Review Focus 2
    write_replays(tmp_path, ok="p1_as01", stopped="p1_as02", no_can="p1_as03", bad_hash="p1_at01", short="p1_b01")
    assert [r for r, _ in R.valid_replays(tmp_path, "p1", 80)] == ["p1_as01"]
    assert capsys.readouterr().out.count("跳过") == 4

def test_build_writes_splits_and_never_mixes_val_into_train(case, tmp_path):
    p = R.build_regen(rows=two_rows(), out=tmp_path, loader=lambda t: case.loader_tuple)
    z = np.load(p)
    assert set(z["split"]) == {"train", "val"} and not set(z["traj_id"][z["split"] == "train"]) & set(z["traj_id"][z["split"] == "val"])
```

- [ ] **Step 2: 用户运行，确认失败**（找不到模块 `b4_regen`）
- [ ] **Step 3: 实现上面的函数和命令行 `build`**
  - `assemble_regen` 每帧用 `t_send[k] − CUT_S` 作截止时刻，调任务 1 的函数。
  - 编码器和 IMU 先按到达时刻排好序，用 `np.searchsorted` 找到截止位置后切片，不要逐帧重新排序。
- [ ] **Step 4: 用户运行，全部通过；再跑 `python -u sim/b4_regen.py build`**。预期：训练约 26 条轨迹、468 遍，验证约 7 条、126 遍，打印跳过清单，写出 `outputs/b4/regen/data.npz`。
- [ ] **Step 5: 用户要求时提交**

### Task 3: 网络与训练

**Files:**
- Modify: `b4_online.py`：加 `RegenNet`、`load_regen`、`GROUP_COLS`。
- Modify: `sim/b4_regen.py`：加 `train_regen`、`select_regen`、`ablate_regen`，命令行 `train`、`ablate`。
- Test: `sim/test_b4_regen.py`

**Interfaces:**
- Consumes：任务 2 的 `data.npz`，内含 past、fut、Y、W、split、data_sha。
- Produces：
  - `GROUP_COLS = {"sent": slice(6, 12), "enc": slice(12, 72), "imu": slice(72, 102)}`。
  - `RegenNet(n_past=102, n_fut=6, d=128, layers=4, heads=4, ff=512, dropout=0.1, drop=())`：
    - buffer：`mu_p`、`sd_p`（102）、`mu_f`、`sd_f`（6）、`mu_y`、`sd_y`（6）；
    - `forward(past: Tensor (B, 8, 102), fut: Tensor (B, 10, 6)) -> Tensor (B, 10, 6)`，输出绝对弧度。
  - `load_regen(path: str) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], str]`：读冻结 JSON，返回推理函数 (8, 102), (10, 6) → (10, 6) 和权重哈希；哈希不符就停。
  - `train_regen(data: Path, seed: int, drop: tuple[str, ...] = (), epochs: int = 100, lr: float = 3e-4, batch: int = 1024, patience: int = 100, out: Path = REGEN / "runs", device: str = "cuda") -> dict`：
    - 返回 `{"path", "val_first", "val_all", "train_loss", "epochs", "history_val_first"}`，最后一项是逐轮的验证第一帧损失；
    - 存 `{"state_dict", "drop", "seed", "data_sha", "arch"}`，文件名为 `s{seed}.pt` 或 `s0_no-{组}.pt`。
  - `select_regen() -> dict`：写 `REGEN / "selection.json"`，内容 `{"chosen", "runs"}`。
  - `ablate_regen() -> dict`：写 `REGEN / "ablation.json"`，三个对照分别去掉 (enc, imu)、(imu)、(sent)。

- [ ] **Step 1: 写测试**

```python
def test_no_path_from_inputs_to_output_when_head_is_zero():     # 输出不是调整量
    net = O.RegenNet()
    torch.nn.init.zeros_(net.head.weight); torch.nn.init.zeros_(net.head.bias)
    net.mu_y.copy_(torch.arange(6.0))
    y = net(torch.randn(3, 8, 102), torch.randn(3, 10, 6))
    assert torch.equal(y, torch.arange(6.0).expand(3, 10, 6))

def test_dropped_groups_do_not_change_the_output():
    net = O.RegenNet(drop=("enc", "imu")).eval()
    p = torch.randn(2, 8, 102); q = p.clone(); q[..., 12:102] += 5.0
    f = torch.randn(2, 10, 6)
    assert torch.equal(net(p, f), net(q, f))

def test_constant_j4_labels_give_finite_outputs(tmp_path):      # Review Focus 4
    r = R.train_regen(tiny_data(tmp_path, j4_zero=True), seed=0, epochs=2, device="cpu", out=tmp_path)
    m = torch.load(r["path"], map_location="cpu")
    assert torch.isfinite(m["state_dict"]["sd_y"]).all() and m["state_dict"]["sd_y"][3] == 1e-3 and np.isfinite(r["val_first"])

def test_overfits_a_tiny_dataset(tmp_path):
    r = R.train_regen(tiny_data(tmp_path, n=64), seed=0, epochs=400, patience=400, device="cpu", out=tmp_path)
    assert r["train_loss"] < 1e-2

def test_first_frame_weight_is_n_minus_one():
    W = torch.ones(1, 10, 6)
    assert R.frame_weights(W)[0, 0].sum() == 9 * W[0, 1].sum()

def test_same_seed_same_weights(tmp_path):
    a, b = (R.train_regen(tiny_data(tmp_path), 0, epochs=3, device="cpu", out=tmp_path / x) for x in ("a", "b"))
    assert all(torch.equal(x, y) for x, y in zip(torch.load(a["path"])["state_dict"].values(), torch.load(b["path"])["state_dict"].values()))

def test_saved_state_is_the_best_validation_epoch(tmp_path):
    r = R.train_regen(noisy_val_data(tmp_path), 0, epochs=30, patience=30, device="cpu", out=tmp_path)
    assert r["val_first"] == min(r["history_val_first"])

def test_load_regen_rejects_changed_weights(tmp_path):
    f = R.freeze_regen(Path(R.train_regen(tiny_data(tmp_path), 0, epochs=1, device="cpu", out=tmp_path)["path"]), out=tmp_path)
    Path(json.loads(f.read_text())["weights"]).write_bytes(b"x")
    with pytest.raises(SystemExit):
        O.load_regen(str(f))
```

  - 上面用到的 `frame_weights(W) -> Tensor`（把第一帧的权重乘 9）属于本任务。
  - `freeze_regen` 在任务 5 实现；本任务先把 `test_load_regen_rejects_changed_weights` 标成 skip，任务 5 再打开。

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
  - 标准化统计量只在训练划分上算。
  - 早停按验证划分的第一帧加权损失，存最好那一轮。
  - 被去掉的组在标准化之后置 0。
- [ ] **Step 4: 用户运行，全部通过；再跑 `train` 和 `ablate`**。预期：打印 3 个种子各自的验证第一帧损失和选中的种子，再打印 3 个对照的损失。
- [ ] **Step 5: 用户要求时提交**

### Task 4: 逐帧检查、指令切换与界

**Files:**
- Modify: `b4_online.py`：加 `STEP_MAX`、`DIFF_MAX`、`POS_PAD`、`STALE_ENC_S`、`STALE_IMU_S`、`regen_bounds`、`step_check`、`regen_command`。
- Modify: `sim/b4_regen.py`：加 `sent_range`，命令行 `bounds`。
- Test: `sim/test_b4_regen.py`

**Interfaces:**
- Produces：
  - `STEP_MAX = np.array([0.060, 0.045, 0.045, 0.005, 0.025, 0.045])`；`DIFF_MAX = np.array([0.100, 0.100, 0.100, 0.010, 0.050, 0.100])`；`POS_PAD = 0.05`；`STALE_ENC_S = 0.030`；`STALE_IMU_S = 0.050`。
  - `regen_bounds(cfg: OnlineCfg, sent_lo: np.ndarray, sent_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]`：返回 6 个关节的上下界，取法见 Global Constraints。
  - `step_check(u6: np.ndarray, prev6: np.ndarray, a6: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str | None`：
    - 越界返回原因，以规则名开头，再写关节和数值；规则名依次是「非有限」「位置」「每帧变化」「与 ACT 指令的差」；
    - 不越界返回 None。
    - 实测中断的原因由任务 6 的 `loop` 生成，以「实测中断」开头。
  - `regen_command(first6: np.ndarray, s7k: np.ndarray, w5k: np.ndarray) -> np.ndarray`：返回 (7,) float32 的实发指令。
  - `sent_range(root: Path = B.TEACH / "sessions") -> dict`：遍历全部 `seq_*.npz` 的 `cmd7[:, :6]`，写 `REGEN / "bounds.json"`，内容 `{"sent_lo", "sent_hi", "n_files", "lo", "hi"}`。

- [ ] **Step 1: 写测试**

```python
def test_each_rule_triggers_and_in_range_passes():
    lo, hi = -np.ones(6), np.ones(6)
    ok = np.zeros(6)
    assert O.step_check(ok, ok, ok, lo, hi) is None
    for u, prev, a, rule in ((np.r_[1.1, 0, 0, 0, 0, 0], ok, ok, "位置"),
                             (np.r_[0, 0, 0, 0, 0.03, 0], ok, np.r_[0, 0, 0, 0, 0.03, 0], "每帧变化"),
                             (np.r_[0.05, 0, 0, 0, 0, 0], np.r_[0.05, 0, 0, 0, 0, 0], np.r_[-0.06, 0, 0, 0, 0, 0], "与 ACT 指令的差"),
                             (np.r_[np.nan, 0, 0, 0, 0, 0], ok, ok, "非有限")):
        assert rule in O.step_check(u, prev, a, lo, hi)

def test_regen_command_switches_per_joint_and_keeps_shift_gripper():
    first, s7 = np.arange(6.0), -np.arange(7.0)
    u = O.regen_command(first, s7, np.array([1, 0, 1, 1, 1.0]))
    assert u.dtype == np.float32 and list(u) == [0, -1, 2, -3, 4, 5, -6]   # j2 窗为 0 ⇒ Shift；j4 取最小值 ⇒ Shift

def test_perfect_teacher_prediction_reproduces_the_teacher_bitwise(case):   # 设计 §6
    u = np.stack([O.regen_command(case.teacher7[k, :6], case.base["s7"][k], case.base["w"][k]) for k in range(case.n)])
    assert np.array_equal(u, case.teacher7.astype(np.float32))

def test_bounds_intersect_sent_range_and_model_limits():
    cfg = toy_cfg(jnt_lo=[-1] * 6, jnt_hi=[1] * 6)
    lo, hi = O.regen_bounds(cfg, np.full(6, -2.0), np.full(6, 0.1))
    assert np.allclose(hi, 0.15) and np.allclose(lo, -1 - 0.035 + np.radians(0.2))
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
  - 检查顺序：非有限值 → 位置 → 每帧变化 → 与 ACT 指令的差；返回第一个违反的规则。
  - 合成夹具里 teacher7 在窗为 0 处等于 s7，在窗大于 0 处等于 s7 + δ；完美复现的测试靠这一点成立。
- [ ] **Step 4: 用户运行，全部通过；再跑 `bounds`**。预期：`sent_lo`、`sent_hi` 与设计 §4.6 的表一致（例如 j1 约 −0.35…0.41 rad），并打印最终的界。
- [ ] **Step 5: 用户要求时提交**

### Task 5: 离线检查与冻结

**Files:**
- Modify: `sim/b4_regen.py`：加 `first_frames`、`preview`、`check_regen`、`freeze_regen`，命令行 `check`、`freeze`。
- Test: `sim/test_b4_regen.py`；打开任务 3 里跳过的那个测试。

**Interfaces:**
- Consumes：任务 1–4 的全部内容；`b4_net._r2`、`b4_net.assemble`；`O.load_frozen`（一帧版）。
- Produces：
  - `first_frames(fn, h: dict, base: dict, log: dict, calib: dict) -> np.ndarray`：返回 (n, 6)，是这一遍的状态上（已发出的指令取记录里的）网络每帧的第一帧输出。
  - `preview(fn, h: dict, base: dict, log: dict, calib: dict, lo: np.ndarray, hi: np.ndarray) -> dict`：按使用方式逐帧跑，已发出的指令用自己的 `regen_command` 结果。越界时记下原因但不停，照样把这一帧当作已发出。返回 `{"u7": (n, 7), "first": (n, 6), "stops": [(k, 原因)]}`。
  - `first_frames`、`preview`（以及任务 6 的 `loop`）都按帧号从 0 到 n−1 的顺序、每帧调用一次 fn。测试里的教师预言函数就是靠这一点知道当前帧号的。
  - `check_regen() -> dict`：写 `REGEN / "check.json"`，内容：
    - `{"gate1": {"r2", "base", "threshold", "passed"}, "gate2": {"stops": {轨迹: 次数}, "passed"}, "reports": {...}, "passed"}`；
    - reports 包括：预演输出对教师补正的 R²；从 ILC 迭代中途（`c02`、`h04`）和教师回放（`at01`–`at03`）的状态出发的 R²，以及从教师回放的状态出发时离教师指令的 RMS（mrad）；三个对照按门槛 1 口径算的 R²；j4 输出的标准差（mrad）；网络、Shift、教师三种指令在窗大于 0 的帧上 ≥ 2 Hz 成分的 RMS（mrad）。
  - `freeze_regen(run: Path, out: Path = REGEN) -> Path`：只能冻结一次。复制成 `frozen.pt`，写 `frozen.json`，内容 `{"weights", "weights_sha", "data_sha", "seed", "drop", "arch", "git_head", "time"}`。

- [ ] **Step 1: 写测试**

```python
def test_r2_against_the_teacher_is_one_for_a_perfect_net(case):
    fn = teacher_oracle(case)                                   # 按帧号返回教师 cmd7 的 10 帧
    first = R.first_frames(fn, case.replays[0], case.base, case.log, case.calib)
    act = list(O.ACTIVE)
    star = case.teacher7[:, act] - case.base["s7"][:, act]
    assert np.allclose(N._r2(first[:, act] - case.base["s7"][:, act], star, case.base["w"]), 1.0)

def test_preview_feeds_back_its_own_commands(case):
    seen = []
    fn = recording_fn(seen, out=lambda past, fut: fut + 0.001)
    R.preview(fn, case.replays[0], case.base, case.log, case.calib, -np.ones(6) * 9, np.ones(6) * 9)
    k = 40
    assert np.allclose(seen[k][0][-1, 6:12], (seen[k - 1][1][0] + 0.001))   # 第 k 帧输入里的上一帧已发指令 = 上一帧自己的输出

def test_preview_counts_every_violation_without_stopping(case):
    fn = lambda past, fut: fut + 0.2                            # 用网络的帧都和 ACT 指令差 0.2 rad
    r = R.preview(fn, case.replays[0], case.base, case.log, case.calib, -np.ones(6) * 9, np.ones(6) * 9)
    assert len(r["u7"]) == case.n and len(r["stops"]) >= int((case.base["w"].max(1) > 0).sum())
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
  - 门槛 1 的 R²：网络与一帧版都在同样 7 条轨迹、同样的帧上算；每遍 Shift 回放的帧都计入。
  - 一帧版的预测用 `b4_net.assemble(...)["X"]` 加冻结权重。
- [ ] **Step 4: 用户运行，全部通过；再跑 `check`**。
  - 预期：打印逐关节 R²、门槛、预演触发次数，以及「门槛：通过 / 不通过（原因）」。
  - **不通过就停下讨论，不自行调参。**
  - 通过后跑 `freeze`。
- [ ] **Step 5: 用户要求时提交**

**——阶段一到此结束。阶段二只在门槛通过、并且用户同意后开始。——**

## 阶段二：在线回放与真机

### Task 6: 在线回放程序

**Files:**
- Create: `lerobot-piper/sim/b4_regen_replay.py`
- Create: `lerobot-piper/sim/test_b4_regen_replay.py`

**Interfaces:**
- Consumes：任务 1–5 的全部内容；`replay_probe.staged_return`、`replay_probe.FPS`、`make_probe_b2.reach_limits`；`step3_canobs.can_hs_recorder`（`CanHighSpdRecorder`、`decode`）；`b4_teacher.session_ctx`。
- Produces：
  - `class Recorded(h: dict)`：`rows(cutoff: float) -> tuple[np.ndarray, np.ndarray]`，返回这一遍记录里到达时刻早于截止时刻的 `can_q200`、`imu100` 行。
  - `class Live(robot, can_rec)`：`rows(cutoff)` 同上。每次调用先把新到的数据取出来（CAN 原始帧当场用 `decode` 解出）再筛选。
  - `loop(n, base, log, fn, calib, offset, lo, hi, source, mode, send, clock) -> dict`：
    - mode 是 "regen" 或 "shadow"；`send(u7) -> t_send`；`clock() -> 本帧开始时刻`。
    - 返回 `{"cmd", "t_send", "t_frame", "cutoff", "net_first", "net_u", "check", "compute_ms", "stopped"}`。
  - `run(sess: Path, tid: str, rid: str, mode: str, net_path: str) -> int`：真机。
  - `dry(tid: str, source_json: Path, net_path: str) -> dict`：不连机械臂，核对和离线逐位一致，并报告计算耗时。
  - 命令行：
    - `dry --tid --source --net`
    - `run --session --tid --rid --mode shadow|regen --net`

`loop` 每帧按这个顺序做：
1. t0 = clock()，截止时刻 = t0 − CUT_S。
2. 取实测，算窗口。实测中断就停（两种模式都停）。
3. 组输入单元，网络推理，`regen_command`，`step_check`。
4. regen 模式下检查不过：**这一帧不发**，记原因并停；shadow 模式只记录。
5. 发出：regen 发网络切换后的指令，shadow 发 `s7[k]`。
6. 记录本帧。

`run` 的流程：
- 连机械臂并做 IMU 自检。
- 预移动到 `s7[0]`，判据照抄 `replay_probe._run_impl`：3 s 余弦插值，5 s 静置，最后 2 s 的位置带 [0.5, 0.5, 0.5, 0.8, 0.8, 0.8]°，到位容差 = 可达范围豁免 + 1°。
- IMU 100 Hz 记录和高速 CAN 记录必须都开着（CAN 要通过 `wait_first`），否则在运动前退出。
- 回放结束后：
  - 写 `replay_{rid}_r1.json`：字段与 `replay_probe` 的相同，另加 `mode`、`net_sha`、`loop` 的各项；
  - 写 `seq_{rid}.npz`：`cmd7` = 实际发出的，`da` = 0，`source` = {rid, label, cmd_hash, net_sha, mode}，`cmd_hash` 和记录里的一致，`replay_result` 可以直接读；
  - 分段归位。
- 异常处理照 `shift_replay.guarded_run`：没有记录时落一份 stopped 记录并归位；已有记录时只补原因；最后解除析构归位。

- [ ] **Step 1: 写测试**
  - 用任务 2 的合成夹具 `case`（从 `test_b4_regen.py` 导入）。
  - 测试工具 `run_loop(case, mode, fn, source=None, return_rec=False)`：默认源是 `Recorded(case.replays[0])`，`clock` 和 `send` 都返回这一遍记录里第 k 帧的 `t_send`，所以截止时刻和离线完全相同。返回发出的指令列表；`return_rec=True` 时再返回 `loop` 的结果。

```python
def test_dry_inputs_equal_the_offline_assembly(case):           # 设计 §6；Review Focus 1
    seen = []
    run_loop(case, mode="shadow", fn=recording_fn(seen, out=lambda p, f: f))   # Shift 回放 + shadow ⇒ 已发出的指令和记录相同
    d = R.assemble_regen(case.replays[0], case.base, case.log, case.teacher7, case.calib)
    for i, k in enumerate(d["k"]):
        assert np.array_equal(seen[k][0], d["past"][i]) and np.array_equal(seen[k][1], d["fut"][i])

def test_only_the_first_output_frame_is_sent(case):             # 设计 §6
    a = run_loop(case, mode="regen", fn=teacher_oracle(case))
    b = run_loop(case, mode="regen", fn=teacher_oracle(case, noise_after_first=0.05))
    assert np.array_equal(np.array(a), np.array(b))

def test_shadow_sends_shift_bitwise(case):
    sent = run_loop(case, mode="shadow", fn=lambda p, f: f + 0.01)
    assert np.array_equal(np.array(sent), case.base["s7"].astype(np.float32))

def test_perfect_teacher_net_sends_the_teacher_commands(case):
    sent = run_loop(case, mode="regen", fn=teacher_oracle(case))
    assert np.array_equal(np.array(sent), case.teacher7.astype(np.float32))

def test_regen_stops_before_sending_a_violating_frame(case):     # Review Focus 5
    sent, rec = run_loop(case, mode="regen", fn=jump_at(frame=40, by=0.2), return_rec=True)
    assert len(sent) == 40 and "每帧变化" in rec["stopped"]

def test_stale_measurements_stop_both_modes(case):              # Review Focus 5
    for mode in ("regen", "shadow"):
        sent, rec = run_loop(case, mode=mode, fn=teacher_oracle(case), source=drop_can_after(t=1.5), return_rec=True)
        assert rec["stopped"] and "实测中断" in rec["stopped"] and max(rec["t_send"]) < 1.5 + 0.031 + 0.05

def test_written_files_pass_replay_result_checks(case, tmp_path):
    p_rec, p_seq = RP2.write_outputs(tmp_path, "p1_gregen1", rec_ok(case), mode="regen", net_sha="abc")
    h, z = json.loads(p_rec.read_text()), np.load(p_seq)
    assert h["cmd_hash"] == json.loads(str(z["source"]))["cmd_hash"] and len(h["t_send"]) == len(z["cmd7"])
```

  - 上面用到的 `write_outputs(sess, rid, rec, mode, net_sha) -> tuple[Path, Path]`（写两个文件）属于本任务。
  - `run_loop` 等是测试文件里的工具函数。

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现 `Recorded`、`Live`、`loop`、`write_outputs`、`dry`、`run`**
  - 推理在 CPU 上做（`torch.set_num_threads(1)`）。
  - `dry` 报告每帧计算耗时的 p50、p99、最大值；p99 ≥ 10 ms 时打印 ⛔，并且不允许上机。
- [ ] **Step 4: 用户运行，全部通过；再跑 `dry`（p44，源用 `sessions/20261106q/replay_p44_qshift1_r1.json`）**。预期：输入逐位一致，耗时 p99 < 10 ms。
- [ ] **Step 5: 上机试跑（用户在场；先当天标定，结束后做收尾核对）**
  - p23、p44 各跑一遍 `--mode shadow`。看网络在真实实测下的输出和逐帧检查记录，**有触发就停下讨论**。
  - 没有问题再各跑一遍 `--mode regen`。**停机就停下讨论（设计 §4.6）。**
  - 会话目录用单独的名字，编号 `{轨迹}_tshadow1`、`{轨迹}_tregen1`。
- [ ] **Step 6: 用户要求时提交**

### Task 7: 评价集 valregen

**Files:**
- Modify: `lerobot-piper/sim/b4_eval.py`：`CONDS`、`ROUNDS`、`PREFIX`、`_ids`、`cmd_replay`、`cmd_analyze`、命令行参数 `--regen-net`，文件头用法同步更新。
- Test: `lerobot-piper/sim/test_b4_eval.py`

**Interfaces:**
- Consumes：任务 6 的命令行 `run`，以及它写出的两个文件；`analyze_results`、`pick_session`、`B.replay_result`。
- Produces：
  - `CONDS["valregen"] = ("shift", "regen")`；`ROUNDS["valregen"] = 2`；`PREFIX["valregen"] = "g"`；`_ids("valregen")` 返回验证集。
  - `cmd_replay("valregen", sess, ids, net_path)`：每遍用子进程跑 `b4_regen_replay.py run --mode shadow|regen`（shift 用 shadow、regen 用 regen）；已开始过的跳过；返回码非零就停。
  - `regen_stops(sessions: list[Path], ids: list[str]) -> list[str]`：返回 regen 条件里 `stopped` 不为空的那些遍的 rid。
  - `cmd_analyze("valregen", net_path)`：`analyze_results(..., judged="regen")`。`regen_stops` 非空时，判定改为「不通过：② 触发停机（rid 列表）」。写 `outputs/b4/eval/valregen/summary.json`。
  - 只报告的两项（设计 §4.8），逐遍算，按条件取中位并给出 regen 相对 shift 的变化，不进判定：
    - `joint_nodelay(ep: dict, a6: np.ndarray, free: np.ndarray) -> float`：五个会动关节的 RMS（mrad），参考是原始指令在 `t_send` 之间的线性插值，也就是不含延迟的参考；
    - `joint_j4(ep: dict, a6: np.ndarray, free: np.ndarray) -> float`：j4 相对原始指令零阶保持的 RMS（mrad）。
    - `free` 同主指标的评价帧（`shift_replay.masks` 的编码器那一份）。

- [ ] **Step 1: 写测试**

```python
def test_valregen_plan_alternates_the_order_each_round():
    assert E.plan("valregen", ["p1"]) == [("p1", "shift", 1), ("p1", "regen", 1), ("p1", "regen", 2), ("p1", "shift", 2)]

def test_a_regen_stop_makes_the_verdict_fail(tmp_path):
    sess = fake_valregen_session(tmp_path, stopped={"p1_gregen2"})
    assert E.regen_stops([sess], ["p1"]) == ["p1_gregen2"]
    out = E.cmd_analyze_from(tmp_path, "valregen", net_sha="abc")
    assert out["verdict"].startswith("不通过：② 触发停机")

def test_old_sets_still_read_only_their_own_prefixes(tmp_path):
    fake_valregen_session(tmp_path)
    assert E.pick_session("p1", [tmp_path], ("shift", "net"), 2, {"net": "abc"}, "e") is None

def test_nodelay_deviation_is_zero_for_motion_on_the_interpolated_command():
    ep, a6, free = ep_following_linear_interp(lag_s=0.0)
    assert E.joint_nodelay(ep, a6, free) < 1e-9
    ep2, _, _ = ep_following_linear_interp(lag_s=0.03)
    assert E.joint_nodelay(ep2, a6, free) > 0.1
```

  - 测试里的 `cmd_analyze_from(root, set_, net_sha)` 是可选的薄包装：只有当 `cmd_analyze` 难以对临时目录做测试时才加；也可以直接改成对 `cmd_analyze` 的测试，执行者自定。
- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**；旧的 `analyze --set val`、`--set valseq` 复算结果不变。
- [ ] **Step 5: 真机评价（用户在场）**
  - 当天标定 → `replay --set valregen` 分两批，每条轨迹的 4 遍在同一批内跑完，每批结束后做收尾核对 → `analyze --set valregen`。
  - 共 40 遍；判定按设计 §4.8。
- [ ] **Step 6: 用户要求时提交**

## 本计划不含（设计 §4.8–4.9，另行决定）

- 「去掉 IMU」的真机对比（② 有效之后再做）。
- DAgger 迭代、超过教师的办法。
- 结果图：复用 `sim/fig_b4_report.py`、`sim/fig_b4_seq.py` 的样式，按需另做。
