# B3（误差反馈）实施计划

> **2026-11-12 待修订，暂不要照着执行。** 审阅发现：标签「teacher24 − δa」学出的反馈在闭环里会发散，只量幅度的增益检查拦不住；另估出误差反馈相对 B4 的上限约 −6…−7%。改法和下一步待用户定。

> **给执行者：** 必须用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐个任务实现。步骤用 `- [ ]` 勾选跟踪。

**Goal：** 按设计 v0.1，在 B4 网络上加「理想 − 实测」的误差输入，用带已知扰动 δa 的闭环回放迭代训练 3 次，最后在验证轨迹上和 B4、shift84 做真机比较。

**Architecture：**
- **训练和在线共用的量放在 `lerobot/common/policies/act/b4_online.py`**：误差输入 `err_tokens`、网络的误差层。离线重算和在线算出的输入必须逐位一致。
- **在线回放不另起一套**：`sim/b4_regen_replay.py` 的 `loop`、`run` 加两个可选参数（误差输入用的理想、δa），B4 的行为不变。
- **B3 的离线部分是新文件 `sim/b3.py`**（δa、建数据、训练、上机前检查、冻结）；**上机排程是新文件 `sim/b3_run.py`**（逐遍回放、试跑判定、每次迭代的偏差报告）。输出到 `outputs/b4/b3/`。
- **评价**加在 `sim/b4_ideal_check.py`（评价集 `b3val`）。
- **两个阶段：** 任务 1–8 只写代码、跑合成数据的测试，不动机械臂；任务 9 是上机流程，每一步都由用户在场运行。

**Tech Stack：** Python（conda 环境 `lerobot`）、numpy、scipy、PyTorch、mujoco（只用运动学）、pytest。

**Spec：** `lerobot-piper/reports/PLAN_b3.md`（v0.1，用户 2026-11-12 批准）。B4 的设计和实现：`reports/PLAN_b4_regen.md`、`reports/PLAN_IMPL_b4_regen.md`。执行者都要读。

**执行方式（CLAUDE.md）：**
- 每个任务开头，按 `reports/ADVISOR_STANDARD.md` 逐条回答第 1–6 条。
- 每个任务做完：说明改了什么、为什么这样改、关键代码怎么工作；给出运行命令和预期结果，**测试由用户运行**；等用户反馈后再做下一个。
- 只在用户要求时提交：在 develop 分支上，提交信息不加 `Co-Authored-By`，只 `git add` 本任务的文件。
- 不改 `sim/replay_probe.py`；不碰用户未提交的删除。
- 写代码遵守四条准则：防过度防御、入口单次校验、拒绝提前抽象、控制复杂度。
- 训练（`b3.py train`）和一切连机械臂的命令由用户在自己的终端里跑。

## Global Constraints

- **帧与网格：** `DT = 0.05` s；保持段 `HOLD = 20` 帧；理想的网格 `ideal_motion.DTG = 0.005` s，第 i 个网格点在第 0 帧发令时刻之后 i·DTG 秒。
- **名词：** teacher24 = `IM.earlier(教师 cmd7, 0.024)`；shift84 = `R.shift_cmd(tid, IM.LEAD_S)`；窗 w = `seq_base.npz` 的 `w`，形状 (n, 5)，列对应五个会动关节 `O.ACTIVE = (0, 1, 2, 4, 5)`。
- **δa：** 五个会动关节各自独立的高斯白噪声（20 Hz）→ 2 阶 Butterworth、截止 1 Hz、零相位 → 整条缩放到 RMS 2 mrad → 限幅 ±6 mrad → 乘窗 w；j4 和夹爪为 0。
  随机种子 = (迭代号, 轨迹号的数字, redo)。每条轨迹每次迭代跑两遍：+δa、−δa。
- **实发与标签：** 实发 = `O.regen_command(网络第一帧, shift84[k], w[k])` 的前 6 维 + da[k]；标签 = teacher24 − da。da 指已经乘过窗和符号的那一份 (n, 6)。
- **误差输入**（每个过去帧 75 维，8 帧）：
  - 编码器误差：理想关节角 − 编码器窗口，10 个采样 × 6 关节（rad）；IMU 误差：理想角速度 − IMU 窗口的角速度，5 个采样 × 3 轴（rad/s）。
  - 都乘采样所在帧的窗：会动关节用自己的，j4 和 IMU 用五个关节里最小的。
  - 采样时刻 → 理想的时间轴：只用已经发出的各帧的发令时刻；最后一帧之后按 1 秒 = 1 秒外推；IMU 采样时刻先减 `calib["delay_s"]`。
  - 理想关节角按网格线性插值；理想角速度取所在格子的值（同 `IdealCmd` 的速度、同 `ideal_motion.gyro_ideal`）。
- **实测截止时刻：** 回放记录里有 `cutoff` 就用记录的（在线回放都有）；没有就用 `t_send[k] − 0.010`。
- **网络：** B4 的 `RegenNet` 不变，过去 8 个单元的嵌入各加上一层 `Linear(75, d)`，权重和偏置初始为 0。过去单元里的已发指令、编码器、IMU 原始窗口照旧置 0（`drop = ("sent", "enc", "imu")`）。
  误差的标准化不减均值（mu_e = 0），sd_e = 训练集各列的标准差（下限 1e-6）。输出仍是之后 10 帧 × 6 关节的绝对指令，没有把 ACT、Shift 或 B4 的输出加回去的通路。
- **B3-0：** B4 冻结网络的权重 + 零误差层，输出与 B4 逐位相同。每次迭代的训练都从 B3-0 重新开始，全部权重都训练。
- **训练：** 数据 = 迭代 1…r 里成对有效的训练轨迹 δa 回放；损失同 B4（mrad²、各关节等权、第一帧 ×9、只算窗 > 0 的帧）；AdamW，lr 1e-4 在 60 轮里按余弦降到 0，weight_decay 0.01，批 1024；
  留验证第一帧损失最小的那一轮；种子 0、1、2，按验证第一帧损失选。去掉 IMU 误差的对照同样训 3 个种子。
- **迭代：** 3 次。每次训练 26 条 × 2 + 验证 7 条（p17 p23 p28 p30 p31 p39 p44）× 2 = 66 遍，空桌，都走在线回放（迭代 r 跑 B3-(r−1)）。
  会话目录 `outputs/b4/b3/sessions/<名字>`；回放编号 `{轨迹}_d{迭代}p`（+δa）、`{轨迹}_d{迭代}m`（−δa）。
- **停机约定：** 停机的一遍和它成对的另一遍都不进训练、不补跑；同一个网络停机满 2 遍就停下讨论；逐遍朝向偏差 > 2.5°（`B.QC_MAX`）就停。
- **上机前检查的门槛：**
  - 误差利用：没进训练的验证 δa 回放上，五个会动关节各自的第一帧平方误差（对标签，按窗加权），B3 都小于 B3-0。
  - 闭环增益：探针幅度 1 mrad；0、0.5、1、2 Hz 增益 ≤ 2.0；3、4 Hz 增益 ≤ 0.7。
  - 试跑：p23、p44 各 1 遍闭环、不加 δa；不停机；窗 > 0 的帧上网络第一帧与 B4 第一帧之差 RMS ≤ 5 mrad、最大 ≤ 15 mrad。
  - 任何一项不过就停下讨论，不自行调参。
- **评价：** 10 条验证轨迹（同 `b4val`）× shift84 / B4 / B3-3 × 2 轮 = 60 遍，都走在线回放，摆方块；主判定 B3-3 相对 B4（`ST.judge(main="joint_ideal")`）；B3 的回放一旦停机，评价不通过。
- **环境：** 在 `lerobot-piper/` 下运行；命令前加 `OPENBLAS_NUM_THREADS=1`，解释器 `~/miniconda3/envs/lerobot/bin/python`；入口 `gc.disable()`；
  主机不稳定：`build` 跑两次，两次的数据哈希相同才采用。
- **测试命令：** `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b3.py sim/test_b4_regen.py sim/test_b4_regen_replay.py sim/test_ideal_motion.py -q`。

## Review Focus

1. **还没发过帧、或采样早于第 0 帧**（k = 0 和开头几帧）。期望：误差输入全 0 或有限值，不越界、不出 NaN。测试在任务 1。
2. **成对的两遍里有一遍停机、无效或缺失。** 期望：两遍都不进训练，并打印原因。测试在任务 6。
3. **δa 把实发指令推过逐帧检查。** 期望：这一帧不发、停机；迭代 1 事先预检，触发就换种子。测试在任务 4、5。
4. **在线回放的截止时刻比发令早约 22 ms，旧回放是 10 ms。** 期望：离线按记录里的截止时刻重算，和在线算出的输入逐位一致。测试在任务 1、4。
5. **窗为 0 的帧上误差被屏蔽，增益探针会量到 0。** 期望：只在五个关节的窗都 > 0 的连续 80 帧上量；找不到这样的段就跳过并打印，不把量不到当成通过。测试在任务 7。

## 文件

| 文件 | 改动 |
|---|---|
| `lerobot/common/policies/act/b4_online.py` | 加 `N_ERR`、`ERR_IMU`、`IDEAL_DT`、`err_tokens`；`RegenNet` 加误差层；`regen_fn` 支持第三个输入 |
| `sim/ideal_motion.py` | 加 `omega_ideal` |
| `sim/b4_regen.py` | 加 `ideal_arrays`；`_token_fn`、`assemble_regen`、`first_frames` 加参数 `ideal`；截止时刻优先用记录里的 |
| `sim/b4_regen_replay.py` | `loop`、`run` 加 `ideal`、`da`；`write_outputs` 存实际加的 da；`dry` 支持 B3 网络；命令行 `--da` |
| `sim/b3.py`（新） | δa、B3-0、建数据、训练、选种子、上机前检查、冻结 |
| `sim/b3_run.py`（新） | 逐遍回放的排程与预检、试跑判定、每次迭代的偏差报告 |
| `sim/b4_ideal_check.py` | 评价集 `b3val` |
| `sim/test_b3.py`（新）、`sim/test_b4_regen_replay.py`、`sim/test_ideal_motion.py` | 测试 |

---

### Task 1: 误差输入

**Files:**
- Modify: `lerobot/common/policies/act/b4_online.py`（`regen_tokens` 之后）、`sim/ideal_motion.py`、`sim/b4_regen.py`（`_token_fn`、`assemble_regen`、`first_frames`）
- Test: `sim/test_b3.py`（新建）

**Interfaces:**
- Produces：
  - `O.N_ERR = 75`；`O.ERR_IMU = slice(60, 75)`；`O.IDEAL_DT = 0.005`。
  - `O.err_tokens(enc, imw, cutoff, t_send, x, om, delay_s, w) -> np.ndarray`，(8, 75) float32。
    `enc` (80, 6)、`imw` (40, 6) 是 `enc_window`、`imu_window` 的结果；`t_send` 是已发出各帧的发令时刻 (k,)；`x` (N, 6)、`om` (N, 3) 是理想关节角和理想角速度的网格；`w` (n, 5)。
  - `IM.omega_ideal(x, model, sid) -> np.ndarray`，(N, 3)：`om[i]` = 理想在第 i 个格子中点 (i + 0.5)·DTG 的 IMU 角速度（`ideal_imu_traj(...)[1][:, 3:]`），最后一行重复前一行。
  - `R.ideal_arrays(tid) -> tuple[np.ndarray, np.ndarray]`：(x, om)，缓存在 `outputs/b4/ideal/{tid}.npz`；模型和安装点同 `C.make_ctx`（`MODEL_PATH`、`ee_imu_site`）。
  - `R._token_fn(h, base, log, calib, ideal=None)`：给了 `ideal = (x, om)` 时 `tokens(k, sent6)` 返回 `(past, fut, err)`，否则同原来。
  - `R.assemble_regen(h, base, log, teacher7, calib, ideal=None)`：给了 `ideal` 时结果多一个键 `err` (m, 8, 75)。
  - `R.first_frames(fn, h, base, log, calib, ideal=None)`。

- [ ] **Step 1: 写测试**

```python
"""B3（设计 `reports/PLAN_b3.md`、实施计划 `reports/PLAN_IMPL_b3.md`）的测试。合成数据，不连机械臂。"""
# 文件头同 test_b4_regen.py（gc.disable、sys.path）；另外：
from test_b4_regen import case, recording_fn, rot_z, teacher_oracle, with_rows_after  # noqa: F401
from test_ideal_motion import smooth_cmd

CAL = lambda case: {**case.calib, "delay_s": 0.004}   # noqa: E731


@pytest.fixture(scope="module")
def ideal(case):
    x = IM.ideal(case.base["a7"])
    return x, 0.1 * np.sin(np.arange(len(x))[:, None] * 0.01 + [0.0, 1.0, 2.0])


def err_inputs(lag=0.03, k=30, n=60):
    """理想：六个关节都是斜率 0.2 rad/s 的斜线、角速度恒为 (0.1, 0.2, 0.3)；实测 = 理想整体晚 lag 秒、角速度少 (0.01, −0.02, 0.03)。
    已发 k 帧，发令时刻 1.0 + 0.05·i；截止时刻在最后一帧之后 28 ms。"""
    N = (n - 1) * 10 + 1
    t_send = 1.0 + 0.05 * np.arange(k)
    cutoff = t_send[-1] + 0.028
    tq = cutoff - 0.4 + 0.005 * np.arange(1, 81)
    return dict(enc=0.2 * (tq - 1.0 - lag)[:, None] * np.ones(6), imw=np.tile([0.09, 0.22, 0.27, 0, 0, 0], (40, 1)), cutoff=cutoff,
                t_send=t_send, x=0.2 * 0.005 * np.arange(N)[:, None] * np.ones(6), om=np.tile([0.1, 0.2, 0.3], (N, 1)),
                delay_s=0.004, w=np.ones((n, 5)))


def test_ideal_grid_constant_matches_ideal_motion():
    assert O.IDEAL_DT == IM.DTG and O.N_ERR == 75


def test_err_tokens_recover_a_pure_delay():
    e = O.err_tokens(**err_inputs())
    assert e.shape == (8, 75) and e.dtype == np.float32
    assert np.allclose(e[:, :60], 0.2 * 0.03, atol=1e-6) and np.allclose(e[:, 60:].reshape(-1, 3), [0.01, -0.02, 0.03], atol=1e-6)


def test_err_tokens_are_zero_where_the_window_is_zero():
    a = err_inputs()
    a["w"][20:26] = 0                                              # 第 20–25 帧所有关节的窗为 0
    a["w"][27, 2] = 0                                              # 第 27 帧只有 j3 的窗为 0
    e = O.err_tokens(**a)
    enc, imu = e[:, :60].reshape(80, 6), e[:, 60:].reshape(40, 3)
    fq = np.floor((a["cutoff"] - 0.4 + 0.005 * np.arange(1, 81) - 1.0) / 0.05).astype(int)
    fi = np.floor((a["cutoff"] - 0.4 + 0.01 * np.arange(1, 41) - 0.004 - 1.0) / 0.05).astype(int)
    assert not enc[(fq >= 20) & (fq <= 25)].any() and not imu[((fi >= 20) & (fi <= 25)) | (fi == 27)].any()
    assert not enc[fq == 27][:, [2, 3]].any() and enc[fq == 27][:, [0, 1, 4, 5]].all() and imu[fi == 28].all()   # j4 取最小的窗


def test_err_tokens_extrapolate_after_the_last_sent_frame():
    a = err_inputs(lag=0.0)
    a["t_send"] = np.r_[a["t_send"][:-1], a["t_send"][-1] + 0.004]          # 最后一帧晚发 4 ms
    e = O.err_tokens(**a)[:, :60].reshape(80, 6)
    late = a["cutoff"] - 0.4 + 0.005 * np.arange(1, 81) > a["t_send"][-1]
    assert late.any() and np.allclose(e[late], -0.2 * 0.004, atol=1e-6)


def test_err_tokens_before_the_first_frames_are_zero_or_finite():           # Review Focus 1
    a = err_inputs()
    assert not O.err_tokens(**{**a, "t_send": a["t_send"][:0]}).any()
    assert np.isfinite(O.err_tokens(**{**a, "t_send": a["t_send"][:1], "cutoff": 1.02})).all()


def test_omega_lookup_matches_gyro_ideal():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    x = IM.ideal(smooth_cmd())
    om, calib = IM.omega_ideal(x, model, sid), {"R": rot_z(30).tolist(), "gyro_bias": [0.01, 0.0, 0.0], "delay_s": 0.004}
    ts = 10.0 + 0.05 * np.arange(80)
    ti = np.arange(ts[0] + 0.5, ts[-1] - 0.5, 0.0103)
    ref = IM.gyro_ideal({"ti": ti, "t_send": ts}, x, calib, model, sid)
    got = om[np.floor(IM.nominal(ti - 0.004, ts) / IM.DTG).astype(int)] @ np.asarray(calib["R"]).T + calib["gyro_bias"]
    assert om.shape == (len(x), 3) and np.abs(ref).max() > 0.05 and np.abs(got - ref).max() < 2e-3


def test_tokens_ignore_future_send_times_and_late_samples(case, ideal):     # 改未来，过去不变
    h, k, sent = case.replays[1], 60, case.replays[1]["cmd"][:, :6]
    late = with_rows_after(h, frame=k, seconds=0.01)
    late.update(t_send=np.r_[h["t_send"][:k], h["t_send"][k:] + 0.003], cutoff=h["t_send"] - O.CUT_S)
    a = R._token_fn(h, case.base, case.log, CAL(case), ideal)(k, sent)
    b = R._token_fn(late, case.base, case.log, CAL(case), ideal)(k, sent)
    assert len(a) == 3 and a[2].shape == (8, 75) and all(np.array_equal(p, q) for p, q in zip(a, b))


def test_tokens_use_the_recorded_cutoff(case, ideal):                       # Review Focus 4
    h, k = case.replays[1], 60
    past = R._token_fn({**h, "cutoff": h["t_send"] - 0.022}, case.base, case.log, CAL(case), ideal)(k, h["cmd"][:, :6])[0]
    enc, _ = O.enc_window(h["can_q200"], 1000.0, h["t_send"][k] - 0.022)
    assert np.array_equal(past[:, 12:72].reshape(80, 6), enc.astype(np.float32))


def test_assemble_adds_err_only_when_an_ideal_is_given(case, ideal):
    h = case.replays[1]
    d = R.assemble_regen(h, case.base, case.log, case.teacher7, CAL(case), ideal)
    i = len(d["k"]) // 2
    assert d["err"].shape == (len(d["k"]), 8, 75)
    assert np.array_equal(d["err"][i], R._token_fn(h, case.base, case.log, CAL(case), ideal)(d["k"][i], h["cmd"][:, :6])[2])
    assert "err" not in R.assemble_regen(h, case.base, case.log, case.teacher7, case.calib)
```

- [ ] **Step 2: 用户运行，确认失败**（`O.err_tokens` 等不存在）
- [ ] **Step 3: 实现**。`err_tokens` 的算法（签名和测试定不下来的部分）：

```
k = len(t_send)；k == 0 时返回全 0
采样时刻：编码器 cutoff − WIN_S + ENC_DT·(1…80)；IMU cutoff − WIN_S + IMU_DT·(1…40) − delay_s
名义时刻 nom(t)：t < t_send[0] 取 t − t_send[0]；t > t_send[-1] 取 DT·(k−1) + (t − t_send[-1])；其余 np.interp(t, t_send, DT·arange(k))
理想关节角 = x 各列在 IDEAL_DT 网格上对 nom 的线性插值（np.interp，两端保持）
理想角速度 = om[clip(floor(nom / IDEAL_DT), 0, N−1)]
帧号 f = clip(floor(nom / DT), 0, n−1)；窗 6 列 = w[f] 放到会动关节的列，j4 列 = w[f].min(1)；IMU 的窗 = w[f].min(1)
返回 concat([((理想关节角 − enc) × 窗).reshape(8, 60), ((理想角速度 − imw[:, :3]) × 窗).reshape(8, 15)], 1).astype(float32)
```

  `_token_fn` 里传给 `err_tokens` 的是 `ts[:k]` 和 `calib["delay_s"]`；截止时刻 `cuts = h["cutoff"] if "cutoff" in h else ts − O.CUT_S`。
- [ ] **Step 4: 用户运行，全部通过**；B4 的旧测试不变。
- [ ] **Step 5: 用户要求时提交**

### Task 2: 网络的误差层

**Files:**
- Modify: `lerobot/common/policies/act/b4_online.py`（`RegenNet`、`regen_fn`）
- Test: `sim/test_b3.py`

**Interfaces:**
- Produces：
  - `O.RegenNet(..., n_err: int = 0, err_drop_imu: bool = False)`；`forward(past, fut, err=None)`。
    `n_err > 0` 时：`emb_e = nn.Linear(n_err, d)`（权重、偏置置 0）；buffer `mu_e`（0）、`sd_e`（1）；不进 state_dict 的 buffer `keep_e`（全 1，`err_drop_imu` 时 `ERR_IMU` 那几列为 0）；
    过去单元的嵌入 = `emb_p(p) + emb_e((err − mu_e) / sd_e × keep_e)`。
  - `O.regen_fn(m, device)`：`m["arch"]` 里有 `n_err` 时返回 `fn(past, fut, err)`，否则 `fn(past, fut)`；两种都带属性 `fn.n_err`。CUDA graph 的输入缓冲多一块 (1, 8, n_err)。

- [ ] **Step 1: 写测试**

```python
ARCH = {"n_past": 102, "n_fut": 6, "d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0}
DROP = ("sent", "enc", "imu")


def b4_and_b3(**kw):
    torch.manual_seed(0)
    b4, b3 = O.RegenNet(**ARCH, drop=DROP).eval(), O.RegenNet(**ARCH, drop=DROP, n_err=O.N_ERR, **kw).eval()
    return b4, b3, b3.load_state_dict(b4.state_dict(), strict=False).missing_keys


def rand_inputs(b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 8, 102, generator=g), torch.randn(b, 10, 6, generator=g), torch.randn(b, 8, 75, generator=g)


def test_zero_error_layer_reproduces_b4_bitwise():
    b4, b3, missing = b4_and_b3()
    p, f, e = rand_inputs()
    assert set(missing) == {"emb_e.weight", "emb_e.bias", "mu_e", "sd_e"} and torch.equal(b3(p, f, e), b4(p, f))


def test_error_layer_moves_the_output_once_it_has_weights():
    _, b3, _ = b4_and_b3()
    torch.nn.init.normal_(b3.emb_e.weight, std=0.1)
    p, f, e = rand_inputs()
    assert not torch.equal(b3(p, f, e), b3(p, f, torch.zeros_like(e)))


def test_err_drop_imu_ignores_the_imu_columns():
    _, b3, _ = b4_and_b3(err_drop_imu=True)
    torch.nn.init.normal_(b3.emb_e.weight, std=0.1)
    p, f, e = rand_inputs()
    e2 = e.clone()
    e2[..., O.ERR_IMU] += 5.0
    assert torch.equal(b3(p, f, e), b3(p, f, e2))


def test_regen_fn_takes_the_error_input_and_reports_n_err():
    b4, b3, _ = b4_and_b3()
    torch.nn.init.normal_(b3.emb_e.weight, std=0.1)
    fn = O.regen_fn({"state_dict": b3.state_dict(), "drop": list(DROP), "arch": {**ARCH, "n_err": 75}})
    p, f, e = (x[0].numpy() for x in rand_inputs(1))
    ref = b3(*(torch.from_numpy(x)[None] for x in (p, f, e)))[0].double().detach().numpy()
    assert fn.n_err == 75 and np.array_equal(fn(p, f, e), ref)
    assert O.regen_fn({"state_dict": b4.state_dict(), "drop": list(DROP), "arch": ARCH}).n_err == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="没有 GPU")
def test_cuda_graph_with_the_error_input_matches_cpu():
    _, b3, _ = b4_and_b3()
    torch.nn.init.normal_(b3.emb_e.weight, std=0.1)
    m = {"state_dict": b3.state_dict(), "drop": list(DROP), "arch": {**ARCH, "n_err": 75}}
    cpu, gpu = O.regen_fn(m), O.regen_fn(m, device="cuda")
    for s in range(3):                                            # 连续几帧：图回放不能串数据
        x = [t[0].numpy() for t in rand_inputs(1, seed=s)]
        assert np.allclose(gpu(*x), cpu(*x), atol=1e-5)
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**（`RegenNet` 和 `regen_fn` 各改几行；B4 的权重文件没有 `n_err`，按默认 0 构造，行为不变）
- [ ] **Step 4: 用户运行，全部通过**
- [ ] **Step 5: 用户要求时提交**

### Task 3: δa 与 B3-0

**Files:**
- Create: `sim/b3.py`
- Test: `sim/test_b3.py`

**Interfaces:**
- Consumes：任务 2 的 `RegenNet(n_err=…)`；`R.freeze_regen(run, out)`；`R.REGEN / "frozen.json"`（B4 冻结网络）。
- Produces：
  - 常量：`B3 = C.OUT / "b3"`；`N_IT = 3`；`DA_RMS, DA_MAX, DA_FC = 2e-3, 6e-3, 1.0`。
  - `delta(w: np.ndarray, it: int, tid: str, redo: int = 0) -> np.ndarray`：(n, 6)，已乘窗，未乘符号。
  - `init0(b4: Path = R.REGEN / "frozen.json", out: Path = B3 / "it0") -> Path`：写 `out/run.pt`（B4 的 state_dict 载入 `n_err = 75` 的网络后再存，`arch` 加 `n_err`，其余键照抄）并冻结成 `out/frozen.json`。
  - `net_path(it: int) -> Path`：`B3 / f"it{it}" / "frozen.json"`。
  - 命令行：`python -u sim/b3.py init0`。

- [ ] **Step 1: 写测试**

```python
def test_delta_follows_the_design():
    w = np.ones((4000, 5))
    w[100:160], w[160:170] = 0, np.linspace(0, 1, 10)[:, None]
    d = B3.delta(w, it=1, tid="p23")
    a = d[200:, list(O.ACTIVE)]
    f, p = np.fft.rfftfreq(len(a), 0.05), (np.abs(np.fft.rfft(a, axis=0)) ** 2).sum(1)
    assert d.shape == (4000, 6) and not d[:, 3].any() and not d[100:160].any()
    assert np.abs(d).max() <= 6e-3 and np.allclose(np.sqrt((a ** 2).mean(0)), 2e-3, rtol=0.1)
    assert p[f > 1.0].sum() / p.sum() < 0.15 and p[f > 2.0].sum() / p.sum() < 0.02
    assert np.abs(np.corrcoef(a.T) - np.eye(5)).max() < 0.2                       # 各关节独立
    assert np.array_equal(d, B3.delta(w, 1, "p23"))
    assert all(not np.array_equal(d, x) for x in (B3.delta(w, 2, "p23"), B3.delta(w, 1, "p24"), B3.delta(w, 1, "p23", redo=1)))


def frozen_b4(d: Path) -> Path:
    """小的随机 B4 网络，按 R.freeze_regen 的格式冻结到 d/b4/frozen.json。"""
    torch.manual_seed(0)
    run = d / "b4run.pt"
    torch.save({"state_dict": O.RegenNet(**ARCH, drop=DROP).state_dict(), "drop": list(DROP), "seed": 0, "data_sha": "t", "arch": ARCH}, run)
    (d / "b4").mkdir()
    return R.freeze_regen(run, out=d / "b4")


def test_b3_0_equals_b4_bitwise(tmp_path):
    f4 = frozen_b4(tmp_path)
    fn4, fn0 = O.load_regen(str(f4))[0], O.load_regen(str(B3.init0(f4, tmp_path / "it0")))[0]
    p, f, e = (x[0].numpy() for x in rand_inputs(1))
    assert fn0.n_err == 75 and np.array_equal(fn0(p, f, e), fn4(p, f))
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**。`delta` 的算法：

```python
rng = np.random.default_rng([it, int(tid[1:]), redo])
z = sosfiltfilt(butter(2, DA_FC, fs=1 / O.DT, output="sos"), rng.normal(size=(len(w), 5)), axis=0)
z = np.clip(z / np.sqrt((z ** 2).mean(0)) * DA_RMS, -DA_MAX, DA_MAX) * w      # 先缩放到 RMS，再限幅，最后乘窗
```

- [ ] **Step 4: 用户运行，全部通过**；再跑 `python -u sim/b3.py init0`。预期：打印 `outputs/b4/b3/it0/frozen.json` 和权重哈希。
- [ ] **Step 5: 用户要求时提交**

### Task 4: 在线回放支持误差输入和 δa

**Files:**
- Modify: `sim/b4_regen_replay.py`（`loop`、`write_outputs`、`dry`、`run`、`main`，文件头用法）
- Test: `sim/test_b4_regen_replay.py`

**Interfaces:**
- Consumes：任务 1 的 `O.err_tokens`、`R.ideal_arrays`、`R.first_frames(..., ideal)`；任务 2 的 `fn.n_err`。
- Produces：
  - `loop(n, base, log, fn, calib, offset, lo, hi, source, mode, send, clock, ideal=None, da=None) -> dict`：
    - `ideal = (x, om)` 时每帧算 `O.err_tokens(enc, imw, cut, 已记录的 t_send, x, om, calib["delay_s"], w)`，调用 `fn(past, fut, err)`；
    - `da` (n, 6) 只在 regen 模式起作用：`u7[:6] += da[k]` 放在 `regen_command` 之后、`step_check` 之前；
    - 记录多一个键 `da`（每帧实际加上的 6 维，没加就是 0）；`net_u` 仍是加 δa 之前的指令，`cmd` 是真正发出的。
  - `write_outputs`：回放序列文件里的 `da` 存 `rec["da"]`。
  - `dry(tid, source_json, net_path)`：网络的 `fn.n_err > 0` 时，在线和离线两边都带 `R.ideal_arrays(tid)`。
  - `run(sess, tid, rid, mode, net_path, da_path=None) -> int`：`fn.n_err > 0` 时带 `R.ideal_arrays(tid)`（连机械臂之前算好）；`da_path` 指向带键 `da` 的 npz。
  - 命令行：`run … [--da <npz>]`。

- [ ] **Step 1: 写测试**。`run_loop` 加三个参数：`early=0.0`（clock 返回 `ts[k] − early`，send 仍返回 `ts[k]`）、`calib=None`（默认 `case.calib`）、`**kw` 原样传给 `loop`。

```python
from test_b3 import CAL, ideal  # noqa: F401


def test_da_is_added_after_the_switch_and_recorded(case):
    da = np.zeros((case.n, 6))
    da[:, 0] = 0.001 * (case.base["w"][:, 0] > 0)
    sent, rec = run_loop(case, "regen", teacher_oracle(case), return_rec=True, da=da)
    assert np.allclose(np.array(sent)[:, :6], case.teacher7[:, :6] + da, atol=1e-7)
    assert np.allclose(np.array(rec["net_u"])[:, :6], case.teacher7[:, :6], atol=1e-7) and np.allclose(rec["da"], da, atol=1e-9)


def test_shadow_does_not_apply_da(case):
    sent, rec = run_loop(case, "shadow", teacher_oracle(case), return_rec=True, da=np.full((case.n, 6), 0.001))
    assert np.array_equal(np.array(sent), case.base["s7"].astype(np.float32)) and not np.asarray(rec["da"]).any()


def test_step_check_sees_the_perturbed_command(case):               # Review Focus 3
    da = np.zeros((case.n, 6))
    da[40, 0] = 0.2
    sent, rec = run_loop(case, "regen", teacher_oracle(case), return_rec=True, da=da)
    assert len(sent) == 40 and "每帧变化" in rec["stopped"]


def test_online_inputs_equal_the_offline_assembly_from_the_record(case, ideal):   # Review Focus 4
    seen = []
    _, rec = run_loop(case, "shadow", recording_fn(seen, lambda p, f, e: f), return_rec=True, early=0.012, calib=CAL(case), ideal=ideal)
    h = {**case.replays[0], "cutoff": np.asarray(rec["cutoff"])}
    d = R.assemble_regen(h, case.base, case.log, case.teacher7, CAL(case), ideal)
    assert np.allclose(rec["cutoff"], case.replays[0]["t_send"] - 0.022)
    for i, k in enumerate(d["k"]):
        assert all(np.array_equal(seen[k][j], d[key][i]) for j, key in enumerate(("past", "fut", "err")))


def test_seq_file_stores_the_applied_da(case, tmp_path):
    da = np.zeros((case.n, 6))
    da[:, 1] = 0.0005 * (case.base["w"][:, 1] > 0)
    _, rec = run_loop(case, "regen", teacher_oracle(case), return_rec=True, da=da)
    _, p_seq = RP2.write_outputs(tmp_path, "p1_d1p", rec, mode="regen", net_sha="abc")
    assert np.allclose(np.load(p_seq)["da"], da, atol=1e-7)
```

  `recording_fn` 的 `fn(past, fut)` 要能多收一个误差输入：改成 `fn(*xs)`，`seen.append(xs)`，`out(*xs)`。
- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**；再跑
  `python -u sim/b4_regen_replay.py dry --tid p44 --source outputs/b4/teacher/sessions/20261106q/replay_p44_qshift1_r1.json --net outputs/b4/b3/it0/frozen.json`。
  预期：在线与离线逐位一致；每帧计算耗时 p99 < 10 ms。
- [ ] **Step 5: 用户要求时提交**

### Task 5: 采数据的排程与预检

**Files:**
- Create: `sim/b3_run.py`
- Test: `sim/test_b3.py`

**Interfaces:**
- Consumes：`B3.delta`、`B3.net_path`；`R.cases()`、`R._load`、`R.preview`、`R.shift_cmd`；`O.step_check`；`RR.run_child`；`B.replay_result`、`B.session_ctx`、`B.check_session`、`B.QC_MAX`；任务 4 的命令行 `run --da`。
- Produces：
  - `rid(tid: str, it: int, sign: int) -> str`：`f"{tid}_d{it}{'p' if sign > 0 else 'm'}"`。
  - `plan(ids: list[str]) -> list[tuple[str, int]]`：每条轨迹的两遍连着跑；第偶数条先 +、第奇数条先 −。
  - `todo(plan: list, done: set[str], stopped: set[str], it: int) -> list[tuple[str, int]]`：去掉已经开始过的遍，以及成对的另一遍已经停机的遍。
  - `preflight(u6: np.ndarray, da: np.ndarray, a6: np.ndarray, start6: np.ndarray, lo, hi) -> str | None`：把 `u6 + da` 逐帧过 `O.step_check`（上一帧 = 自己的上一帧，第 0 帧之前是 `start6`），返回第一个 `"第 k 帧：原因"` 或 None。
  - `stops(it: int, root: Path = B3.B3 / "sessions") -> list[str]`：这次迭代里 `stopped` 不为空的回放编号。
  - `cmd_replay(it: int, sess: Path, n_next: int | None) -> int`，命令行 `python -u sim/b3_run.py replay --it r --session <目录> [--next N]`：
    1. `B.check_session`、写 `session.json`（同 `b4_ideal_check.cmd_replay`）；`stops(it)` 已满 2 遍就 ⛔ 退出。
    2. 轨迹 = `R.cases()` 的训练和验证轨迹；编号 = 所有会话里已有的回放记录 → `todo`；`--next N` 只取前 N 条轨迹。
    3. 每条轨迹先生成 da：迭代 1 用 B4 冻结网络预演出整条指令（`R.preview`，实测窗口取这条轨迹教师会话的第一遍有效回放，Shift = shift84），
       `preflight` 对 +、− 两遍都过才用；不过就 `redo += 1` 重新生成，最多 5 次，仍不过就跳过这条轨迹并打印。B4 自己的预演就触发检查的轨迹同样跳过并打印。
       迭代 2、3 不预检，`redo = 0`。两遍的 da 文件 `da_{rid}.npz`（键 `da` = 符号 × `delta`，`redo`）在跑第一遍之前都写好。
    4. 开始前提示一次「空桌，可以运动就按回车（q 退出）」。每遍用子进程跑 `b4_regen_replay.py run --mode regen --net net_path(it − 1) --da …`。
    5. 子进程返回 0：`B.replay_result` 取朝向偏差，无效就打印，> `B.QC_MAX` 就 ⛔ 退出。返回非 0：打印「停机，这一遍和成对的另一遍作废」，
       `stops(it)` 满 2 遍就 ⛔ 退出，否则接着跑下一条轨迹。

- [ ] **Step 1: 写测试**

```python
def test_plan_runs_both_signs_back_to_back_and_alternates_which_goes_first():
    assert BR.plan(["p1", "p2"]) == [("p1", 1), ("p1", -1), ("p2", -1), ("p2", 1)]
    assert BR.rid("p23", 2, 1) == "p23_d2p" and BR.rid("p23", 2, -1) == "p23_d2m"


def test_todo_skips_started_runs_and_the_partner_of_a_stopped_run():
    plan = BR.plan(["p1", "p2", "p3"])
    assert BR.todo(plan, done={"p1_d1p", "p2_d1m"}, stopped={"p2_d1m"}, it=1) == [("p1", -1), ("p3", 1), ("p3", -1)]


def test_preflight_reports_the_first_violation():                           # Review Focus 3
    u = np.zeros((50, 6))
    lo, hi = -np.ones(6), np.ones(6)
    da = np.zeros((50, 6))
    assert BR.preflight(u, da, u, u[0], lo, hi) is None
    da[30, 4] = 0.03                                                        # j5 每帧变化上限 0.025
    assert BR.preflight(u, da, u, u[0], lo, hi).startswith("第 30 帧：每帧变化")


def test_stops_lists_only_this_iteration(tmp_path):
    s = tmp_path / "a"
    s.mkdir()
    for name, stopped in (("p1_d1p", "x"), ("p1_d1m", None), ("p2_d2p", "y"), ("p3_t1", "z")):
        (s / f"replay_{name}_r1.json").write_text(json.dumps({"stopped": stopped}))
    assert BR.stops(1, tmp_path) == ["p1_d1p"] and BR.stops(2, tmp_path) == ["p2_d2p"]
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**
- [ ] **Step 5: 用户要求时提交**

### Task 6: 建数据与训练

**Files:**
- Modify: `sim/b3.py`
- Test: `sim/test_b3.py`

**Interfaces:**
- Consumes：`R.cases()`、`R.valid_replays`、`R._calib`、`R._log`、`R.ideal_arrays`、`R.assemble_regen(..., ideal)`、`R.frame_weights`、`R.MRAD`；`BR.rid`（或在本文件里重复那一行格式，二选一，执行者定）。
- Produces：
  - `build(it: int, out: Path | None = None, loader=None) -> Path`：`B3/it{it}/data.npz`，键 `past fut err Y W k da traj_id rid split it data_sha`。
    - `loader(tid, i) -> (base, log, t24, ideal, [(rid, h, calib, da), …])`：轨迹 tid 在迭代 i 的有效回放（0、1 或 2 遍），默认从磁盘读：
      在 `B3/sessions/*` 里找这两个编号，有效性用 `R.valid_replays`，da 取回放序列文件里的 `da`。
    - 只有 +、− 两遍都有效才收，否则打印原因、两遍都不收。标签 = `t24[:, :6] − da`（第 7 列照抄）；`da` 键存每个样本第一帧的 da (m, 6)。
    - `split`：训练轨迹 `train`、验证轨迹 `val`（同 `R.cases()`）。`data_sha` = `past`、`err`、`Y` 字节的 sha256 前 16 位。
  - `EPOCHS, LR = 60, 1e-4`。
  - `train_b3(data: Path, seed: int, init: Path, drop_imu: bool = False, epochs: int = EPOCHS, lr: float = LR, batch: int = 1024, out: Path | None = None, device: str = "cuda") -> dict`：
    从 `init`（B3-0 的 `run.pt`）载入权重，`mu_*`、`sd_*` 除 `sd_e` 外都沿用；循环同 `R.train_regen`，前向多传 `err`。
    存 `out/s{seed}.pt` 或 `out/s{seed}_noimu.pt`（键同 B4 的权重文件）。返回 `{"path", "val_first", "val_init", "train_loss", "epochs"}`，
    `val_init` = 起点网络在同一批验证样本上的第一帧损失（mrad²）。
  - `select(it: int) -> dict`：种子 0、1、2 各训一次，再各训一次去掉 IMU 的 → `B3/it{it}/selection.json`（`chosen` = 验证第一帧损失最小的完整版，`runs`，`noimu`）。
  - 命令行：`build --it r`、`train --it r`。

- [ ] **Step 1: 写测试**

```python
def b3_runs(case, ideal, kinds: dict):
    """假的 loader：kinds = {(轨迹, 迭代): 有效的符号列表}。每遍的 da 只在 j1 上、±0.5 mrad、乘窗。"""
    def loader(tid, i):
        da = np.zeros((case.n, 6))
        da[:, 0] = 0.0005 * (case.base["w"][:, 0] > 0)
        runs = [(f"{tid}_d{i}{'p' if s > 0 else 'm'}", case.replays[0], CAL(case), s * da) for s in kinds.get((tid, i), ())]
        return case.base, case.log, case.teacher7, ideal, runs
    return loader


ROWS = {"p1": {"split": "train", "passed": True, "session": "s"}, "p2": {"split": "val", "passed": True, "session": "s"}}


def test_build_labels_are_teacher_minus_da_and_pairs_are_kept_whole(case, ideal, tmp_path, capsys):   # Review Focus 2
    z = np.load(B3.build(2, tmp_path, b3_runs(case, ideal, {("p1", 1): (1, -1), ("p1", 2): (1,), ("p2", 1): (1, -1)}), rows=ROWS))
    assert set(z["rid"]) == {"p1_d1p", "p1_d1m", "p2_d1p", "p2_d1m"} and "p1_d2p" in capsys.readouterr().out
    assert set(z["split"][z["traj_id"] == "p1"]) == {"train"} and set(z["split"][z["traj_id"] == "p2"]) == {"val"}
    m = z["rid"] == "p1_d1m"
    t = np.asarray(case.teacher7, np.float32)[np.minimum(z["k"][m][:, None] + np.arange(10), case.n - 1), :6]
    assert np.allclose(z["Y"][m][:, 0] - t[:, 0], -z["da"][m], atol=1e-7) and (z["da"][m][:, 0] <= 0).all() and z["da"][m][:, 0].min() < 0
    assert z["err"].shape[1:] == (8, 75) and z["past"].shape[0] == z["err"].shape[0]


def toy(d: Path, blind: bool = False, n: int = 512) -> tuple[Path, Path]:
    """(数据, 起点)。标签 10 帧都是 −δ（δ 每个样本、每个关节独立，σ = 2 mrad）；误差输入最后一帧 j 关节的 10 个编码器采样 = −δ_j + 0.1 mrad 噪声；
    其余输入是噪声。blind：误差输入全置 0。起点 = 小的随机网络（ARCH + n_err），mu_y = 0、sd_y = 2e-3。后四分之一是验证集。"""


def test_the_error_input_is_what_gets_learned(tmp_path):
    kw = dict(epochs=40, lr=1e-3, batch=64, device="cpu")
    full = B3.train_b3(*toy(tmp_path), seed=0, out=tmp_path / "full", **kw)
    blind = B3.train_b3(*toy(tmp_path, blind=True), seed=0, out=tmp_path / "blind", **kw)
    assert full["val_first"] < 1.0 < 3.0 < blind["val_first"]                # mrad²；δ 的方差是 4


def test_training_starts_from_the_init_and_moves_the_error_layer(tmp_path):
    data, init = toy(tmp_path)
    r = B3.train_b3(data, 0, init, epochs=3, lr=1e-3, batch=64, out=tmp_path / "r", device="cpu")
    a, b = torch.load(init)["state_dict"], torch.load(r["path"])["state_dict"]
    assert torch.equal(a["mu_p"], b["mu_p"]) and torch.equal(a["sd_y"], b["sd_y"]) and not torch.equal(a["emb_e.weight"], b["emb_e.weight"])
    assert not b["mu_e"].any() and (b["sd_e"] != 1).any()                   # 误差不减均值，标准差按训练集
    z, net = np.load(data), O.RegenNet(**torch.load(init)["arch"], drop=DROP)
    net.load_state_dict(a)
    v = z["split"] == "val"
    with torch.no_grad():
        ref = ((net(*(torch.from_numpy(z[k][v]) for k in ("past", "fut", "err")))[:, 0] - torch.from_numpy(z["Y"][v][:, 0])) / 1e-3).pow(2).mean().item()
    assert abs(r["val_init"] - ref) < 1e-3 * ref


def test_noimu_runs_get_their_own_file_and_ignore_imu(tmp_path):
    data, init = toy(tmp_path)
    r = B3.train_b3(data, 1, init, drop_imu=True, epochs=1, batch=64, out=tmp_path / "r", device="cpu")
    assert Path(r["path"]).name == "s1_noimu.pt" and torch.load(r["path"])["arch"]["err_drop_imu"] is True
```

  `build` 的测试用了参数 `rows`（同 `R.build_regen`，默认读 `summary.json`）。`toy` 的两组阈值若在本机不稳，只许调测试里的 `epochs`、`lr`、`n`，不改断言的含义。
- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**
- [ ] **Step 5: 用户要求时提交**

### Task 7: 上机前的检查、冻结、试跑判定、偏差报告

**Files:**
- Modify: `sim/b3.py`（`use_gate`、`open_stretch`、`gain_probe`、`gain_gate`、`check`、`freeze`）、`sim/b3_run.py`（`trial_gate`、`trial`、`report`）
- Test: `sim/test_b3.py`

**Interfaces:**
- Consumes：任务 6 的 `data.npz`、`selection.json`；`R._token_fn(..., ideal)`；`IM.omega_ideal`；`R.first_frames`；`IM.joint_ideal`；`S.masks`。
- Produces（`b3.py`）：
  - `GAIN_F = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0)`；`GAIN_A = 1e-3`；`GAIN_L = 80`。
  - `use_gate(sse_b3: np.ndarray, sse_b0: np.ndarray) -> list[str]`：五个会动关节里 `not sse_b3 < sse_b0` 的项（NaN 算不过），空 = 通过。
  - `open_stretch(w: np.ndarray, L: int = GAIN_L) -> int | None`：第一段连续 L 帧五个关节的窗都 > 0 的起始帧号；没有返回 None。
  - `gain_probe(fn, h, base, log, calib, ideal, pert, j: int, f: float, k0: int, L: int = GAIN_L) -> float`：
    `pert = (dx, dom)` 加到理想上；对第 k0 … k0+L−1 帧分别用原来的和加了扰动的理想算输入（已发指令都取记录里的），网络第一帧第 j 个关节之差 d；
    f = 0 时增益 = |mean(d)| / GAIN_A；否则把 d 对 sin、cos(2π f k·DT) 做最小二乘，增益 = 幅度 / GAIN_A。
  - `gain_gate(g: dict[float, np.ndarray]) -> list[str]`：f ≤ 2 时 > 2.0、f ≥ 3 时 > 0.7 的项（NaN 算不过）。
  - `check(it: int) -> dict` → `B3/it{it}/check.json`，命令行 `check --it r`：
    - 误差利用：`data.npz` 的验证样本上，选中网络和 B3-0 各自的第一帧平方误差（对 `Y[:, 0]`，按 `W[:, 0]` 加权，逐关节）→ `use_gate`；
      另报「选中网络 − B3-0」对「−da」的 R²，以及完整版和去掉 IMU 版 3 个种子的验证第一帧损失（均值和最大最小）。
    - 闭环增益：每条验证轨迹取本次迭代的 +δa 回放，`open_stretch` 找不到就跳过并打印；对五个会动关节、`GAIN_F` 各频率，
      `dx` = 该关节加 `GAIN_A·sin(2π f t)`（f = 0 时加常数 `GAIN_A`），`dom = IM.omega_ideal(x + dx) − om`；各轨迹取最大 → `gain_gate`。
      一条能量的轨迹都没有时算不过。
    - 打印两张表和「误差利用检查：通过／不通过（…）」「闭环增益检查：通过／不通过（…）」。
  - `freeze(it: int) -> Path`，命令行 `freeze --it r`：`check.json` 不存在或没通过就 ⛔；否则 `R.freeze_regen(选中的权重, out=B3/it{it})`。
- Produces（`b3_run.py`）：
  - `TRIAL_IDS = ("p23", "p44")`；`TRIAL_RMS, TRIAL_MAX = 5e-3, 15e-3`。
  - `trial_gate(diff: np.ndarray, stopped: bool) -> list[str]`：停机、RMS > 5 mrad、最大 > 15 mrad 的项。
  - `trial(it: int, sess: Path) -> dict`，命令行 `trial --it r --session <目录>`：读 `replay_{tid}_t{it}_r1.json`；
    `diff` = 记录里的 `net_first` − B4 冻结网络在同一遍上的 `R.first_frames`，只取窗 > 0 的帧和关节（j4 用最小的窗）；逐条打印并给出通过／不通过。
  - `report(it: int) -> dict`，命令行 `report --it r`：迭代 1…it 每遍有效 δa 回放的相对理想的关节偏差（`IM.joint_ideal(ep, x, fq)["dev"]`，口径同 `b4_ideal_check.cmd_analyze`），
    按迭代、按训练／验证取中位，打印一张表。

- [ ] **Step 1: 写测试**

```python
def test_use_gate_fails_on_nan_and_on_ties():
    assert B3.use_gate(np.ones(5), np.full(5, 2.0)) == []
    assert len(B3.use_gate(np.array([1, 2, np.nan, 1, 1.0]), np.full(5, 2.0))) == 2


def test_gain_gate_thresholds():
    g = {f: np.full(5, 2.0 if f <= 2 else 0.7) for f in B3.GAIN_F}
    assert B3.gain_gate(g) == []
    g[1.0][4], g[3.0][1] = 2.01, 0.71
    bad = B3.gain_gate(g)
    assert len(bad) == 2 and "j6" in bad[0] and "1 Hz" in bad[0] and "j2" in bad[1] and "3 Hz" in bad[1]
    assert B3.gain_gate({**g, 0.0: np.array([np.nan, 0, 0, 0, 0])})


def test_open_stretch_needs_every_joint_open():                             # Review Focus 5
    w = np.ones((200, 5))
    w[50, 3] = 0
    assert B3.open_stretch(w) == 51 and B3.open_stretch(w[:120]) is None and B3.open_stretch(np.ones((80, 5))) == 0


def test_gain_probe_measures_a_known_gain(case, ideal):
    base = {**case.base, "w": np.ones((case.n, 5))}
    fn = lambda p, f, e: f + 2.5 * e[-1, :60].reshape(10, 6).mean(0)        # noqa: E731  第一帧 = ACT + 2.5 × 最近一帧的编码器误差平均
    x, om = ideal
    t = IM.DTG * np.arange(len(x))
    for f, want in ((0.0, 2.5), (4.0, 2.5 * np.sin(np.pi * 4 * 0.05) / (10 * np.sin(np.pi * 4 * 0.005)))):
        dx = np.zeros_like(x)
        dx[:, 1] = B3.GAIN_A * (np.sin(2 * np.pi * f * t) if f else 1.0)
        g = B3.gain_probe(fn, case.replays[0], base, case.log, CAL(case), ideal, (dx, np.zeros_like(om)), j=1, f=f, k0=25, L=40)
        assert abs(g - want) < 0.03 * want


def test_trial_gate_thresholds():
    ok = np.full(100, 0.004)
    assert BR.trial_gate(ok, stopped=False) == [] and len(BR.trial_gate(ok, stopped=True)) == 1
    assert len(BR.trial_gate(np.r_[ok, 0.016], stopped=False)) == 1 and len(BR.trial_gate(np.full(100, 0.006), stopped=False)) == 1
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**
- [ ] **Step 5: 用户要求时提交**

### Task 8: 评价集 b3val

**Files:**
- Modify: `sim/b4_ideal_check.py`（`SETS`、`ONLINE`、`PAIRS`、`cmd_replay`、`cmd_analyze`、文件头）
- Test: `sim/test_ideal_motion.py`

**Interfaces:**
- Consumes：任务 4 的命令行 `run`；`B3.net_path`。
- Produces：
  - `SETS["b3val"] = (SETS["b4val"][0], ("shift84", "b4", "b3"), "e")`。
  - `ONLINE = {"b3val": {"shift84": ("shadow", 3), "b4": ("regen", 0), "b3": ("regen", 3)}}`：条件 → (在线回放的模式, 用第几次迭代的网络)。
  - `pairs(set_: str) -> tuple[tuple[str, str], ...]`：`b3val` 是 `(("b3", "b4"), ("b3", "shift84"), ("b4", "shift84"))`；其余评价集是每个非基线条件对基线。
  - `cmd_replay`：评价集在 `ONLINE` 里时不写回放序列，每遍用子进程跑 `b4_regen_replay.py run --mode … --net B3.net_path(…)`；提示摆方块、跳过已开始的、返回码非 0 就停，同原来。
  - `cmd_analyze`：按 `pairs(set_)` 逐对比较，结果键为 `"b3 相对 b4"` 这样的名字；`b3val` 里条件 `b3` 有任何一遍 `stopped` 不为空时，
    含 `b3` 的比较判定改为「不通过：B3 触发停机（编号列表）」。

- [ ] **Step 1: 写测试**

```python
def test_b3val_plan_pairs_and_online_modes():
    import b4_ideal_check as V
    assert V.plan(["p1"], V.SETS["b3val"][1]) == [("p1", "shift84", 1), ("p1", "b4", 1), ("p1", "b3", 1),
                                                  ("p1", "b3", 2), ("p1", "b4", 2), ("p1", "shift84", 2)]
    assert V.pairs("b3val") == (("b3", "b4"), ("b3", "shift84"), ("b4", "shift84"))
    assert V.pairs("b4val") == (("regen", "shift84"),) and V.pairs("ideal") == (("shift84", "shift60"), ("teacher24", "shift60"))
    assert V.ONLINE["b3val"] == {"shift84": ("shadow", 3), "b4": ("regen", 0), "b3": ("regen", 3)}
```

- [ ] **Step 2: 用户运行，确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 用户运行，全部通过**；`python -u sim/b4_ideal_check.py analyze --set b4val --session outputs/b4/teacher/sessions/20261112n` 复算，
  regen 相对 shift84 的结果不变（理想 −16.4% [−22.0, −10.1]）。
- [ ] **Step 5: 用户要求时提交**

### Task 9: 上机流程（不写代码；每一步用户在场运行，先给命令和预期结果）

- [ ] **Step 1: 准备**（任务 3、4 已跑过 `init0` 和 `dry`）。
- [ ] **Step 2: 迭代 r = 1、2、3**，每次：
  1. 当天 IMU 标定。r ≥ 2 时先试跑上一次冻结的网络：
     `python -u sim/b4_regen_replay.py run --session outputs/b4/b3/sessions/<名字> --tid p23 --rid p23_t{r−1} --mode regen --net outputs/b4/b3/it{r−1}/frozen.json`，p44 同；
     然后 `python -u sim/b3_run.py trial --it {r−1} --session …`。**不通过就停下讨论。**
  2. `python -u sim/b3_run.py replay --it r --session outputs/b4/b3/sessions/<名字>`（66 遍，可用 `--next N` 分几次跑）；结束后照旧做收尾核对。
  3. `python -u sim/b3_run.py report --it r`：看带 δa 时的关节偏差（只报告）。
  4. `python -u sim/b3.py build --it r` 跑两次，两次的数据哈希相同才继续。预期：训练 26r 条次、验证 7r 条次（成对有效的）。
  5. `python -u sim/b3.py train --it r`（用户终端）。
  6. `python -u sim/b3.py check --it r`。**任何一项不过就停下讨论。** 通过后 `python -u sim/b3.py freeze --it r`，
     再用 `dry` 核对 `outputs/b4/b3/it{r}/frozen.json` 的逐位一致和计算耗时。
- [ ] **Step 3: B3-3 的试跑**（同 Step 2 的第 1 项，`--rid p23_t3`、`p44_t3`，`trial --it 3`）。
- [ ] **Step 4: 评价**：当天标定 → `python -u sim/b4_ideal_check.py replay --set b3val --session outputs/b4/teacher/sessions/<名字>`（60 遍，摆方块）→ 收尾核对 →
  `python -u sim/b4_ideal_check.py analyze --set b3val --session …`。判定按设计 §3.8。
- [ ] **Step 5: 结果记入记忆和设计文档；用户要求时提交。**

## 用户运行的节点

| 在哪个任务之后 | 做什么 |
|---|---|
| 1–8 每个 | 跑测试命令，把输出贴回来 |
| 3 | `python -u sim/b3.py init0` |
| 4 | `dry`（B3-0）：逐位一致、p99 < 10 ms |
| 9 | 按 Step 2–4 上机；每个「不通过就停下讨论」的节点都先回来 |

## 本计划不含（另行决定）

- B3 有效之后的真机 IMU 对照（B3 对去掉 IMU 误差的 B3，40 遍，设计 §3.9）。
- 测试集上的最终结论；结果图。
