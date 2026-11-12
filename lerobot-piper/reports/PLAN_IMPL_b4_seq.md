# 整段版 ① 实现计划（接收 ACT 指令，直接输出未来 10 帧指令）

> **给执行者：** 必须用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐个任务实现。步骤用 `- [ ]` 勾选跟踪。

**Goal：** 训练「接收 ACT 指令、直接输出未来 10 帧指令」的整段版 ①。离线过门槛后，在验证集上与 Shift、一帧版 ① 同会话做真机回放比较。

**Architecture：**
- 标签、训练、选种子、离线检查都加在 `sim/b4_net.py`，输出目录为 `outputs/b4/net_seq/`。
- 网络类只加一个输出维数参数（`b4_online.py`）。
- 回放与分析在 `sim/b4_eval.py` 里加一个评价集 `valseq`：条件 shift / net / seq，回放编号前缀 `q`。
- 实发一律经过现有 `apply_correction`；部署包装 `b4_deploy.py` 不改。

**Tech Stack：** Python（conda 环境 `lerobot`）、numpy、scipy、PyTorch、pytest；真机回放走现有的 `b4_teacher.py run --one`，只调用。

**Spec：** `lerobot-piper/reports/PLAN_b4_seq.md`（v0.2 定稿）。上位文档：`reports/PLAN_b4.md`、`reports/PLAN_IMPL_B4.md`。执行者都要读。

**执行方式（CLAUDE.md）：**
- 每个任务开头按 `reports/ADVISOR_STANDARD.md` 逐条回答第 1–6 条。
- 每个任务做完：
  - 说明改了什么、为什么这样改、关键代码怎么工作；
  - 给用户运行命令和预期结果，**测试由用户运行**；
  - 等用户反馈后再做下一个。
- 只在用户要求时提交：develop 分支，提交信息不加 `Co-Authored-By`，只 `git add` 本任务的文件。
- 不改 `sim/replay_probe.py`；不碰用户未提交的删除。
- 写代码遵守四条准则：防过度防御、入口单次校验、拒绝提前抽象、控制复杂度。

## Global Constraints

- **网络**：
  - 输入 `online_input(a_pi, ens[k], k, P=8, F=16)`，168 维。
  - 结构 MLP 168 → 128 → 128 → 50（N = 10 帧 × 5 个活动关节 `ACTIVE = (0, 1, 2, 4, 5)`）。
  - 输出按帧优先展平：`y.reshape(10, 5)[j]` 是第 k+j 帧。
- **参考与指令**：参考 r_k = `predicted(a_pi, ens[k], k, k, k+N−1)[:, ACTIVE]`；指令 = r_k + y；**不加 Shift**。
- **标签**：
  - 标签 = 教师 `cmd7[HOLD+k+j, ACTIVE] − r_k[j]`；
  - 权重 = 录制的在线窗 `log["w"][k+j]`；
  - 第 k+j 帧超出轨迹末尾时，标签 0、权重 0；
  - N 帧等权。
- **实发只用第一帧**：
  - Δ_eq(k) = `a_pi[k, ACTIVE] + y[0] − shift[k, ACTIVE]`；
  - 实发 = `apply_correction(shift[k], w[k], Δ_eq(k), cfg)`。窗门、±15 mrad、离硬界 0.2° 都不改；j4、夹爪 = Shift。
- **训练**：
  - Adam、批 4096、lr 1e-3、epochs 300、patience 30；按验证集损失（N 帧全部）早停。
  - 种子 0/1/2，按验证集**第一帧**加权损失选种子。
  - 数据 = 训练、备用、验证里验收通过的教师；测试集不进。
- **离线门槛（用户 11-06）**：
  - 每个关节：第一帧 Δ_eq 对教师 Δ* 的 R²_j ≥ 一帧版 R²_j − 0.10。一帧版取 `outputs/b4/net/selection.json` 的 `chosen.val_r2` 原值，不取四舍五入后的值。
  - 每个关节：第一帧限幅触发率（窗 > 0 的帧里 |Δ_eq| > 0.015 rad 的比例）≤ 0.01。
  - 任一项不过就停下，不自行调参。平滑、五关节平均 R²、其余 9 帧、离线平均都只报告。
- **文件隔离**：
  - 整段版只写 `outputs/b4/net_seq/` 和 `outputs/b4/eval/valseq/`；回放编号 `{id}_q{条件}{轮}`。
  - 不写 `outputs/b4/net/`、`outputs/b4/eval/val/`；不复用旧的 `{id}_e…` 编号。
- **真机**：
  - 评价集 `valseq` = 验证集 10 条（含 p27）× 条件 (shift, seq) × 2 轮 = 40 遍（用户 11-06 去掉一帧版；原为三条件 60 遍）。
  - 顺序：第 1 轮 shift → seq，第 2 轮倒序。
  - 分两批，用两个会话目录；每条轨迹的 6 遍在同一批内跑完；每批后收尾核对。
- **判定**：
  - seq 相对 shift 的关节偏差（相对原始策略指令）：两级自助 95% 置信区间上限 < 0，且路径长度、关节速度在 ±5% 以内（现有 `ST.judge`）。
  - 只报告：末端角速度、实发限幅触发率；与一帧版的比较用 11-05 的 val 结果（跨会话粗比）。
- **测试命令**：在 `lerobot-piper/` 下执行
  `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_net.py sim/test_b4_eval.py -q`。

## Review Focus

1. **轨迹末尾**：最后 N−1 帧的目标超出轨迹末尾。期望：标签 0、权重 0，不报错，第一帧照常。测试在任务 1。
2. **旧权重文件**：一帧版的旧权重文件没有 `n_out` 键。期望：仍按 5 维加载，输出不变。测试在任务 2。
3. **旧分析不读新会话**：新会话建好后再跑 `analyze --set val`。期望：仍读 20261105v，数字不变（靠编号前缀隔离）。测试在任务 4。
4. **P、F 与录制时不同**：整段版用 8、16，录制时是 8、8。期望：输入由整段预测复算，而且第 k 帧只用第 k 帧以前的 chunk。测试在任务 4。
5. **一批中途停下**：某条轨迹的 6 遍没跑完，或跨了两个会话。期望：不跨会话拼轮次，该条记「未完成」，不下判定。测试在任务 4。

## 真机与长时间运行节点（用户运行；每次先给命令和预期结果）

| 在哪个任务之后 | 做什么 |
|---|---|
| 1 | `b4_net.py build --P 8 --F 16 --N 10`：看条数、帧数、数据哈希 |
| 2 | `b4_net.py seq`：3 次训练、选种子（不冻结） |
| 3 | `b4_net.py check`：离线门槛；不过就停下商量 |
| 4 | 先核对 `analyze --set val` 结果不变；再两批真机 `replay --set valseq`（每批先标定、跑完收尾核对）；然后 `analyze --set valseq`；判定有效后 `b4_net.py freeze --seq` |
| 5 | 画结果图（中、日） |

---

### Task 1: 整段标签与数据

**Files:**
- Modify: `lerobot-piper/sim/b4_net.py`：加 `NET_SEQ`、`assemble_seq`；`build` 加参数 `N`；命令行加 `build --N`；文件头的用法说明同步更新。
- Test: `lerobot-piper/sim/test_b4_net.py`：沿用 `case` 夹具，其中教师 `cmd7` = `s7` + 含窗的小补正。

**Interfaces:**
- Consumes: `O.te_states`、`O.online_input`、`O.predicted`、`O.ACTIVE`、`RR.HOLD`。
- Produces:
  - `NET_SEQ = C.OUT / "net_seq"`。
  - `assemble_seq(log: dict, seq: dict, teacher: dict, P: int, F: int, N: int) -> dict`，n = 录制帧数，返回：
    - `X` (n, (P+F)·7) float32：与 `assemble` 相同，含与 `log["x"]` 的核对；
    - `R` (n, N, 5)：r_k；
    - `Y`、`Wt` (n, N·5)：见 Global Constraints，按帧优先展平；
    - `D` (n, 5) = `seq["s7"][HOLD:, ACTIVE] − a_pi[:, ACTIVE]`，于是教师补正 Δ* = `Y[:, :5] − D`。
  - `build(P, F, rows=None, loader=None, out=NET, N=None) -> Path`：
    - `N is None` 时，行为和文件名都完全不变；
    - 给了 N 时用 `assemble_seq`，写 `out / f"data_P{P}_F{F}_N{N}.npz"`，内容为 X、Y、Wt、D、traj_id、split、data_sha、P、F、N。
    - 命令行 `build --N` 写到 `NET_SEQ`。

- [ ] **Step 1: 写测试**

```python
def test_seq_first_frame_recovers_teacher_delta(case):
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    sent = case.teacher["cmd7"][HOLD:, ACT].astype(float) - case.seq["s7"][HOLD:, ACT].astype(float)
    assert np.allclose(d["Y"][:, :5] - d["D"], sent, rtol=0, atol=1e-12)    # 只差 float64 舍入
    assert np.allclose(d["X"], case.log["x"], atol=1e-6)

def test_seq_label_column_j_is_frame_k_plus_j(case):
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    n = len(d["X"])
    Y, W = d["Y"].reshape(n, 10, 5), d["Wt"].reshape(n, 10, 5)
    cmd = case.teacher["cmd7"][HOLD:, ACT].astype(float)
    for k, j in ((0, 0), (20, 3), (n - 10, 9)):
        assert np.allclose(Y[k, j] + d["R"][k, j], cmd[k + j], rtol=0, atol=1e-12)
        assert np.array_equal(W[k, j], case.log["w"][k + j])

def test_seq_targets_past_the_end_are_zero(case):                      # Review Focus 1
    d = N.assemble_seq(case.log, case.seq, case.teacher, 8, 8, 10)
    n = len(d["X"])
    Y, W = d["Y"].reshape(n, 10, 5), d["Wt"].reshape(n, 10, 5)
    assert not Y[n - 1, 1:].any() and not W[n - 1, 1:].any() and not W[n - 3, 3:].any()

def test_seq_reference_uses_only_chunks_up_to_k(case):                 # 约束 2；P、F ≠ 录制时，走复算
    chunk = case.log["chunk"].copy()
    chunk[31:] += 0.05
    a, b = (N.assemble_seq({**case.log, "chunk": c}, case.seq, case.teacher, 4, 16, 10)
            for c in (case.log["chunk"], chunk))
    assert np.array_equal(a["R"][:31], b["R"][:31]) and np.array_equal(a["X"][:31], b["X"][:31])
    assert not np.array_equal(a["R"][31:], b["R"][31:])

def test_build_with_N_writes_a_separate_file(case, tmp_path):
    rows = {"a": {"split": "train", "passed": True}, "b": {"split": "val", "passed": True}}
    p = N.build(8, 8, rows=rows, loader=lambda t: (case.log, case.seq, case.teacher), out=tmp_path, N=10)
    z = np.load(p)
    assert p.name == "data_P8_F8_N10.npz" and z["Y"].shape[1] == 50 and z["D"].shape[1] == 5 and int(z["N"]) == 10
```

- [ ] **Step 2: 实现 `assemble_seq`、`build(..., N)` 和命令行 `--N`**
- [ ] **Step 3: 交给用户**
  - 先跑测试命令（Global Constraints）。Expected：原有测试和新增 5 个全部通过。
  - 再运行 `OPENBLAS_NUM_THREADS=1 python -u sim/b4_net.py build --P 8 --F 16 --N 10`。
    Expected：train 30 条、val 9 条，帧数与一帧版 `outputs/b4/net/data_P8_F16.npz` 相同；输出 `→ …/outputs/b4/net_seq/data_P8_F16_N10.npz`，以及数据哈希。
  - 等反馈。

### Task 2: 多帧输出的网络、训练、选种子与冻结

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`：`B4Net` 加 `n_out`；`load_frozen` 读 `n_out`。
- Modify: `lerobot-piper/sim/b4_net.py`：`train` 的输出维数与 `val_loss_first`；`select_seq`；命令行 `seq`、`freeze --seq`。
- Test: `lerobot-piper/sim/test_b4_net.py`

**Interfaces:**
- Consumes: 任务 1 的 `build(..., N)`、`NET_SEQ`。
- Produces:
  - `B4Net(n_in: int, hidden: int = 128, n_out: int = len(ACTIVE))`：末层改为 `Linear(hidden, n_out)`，其余不变。
  - `load_frozen(path)`：
    - 网络维数取权重文件里的 `n_out`；没有这个键（一帧版旧文件）就是 5。
    - 返回值不变：`(fn, sha, P, F)`，`fn` 输出 (n_out,)。
  - `train(...)`：
    - 网络输出维数 = `Y.shape[1]`；权重文件多存 `"n_out"`；
    - 整段帧数统一由 `N = n_out // len(O.ACTIVE)` 还原，不要求权重文件另存 `N`；
    - 返回值多一个 `"val_loss_first"` = 验证集前 5 列（第一帧）的 `weighted_mse`；
    - 早停仍按全部列的验证损失。
  - `select_seq(N: int = 10, P: int = 8, F: int = 16, hidden: int = 128) -> dict`：
    - 先 `build(P, F, N=N, out=NET_SEQ)`；
    - 种子 0/1/2 各跑一次 `train(..., out=NET_SEQ / "runs")`，取 `val_loss_first` 最小的；
    - 写 `NET_SEQ / "selection.json"`：`{"chosen": …, "runs": […]}`，每项含 P、F、N、hidden、seed 和 `train` 的返回值。
  - 命令行：
    - `seq` → `select_seq()`；
    - `freeze --seq` → `freeze(NET_SEQ 里 chosen 的 path, out=NET_SEQ)`。`freeze` 本身不改，仍只能冻结一次。

- [ ] **Step 1: 写测试**

```python
def make_seq_data(path, n=64, P=4, F=4, N_=3, seed=0):
    """同 make_data，但标签是 N_ 帧 × 5（Y、Wt 为 (2n, 5·N_)），另存 D、N。"""

def test_net_output_size_defaults_to_five():
    assert O.B4Net(56, 16)(torch.zeros(3, 56)).shape == (3, 5)
    assert O.B4Net(56, 16, 50)(torch.zeros(3, 56)).shape == (3, 50)

def test_old_checkpoint_without_n_out_loads_as_five(tmp_path):          # Review Focus 2
    r = N.train(make_data(tmp_path / "d.npz"), 0, 16, epochs=2, out=tmp_path)
    m = torch.load(r["path"])
    m.pop("n_out")
    torch.save(m, r["path"])
    fn, *_ = O.load_frozen(str(N.freeze(Path(r["path"]), out=tmp_path)))
    assert fn(np.zeros(56, np.float32)).shape == (5,)

def test_multi_frame_training_saves_n_out_and_first_frame_loss(tmp_path):
    r = N.train(make_seq_data(tmp_path / "d.npz"), 0, 16, epochs=5, out=tmp_path)
    assert torch.load(r["path"])["n_out"] == 15
    assert 0 < r["val_loss_first"] != r["val_loss"]

def test_overfits_tiny_multi_frame_dataset(tmp_path):
    r = N.train(make_seq_data(tmp_path / "d.npz"), 0, 64, epochs=2000, patience=2000, out=tmp_path)
    assert r["train_loss"] < 1e-8

def test_select_seq_picks_lowest_first_frame_loss(monkeypatch, tmp_path):
    monkeypatch.setattr(N, "NET_SEQ", tmp_path)
    monkeypatch.setattr(N, "build", lambda *a, **k: tmp_path / "d.npz")
    fake = {0: (1.0, 0.3), 1: (0.5, 0.4), 2: (0.9, 0.2)}               # 种子：(N 帧验证损失, 第一帧验证损失)
    monkeypatch.setattr(N, "train", lambda d, s, h, **k: {"path": f"s{s}", "train_loss": 0.0, "val_loss": fake[s][0],
                                                          "val_loss_first": fake[s][1], "val_r2": []})
    assert N.select_seq()["chosen"]["seed"] == 2
    assert json.loads((tmp_path / "selection.json").read_text())["chosen"]["seed"] == 2
```

- [ ] **Step 2: 实现上面的接口**
- [ ] **Step 3: 交给用户**
  - 先跑测试命令。Expected：全部通过，包括一帧版原有的冻结与加载测试。
  - 再运行 `OPENBLAS_NUM_THREADS=1 python -u sim/b4_net.py seq`。
    Expected：3 个种子各打印 N 帧验证损失、第一帧验证损失；最后打印 `选中：…/outputs/b4/net_seq/runs/P8_F16_h128_s?.pt`。
  - 先**不要** freeze。等反馈。

### Task 3: 离线检查与上真机门槛

**Files:**
- Modify: `lerobot-piper/sim/b4_net.py`：加 `act_average`、`gate`、`check_seq`；命令行加 `check`。
- Test: `lerobot-piper/sim/test_b4_net.py`

**Interfaces:**
- Consumes:
  - 任务 1 的 `assemble_seq`、`_load_case`；
  - 任务 2 的 `NET_SEQ / "selection.json"`；
  - `assemble` 与 `O.load_frozen(NET / "frozen.json")`：一帧版 ① 的补正，只用于平滑对比；
  - `_r2`；
  - `D.lp`（`sim/step4_gvnext/data.py`，4 阶零相位 Butterworth：`lp(x, fc, fs)`）。
- Produces:
  - `act_average(C: np.ndarray, coeff: float = 0.01) -> np.ndarray`：
    - C (n, N, 5) 的第 k 段 = 第 k 帧生成的、第 k … k+N−1 帧的指令；返回 (n, 5)。
    - 第 k 帧的值 = 能用的 c[k−i, i]（i = 0 … min(N−1, k)）的加权平均。
    - 权重 exp(−coeff·idx)，idx = min(N−1, k) − i：最旧的一段 idx = 0，同 `te_states`。
  - `gate(r2: np.ndarray, clip: np.ndarray, base: np.ndarray) -> tuple[bool, list[str]]`：
    - 通过 ⇔ 每个关节 `r2 >= base − 0.10` 且 `clip <= 0.01`；
    - 第二项用中文列出不过的关节（j1 j2 j3 j5 j6）和数值。
  - `check_seq() -> dict`：只用验证集里验收通过的 9 条教师（`summary.json` 里 split == "val" 且 passed）。
    - **整段版**：用选中的权重算出 y (n, N, 5)，指令 C = R + y。
      - 第一帧 Δ_eq = `y[:, 0] − D`；
      - 离线平均 Δ_avg = `act_average(C) − R[:, 0] − D`；
      - 第 j 帧另报 R²（比较 C[k, j] 与教师 `cmd7[k+j]`，按 `w[k+j]` 加权）。
    - **R²**：用 `_r2`，目标为 Δ* = `Y[:, :5] − D`，权重 = 第一帧的窗 `log["w"]`，与一帧版同口径。分别算第一帧、平均两份。
    - **限幅触发率**：窗 > 0 的帧里 |Δ| > `O.CLIP` 的比例，逐关节；算第一帧、平均两份。
    - **平滑**：≥ 2 Hz 成分 = `x − D.lp(x, 2.0, 20.0)`，每条轨迹单独滤，在窗 > 0 的帧上取 RMS（mrad），逐关节。共四份：整段版第一帧、整段版平均、一帧版 ①（输入用 `assemble(P=8, F=16)` 的 X）、教师 Δ*。
    - **门槛**：基线 = `NET / "selection.json"` 的 `chosen.val_r2`；判定 `gate(第一帧 R², 第一帧触发率, 基线)`。
    - **输出**：写 `NET_SEQ / "check.json"`；打印逐关节表，最后一行「门槛：通过」或「门槛：不通过（…）」。

- [ ] **Step 1: 写测试**

```python
def test_act_average_weights_the_oldest_segment_most():
    C = np.zeros((6, 3, 5))
    C[:, 0], C[:, 1], C[:, 2] = 1.0, 2.0, 3.0                             # 第 k 段对第 k+j 帧给 j+1
    w = np.exp(-0.01 * np.arange(3))                                      # 最旧的一段（i = 2，值 3）权重最大
    out = N.act_average(C)
    assert np.allclose(out[5], (w[0] * 3 + w[1] * 2 + w[2] * 1) / w.sum())
    assert np.allclose(out[1], (w[0] * 2 + w[1] * 1) / w[:2].sum()) and np.allclose(out[0], 1.0)

def test_act_average_of_consistent_segments_is_exact():
    cmd = np.random.default_rng(0).normal(size=(8, 5))
    C = np.stack([cmd[np.minimum(np.arange(k, k + 4), 7)] for k in range(8)])
    assert np.allclose(N.act_average(C), cmd)

def test_gate_is_per_joint_against_the_exact_baseline():
    base = np.array([0.5581268668174744, 0.7319753766059875, 0.6685474216938019,
                     0.20204311609268188, 0.35309410095214844])
    assert N.gate(base - 0.10, np.full(5, 0.01), base) == (True, [])
    r2 = base.copy()
    r2[3] -= 0.1001                                                       # 平均几乎不变，只有 j5 不过
    ok, why = N.gate(r2, np.zeros(5), base)
    assert not ok and len(why) == 1 and "j5" in why[0]
    ok, why = N.gate(base, np.array([0, 0, 0, 0, 0.0101]), base)
    assert not ok and len(why) == 1 and "j6" in why[0]
```

- [ ] **Step 2: 实现 `act_average`、`gate`、`check_seq` 与命令行 `check`**
- [ ] **Step 3: 交给用户**
  - 先跑测试命令。Expected：全部通过。
  - 再运行 `OPENBLAS_NUM_THREADS=1 python -u sim/b4_net.py check`。
    Expected：
    - 逐关节表：第一帧 R²、门槛值、平均 R²、触发率，以及四份 ≥ 2 Hz RMS；
    - 第 1…10 帧的 R²（五关节平均）；
    - 最后一行「门槛：通过」或「门槛：不通过（…）」；
    - 写出 `outputs/b4/net_seq/check.json`。
  - **不通过就停下和用户商量，不自行调参。** 等反馈。

### Task 3b: 整段版 v0.3：相对输入、第一帧加权（设计 v0.3 §4.2、§4.6；用户 11-06 选 A）

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`：`B4Net` 加 `rel_P` 与 `prep`；`load_frozen` 读 `rel_P`。
- Modify: `lerobot-piper/sim/b4_net.py`：`train` 加 `rel_P`、`first`；`select_seq` 用 `rel_P=P, first=True`；`check_seq` 建网络时带 `rel_P`。
- Test: `lerobot-piper/sim/test_b4_net.py`

**Interfaces:**
- `B4Net(n_in, hidden=128, n_out=5, rel_P: int | None = None)`：
  - `prep(x)`：`rel_P is None` 时原样返回；否则把 x 看成 (P+F, 7) 帧，第 rel_P−1 帧是 a_π(k)，返回 `[各帧 − a_π(k)（展平）, a_π(k)]`，维数 n_in + 7。
  - `mu`、`sd` 的维数 = prep 之后的维数；`forward(x) = mlp((prep(x) − mu) / sd)`。
- `load_frozen`：`B4Net(m["n_in"], m["hidden"], m.get("n_out", 5), m.get("rel_P"))`。
- `train(..., rel_P: int | None = None, first: bool = False)`：
  - mu、sd 用 `net.prep(Xt)` 的统计量；权重文件多存 `rel_P`、`first`；
  - `first=True`：训练损失里前 5 列的权重 × (N − 1)（N = 标签列数 / 5，要求 N ≥ 2）；早停按验证集第一帧损失；
  - 返回的 `val_loss` = 存下那一轮的全部列验证损失（`first=False` 时与原来相同）。
- `select_seq` 调 `train(..., rel_P=P, first=True)`；`selection.json` 每项多记 `rel_P`、`first`。
- 先把 v0.2 的产物挪到 `outputs/b4/net_seq/v1/`（用户运行），再跑 `seq`、`check`。

- [ ] **Step 1: 写测试**

```python
def test_relative_input_is_frames_minus_current_plus_current():
    x = torch.arange(56, dtype=torch.float32)[None]
    net = O.B4Net(56, 16, 15, rel_P=4)
    f = x.reshape(1, 8, 7)
    want = torch.cat([(f - f[:, 3:4]).reshape(1, -1), f[:, 3]], 1)
    assert torch.equal(net.prep(x), want) and net.mu.shape == (63,)
    assert torch.equal(O.B4Net(56, 16).prep(x), x)                       # 一帧版：不变换

def test_frozen_relative_net_reproduces_the_trained_net(tmp_path):
    r = N.train(make_seq_data(tmp_path / "d.npz"), 0, 16, epochs=3, out=tmp_path, rel_P=4, first=True)
    m = torch.load(r["path"])
    net = O.B4Net(m["n_in"], m["hidden"], m["n_out"], m["rel_P"])
    net.load_state_dict(m["state_dict"])
    fn, *_ = O.load_frozen(str(N.freeze(Path(r["path"]), out=tmp_path)))
    x = np.load(tmp_path / "d.npz")["X"][0]
    assert m["rel_P"] == 4 and m["first"] is True
    assert np.allclose(fn(x), net(torch.from_numpy(x)[None])[0].detach().numpy())

def test_first_frame_early_stopping_keeps_a_better_first_frame(tmp_path):
    """验证集第一帧的规律与训练相反（越练越差），后两帧一致（越练越好）：按第一帧早停留下的第一帧更好。"""
    d = make_seq_data(tmp_path / "d.npz", n=32)
    z = dict(np.load(d))
    z["Y"][32:, :5] = -z["Y"][32:, :5]
    np.savez(d, **z)
    a = N.train(d, 0, 64, epochs=200, patience=200, out=tmp_path / "a", first=True)
    b = N.train(d, 0, 64, epochs=200, patience=200, out=tmp_path / "b")
    assert a["val_loss_first"] < b["val_loss_first"]
```

- [ ] **Step 2: 实现上面的接口**
- [ ] **Step 3: 交给用户**
  - 测试命令。Expected：全部通过（`38 passed`）。
  - 挪走 v0.2 的产物：
    ```
    S=outputs/b4/net_seq
    mkdir -p $S/v1
    mv $S/runs $S/selection.json $S/check.json $S/v1/
    ```
  - 再依次运行 `python -u sim/b4_net.py seq` 与 `python -u sim/b4_net.py check`。
    Expected：同任务 2、3 的格式；门槛按原规则判定。不过就停下商量。

### Task 3c: 只试一次 N = 5（设计 v0.4；用户 11-06 选 A2）

- Modify: `sim/b4_net.py`：命令行 `seq` 加 `--N`（默认 10），传给 `select_seq(N=…)`；Test: `sim/test_b4_net.py`。

```python
def test_seq_command_passes_N(monkeypatch):
    got = {}
    monkeypatch.setattr(N, "select_seq", lambda **k: got.update(k) or {"chosen": {"path": "x"}})
    assert N.main(["seq", "--N", "5"]) == 0 and got == {"N": 5}
```

- 用户：测试（`39 passed`）→ 把 v0.3 的产物挪到 `outputs/b4/net_seq/v2/` → `seq --N 5` → `check`。
- 门槛规则不变。**不过就停掉整段版 ①，转 ② 的设计**；过了才做任务 4。
- **结果（11-06）：没过（j1、j3、j5、j6），按规则本应停止。** 结论见设计 §4.6、§4.7。
- **用户随后决定以 v0.3（N = 10）上真机（设计 v0.5，偏离）→ 做任务 4。** 先把 v0.3 的产物放回 `outputs/b4/net_seq/`
  （v0.4 的挪到 `v3/`），这样 `valseq` 默认读到的、以后 `freeze --seq` 冻结的都是 v0.3。

### Task 4: 验证集三条件回放与分析（`valseq`）

**Files:**
- Modify: `lerobot-piper/sim/b4_eval.py`：文件头的用法说明同步更新。
- Test: `lerobot-piper/sim/test_b4_eval.py`：
  - 加 `import b4_net as N` 和 `ACT = [0, 1, 2, 4, 5]`；
  - `_fake_session(root, name, tid, rounds, shas, conds=("shift", "net"), prefix="e")` 改成按条件记网络哈希、可指定前缀；
  - 原有调用跟着改，断言不变。

**Interfaces:**
- Consumes:
  - 任务 2：权重文件里的 P、F、`n_out`；`NET_SEQ / "selection.json"`；
  - `O.apply_correction`、`O.load_frozen`、`ST.hier2`、`ST.rel_change`、`ST.judge`。
- Produces:
  - `_inputs(log: dict, P: int, F: int) -> np.ndarray`：
    - P、F 与录制时相同就用 `log["x"]`，否则由整段预测复算；
    - 从 `net_command` 里提出来，`net_command` 改用它，行为不变。
  - `seq_delta(log: dict, net, P: int, F: int, N: int) -> np.ndarray`，形状 (n, 5)：
    Δ_eq(k) = `a_pi[k, ACTIVE] + net(x_k).reshape(N, 5)[0] − shift[k, ACTIVE]`。
  - `seq_command(log: dict, net, P: int, F: int, N: int, cfg) -> np.ndarray`，形状 (HOLD + n, 7)，float32：
    - 逐帧 `apply_correction(shift[k], w[k], seq_delta[k], cfg)`；
    - 开头补 HOLD 帧，保持首帧，同 `net_command`。
  - 评价集与编号：
    - `CONDS["valseq"] = ("shift", "net", "seq")`，`ROUNDS["valseq"] = 2`；
    - `PREFIX = {"val": "e", "test": "e", "valseq": "q"}`；
    - `_rid(tid, cond, r, set_)` = `f"{tid}_{PREFIX[set_]}{cond}{r}"`。
  - `pick_session(tid, sessions, conds, n_rounds, shas: dict[str, str], prefix: str = "e")`：
    - 对有网络的条件，逐个核对 seq 文件里记的 `net_sha == shas[条件]`；
    - 其余规则不变：一条轨迹的全部轮次在同一会话，取最新的会话。
  - `analyze_results(res, conds, n_rounds, judged: str = "net")`：
    - 判定对象 = `judged` 相对 shift；
    - conds 同时含 net 和 seq 时另报两项：
      - `out["net"] = {键: hier2(net 相对 shift)}`，覆盖 `KEYS` 的每个键；
      - `out["seq_vs_net_joint_raw"]`：每轮 seq 相对同轮 net 的变化，再 `hier2`。
    - val、test 原有输出不变。
  - `replay --set valseq --session <dir> [--net 一帧版冻结文件] [--seq-net 整段版权重] [id…]`：
    - 默认 `--net` = `outputs/b4/net/frozen.json`，`--seq-net` = `NET_SEQ / "selection.json"` 的 chosen path；
    - net 条件用 `net_command`，seq 条件用 `seq_command`；P、F 取自权重文件，整段版 N 由 `n_out // len(O.ACTIVE)` 还原；
    - 加载 .pt 权重时建 `B4Net(m["n_in"], m["hidden"], m.get("n_out", 5), m.get("rel_P"))`（任务 3b）；
    - 每个 seq 文件的 `source` 记它自己那个网络的 `net_sha`；
    - id 默认取 split == "val" 的 10 条。
  - `analyze --set valseq`：
    - `shas = {"net": 一帧版, "seq": 整段版}`，`judged="seq"`；写 `outputs/b4/eval/valseq/summary.json`；
    - 另算实发限幅触发率：用 `seq_delta`，窗 > 0 的帧里 |Δ| > `O.CLIP` 的比例，逐关节、10 条合计；打印并写进 summary。

- [ ] **Step 1: 写测试**

```python
@pytest.fixture(scope="module")
def teacher(case):                                                     # 同 test_b4_net 的教师：Shift + 含窗的小补正
    delta = case.seq["w"] * 0.003 * np.sin(np.arange(len(case.seq["w"]))[:, None] * 0.2 + np.arange(5))
    cmd7 = case.seq["s7"].copy()
    cmd7[:, ACT] += delta.astype(np.float32)
    return {"cmd7": cmd7}

def test_seq_command_with_perfect_prediction_reproduces_teacher(case, teacher):
    rows = iter(N.assemble_seq(case.log, case.seq, teacher, 8, 16, 10)["Y"])   # 把标签当网络输出，按帧顺序给
    cmd7 = E.seq_command(case.log, lambda x: next(rows), 8, 16, 10, CFG)
    assert np.abs(cmd7[RR.HOLD:] - teacher["cmd7"][RR.HOLD:]).max() <= 1e-6       # 只差 float32 舍入
    assert np.array_equal(cmd7[RR.HOLD:, [3, 6]], case.seq["s7"][RR.HOLD:, [3, 6]])  # j4、夹爪 = Shift

def test_seq_command_sends_only_the_first_frame(case):
    f = lambda x: np.tile(0.002 * np.tanh(x[-7:-2]), 10)
    g = lambda x: np.r_[0.002 * np.tanh(x[-7:-2]), np.full(45, 0.5)]
    assert np.array_equal(E.seq_command(case.log, f, 8, 16, 10, CFG), E.seq_command(case.log, g, 8, 16, 10, CFG))

def test_seq_command_uses_only_chunks_up_to_k(case):                   # Review Focus 4
    chunk = case.log["chunk"].copy()
    chunk[31:] += 0.05
    f = lambda x: np.tile(0.002 * np.tanh(x[-7:-2]), 10)                # 取最远那一帧未来值，改 chunk 一定看得到
    a, b = (E.seq_command({**case.log, "chunk": c}, f, 8, 16, 10, CFG) for c in (case.log["chunk"], chunk))
    assert np.array_equal(a[:RR.HOLD + 31], b[:RR.HOLD + 31]) and not np.array_equal(a, b)

def test_valseq_plan_reverses_three_conditions():
    assert E.plan("valseq", ["a"]) == [("a", "shift", 1), ("a", "net", 1), ("a", "seq", 1),
                                       ("a", "seq", 2), ("a", "net", 2), ("a", "shift", 2)]

def test_valseq_session_never_feeds_the_old_val_analysis(tmp_path):    # Review Focus 3
    old = _fake_session(tmp_path, "20261105v", "a", (1, 2), {"net": "NET1"})
    new = _fake_session(tmp_path, "20261107q1", "a", (1, 2), {"net": "NET1", "seq": "SEQ"}, ("shift", "net", "seq"), "q")
    assert E.pick_session("a", [old, new], ("shift", "net"), 2, {"net": "NET1"}) == old
    assert E.pick_session("a", [old, new], ("shift", "net", "seq"), 2, {"net": "NET1", "seq": "SEQ"}, "q") == new
    assert E.pick_session("a", [old, new], ("shift", "net", "seq"), 2, {"net": "NET1", "seq": "X"}, "q") is None

def test_valseq_trajectory_split_across_batches_is_not_joined(tmp_path):   # Review Focus 5
    sh = {"net": "N", "seq": "S"}
    s1 = _fake_session(tmp_path, "q1", "a", (1,), sh, ("shift", "net", "seq"), "q")
    s2 = _fake_session(tmp_path, "q2", "a", (2,), sh, ("shift", "net", "seq"), "q")
    assert E.pick_session("a", [s1, s2], ("shift", "net", "seq"), 2, sh, "q") is None

def results3(n_traj=10, net=0.93, seq=0.90):
    """合成：每条 2 轮；一帧版关节偏差是 Shift 的 net 倍，整段版是 seq 倍。"""

def test_analyze_judges_seq_and_reports_net():
    out = E.analyze_results(results3(), ("shift", "net", "seq"), 2, judged="seq")
    assert out["verdict"] == "有效" and out["joint_raw"]["effect"] == pytest.approx(-10, abs=0.5)
    assert out["net"]["joint_raw"]["effect"] == pytest.approx(-7, abs=0.5)
    assert out["seq_vs_net_joint_raw"]["effect"] == pytest.approx((0.90 / 0.93 - 1) * 100, abs=0.5)
```

- [ ] **Step 2: 实现上面的接口**
- [ ] **Step 3: 交给用户（真机前）**
  - 先跑测试命令。Expected：全部通过，包括原有的 val、test 测试。
  - 再核对旧结果没变。在 `lerobot-piper/` 下依次执行：
    ```
    cp outputs/b4/eval/val/summary.json ~/val_before.json
    python -u sim/b4_eval.py analyze --set val
    cmp ~/val_before.json outputs/b4/eval/val/summary.json && echo 一致
    rm ~/val_before.json
    ```
    Expected：打印 −6.5% [−8.2, −4.8]，并输出「一致」。
- [ ] **Step 4: 真机（用户在场；任务 3 门槛通过后才做）**
  - 两批，每批 5 条：前 5 条和后 5 条按 `_ids("val")` 的顺序分，执行时给出具体 id。
  - 每批的做法：
    1. 当天标定，第二批用第一批的收尾标定（同教师采集）；
    2. 新会话目录，`replay --set valseq --session <目录> <5 个 id>`，30 遍约 40 分钟；
    3. 跑完做收尾核对。
  - 两批都跑完后：`analyze --set valseq`。
    Expected：打印判定、整段版相对 Shift、一帧版相对 Shift、整段版相对一帧版、末端角速度、实发限幅触发率。
  - 判定「有效」才执行 `b4_net.py freeze --seq`。
  - 等反馈。

### Task 5: 结果图

**Files:**
- Create: `lerobot-piper/sim/fig_b4_seq.py`：读 `outputs/b4/eval/valseq/summary.json`，写 `outputs/b4/report/fig_seq_val.png` 和 `fig_seq_val_ja.png`。

**Interfaces:**
- Consumes:
  - 任务 4 的 summary：`joint_raw` / `ee_raw` 的 `per_group`（整段版）、`out["net"]` 的 `per_group`（一帧版）、`seq_vs_net_joint_raw`；
  - `sim/fig_b4_report.py` 的样式：字体、`T` 配色、底色。
- Produces: 和 `fig_net1_val` 同版式的两栏图。
  - 左栏关节偏差，右栏末端角速度偏差，每条轨迹一行。
  - 每行两个点：一帧版 ①、整段版 ①。整段版标出效应虚线和 95% 区间带。
  - 图例与总标题同一行；图注一行，内容为「判定只看整段版相对 Shift 的关节偏差；与一帧版的差只报告」。
  - 中、日两版文字放在 `TEXT` 字典里，一次运行出两版。

- [ ] **Step 1: 实现并运行**：`python -u sim/fig_b4_seq.py`，几秒内出两张图。
- [ ] **Step 2: 看图核对**：数字与 summary 一致；标注、图例不压数据点（经验见 `fig_b4_report.py` 的布局修正）。
- [ ] **Step 3: 交给用户**：图的路径与读法。等反馈。
