# B4 实现计划（真机 ILC 当教师 → 能泛化的补正网络）

> **给执行者：** 必须用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐个任务实现。步骤用 `- [ ]` 勾选跟踪。

**Goal：** 实现 B4 五步所需的全部软件：开 TE 的闭环 Shift 与每集记录、位置与分组、教师（固定 12 遍学习 + 独立验收）、网络的数据与训练、回放与闭环的评价和统计。

**Architecture：** 所有在线量（TE 重建、在线 Shift、在线窗、网络输入、加补正）集中在纯函数模块 `lerobot/common/policies/act/b4_online.py`。控制回路里的部署包装和 `sim/` 下的离线脚本调用同一批函数（约束 2）。部署包装按环境变量挂在 `control_robot.py` 上，每集把整段预测、在线窗、网络输入等写成 npz。教师复用 `sim/ilc_route1.py` 的 `Learner` 与回放管线；统计另写，不沿用旧判定（约束 3）。

**Tech Stack：** Python（conda 环境 `lerobot`）、numpy、PyTorch、mujoco（只做正运动学）、pytest；真机回放走 `sim/replay_probe.py`（只调用）。

**Spec：** `lerobot-piper/reports/PLAN_b4.md`（v0.3）。执行者两份都读。

**执行方式（CLAUDE.md）：**
- 每个任务做完，给用户运行命令和预期结果，等用户反馈后再做下一个。
- 每次改动都向用户说明：改了什么、为什么这样改、关键代码怎么工作。
- 只在用户要求时提交：在 develop 分支，提交信息不加 `Co-Authored-By`，只 `git add` 本任务的文件。
- 每个任务开头按 `reports/ADVISOR_STANDARD.md` 逐条回答第 1–6 条。

## Global Constraints

- 开 TE：`temporal_ensemble_coeff=0.01`、`n_action_steps=1`、`cuda_graph=true`、20 Hz。
  模型 = `outputs/train/2026-10-17/ACT_pickcube_noimu_seed42/checkpoints/100000/pretrained_model`。
- 在线 Shift：提前 60 ms，7 维（含夹爪）一起提前，= 0.8 × 第 k+1 帧 + 0.2 × 第 k+2 帧的部分融合值。
- 补正只加在 `ACTIVE = (0, 1, 2, 4, 5)`：|Δ| ≤ 15 mrad，且 Shift＋Δ 离管线硬界（模型限位 ± 0.035 rad）至少 0.2°。
- 在线窗：以下帧为 0，向外 10 帧（0.5 s）余弦过渡；没有片尾过渡。
  - 第 0 帧之前；
  - 疑似接触：夹爪切换前后 0.5 s 再多留 0.1 s（偏保守），或指令末端离目标中心 ≤ 0.03 m；指令与在线 Shift 两份取并集，时间比较同 `cmd_tracking.near`；
  - 限位饱和：逐关节，看 c(j)、s(j)、c(j + 1)。
- **约束 1**：标签 Δ* = 教师实发指令 − 冻结的在线 Shift，已含窗。部署时补正只乘二值门 `w > 0`，**不乘 w**；实发指令与教师的 `cmd7` 用同一个 float32 加法。
- **约束 2**：Shift 基准、网络输入、窗都只用第 k 帧当时已有的 chunk。教师的 `s7` = 录制时实发的 u（逐字节），窗 = 录制时记下的 w。
  评价指标与 ILC 误差里的「疑似接触帧」是测量口径，照 v0.10 在最终指令上算，各条件相同；它们不进补正本身。
- **约束 3**：
  - 回放：两级自助，先抽轨迹，再抽轮次；闭环：两级，先抽位置，再抽配对。
  - 判定只用 `joint_raw` 的 95% 区间上限 < 0，加护栏。
  - 不调用 `ilc_route1.compare` / `verdict` / `hier`。
- **教师**：
  - 学习预算 12 遍 = Shift 基准 2 + 共同关节阶段 4 + 加 IMU 支 6（kind "h"，IMU 独有部分权重 ½）。试探、确认、退回都占这 12 遍。
  - 用完就冻结最后接受的补正；不要求收敛，不追加。
  - 冻结后独立验收 3 轮：Shift 与 Shift＋Δ* 交替，共 6 遍，另计；验收数据不进任何更新。
- **验收通过**：
  - 每轮 Shift＋Δ* 的 `joint_raw` 都小于同轮 Shift；
  - 路径长度、关节速度相对同轮 Shift 都在 ±5% 内；
  - 6 遍都有效，朝向偏差 ≤ 2.5°。
- **护栏**：
  - 路径长度、关节速度：效应量在 ±5% 内。
  - 闭环另加：任务时长增加 ≤ 5%；策略指令自身速度 RMS 下降 ≤ 5%。
  - 闭环成功：严格计数，「成功→失败」不多于「失败→成功」，对数定死 20。
- **成败标注**：用现有的 `scripts/review_eval.py` 逐条看录像标阶段 0–3。
  - 抓起 = 阶段 ≥ 1；成功 = 阶段 3（方块中心在框内）。
  - 没有完全在框内的，备注写「压线」，另计、只报告。
  - 这替代了设计里「每遍结束后当场记」的默认做法，不用新写代码。
- **真实数据的复算容差**：录制在 GPU 上算，离线复算在 CPU 上，float32 舍入不同，所以复算与记录之差 ≤ 1e-5 rad 就算一致；纯 CPU 的单元测试用 1e-6。
- **位置与分组**：
  - 位置在 0…+4 cm 内均匀抽（用户 10-29 第一步检查后定：只放正的一侧；桌上标记 = −4、−2、0、+2、+4 cm），种子 20261029，取到 0.1 cm；打印成「从哪个标记往哪边挪多少 cm」。
  - 主位置 50 个，排序后每 10 个一段，每段 6 : 2 : 2 分到训练 / 验证 / 测试；另备 6 个训练位置；第一步检查另用 10 个位置。
- **IMU**：当天标定；逐遍朝向检查，超过 2.5° 就停；一条教师（学习加验收）在同一会话内跑完。
- `sim/replay_probe.py` 只调用，不修改、不提交。
- **运行环境**：`~/miniconda3/envs/lerobot/bin/python`，`OPENBLAS_NUM_THREADS=1`；`sim/` 入口与测试先 `gc.disable()`，运行加 `-u`。

## Review Focus

1. 新的一集开始时 TE 没清（`reset()` 只清了记录）→ 期望：第一帧的 a_π 等于第一段预测的第 0 行。测试在任务 3。
2. 一集被 Ctrl-C 打断 → 期望：已录的帧照样存成 `_b4.npz` 并标为不完整，原异常照常抛出。测试在任务 3。
3. 窗或网络输入需要的未来帧超出 chunk 末尾 → 期望：用最后一行补齐，不抛错，离线复算结果相同。测试在任务 2。
4. 数据集里记的动作与包装记的 u 不一致（发送时被截断）→ 期望：教师 `seq` 拒绝这一条并说明原因。测试在任务 6。
5. 闭环某一对缺了一边（中途停、成败没标）→ 期望：不补跑，该对不进统计，判定显示「只报告、未完成（x/20）」。测试在任务 11。

## 真机节点（用户运行；每次先给命令和预期结果）

| 在哪个任务之后 | 做什么 |
|---|---|
| 5 | 第一步：`bash scripts/run_b4.sh check`（20 条）→ `review_eval.py` 标成败 → `b4_closed.py check`；没过就停下商量 |
| 5 | 第二步录制：`bash scripts/run_b4.sh record`，再 `b4_positions.py register`，失败的位置按提示重录 |
| 7 | 训练集、验证集的教师会话（每天若干条，每条 18 遍） |
| 9、10 | 选模型 → 验证集回放检查（`b4_eval.py replay --set val`）→ `b4_net.py freeze` |
| 10 | 测试日：每条先跑教师（空桌），再跑评价（放方块），同一会话 |
| 11 | 第五步：`bash scripts/run_b4.sh final`（60 条）→ 标成败 → `b4_closed.py final` |

---

### Task 1: TE 重建与在线 Shift

**Files:**
- Create: `lerobot-piper/lerobot/common/policies/act/b4_online.py`
- Test: `lerobot-piper/tests/test_b4_online.py`

**Interfaces:**
- Produces:
  - 常量 `ACTIVE = (0, 1, 2, 4, 5)`、`DT = 0.05`、`LEAD_S = 0.060`、`CLIP = 0.015`、`HARD_SLACK = 0.035`、`BOUND_MARGIN = np.radians(0.2)`。
  - `te_states(chunks: np.ndarray, coeff: float = 0.01) -> tuple[np.ndarray, np.ndarray]`：
    - 输入 chunks (N, C, 7)，返回 `a_pi` (N, 7) 和 `ens` (N, C−1, 7)。
    - `ens[k, j]` = 第 k 帧更新之后，第 k+1+j 帧的部分融合值。
    - 算法与 `modeling_act.py:339-373` 的 `ACTTemporalEnsembler.update` 相同，用 float32，运算顺序也相同。
  - `online_shift(ens_k: np.ndarray, lead_s: float = LEAD_S, dt: float = DT) -> np.ndarray`：(7,)，在 k + lead_s/dt 处对 ens_k 线性插值（ens_k[0] 是第 k+1 帧）。

- [ ] **Step 1: 写失败的测试**

```python
def test_te_states_matches_torch_ensembler():
    rng = np.random.default_rng(0)
    chunks = rng.normal(size=(40, 100, 7)).astype(np.float32)
    te, acts, ens = ACTTemporalEnsembler(0.01, 100), [], []
    for c in chunks:
        acts.append(te.update(torch.from_numpy(c[None]))[0].numpy())
        ens.append(te.ensembled_actions[0].numpy().copy())
    a_pi, e = O.te_states(chunks, 0.01)
    assert np.allclose(a_pi, acts, atol=1e-6)
    assert all(np.allclose(e[k], ens[k], atol=1e-6) for k in range(40))

def test_online_shift_is_0p8_0p2():
    ens_k = np.arange(21, dtype=float).reshape(3, 7)
    assert np.allclose(O.online_shift(ens_k), 0.8 * ens_k[0] + 0.2 * ens_k[1])

def test_online_shift_equals_offline_shift_when_all_chunks_agree():
    t = np.arange(160) * O.DT
    x = np.stack([np.sin(t * (j + 1)) for j in range(7)], 1).astype(np.float32)
    _, ens = O.te_states(np.stack([x[k:k + 100] for k in range(60)]))   # 每段都预测同一条真实轨迹
    for k in range(59):
        off = [np.interp(t[k] + 0.060, t, x[:, j]) for j in range(7)]  # 同 shift_replay.shifted
        assert np.allclose(O.online_shift(ens[k]), off, atol=1e-6)
```

- [ ] **Step 2: 运行，确认失败**
  Run: `cd ~/claude-test-2/lerobot-piper && ~/miniconda3/envs/lerobot/bin/python -m pytest tests/test_b4_online.py -q`
  Expected: 导入 `b4_online` 失败。
- [ ] **Step 3: 实现 `te_states`、`online_shift`**
- [ ] **Step 4: 运行，确认通过**：同上命令，Expected: `3 passed`。
- [ ] **Step 5: 交给用户**：给出 Step 4 的命令与预期；说明为什么离线要能逐帧复现 TE（约束 2）。等反馈。

### Task 2: 在线窗、网络输入与配置

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`
- Create: `lerobot-piper/sim/b4_common.py`
- Test: `lerobot-piper/tests/test_b4_online.py`（纯函数）、`lerobot-piper/sim/test_b4_common.py`（与旧定义对照）

**Interfaces:**
- Consumes: Task 1。
- Produces（`b4_online.py`）：
  - `OnlineCfg`：`@dataclass(frozen=True)`，字段如下；配套 `load_cfg(path) -> OnlineCfg` 和 `cfg_sha(path) -> str`（文件内容 sha256 的前 16 位）。
    - `center: tuple[float, float, float]`
    - `jnt_lo: tuple[float, ...]`、`jnt_hi: tuple[float, ...]`：6 个模型限位，rad
    - `model_path: str`
    - `contact_s = 0.5`、`contact_pad_s = 0.1`、`zone_m = 0.03`、`grip_open = 0.04`
    - `taper = 10`、`tau_frames = 1`、`P = 8`、`F = 8`
  - `class Tcp`：`__init__(self, model_path: str)`；`__call__(self, q6: np.ndarray) -> np.ndarray`，返回 (n, 3) 的两指中点（link7、link8），同 `sim/cmd_tracking.py:132`。
  - `predicted(a_hist, ens_k, k: int, lo: int, hi: int) -> np.ndarray`：第 k 帧时已知的指令，形状 (hi − lo + 1, 7)。
    - j < 0 取 `a_hist[0]`；
    - 0 ≤ j ≤ k 取 `a_hist[j]`；
    - j > k 取 `ens_k[min(j − k − 1, len(ens_k) − 1)]`。
  - `online_window(a_hist, ens_k, k: int, cfg: OnlineCfg, tcp: Tcp) -> np.ndarray`：返回 (5,)。
    - 在 [k − 22, k + 22] 上取 predicted 指令 c，以及预测 Shift s(j) = 0.8 c(j+1) + 0.2 c(j+2)。
    - 零帧按 Global Constraints「在线窗」的规则。
    - W = 1，若离最近零帧的距离 d ≥ 10；否则 W = 0.5 − 0.5 cos(π d / 10)。
  - `online_input(a_hist, ens_k, k: int, P: int, F: int) -> np.ndarray`：((P + F) · 7,) float32。
    - 前半段是 `a_hist[k−P+1 … k]`，下标 < 0 时用 `a_hist[0]`；
    - 后半段是 `ens_k[0 … F−1]`，不够长就用最后一行补齐。
- Produces（`sim/b4_common.py`）：
  - `OUT = HERE.parent / "outputs/b4"`、`CFG = OUT / "online_cfg.json"`。
  - `make_cfg() -> Path`：
    - `center` 来自 `T.target_center("pickcube_raw", model)`，限位来自 `model.jnt_range[:6]`，模型是 `sim/piper_model.xml`。
    - 文件已存在且内容不同就停，不覆盖。
    - 命令：`python -u sim/b4_common.py cfg`。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_b4_online.py（合成一集：有夹爪合拢、张开，末端经过目标区）
def test_online_window_ignores_information_after_k(ep):             # 约束 2
    a, ens, k = ep.a, ep.ens, 40
    w = O.online_window(a, ens[k], k, ep.cfg, ep.tcp)
    a2 = a.copy(); a2[k + 1:] += 0.3
    assert np.array_equal(O.online_window(a2, ens[k], k, ep.cfg, ep.tcp), w)

def test_online_input_layout_and_causality(ep):
    a, ens, k = ep.a, ep.ens, 40
    x = O.online_input(a, ens[k], k, P=3, F=2)
    assert np.array_equal(x[:21].reshape(3, 7), a[k - 2:k + 1].astype(np.float32))
    assert np.array_equal(x[21:].reshape(2, 7), ens[k][:2].astype(np.float32))
    a2 = a.copy(); a2[k + 1:] += 1
    assert np.array_equal(O.online_input(a2, ens[k], k, 3, 2), x)

def test_short_future_is_padded_not_an_error(ep):                   # Review Focus 3
    w = O.online_window(ep.a, ep.ens[40][:3], 40, ep.cfg, ep.tcp)
    x = O.online_input(ep.a, ep.ens[40][:3], 40, 8, 8)
    assert w.shape == (5,) and np.all((w >= 0) & (w <= 1)) and x.shape == (112,)

# sim/test_b4_common.py
def test_tcp_matches_cmd_tracking(model):
    q = np.random.default_rng(1).uniform(-1, 1, (20, 6))
    assert np.allclose(O.Tcp(str(MODEL_PATH))(q), T.tcp(q, model), atol=1e-12)

def test_online_window_equals_v010_rule_when_predictions_exact(model, cfg):
    a7 = RR.episode_cmd("eval_pickcube_base_TE", 19)[RR.HOLD:].astype(float)
    ens = [a7[k + 1:k + 100] for k in range(len(a7))]               # 预测 = 最终值
    c0 = dataclasses.replace(cfg, contact_pad_s=0.0)
    w_on = np.stack([O.online_window(a7, ens[k], k, c0, O.Tcp(cfg.model_path)) for k in range(len(a7))])
    w_off = v010_window_without_last_frame(a7, model, cfg)           # 测试里用 RR.contact_frames、I.saturated、I.window 按旧规则算，开头补 HOLD 帧再去掉
    assert np.allclose(w_on, w_off, atol=1e-12)

def test_contact_pad_only_adds_zero_frames(model, cfg):              # 偏保守：多留的 0.1 s 只会让窗更小
    a7 = RR.episode_cmd("eval_pickcube_base_TE", 19)[RR.HOLD:].astype(float)
    ens = [a7[k + 1:k + 100] for k in range(len(a7))]
    tcp = O.Tcp(cfg.model_path)
    w0 = np.stack([O.online_window(a7, ens[k], k, dataclasses.replace(cfg, contact_pad_s=0.0), tcp) for k in range(len(a7))])
    w1 = np.stack([O.online_window(a7, ens[k], k, cfg, tcp) for k in range(len(a7))])
    assert np.all(w1 <= w0) and np.any(w1 < w0)
```

- [ ] **Step 2: 运行，确认失败**
  Run: `~/miniconda3/envs/lerobot/bin/python -m pytest tests/test_b4_online.py -q && OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_common.py -q`
  Expected: 新测试因函数不存在而失败。
- [ ] **Step 3: 实现上面列出的函数与 `make_cfg`**：接触判定用与 `T.near` 相同的时间比较（t = DT × 帧号），保证和旧定义逐位一致。
- [ ] **Step 4: 运行，确认通过**：同上，Expected: 两组全部通过（`tests/` 6 个，`sim/` 3 个）。
- [ ] **Step 5: 生成配置并交给用户**：`OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -u sim/b4_common.py cfg`
  Expected: 打印目标中心（m）和 cfg 哈希，写出 `outputs/b4/online_cfg.json`。说明在线窗的规则、它和 v0.10 窗的唯一差别（没有片尾过渡）。等反馈。

### Task 3: 部署包装与每集记录

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`（加 `apply_correction`）
- Create: `lerobot-piper/lerobot/common/policies/act/b4_deploy.py`
- Modify: `lerobot-piper/lerobot/scripts/control_robot.py`（第 303 行 BCORR 段之后，加 B4 环境变量门控）
- Modify: `lerobot-piper/lerobot/common/robot_devices/control_utils.py:238-246`（`record_episode` 在正常结束和异常两处都存包装的记录）
- Test: `lerobot-piper/tests/test_b4_deploy.py`、`lerobot-piper/tests/test_high_rate_log.py`（加接入测试）

**Interfaces:**
- Consumes: Task 1–2。
- Produces:
  - `apply_correction(shift7, w5, d5, cfg: OnlineCfg) -> np.ndarray`：返回 float32 的 u7。
    - u = shift7（转成 float32）；`u[ACTIVE] += ((w5 > 0) * clip(d5, lo, hi)).astype(np.float32)`。
    - lo、hi 同 `ilc_route1.bounds`：±CLIP，且离 `jnt_lo − HARD_SLACK`、`jnt_hi + HARD_SLACK` 至少 BOUND_MARGIN。
    - 实发补正只经过这一个函数：部署和离线复算都调它。**不乘 w。**
  - `class B4Deploy`：`__init__(self, act_policy, mode: str, cfg: OnlineCfg, cfg_sha: str, net=None, net_sha: str = "", P: int | None = None, F: int | None = None)`。
    - mode ∈ {"raw", "shift", "net"}；`net: Callable[[np.ndarray], np.ndarray]`；P、F 默认取 cfg 的值。
    - 方法：`select_action(self, batch) -> torch.Tensor`，返回 (1, 7)；`reset(self)`；`save_episode_log(self, path: Path, error: str | None = None)`。
    - 其他属性透传给 act_policy。
  - `select_action` 的流程：
    - `chunk = act.predict_chunk(batch)` → `a = act.temporal_ensembler.update(chunk)` → `ens_k = act.temporal_ensembler.ensembled_actions[0]`；
    - 再算 shift、w、x；
    - raw 发 a_pi，shift 发 shift，net 发 `apply_correction(shift, w, net(x), cfg)`。
  - 每帧记录：`a_pi` (N,7)、`chunk` (N,C,7) float32、`shift` (N,7)、`w` (N,5)、`x` (N,(P+F)·7) float32、`delta` (N,5)、`u` (N,7)。
    npz 另存 `mode`、`cfg_sha`、`net_sha`、`P`、`F`、`complete`、`error`。
  - `build_b4_policy(act_policy, mode: str, cfg_path: str) -> B4Deploy`：本任务只支持 raw 和 shift；net 在任务 9 加。
  - 环境变量：
    - `B4_MODE`：raw | shift | net；
    - `B4_CFG`：默认 `outputs/b4/online_cfg.json`；
    - `B4_NET`：net 模式必须给。
    - 与 `REFINE_CKPT`、`BCORR_CKPT` 互斥；`policy.config.temporal_ensemble_coeff` 为空时报错。
  - `record_episode`：
    - `policy` 有 `save_episode_log` 时，存到 `hr_path.with_name(hr_path.stem + "_b4.npz")`；
    - 异常时传 `error`；保存失败不掩盖原异常（同 HighRateLog 的做法）。

- [ ] **Step 1: 写失败的测试**

```python
# tests/test_b4_deploy.py：小 ACT 用 tests/test_act_cuda_graph.py 的 tiny_config 建在 CPU 上；ChunkReplayer = predict_chunk 按顺序返回给定 chunk 的假策略（带真的 ACTTemporalEnsembler）
def test_raw_mode_equals_act_select_action():
    a, b = tiny_act(seed=0), tiny_act(seed=0)
    dep = B4Deploy(a, "raw", CFG, "x")
    for batch in batches(10):
        assert torch.equal(dep.select_action(batch), b.select_action(batch))

def test_log_reproduces_online_quantities_offline():               # 约束 2
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x"), 60)
    a_pi, ens = O.te_states(log["chunk"])
    assert np.allclose(a_pi, log["a_pi"], atol=1e-6)
    for k in range(60):
        assert np.allclose(O.online_shift(ens[k]), log["shift"][k], atol=1e-6)
        assert np.array_equal(O.online_window(log["a_pi"], ens[k], k, CFG, TCP), log["w"][k])
        assert np.allclose(O.online_input(log["a_pi"], ens[k], k, 8, 8), log["x"][k], atol=1e-6)
    assert np.array_equal(log["u"], log["shift"].astype(np.float32))

def test_net_mode_does_not_multiply_by_window():                   # 约束 1
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: np.full(5, 0.01)), 60)
    taper = (log["w"] > 0) & (log["w"] < 1)
    assert taper.any()
    assert np.allclose(log["delta"][taper], 0.01)
    assert not log["delta"][log["w"] == 0].any()

def test_net_mode_oracle_reproduces_teacher_cmd7_bitwise():         # 约束 1
    base = run(B4Deploy(ChunkReplayer(CHUNKS), "shift", CFG, "x"), 60)
    d_star = teacher_like(base["w"])                                 # w == 0 处为 0，且在上下限之内
    cmd7 = base["u"].copy(); cmd7[:, list(O.ACTIVE)] += d_star.astype(np.float32)   # 同 ilc_route1.write_seq
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=Oracle(d_star)), 60)
    assert np.array_equal(log["u"], cmd7)

def test_reset_clears_ensembler_and_log():                          # Review Focus 1
    dep = B4Deploy(ChunkReplayer(CHUNKS + CHUNKS), "shift", CFG, "x")
    run(dep, 60); dep.reset(); log = run(dep, 5)
    assert len(log["a_pi"]) == 5 and np.allclose(log["a_pi"][0], CHUNKS[0][0], atol=1e-7)

# tests/test_high_rate_log.py（沿用文件里的 FakeRobot 与 record_episode 夹具）
def test_b4_log_saved_next_to_high_rate_and_on_interrupt(tmp_path, monkeypatch, cu):   # Review Focus 2
    # 正常：调用一次，路径 = high_rate/episode_000007_b4.npz，error=None
    # 中断（假机器人抛 KeyboardInterrupt）：调用一次且 error 非空，KeyboardInterrupt 照常抛出
```

- [ ] **Step 2: 运行，确认失败**
  Run: `~/miniconda3/envs/lerobot/bin/python -m pytest tests/test_b4_deploy.py tests/test_high_rate_log.py -q`
  Expected: 新测试失败，原有测试照常通过。
- [ ] **Step 3: 实现 `apply_correction`、`B4Deploy`、`build_b4_policy`、两处接入**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: 全部通过。
- [ ] **Step 5: 交给用户**：说明包装在控制回路里做了什么、为什么不乘渐变窗（约束 1）、每集记录了什么。等反馈。

### Task 4: 统计与判定（约束 3）

**Files:**
- Create: `lerobot-piper/sim/b4_stats.py`
- Test: `lerobot-piper/sim/test_b4_stats.py`

**Interfaces:**
- Produces:
  - `rel_change(x: float, y: float) -> float`：(x − y) / y × 100。
  - `hier2(rel: dict[str, list[float]], n_boot: int = 10000, seed: int = 0) -> dict`：
    - 效应 = 各组中位数的平均。
    - 95% 区间：每次自助先有放回地抽组，再在每个抽到的组里有放回地抽单位，取中位数后对组平均。
    - 返回 `{"effect", "ci95": [lo, hi], "per_group": {g: {"median", "n"}}, "n_groups"}`。
  - `judge(r: dict) -> str`：r 里有 `"joint_raw"`、`"path"`、`"speed"`（都是 hier2 的结果），其他键只报告。
    - 以下三条都满足时返回 `"有效"`：|path.effect| ≤ 5，|speed.effect| ≤ 5，joint_raw.ci95[1] < 0。
    - 否则返回写明原因的字符串。
  - `judge_closed(r: dict, pairs: list[tuple[bool, bool]], n_required: int = 20) -> str`：
    - 在 `judge` 之外再要求：`duration.effect ≤ 5`、`cmd_speed.effect ≥ −5`、严格计数通过。
    - `len(pairs) != n_required` 时返回 `"只报告、未完成（完整对数 x/20）"`。
  - `strict_count(pairs) -> tuple[int, int, bool]`：pair = (Shift 成功, Shift＋网络成功)；返回 (成功→失败数, 失败→成功数, 前者 ≤ 后者)。
  - `chance_fail_prob(p: float, n: int) -> float`：两组真实失败率都为 p 时，n 对里「成功→失败」严格多于「失败→成功」的精确概率。

- [ ] **Step 1: 写失败的测试**

```python
GOOD = {"effect": -3.0, "ci95": [-4.0, -2.0]}
FLAT = {"effect": 0.5, "ci95": [-1.0, 2.0]}

def test_hier2_effect_is_mean_of_group_medians():
    r = S.hier2({"a": [-10, -12, -11, -9], "b": [-2, -4, -3, -1]})
    assert r["effect"] == pytest.approx((-10.5 + -2.5) / 2)

def test_hier2_resamples_groups_not_only_rounds():
    r = S.hier2({"a": [-20.0, -20.1, -19.9, -20.0], "b": [10.0, 10.1, 9.9, 10.0]})
    assert r["ci95"][0] < 0 < r["ci95"][1]        # 只抽轮次时区间约 [-5.05, -4.95]，不跨 0

def test_judge_uses_joint_only():
    assert S.judge({"joint_raw": GOOD, "path": FLAT, "speed": FLAT, "ee_raw": {"effect": 40.0, "ci95": [20, 60]}}) == "有效"
    assert S.judge({"joint_raw": {"effect": -1.0, "ci95": [-3.0, 0.5]}, "path": FLAT, "speed": FLAT}) != "有效"
    assert "护栏" in S.judge({"joint_raw": GOOD, "path": {"effect": 6.0, "ci95": [5, 7]}, "speed": FLAT})

def test_strict_count():
    assert S.strict_count([(True, True)] * 19 + [(True, False)]) == (1, 0, False)
    assert S.strict_count([(True, False), (False, True)] + [(True, True)] * 18) == (1, 1, True)

def test_chance_fail_prob_values():
    assert S.chance_fail_prob(0.0, 20) == 0.0
    assert S.chance_fail_prob(0.02, 20) == pytest.approx(0.2386, abs=1e-3)
    assert S.chance_fail_prob(0.05, 20) == pytest.approx(0.3449, abs=1e-3)
    assert S.chance_fail_prob(0.05, 200) == pytest.approx(0.454, abs=1e-3)

def test_judge_closed_needs_20_pairs_and_extra_guards():
    r = {"joint_raw": GOOD, "path": FLAT, "speed": FLAT, "duration": FLAT, "cmd_speed": FLAT}
    assert S.judge_closed(r, [(True, True)] * 20) == "有效"
    assert "未完成" in S.judge_closed(r, [(True, True)] * 19)
    assert S.judge_closed({**r, "duration": {"effect": 6.0, "ci95": [5, 7]}}, [(True, True)] * 20) != "有效"
    assert S.judge_closed(r, [(True, False)] + [(True, True)] * 19) != "有效"
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_stats.py -q`
  Expected: 导入失败。
- [ ] **Step 3: 实现**：`chance_fail_prob` 用三项分布精确求和（每对「成功→失败」「失败→成功」的概率都是 p(1−p)）。
- [ ] **Step 4: 运行，确认通过**：Expected: `6 passed`。
- [ ] **Step 5: 交给用户**：说明两级自助和旧 `hier()` 的区别，用 `test_hier2_resamples_groups_not_only_rounds` 举例。等反馈。

### Task 5: 位置、录制脚本与第一步检查

**Files:**
- Create: `lerobot-piper/sim/b4_positions.py`
- Create: `lerobot-piper/scripts/run_b4.sh`
- Modify: `lerobot-piper/sim/b4_common.py`（读闭环一集、算指标）
- Create: `lerobot-piper/sim/b4_closed.py`（本任务只做 `check`；`final` 在任务 11）
- Test: `lerobot-piper/sim/test_b4_positions.py`、`lerobot-piper/sim/test_b4_closed.py`

**Interfaces:**
- Consumes: Task 3（`B4_MODE`）、Task 4（`hier2`、`strict_count`、`rel_change`）、`scripts/review_eval.py` 的 CSV（阶段 0–3）。
- Produces:
  - `b4_positions.generate(seed: int = 20261029) -> dict`：返回 `{"check": [10 个 cm], "pool": [{"id": "p00", "cm", "split"}...] (56 个), "order": [id...]}`。
    - 分组规则见 Global Constraints；备用位置的 split = `"spare"`。
    - order 是全部 56 个 id 打乱后的录制顺序。
    - 写 `outputs/b4/positions.json`；已存在且内容不同就停。
  - `b4_positions.py block <mode> <n>`：打印第 n 块的 5 个位置（cm），供 `run_b4.sh` 用。
    - check 模式：ABBA 四块分别用 check[0:5]、[0:5]、[5:10]、[5:10]。
  - `b4_positions.assign(order: list[str], stages: list[int]) -> dict[str, int]`：把录制的集号对应到位置 id。
    - 队列 = order，失败（阶段 < 3）的 id 依次追加到队尾；第 i 集对应队列第 i 个；返回成功的 id → 集号。
  - `b4_positions.py register <repo>`：读 review CSV，调 `assign`，写 `outputs/b4/recordings.json`（id → repo、集号、阶段），并打印还要重录的 id。
  - `b4_common.make_ctx() -> SimpleNamespace`：`model`、`sid`（`ee_imu_site`）、`calib`（当前标定），供算指标用。
  - `b4_common.load_closed(repo: str, i: int) -> dict`：
    - 读 high_rate JSON、`_b4.npz` 和数据集动作，用 `T.from_record(h, log["u"])` 建一集。
    - 另加 `a_pi`、`w`、`mode`；记录 `act_mismatch` = 数据集动作与 u 的最大差；帧数不一致或记录不完整就报错。
  - `b4_common.closed_metrics(ep: dict, ctx) -> dict`：
    - 参考 = `ep["a_pi"]`；`marked = RR.contact_frames(a_pi)`；`S.masks(ep, marked, MARGIN)`。
    - 返回 `S.metrics` 的 `joint_raw`、`path`、`speed`，另加两项：
      - `duration`：第一次合拢之后第一次张开的时刻 − `t_send[0]`（s），没张开记 NaN；
      - `cmd_speed`：a_π 五个活动关节的速度 RMS（rad/s）。
  - `b4_closed.py check`：
    - 读 `eval_pickcube_b4check_raw`、`eval_pickcube_b4check_shift` 和两份 review CSV，按位置配对（10 对）。
    - 通过 = 严格计数通过，且 `hier2`（组 = 位置）的 joint_raw 区间上限 < 0。
    - 报告效应，并与回放 Shift 的 −67% 对照；写 `outputs/b4/check/summary.json`；没过就打印「停下，和用户商量」。
    - 另报：shift 组记录的 u 与离线 `S.shifted(a_pi, 0.060)` 之差（最大、RMS），作为闭环 Shift 实现正确的证据。
  - `scripts/run_b4.sh check | record [起始块] | final [起始块]`：
    - 照 `scripts/run_baseline.sh`：每块 5 条一个进程；开录前打印这 5 条的位置，`read -rp` 等用户摆好。
    - check：A = `B4_MODE=raw`，B = `B4_MODE=shift`，顺序 ABBA；数据集 `eval_pickcube_b4check_{raw,shift}`。
    - record：`B4_MODE=shift`，按 order 每 5 个一块，数据集 `eval_pickcube_b4rec`；位置用完后，按 `register` 打印的重录 id 接着录。

- [ ] **Step 1: 写失败的测试**

```python
def test_positions_deterministic_and_in_range():
    a = P.generate(); assert a == P.generate()
    assert all(-6 <= c <= 6 for c in [p["cm"] for p in a["pool"]] + a["check"])

def test_split_is_6_2_2_per_bin():
    main = sorted((p for p in P.generate()["pool"] if p["split"] != "spare"), key=lambda p: p["cm"])
    for i in range(0, 50, 10):
        assert Counter(p["split"] for p in main[i:i + 10]) == {"train": 6, "val": 2, "test": 2}

def test_register_requeues_failures(tmp_path):
    order = ["p00", "p01", "p02"]; stages = [3, 1, 3, 3]          # 第 1 集失败 → p01 追加到队尾
    assert P.assign(order, stages) == {"p00": 0, "p02": 2, "p01": 3}

def test_closed_metrics_reference_is_policy_command():
    ep = synthetic_closed(offset_rad=0.002)                        # 实测完全跟着 u = a_pi + 0.002
    assert C.closed_metrics(ep, CTX)["joint_raw"] == pytest.approx(2.0, abs=0.05)   # mrad：相对 a_pi，不是相对 u
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_positions.py sim/test_b4_closed.py -q`
- [ ] **Step 3: 实现上面列出的函数、命令和脚本**（`P.assign` 是 `register` 的纯函数部分）
- [ ] **Step 4: 运行，确认通过**；再跑 `DRY=1 bash scripts/run_b4.sh check`，Expected: 打印 4 条 control_robot 命令和每块的 5 个位置，不连机械臂。
- [ ] **Step 5: 交给用户**：给出第一步的上机命令与预期（20 条、约 25 分钟），以及跑完后的 `review_eval.py` 与 `b4_closed.py check` 命令。等反馈。

### Task 6: 教师序列（约束 2）

**Files:**
- Create: `lerobot-piper/sim/b4_teacher.py`（本任务：`build_seq` 与 `seq` 子命令）
- Test: `lerobot-piper/sim/test_b4_teacher.py`

**Interfaces:**
- Consumes: Task 2（`load_cfg`、`cfg_sha`、`te_states`、`online_shift`）、Task 5（`recordings.json`）、`ilc_route1.bounds`、`RR.HOLD`、`RR.contact_frames`。
- Produces:
  - `build_seq(log: dict, act: np.ndarray, cfg: OnlineCfg, model) -> dict`：返回 `{"a7", "s7", "w", "lo", "hi", "marked"}`。
    - `a7 = [a_pi[0] × HOLD, a_pi]`。
    - `s7 = [u[0] × HOLD, u]`：float32，u 逐字节取自记录。
    - `w = [0 × HOLD, log w]`。
    - `lo, hi = I.bounds(s7[:, :6], jnt_lo − 0.035, jnt_hi + 0.035)`。
    - `marked = RR.contact_frames(a7) | RR.contact_frames(s7)`：只用于评价口径和 ILC 误差，同 v0.10。
  - `build_seq` 在以下任一情况下停下（SystemExit），并写明原因：
    - mode 不是 shift，或记录不完整；
    - 数据集动作与 u 的最大差 > 1e-6（两者都是同一份 float32，应逐位相同）；
    - 用 `te_states(chunk)` 复算的 a_pi，或 `online_shift` 复算的 u，与记录的差 > 1e-5；
    - Δ = 0 也不满足上下限。
  - 另报 `n_unsafe`：记录的 w > 0、但 `marked` 为真的帧数（在线窗没盖住事后判定的接触帧）；只报告，写进来源信息。
  - `b4_teacher.py seq <id…>`：写 `outputs/b4/teacher/<id>/seq_base.npz`，并记来源哈希；已存在且内容不同就停。

- [ ] **Step 1: 写失败的测试**

```python
def test_s7_is_logged_online_shift_bitwise(shift_log):              # 约束 2
    q = B.build_seq(shift_log, shift_log["u"], CFG, MODEL)
    assert np.array_equal(q["s7"][RR.HOLD:], shift_log["u"].astype(np.float32))
    assert np.array_equal(q["w"][RR.HOLD:], shift_log["w"]) and not q["w"][:RR.HOLD].any()

def test_rejects_dataset_action_mismatch(shift_log):                 # Review Focus 4
    act = shift_log["u"].copy(); act[50, 1] += 1e-3
    with pytest.raises(SystemExit, match="数据集动作"):
        B.build_seq(shift_log, act, CFG, MODEL)

def test_rejects_raw_or_incomplete_log(shift_log):
    for bad in ({**shift_log, "mode": "raw"}, {**shift_log, "complete": False}):
        with pytest.raises(SystemExit):
            B.build_seq(bad, bad["u"], CFG, MODEL)
```
`shift_log` 由任务 3 的 `B4Deploy(ChunkReplayer(...), "shift", ...)` 跑出。

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_teacher.py -q`
  Expected: 导入 `build_seq` 失败。
- [ ] **Step 3: 实现 `build_seq` 与 `seq` 子命令**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: `3 passed`。
- [ ] **Step 5: 交给用户**：录完第二步后跑 `python -u sim/b4_teacher.py seq $(列出 id)`，Expected: 每条打印帧数和哈希，没有拒绝；有拒绝就说明原因。等反馈。

### Task 7: 教师运行与独立验收

**Files:**
- Modify: `lerobot-piper/sim/b4_teacher.py`（`run`、`status` 子命令）
- Test: `lerobot-piper/sim/test_b4_teacher.py`

**Interfaces:**
- Consumes:
  - Task 6 的 `seq_base.npz`；
  - `ilc_route1`：`Learner`、`mean_res`、`joint_err`、`ang_err`、`enc_gyro`、`orientation_deg`；
  - `shift_replay`：`metrics`、`masks`、`raw_gyro`、`guarded_run`；
  - `repeat_reversal`：`run_child`、`attempted`；
  - `cmd_tracking.from_record`。
- Produces:
  - `SCHEDULE = [("b", 1), ("b", 2), ("c", 1), …, ("c", 4), ("h", 1), …, ("h", 6)]`：12 遍学习。
  - `ACCEPT = [("as", 1), ("at", 1), ("at", 2), ("as", 2), ("as", 3), ("at", 3)]`：as = Shift，at = Shift＋Δ*；3 轮，先后交替。
  - `walk_teacher(tid: str, sess: Path, execute, result) -> dict`：
    - 同 `ilc_route1.walk` 的确定性重建：已开始过的遍按顺序喂给学习器。
    - 学习器：`Learner("c", 4, 0, …)` 从两遍基准出发；`Learner("h", 6, c.acc, c.err, c.met, box, c.hist[-1][0], enc=False, wr=0.5)`；box = (seq w, lo, hi)。
    - 第 12 遍之后冻结 `delta = st["h"].acc`，写 `teacher_<tid>.npz`：delta、`cmd7 = s7 + delta`（float32，同 `write_seq`）、哈希。
    - 再跑 ACCEPT；验收结果只进 `st["accept"]`，不调用任何 Learner 方法。
    - `result(rid) -> (met, err, qc) | None` 由调用方注入，测试用合成结果。
    - 返回状态 dict：学习过程、冻结的 delta、验收结果、结论。
  - `accept_verdict(acc: dict, qc_max: float = 2.5) -> tuple[bool, list[str]]`：按 Global Constraints「验收通过」判；返回 (是否通过, 原因列表)。
  - 会话：`outputs/b4/teacher/sessions/<YYYYMMDD>/session.json`，记标定哈希和 `rule: "b4-v1"`。
    - `check_session(sess: Path, calib: Path) -> None`：标定不是当天做的，或哈希和会话记录不同，就停（SystemExit）。
    - 同一条教师的 18 遍必须在同一会话。
  - `run --session <dir> <tid…>`：一条跑完 18 遍再跑下一条；每遍提示「桌面已清空、可以运动就按回车」；回放用 `RR.run_child` + `S.guarded_run`。
  - `status`：列出每条的进度、冻结与验收结论，写 `outputs/b4/teacher/summary.json`。

- [ ] **Step 1: 写失败的测试**

```python
def test_schedule_is_12_learning_plus_6_acceptance():
    assert len(B.SCHEDULE) == 12 and [k for k, _ in B.SCHEDULE].count("h") == 6
    assert B.ACCEPT == [("as", 1), ("at", 1), ("at", 2), ("as", 2), ("as", 3), ("at", 3)]

def test_probes_and_confirmations_use_the_12(fake):                  # 预算记账
    st = B.walk_teacher("t", fake.sess, fake.execute, fake.result_with_guard_trips)
    assert len(st["learning"]) == 12 and st["h"].left == 0

def test_frozen_delta_is_last_accepted(fake):
    st = B.walk_teacher("t", fake.sess, fake.execute, fake.result_last_two_rejected)
    assert np.array_equal(st["delta"], fake.last_accepted_delta)

def test_acceptance_never_reaches_learners(fake, monkeypatch):
    calls, orig = [], I.Learner.add
    monkeypatch.setattr(I.Learner, "add", lambda self, *a, **k: calls.append(self.kind) or orig(self, *a, **k))
    st = B.walk_teacher("t", fake.sess, fake.execute, fake.result)
    assert len(st["accept"]) == 6
    assert Counter(calls) == {"c": 4, "h": 6}                        # 验收的 6 遍一遍也没进学习器

def test_accept_verdict_cases():
    assert B.accept_verdict(acc(all_better=True)) == (True, [])
    assert not B.accept_verdict(acc(round2_worse=True))[0]
    assert not B.accept_verdict(acc(path_plus=0.06))[0]
    assert not B.accept_verdict(acc(invalid=("at", 3)))[0]
    assert not B.accept_verdict(acc(qc=3.0))[0]

def test_session_rejects_other_calibration(tmp_path):
    calib = tmp_path / "imu_calib.json"; calib.write_text("{}")
    sess = tmp_path / "s"; sess.mkdir(); (sess / "session.json").write_text(json.dumps({"calib_sha": "0" * 16}))
    with pytest.raises(SystemExit):
        B.check_session(sess, calib)
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_teacher.py -q`
  Expected: 新测试失败，任务 6 的 3 个照常通过。
- [ ] **Step 3: 实现**：调度、重建、冻结、验收、会话检查和两个子命令。
- [ ] **Step 4: 运行，确认通过**：`sim/test_b4_teacher.py` 全部通过；`sim/test_ilc_route1.py` 仍是 39 passed（`ilc_route1.py` 没改）。
- [ ] **Step 5: dry-run 并交给用户**：`python -u sim/b4_teacher.py run --session <今天> --dry-run <一条 id>`
  Expected: 逐遍打印 18 个 rid，`replay_probe` 的安全检查全部通过，不连机械臂。说明每天的上机步骤：标定 → run → status。等反馈。

### Task 8: 网络的数据（约束 1、2）

**Files:**
- Create: `lerobot-piper/sim/b4_net.py`（本任务：`assemble` 与 `build` 子命令）
- Test: `lerobot-piper/sim/test_b4_net.py`

**Interfaces:**
- Consumes: Task 1–2（`te_states`、`online_input`）、Task 6–7（`seq_base.npz`、`teacher_<tid>.npz`、`summary.json` 的验收结论）、`positions.json`、`recordings.json`。
- Produces:
  - `assemble(log: dict, seq: dict, teacher: dict, P: int, F: int) -> dict`，按录制帧下标：
    - `X` (N, (P+F)·7) float32：用 `te_states(log["chunk"])` 与 `online_input` 复算；P、F 等于记录时的值时，必须与 `log["x"]` 相差 ≤ 1e-5，否则停。
    - `Y` (N, 5) = `teacher["cmd7"][HOLD:, ACTIVE] − seq["s7"][HOLD:, ACTIVE]`，即教师实发的补正。
    - `Wt` (N, 5) = `log["w"]`，作损失权重。
    - 断言：`Wt == 0` 处 `Y == 0`。
  - `build(P: int, F: int) -> Path`：
    - 只收训练集、验证集里验收通过的教师；测试集一律不进。
    - 写 `outputs/b4/net/data_P{P}_F{F}.npz`（X、Y、Wt、traj_id、split），并记数据哈希。

- [ ] **Step 1: 写失败的测试**

```python
def test_label_is_teacher_sent_delta(case):                           # 约束 1
    d = N.assemble(case.log, case.seq, case.teacher, 8, 8)
    sent = case.teacher["cmd7"][HOLD:, ACT].astype(float) - case.seq["s7"][HOLD:, ACT].astype(float)
    assert np.array_equal(d["Y"], sent)
    assert not d["Y"][d["Wt"] == 0].any()

def test_inputs_equal_logged_online_inputs(case):                     # 约束 2
    assert np.allclose(N.assemble(case.log, case.seq, case.teacher, 8, 8)["X"], case.log["x"], atol=1e-6)

def test_build_excludes_test_split_and_failed_teachers(tmp_layout):
    z = np.load(N.build(8, 8))
    assert set(z["split"]) <= {"train", "val"}
    assert tmp_layout.failed_id not in set(z["traj_id"])
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_net.py -q`
  Expected: 导入 `b4_net` 失败。
- [ ] **Step 3: 实现 `assemble` 与 `build`**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: `3 passed`。
- [ ] **Step 5: 交给用户**：教师跑完后执行 `python -u sim/b4_net.py build --P 8 --F 8`。
  Expected: 打印训练、验证各多少条、多少帧，以及数据哈希。等反馈。

### Task 9: 网络模型、训练、选择、冻结与部署加载

**Files:**
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_online.py`（加 `B4Net`、`load_frozen`）
- Modify: `lerobot-piper/sim/b4_net.py`（`train`、`select`、`curve`、`freeze` 子命令）
- Modify: `lerobot-piper/lerobot/common/policies/act/b4_deploy.py`（`build_b4_policy` 支持 net）
- Test: `lerobot-piper/sim/test_b4_net.py`、`lerobot-piper/tests/test_b4_deploy.py`

**Interfaces:**
- Consumes: Task 8 的数据文件；Task 3 的 `B4Deploy`。
- Produces:
  - `class B4Net(nn.Module)`：`__init__(self, n_in: int, hidden: int = 128)`。
    - 输入标准化 buffer `mu`、`sd`，用训练集的统计量。
    - 结构：Linear(n_in, hidden) → ReLU → Linear(hidden, hidden) → ReLU → Linear(hidden, 5)。
    - `forward(x) -> (…, 5)`，单位 rad。
  - `weighted_mse(pred, y, w) -> Tensor`：Σ w (pred − y)² / Σ w。
  - `load_frozen(path: str) -> tuple[Callable[[np.ndarray], np.ndarray], str, int, int]`：读冻结 JSON，核对权重 sha，返回 CPU 上的推理函数、权重 sha、P、F。
  - `train(data: Path, seed: int, hidden: int, epochs: int = 300, lr: float = 1e-3, patience: int = 30) -> dict`：
    - Adam，批大小 4096，按验证集损失早停；`torch.manual_seed(seed)`。
    - 保存 `outputs/b4/net/runs/P{P}_F{F}_h{hidden}_s{seed}.pt`，记录验证集逐关节 R²（按 Wt 加权）。
    - 返回 `{"path", "train_loss", "val_loss", "val_r2"}`。
  - `select()`：
    - 候选 (P, F) ∈ {(4, 4), (8, 8), (8, 16)}，hidden ∈ {64, 128}，种子 {0, 1, 2}。
    - 先按 3 个种子的验证损失平均选组合，再取该组合里验证损失最低的种子。
    - 写 `outputs/b4/net/selection.json`。
  - `curve()`：选中的组合下，训练集取 10 / 20 / 30 条（固定种子抽子集），算验证集 R²；写 `curve.json` 和一张简洁的 `curve.png`（画图前先读 dataviz 技能；图上只写名称和一行口径）。
  - `freeze(run: Path) -> Path`：写 `outputs/b4/net/frozen.json`，内容为权重 sha、P、F、hidden、cfg_sha、数据 sha、`git rev-parse HEAD`、时间；已存在就停（只冻结一次）。
  - `build_b4_policy(act_policy, mode, cfg_path, net_path=None)`：net 模式调用 `load_frozen`，`B4Deploy` 的 P、F 用冻结文件里的值。

- [ ] **Step 1: 写失败的测试**

```python
def test_weighted_mse_ignores_zero_weight_frames():
    y = torch.zeros(4, 5); p = y.clone(); p[0] = 9.0; w = torch.ones(4, 5); w[0] = 0
    assert N.weighted_mse(p, y, w).item() == 0.0

def test_overfits_tiny_dataset(tiny_data):
    assert N.train(tiny_data, seed=0, hidden=64, epochs=2000, patience=2000)["train_loss"] < 1e-8

def test_same_seed_same_weights(tiny_data):
    a, b = (torch.load(N.train(tiny_data, 0, 64, epochs=5)["path"]) for _ in range(2))
    assert all(torch.equal(a[k], b[k]) for k in a)

def test_one_step_changes_params_with_nonzero_grad(tiny_batch):
    m = O.B4Net(56, 16); opt = torch.optim.Adam(m.parameters(), 1e-3)
    x, y, w = tiny_batch
    N.weighted_mse(m(x), y, w).backward()
    assert any(p.grad.abs().sum() > 0 for p in m.parameters())
    before = [p.detach().clone() for p in m.parameters()]; opt.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))

def test_load_frozen_rejects_hash_mismatch(frozen_tmp):
    frozen_tmp.weights.write_bytes(frozen_tmp.weights.read_bytes() + b"x")
    with pytest.raises(SystemExit):
        O.load_frozen(str(frozen_tmp.json))

def test_net_mode_inputs_use_frozen_P_F():                             # 约束 2（tests/test_b4_deploy.py）
    log = run(B4Deploy(ChunkReplayer(CHUNKS), "net", CFG, "x", net=lambda x: np.zeros(5), P=4, F=4), 30)
    a_pi, ens = O.te_states(log["chunk"])
    assert log["x"].shape[1] == 56
    assert all(np.allclose(log["x"][k], O.online_input(a_pi, ens[k], k, 4, 4), atol=1e-6) for k in range(30))
    assert np.array_equal(log["u"], log["shift"].astype(np.float32))    # 网络输出 0 时就是 Shift
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_net.py -q && ~/miniconda3/envs/lerobot/bin/python -m pytest tests/test_b4_deploy.py -q`
  Expected: 新测试失败，之前的照常通过。
- [ ] **Step 3: 实现 `B4Net`、`weighted_mse`、`load_frozen`、`train`、`select`、`curve`、`freeze`，以及 `build_b4_policy` 的 net 模式**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: 全部通过。
- [ ] **Step 5: 交给用户**：依次执行 `python -u sim/b4_net.py select`（18 次训练）和 `python -u sim/b4_net.py curve`。
  Expected：打印选中的组合、验证集逐关节 R² 与学习曲线三个点。
  先**不要** freeze：冻结前要先做任务 10 的验证集回放检查。等反馈。

### Task 10: 回放评价（验证集冻结前检查、最终测试）

**Files:**
- Create: `lerobot-piper/sim/b4_eval.py`（`replay`、`analyze` 子命令）
- Test: `lerobot-piper/sim/test_b4_eval.py`

**Interfaces:**
- Consumes: Task 3（`apply_correction`、`B4Deploy`）、Task 4、Task 6–7（`seq_base.npz`、`walk_teacher` 用的 `result` 口径）、Task 9（`load_frozen`）。
- Produces:
  - `net_command(log: dict, seq: dict, net, P: int, F: int, cfg) -> np.ndarray`：
    - 在录制的整段预测上逐帧复算 shift、w、x，再经 `apply_correction` 得到 u；
    - 返回 `cmd7 = [u[0] × HOLD, u]`（float32）。
  - `plan(set_: str, ids: list[str]) -> list[tuple[str, str, int]]`，返回 (id, 条件, 轮)：
    - val：每条 2 轮 × {shift, net}，轮内先后交替；
    - test：每条 4 轮 × {shift, net, teacher}；第 r 轮的顺序从第 (r − 1) mod 3 个条件开始。
  - `replay --set val|test --session <dir> <id…>`：
    - 序列写进会话目录，回放同任务 7：`RR.run_child` + `S.guarded_run`，会话检查用 `check_session`。
    - 每遍提示「方块摆在 X cm 处（这一条录制时的位置），可以运动就按回车」；验证集与测试集都放方块，和闭环同条件。
    - 测试日：同一会话里，每条先跑任务 7 的 `run`（空桌），再跑这一条的评价。
  - `analyze_results(results: dict) -> dict`：results = {轨迹: {(条件, 轮): 指标 | None}}。
    - 组 = 轨迹，单位 = 轮；各键分别过 `rel_change` → `hier2`，再 `judge`。
    - 规定轮次不全的轨迹不进判定，`per_traj[轨迹]["status"] = "未完成"`。
    - 报告：网络 / ILC 收益比 =（Shift − 网络）/（Shift − 教师）；放方块后「Shift＋教师」相对 Shift 的变化（看有没有方块会不会让标签失效）；末端角速度等只报告。
  - `analyze --set val|test`：每遍的指标用与任务 7 相同的 `result` 口径，调 `analyze_results`，写 `outputs/b4/eval/<set>/summary.json`。

- [ ] **Step 1: 写失败的测试**

```python
def test_net_command_equals_deploy_output(case):                       # 约束 1、2：回放指令 = 部署时会发的
    live = run(B4Deploy(ChunkReplayer(case.log["chunk"]), "net", CFG, "x", net=case.net), len(case.log["chunk"]))
    cmd7 = E.net_command(case.log, case.seq, case.net, 8, 8, CFG)
    assert np.array_equal(cmd7[HOLD:], live["u"])

def test_test_plan_is_balanced():
    p = E.plan("test", ["a"])
    assert Counter(c for _, c, _ in p) == {"shift": 4, "net": 4, "teacher": 4}
    firsts = [next(c for _, c, r in p if r == rr) for rr in (1, 2, 3)]
    assert sorted(firsts) == ["net", "shift", "teacher"]

def test_analyze_never_calls_old_verdict(results, monkeypatch):       # 约束 3
    for name in ("verdict", "compare", "hier"):
        monkeypatch.setattr(I, name, lambda *a, _n=name, **k: pytest.fail(f"调用了 ilc_route1.{_n}"))
    out = E.analyze_results(results)                                   # 合成 10 条：网络关节更小、末端更差
    assert out["verdict"] == "有效"
    assert out["joint_raw"]["n_groups"] == 10                          # 组 = 轨迹

def test_incomplete_trajectory_is_reported_not_judged(results):
    results["t3"][("net", 4)] = None
    out = E.analyze_results(results)
    assert out["per_traj"]["t3"]["status"] == "未完成" and out["joint_raw"]["n_groups"] == 9
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_eval.py -q`
  Expected: 导入 `b4_eval` 失败。
- [ ] **Step 3: 实现 `net_command`、`plan`、`analyze_results` 与两个子命令**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: `4 passed`。
- [ ] **Step 5: 交给用户**：
  - 验证集：`replay --set val` 的命令与预期（10 条 × 4 遍，约 55 分钟）；`analyze --set val` 过了才执行 `b4_net.py freeze`。
  - 测试日：每条的命令顺序（`b4_teacher.py run`，再 `b4_eval.py replay --set test`）。
  等反馈。

### Task 11: 闭环最终评价（第五步）

**Files:**
- Modify: `lerobot-piper/scripts/run_b4.sh`（`final` 模式）
- Modify: `lerobot-piper/sim/b4_closed.py`（`final` 子命令）
- Test: `lerobot-piper/sim/test_b4_closed.py`

**Interfaces:**
- Consumes: Task 3（net 模式与冻结网络）、Task 4、Task 5（`load_closed`、`closed_metrics`、review CSV）、`positions.json` 的测试位置。
- Produces:
  - `final_plan() -> list[tuple[str, list[float]]]`：12 块 × 5 条。
    - 位置块：A = 测试位置 1–5，B = 6–10，按 A、B、A、B 的顺序。
    - 每个位置块内三个条件依次按 (S, N, R)、(N, R, S)、(R, S, N)、(S, N, R) 轮换；每个条件 20 条。
    - 数据集 `eval_pickcube_b4final_{shift,net,raw}`。
  - `success(stage: int) -> bool`：阶段 3 为成功。
  - `final_results(runs: dict) -> dict`：runs = {(条件, 块, 位置序号): {"metrics", "stage"} | None}；返回各指标的 hier2 结果、`n_pairs`、严格计数、`verdict`。
  - `final` 子命令：读三个数据集和 review CSV，调 `final_results`，写文件并打印结论。规则如下：
    - 配对 = 同一位置块、同一位置上的 S 与 N，共 20 对；缺一边的那对不进统计。
    - 每条用 `closed_metrics`；joint_raw、path、speed、duration、cmd_speed 各过 `hier2`（组 = 位置，单位 = 配对）。
    - 成功 = 阶段 3，做严格计数；结论用 `judge_closed`。
    - 报告 `chance_fail_prob(p, 20)`，p = 第二步录制（`eval_pickcube_b4rec`）的失败率。
    - 原始 ACT 组只报告：相对 S、相对 N 的关节偏差变化。
    - 写 `outputs/b4/final/summary.json`。

- [ ] **Step 1: 写失败的测试**

```python
def test_final_plan_counts():
    p = C.final_plan()
    assert len(p) == 12 and all(len(pos) == 5 for _, pos in p)
    assert Counter(c for c, _ in p) == {"shift": 4, "net": 4, "raw": 4}

def test_pair_with_missing_side_is_dropped_and_not_judged(runs_missing_one):   # Review Focus 5
    out = C.final_results(runs_missing_one)
    assert out["n_pairs"] == 19 and "未完成" in out["verdict"]

def test_raw_arm_never_enters_judgment(runs, runs_raw_terrible):
    assert C.final_results(runs)["verdict"] == C.final_results(runs_raw_terrible)["verdict"]

def test_success_is_stage_3():
    assert C.success(3) and not C.success(2)
```

- [ ] **Step 2: 运行，确认失败**
  Run: `OPENBLAS_NUM_THREADS=1 ~/miniconda3/envs/lerobot/bin/python -m pytest sim/test_b4_closed.py -q`
  Expected: 新测试失败，任务 5 的照常通过。
- [ ] **Step 3: 实现 `final_plan`、`success`、`final_results`、`final` 子命令和脚本的 final 模式**
- [ ] **Step 4: 运行，确认通过**：同上，Expected: 全部通过；再跑 `DRY=1 bash scripts/run_b4.sh final`，Expected: 打印 12 条命令和每块的位置，不连机械臂。
- [ ] **Step 5: 交给用户**：给出第五步的上机命令与预期（60 条，约 1 小时 15 分），以及标成败和 `b4_closed.py final` 的命令。等反馈。
