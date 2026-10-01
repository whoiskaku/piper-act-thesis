# G→π v2 阶段一（Q1）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用现有 100 集裸 ACT 数据、零机时，按留一会话检验 6 个 G 候选（3 种结构 × {只用指令, +实测状态历史}）预测原始误差（Q1）的能力。比较对象是按同口径重预测的冻结 G-2，另附一个折内拟合的线性 AR 参照。

**Architecture:** 新目录 `lerobot-piper/sim/q1/`，按职责拆成小模块，链路如下：
- 数据层；
- 时间轴规则（防泄漏规则只在这一处实现）；
- 两个参照：G-2 重预测、AR；
- 模型：分层伺服关节层 + 闭式传感器层，以及端到端 TCN / Transformer；
- 拼接损失 → 训练与预测 → 命令行与产物 → 汇总判据 → pp2 波形图。

所有可拟合部分只在每折的 4 个训练会话上拟合；计分沿用 `train_fabs_g0` 的 `P(·)` 口径。正式运行前先写预注册，由用户冻结。

**Tech Stack:** Python 3.10（`lerobot` conda 环境）· numpy 1.26 · scipy 1.15 · torch 2.11（CPU 单线程、确定性算法）· MuJoCo 3.9（`kin_gyro_fs`、G-q-v1）· pytest 9.1 · matplotlib

**Spec:** `lerobot-piper/reports/DESIGN_G_PI_V2.md`，用到 §2（G 的设计）、§4（阶段一）、§5 #1–4（用户 2026-10-02 定），含 `60291ce` 的两处核实订正。

⚠ **本计划授权的范围**：写代码与测试；Task 13 的 30 步耗时测量；预注册草稿。
**正式训练与评分（Task 14）必须等用户冻结 `reports/PREREG_Q1_STAGE1.md`。** 零机时，不连接机械臂，不做阶段二及以后。
**执行边界**（用户 2026-10-02）：Native 方式实施 Task 1–13，交付预注册草稿即停。

## Global Constraints

每条都是所有任务的隐含要求。引号内是 spec 或用户原文。

**授权与既有判决**
- 「本文件只是设计，不授权任何训练、采集或上机。每个阶段开工前另写预注册。」⇒ Task 14 只在 `PREREG_Q1_STAGE1.md` 冻结后执行。
- 「不改变任何既有判决：`step* = 0`、`F_inc` 限定 ab4 的 `d̄ = −42%`、全会话 `G0` 不通过、`ab2` 保留、③c『离线完成、运动未放行、无真机数据』。」
- 「G-2 及其结果冻结，2026-09-02 的『不再反向修改 G』不撤销；本设计是平行的新线…不修改 G-2。」⇒ 不改以下文件：
  - `outputs/step4_gvnext/{rq,rs}/` 和 `g2_incr_v1.npz`；
  - `sim/freeze_pi_g2wall.py`、`sim/g2_on_ab4.py`。
- 「不再扩展路线。」「不做：GRU、SUBNET / 深层 SSM、单独的容量对照。」

**时间轴与计分**
- 「`q̂_k`、`ω̂_k` 预测的就是第 `k` 周期记录的 `q_k`、`ω_k`」；「`e_t = ω_{t+3} − ω_ideal,t`（ALIGN_L = 3）」。
- 可见性：「实测可用到第 `t0` 帧（含）；指令可用 `u_{<t0}` 与当前 chunk 的 `u_{t0…t1}`」。
- 钳位：「超出 `t1` 的指令输入钳位到 `u_{t1}`（与 `build_command_features` 的 chunk 末端钳位一致）」。
- 计分范围：「计分的是当前 chunk 内的误差 `ê_t, t ∈ [t0, t1]`」。
- 计分口径：「逐集去均值 → bandpass41（0.5–9 Hz）→ 丢首尾 20 帧。集尾最后一个 chunk 按实际帧数截断。」
- 起点：「主分析的起点 = 每个 chunk 的开头…chunk 中段起点只作辅助报告，不混入主分数、不用于挑选模型。」
- 拟合范围：「所有可拟合部分…只在当前留一折的训练会话上拟合；不复用 G-2 已拟合的 `r_q`、`r_s`。」

**模型**
- 1a「不读任何实测」；「不能直接截取整集连续滚动，否则窗口尾部会用到下一 chunk 的指令。」1b「起点之后全部改用自身预测。」
- 「1a / 1b 分开训练，结构、规模与训练预算一致。」
- 历史窗：「历史窗长 `n = 20` 帧（1 s）…不扫描窗长。」「历史不足或实测无效时显式处理、不借未来数据…并加有效位掩码通道；所有模型同一规则。」
- 端到端：「直接多步预测——在起点一次输出窗口内的 `q̂` 与 `ω̂`，窗口内不递推。」「Transformer 规模与 TCN 相近。」
- 「监督信号相同：6 个模型都输出 `q̂` 与 `ω̂`，都用编码器 + IMU 训练。」

**参照**
- G-2：「冻结权重，按与候选相同的起点、可见指令和末端钳位重新预测，再按 §2.3 拼接评分。」整集前向的旧口径只作对照，不作判据。
- AR：「折内拟合的线性 AR 参照，不训练神经网络」；「系数只用每折的 4 个训练会话拟合；不复用旧脚本（`oracle_causal_feedback.py`）的权重与残差目标；输入只用起点可得的历史。」
- `F_abs`：「不做新口径重评分…只引用旧结果，并标明旧口径。」

**判据与措辞**
- 推进判据（§5 #1）：「每个留出会话的墙预测误差都不高于同口径 G-2，且五会话等权平均严格改善…每个候选单独判，不要求六个全部通过。这是开发筛查规则，不代表统计上已证明泛化。」
- 状态历史：「逐会话报告 b 相对 a 的变化，不用它替代①。」
- 「b 未优于 a 时，只能写『本轮未测到实测历史的预测收益』，不能写『状态历史无信息』，也不能推出『π 学不到有用反馈』。」
- 「阶段一即使成功，也只支持『20 Hz 口径下、这 5 个会话上的 Q1 预测收益』，不代表 Q2 / Q3。」

**单位、产物与环境**
- 单位：`state` 为 mdeg（`× π/180/1000` → rad），指令为 rad；20 Hz 朴素抽取口径；`R, b` 取 `outputs/step1_H/step1_H_v1.json`。
- 产物：保存每折每种子的权重与 sha256（`F_abs` 的教训：五折权重未保存 ⇒ 无法按新口径重评分）。
- 环境：入口 `gc.disable()`，用 `python -u`（MuJoCo + scipy 的 GC 段错误会吞掉缓冲输出）。
- 「改完 G 必出 pp2 波形图」：中位集 + 最难集、0.5–9 Hz、带口径。

**用户审阅订正（2026-10-02，静态审阅后定）**
- 端到端的 `q̂` 基线必须用**钳位后的可见指令**：`q̂` 进联合损失，同样不得读下一 chunk。`ê` 和 `q̂` 都要测「改 `t1` 之后的指令不变」。
- 1b 的附加线性项**只用掩码后的残差数值**，去掉截距和单独的有效比例通道 ⇒ 没有有效残差时附加项**精确为零**（1a 没有对应项）。
  - 同一原则也用于折内 AR 参照（无截距、无有效比例）。这是我按同理补的，列在预注册 §11，可撤回。
- 传感器层要处理无效关节样本：无效值（存为 0）不得经中心差分、FIR、带通污染邻近有效帧。
  - 拟合与目标只用支撑全有效的行；
  - 推理时支撑内有效/无效混合即报错。
- 正式入口（`prepare / job / refs / fullfit`）检查预注册已冻结、且 `code_sha` 与当前代码一致；`timing` 单独允许。
- 汇总强制核验：六模型 × 五折 × 三种子，以及权重、预测文件、配置指纹完整一致。
- `r_s` 用实测关节拟合，保留为首版；须注明两点，**不能把胜负单独归因于网络结构**：
  - 训练用实测关节、推理用 q̂（训练—推理差异）；
  - 分层方案与端到端方案的训练方法不同。

**Git 与安全**
- 在 `develop` 上开发与提交（CLAUDE.md）；提交信息**不加 `Co-Authored-By`**。
- **只 `git add` 本计划新建的文件**；永远不用 `git add -A` / `git add .`。另一会话的未提交工作一律不碰、不提交：
  - 已修改：`reports/PLAN_INDEPENDENT_RESPONSE_DATA.md`、`reports/PLAN_PILOT_RESPONSE_14.md`、`sim/replay_probe.py`；
  - 未跟踪：`reports/RELEASE_PILOT_RESPONSE_14.json`、`reports/RUNBOOK_PILOT_RESPONSE_14.md`、`sim/make_pilot_response14.py`、`sim/run_pilot_response14.py`、`sim/test_make_pilot_response14.py`、`sim/test_replay_pilot_scope.py`、`sim/test_run_pilot_response14.py`。
- 安全：不启动 pilot，不解除运动阻断，不碰 `replay_probe.py` 的硬阻断。

## Review Focus

1. **下一 chunk 的信息经派生量泄漏**（例如 `q̂` 的指令基线、理想陀螺的尾部）。窗口内一切输入与输出都只能依赖 `u_{≤t1}` 和 `≤ t0` 的实测。
   → Task 7 `test_no_leakage`（同时查 `ê` 与 `q̂`）；Task 9 `test_forward_windows_ignore_commands_after_t1`（训练用的前向，6 个模型）与 `test_prediction_ignores_measurements_after_origin`。
2. **无效样本污染邻近有效帧**：编码器前 2 帧、中途掉线（存为 0）经中心差分、FIR、带通扩散；chunk 结构不符也属此类。应掩码、排除或报错，绝不能当成真实读数。
   → Task 1 `test_measured_joints_masks_head_and_dropout`、`test_chunk_contract`；Task 2 `test_past_block_masks_mid_episode_dropout`、`test_band_clean_requires_full_support`；Task 5 `test_invalid_joint_values_never_enter_the_fit`、`test_window_support_must_not_mix_valid_and_invalid`；Task 8 `test_stored_value_at_invalid_frame_changes_nothing`。
3. **附加项在没有有效历史时不为零**：给 b 版（或 AR）混入与历史无关的校正能力。
   → Task 5 `test_cmap_term_is_exactly_zero_without_valid_history`；Task 4 `test_no_history_predicts_exactly_zero`。
4. **留出会话经统计量或闭式拟合泄漏**：归一化、`r_s`、1b 前瞻项、AR 系数都只能用训练折。
   → Task 5 `test_fit_rs_uses_only_given_episodes_and_is_deterministic`；Task 7 `test_norm_stats_use_only_given_episodes`；Task 9 `test_split_fold_excludes_held_out_session`；Task 10 `test_load_sensor_rejects_leaky_manifest`、`test_prepare_and_job_end_to_end_on_synthetic`；Task 11 `leaky_train` 用例。
5. **未冻结就运行，或混用、缺失、篡改产物，或结果不可复现**。
   → Task 10 `test_formal_commands_refuse_unless_frozen`、`test_list_jobs_skips_done_and_requeues_stale`、`test_code_sha_covers_the_run_entry`；Task 11 `test_load_runs_rejects_incomplete_or_inconsistent`（8 种损坏）；Task 9 `test_train_is_bit_reproducible_and_predicts`。

---

## 文件结构

全部新建于 `lerobot-piper/sim/q1/`。目录不放 `__init__.py`；模块名统一加 `q1_` 前缀，避免与 `sim/step4_gvnext/data.py` 等同名模块冲突。

| 文件 | 职责 |
|---|---|
| `conftest.py` | 关 GC、注册 `slow` 标记、会话级夹具 `bare_eps` / `calib` |
| `q1_data.py` | 100 集 → `Q1Episode`；数据契约（chunk 结构、编码器前 2 帧） |
| `q1_testing.py` | 合成集工厂（测试用，不读真实数据） |
| `q1_windows.py` | 起点、窗口、钳位、填充、拼接、计分、前瞻分层 —— **防泄漏规则的唯一实现处** |
| `q1_g2ref.py` | 冻结 G-2 的同口径窗口重预测；旧口径整集前向（只对照） |
| `q1_ar.py` | 折内线性 AR 参照 |
| `q1_sensor.py` | 分层模型的传感器层：`R·kin + r_s` 闭式拟合，1b 的实测残差前瞻项 |
| `q1_layered.py` | 分层伺服关节层：逐关节 MLP，1a 开环 / 1b 实测历史滚动 |
| `q1_e2e.py` | 端到端直接多步 TCN / Transformer |
| `q1_losses.py` | 拼接后的 ω 损失与关节损失（torch） |
| `q1_train.py` | 训练（固定步数、取末步）、预测、复现指纹 |
| `q1_run.py` | 命令行 `timing / prepare / list-jobs / job / refs / fullfit`；原子写产物 |
| `q1_report.py` | 逐会话 ρ、推进判据、b 相对 a、前瞻分层、辅助起点；输出 `summary.json` 与 Markdown 表 |
| `fig_pp2_q1.py` | pp2 17 集波形图（候选 / 实机 / G-2） |
| `run_q1_stage1.sh` | 正式运行（可重入、并行）；**冻结后**在用户终端执行 |
| `test_*.py` | 每个模块一份 |

新建报告：
- `reports/PREREG_Q1_STAGE1.md`：Task 13 写草稿，待用户冻结；
- `reports/RESULT_Q1_STAGE1.md`：Task 14 写。

产物目录 `outputs/q1_stage1/`。`outputs/` 被 `.gitignore` 忽略，不提交；报告里记哈希。

## 通用约定

- 所有命令在 `lerobot-piper/` 下执行；`PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python`。
- 测试命令：
  - 快速测试：`$PY -m pytest sim/q1 -q -m "not slow"`；
  - 集成测试：`$PY -m pytest sim/q1 -q -m slow`（读 100 集约 45 s，G-2 复现约 1 分钟）。
- 本计划的代码已在暂存区镜像里跑通：快速测试 94 个、集成测试 4 个全部通过。镜像通过软链接指向真实的 `outputs/`、`lerobot/` 和既有 `sim/` 模块。
  - 并且按任务顺序逐个重放过：每个任务在实现前测试失败，实现后通过；最终文件与镜像逐字节相同。
  - 照抄即可。若与仓库现状不符，先查原因再改；**不要为了让测试通过而放宽测试**。
- 提交信息格式 `阶段一 Q1：<内容>`，**不加 `Co-Authored-By`**。

---

### Task 1: 数据层（100 集 → `Q1Episode`）

**Files:**
- Create: `sim/q1/conftest.py`, `sim/q1/q1_data.py`, `sim/q1/q1_testing.py`
- Test: `sim/q1/test_q1_data.py`

**Interfaces:**
- Consumes:
  - `fit_step1_candidate.load_episode / _render_logged_chunks / calibrate_ideal_imu`
  - `lerobot.common.policies.act.b_runtime.IdealChunkRenderer`
  - `outputs/step1_H/step1_H_v1.json`
- Produces:
  - 常量：`ROOT, HERE, SESSIONS, EXPECTED_COUNTS, MIN_FRAMES=550, FS=20.0, ALIGN_L=3, EDGE=20, HIST=20, CHUNK_MAX=100, Q_VALID_FROM=2, MDEG, CALIB_JSON`
  - `class DataContractError(RuntimeError)`
  - `@dataclass Q1Episode(name, sess, path, T, cmd (Tc,7), ideal (Tc,3), caps (Tc,), q (T,6) rad, q_valid (T,), w (T,3), w_valid (T,), gap (n,3))`，属性 `n = T − 3`
  - 函数：
    - `calibration() -> (R (3,3), b (3,))`
    - `renderer()`
    - `check_chunks(chunks) -> list[dict]`
    - `measured_joints(state) -> (q, valid)`
    - `load_episode(path, sess) -> Q1Episode`
    - `load_bare(sessions=SESSIONS) -> list[Q1Episode]`
    - `by_session(eps) -> dict[str, list]`
  - `q1_testing.synthetic_episode(seed=0, T=587, sess="syn/A", chunk=100) -> Q1Episode`
  - `conftest` 夹具：`bare_eps`（100 集）、`calib`（`(R, b)`）

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_data.py`

```python
import numpy as np
import pytest

import q1_data as D
import q1_testing as T


def test_synthetic_episode_invariants():
    e = T.synthetic_episode(0)
    assert e.n == e.T - D.ALIGN_L
    assert e.cmd.shape == (600, 7) and e.ideal.shape == (600, 3) and e.caps.shape == (600,)
    assert not e.q_valid[:D.Q_VALID_FROM].any() and e.q_valid[D.Q_VALID_FROM:].all()
    assert np.allclose(e.gap, e.w[D.ALIGN_L:] - e.ideal[:e.n])


def test_measured_joints_masks_head_and_dropout():
    st = np.zeros((10, 7))
    st[2:, :6] = 1000.0
    st[6, :6] = 0.0                                   # 第 6 帧掉线（全零）
    q, valid = D.measured_joints(st)
    assert valid.tolist() == [False, False, True, True, True, True, False, True, True, True]
    assert np.allclose(q[2], np.radians(1.0)) and (q[6] == 0).all()


def test_measured_joints_rejects_nonzero_head():
    with pytest.raises(D.DataContractError):
        D.measured_joints(np.ones((10, 7)))


def test_chunk_contract():
    row = [0.0] * 7
    D.check_chunks([{"start": 100, "raw": [row] * 100}, {"start": 0, "raw": [row] * 100}])
    with pytest.raises(D.DataContractError):
        D.check_chunks([{"start": 0, "raw": [row] * 101}])
    with pytest.raises(D.DataContractError):
        D.check_chunks([{"start": 0, "raw": [row] * 100}, {"start": 120, "raw": [row] * 100}])


@pytest.mark.slow
def test_load_bare_matches_fabs_loader(bare_eps, calib):
    import fit_step1_candidate as S1
    import train_fabs_g0 as A
    assert {k: len(v) for k, v in D.by_session(bare_eps).items()} == D.EXPECTED_COUNTS
    old = A.load_all(S1, D.renderer(), *calib)
    assert [o.path for o in old] == [e.path for e in bare_eps]
    for e, o in zip(bare_eps, old):
        assert np.abs(e.gap - o.y).max() <= 2e-7                # o.y 存为 float32
        assert np.array_equal(e.ideal[:e.n], o.ideal)
        assert e.q_valid[D.Q_VALID_FROM:].all() and not e.q_valid[:D.Q_VALID_FROM].any()
        assert len(e.cmd) == 600 and set(e.caps.tolist()) == {99, 199, 299, 399, 499, 599}
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_data.py -q`
Expected: 收集时报 `ImportError`（`ModuleNotFoundError: No module named 'q1_data'`）

- [ ] **Step 3: 实现**

`sim/q1/conftest.py`：

```python
"""`sim/q1` 的 pytest 配置。

MuJoCo + scipy 共存时 `json.loads` 期间的 GC 会 SIGSEGV（见 infra 记录）⇒ 导入即关 GC。
"""
import gc
import sys
from pathlib import Path

import pytest

gc.disable()
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 读 100 集真实数据或跑 G-2/MuJoCo 的集成测试")


@pytest.fixture(scope="session")
def bare_eps():
    import q1_data as D
    return D.load_bare()


@pytest.fixture(scope="session")
def calib():
    import q1_data as D
    return D.calibration()
```

`sim/q1/q1_data.py`：

```python
"""阶段一（Q1）数据层 —— `reports/DESIGN_G_PI_V2.md` §2.3/§2.4。

100 集裸 ACT（5 会话）→ `Q1Episode`。标签直接取 `fit_step1_candidate.load_episode`
（`gap[t] = ω_real[t+3] − ω_ideal[t]`），与 `train_fabs_g0.load_all` 逐集相同。
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs",
           ROOT / "sim/step4_gvnext", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SESSIONS = (("ab5", "A"), ("ab6", "A"), ("ab7", "Apost"), ("ab2", "A"), ("ab4", "P0"))
EXPECTED_COUNTS = {"ab5/A": 40, "ab6/A": 30, "ab7/Apost": 15, "ab2/A": 10, "ab4/P0": 5}
MIN_FRAMES = 550
FS = 20.0
ALIGN_L = 3
EDGE = 20
HIST = 20                     # n = 20 帧（1 s）：初始设计选择，不扫描（用户 2026-10-02）
CHUNK_MAX = 100
Q_VALID_FROM = 2              # 核实 2026-10-02：100 集编码器 state 前 2 帧全为零
MDEG = np.pi / 180.0 / 1000.0
CALIB_JSON = ROOT / "outputs/step1_H/step1_H_v1.json"


class DataContractError(RuntimeError):
    """数据不满足已核实的契约（chunk 结构、编码器前 2 帧）。**不被 `load_bare` 吞掉。**"""


@dataclass
class Q1Episode:
    name: str                 # "ab5/A_ep0"
    sess: str                 # "ab5/A"
    path: str                 # 相对 ROOT
    T: int                    # 记录帧数
    cmd: np.ndarray           # (Tc,7) 完整 chunk 指令（含未执行尾部），rad
    ideal: np.ndarray         # (Tc,3) 渲染理想陀螺，已旋到传感器系
    caps: np.ndarray          # (Tc,) 每帧所在 chunk 的末帧
    q: np.ndarray             # (T,6) 实测关节 rad；无效行 = 0
    q_valid: np.ndarray       # (T,) bool
    w: np.ndarray             # (T,3) 实测陀螺（传感器系）
    w_valid: np.ndarray       # (T,) bool
    gap: np.ndarray           # (n,3) = w[t+3] − ideal[t]，n = T − 3

    @property
    def n(self) -> int:
        return len(self.gap)


def calibration() -> tuple[np.ndarray, np.ndarray]:
    cal = json.loads(CALIB_JSON.read_text())["calibration"]
    return (np.asarray(cal["R"], np.float64),
            np.asarray(cal.get("b", [0, 0, 0]), np.float64))


_RENDERER = None


def renderer():
    global _RENDERER
    if _RENDERER is None:
        from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
        _RENDERER = IdealChunkRenderer()
    return _RENDERER


def check_chunks(chunks: list[dict]) -> list[dict]:
    """chunk 契约：从 0 起连续、每个长度 1…CHUNK_MAX。→ 按起点排序的列表。"""
    cs = sorted(chunks, key=lambda c: int(c["start"]))
    end = 0
    for c in cs:
        if int(c["start"]) != end:
            raise DataContractError(f"chunk 不连续：期望起点 {end}，实际 {c['start']}")
        n = len(c["raw"])
        if not 1 <= n <= CHUNK_MAX:
            raise DataContractError(f"chunk 长度 {n} 不在 1…{CHUNK_MAX}")
        end += n
    return cs


def measured_joints(state) -> tuple[np.ndarray, np.ndarray]:
    """记录的 state（mdeg，取前 6 维）→ (q rad, 有效位)。全零行 = 无效（未就绪/掉线）。"""
    st = np.asarray(state, np.float64)[:, :6]
    nz = np.abs(st).sum(1) > 0
    if nz[:Q_VALID_FROM].any():
        raise DataContractError(f"前 {Q_VALID_FROM} 帧 state 非零 —— 与核实的数据事实不符")
    valid = nz & (np.arange(len(st)) >= Q_VALID_FROM)
    return np.where(valid[:, None], st * MDEG, 0.0), valid


def load_episode(path, sess: str) -> Q1Episode:
    """一集 → `Q1Episode`。帧数不足/缺 IMU 抛 `ValueError`；违反契约抛 `DataContractError`。"""
    import fit_step1_candidate as S1
    path = Path(path)
    pay = json.loads(path.read_text())
    fr = pay["frames"]
    T = len(fr)
    if T < MIN_FRAMES or any(f.get("imu") is None for f in fr):
        raise ValueError(f"{path}: 少于 {MIN_FRAMES} 帧或缺 IMU")
    chunks = check_chunks(pay["chunks"])
    R, b = calibration()
    e = S1.load_episode(path, "P0", R, b, renderer(), MIN_FRAMES)    # 标签的唯一来源
    cmd, ideal_sim, caps = S1._render_logged_chunks(chunks, renderer(), T)
    ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R.astype(np.float32),
                                   b.astype(np.float32))[:, 3:6].astype(np.float64)
    q, q_valid = measured_joints([f["state"] for f in fr])
    w = np.asarray([f["imu"] for f in fr], np.float64)[:, 3:6]
    gap = np.asarray(e.gap, np.float64)
    if gap.shape != (T - ALIGN_L, 3):
        raise DataContractError(f"{path}: gap 形状 {gap.shape}")
    try:
        rel = str(path.relative_to(ROOT))
    except ValueError:                                 # ROOT 之外的文件：记绝对路径
        rel = str(path)
    return Q1Episode(name=f"{sess.split('/')[0]}/{path.stem}", sess=sess,
                     path=rel, T=T, cmd=cmd, ideal=ideal, caps=caps,
                     q=q, q_valid=q_valid, w=w, w_valid=np.isfinite(w).all(1), gap=gap)


def load_bare(sessions=SESSIONS) -> list[Q1Episode]:
    """5 会话的裸 ACT 集；与 `train_fabs_g0.load_all` 同筛选、同顺序。"""
    out = []
    for s, arm in sessions:
        for p in sorted((ROOT / "outputs" / s).glob(f"{arm}_ep*.json"),
                        key=lambda q: int(q.stem.split("ep")[-1])):
            try:
                out.append(load_episode(p, f"{s}/{arm}"))
            except ValueError:
                continue
    return out


def by_session(eps) -> dict:
    d = {}
    for e in eps:
        d.setdefault(e.sess, []).append(e)
    return d
```

`sim/q1/q1_testing.py`：

```python
"""测试用合成集（不读真实数据）：100 帧一个 chunk、编码器前 2 帧无效、一阶滞后伺服。"""
from __future__ import annotations
import numpy as np
import q1_data as D


def synthetic_episode(seed: int = 0, T: int = 587, sess: str = "syn/A",
                      chunk: int = 100) -> D.Q1Episode:
    rng = np.random.default_rng(seed)
    Tc = -(-T // chunk) * chunk
    base = np.array([0.0, 0.8, -0.8, 0.0, 0.4, 0.0, 0.02])
    u = base + np.cumsum(rng.normal(0, 0.004, (Tc, 7)), 0)
    u[:, 3] = 0.0
    u[:, 6] = np.clip(u[:, 6], 0.0, 0.035)
    q = np.empty((T, 6))
    q[0] = u[0, :6]
    for k in range(1, T):
        q[k] = q[k - 1] + 0.3 * (u[k - 1, :6] - q[k - 1])
    q_valid = np.arange(T) >= D.Q_VALID_FROM
    q[~q_valid] = 0.0
    ideal = rng.normal(0, 0.05, (Tc, 3))
    w = rng.normal(0, 0.05, (T, 3))
    caps = np.repeat(np.arange(chunk - 1, Tc, chunk), chunk)
    gap = w[D.ALIGN_L:] - ideal[:T - D.ALIGN_L]
    return D.Q1Episode(name=f"{sess.split('/')[0]}/ep{seed}", sess=sess, path=f"synthetic/{seed}",
                       T=T, cmd=u, ideal=ideal, caps=caps, q=q, q_valid=q_valid,
                       w=w, w_valid=np.ones(T, bool), gap=gap)
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_data.py -q`
Expected: `5 passed`。其中 `test_load_bare_matches_fabs_loader` 约 90 s，因为要两次载入 100 集。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/conftest.py sim/q1/q1_data.py sim/q1/q1_testing.py sim/q1/test_q1_data.py
git commit -m "阶段一 Q1：数据层（100 集 → Q1Episode；chunk 与编码器前 2 帧契约）"
```

---

### Task 2: 时间轴规则（起点、窗口、钳位、填充、拼接、计分）

**Files:**
- Create: `sim/q1/q1_windows.py`
- Test: `sim/q1/test_q1_windows.py`

**Interfaces:**
- Consumes: `q1_data`（`HIST, ALIGN_L, EDGE, FS`）；`Q1Episode` 的 `cmd / ideal / caps / q / q_valid / w / w_valid / gap / n`
- Produces:
  - 常量：`MID_OFFSET=50`、`LEAD_BINS`
  - `@dataclass(frozen=True) Window(t0, t1, s1)`，属性 `n_scored`
  - 窗口与输入：
    - `chunk_windows(ep) -> list[Window]`
    - `mid_windows(ep, offset=50) -> list[Window]`
    - `visible(arr, t1, upto) -> ndarray`（第 0…upto 行，> t1 钳位）
    - `span(arr, lo, hi, fill) -> ndarray`（< 0 行填充）
    - `past_block(ep, t0, measured) -> dict`，键为 `u, ideal, m_cmd`（+ `q, w, m_q, m_w`）
  - 拼接与计分：
    - `stitch(n, windows, preds) -> (n,3)`
    - `overlay(base, windows, preds) -> (n,3)`
    - `band_clean(ok) -> (len,) bool`（带通后干净 ⇔ 原始 ±20 帧全有效；首尾 20 帧为否）
    - `kernel() -> float32 (41,)`
    - `P(v, K)`
    - `score(gap, ehat, K) -> (num, den, f3)`
    - `lead_of(n, windows) -> (n,)`
    - `lead_score(gap, ehat, windows, K) -> {"3-6": [num, den], …}`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_windows.py`

```python
import numpy as np
import pytest

import q1_data as D
import q1_testing as T
import q1_windows as W


def test_chunk_windows_cover_scored_frames_once_and_truncate_last_chunk():
    e = T.synthetic_episode(0, T=587)
    ws = W.chunk_windows(e)
    assert [(w.t0, w.t1) for w in ws] == [(0, 99), (100, 199), (200, 299),
                                          (300, 399), (400, 499), (500, 599)]
    assert ws[-1].s1 == e.n - 1 == 583
    W.stitch(e.n, ws, [np.zeros((w.n_scored, 3)) for w in ws])


def test_stitch_rejects_gaps_and_bad_shapes():
    e = T.synthetic_episode(0)
    ws = W.chunk_windows(e)
    with pytest.raises(ValueError):
        W.stitch(e.n, ws[:-1], [np.zeros((w.n_scored, 3)) for w in ws[:-1]])
    with pytest.raises(ValueError):
        W.stitch(e.n, ws, [np.zeros((w.n_scored + 1, 3)) for w in ws])


def test_visible_clamps_beyond_t1():
    a = np.arange(20, dtype=float)[:, None]
    v = W.visible(a, t1=9, upto=14)
    assert v.shape == (15, 1) and v[:10, 0].tolist() == list(range(10)) and (v[10:, 0] == 9).all()
    assert np.array_equal(W.visible(a, t1=9, upto=5), a[:6])


def test_span_pads_negative_frames():
    a = np.arange(10, dtype=float)[:, None] + 1
    assert W.span(a, -3, 2, fill=-1.0)[:, 0].tolist() == [-1, -1, -1, 1, 2, 3]


def test_past_block_padding_and_masks():
    e = T.synthetic_episode(0)
    pb = W.past_block(e, 0, measured=True)                 # t0 = 0：只有第 0 帧存在，且编码器无效
    assert pb["m_cmd"].tolist() == [0.0] * 19 + [1.0]
    assert not pb["m_q"].any() and pb["m_w"].tolist() == [0.0] * 19 + [1.0]
    assert np.allclose(pb["q"], e.cmd[0, :6]) and np.allclose(pb["u"][:19], e.cmd[0])
    assert (pb["ideal"][:19] == 0).all() and (pb["w"][:19] == 0).all()
    pb = W.past_block(e, 100, measured=True)
    assert pb["m_q"].all() and np.allclose(pb["q"], e.q[81:101])


def test_past_block_masks_mid_episode_dropout():
    e = T.synthetic_episode(0)
    e.q_valid = e.q_valid.copy(); e.q_valid[90] = False; e.q = e.q.copy(); e.q[90] = 0.0
    pb = W.past_block(e, 100, measured=True)
    i = 90 - 81
    assert pb["m_q"][i] == 0.0 and np.allclose(pb["q"][i], e.cmd[0, :6])


def test_score_matches_fabs_err_energy():
    import fit_step1_candidate as S1
    import train_fabs_g0 as A
    K = W.kernel()
    assert np.array_equal(K, A.bp_kernel(S1))
    e = T.synthetic_episode(1)
    eh = np.random.default_rng(0).normal(0, 0.01, e.gap.shape)
    num, den, f3 = W.score(e.gap, eh, K)
    a, c = A.err_energy(e.gap - eh, K)
    b, _ = A.err_energy(e.gap, K)
    assert np.isclose(num, a, rtol=1e-12) and np.isclose(den, b, rtol=1e-12) and f3 == c
    assert W.score(e.gap, np.zeros_like(e.gap), K)[0] == den
    assert W.score(e.gap, e.gap, K)[0] == 0.0


def test_mid_windows_and_overlay():
    e = T.synthetic_episode(0)
    mws = W.mid_windows(e)
    assert [w.t0 for w in mws] == [50, 150, 250, 350, 450, 550]
    out = W.overlay(np.zeros((e.n, 3)), mws, [np.ones((w.n_scored, 3)) for w in mws])
    assert out[49].sum() == 0 and out[50].sum() == 3 and out[100].sum() == 0


def test_band_clean_requires_full_support():
    ok = np.ones(200, bool); ok[100] = False
    bc = W.band_clean(ok)
    assert not bc[:D.EDGE].any() and not bc[-D.EDGE:].any()
    assert not bc[80:121].any() and bc[79] and bc[121]


def test_lead_bins_partition_scored_frames():
    e = T.synthetic_episode(2)
    ws = W.chunk_windows(e)
    K = W.kernel()
    tab = W.lead_score(e.gap, np.zeros_like(e.gap), ws, K)
    assert np.isclose(sum(v[1] for v in tab.values()), W.score(e.gap, 0 * e.gap, K)[1])
    lead = W.lead_of(e.n, ws)
    assert lead[0] == 3 and lead[99] == 102 and lead[100] == 3
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_windows.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_windows'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_windows.py`

```python
"""阶段一的时间轴规则 —— `DESIGN_G_PI_V2.md` §2.3（**防泄漏规则只在这里实现**）。

- 起点 `t0` = chunk 开头；实测可用到 `t0`（含）；指令/理想陀螺可用到 `t1`（当前 chunk 末帧），
  超出 `t1` 一律钳位到第 `t1` 帧（与 `build_command_features` 同规）。
- 计分 `ê_t, t ∈ [t0, s1]`，`s1 = min(t1, n−1)`；前瞻 = `t − t0 + 3`。
- 逐 chunk 拼接成整集，再按 `P(·)` 计分：逐集去均值 → bandpass41（0.5–9 Hz）→ 丢首尾 20。
- 历史不足或实测无效：指令填 `u_0`、理想陀螺填 0、编码器填 `u_0[:6]`、陀螺填 0，并给有效位掩码。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import q1_data as D

MID_OFFSET = 50               # 辅助报告的中段起点（默认值，可改）
LEAD_BINS = ((3, 6), (7, 12), (13, 27), (28, 52), (53, 102))   # 3–6 = chunk 头 4 帧


@dataclass(frozen=True)
class Window:
    t0: int                   # 起点：首个计分帧；实测可见到此帧（含）
    t1: int                   # 当前 chunk 末帧：指令可见到此帧（含）
    s1: int                   # 末个计分帧 = min(t1, n−1)

    @property
    def n_scored(self) -> int:
        return self.s1 - self.t0 + 1


def chunk_windows(ep) -> list[Window]:
    """主分析：每个 chunk 一个窗口，恰好覆盖 `[0, n−1]` 一次。"""
    out, t0 = [], 0
    while t0 < ep.n:
        t1 = int(ep.caps[t0])
        out.append(Window(t0, t1, min(t1, ep.n - 1)))
        t0 = t1 + 1
    return out


def mid_windows(ep, offset: int = MID_OFFSET) -> list[Window]:
    """辅助报告：chunk 中段起点 `t0 + offset`。**不进主分数、不用于挑模型。**"""
    return [Window(w.t0 + offset, w.t1, w.s1) for w in chunk_windows(ep)
            if w.t0 + offset <= w.s1]


def visible(arr, t1: int, upto: int) -> np.ndarray:
    """第 `0 … upto` 行；`> t1` 的行钳位到 `arr[t1]`（下一 chunk 不可见）。"""
    a = np.asarray(arr)
    head = a[:min(upto, t1) + 1]
    if upto <= t1:
        return np.array(head, copy=True)
    return np.concatenate([head, np.repeat(a[t1:t1 + 1], upto - t1, axis=0)], 0)


def span(arr, lo: int, hi: int, fill) -> np.ndarray:
    """第 `lo … hi` 行（含）；`< 0` 的行填 `fill`（历史不足，不借未来）。"""
    a = np.asarray(arr)
    k = np.arange(lo, hi + 1)
    out = np.empty((len(k),) + a.shape[1:], dtype=np.float64)
    neg = k < 0
    out[neg] = fill
    out[~neg] = a[k[~neg]]
    return out


def past_block(ep, t0: int, measured: bool) -> dict:
    """起点前 `HIST` 帧（`t0−19 … t0`）的输入与有效位。"""
    lo = t0 - D.HIST + 1
    k = np.arange(lo, t0 + 1)
    out = {"u": span(ep.cmd, lo, t0, ep.cmd[0]),
           "ideal": span(ep.ideal, lo, t0, 0.0),
           "m_cmd": (k >= 0).astype(np.float64)}
    if measured:
        ok = k >= 0
        qv = np.zeros(len(k), bool)
        wv = np.zeros(len(k), bool)
        qv[ok] = ep.q_valid[k[ok]]
        wv[ok] = ep.w_valid[k[ok]]
        q = np.repeat(ep.cmd[0:1, :6], len(k), 0).astype(np.float64)
        w = np.zeros((len(k), 3))
        q[qv] = ep.q[k[qv]]
        w[wv] = ep.w[k[wv]]
        out.update({"q": q, "w": w, "m_q": qv.astype(np.float64), "m_w": wv.astype(np.float64)})
    return out


def stitch(n: int, windows, preds) -> np.ndarray:
    """逐窗 `ê`（各 `(n_scored,3)`）拼成整集 `(n,3)`；要求恰好覆盖每帧一次。"""
    out = np.full((n, 3), np.nan)
    seen = np.zeros(n, int)
    for w, p in zip(windows, preds):
        p = np.asarray(p, np.float64)
        if p.shape != (w.n_scored, 3):
            raise ValueError(f"窗口 {w} 的预测形状 {p.shape}")
        out[w.t0:w.s1 + 1] = p
        seen[w.t0:w.s1 + 1] += 1
    if not (seen == 1).all():
        raise ValueError("窗口没有恰好覆盖每帧一次")
    return out


def overlay(base, windows, preds) -> np.ndarray:
    """辅助报告：在主拼接序列上，用中段起点的预测覆盖 `[t0', s1]`。"""
    out = np.array(base, dtype=np.float64, copy=True)
    for w, p in zip(windows, preds):
        out[w.t0:w.s1 + 1] = p
    return out


def band_clean(ok) -> np.ndarray:
    """带通（41 tap 居中）后第 k 帧「干净」⇔ 原始第 k−20 … k+20 帧全部有效；首尾 20 帧一律否。"""
    ok = np.asarray(ok, bool)
    bad = np.convolve((~ok).astype(np.float64), np.ones(2 * D.EDGE + 1), mode="same") > 0
    out = ~bad
    out[:D.EDGE] = False
    out[len(out) - D.EDGE:] = False
    return out


def kernel() -> np.ndarray:
    """与 `train_fabs_g0.bp_kernel` 逐位相同（0.5–9 Hz、41 tap、float32）。"""
    from scipy.signal import firwin
    return firwin(41, [0.5, 9.0], fs=D.FS, pass_zero=False).astype(np.float32)


def P(v, K) -> np.ndarray:
    r = np.asarray(v, np.float64)
    r = r - r.mean(0, keepdims=True)
    f = np.stack([np.convolve(r[:, j], K, mode="same") for j in range(r.shape[1])], 1)
    return f[D.EDGE:-D.EDGE]


def score(gap, ehat, K) -> tuple[float, float, int]:
    """→ (num = ‖P(gap − ê)‖², den = ‖P(gap)‖², 3·T_e)。与 `train_fabs_g0.err_energy` 同口径。"""
    a = P(np.asarray(gap) - np.asarray(ehat), K)
    d = P(gap, K)
    return float((a ** 2).sum()), float((d ** 2).sum()), int(a.size)


def lead_of(n: int, windows) -> np.ndarray:
    lead = np.full(n, -1, int)
    for w in windows:
        lead[w.t0:w.s1 + 1] = np.arange(w.n_scored) + D.ALIGN_L
    return lead


def lead_score(gap, ehat, windows, K, bins=LEAD_BINS) -> dict:
    """带通后的逐帧残差按前瞻归组。⚠ 带通混合相邻帧 ⇒ 分层结果是近似。"""
    n = len(gap)
    L = lead_of(n, windows)[D.EDGE:n - D.EDGE]
    a = P(np.asarray(gap) - np.asarray(ehat), K)
    d = P(gap, K)
    out = {}
    for lo, hi in bins:
        m = (L >= lo) & (L <= hi)
        out[f"{lo}-{hi}"] = [float((a[m] ** 2).sum()), float((d[m] ** 2).sum())]
    return out
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_windows.py -q`
Expected: `10 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_windows.py sim/q1/test_q1_windows.py
git commit -m "阶段一 Q1：时间轴规则（chunk 起点、末端钳位、填充掩码、拼接与 P(·) 计分）"
```

---

### Task 3: G-2 同口径重预测

**Files:**
- Create: `sim/q1/q1_g2ref.py`
- Test: `sim/q1/test_q1_g2ref.py`

**Interfaces:**
- Consumes:
  - `freeze_pi_g2wall.G2v0`：冻结 G-2，只读，`omega(dd) -> (N,3)` 机体系
  - `fit_step1_candidate.calibrate_ideal_imu`
  - `q1_windows.visible / stitch / Window`
- Produces:
  - `PAD=2`
  - `_dd(cmd, name) -> dict`
  - `load_g2()`
  - `omega_sensor(g2, cmd, name, R, b) -> (N,3)`
  - `window_ehat(g2, ep, w, R, b) -> (n_scored,3)`
  - `episode_ehat(g2, ep, windows, R, b) -> (n,3)`
  - `whole_episode_ehat(g2, ep, R, b) -> (n,3)`：旧口径，只对照

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_g2ref.py`

```python
import copy
import json

import numpy as np
import pytest

import q1_data as D
import q1_g2ref as G
import q1_testing as T
import q1_windows as W


@pytest.fixture(scope="module")
def g2():
    return G.load_g2()


def test_gyro100_is_not_read(g2):
    e = T.synthetic_episode(3)
    dd = G._dd(e.cmd[:150], "x")
    a = np.asarray(g2.omega(dd))
    dd["gyro100"] = np.full_like(dd["gyro100"], 9.0)
    assert np.array_equal(np.asarray(g2.omega(dd)), a)


def test_window_ignores_commands_after_t1(g2, calib):
    e = T.synthetic_episode(4)
    w = W.chunk_windows(e)[2]
    base = G.window_ehat(g2, e, w, *calib)
    e2 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e2.cmd = c
    assert np.array_equal(G.window_ehat(g2, e2, w, *calib), base)


@pytest.mark.slow
def test_whole_episode_reproduces_fabs_g0(bare_eps, calib, g2):
    """旧口径必须逐会话复现 `outputs/fabs_g0/fabs_g0_v1.json` 的 G-2 redu（实现自检）。"""
    old = {r["session"]: r["g2_redu"] for r in json.loads(
        (D.ROOT / "outputs/fabs_g0/fabs_g0_v1.json").read_text())["folds"]}
    K = W.kernel()
    acc = {}
    for e in bare_eps:
        n, d, _ = W.score(e.gap, G.whole_episode_ehat(g2, e, *calib), K)
        a = acc.setdefault(e.sess, [0.0, 0.0]); a[0] += n; a[1] += d
    for s, (n, d) in acc.items():
        assert abs((1 - n / d) - old[s]) < 1e-5, (s, 1 - n / d, old[s])


@pytest.mark.slow
def test_clamping_changes_only_last_scored_frames(bare_eps, calib, g2):
    e = bare_eps[0]
    whole = G.whole_episode_ehat(g2, e, *calib)
    for w in W.chunk_windows(e)[:-1]:                    # 末 chunk 的旧口径另有序列末端效应
        d = np.abs(G.window_ehat(g2, e, w, *calib) - whole[w.t0:w.s1 + 1]).max(1)
        changed = np.where(d > 0)[0] + w.t0
        assert len(changed) > 0 and changed.min() >= w.t1 - 5, (w, changed)


def test_mid_origin_equals_main_for_g2(g2, calib):
    """G-2 不读实测 ⇒ 中段起点（同一 t1）的预测与主窗口逐位相同（`refs` 据此令 aux = main）。"""
    e = T.synthetic_episode(5)
    w = W.chunk_windows(e)[1]
    m = W.mid_windows(e)[1]
    assert np.array_equal(G.window_ehat(g2, e, m, *calib), G.window_ehat(g2, e, w, *calib)[50:])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_g2ref.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_g2ref'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_g2ref.py`

```python
"""G-2（冻结）的同口径重预测 —— `DESIGN_G_PI_V2.md` §2.5。

**冻结权重，按与候选相同的起点、可见指令和末端钳位重新预测，再拼接评分。**
G-2 只读指令（`rs_features` 只用指令与预测的 q̂，不读实测），故起点只通过「可见指令」起作用。
⚠ `rq_X`/`rs_features`/`kin_gyro_fs` 用中心差分 ⇒ 钳位会改变 `t1` 附近的帧，不只尾部 3 帧。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

PAD = 2                       # 让 t1+3 处的中心差分落在序列内部


def _dd(cmd, name: str) -> dict:
    """G-2 前向所需的 `dd`（均匀 20 Hz 时基；`gyro100` 只为满足接口，不进前向）。"""
    N = len(cmd)
    t = np.arange(N) / D.FS
    return {"name": name, "cmd": np.asarray(cmd, np.float64), "t_send": t, "t_read": t,
            "tg_i": np.arange(N * 5) / (D.FS * 5), "gyro100": np.zeros((N * 5, 3))}


def load_g2():
    from freeze_pi_g2wall import G2v0
    return G2v0()


def omega_sensor(g2, cmd, name: str, R, b) -> np.ndarray:
    """指令 → G-2 陀螺并旋到传感器系（与 `train_fabs_g0.g2_score` 同变换）。"""
    import fit_step1_candidate as S1
    om = np.asarray(g2.omega(_dd(cmd, name)), np.float64)
    w6 = np.zeros((len(om), 6), np.float32)
    w6[:, 3:] = om
    return S1.calibrate_ideal_imu(w6, np.asarray(R, np.float32),
                                  np.asarray(b, np.float32))[:, 3:].astype(np.float64)


def window_ehat(g2, ep, w: W.Window, R, b) -> np.ndarray:
    """一个窗口的 `ê_t, t ∈ [t0, s1]`：只用 `u_{0…t1}`，其后钳位到 `u_{t1}`。"""
    ws = omega_sensor(g2, W.visible(ep.cmd, w.t1, w.t1 + D.ALIGN_L + PAD), ep.name, R, b)
    ts = np.arange(w.t0, w.s1 + 1)
    return ws[ts + D.ALIGN_L] - ep.ideal[ts]


def episode_ehat(g2, ep, windows, R, b) -> np.ndarray:
    return W.stitch(ep.n, windows, [window_ehat(g2, ep, w, R, b) for w in windows])


def whole_episode_ehat(g2, ep, R, b) -> np.ndarray:
    """**旧口径**（整集前向，窗口尾部用到下一 chunk 的指令）—— 只作对照，**不作判据**。"""
    ws = omega_sensor(g2, ep.cmd[:ep.T], ep.name, R, b)
    return ws[D.ALIGN_L:D.ALIGN_L + ep.n] - ep.ideal[:ep.n]
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_g2ref.py -q`
Expected: `5 passed`，约 2.5 分钟。

两条 slow 测试同时是**实现自检**：
- 旧口径必须逐会话复现 `fabs_g0_v1.json` 里的 G-2 redu（39.56 / 36.49 / 33.36 / 23.41 / 50.03%，容差 1e-5）；
- 钳位只改变每个 chunk 末 6 个计分帧（`t1−5 … t1`）。

两条任一失败都**停下查原因**，不要放宽。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_g2ref.py sim/q1/test_q1_g2ref.py
git commit -m "阶段一 Q1：G-2 冻结权重的同口径窗口重预测（旧口径整集前向只对照）"
```

---

### Task 4: 折内线性 AR 参照

**Files:**
- Create: `sim/q1/q1_ar.py`
- Test: `sim/q1/test_q1_ar.py`

**Interfaces:**
- Consumes: `q1_windows.chunk_windows / Window`；`Q1Episode.gap`
- Produces:
  - `N_FEAT = 60`（只有掩码后的历史数值；**无截距、无有效比例** ⇒ 没有有效历史时预测精确为 0）
  - `design_row(ep, t0) -> (60,)`
  - `fit(eps, windows_of=chunk_windows) -> {h: (60,3)}`
  - `window_ehat(coef, ep, w) -> (n_scored,3)`：缺前瞻时抛 `ValueError`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_ar.py`

```python
import copy

import numpy as np
import pytest

import q1_ar as AR
import q1_data as D
import q1_testing as T
import q1_windows as W


def test_design_row_uses_only_visible_history():
    e = T.synthetic_episode(0)
    x = AR.design_row(e, 100)
    assert x.shape == (AR.N_FEAT,) == (3 * D.HIST,)               # 只有历史数值：无截距、无有效比例
    assert np.allclose(x[:3], e.gap[97]) and np.allclose(x[57:60], e.gap[78])
    assert (AR.design_row(e, 0) == 0).all()


def test_no_history_predicts_exactly_zero():
    eps = [T.synthetic_episode(s) for s in range(6)]
    coef = AR.fit(eps[:5])
    assert (AR.window_ehat(coef, eps[5], W.chunk_windows(eps[5])[0]) == 0).all()


def test_prediction_ignores_gap_after_origin():
    eps = [T.synthetic_episode(s) for s in range(6)]
    coef = AR.fit(eps[:5])
    e = eps[5]
    w = W.chunk_windows(e)[2]
    base = AR.window_ehat(coef, e, w)
    e2 = copy.copy(e)
    g = e.gap.copy(); g[w.t0 - 2:] += 1.0; e2.gap = g          # e_{t0−2} 已用到 ω_{t0+1}
    assert np.array_equal(AR.window_ehat(coef, e2, w), base)


def test_fit_is_exact_on_a_linear_target():
    rng = np.random.default_rng(0)
    eps = [T.synthetic_episode(s) for s in range(16)]
    M = rng.normal(0, 0.1, (AR.N_FEAT, 3))
    sel = lambda e: [w for w in W.chunk_windows(e) if w.t0 >= 100]
    for e in eps:
        for w in sel(e):
            e.gap[w.t0] = AR.design_row(e, w.t0) @ M          # 只改 h = 0 的目标
    coef = AR.fit(eps, windows_of=sel)
    w = sel(eps[0])[1]
    assert np.allclose(AR.window_ehat(coef, eps[0], w)[0], AR.design_row(eps[0], w.t0) @ M)


def test_missing_lead_raises():
    e = T.synthetic_episode(1)
    with pytest.raises(ValueError):
        AR.window_ehat({0: np.zeros((AR.N_FEAT, 3))}, e, W.chunk_windows(e)[0])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_ar.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_ar'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_ar.py`

```python
"""折内拟合的线性 AR 参照（**不训练神经网络**）—— `DESIGN_G_PI_V2.md` §2.5、§5 #4。

目标 = gap `e_t`（t ∈ [t0, s1]）；输入 = 起点可得的最近 20 个**原始**实测 gap
`e_{t0−3−l}`（l = 0…19；`e_{t0−3}` 只用到 `ω_{t0}`），三轴联合；每个前瞻一组系数；
普通最小二乘、不调超参。历史不足（`t0−3−l < 0`）填 0。
**无截距、无有效比例列**：没有有效历史时预测精确为 0；逐前瞻截距会学到 chunk 相位的平均
模板，那不是「误差延续」（与 1b 附加项同一原则，用户 2026-10-02 订正 1b 时定）。
不复用 `oracle_causal_feedback.py` 的权重与残差目标。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

N_FEAT = 3 * D.HIST


def design_row(ep, t0: int) -> np.ndarray:
    """→ (N_FEAT,) = [e_{t0−3}, …, e_{t0−22}]（各 3 轴；无效 = 0）。"""
    idx = t0 - D.ALIGN_L - np.arange(D.HIST)
    ok = idx >= 0
    h = np.zeros((D.HIST, 3))
    h[ok] = ep.gap[idx[ok]]
    return h.ravel()


def fit(eps, windows_of=W.chunk_windows) -> dict:
    """→ {前瞻 h: (N_FEAT, 3) 系数}。**只用传入的（训练折）集。**"""
    rows: dict = {}
    for e in eps:
        for w in windows_of(e):
            x = design_row(e, w.t0)
            for h in range(w.n_scored):
                X, Y = rows.setdefault(h, ([], []))
                X.append(x)
                Y.append(e.gap[w.t0 + h])
    return {h: np.linalg.lstsq(np.asarray(X), np.asarray(Y), rcond=None)[0]
            for h, (X, Y) in rows.items()}


def window_ehat(coef: dict, ep, w: W.Window) -> np.ndarray:
    missing = [h for h in range(w.n_scored) if h not in coef]
    if missing:
        raise ValueError(f"前瞻 {missing[:3]} 在训练折里没有样本")
    x = design_row(ep, w.t0)
    return np.stack([x @ coef[h] for h in range(w.n_scored)])
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_ar.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_ar.py sim/q1/test_q1_ar.py
git commit -m "阶段一 Q1：折内拟合的线性 AR 参照（不训练神经网络；无截距）"
```

---

### Task 5: 分层模型的传感器层

**Files:**
- Create: `sim/q1/q1_sensor.py`
- Test: `sim/q1/test_q1_sensor.py`

**Interfaces:**
- Consumes:
  - `ceiling_200hz.kin_gyro_fs`（MuJoCo，机体系）
  - `train_rq.fir_design`（因果 FIR + 常数列）
  - `q1_windows.chunk_windows`
- Produces:
  - 常量：`NLAG=16, LAM=1e-4, N_FEAT=37, STENCIL_BACK=17, STENCIL_FWD=2, SPAN_WARM=20, SUPPORT_BACK=14`
  - 基础：
    - `kin_body(q) -> (L,3)`
    - `features(u7, q, ideal, kin) -> (L,37)`
    - `fir(X) -> (L, 16·37+1)`
    - `bp(x)`
  - 无效样本：
    - `clean_rows(q_valid, w_valid) -> (T,) bool`（`q_{k−17…k+2}` 全有效且 `ω_k` 有效）
    - `check_window_support(ep, t0)`（支撑 `t0−14…t0` 有效/无效混合 ⇒ `DataContractError`）
  - 闭式层：
    - `omega_span(q, u7, ideal, R, Wrs) -> (L,3)`（传感器系）
    - `fit_rs(eps, R) -> Wrs (593,3)`（只用 `band_clean(clean_rows)` 的行）
  - 1b 前瞻项（**无截距、无有效比例**）：
    - `residual_history(ep, t0, R, Wrs) -> (60,)`（没有有效残差时全 0）
    - `sensor_residual(ep, R, Wrs) -> (T,3)`（不干净的帧 = NaN）
    - `fit_cmap(eps, R, Wrs) -> {h: (60,3)}`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_sensor.py`

```python
import copy

import numpy as np
import pytest

import q1_data as D
import q1_sensor as S
import q1_testing as T
import q1_windows as W


def test_features_layout():
    e = T.synthetic_episode(0)
    q = e.q[2:60]
    X = S.features(e.cmd[2:60], q, e.ideal[2:60], S.kin_body(q))
    assert X.shape == (58, S.N_FEAT)
    assert np.allclose(X[:, :6], e.cmd[2:60, :6]) and np.allclose(X[:, 6:12], q)
    assert np.allclose(X[:, 33], e.cmd[2:60, 6]) and np.allclose(X[:, 34:], e.ideal[2:60])


def test_fir_is_causal():
    X = np.random.default_rng(0).normal(size=(50, 4))
    X2 = X.copy(); X2[30:] += 1.0
    assert np.array_equal(S.fir(X2)[:30], S.fir(X)[:30])


def test_fit_rs_uses_only_given_episodes_and_is_deterministic():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    W1, W2, W3 = S.fit_rs(eps[:3], R), S.fit_rs(eps[:3], R), S.fit_rs(eps[1:], R)
    assert W1.shape == (S.NLAG * S.N_FEAT + 1, 3)
    assert np.array_equal(W1, W2) and not np.allclose(W1, W3)


def test_residual_history_uses_only_data_up_to_origin():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps[:3], R)
    e = eps[3]
    x = S.residual_history(e, 100, R, Wrs)
    e2 = copy.copy(e)
    q = e.q.copy(); q[101:] += 0.05; e2.q = q
    ww = e.w.copy(); ww[101:] += 1.0; e2.w = ww
    assert np.array_equal(S.residual_history(e2, 100, R, Wrs), x)
    assert x.shape == (3 * D.HIST,) and (x != 0).all()
    assert (S.residual_history(e, 0, R, Wrs) == 0).all()


def test_cmap_term_is_exactly_zero_without_valid_history():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps, R)
    C = S.fit_cmap(eps, R, Wrs)
    x0 = S.residual_history(eps[0], 0, R, Wrs)
    assert all(C[h].shape == (3 * D.HIST, 3) and (x0 @ C[h] == 0).all() for h in C)


def _dropout(e, k, value):
    e.q_valid = e.q_valid.copy(); e.q_valid[k] = False
    e.q = e.q.copy(); e.q[k] = value
    return e


def test_clean_rows_and_band_clean_cover_the_stencils():
    e = _dropout(T.synthetic_episode(0), 300, 0.0)
    ok = S.clean_rows(e.q_valid, e.w_valid)
    assert not ok[:S.STENCIL_BACK + D.Q_VALID_FROM].any() and ok[S.STENCIL_BACK + D.Q_VALID_FROM]
    assert not ok[298:318].any() and ok[297] and ok[318]           # k−17 ≤ 300 ≤ k+2
    bc = W.band_clean(ok)
    assert not bc[278:338].any() and bc[277] and bc[338]


def test_invalid_joint_values_never_enter_the_fit():
    """无效帧里存的数（0 或任意值）不得经中心差分 / FIR / 带通影响拟合与有效帧的残差。"""
    R = np.eye(3)
    a = [_dropout(T.synthetic_episode(s), 300, 0.0) for s in range(4)]
    b = [_dropout(T.synthetic_episode(s), 300, 5.0) for s in range(4)]
    Wa, Wb = S.fit_rs(a, R), S.fit_rs(b, R)
    assert np.array_equal(Wa, Wb)
    ya, yb = S.sensor_residual(a[0], R, Wa), S.sensor_residual(b[0], R, Wa)
    assert np.array_equal(np.isnan(ya), np.isnan(yb)) and np.array_equal(ya[~np.isnan(ya)], yb[~np.isnan(yb)])
    assert np.array_equal(S.residual_history(a[0], 310, R, Wa), S.residual_history(b[0], 310, R, Wa))


def test_window_support_must_not_mix_valid_and_invalid():
    e = T.synthetic_episode(0)
    S.check_window_support(e, 0)                                   # t0 = 0：支撑全是填充 ⇒ 允许
    S.check_window_support(e, 100)                                 # 全是有效实测 ⇒ 允许
    with pytest.raises(D.DataContractError):
        S.check_window_support(_dropout(e, 95, 0.0), 100)


def test_fit_cmap_covers_all_leads():
    eps = [T.synthetic_episode(s) for s in range(4)]
    R = np.eye(3)
    Wrs = S.fit_rs(eps, R)
    C = S.fit_cmap(eps, R, Wrs)
    assert sorted(C) == list(range(100)) and C[0].shape == (3 * D.HIST, 3)


@pytest.mark.slow
def test_measured_joint_sensor_layer_explains_gap(bare_eps, calib):
    """实现自检（**不是模型分数**：用了未来的实测关节）：带通后应解释大部分 gap。"""
    R, _ = calib
    tr = [e for e in bare_eps if e.sess != "ab5/A"]
    te = [e for e in bare_eps if e.sess == "ab5/A"][:10]
    Wrs = S.fit_rs(tr, R)
    K = W.kernel()
    num = den = 0.0
    for e in te:
        a = D.Q_VALID_FROM
        full = np.zeros((e.T, 3))
        full[a:] = S.omega_span(e.q[a:], e.cmd[a:e.T], e.ideal[a:e.T], R, Wrs)
        eh = full[D.ALIGN_L:D.ALIGN_L + e.n] - e.ideal[:e.n]
        n_, d_, _ = W.score(e.gap[30:], eh[30:], K)
        num += n_; den += d_
    assert 1 - num / den > 0.3
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_sensor.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_sensor'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_sensor.py`

```python
"""分层模型的传感器层 —— `ω̂ = R·kin(q̂) + r_s`（+ 1b 的实测残差前瞻项）。§2.4。

`r_s`：因果 FIR（nlag 16、相对岭 1e-4 —— 沿用 G-2 `r_s` 的选定值，不调参），
特征 = G-2 `rs_features` 的 34 维 + 理想陀螺 3 维（§2.6 输入相同）= 37 维。
**折内、用实测关节闭式拟合**：目标 = `ω_real − R·kin(q_real)`（与 G-2「目标用真关节」同理），
特征也取自实测关节 ⇒ 不需要关节层的折外预测、无折内泄漏。
⚠ 代价 = **训练用实测关节、推理用 q̂**（训练—推理差异，首版保留，报告须注明）。
拟合在评分频带（0.5–9 Hz）内做。**不复用 G-2 已拟合的 `r_s`。**

无效关节样本（存为 0）：FIR 第 k 行用到 `q_{k−17 … k+2}`（16 个滞后 + 二阶中心差分），
只有这段支撑全有效的行才进拟合/目标/历史；带通拟合再要求 ±20 帧全干净。
1b 的附加项**无截距、无有效比例**：没有有效残差时精确为 0（用户 2026-10-02 订正）。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

NLAG = 16
LAM = 1e-4
N_FEAT = 37
STENCIL_BACK = NLAG + 1        # FIR 第 k 行 ← 特征 k−15…k ← q_{k−17…k+2}
STENCIL_FWD = 2
SPAN_WARM = NLAG + 4
SUPPORT_BACK = STENCIL_BACK - D.ALIGN_L    # 首个计分输出（帧 t0+3）的支撑回到 t0−14


def kin_body(q) -> np.ndarray:
    from ceiling_200hz import kin_gyro_fs
    return kin_gyro_fs(np.asarray(q, np.float64), D.FS)


def features(u7, q, ideal, kin) -> np.ndarray:
    """→ (L,37) = [u6, q, dq, ddq, kin(机体系), |dq|, 夹爪, 理想陀螺]。"""
    dq = np.gradient(q, 1 / D.FS, axis=0)
    ddq = np.gradient(dq, 1 / D.FS, axis=0)
    return np.concatenate([np.asarray(u7)[:, :6], q, dq, ddq, kin, np.abs(dq),
                           np.asarray(u7)[:, 6:7], ideal], 1)


def fir(X, nlag: int = NLAG) -> np.ndarray:
    from train_rq import fir_design
    return fir_design(np.asarray(X, np.float64), nlag)


def bp(x) -> np.ndarray:
    from scipy.signal import firwin
    k = firwin(41, [0.5, 9.0], fs=D.FS, pass_zero=False)
    x = np.asarray(x, np.float64)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def clean_rows(q_valid, w_valid) -> np.ndarray:
    """原始行 k「干净」⇔ `q_{k−17 … k+2}` 全有效且 `ω_k` 有效（无效样本不经差分/FIR 进入该行）。"""
    qv = np.asarray(q_valid, bool)
    T = len(qv)
    ok = np.zeros(T, bool)
    for k in range(STENCIL_BACK, T - STENCIL_FWD):
        ok[k] = bool(qv[k - STENCIL_BACK:k + STENCIL_FWD + 1].all() and w_valid[k])
    return ok


def check_window_support(ep, t0: int) -> None:
    """1b 推理：首个计分输出的支撑（帧 t0−14 … t0，来自起点前的实测历史）要么全有效、要么全是填充。
    混合 ⇒ 填充值会经差分/FIR 污染有效帧 ⇒ 按数据契约报错（本数据只有 t0 = 0 全填充）。"""
    k = np.arange(t0 - SUPPORT_BACK, t0 + 1)
    v = np.zeros(len(k), bool)
    ok = k >= 0
    v[ok] = ep.q_valid[k[ok]]
    if v.any() and not v.all():
        raise D.DataContractError(f"{ep.name} t0={t0}: 传感器层支撑内实测有效/无效混合")


def omega_span(q, u7, ideal, R, Wrs) -> np.ndarray:
    """一段跨度上的 `ω̂`（传感器系）= R·kin(q) + r_s；FIR 在跨度起点零历史。"""
    kb = kin_body(q)
    return kb @ np.asarray(R).T + fir(features(u7, q, ideal, kb)) @ Wrs


def fit_rs(eps, R) -> np.ndarray:
    """折内闭式拟合：带内岭回归，`λ = LAM·tr(G)/dim`（同 `train_rq.fit`）；只用干净行。"""
    G = B = None
    for e in eps:
        kb = kin_body(e.q)
        Dm = fir(features(e.cmd[:e.T], e.q, e.ideal[:e.T], kb))
        Y = e.w - kb @ np.asarray(R).T
        m = W.band_clean(clean_rows(e.q_valid, e.w_valid))
        Db, Yb = bp(Dm)[m], bp(Y)[m]
        G = Db.T @ Db if G is None else G + Db.T @ Db
        B = Db.T @ Yb if B is None else B + Db.T @ Yb
    G = G + LAM * (np.trace(G) / G.shape[0]) * np.eye(G.shape[0])
    return np.linalg.solve(G, B)


def residual_history(ep, t0: int, R, Wrs) -> np.ndarray:
    """1b：起点前 20 帧的实测传感器残差 `s_k = ω_k − (R·kin(q_k) + r_s)`，**只用 ≤ t0 的实测**
    （跨度止于 t0 ⇒ 末端单侧差分），且只保留支撑内实测全有效的样本，其余置 0。
    → (3·HIST,)；没有有效样本时全为 0 ⇒ 1b 的附加项精确为 0。"""
    lo = t0 - D.HIST + 1
    a = max(0, lo - SPAN_WARM)
    h = np.zeros((D.HIST, 3))
    if t0 - a + 1 > STENCIL_BACK + 1:
        qv, wv = ep.q_valid[a:t0 + 1], ep.w_valid[a:t0 + 1]
        s = ep.w[a:t0 + 1] - omega_span(ep.q[a:t0 + 1], ep.cmd[a:t0 + 1], ep.ideal[a:t0 + 1], R, Wrs)
        for i, k in enumerate(range(lo, t0 + 1)):
            j = k - a
            if j >= STENCIL_BACK and wv[j] and qv[j - STENCIL_BACK:min(j + STENCIL_FWD, t0 - a) + 1].all():
                h[i] = s[j]
    return h.ravel()


def sensor_residual(ep, R, Wrs) -> np.ndarray:
    """拟合目标：整集实测关节上的传感器残差，帧 0…T−1（不干净的帧 = NaN）。"""
    out = ep.w - omega_span(ep.q, ep.cmd[:ep.T], ep.ideal[:ep.T], R, Wrs)
    out[~clean_rows(ep.q_valid, ep.w_valid)] = np.nan
    return out


def fit_cmap(eps, R, Wrs, windows_of=W.chunk_windows) -> dict:
    """1b 的逐前瞻线性项：`s_{t0+h+3} ≈ x(t0)·C_h`（无截距），只用训练折，普通最小二乘。"""
    rows: dict = {}
    for e in eps:
        y = sensor_residual(e, R, Wrs)
        for w in windows_of(e):
            x = residual_history(e, w.t0, R, Wrs)
            for h in range(w.n_scored):
                k = w.t0 + h + D.ALIGN_L
                if k < e.T and np.isfinite(y[k]).all():
                    X, Y = rows.setdefault(h, ([], []))
                    X.append(x)
                    Y.append(y[k])
    return {h: np.linalg.lstsq(np.asarray(X), np.asarray(Y), rcond=None)[0]
            for h, (X, Y) in rows.items()}
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_sensor.py -q`
Expected: `10 passed`

slow 测试 `test_measured_joint_sensor_layer_explains_gap` 是实现自检：用了未来实测关节，**不是模型分数**。镜像里实测 redu 0.80，门槛 0.3 只为抓旋转、符号一类的粗错。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_sensor.py sim/q1/test_q1_sensor.py
git commit -m "阶段一 Q1：分层模型传感器层（折内实测关节闭式拟合 r_s；无效样本按支撑排除；1b 前瞻项无截距）"
```

---

### Task 6: 分层伺服关节层（1a / 1b）

**Files:**
- Create: `sim/q1/q1_layered.py`
- Test: `sim/q1/test_q1_layered.py`

**Interfaces:**
- Consumes: `q1_windows.span / visible / Window`；`Q1Episode`
- Produces:
  - 常量：`HID=32, G_INIT=0.3, ROLL_PAD=2`
  - `joint_stats(eps) -> {"s_eps": (6,), "s_v": (6,)}`
  - `make_servo(stats) -> nn.Module`，方法：
    - `.step(qh, uh, mh, u_next)`
    - `.rollout(qh, uh, mh, u_future) -> (B,K,6)`
    - 共 24 780 参数
  - 历史与输入：
    - `history_measured(ep, t0) -> (q (6,21), u (6,21), m (21,))`
    - `history_from_traj(ep, traj, t0)`（同上形状）
    - `open_loop_traj(net, ep, upto) -> (upto+1, 6)`
    - `future_commands(ep, w, K) -> (K,6)`
    - `window_inputs(net, ep, windows, mode, traj=None) -> (Q, U, M, F, K)`（torch）

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_layered.py`

```python
import copy

import numpy as np
import pytest
import torch

import q1_data as D
import q1_layered as LM
import q1_testing as T
import q1_windows as W


def _net(eps, seed=0, jitter=0.05):
    torch.manual_seed(seed)
    net = LM.make_servo(LM.joint_stats(eps))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * jitter)          # 打破零初始化才测得出依赖
    return net


def _pred(net, e, mode):
    ws = W.chunk_windows(e)
    traj = LM.open_loop_traj(net, e, max(w.t0 for w in ws)) if mode == "a" else None
    Q, U, M, F, _ = LM.window_inputs(net, e, ws, mode, traj)
    with torch.no_grad():
        return net.rollout(Q, U, M, F).numpy()


def test_zero_head_is_first_order_lag():
    eps = [T.synthetic_episode(s) for s in range(3)]
    torch.manual_seed(0)
    net = LM.make_servo(LM.joint_stats(eps))
    q = torch.zeros(1, 6, D.HIST + 1); u = torch.ones(1, 6, D.HIST + 1); m = torch.ones(1, D.HIST + 1)
    assert torch.allclose(net.step(q, u, m, torch.ones(1, 6)), torch.full((1, 6), LM.G_INIT))


@pytest.mark.parametrize("mode", ["a", "b"])
def test_no_commands_after_t1_and_no_measured_after_t0(mode):
    eps = [T.synthetic_episode(s) for s in range(3)]
    net = _net(eps)
    e = eps[2]
    w = W.chunk_windows(e)[2]
    base = _pred(net, e, mode)
    e1 = copy.copy(e); c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    assert np.array_equal(_pred(net, e1, mode)[2], base[2])
    e2 = copy.copy(e); q = e.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    assert np.array_equal(_pred(net, e2, mode)[2], base[2])


def test_1a_reads_no_measured_data_and_1b_does():
    eps = [T.synthetic_episode(s) for s in range(3)]
    net = _net(eps)
    e = eps[2]
    e2 = copy.copy(e); e2.q = e.q + 0.01
    assert np.array_equal(_pred(net, e2, "a"), _pred(net, e, "a"))
    assert not np.array_equal(_pred(net, e2, "b"), _pred(net, e, "b"))


def test_1b_first_chunk_history_is_fully_masked():
    e = T.synthetic_episode(0)
    q, u, m = LM.history_measured(e, 0)
    assert m.sum() == 0 and np.allclose(q, e.cmd[0, :6][:, None])
    q, u, m = LM.history_measured(e, 100)
    assert m.all() and np.allclose(q.T, e.q[80:101])


def test_open_loop_traj_starts_at_rest_on_first_command():
    eps = [T.synthetic_episode(s) for s in range(2)]
    net = _net(eps)
    tr = LM.open_loop_traj(net, eps[1], 120)
    assert tr.shape == (121, 6) and np.allclose(tr[0], eps[1].cmd[0, :6])


def test_param_count():
    eps = [T.synthetic_episode(s) for s in range(2)]
    torch.manual_seed(0)
    assert sum(p.numel() for p in LM.make_servo(LM.joint_stats(eps)).parameters()) == 24780
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_layered.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_layered'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_layered.py`

```python
"""分层伺服响应网络的关节层（1a / 1b）—— `DESIGN_G_PI_V2.md` §2.4。

执行器网络（Hwangbo 2019）改编到运动学层：每个关节一套 MLP（3×32 softsign，权重不共享），
输入 = 20 帧跟踪误差 + 20 帧速度 + 20 帧有效位 + 下一帧指令误差（共 61 维），
输出下一帧关节增量：`Δq̂ = g_j·(u_{k+1} − q̂_k) + s_v,j·MLP_j(x)`。
头部零初始化 ⇒ 起点是一阶滞后伺服（`g = G_INIT`）。
- 1a：**不读任何实测**；起点状态取自从静止（`q̂_0 = u_0`）起、只用 `u_{≤t0}` 的开环滚动。
- 1b：起点读实测编码器 `q_{t0−20 … t0}`（无效 = `u_0`、掩码 0），之后全用自身预测。
窗口内指令超出 `t1` 钳位到 `u_{t1}`。传感器层见 `q1_sensor.py`。
"""
from __future__ import annotations

import numpy as np

import q1_data as D
import q1_windows as W

HID = 32
G_INIT = 0.3                  # 一阶滞后先验（≈ 3 帧滞后），默认值
ROLL_PAD = 2                  # 多滚 2 帧，让 t1+3 的中心差分落在内部
SD_FLOOR = 1e-6


def joint_stats(eps) -> dict:
    """关节层归一化尺度（只用传入的训练折、有效实测帧）。"""
    E, V = [], []
    for e in eps:
        k = np.where(e.q_valid)[0]
        E.append(e.cmd[k, :6] - e.q[k])
        kk = k[1:][np.diff(k) == 1]
        V.append(e.q[kk] - e.q[kk - 1])
    return {"s_eps": np.maximum(np.concatenate(E).std(0), SD_FLOOR),
            "s_v": np.maximum(np.concatenate(V).std(0), SD_FLOOR)}


def make_servo(stats: dict):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ServoNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [3 * D.HIST + 1, HID, HID, HID, 1]
            self.W = nn.ParameterList()
            self.b = nn.ParameterList()
            for i in range(4):
                w = torch.empty(6, dims[i], dims[i + 1])
                for j in range(6):
                    nn.init.kaiming_uniform_(w[j].T, a=5 ** 0.5)
                if i == 3:
                    w.zero_()                                    # 头部零初始化
                self.W.append(nn.Parameter(w))
                self.b.append(nn.Parameter(torch.zeros(6, dims[i + 1])))
            self.g = nn.Parameter(torch.full((6,), G_INIT))
            self.register_buffer("s_eps", torch.tensor(stats["s_eps"], dtype=torch.float32))
            self.register_buffer("s_v", torch.tensor(stats["s_v"], dtype=torch.float32))

        def step(self, qh, uh, mh, u_next):
            """qh/uh (B,6,H+1) = 帧 k−H…k；mh (B,H+1)；u_next (B,6) → q̂_{k+1} (B,6)。"""
            m = mh[:, None, 1:]
            eps = (uh[..., 1:] - qh[..., 1:]) / self.s_eps[None, :, None] * m
            v = (qh[..., 1:] - qh[..., :-1]) / self.s_v[None, :, None] * m * mh[:, None, :-1]
            e_next = ((u_next - qh[..., -1]) / self.s_eps)[..., None]
            h = torch.cat([eps, v, m.expand_as(eps), e_next], -1)          # (B,6,61)
            for i in range(4):
                h = torch.einsum("bji,jio->bjo", h, self.W[i]) + self.b[i]
                if i < 3:
                    h = F.softsign(h)
            return qh[..., -1] + self.g * (u_next - qh[..., -1]) + self.s_v * h[..., 0]

        def rollout(self, qh, uh, mh, u_future):
            """从历史出发逐帧喂 `u_future` (B,K,6) → q̂ (B,K,6)。"""
            out = []
            for k in range(u_future.shape[1]):
                un = u_future[:, k]
                qn = self.step(qh, uh, mh, un)
                out.append(qn)
                qh = torch.cat([qh[..., 1:], qn[..., None]], -1)
                uh = torch.cat([uh[..., 1:], un[..., None]], -1)
                mh = torch.cat([mh[:, 1:], torch.ones_like(mh[:, :1])], -1)
            return torch.stack(out, 1)

    return ServoNet()


def _t(net, x):
    import torch
    return torch.tensor(np.asarray(x), dtype=torch.float32, device=net.s_eps.device)


def history_measured(ep, t0: int):
    """1b：帧 `t0−H … t0` 的实测编码器（无效 = u_0）、对应指令、有效位 → ((6,H+1),(6,H+1),(H+1,))。"""
    lo = t0 - D.HIST
    k = np.arange(lo, t0 + 1)
    u = W.span(ep.cmd[:, :6], lo, t0, ep.cmd[0, :6])
    q = np.repeat(ep.cmd[0:1, :6], len(k), 0).astype(np.float64)
    ok = k >= 0
    ok[ok] = ep.q_valid[k[ok]]
    q[ok] = ep.q[k[ok]]
    return q.T, u.T, ok.astype(np.float64)


def history_from_traj(ep, traj, t0: int):
    """1a：帧 `t0−H … t0` 取自开环轨迹 `traj`（帧 0 = u_0）；`< 0` 帧填 u_0、掩码 0。"""
    lo = t0 - D.HIST
    k = np.arange(lo, t0 + 1)
    u = W.span(ep.cmd[:, :6], lo, t0, ep.cmd[0, :6])
    q = W.span(traj, lo, t0, ep.cmd[0, :6])
    return q.T, u.T, (k >= 0).astype(np.float64)


def open_loop_traj(net, ep, upto: int) -> np.ndarray:
    """1a：从静止（q̂_0 = u_0）起、只用真实指令 `u_1 … u_upto` 的开环滚动 → (upto+1,6)。无梯度。"""
    import torch
    u0 = ep.cmd[0, :6]
    if upto <= 0:
        return u0[None].astype(np.float64)
    with torch.no_grad():
        qh = _t(net, np.repeat(u0[:, None], D.HIST + 1, 1)[None])
        mh = torch.zeros(1, D.HIST + 1, device=qh.device)
        mh[:, -1] = 1.0
        q = net.rollout(qh, qh.clone(), mh, _t(net, ep.cmd[1:upto + 1, :6][None]))[0]
    return np.concatenate([u0[None], q.cpu().numpy().astype(np.float64)], 0)


def future_commands(ep, w: W.Window, K: int) -> np.ndarray:
    """窗口内第 `t0+1 … t0+K` 帧的可见指令（超出 t1 钳位）→ (K,6)。"""
    return W.visible(ep.cmd[:, :6], w.t1, w.t0 + K)[w.t0 + 1:]


def window_inputs(net, ep, windows, mode: str, traj=None):
    """批量窗口的 (历史 q, 历史 u, 有效位, 未来指令, K)。mode 'a' 用开环轨迹，'b' 用实测。"""
    K = max(w.t1 - w.t0 for w in windows) + D.ALIGN_L + ROLL_PAD
    Q, U, M, F = [], [], [], []
    for w in windows:
        q, u, m = history_from_traj(ep, traj, w.t0) if mode == "a" else history_measured(ep, w.t0)
        Q.append(q); U.append(u); M.append(m); F.append(future_commands(ep, w, K))
    return (_t(net, np.stack(Q)), _t(net, np.stack(U)), _t(net, np.stack(M)),
            _t(net, np.stack(F)), K)
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_layered.py -q`
Expected: `7 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_layered.py sim/q1/test_q1_layered.py
git commit -m "阶段一 Q1：分层伺服关节层（逐关节 MLP；1a 开环 / 1b 实测历史；末端钳位）"
```

---

### Task 7: 端到端 TCN / Transformer（2a/2b、3a/3b）

**Files:**
- Create: `sim/q1/q1_e2e.py`
- Test: `sim/q1/test_q1_e2e.py`

**Interfaces:**
- Consumes: `q1_windows.past_block / visible / Window`
- Produces:
  - 常量：`K_OUT=102, L_SEQ=122, CH_A=13, CH_B=24`
  - `norm_stats(eps) -> dict`：键 `u, ideal, q, w` 为 `(mean, sd)`，另有 `s_eps (6,)`、`s_gap (3,)`
  - `window_input(ep, w, st, measured) -> float32 (122, 13|24)`
  - `make_tcn(in_ch)`：参数 66 441（a）/ 69 961（b）
  - `make_transformer(in_ch)`：参数 76 361（a）/ 77 065（b）
  - `decode(out, ep, windows, st) -> (每窗 ê, 每窗 q̂)`（torch）；`q̂` 的基线是**钳位后的可见指令**

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_e2e.py`

```python
import copy

import numpy as np
import pytest
import torch

import q1_data as D
import q1_e2e as E
import q1_testing as T
import q1_windows as W

FAMS = [("tcn", E.make_tcn), ("tr", E.make_transformer)]


def _jit(m, s=0.05):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * s)
    return m


def _pred(m, e, st, meas):
    ws = W.chunk_windows(e)
    X = torch.tensor(np.stack([E.window_input(e, w, st, meas) for w in ws]))
    with torch.no_grad():
        Ee, Qq = E.decode(m(X), e, ws, st)
    return [x.numpy() for x in Ee], [x.numpy() for x in Qq]


def test_input_shapes_and_future_has_no_measured_values():
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    w = W.chunk_windows(eps[2])[1]
    xa = E.window_input(eps[2], w, st, False)
    xb = E.window_input(eps[2], w, st, True)
    assert xa.shape == (E.L_SEQ, E.CH_A) and xb.shape == (E.L_SEQ, E.CH_B)
    assert (xb[D.HIST:, E.CH_A:] == 0).all()


@pytest.mark.parametrize("name,mk", FAMS)
def test_zero_init_is_zero_prediction(name, mk):
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    e = eps[2]
    torch.manual_seed(0)
    Ee, Qq = _pred(mk(E.CH_A), e, st, False)
    w = W.chunk_windows(e)[1]
    assert all((x == 0).all() for x in Ee)
    u_vis = W.visible(e.cmd[:, :6], w.t1, w.s1 + D.ALIGN_L)[np.arange(w.t0, w.s1 + 1) + D.ALIGN_L]
    assert np.allclose(Qq[1], u_vis, atol=1e-6)                 # q̂ 基线 = 钳位后的可见指令
    assert np.allclose(Qq[1][-3:], e.cmd[w.t1, :6], atol=1e-6)  # 末 3 帧落在 t1 之后 ⇒ 等于 u_{t1}


@pytest.mark.parametrize("name,mk", FAMS)
@pytest.mark.parametrize("meas", [False, True])
def test_no_leakage(name, mk, meas):
    eps = [T.synthetic_episode(s) for s in range(3)]
    st = E.norm_stats(eps[:2])
    e = eps[2]
    w = W.chunk_windows(e)[2]
    torch.manual_seed(0)
    m = _jit(mk(E.CH_B if meas else E.CH_A))
    bE, bQ = (x[2] for x in _pred(m, e, st, meas))
    e1 = copy.copy(e)
    c = e.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    i_ = e.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = (x[2] for x in _pred(m, e1, st, meas))
    assert np.array_equal(E1, bE) and np.array_equal(Q1, bQ)     # ê 与 q̂ 都不读下一 chunk
    e2 = copy.copy(e)
    q = e.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    ww = e.w.copy(); ww[w.t0 + 1:] += 0.5; e2.w = ww
    E2, Q2 = (x[2] for x in _pred(m, e2, st, meas))
    assert np.array_equal(E2, bE) and np.array_equal(Q2, bQ)
    e3 = copy.copy(e); e3.q = e.q + 0.01; e3.w = e.w + 0.1
    assert (not np.array_equal(_pred(m, e3, st, meas)[0][2], bE)) == meas


@pytest.mark.parametrize("name,mk", FAMS)
def test_causal_over_sequence(name, mk):
    torch.manual_seed(0)
    m = _jit(mk(E.CH_A))
    x = torch.randn(1, E.L_SEQ, E.CH_A)
    x2 = x.clone(); x2[:, 60:] += 1.0
    with torch.no_grad():
        assert torch.equal(m(x)[:, :60], m(x2)[:, :60])


def test_param_counts_are_comparable():
    for ch in (E.CH_A, E.CH_B):
        torch.manual_seed(0)
        a = sum(p.numel() for p in E.make_tcn(ch).parameters())
        b = sum(p.numel() for p in E.make_transformer(ch).parameters())
        assert abs(b / a - 1) <= 0.25, (a, b)


def test_norm_stats_use_only_given_episodes():
    eps = [T.synthetic_episode(s) for s in range(3)]
    assert not np.allclose(E.norm_stats(eps[:2])["u"][0], E.norm_stats(eps)["u"][0])
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_e2e.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_e2e'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_e2e.py`

```python
"""端到端直接多步模型（2a/2b TCN、3a/3b Transformer）—— `DESIGN_G_PI_V2.md` §2.4。

在起点一次输出窗口内全部帧，窗口内不递推。序列 = 过去段 20 帧（`t0−19 … t0`）+ 未来段
`K_OUT` 帧（`t0+1 … t0+K_OUT`）。未来段只含当前 chunk 的指令与理想陀螺（超出 t1 钳位）；
b 版过去段另含实测编码器/陀螺（只到 t0）。两种网络都**因果**（只看到本帧及之前的输入）。
输出头零初始化：`q̂_k = u_k + s_ε·o_q`，`ê_{k−3} = s_e·o_e` ⇒ 起点 = 零预测（G0 分母）。
"""
from __future__ import annotations
import numpy as np
import q1_data as D
import q1_windows as W

K_OUT = D.CHUNK_MAX + D.ALIGN_L - 1          # 102：覆盖 t0+1 … t1+3
L_SEQ = D.HIST + K_OUT
CH_A = 13                                     # u7 ideal3 seg inchunk m_cmd
CH_B = CH_A + 11                              # + q6 w3 m_q m_w
SD_FLOOR = 1e-6


def norm_stats(eps) -> dict:
    """输入归一化（只用训练会话）：u、ideal、q（有效帧）、w（有效帧）；以及输出尺度。"""
    U = np.concatenate([e.cmd[:e.T] for e in eps]); I = np.concatenate([e.ideal[:e.T] for e in eps])
    Q = np.concatenate([e.q[e.q_valid] for e in eps]); Wm = np.concatenate([e.w[e.w_valid] for e in eps])
    G = np.concatenate([e.gap for e in eps])
    E = np.concatenate([e.cmd[np.where(e.q_valid)[0], :6] - e.q[e.q_valid] for e in eps])
    f = lambda x: (x.mean(0), np.maximum(x.std(0), SD_FLOOR))
    st = {k: f(v) for k, v in (("u", U), ("ideal", I), ("q", Q), ("w", Wm))}
    st["s_eps"] = np.maximum(E.std(0), SD_FLOOR)
    st["s_gap"] = np.maximum(G.std(0), SD_FLOOR)
    return st


def window_input(ep, w: W.Window, st: dict, measured: bool) -> np.ndarray:
    """→ (L_SEQ, CH_A|CH_B) float32。"""
    nz = lambda x, k: (x - st[k][0]) / st[k][1]
    pb = W.past_block(ep, w.t0, measured)
    fu = W.visible(ep.cmd, w.t1, w.t0 + K_OUT)[w.t0 + 1:]
    fi = W.visible(ep.ideal, w.t1, w.t0 + K_OUT)[w.t0 + 1:]
    kk = np.arange(w.t0 + 1, w.t0 + K_OUT + 1)
    past = [nz(pb["u"], "u"), nz(pb["ideal"], "ideal"), np.zeros((D.HIST, 1)),
            np.zeros((D.HIST, 1)), pb["m_cmd"][:, None]]
    fut = [nz(fu, "u"), nz(fi, "ideal"), np.ones((K_OUT, 1)),
           (kk <= w.t1).astype(np.float64)[:, None], np.ones((K_OUT, 1))]
    if measured:
        past += [nz(pb["q"], "q") * pb["m_q"][:, None], nz(pb["w"], "w") * pb["m_w"][:, None],
                 pb["m_q"][:, None], pb["m_w"][:, None]]
        fut += [np.zeros((K_OUT, 11))]
    return np.concatenate([np.concatenate(past, 1), np.concatenate(fut, 1)], 0).astype(np.float32)


def make_tcn(in_ch: int):
    import torch.nn as nn

    class TCN(nn.Module):
        """因果空洞卷积 4 层 k=5 d=1/2/4/8 64ch（与 `F_abs` 主干同构）→ 1×1 → 9。"""

        def __init__(self):
            super().__init__()
            layers, c = [], in_ch
            for d in (1, 2, 4, 8):
                layers += [nn.ConstantPad1d(((5 - 1) * d, 0), 0.0), nn.Conv1d(c, 64, 5, dilation=d), nn.GELU()]
                c = 64
            self.body = nn.Sequential(*layers)
            self.head = nn.Conv1d(64, 9, 1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

        def forward(self, x):                          # (B,L,C) → (B,L,9)
            return self.head(self.body(x.transpose(1, 2))).transpose(1, 2)

    return TCN()


def make_transformer(in_ch: int):
    import torch, torch.nn as nn

    class TinyTransformer(nn.Module):
        """因果自注意力 2 层 d=64 4 头 FF128（规模与 TCN 相近）→ 9。"""

        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(in_ch, 64)
            self.pos = nn.Parameter(torch.zeros(1, L_SEQ, 64)); nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(64)
            self.head = nn.Linear(64, 9)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(L_SEQ))

        def forward(self, x):
            L = x.shape[1]
            h = self.inp(x) + self.pos[:, :L]
            h = self.enc(h, mask=self.causal[:L, :L], is_causal=True)
            return self.head(self.norm(h))

    return TinyTransformer()


def decode(out, ep, windows, st: dict):
    """网络输出 (B,L_SEQ,9) → 每窗 (ê (n_scored,3) torch, q̂ 帧 t0+3…s1+3 (n_scored,6) torch)。

    q̂ 的基线是**钳位后的可见指令**（帧 > t1 用 u_{t1}）—— q̂ 进联合损失，同样不得读下一 chunk。
    """
    import torch
    s_e = torch.tensor(st["s_gap"], dtype=out.dtype, device=out.device)
    s_q = torch.tensor(st["s_eps"], dtype=out.dtype, device=out.device)
    E, Q = [], []
    for b, w in enumerate(windows):
        i = np.arange(w.n_scored) + 2 + D.HIST          # 帧 t+3 在序列中的位置
        E.append(out[b, i, 6:9] * s_e)
        uv = W.visible(ep.cmd[:, :6], w.t1, w.s1 + D.ALIGN_L)[np.arange(w.t0, w.s1 + 1) + D.ALIGN_L]
        u = torch.tensor(uv, dtype=out.dtype, device=out.device)
        Q.append(u + out[b, i, :6] * s_q)
    return E, Q
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_e2e.py -q`
Expected: `11 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_e2e.py sim/q1/test_q1_e2e.py
git commit -m "阶段一 Q1：端到端直接多步 TCN / 因果 Transformer（规模相近、零初始化输出头、q̂ 基线钳位）"
```

---

### Task 8: 拼接损失

**Files:**
- Create: `sim/q1/q1_losses.py`
- Test: `sim/q1/test_q1_losses.py`

**Interfaces:**
- Consumes: `q1_windows.kernel / score`；`Q1Episode`
- Produces:
  - 常量：`W_BAND=0.5, Q_BAND=(3,9)`
  - `q_kernel()`
  - `loss_norms(eps) -> {"c_w", "s_eps", "s_qb"}`
  - `_bp(x, K)`（torch）
  - `omega_loss(ep, ehat (n,3), norms)`
  - `joint_loss(ep, qhat (n,6), norms)`：带内项只取 ±20 帧支撑全有效的帧

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_losses.py`

```python
import numpy as np
import torch

import q1_data as D
import q1_losses as LS
import q1_testing as T
import q1_windows as W


def test_torch_band_matches_numpy_P():
    e = T.synthetic_episode(0)
    r = np.random.default_rng(0).normal(size=(e.n, 3))
    b = LS._bp(torch.tensor(r - r.mean(0)), W.kernel()).numpy()
    assert np.abs(W.P(r, W.kernel()) - b).max() < 1e-12


def test_omega_loss_at_zero_is_relative_wall_energy():
    eps = [T.synthetic_episode(s) for s in range(3)]
    nm = LS.loss_norms(eps)
    K = W.kernel()
    e0 = [W.score(x.gap, 0 * x.gap, K) for x in eps]
    z = LS.omega_loss(eps[0], torch.zeros(eps[0].n, 3, dtype=torch.float64), nm).item()
    assert np.isclose(z, e0[0][0] / e0[0][2] / nm["c_w"], rtol=1e-12)
    assert np.isclose(np.mean([a / c for a, _, c in e0]) / nm["c_w"], 1.0)


def test_joint_loss_ignores_invalid_frames():
    e = T.synthetic_episode(0)
    e.q_valid = e.q_valid.copy(); e.q_valid[300] = False
    nm = LS.loss_norms([e])
    q = torch.tensor(e.cmd[D.ALIGN_L:D.ALIGN_L + e.n, :6])
    base = LS.joint_loss(e, q.clone(), nm).item()
    q2 = q.clone(); q2[300 - D.ALIGN_L] += 10.0
    assert LS.joint_loss(e, q2, nm).item() == base


def test_stored_value_at_invalid_frame_changes_nothing():
    """无效帧里存的实测值不得经带通进入关节损失或其归一化尺度。"""
    import copy
    a = T.synthetic_episode(0)
    a.q_valid = a.q_valid.copy(); a.q_valid[300] = False
    a.q = a.q.copy(); a.q[300] = 0.0
    b = copy.copy(a); b.q = a.q.copy(); b.q[300] = 5.0
    na, nb = LS.loss_norms([a]), LS.loss_norms([b])
    assert all(np.array_equal(na[k], nb[k]) for k in na)
    qh = torch.tensor(a.cmd[D.ALIGN_L:D.ALIGN_L + a.n, :6])
    assert LS.joint_loss(a, qh, na).item() == LS.joint_loss(b, qh, na).item()


def test_losses_are_differentiable():
    e = T.synthetic_episode(1)
    nm = LS.loss_norms([e])
    eh = torch.zeros(e.n, 3, dtype=torch.float64, requires_grad=True)
    qh = torch.tensor(e.cmd[D.ALIGN_L:D.ALIGN_L + e.n, :6], requires_grad=True)
    (LS.omega_loss(e, eh, nm) + LS.joint_loss(e, qh, nm)).backward()
    assert eh.grad.abs().sum() > 0 and qh.grad.abs().sum() > 0
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_losses.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_losses'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_losses.py`

```python
"""拼接后的训练损失（torch）。与评分口径一致：逐集去均值 → bandpass41 → 丢首尾 20。"""
from __future__ import annotations
import numpy as np
import q1_data as D
import q1_windows as W

W_BAND = 0.5                  # 关节损失的带内权重（同 G-2 `r_q`），默认值
Q_BAND = (3.0, 9.0)


def q_kernel() -> np.ndarray:
    from scipy.signal import firwin
    return firwin(41, list(Q_BAND), fs=D.FS, pass_zero=False).astype(np.float32)


def loss_norms(eps) -> dict:
    """损失归一化（只用训练会话）：ω 项除以零预测误差的均值；关节项按自身尺度。"""
    K = W.kernel(); Kq = q_kernel()
    e0 = [W.score(e.gap, np.zeros_like(e.gap), K) for e in eps]
    c_w = float(np.mean([a / c for a, _, c in e0]))
    E, B = [], []
    for e in eps:
        k = np.where(e.q_valid)[0]
        E.append(e.cmd[k, :6] - e.q[k])
        qv = e.q_valid[D.ALIGN_L:]
        qb = np.stack([np.convolve(e.q[D.ALIGN_L:, j] * qv, Kq, mode="same") for j in range(6)], 1)
        B.append(qb[W.band_clean(qv)])                  # 只用 ±20 帧支撑全有效的帧
    return {"c_w": c_w, "s_eps": np.maximum(np.concatenate(E).std(0), 1e-6),
            "s_qb": np.maximum(np.concatenate(B).std(0), 1e-6)}


def _bp(x, K):
    """(L,C) → 'same' 卷积后丢首尾 EDGE。"""
    import torch
    Kt = torch.as_tensor(K, dtype=x.dtype, device=x.device).view(1, 1, -1).repeat(x.shape[1], 1, 1)
    y = torch.nn.functional.conv1d(x.T[None], Kt, padding=len(K) // 2, groups=x.shape[1])[0].T
    return y[D.EDGE:-D.EDGE]


def omega_loss(ep, ehat, norms):
    """`ehat` (n,3) torch（已拼接）→ ‖P(gap−ê)‖²/(3T_e)/c_w。"""
    import torch
    g = torch.as_tensor(ep.gap, dtype=ehat.dtype, device=ehat.device)
    r = g - ehat
    r = r - r.mean(0, keepdim=True)
    v = _bp(r, W.kernel())
    return (v ** 2).sum() / v.numel() / norms["c_w"]


def joint_loss(ep, qhat, norms):
    """`qhat` (n,6) torch = 帧 3 … T−1（已拼接）→ 0.5·宽带 + 0.5·带内（3–9 Hz），各按自身尺度。

    无效帧不进宽带项；带内项只取 ±20 帧支撑全有效的帧（无效帧的值不经带通污染邻帧）。
    """
    import torch
    q = torch.as_tensor(ep.q[D.ALIGN_L:D.ALIGN_L + ep.n], dtype=qhat.dtype, device=qhat.device)
    qv = ep.q_valid[D.ALIGN_L:D.ALIGN_L + ep.n]
    m = torch.as_tensor(qv, dtype=qhat.dtype, device=qhat.device)
    cm = torch.as_tensor(W.band_clean(qv)[D.EDGE:-D.EDGE], dtype=qhat.dtype, device=qhat.device)
    d = qhat - q
    s_e = torch.as_tensor(norms["s_eps"], dtype=qhat.dtype, device=qhat.device)
    s_b = torch.as_tensor(norms["s_qb"], dtype=qhat.dtype, device=qhat.device)
    broad = (((d / s_e) ** 2) * m[:, None]).sum() / (m.sum() * 6)
    bd = _bp(d * m[:, None], q_kernel()) / s_b
    band = ((bd ** 2) * cm[:, None]).sum() / (cm.sum() * 6)
    return (1 - W_BAND) * broad + W_BAND * band
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_losses.py -q`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_losses.py sim/q1/test_q1_losses.py
git commit -m "阶段一 Q1：拼接后的 ω 损失与关节损失（与 P(·) 计分同口径）"
```

---

### Task 9: 训练与预测

**Files:**
- Create: `sim/q1/q1_train.py`
- Test: `sim/q1/test_q1_train.py`

**Interfaces:**
- Consumes: Task 1–8 的全部接口
- Produces:
  - 常量：
    - `MODELS = {"1a": ("layered","a"), …, "3b": ("tr","b")}`
    - `CONFIG = {"steps": 3000, "lr": 1e-3, "clip": 1.0, "threads": 1, "device": "cpu"}`
    - `CODE_FILES`：本任务为 9 个模块；Task 10 加入 `q1_run.py`（它此时还不存在）
  - 指纹：`code_sha() -> str`；`config_hash(steps) -> str`（16 位）
  - 折与建模：
    - `split_fold(eps, held) -> (tr, te)`
    - `rebuild(model_id, st)`
    - `build(model_id, tr_eps) -> (model, st)`
  - 训练与预测：
    - `forward_windows(...)`
    - `train_one(model_id, tr_eps, seed, steps=None, log=None) -> {"model", "stats", "norms", "loss", "sec"}`
    - `predict_windows(model_id, model, st, ep, windows, R=None, Wrs=None, C=None) -> list[(n_scored,3)]`（1b 先 `check_window_support`）
    - `predict_episode(...) -> (n,3)`
  - `state_sha256(model) -> str`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_train.py`

```python
import copy

import numpy as np
import pytest

import q1_sensor as S
import q1_testing as T
import q1_train as TR
import q1_windows as W


@pytest.fixture(scope="module")
def syn():
    tr = [T.synthetic_episode(s, sess=f"s{s % 4}/A") for s in range(8)]
    te = T.synthetic_episode(99, sess="hold/A")
    R = np.eye(3)
    Wrs = S.fit_rs(tr, R)
    return tr, te, R, Wrs, S.fit_cmap(tr, R, Wrs)


@pytest.mark.parametrize("mid", list(TR.MODELS))
def test_train_is_bit_reproducible_and_predicts(mid, syn):
    import torch
    tr, te, R, Wrs, C = syn
    a = TR.train_one(mid, tr, seed=0, steps=3)
    assert torch.get_num_threads() == TR.CONFIG["threads"]
    b = TR.train_one(mid, tr, seed=0, steps=3)
    assert TR.state_sha256(a["model"]) == TR.state_sha256(b["model"]) and a["loss"] == b["loss"]
    c = TR.train_one(mid, tr, seed=1, steps=3)
    assert TR.state_sha256(c["model"]) != TR.state_sha256(a["model"])
    eh = TR.predict_episode(mid, a["model"], a["stats"], te, W.chunk_windows(te), R, Wrs, C)
    assert eh.shape == (te.n, 3) and np.isfinite(eh).all()


@pytest.mark.parametrize("mid", ["1b", "2b", "3b"])
def test_prediction_ignores_measurements_after_origin(mid, syn):
    tr, te, R, Wrs, C = syn
    out = TR.train_one(mid, tr, seed=0, steps=2)
    ws = W.chunk_windows(te)
    base = TR.predict_episode(mid, out["model"], out["stats"], te, ws, R, Wrs, C)
    w = ws[3]
    e2 = copy.copy(te)
    q = te.q.copy(); q[w.t0 + 1:] += 0.05; e2.q = q
    ww = te.w.copy(); ww[w.t0 + 1:] += 0.5; e2.w = ww
    p2 = TR.predict_episode(mid, out["model"], out["stats"], e2, ws, R, Wrs, C)
    assert np.array_equal(p2[w.t0:w.s1 + 1], base[w.t0:w.s1 + 1])


@pytest.mark.parametrize("mid", list(TR.MODELS))
def test_forward_windows_ignore_commands_after_t1(mid, syn):
    """训练用的前向（进联合损失的 q̂ 与 ê）也不得读下一 chunk 的指令/理想陀螺。"""
    import torch
    tr, te, R, Wrs, C = syn
    torch.manual_seed(0)
    model, st = TR.build(mid, tr)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    ws = W.chunk_windows(te)
    w = ws[2]

    def fw(ep):
        with torch.no_grad():
            Ee, Qq, _ = TR.forward_windows(mid, model, st, ep, ws)
        return (None if Ee is None else Ee[2].numpy()), Qq[2].numpy()

    bE, bQ = fw(te)
    e1 = copy.copy(te)
    c = te.cmd.copy(); c[w.t1 + 1:] += 0.05; e1.cmd = c
    i_ = te.ideal.copy(); i_[w.t1 + 1:] += 0.5; e1.ideal = i_
    E1, Q1 = fw(e1)
    assert np.array_equal(Q1, bQ) and (bE is None or np.array_equal(E1, bE))


def test_split_fold_excludes_held_out_session():
    eps = [T.synthetic_episode(s, sess=f"s{s % 3}/A") for s in range(6)]
    tr, te = TR.split_fold(eps, "s1/A")
    assert {e.sess for e in te} == {"s1/A"} and "s1/A" not in {e.sess for e in tr}
    with pytest.raises(ValueError):
        TR.split_fold(eps, "nope/A")


def test_config_hash_tracks_steps():
    assert TR.config_hash(3000) != TR.config_hash(2999)
    assert TR.config_hash(3000) == TR.config_hash(3000)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_train.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_train'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_train.py`

```python
"""阶段一的训练与预测 —— 6 个模型 × 留一会话 × 训练种子。

每步均匀抽一个训练集，取它的全部主窗口（chunk 开头起点），拼接后算损失；
**固定步数、取最后一步权重**；留出会话只在最后预测一次，不参与任何选择。
CPU 单线程 + 确定性算法 ⇒ 同种子逐位复现。
"""
from __future__ import annotations
import hashlib
import json
import time

import numpy as np

import q1_data as D
import q1_e2e as E
import q1_layered as LM
import q1_losses as LS
import q1_sensor as S
import q1_windows as W

MODELS = {"1a": ("layered", "a"), "1b": ("layered", "b"),
          "2a": ("tcn", "a"), "2b": ("tcn", "b"),
          "3a": ("tr", "a"), "3b": ("tr", "b")}
CONFIG = {"steps": 3000, "lr": 1e-3, "clip": 1.0, "threads": 1, "device": "cpu"}
CODE_FILES = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_layered.py", "q1_e2e.py",
              "q1_sensor.py", "q1_train.py", "q1_ar.py", "q1_g2ref.py")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((D.HERE / f).read_bytes())
    return h.hexdigest()


def config_hash(steps) -> str:
    """训练/预测相关的配置 + 代码指纹。改了代码或步数 ⇒ 旧产物作废、需重跑。"""
    blob = json.dumps({"config": {**CONFIG, "steps": steps}, "code": code_sha()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def split_fold(eps, held: str):
    """→ (训练折, 留出会话)。留出会话不存在即报错。"""
    te = [e for e in eps if e.sess == held]
    tr = [e for e in eps if e.sess != held]
    if not te:
        raise ValueError(f"留出会话 {held} 不存在")
    return tr, te


def _determinism(seed: int):
    import torch
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def rebuild(model_id: str, st: dict):
    """按统计量重建网络结构（载入权重前用）。"""
    family, mode = MODELS[model_id]
    if family == "layered":
        return LM.make_servo(st)
    ch = E.CH_B if mode == "b" else E.CH_A
    return E.make_tcn(ch) if family == "tcn" else E.make_transformer(ch)


def build(model_id: str, tr_eps):
    """→ (模型, 统计量)。统计量只用训练折。调用前须已 `_determinism(seed)`。"""
    st = LM.joint_stats(tr_eps) if MODELS[model_id][0] == "layered" else E.norm_stats(tr_eps)
    return rebuild(model_id, st), st


def forward_windows(model_id, model, st, ep, windows):
    """→ (每窗 ê 或 None, 每窗 q̂（帧 t0+3 … s1+3）, 分层模型的 (历史, 滚动))。"""
    import torch
    family, mode = MODELS[model_id]
    if family == "layered":
        traj = LM.open_loop_traj(model, ep, max(w.t0 for w in windows)) if mode == "a" else None
        Q, U, M, F, _ = LM.window_inputs(model, ep, windows, mode, traj)
        qh = model.rollout(Q, U, M, F)
        return None, [qh[i, np.arange(w.n_scored) + 2] for i, w in enumerate(windows)], (Q, qh)
    X = torch.tensor(np.stack([E.window_input(ep, w, st, mode == "b") for w in windows]))
    Ee, Qq = E.decode(model(X), ep, windows, st)
    return Ee, Qq, None


def train_one(model_id: str, tr_eps, seed: int, steps=None, log=None) -> dict:
    import torch
    steps = CONFIG["steps"] if steps is None else steps
    _determinism(seed)
    model, st = build(model_id, tr_eps)
    norms = LS.loss_norms(tr_eps)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = np.random.default_rng(seed)
    wins = [W.chunk_windows(e) for e in tr_eps]
    hist, t_start = [], time.time()
    for it in range(steps):
        i = int(rng.integers(len(tr_eps)))
        Ee, Qq, _ = forward_windows(model_id, model, st, tr_eps[i], wins[i])
        loss = LS.joint_loss(tr_eps[i], torch.cat(Qq).double(), norms)
        if Ee is not None:
            loss = loss + LS.omega_loss(tr_eps[i], torch.cat(Ee).double(), norms)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
        opt.step()
        hist.append(float(loss.item()))
        if log and (it + 1) % 500 == 0:
            log(f"    {model_id} seed{seed} step {it + 1}/{steps} "
                f"loss {np.mean(hist[-500:]):.5f} {time.time() - t_start:.0f}s")
    return {"model": model, "stats": st, "norms": norms, "loss": hist, "sec": time.time() - t_start}


def predict_windows(model_id, model, st, ep, windows, R=None, Wrs=None, C=None) -> list:
    """→ 每窗 ê `(n_scored,3)`。分层模型需要该折的 `Wrs`（1b 另需 `C`）。"""
    import torch
    family, mode = MODELS[model_id]
    with torch.no_grad():
        Ee, _, extra = forward_windows(model_id, model, st, ep, windows)
    if family != "layered":
        return [e.double().numpy() for e in Ee]
    Q, qh = extra
    Kr = qh.shape[1]
    preds = []
    for i, w in enumerate(windows):
        lo = w.t0 - D.HIST
        q_span = np.concatenate([Q[i].double().numpy().T, qh[i].double().numpy()], 0)
        u_span = W.span(W.visible(ep.cmd, w.t1, w.t0 + Kr), lo, w.t0 + Kr, ep.cmd[0])
        i_span = W.span(W.visible(ep.ideal, w.t1, w.t0 + Kr), lo, w.t0 + Kr, 0.0)
        om = S.omega_span(q_span, u_span, i_span, R, Wrs)
        ts = np.arange(w.t0, w.s1 + 1)
        eh = om[ts + D.ALIGN_L - lo] - ep.ideal[ts]
        if mode == "b":
            S.check_window_support(ep, w.t0)
            x = S.residual_history(ep, w.t0, R, Wrs)
            eh = eh + np.stack([x @ C[h] for h in range(w.n_scored)])
        preds.append(eh)
    return preds


def predict_episode(model_id, model, st, ep, windows, R=None, Wrs=None, C=None) -> np.ndarray:
    return W.stitch(ep.n, windows, predict_windows(model_id, model, st, ep, windows, R, Wrs, C))


def state_sha256(model) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_train.py -q`
Expected: `17 passed`，约 15 s。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_train.py sim/q1/test_q1_train.py
git commit -m "阶段一 Q1：训练（固定步数取末步、单线程确定性）与逐窗预测"
```

---

### Task 10: 命令行与产物

**Files:**
- Create: `sim/q1/q1_run.py`
- Modify: `sim/q1/q1_train.py`（`CODE_FILES` 加入 `q1_run.py`）
- Test: `sim/q1/test_q1_run.py`

**Interfaces:**
- Consumes: `q1_train`；`q1_sensor`；`q1_g2ref`；`q1_ar`；`q1_windows`
- Produces:
  - CLI 子命令（正式入口另有 `--prereg`，缺省 `reports/PREREG_Q1_STAGE1.md`）：
    - `timing [--steps 30]`（不经冻结闸门）
    - `prepare --out`
    - `list-jobs --out`
    - `job --out MODEL FOLD SEED`
    - `refs --out`
    - `fullfit --out MODEL`
  - 冻结闸门：`PREREG`、`class NotFrozenError(RuntimeError)`、`require_frozen(prereg=None)`。状态行须以「**状态：已冻结」开头，且 `code_sha=<16 位>` 与当前代码一致；`main` 遇到它返回 2。
  - 产物布局（`OUT = outputs/q1_stage1`）：
    - `OUT/sensor/{fold}.npz`：`Wrs`、`C`、`manifest`（训练会话 + 哈希 + 代码指纹）
    - `OUT/jobs/{model}/seed{k}/{fold}.json`：逐集 `num/den/f3/num_aux/lead`，以及 `config_hash`、`code_sha`、`state_sha256`、`pt_sha256`、`npz_sha256`、`train_sessions`
    - 同目录的 `.pt`（权重 + 统计量）、`.npz`（逐集拼接 ê）
    - `OUT/refs/{g2,g2_whole,ar}/{fold}.json`
    - `OUT/fullfit/{model}.pt`（+ `sensor_all.npz`）
  - 函数：
    - 工具：`tag(fold)`、`sha(a)`、`file_sha(path)`、`atomic_json / atomic_npz / atomic_torch`
    - 计分：`episode_metrics(ep, main, aux, K)`
    - 传感器层：`sensor_file`、`fit_sensor`、`load_sensor`
    - 作业状态：`job_json`、`is_done`
    - 子命令实现：`cmd_*`、`main(argv)`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_run.py`

```python
import json

import numpy as np
import pytest

import q1_data as D
import q1_run as RUN
import q1_testing as T
import q1_train as TR


@pytest.fixture()
def syn_data(monkeypatch):
    eps = [T.synthetic_episode(i, sess=f"{s}/{a}") for i, (s, a) in
           enumerate([x for x in D.SESSIONS for _ in range(2)])]
    monkeypatch.setattr(D, "load_bare", lambda *a, **k: eps)
    monkeypatch.setattr(D, "calibration", lambda: (np.eye(3), np.zeros(3)))
    return eps


def _prereg(tmp_path, status="**状态：已冻结（测试）**", sha=None):
    p = tmp_path / "PREREG.md"
    p.write_text(f"# 预注册\n\n{status}\n代码 code_sha={sha or TR.code_sha()[:16]}\n")
    return p


@pytest.mark.parametrize("case", ["missing", "draft", "stale_sha"])
def test_formal_commands_refuse_unless_frozen(tmp_path, monkeypatch, case):
    def boom(*a, **k):
        raise AssertionError("闸门必须在读数据之前")
    monkeypatch.setattr(D, "load_bare", boom)
    pre = {"missing": tmp_path / "none.md",
           "draft": _prereg(tmp_path, status="**状态：草稿（测试），待用户冻结。**"),
           "stale_sha": _prereg(tmp_path, sha="0" * 16)}[case]
    calls = [lambda: RUN.cmd_prepare(tmp_path, prereg=pre),
             lambda: RUN.cmd_job(tmp_path, "2a", "ab2/A", 0, 2, prereg=pre),
             lambda: RUN.cmd_refs(tmp_path, prereg=pre),
             lambda: RUN.cmd_fullfit(tmp_path, "2a", 2, prereg=pre)]
    for call in calls:
        with pytest.raises(RUN.NotFrozenError):
            call()
    assert RUN.main(["prepare", "--out", str(tmp_path), "--prereg", str(pre)]) == 2


def test_timing_needs_no_prereg(syn_data, monkeypatch, capsys):
    monkeypatch.setattr(RUN, "PREREG", D.ROOT / "reports" / "does_not_exist.md")
    RUN.cmd_timing(1)
    assert "s/步" in capsys.readouterr().out


def test_code_sha_covers_the_run_entry():
    """`q1_run.py` 决定写什么进产物（逐集计分、哈希）⇒ 它也必须在代码指纹里。"""
    assert "q1_run.py" in TR.CODE_FILES


def test_list_jobs_skips_done_and_requeues_stale(tmp_path, capsys):
    ch = TR.config_hash(5)
    RUN.atomic_json(RUN.job_json(tmp_path, "1a", "ab5/A", 0), {"config_hash": ch})
    RUN.atomic_json(RUN.job_json(tmp_path, "1a", "ab6/A", 0), {"config_hash": "stale"})
    RUN.cmd_list_jobs(tmp_path, 5)
    lines = capsys.readouterr().out.split("\n")
    assert "1a ab5/A 0" not in lines and "1a ab6/A 0" in lines
    assert len([x for x in lines if x]) == 6 * 5 * 3 - 1


def test_load_sensor_rejects_leaky_manifest(tmp_path):
    man = {"fold": "ab5/A", "train_sessions": ["ab5/A", "ab6/A"], "Wrs_sha256": RUN.sha(np.zeros(2)),
           "C_sha256": RUN.sha(np.zeros(2)), "code": TR.code_sha()}
    RUN.atomic_npz(RUN.sensor_file(tmp_path, "ab5/A"), Wrs=np.zeros(2), C=np.zeros(2),
                   manifest=json.dumps(man))
    with pytest.raises(RuntimeError):
        RUN.load_sensor(tmp_path, "ab5/A")


@pytest.mark.parametrize("model", ["1b", "2a"])
def test_prepare_and_job_end_to_end_on_synthetic(tmp_path, syn_data, model):
    pre = _prereg(tmp_path)
    RUN.cmd_prepare(tmp_path, prereg=pre)
    Wrs, C, man = RUN.load_sensor(tmp_path, "ab2/A")
    assert "ab2/A" not in man["train_sessions"] and len(man["train_sessions"]) == 4
    RUN.cmd_job(tmp_path, model, "ab2/A", 0, steps=2, prereg=pre)
    base = RUN.job_json(tmp_path, model, "ab2/A", 0)
    j = json.loads(base.read_text())
    assert {r["sess"] for r in j["episodes"]} == {"ab2/A"} and "ab2/A" not in j["train_sessions"]
    assert j["config_hash"] == TR.config_hash(2) and j["code_sha"] == TR.code_sha()
    assert j["pt_sha256"] == RUN.file_sha(base.with_suffix(".pt"))
    assert j["npz_sha256"] == RUN.file_sha(base.with_suffix(".npz"))
    r = j["episodes"][0]
    assert 0 <= r["num"] and r["den"] > 0 and set(r["lead"]) == {"3-6", "7-12", "13-27", "28-52", "53-102"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_run.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_run'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_run.py`

```python
"""阶段一命令行入口。

⚠ `prepare` / `job` / `refs` / `fullfit` 是**正式运行**：入口先查 `reports/PREREG_Q1_STAGE1.md`
—— 状态行须以「**状态：已冻结」开头，且记录的 `code_sha=` 与当前代码一致，否则拒绝运行。
`timing` 不经过这道闸门：只测每步耗时（在 ab5/A 折的训练会话上训几十步，不预测留出、不存权重），
用来填预注册的预算。

    python -u sim/q1/q1_run.py timing --steps 30
    python -u sim/q1/q1_run.py prepare   --out outputs/q1_stage1
    python -u sim/q1/q1_run.py list-jobs --out outputs/q1_stage1
    python -u sim/q1/q1_run.py job       --out outputs/q1_stage1 1a ab5/A 0
    python -u sim/q1/q1_run.py refs      --out outputs/q1_stage1
    python -u sim/q1/q1_run.py fullfit   --out outputs/q1_stage1 2b
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_train as TR          # noqa: E402
import q1_windows as W         # noqa: E402

SEEDS = (0, 1, 2)
FOLDS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
TIMING_FOLD = "ab5/A"
PREREG = D.ROOT / "reports/PREREG_Q1_STAGE1.md"


class NotFrozenError(RuntimeError):
    """预注册不存在、未冻结，或冻结时的代码指纹与当前代码不符。"""


def require_frozen(prereg=None) -> None:
    p = Path(prereg) if prereg is not None else PREREG
    if not p.exists():
        raise NotFrozenError(f"拒绝运行：{p} 不存在 —— 正式运行须先冻结预注册")
    text = p.read_text()
    if not re.search(r"^\*\*状态：已冻结", text, re.M):
        raise NotFrozenError(f"拒绝运行：{p.name} 尚未冻结（状态行须以「**状态：已冻结」开头）")
    m = re.search(r"code_sha=([0-9a-f]{16})", text)
    if not m or m.group(1) != TR.code_sha()[:16]:
        raise NotFrozenError("拒绝运行：预注册记录的 code_sha 与当前代码不符 —— 代码改动后须重新冻结")


def tag(fold: str) -> str:
    return fold.replace("/", "_")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tmp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def atomic_json(path: Path, obj) -> None:
    t = _tmp(path)
    t.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    os.replace(t, path)


def atomic_npz(path: Path, **arrays) -> None:
    t = _tmp(path)
    with open(t, "wb") as f:
        np.savez(f, **arrays)
    os.replace(t, path)


def atomic_torch(path: Path, obj) -> None:
    import torch
    t = _tmp(path)
    torch.save(obj, t)
    os.replace(t, path)


def episode_metrics(ep, main, aux, K) -> dict:
    num, den, f3 = W.score(ep.gap, main, K)
    return {"name": ep.name, "sess": ep.sess, "num": num, "den": den, "f3": f3,
            "num_aux": W.score(ep.gap, aux, K)[0],
            "lead": W.lead_score(ep.gap, main, W.chunk_windows(ep), K)}


def sensor_file(out, fold: str) -> Path:
    return Path(out) / "sensor" / f"{tag(fold)}.npz"


def fit_sensor(tr, R) -> tuple[np.ndarray, np.ndarray]:
    import q1_sensor as S
    Wrs = S.fit_rs(tr, R)
    C = S.fit_cmap(tr, R, Wrs)
    if sorted(C) != list(range(len(C))):
        raise RuntimeError("1b 的前瞻项不连续")
    return Wrs, np.stack([C[h] for h in range(len(C))])


def cmd_prepare(out, prereg=None) -> None:
    """每折的传感器层（`r_s` + 1b 前瞻项），只用该折 4 个训练会话。"""
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    for fold in FOLDS:
        tr, _ = TR.split_fold(eps, fold)
        Wrs, C = fit_sensor(tr, R)
        man = {"fold": fold, "train_sessions": sorted({e.sess for e in tr}), "n_train": len(tr),
               "Wrs_sha256": sha(Wrs), "C_sha256": sha(C), "code": TR.code_sha()}
        atomic_npz(sensor_file(out, fold), Wrs=Wrs, C=C, manifest=json.dumps(man))
        print(f"  传感器层 {fold}: 训练 {man['train_sessions']} ({len(tr)} 集)", flush=True)


def load_sensor(out, fold: str):
    z = np.load(sensor_file(out, fold))
    man = json.loads(str(z["manifest"]))
    if man["fold"] != fold or fold in man["train_sessions"]:
        raise RuntimeError(f"{fold}: 留出会话进入了传感器层拟合 / 文件错配")
    if sha(z["Wrs"]) != man["Wrs_sha256"] or sha(z["C"]) != man["C_sha256"]:
        raise RuntimeError(f"{fold}: 传感器层文件与清单哈希不符")
    if man["code"] != TR.code_sha():
        raise RuntimeError(f"{fold}: 传感器层由另一版代码生成 —— 重跑 prepare")
    return z["Wrs"], {h: z["C"][h] for h in range(len(z["C"]))}, man


def job_json(out, model: str, fold: str, seed: int) -> Path:
    return Path(out) / "jobs" / model / f"seed{seed}" / f"{tag(fold)}.json"


def is_done(path: Path, chash: str) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("config_hash") == chash
    except json.JSONDecodeError:
        return False


def cmd_list_jobs(out, steps: int) -> None:
    ch = TR.config_hash(steps)
    for model in TR.MODELS:
        for fold in FOLDS:
            for seed in SEEDS:
                if not is_done(job_json(out, model, fold, seed), ch):
                    print(f"{model} {fold} {seed}")


def predict_main_aux(model_id, res, e, R, Wrs, C):
    ws, mws = W.chunk_windows(e), W.mid_windows(e)
    main = W.stitch(e.n, ws, TR.predict_windows(model_id, res["model"], res["stats"], e, ws, R, Wrs, C))
    aux = W.overlay(main, mws, TR.predict_windows(model_id, res["model"], res["stats"], e, mws, R, Wrs, C))
    return main, aux


def cmd_job(out, model: str, fold: str, seed: int, steps: int, prereg=None) -> None:
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    K = W.kernel()
    tr, te = TR.split_fold(eps, fold)
    Wrs = C = None
    if TR.MODELS[model][0] == "layered":
        Wrs, C, _ = load_sensor(out, fold)
    res = TR.train_one(model, tr, seed, steps, log=lambda m: print(m, flush=True))
    rows, preds = [], {}
    for e in te:                                     # 留出会话：只在最后预测一次
        main, aux = predict_main_aux(model, res, e, R, Wrs, C)
        rows.append(episode_metrics(e, main, aux, K))
        preds[e.name.replace("/", "__")] = main
    base = job_json(out, model, fold, seed)
    atomic_torch(base.with_suffix(".pt"), {"model_id": model, "state_dict": res["model"].state_dict(),
                                           "stats": res["stats"], "norms": res["norms"]})
    atomic_npz(base.with_suffix(".npz"), **preds)
    atomic_json(base, {"model": model, "fold": fold, "seed": seed, "steps": steps,
                       "config_hash": TR.config_hash(steps), "code_sha": TR.code_sha(),
                       "state_sha256": TR.state_sha256(res["model"]),
                       "pt_sha256": file_sha(base.with_suffix(".pt")),
                       "npz_sha256": file_sha(base.with_suffix(".npz")),
                       "train_sessions": sorted({e.sess for e in tr}), "sec": res["sec"],
                       "loss_tail": float(np.mean(res["loss"][-100:])), "episodes": rows})
    print(f"  完成 {model} {fold} seed{seed}  {res['sec']:.0f}s", flush=True)


def cmd_refs(out, prereg=None) -> None:
    """G-2 同口径重预测、旧口径整集 G-2（只对照）、折内 AR 参照。"""
    require_frozen(prereg)
    import q1_ar as AR
    import q1_g2ref as G
    eps = D.load_bare()
    R, b = D.calibration()
    K = W.kernel()
    g2 = G.load_g2()
    ch = TR.code_sha()[:16]
    for fold in FOLDS:
        tr, te = TR.split_fold(eps, fold)
        coef = AR.fit(tr)
        rows = {"g2": [], "g2_whole": [], "ar": []}
        for e in te:
            ws, mws = W.chunk_windows(e), W.mid_windows(e)
            g = G.episode_ehat(g2, e, ws, R, b)
            rows["g2"].append(episode_metrics(e, g, g, K))     # 不读实测 ⇒ 中段起点预测与主窗口逐位相同
            wh = G.whole_episode_ehat(g2, e, R, b)
            rows["g2_whole"].append(episode_metrics(e, wh, wh, K))
            a = W.stitch(e.n, ws, [AR.window_ehat(coef, e, w) for w in ws])
            rows["ar"].append(episode_metrics(
                e, a, W.overlay(a, mws, [AR.window_ehat(coef, e, w) for w in mws]), K))
        for name, r in rows.items():
            atomic_json(Path(out) / "refs" / name / f"{tag(fold)}.json",
                        {"model": name, "fold": fold, "seed": "-", "config_hash": ch,
                         "code_sha": TR.code_sha(),
                         "train_sessions": sorted({e.sess for e in tr}) if name == "ar" else [],
                         "episodes": r})
        print(f"  参照 {fold} 完成", flush=True)


def cmd_timing(steps: int) -> None:
    eps = D.load_bare()
    tr, _ = TR.split_fold(eps, TIMING_FOLD)
    per = {}
    for model in TR.MODELS:
        per[model] = TR.train_one(model, tr, 0, steps)["sec"] / steps
        print(f"  {model}: {per[model]:.3f} s/步", flush=True)
    n = len(FOLDS) * len(SEEDS)
    S_ = TR.CONFIG["steps"]
    core_h = sum(per.values()) * S_ * n / 3600
    print(f"  预计 {S_} 步 × {n} 个（折×种子）× 6 模型 ≈ {core_h:.1f} 核·小时"
          f"（另加每个作业载入数据约 1 分钟）；最长单作业 ≈ {max(per.values()) * S_ / 60:.0f} 分钟")


def cmd_fullfit(out, model: str, steps: int, prereg=None) -> None:
    """**只供 pp2 波形图**：用全部 100 集训练（seed 0），不进任何判据。"""
    require_frozen(prereg)
    eps = D.load_bare()
    R, _ = D.calibration()
    fdir = Path(out) / "fullfit"
    if TR.MODELS[model][0] == "layered":
        sp = fdir / "sensor_all.npz"
        if not sp.exists():
            Wrs, C = fit_sensor(eps, R)
            atomic_npz(sp, Wrs=Wrs, C=C, code=np.asarray(TR.code_sha()))
    res = TR.train_one(model, eps, 0, steps, log=lambda m: print(m, flush=True))
    atomic_torch(fdir / f"{model}.pt", {"model_id": model, "state_dict": res["model"].state_dict(),
                                        "stats": res["stats"], "config_hash": TR.config_hash(steps)})


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "list-jobs", "refs"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
        p.add_argument("--prereg", default=None)
    p = sub.add_parser("job")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p.add_argument("fold", choices=FOLDS)
    p.add_argument("seed", type=int, choices=SEEDS)
    p = sub.add_parser("fullfit")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=TR.CONFIG["steps"])
    p.add_argument("--prereg", default=None)
    p.add_argument("model", choices=list(TR.MODELS))
    p = sub.add_parser("timing")
    p.add_argument("--steps", type=int, default=30)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            cmd_prepare(a.out, a.prereg)
        elif a.cmd == "list-jobs":
            cmd_list_jobs(a.out, a.steps)
        elif a.cmd == "job":
            cmd_job(a.out, a.model, a.fold, a.seed, a.steps, a.prereg)
        elif a.cmd == "refs":
            cmd_refs(a.out, a.prereg)
        elif a.cmd == "fullfit":
            cmd_fullfit(a.out, a.model, a.steps, a.prereg)
        else:
            cmd_timing(a.steps)
    except NotFrozenError as err:
        print(err, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
```

再把 `q1_run.py` 加进 `sim/q1/q1_train.py` 的代码指纹（它决定写进产物的计分与哈希），把：

```python
CODE_FILES = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_layered.py", "q1_e2e.py",
              "q1_sensor.py", "q1_train.py", "q1_ar.py", "q1_g2ref.py")
```

改为：

```python
CODE_FILES = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_layered.py", "q1_e2e.py",
              "q1_sensor.py", "q1_train.py", "q1_ar.py", "q1_g2ref.py", "q1_run.py")
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_run.py sim/q1/test_q1_train.py -q`
Expected: `26 passed`（run 9 + train 17）。端到端用例在合成数据上跑 `prepare` + `job`，约 40 s。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_run.py sim/q1/test_q1_run.py sim/q1/q1_train.py
git commit -m "阶段一 Q1：命令行与产物（冻结闸门、可重入作业、原子写、文件哈希、留出会话不进拟合的清单检查）"
```

---

### Task 11: 汇总与推进判据

**Files:**
- Create: `sim/q1/q1_report.py`
- Test: `sim/q1/test_q1_report.py`

**Interfaces:**
- Consumes: Task 10 的 JSON 产物
- Produces:
  - 常量：`CANDIDATES`、`REFS`、`SEEDS`、`SESS`、`N_EPISODES = 100`
  - 完整性：`file_sha(path)`；`verify(raw) -> list[str]`，空列表 = 通过。检查以下各项：
    - 6 模型 × 3 种子 × 5 折，加 3 个参照 × 5 折；
    - `.pt` / `.npz` 存在，且哈希与记录一致；
    - 候选的 `config_hash` 唯一；
    - 全部 `code_sha` = 当前代码；
    - 训练会话不含留出会话；
    - 每份产物恰好覆盖同一组 100 集。
  - 载入与聚合：
    - `load_runs(out) -> {model: {seed: [rows]}}`：先 `verify`，有问题即抛 `RuntimeError`（列出全部）
    - `rho_by_session(rows, key="num")`
    - `seed_mean(by_seed, key="num")`
  - 判据与报告：
    - `gate(rho_c, rho_g2) -> {"per_session", "mean", "mean_g2", "pass"}`
    - `b_vs_a(rho_a, rho_b)`
    - `lead_rho(by_seed)`
    - `summarize(runs)`
    - `markdown(summary) -> str`
    - `main(argv)`：写 `OUT/summary.json`

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_q1_report.py`

```python
import json

import pytest

import q1_data as D
import q1_report as REP
import q1_train as TR

S = REP.SESS
BINS = ("3-6", "7-12", "13-27", "28-52", "53-102")


def _rho(vals):
    return dict(zip(S, vals))


def test_gate_ties_pass_per_session_but_mean_must_strictly_improve():
    g2 = _rho([0.6, 0.6, 0.6, 0.6, 0.6])
    assert not REP.gate(_rho([0.6] * 5), g2)["pass"]                     # 全相等：平均未严格改善
    assert REP.gate(_rho([0.6, 0.6, 0.6, 0.6, 0.5]), g2)["pass"]
    g = REP.gate(_rho([0.1, 0.1, 0.1, 0.1, 0.61]), g2)
    assert not g["pass"] and g["per_session"][S[4]] is False            # 平均好很多也不能救一个会话


def test_seed_mean_averages_errors_not_predictions():
    rows = lambda num: [{"sess": s, "num": num, "den": 1.0} for s in S]
    m = REP.seed_mean({"0": rows(0.2), "1": rows(0.4)})
    assert set(m) == set(S) and all(abs(v - 0.3) < 1e-12 for v in m.values())


def test_b_vs_a_sign():
    d = REP.b_vs_a(_rho([0.5] * 5), _rho([0.4] * 5))
    assert all(abs(v + 0.2) < 1e-12 for v in d.values())


def _episodes(fold):
    return [{"name": f"{fold}_ep{i}", "sess": fold, "num": 0.4, "den": 1.0, "f3": 3, "num_aux": 0.5,
             "lead": {b: [0.1, 0.2] for b in BINS}} for i in range(D.EXPECTED_COUNTS[fold])]


def _tree(tmp):
    """完整产物树：6 模型 × 3 种子 × 5 折（含 .pt/.npz）+ 3 个参照 × 5 折。"""
    code = TR.code_sha()
    for m in REP.CANDIDATES:
        for seed in REP.SEEDS:
            for f in S:
                base = tmp / "jobs" / m / f"seed{seed}" / f.replace("/", "_")
                base.parent.mkdir(parents=True, exist_ok=True)
                base.with_suffix(".pt").write_bytes(f"pt{m}{seed}{f}".encode())
                base.with_suffix(".npz").write_bytes(f"npz{m}{seed}{f}".encode())
                base.with_suffix(".json").write_text(json.dumps({
                    "model": m, "seed": int(seed), "fold": f, "config_hash": "cfg", "code_sha": code,
                    "train_sessions": [x for x in S if x != f],
                    "pt_sha256": REP.file_sha(base.with_suffix(".pt")),
                    "npz_sha256": REP.file_sha(base.with_suffix(".npz")), "episodes": _episodes(f)}))
    for r in REP.REFS:
        for f in S:
            p = tmp / "refs" / r / (f.replace("/", "_") + ".json")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"model": r, "seed": "-", "fold": f, "config_hash": code[:16],
                                     "code_sha": code, "episodes": _episodes(f)}))


def test_load_runs_accepts_complete_tree(tmp_path):
    _tree(tmp_path)
    s = REP.summarize(REP.load_runs(tmp_path))
    assert set(s["candidates"]) == set(REP.CANDIDATES) and set(s["b_vs_a"]) == {"1", "2", "3"}


@pytest.mark.parametrize("damage", ["missing_npz", "tampered_pt", "mixed_config", "missing_seed",
                                    "missing_episode", "stale_code", "missing_ref", "leaky_train"])
def test_load_runs_rejects_incomplete_or_inconsistent(tmp_path, damage):
    _tree(tmp_path)
    j1 = tmp_path / "jobs" / "2b" / "seed1" / "ab6_A.json"
    if damage == "missing_npz":
        j1.with_suffix(".npz").unlink()
    elif damage == "tampered_pt":
        j1.with_suffix(".pt").write_bytes(b"changed")
    elif damage == "missing_seed":
        for p in (tmp_path / "jobs" / "3a" / "seed2").iterdir():
            p.unlink()
    elif damage == "missing_ref":
        (tmp_path / "refs" / "ar" / "ab7_Apost.json").unlink()
    else:
        j = json.loads(j1.read_text())
        if damage == "mixed_config":
            j["config_hash"] = "other"
        elif damage == "missing_episode":
            j["episodes"] = j["episodes"][1:]
        elif damage == "stale_code":
            j["code_sha"] = "0" * 64
        elif damage == "leaky_train":
            j["train_sessions"] = list(S)
        j1.write_text(json.dumps(j))
    with pytest.raises(RuntimeError):
        REP.load_runs(tmp_path)
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_q1_report.py -q`
Expected: `ModuleNotFoundError: No module named 'q1_report'`

- [ ] **Step 3: 实现** —— `sim/q1/q1_report.py`

```python
"""阶段一汇总：逐会话误差比、推进判据（§5 #1）、b 相对 a、前瞻分层、辅助起点。

误差比 `ρ_s = Σ_e ‖P(gap−ê)‖² / Σ_e ‖P(gap)‖²`（会话 s 内按集池化；`redu = 1 − ρ`）。
训练种子：取各种子 `ρ_s` 的平均（= 平均误差，不平均预测）。
**推进判据**（每个候选单独判，不要求六个全过）：5 个留出会话都 `ρ_s ≤ ρ_s(G-2)`，
且 5 会话等权平均 `mean ρ_s < mean ρ_s(G-2)`。G-2 = 同口径窗口重预测。
⚠ 开发筛查规则，不代表统计上已证明泛化。
汇总前强制完整性检查：6 模型 × 3 种子 × 5 折 + 3 个参照 × 5 折；每个作业的权重与预测文件存在且
哈希与记录一致；候选的 config_hash 唯一；全部产物的 code_sha = 当前代码；每份产物恰好覆盖同一组 100 集。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_train as TR          # noqa: E402

CANDIDATES = ("1a", "1b", "2a", "2b", "3a", "3b")
REFS = ("g2", "g2_whole", "ar")
SEEDS = ("0", "1", "2")
SESS = tuple(f"{s}/{a}" for s, a in D.SESSIONS)
N_EPISODES = sum(D.EXPECTED_COUNTS.values())


def file_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(raw) -> list:
    """完整性检查 → 问题列表（空 = 通过）。"""
    P = []
    code_now = TR.code_sha()
    for m in CANDIDATES + REFS:
        by_seed = raw.get(m, {})
        want = SEEDS if m in CANDIDATES else ("-",)
        if sorted(by_seed) != sorted(want):
            P.append(f"{m}: 种子 {sorted(by_seed)}，应为 {list(want)}")
        for s, js in by_seed.items():
            if sorted(j["fold"] for j in js) != sorted(SESS):
                P.append(f"{m} seed {s}: 折 {sorted(j['fold'] for j in js)} 不齐")
            names = [r["name"] for j in js for r in j["episodes"]]
            if len(names) != N_EPISODES or len(set(names)) != N_EPISODES:
                P.append(f"{m} seed {s}: {len(set(names))} 个不同的集，应为 {N_EPISODES}")
            for j in js:
                if any(r["sess"] != j["fold"] for r in j["episodes"]):
                    P.append(f"{j['_path']}: 含非留出会话的集")
                if j.get("code_sha") != code_now:
                    P.append(f"{j['_path']}: code_sha 与当前代码不一致")
                if m not in CANDIDATES:
                    continue
                ts = j.get("train_sessions", [])
                if j["fold"] in ts or len(ts) != len(SESS) - 1:
                    P.append(f"{j['_path']}: 训练会话 {ts} 不对")
                for key, suf in (("pt_sha256", ".pt"), ("npz_sha256", ".npz")):
                    f = Path(j["_path"]).with_suffix(suf)
                    if not f.exists():
                        P.append(f"缺 {f}")
                    elif file_sha(f) != j.get(key):
                        P.append(f"{f}: 与记录的 {key} 不符")
    cfg = {j["config_hash"] for m in CANDIDATES for js in raw.get(m, {}).values() for j in js}
    if len(cfg) > 1:
        P.append(f"候选作业的 config_hash 不唯一：{sorted(cfg)}")
    sets = {tuple(sorted(r["name"] for j in js for r in j["episodes"]))
            for by_seed in raw.values() for js in by_seed.values()}
    if len(sets) > 1:
        P.append("各产物覆盖的集合不一致")
    return P


def load_runs(out) -> dict:
    """→ {model: {seed: [逐集行]}}。先做完整性检查；任何问题 ⇒ RuntimeError（列出全部）。"""
    raw: dict = {}
    for p in sorted((Path(out) / "jobs").glob("*/seed*/*.json")) + \
            sorted((Path(out) / "refs").glob("*/*.json")):
        j = json.loads(p.read_text())
        j["_path"] = str(p)
        raw.setdefault(j["model"], {}).setdefault(str(j["seed"]), []).append(j)
    problems = verify(raw)
    if problems:
        raise RuntimeError("产物不完整或不一致：\n  " + "\n  ".join(problems))
    return {m: {s: [r for j in js for r in j["episodes"]] for s, js in by_seed.items()}
            for m, by_seed in raw.items()}


def rho_by_session(rows, key: str = "num") -> dict:
    acc: dict = {}
    for r in rows:
        a = acc.setdefault(r["sess"], [0.0, 0.0])
        a[0] += r[key]
        a[1] += r["den"]
    return {s: n / d for s, (n, d) in acc.items()}


def seed_mean(by_seed, key: str = "num") -> dict:
    per = [rho_by_session(rows, key) for rows in by_seed.values()]
    return {s: float(np.mean([p[s] for p in per])) for s in per[0]}


def gate(rho_c: dict, rho_g2: dict) -> dict:
    per = {s: bool(rho_c[s] <= rho_g2[s]) for s in rho_g2}
    mc = float(np.mean([rho_c[s] for s in rho_g2]))
    mg = float(np.mean(list(rho_g2.values())))
    return {"per_session": per, "mean": mc, "mean_g2": mg, "pass": all(per.values()) and mc < mg}


def b_vs_a(rho_a: dict, rho_b: dict) -> dict:
    """逐会话相对变化 `(ρ_b − ρ_a)/ρ_a`；负 = b 更好。**只报告，不替代推进判据。**"""
    return {s: (rho_b[s] - rho_a[s]) / rho_a[s] for s in rho_a}


def lead_rho(by_seed) -> dict:
    acc: dict = {}
    k = len(by_seed)
    for rows in by_seed.values():
        for r in rows:
            for b, (n, d) in r["lead"].items():
                a = acc.setdefault(b, [0.0, 0.0])
                a[0] += n / k
                a[1] += d / k
    return {b: n / d for b, (n, d) in acc.items()}


def summarize(runs) -> dict:
    g2 = seed_mean(runs["g2"])
    out = {"g2": g2, "ar": seed_mean(runs["ar"]), "g2_whole": seed_mean(runs["g2_whole"]),
           "lead_refs": {m: lead_rho(runs[m]) for m in ("g2", "ar")},
           "aux_refs": {m: seed_mean(runs[m], "num_aux") for m in ("g2", "ar")},
           "candidates": {}, "b_vs_a": {}}
    for m in CANDIDATES:
        if m in runs:
            rho = seed_mean(runs[m])
            out["candidates"][m] = {
                "rho": rho, "gate": gate(rho, g2), "rho_aux": seed_mean(runs[m], "num_aux"),
                "rho_per_seed": {s: rho_by_session(r) for s, r in runs[m].items()},
                "lead": lead_rho(runs[m])}
    for f in ("1", "2", "3"):
        c = out["candidates"]
        if f + "a" in c and f + "b" in c:
            out["b_vs_a"][f] = b_vs_a(c[f + "a"]["rho"], c[f + "b"]["rho"])
    return out


def markdown(s: dict) -> str:
    pct = lambda r: f"{(1 - r) * 100:+.2f}%"
    head = "| 模型 | " + " | ".join(SESS) + " | 等权平均 | 推进判据 |"
    sep = "|" + "---|" * (len(SESS) + 3)
    L = ["### 留出会话 redu（= 1 − ρ，种子平均）", "", head, sep]

    def row(name, rho, verdict=""):
        return f"| {name} | " + " | ".join(pct(rho[x]) for x in SESS) + \
            f" | {pct(float(np.mean([rho[x] for x in SESS])))} | {verdict} |"

    L.append(row("G-2（同口径重预测）", s["g2"], "比较对象"))
    for m, c in s["candidates"].items():
        g = c["gate"]
        bad = [x for x in SESS if not g["per_session"][x]]
        L.append(row(m, c["rho"], "**过**" if g["pass"] else f"不过（{', '.join(bad) or '平均未严格改善'}）"))
    L.append(row("折内 AR 参照", s["ar"], "参照"))
    L.append(row("G-2 旧口径（整集前向，只对照）", s["g2_whole"], "不作判据"))
    L += ["", "### 状态历史：b 相对 a 的逐会话变化（负 = b 更好；只报告）", "",
          "| 结构 | " + " | ".join(SESS) + " |", "|" + "---|" * (len(SESS) + 1)]
    for f, d in s["b_vs_a"].items():
        L.append(f"| {f}b vs {f}a | " + " | ".join(f"{d[x] * 100:+.1f}%" for x in SESS) + " |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    s = summarize(load_runs(a.out))
    (Path(a.out) / "summary.json").write_text(json.dumps(s, indent=1, ensure_ascii=False))
    print(markdown(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_q1_report.py -q`
Expected: `12 passed`

- [ ] **Step 5: 提交**

```bash
git add sim/q1/q1_report.py sim/q1/test_q1_report.py
git commit -m "阶段一 Q1：汇总与推进判据（强制完整性核验；逐会话不高于 G-2 且等权平均严格改善；b 相对 a 只报告）"
```

---

### Task 12: pp2 波形图与正式运行脚本

**Files:**
- Create: `sim/q1/fig_pp2_q1.py`, `sim/q1/run_q1_stage1.sh`
- Test: `sim/q1/test_fig_pp2_q1.py`

**Interfaces:**
- Consumes:
  - `q1_train.rebuild / predict_episode`
  - `q1_g2ref.episode_ehat`
  - `sim.step2_gap_baseline.bare_logs()`：17 集 pp2，已核实满足同一数据契约（584–587 帧、6 chunk、编码器第 2 帧起有效）
  - `fit_step1_candidate.bandpass41`
- Produces:
  - 函数：
    - 信号：`bp(x)`、`imu_pair(ep, ehat) -> (real_bp, pred_bp)`
    - 位移：`ratio(real, pred, L=0)`、`best_shift(pairs)`
    - 画图：`plot(rows, label, Ln, Lg, path)`、`main(argv)`
  - 输出图：`OUT/fig/pp2_{model}_vs_real.png`
  - 运行脚本 `run_q1_stage1.sh`，依次：
    1. `prepare`
    2. 并行 `job`
    3. 复查未完成作业
    4. `refs`
    5. `fullfit`
    6. `report`
    7. 6 张图

- [ ] **Step 1: 写失败的测试** —— `sim/q1/test_fig_pp2_q1.py`

```python
import numpy as np

import fig_pp2_q1 as F


def test_best_shift_recovers_known_lag():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    pairs = [(x[1:301], x[:300])]                  # real[k] = x[k+1] ⇒ real[k−1] 对 pred[k]，L = −1
    assert F.best_shift(pairs) == -1
    assert F.ratio(x[:300], x[:300], 0) == 0.0


def test_plot_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    rows = [{"name": f"ep{i}", "real": rng.normal(size=(300, 3)), "new": rng.normal(size=(300, 3)),
             "g2": rng.normal(size=(300, 3)), "r_new": float(i), "r_g2": 1.0} for i in range(5)]
    p = tmp_path / "f.png"
    F.plot(rows, "2b", 0, 1, p)
    assert p.exists() and p.stat().st_size > 10_000
```

- [ ] **Step 2: 运行，确认失败**

Run: `$PY -m pytest sim/q1/test_fig_pp2_q1.py -q`
Expected: `ModuleNotFoundError: No module named 'fig_pp2_q1'`

- [ ] **Step 3: 实现**

`sim/q1/fig_pp2_q1.py`（版式沿用 `sim/step2_g/fig_pp2_gq_vs_real.py`）：

```python
"""pp2（17 集开发基准集）：候选 G 的 IMU 输出 vs 実機 IMU（+ G-2 参照线）。

规则「改完 G 必出 pp2 波形图」：沿用 `sim/step2_g/fig_pp2_gq_vs_real.py` 的版式 ——
中位集 + 最难集、0.5–9 Hz、各模型取自己的最优大域位移并写在图上、两条口径注记。
候选 = `q1_run.py fullfit` 的全数据拟合（seed 0，**只作图，不进任何判据**）；
G-2 = 同口径窗口重预测。窗口规则与阶段一相同（chunk 开头起点、末端钳位）。

    python -u sim/q1/fig_pp2_q1.py --out outputs/q1_stage1 --model 2b
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_windows as W         # noqa: E402

SHIFTS = (-2, -1, 0, 1, 2)
C_REAL, C_NEW, C_G2 = "#1f6fb2", "#8b2fb0", "#9aa0a6"


def bp(x) -> np.ndarray:
    import fit_step1_candidate as S1
    x = np.asarray(x, np.float64)
    return S1.bandpass41(x - x.mean(0, keepdims=True))


def imu_pair(ep, ehat) -> tuple[np.ndarray, np.ndarray]:
    """→ (帧 3…T−1 的实测陀螺, 预测陀螺 `ω̂_{t+3} = ideal_t + ê_t`)，均已带通。"""
    real = ep.w[D.ALIGN_L:D.ALIGN_L + ep.n]
    return bp(real), bp(ep.ideal[:ep.n] + ehat)


def ratio(real, pred, L: int = 0) -> float:
    """残差/实测（1.0 = 输出全零）；`L` = 大域位移（比较 real[k+L] 与 pred[k]）。"""
    if L >= 0:
        a, b = real[L + D.EDGE:len(real) - D.EDGE], pred[D.EDGE:len(pred) - D.EDGE - L]
    else:
        a, b = real[D.EDGE:len(real) - D.EDGE + L], pred[D.EDGE - L:len(pred) - D.EDGE]
    return float(((a - b) ** 2).sum() / max((a ** 2).sum(), 1e-12))


def best_shift(pairs) -> int:
    """全部集池化后残差最小的位移。"""
    def pooled(L):
        return sum(ratio(r, p, L) * float((r ** 2).sum()) for r, p in pairs)
    return min(SHIFTS, key=pooled)


def plot(rows, label: str, Ln: int, Lg: int, path: Path) -> None:
    """rows: [{"name","real","new","g2","r_new","r_g2"}]；按 r_new 取中位集与最难集。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort([r["r_new"] for r in rows])
    med, hard = rows[order[len(order) // 2]], rows[order[-1]]
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.33, 7.5))
    gs = fig.add_gridspec(2, 1, hspace=0.42, left=0.065, right=0.985, top=0.845, bottom=0.105)

    def wave(ax, e, title, span=200):
        ax_i = int(np.argmax(np.abs(e["real"]).max(0)))
        c = int(np.argmax(np.abs(e["real"][:, ax_i])))
        s = max(0, min(c - span // 2, len(e["real"]) - span))
        t = np.arange(span) / D.FS
        ax.plot(t, e["real"][s:s + span, ax_i], color=C_REAL, lw=2.4, label="実機 IMU")
        ax.plot(t, e["new"][s:s + span, ax_i], color=C_NEW, lw=2.0, ls="--", label=label)
        ax.plot(t, e["g2"][s:s + span, ax_i], color=C_G2, lw=1.6, ls=":", label="G-2（参照）")
        ax.set_title(f"{title}  {e['name']}   残差/実測 {label} {e['r_new']:.3f} · G-2 {e['r_g2']:.3f}"
                     f"   軸 ω{'xyz'[ax_i]}", fontsize=12, color="#22223b", loc="left")
        ax.set_xlabel("時間 (s)", fontsize=11)
        ax.set_ylabel("角速度 (rad/s)\n0.5–9 Hz 帯域通過", fontsize=10.5)
        ax.grid(alpha=0.25)

    a1 = fig.add_subplot(gs[0])
    wave(a1, med, "① 中央値のエピソード")
    a1.legend(fontsize=11, frameon=False, ncol=3, loc="upper right")
    wave(fig.add_subplot(gs[1]), hard, "② 最も難しいエピソード")
    fig.suptitle(f"pp2（pick-and-place）：{label} の IMU 出力 と 実機 IMU",
                 fontsize=18, fontweight="bold", color="#22223b", y=0.982)
    fig.text(0.5, 0.912, f"帯域 0.5–9 Hz・大域シフト {label} L={Ln} / G-2 L={Lg}・"
             "chunk 先頭起点・末端クランプ・候補は 100 集全数学習（この 17 集は未使用）",
             ha="center", fontsize=11.5, color="#5a5a72")
    fig.text(0.5, 0.018, "※ 17 集は「pp2 開発基準集」であり最終密封 holdout ではない。"
             "残差/実測 は 1.0 が「出力ゼロ」相当。", ha="center", fontsize=10, color="#c1121f")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main(argv=None) -> int:
    gc.disable()
    import torch
    import q1_g2ref as G
    import q1_train as TR
    from sim.step2_gap_baseline import bare_logs
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, choices=list(TR.MODELS))
    a = ap.parse_args(argv)
    R, b = D.calibration()
    ck = torch.load(Path(a.out) / "fullfit" / f"{a.model}.pt", weights_only=False)
    model = TR.rebuild(a.model, ck["stats"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    Wrs = C = None
    if TR.MODELS[a.model][0] == "layered":
        z = np.load(Path(a.out) / "fullfit" / "sensor_all.npz")
        Wrs, C = z["Wrs"], {h: z["C"][h] for h in range(len(z["C"]))}
    g2 = G.load_g2()
    eps = [D.load_episode(p, "pp2/bare") for p in bare_logs()]
    raw = []
    for e in eps:
        ws = W.chunk_windows(e)
        new = TR.predict_episode(a.model, model, ck["stats"], e, ws, R, Wrs, C)
        real, pn = imu_pair(e, new)
        _, pg = imu_pair(e, G.episode_ehat(g2, e, ws, R, b))
        raw.append((e.name, real, pn, pg))
    Ln = best_shift([(r, p) for _, r, p, _ in raw])
    Lg = best_shift([(r, g) for _, r, _, g in raw])
    rows = [{"name": n, "real": r, "new": p, "g2": g, "r_new": ratio(r, p, Ln), "r_g2": ratio(r, g, Lg)}
            for n, r, p, g in raw]
    for r in rows:
        print(f"  {r['name']:30s} {a.model} {r['r_new']:.3f}   G-2 {r['r_g2']:.3f}", flush=True)
    print(f"  逐集中位: {a.model} {np.median([r['r_new'] for r in rows]):.3f} · "
          f"G-2 {np.median([r['r_g2'] for r in rows]):.3f}   L={Ln}/{Lg}")
    p = Path(a.out) / "fig" / f"pp2_{a.model}_vs_real.png"
    plot(rows, a.model, Ln, Lg, p)
    print(f"  → {p}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
```

`sim/q1/run_q1_stage1.sh`：

```bash
#!/usr/bin/env bash
# 阶段一（Q1）正式运行 —— **只在 reports/PREREG_Q1_STAGE1.md 冻结后**，在用户自己的终端里运行。
# 可重入：配置/代码指纹一致的已完成作业会被跳过；中断后直接重跑本脚本即可。
#   bash sim/q1/run_q1_stage1.sh            # 默认 OUT=outputs/q1_stage1 JOBS=15
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/enolab-guo/miniconda3/envs/lerobot/bin/python}
OUT=${OUT:-outputs/q1_stage1}
JOBS=${JOBS:-15}
MODELS="1a 1b 2a 2b 3a 3b"
mkdir -p "$OUT/logs"
"$PY" -u sim/q1/q1_run.py prepare --out "$OUT" 2>&1 | tee "$OUT/logs/prepare.log"
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" > "$OUT/logs/todo.txt"
echo "待跑作业 $(wc -l < "$OUT/logs/todo.txt") 个，并行 $JOBS"
xargs -P "$JOBS" -L 1 sh -c \
  '"$0" -u sim/q1/q1_run.py job --out "$1" "$2" "$3" "$4" > "$1/logs/job_$2_$(echo "$3" | tr / _)_$4.log" 2>&1' \
  "$PY" "$OUT" < "$OUT/logs/todo.txt"
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" > "$OUT/logs/todo_after.txt"
if [ -s "$OUT/logs/todo_after.txt" ]; then
  echo "仍有未完成作业（见 $OUT/logs/todo_after.txt 与 job_*.log）—— 修复后重跑本脚本"; exit 1
fi
"$PY" -u sim/q1/q1_run.py refs --out "$OUT" 2>&1 | tee "$OUT/logs/refs.log"
for m in $MODELS; do
  "$PY" -u sim/q1/q1_run.py fullfit --out "$OUT" "$m" > "$OUT/logs/fullfit_$m.log" 2>&1 &
done
wait
"$PY" -u sim/q1/q1_report.py --out "$OUT" | tee "$OUT/logs/report.md"
for m in $MODELS; do
  "$PY" -u sim/q1/fig_pp2_q1.py --out "$OUT" --model "$m" | tee "$OUT/logs/fig_$m.log"
done
echo "完成：$OUT/summary.json、$OUT/logs/report.md、$OUT/fig/"
```

- [ ] **Step 4: 运行，确认通过**

Run: `$PY -m pytest sim/q1/test_fig_pp2_q1.py -q && bash -n sim/q1/run_q1_stage1.sh && echo ok`
Expected: `2 passed` 与 `ok`

接着用假解释器空跑一遍脚本，确认 `xargs` 传参正确（**不会真的训练**）：

```bash
SP=$(mktemp -d); cat > $SP/fakepy.sh <<'EOF'
#!/usr/bin/env bash
echo "$*" >> "$(dirname "$0")/calls.log"
case "$*" in *list-jobs*) [ -f "$(dirname "$0")/listed" ] || { touch "$(dirname "$0")/listed"; printf '1a ab5/A 0\n2b ab7/Apost 2\n'; } ;; esac
EOF
chmod +x $SP/fakepy.sh
PY=$SP/fakepy.sh OUT=$SP/out JOBS=2 bash sim/q1/run_q1_stage1.sh > /dev/null && grep -c " job " $SP/calls.log
```

Expected: 打印 `2`。另外 `$SP/out/logs/` 下应有 `job_1a_ab5_A_0.log` 与 `job_2b_ab7_Apost_2.log`。

- [ ] **Step 5: 提交**

```bash
git add sim/q1/fig_pp2_q1.py sim/q1/run_q1_stage1.sh sim/q1/test_fig_pp2_q1.py
git commit -m "阶段一 Q1：pp2 波形图（候选 / 实机 / G-2）与可重入正式运行脚本"
```

---

### Task 13: 全量自测 + 耗时测量 + 预注册草稿 → **停，等用户冻结**

**Files:**
- Create: `reports/PREREG_Q1_STAGE1.md`

- [ ] **Step 1: 全量自测**

Run: `$PY -m pytest sim/q1 -q`
Expected: `98 passed`，约 3–4 分钟。任何失败都先修，不进入 Step 2。

- [ ] **Step 2: 耗时测量**

这一步只测耗时：在 ab5/A 折的训练会话上训 30 步，**不预测留出、不存权重**；数字只用于填预算。

Run: `$PY -u sim/q1/q1_run.py timing --steps 30`
Expected: 6 行 `<模型>: x.xxx s/步`，再加一行「预计 … 核·小时 … 最长单作业 … 分钟」。镜像里的单线程参考值：
- 1a ≈ 0.32 s/步，1b ≈ 0.15 s/步；
- 端到端 ≈ 0.01–0.02 s/步。

- [ ] **Step 3: 写预注册草稿** —— `reports/PREREG_Q1_STAGE1.md`

填入以下三项：
- 抬头：`git rev-parse --short HEAD`，以及 `code_sha=<16 位>`（**照此格式**，冻结闸门按 `code_sha=` 后的 16 位十六进制核对）；后者用 `$PY -c "import sys; sys.path.insert(0,'sim/q1'); import q1_train as TR; print(TR.code_sha()[:16])"` 得到；
- §7：Step 2 的 6 个数字与两个预计值；
- 日期。

全文如下：

```markdown
# 预注册 —— G→π v2 阶段一（Q1：原始误差预测）

**状态：草稿（<日期>），待用户冻结。冻结前不运行 `prepare / job / refs / fullfit`。**
- 依据：`reports/DESIGN_G_PI_V2.md` §2、§4 阶段一、§5 #1–4（用户 2026-10-02 定）。
- 实施：`reports/PLAN_IMPL_Q1_STAGE1.md`；代码 `sim/q1/`（提交 `<git 短哈希>`，`code_sha=<16 位十六进制>`）。
- 冻结时把状态行整行改为 `**状态：已冻结（<日期>，用户确认）**`。正式入口只认这一行，并核对 `code_sha=`；代码改动后须重新冻结。
- 零机时；不训练 π；不做 Q2 / Q3；不修改 G-2。

## 0 冻结前已经看到的数（如实披露）

- G-2 旧口径（整集前向）逐会话 redu：ab5 39.56 / ab6 36.49 / ab7 33.36 / ab2 23.41 / ab4 50.03%。来源 `outputs/fabs_g0/fabs_g0_v1.json`，已发表；Task 3 复现到 1e-5。
- 写实施计划时的原型检查（2026-10-02，在暂存区镜像里做，未提交）：
  - 折内 AR 参照在 ab5/A 留出上的冒烟值 redu **+37.7%**（一次，非正式）。⚠ 当时的 AR 含逐前瞻截距与有效比例列（可学到 chunk 相位模板），现已去掉；**新形式未计算过**；
  - 用**实测关节（含未来）**的传感器层在 ab5/A 前 10 集 redu 0.80 —— 实现自检，**不是模型分数**；
  - G-2 窗口重预测与整集前向只在每个 chunk 末 6 个计分帧不同（最大 0.55 rad/s）。
- **未计算过**任何候选模型的留出分数，也**未计算过** G-2 同口径重预测的分数。

## 1 数据

- 100 集裸 ACT：ab5/A 40 · ab6/A 30 · ab7/Apost 15 · ab2/A 10 · ab4/P0 5。筛选与 `train_fabs_g0.load_all` 相同，Task 1 逐集核对。
- 标签 `gap[t] = ω_real[t+3] − ω_ideal[t]`；`R, b` 取 `step1_H_v1.json`；20 Hz 朴素抽取口径。
- 编码器 `state`（mdeg）前 2 帧全为零 ⇒ 判无效、掩码；IMU 全部有效。每集 6 个 100 帧 chunk，记录 580–596 帧。
- 留一会话 5 折。归一化统计、`r_s`、1b 前瞻项、AR 系数**只用该折的 4 个训练会话**。

## 2 时间轴与计分（设计稿 §2.3）

- 主起点 = 每个 chunk 的开头 `t0`。实测可用到 `t0`；指令和理想陀螺可用到 `t1`，之后钳位到第 `t1` 帧。
- 计分范围 `ê_t, t ∈ [t0, min(t1, n−1)]`。逐 chunk 拼接成整集，再算 `P(·)`：去均值 → bandpass41（0.5–9 Hz）→ 丢首尾 20。
- 会话误差比 `ρ_s = Σ_e ‖P(gap−ê)‖² / Σ_e ‖P(gap)‖²`，`redu_s = 1 − ρ_s`。
- 辅助项（不进主分数、不挑模型）：
  - chunk 中段起点 `t0 + 50`，覆盖后半个 chunk；
  - 前瞻分层 3–6 / 7–12 / 13–27 / 28–52 / 53–102 帧（带通会混合相邻帧，只是近似）。

## 3 模型与参照

| 编号 | 结构 | 输入 | 参数 |
|---|---|---|---|
| 1a / 1b | 分层：逐关节伺服 MLP（3×32 softsign）→ `R·kin(q̂) + r_s`；1b 另加实测传感器残差的逐前瞻线性项（**无截距**，没有有效残差时精确为 0） | 1a 只用指令（开环起点状态）；1b 在起点读实测编码器与陀螺历史 20 帧 | 24 780 + 闭式层 |
| 2a / 2b | 端到端 TCN（因果空洞卷积 4 层 64ch，与 `F_abs` 主干同构），直接多步 | 过去 20 帧 + 窗口 102 帧；b 版加实测历史 | 66 441 / 69 961 |
| 3a / 3b | 端到端因果 Transformer（2 层、d64、4 头、FF128），直接多步 | 同上 | 76 361 / 77 065 |
| G-2 | 冻结；同口径窗口重预测 | 只读指令 | — |
| AR | 折内线性 AR：逐前瞻 OLS，三轴联合，20 个原始 gap，**无截距** | 只用起点可得的 gap | — |
| G-2 旧口径 | 整集前向（只对照，不作判据） | — | — |

结构性说明：
- TCN 感受野 61 帧 ⇒ b 版的实测历史只能影响前瞻 ≤ 60 帧的预测；Transformer 能看到全窗口。
- 1a 的起点状态来自整集开环，隐含使用了全部过去指令；端到端 a 版只看过去 20 帧指令。这是两种结构固有的差别。
- **训练方法不同**（用户 2026-10-02 要求注明）：
  - 分层：关节层只用关节损失训练，传感器层用实测关节闭式拟合 —— 训练用实测关节、推理用 q̂（训练—推理差异，首版保留）；
  - 端到端：ω + 关节联合损失端到端训练，还能从窗口位置学到 chunk 相位模板；分层没有对应项。
  - ⇒ 三行之间的胜负**不能单独归因于网络结构**。
- 无效样本：编码器无效值不经中心差分、FIR、带通进入任何拟合或目标（按支撑排除）；推理时支撑内有效/无效混合即按数据契约报错。本数据中只有 t0 = 0 是全填充。

## 4 训练（§11 列出的默认值待确认）

- 每步均匀抽 1 个训练集，取其 6 个主窗口，拼接后算损失。
- Adam lr 1e-3，梯度裁剪 1.0，**3000 步，取最后一步**；不早停，不挑检查点。
- 损失：
  - 端到端 = ω 项 + 关节项。ω 项为 `‖P(gap−ê)‖²/(3T_e)`，除以训练折零预测误差的均值；关节项为 0.5·宽带 MSE/尺度 + 0.5·3–9 Hz 带内 MSE/尺度。
  - 分层模型的关节层只用关节项；传感器层闭式拟合。
- 训练种子 0 / 1 / 2，全部报告。判据用各种子 `ρ_s` 的平均（平均误差，不平均预测）。
- CPU 单线程 + 确定性算法：同种子逐位复现（已测）。
- 留出会话只在最后预测一次。

## 5 判据（设计稿 §5 #1，用户定）

每个候选单独判，不要求六个全部通过。
- **推进**：5 个留出会话都满足 `ρ_s(候选) ≤ ρ_s(G-2 同口径)`，**且** `mean_s ρ_s(候选) < mean_s ρ_s(G-2)`（5 会话等权）。
- **状态历史**：逐会话报告 `(ρ_b − ρ_a) / ρ_a`，3 个结构各一行；**不替代**推进判据。
- ⚠ 这是开发筛查规则，不代表统计上已证明泛化。
- 超预算（§7）⇒ 记为未完成；不临时改判据，不补跑挑结果。

## 6 报告项（不作判据）

- 折内 AR 参照的逐会话 redu：判断 b 版收益是否只是误差延续。
- G-2 旧口径与同口径重预测之差。
- 前瞻分层表；chunk 中段起点的辅助 redu。所有模型只在 chunk 开头训练过 ⇒ 中段起点含分布外成分。
- 逐种子的 `ρ_s`。
- pp2 17 集波形图，6 张。候选用 100 集全数据、seed 0 拟合，只作图。
- `F_abs` 只引用旧结果 +40.70%（旧口径），不并列比较。

## 7 预算（Task 13 实测）

| 模型 | 1a | 1b | 2a | 2b | 3a | 3b |
|---|---|---|---|---|---|---|
| s/步（单线程，30 步实测） | <…> | <…> | <…> | <…> | <…> | <…> |

- 预计：<…> 核·小时；最长单作业 <…> 分钟。
- 15 路并行，含参照与 6 个全数据拟合，墙钟约 <…>。
- **上限：墙钟 ≤ 4 小时**；超出 ⇒ 记为未完成。

## 8 证据边界与措辞（设计稿 §6）

- 阶段一即使成功，也只支持「20 Hz 口径下、这 5 个会话上的 Q1 预测收益」，不代表 Q2 / Q3。
- b 未优于 a 时，只能写「本轮未测到实测历史的预测收益」。不能写「状态历史无信息」，也不能推出「π 学不到有用反馈」（排除 ≠ 证明）。
- G 内部的任何改善都不是补正有效的证据；直到阶段四才检验 π 的真机效果。
- 比较三种结构时，只能写「在本训练方法组合下……」，**不能把胜负单独归因于网络结构**（见 §3）。

## 9 不做

- GRU、SUBNET / 深层 SSM、单独的容量对照。
- 不扫描历史窗长。
- 不用中段起点挑模型。
- 不重训或重评分 `F_abs`。
- 不修改 G-2。
- 不进入阶段二。

## 10 产物

`outputs/q1_stage1/` 下保存：
- `sensor/`；
- `jobs/{model}/seed{k}/{fold}.{json,pt,npz}`（每折每种子的权重、逐集计分与拼接预测）；
- `refs/`、`fullfit/`、`fig/`；
- `summary.json`、`logs/`。

- 每个作业的 JSON 记录 `config_hash`、`code_sha`、`state_sha256`，以及 `.pt` / `.npz` 的文件 sha256。
- 汇总前强制核验以下各项，任一不满足即拒绝汇总：
  - 6 模型 × 3 种子 × 5 折，加 3 个参照 × 5 折；
  - 文件存在且哈希一致；
  - 候选的 `config_hash` 唯一；
  - `code_sha` = 当前代码；
  - 训练会话不含留出会话；
  - 每份产物覆盖同一组 100 集。

结果报告记录冻结提交哈希与 `code_sha`。

## 11 用户审阅订正（2026-10-02，已定，不再是默认值）

1. 端到端 `q̂` 的基线用钳位后的可见指令；`ê` 与 `q̂` 都测「改 `t1` 之后的指令不变」。
2. 1b 附加项只用掩码后的残差数值，无截距、无有效比例；没有有效残差时精确为 0。
3. 传感器层的无效关节样本按支撑排除，推理时混合即报错。
4. 正式入口检查冻结状态与 `code_sha`；`timing` 单独允许。
5. 汇总强制核验产物完整一致（见 §10）。
6. `r_s` 用实测关节拟合保留为首版，并注明训练—推理差异与训练方法差异（见 §3）。

## 12 我定的默认值（逐条待你确认或改）

1. 分层关节层：逐关节 MLP 3×32 softsign，输入 61 维（20 帧跟踪误差 + 20 帧速度 + 20 帧有效位 + 下一帧指令误差），一阶滞后先验 `g = 0.3`，头部零初始化。
2. 1a 训练时，起点状态取自无梯度的开环滚动；窗口内有梯度。
3. 传感器层 `r_s`（是否用实测关节拟合已由你定为首版，见 §11-6）：
   - FIR nlag 16、相对岭 1e-4，沿用 G-2 的选定值，不调参；
   - 在 0.5–9 Hz 带内拟合，特征加理想陀螺。
4. 1b 读实测 IMU 的方式：起点前 20 帧的实测传感器残差 → 逐前瞻线性项（普通最小二乘；无截距按 §11-2）。
5. 端到端：
   - 未来段 102 帧，外加「是否仍在当前 chunk」指示通道；
   - 两种网络都因果；
   - 输出头零初始化（起点 = 零预测）。
6. 填充：指令 `u_0`、理想陀螺 0、编码器 `u_0`、陀螺 0，并加有效位。这是设计稿 §2.4 的默认值。
7. 损失权重：ω 项与关节项 1:1；关节项内，宽带与 3–9 Hz 带内 0.5:0.5（同 G-2 `r_q`）。
8. 训练：3000 步、lr 1e-3、每步 1 集、种子 0/1/2、CPU 单线程。
9. 推进判据的操作化：
   - 「误差」= 会话池化误差比 `ρ_s`；
   - 「等权平均」= 5 个会话 `ρ_s` 的算术平均；
   - 多种子 = 取 `ρ_s` 的平均。
10. 辅助项：中段起点 = chunk 开头 + 50 帧；前瞻分组 3–6 / 7–12 / 13–27 / 28–52 / 53–102。
11. pp2 图：
    - 候选用全部 100 集训练的 seed 0 模型，只作图；
    - 三条线：实机 / 候选 / G-2；
    - 不画逐集条形，沿用脚本 2026-09-02 的版式；逐集数值打印到日志。
12. AR 参照的具体形式：设计稿 §2.5 的默认值，**另按 §11-2 的同一原则去掉截距与有效比例列**。这是我补的，你可撤回；撤回则恢复带截距的形式。
13. 预算上限：墙钟 ≤ 4 小时。
```

- [ ] **Step 4: 提交草稿**

```bash
git add reports/PREREG_Q1_STAGE1.md
git commit -m "阶段一 Q1：预注册草稿（待用户冻结；不授权正式运行）"
```

- [ ] **Step 5: 停**

把预注册 §12「我定的默认值」逐条交给用户确认或修改（§11 是已定的用户订正，一并给出）。

- 用户冻结前**不执行 Task 14**。
- 用户若改了任何默认值，按顺序处理：
  1. 先改代码与测试；
  2. 重跑 Step 1–2；
  3. 更新草稿；
  4. 再请用户冻结。

---

### Task 14: 正式运行与结果（**仅在预注册冻结后**）

**Files:**
- Modify: `reports/PREREG_Q1_STAGE1.md`（只改状态行）
- Create: `reports/RESULT_Q1_STAGE1.md`

- [ ] **Step 1: 记录冻结**

把预注册状态行整行改为 `**状态：已冻结（<日期>，用户确认）**`（冻结闸门只认以「**状态：已冻结」开头的行），并确认 `code_sha=` 仍等于当前代码，然后提交：

```bash
git add reports/PREREG_Q1_STAGE1.md
git commit -m "阶段一 Q1：预注册冻结"
```

记下这个提交哈希。冻结后若代码变化，`config_hash` 会变，旧产物自动作废。

- [ ] **Step 2: 正式运行**

在用户自己的终端运行（也可在本会话输入 `! bash sim/q1/run_q1_stage1.sh`）：

Run: `bash sim/q1/run_q1_stage1.sh`
Expected: 结尾打印「完成：outputs/q1_stage1/summary.json …」。脚本可重入：中断或某个作业失败时，查 `outputs/q1_stage1/logs/job_*.log`，修复后重跑即可。

- [ ] **Step 3: 核对产物**

依次核对：
- 作业 JSON 数：`ls outputs/q1_stage1/jobs/*/seed*/*.json | wc -l` → `90`；
- 参照 JSON 数：`ls outputs/q1_stage1/refs/*/*.json | wc -l` → `15`；
- 汇总：`$PY -u sim/q1/q1_report.py --out outputs/q1_stage1` 不报错，即配置指纹唯一、折齐全；
- 图：`ls outputs/q1_stage1/fig/` 有 6 张 `pp2_*_vs_real.png`。

墙钟或核·小时超出预注册 §7 的上限 ⇒ 按预注册记为未完成，**不临时改判据、不补跑挑结果**。

- [ ] **Step 4: 写结果** —— `reports/RESULT_Q1_STAGE1.md`

数字一律取自 `summary.json` 与 `logs/report.md`。结构如下：

```markdown
# 结果 —— G→π v2 阶段一（Q1：原始误差预测）

预注册 `reports/PREREG_Q1_STAGE1.md`（冻结于 `<哈希>`）· 代码 `sim/q1/`（`<哈希>`，code_sha `<16 位>`）· 零机时 · 用时 <墙钟> / <核·小时>（预算 <…>）

## 一、推进判据（§5 #1）
<粘贴 report.md 第一张表>
- 通过：<编号>；不过：<编号 + 哪个会话不满足 / 平均未严格改善>。
- ⚠ 这是开发筛查规则，不代表统计上已证明泛化。
- ⚠ 三行（结构）之间的胜负**不能单独归因于网络结构**，原因有三：
  - 分层与端到端的训练方法不同：分层的关节层只用关节损失，传感器层用实测关节闭式拟合；端到端用 ω + 关节联合损失；
  - `r_s` 训练用实测关节、推理用 q̂；
  - 端到端能从窗口位置学到 chunk 相位模板，分层没有对应项。

## 二、状态历史：b 相对 a（只报告，不替代推进判据）
<粘贴 report.md 第二张表>
- b 未优于 a 的结构，只写「本轮未测到实测历史的预测收益」。
- 不写「状态历史无信息」，也不推出「π 学不到有用反馈」。

## 三、参照
- 折内 AR 参照：<逐会话>。用来判断 b 版的收益是否只是误差延续。
- G-2 旧口径（整集前向）vs 同口径重预测：<逐会话差>。旧口径的窗口尾部用到了下一 chunk 的指令。
- `F_abs`：只引用旧结果 +40.70%（旧口径：逐帧因果、整集、无预测起点），不与本表并列比较。

## 四、前瞻分层与 chunk 中段起点（只报告，不作判据）
<summary.json 的 lead / rho_aux>
- 带通混合相邻帧 ⇒ 前瞻分层是近似。
- 中段起点上的所有模型只在 chunk 开头训练过 ⇒ 含分布外成分。

## 五、pp2 波形图
<6 张图的路径 + 逐集中位残差/实测>
- 17 集是 pp2 开发基准集，不是最终密封 holdout。
- 残差/实测 1.0 = 输出零。
- 候选用的是全数据拟合（只作图）。

## 六、证据边界（spec §6 原文）
- 阶段一即使成功，也只支持「20 Hz 口径下、这 5 个会话上的 Q1 预测收益」，不代表 Q2 / Q3。阶段一的推进判据是开发筛查规则，不代表统计上已证明泛化。
- b 未优于 a 时，只能写「本轮未测到实测历史的预测收益」，不能写「状态历史无信息」，也不能推出「π 学不到有用反馈」（排除 ≠ 证明）。
- G 内部的任何改善都不是补正有效的证据。
- 直到阶段四才检验 π 的真机效果。

## 七、产物
<outputs/q1_stage1 下各文件的 sha256 / state_sha256 摘要；冻结提交哈希>
```

- [ ] **Step 5: 提交结果**

```bash
git add reports/RESULT_Q1_STAGE1.md
git commit -m "阶段一 Q1：结果（推进判据逐候选、b 相对 a、参照与 pp2 波形图）"
```

- [ ] **Step 6: 更新记忆**

- 改 `memory/plan_g_pi_design_brainstorm.md`：写入阶段一结论、措辞约束、产物路径；
- 视需要新建一条 `finding_q1_stage1.md`，并在 `MEMORY.md` 加一行索引。

- [ ] **Step 7: 停**

阶段二（Q2 检验）需另写预注册（§5 #5 门槛仍待定），**不自动开始**。

---

## 修订记录

- 2026-10-02 初版（`ad9fa7a`）。
- 2026-10-02 按用户静态审阅订正（执行方式定为 Native；只实施 Task 1–13，交付预注册草稿即停）：
  1. 端到端 `decode` 的 `q̂` 基线改用钳位后的可见指令。新增测试：`test_no_leakage` 同时查 `ê` 与 `q̂`；`test_forward_windows_ignore_commands_after_t1` 覆盖 6 个模型的训练前向。
  2. 1b 附加项去掉截距与有效比例 ⇒ 没有有效残差时精确为 0。AR 参照按同一原则去掉截距（我补的，可撤回）。
  3. 传感器层与关节损失：无效样本按支撑排除（`clean_rows` / `band_clean`），推理时支撑混合即报错（`check_window_support`）。
  4. 正式入口加冻结闸门（`require_frozen`：状态行 + `code_sha`）；`timing` 不受限。`q1_run.py` 计入代码指纹。
  5. 汇总强制核验产物完整一致（`verify`：模型/种子/折/文件哈希/配置与代码指纹/集合覆盖）。
  6. 预注册与结果模板注明：`r_s` 训练—推理差异；分层与端到端训练方法不同，不能把胜负单独归因于网络结构。
  - 测试由 72 个增至 98 个（快速 94 + 集成 4）。按任务顺序逐个重放：每个任务实现前测试失败，实现后通过。
- 2026-10-02 Native 实施 Task 1–13（`dd4cea5`…`df242b8`，代码与本计划代码块逐字节一致），随后按终审（独立复审）修复，见 `6aa78fb`。**自此仓库代码在以下几处与本计划代码块不同，以仓库为准**：
  1. `run_q1_stage1.sh`：`xargs -r`；参照先于作业；全数据拟合逐个 `wait` 并检查退出码。
  2. `q1_train.CODE_FILES` 加入 `q1_report.py`（判据代码进冻结指纹）。
  3. `q1_windows.LEAD_BINS` 末组拆成 53–96 / 97–102（只报告）。
  4. `q1_run`：`prepare` 一次写好全数据传感器层；`fullfit` 只读并核对；新增 `load_sensor_all` / `load_fullfit`。`fig_pp2_q1` 改用 `load_fullfit`。
  5. 新测试：分层计分输出不读下一 chunk、参照合成端到端（slow）、全数据拟合产物核验、运行脚本空跑 4 例 —— 共 112 个。
  - 预注册草稿同步：新 `code_sha`、§4 交叉引用改为 §12、前瞻分组、§12-4 的 OLS 过拟合风险、全数据模型核验。
