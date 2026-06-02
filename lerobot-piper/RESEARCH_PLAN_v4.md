# Research Plan v4.3 — Sim-Based Pretraining + Context-Aware RefineNet

**Title**: Smoothness-Regularized Imitation Learning for Robot Manipulation via Sim-Based Denoising Pretraining and Context-Aware Action Refinement

**Author**: 郭鹏 (guo.peng.r6@s.mail.nagoya-u.ac.jp)
**Date**: 2026-05-27
**Status**: Active. Replaces all prior Phase 2 RL plans. To be implemented sequentially from Stage 0.

---

## §1 Project Principles (Non-Negotiable)

These three principles anchor the project against scope drift.

### Principle 1: Main contribution is RefineNet, not RL

Phase 2 real-world RL experiments (Round A/B/v4a/v4b/v4c) have been diagnosed as fundamentally constrained by (a) sample inefficiency (~240 episodes vs. PPO's typical 10⁶+ requirement) and (b) reward direction misalignment (symmetric L2 IMU reward penalizes "smoother than demo" trajectories).

Thesis main contribution is a **sim-based denoising pretraining + context-aware RefineNet** under a supervised learning paradigm.

**Stage 6 (minimal real-world RL fine-tune)** is a deferred optional, evaluated *after* Stage 5 completes. If Stage 5 demonstrates sufficient smoothness improvement, Stage 6 is written into Future Work and not implemented.

### Principle 2: ACT is frozen by default

Stage 3b and Stage 5 main pipeline keeps the existing C1-v3 ACT model **frozen** as an action generator. RefineNet is trained on top to produce a residual correction.

The only exception is **Stage 3a** (denoising ACT pretraining), which serves as a **negative ablation** to demonstrate that modifying ACT directly compromises task ability — thereby motivating the RefineNet (Stage 3b) design.

Stage 3a outputs are NOT used in Stage 5 main pipeline (only as ablation comparison).

### Principle 3: MuJoCo for training only, not deployment

MuJoCo simulation roles (strictly bounded):
- Training data generation (Stage 2)
- Sim-side ablation validation (Phase 1/2 sim deploy in Stages 3-4)
- Sim-side replay for consistency filter (Stage 5)

At **deployment time**, RefineNet uses real-robot IMU and history feedback only:

```
real_state    → ACT → ACT_output
real_history  ─┐
real_state    ─┴─→ RefineNet(ACT_output, real_history, real_state) → A_refined
A_refined → real robot execution
```

No MuJoCo runtime in the deployment loop.

---

## §2 Background (Brief)

Phases 0, 1, and the Phase 2 RL diagnostic study are completed. The current v4.3 plan starts from Phase 2's diagnostic outcomes.

### Existing baselines (do not modify)
- **C1-v3**: Phase 1 BC-trained ACT with autoregressive history input, deployed 4/5 task success on real robot. Used as frozen action generator for Stage 3b onward.
- **eval_C1_v3**: 5 deterministic deploy episodes from C1-v3, used as smoothness baseline.
- **eval_phase2_v3**: 5 deterministic deploy episodes from Phase 2 Round A (already shows jerk +84% regression vs C1-v3, motivating supervised pivot).

### Existing diagnostic figures (already in outputs/eval/)
- `phase2_v3_vs_C1_v3_with_gt/bar_chart_3way.png` — 4 IMU metrics, 3-way
- `phase2_v3_vs_C1_v3_with_gt/psd_3way.png` — PSD overlay
- `phase2_v3_vs_C1_v3_with_gt/per_episode_3way.png` — time-domain per-ep

Key data point: gt (demo) IMU >5Hz % is ~11%, while C1-v3 is 21.6% and Phase2-v3 is 27.8%. Demo is meaningfully smoother than current BC output → improvement headroom exists in supervised regime.

---

## §3 Pipeline Overview

```
Stage 0  Noise calibration from v3 demo data                          ~0.5 day
   ↓
Stage 1  MuJoCo sim setup (no object), inject calibrated noise        ~1 week
   ↓
Stage 2  Generate Phase 1 sim paired data (6-tuple)                   ~3-4 days
   ↓
Stage 3  (parallel)
   ├ Stage 3a  Denoising ACT pretraining (negative ablation)          ~3 days GPU
   └ Stage 3b  Vanilla RefineNet training (main path)                  ~3 days GPU
   ↓ (RefineNet validated on Phase 1 sim)
   ↓
Stage 4  Object-aware sim incremental (fine-tune, not full retrain)   ~1-2 weeks
   ↓
Stage 5  Real-world supervised fine-tune (consistency-filtered)       ~1 week
   ↓
Stage 6  [Deferred] Minimal real-world RL fine-tune                   ~3-5 days
         (Only if Stage 5 smoothness improvement < expected)
```

**Total core (Stage 0-5)**: 5-7 weeks
**With optional Stage 6**: 6-8 weeks

---

## §4 Design Decisions (T1-T5 Finalized)

### T1: Stage 5 consistency filter — Hybrid (task success + state distance)

For each real episode collected, replay the recorded action sequence in MuJoCo from the same start state:

```python
real_success, real_imu, real_state_traj = real_episode_record
sim_success, sim_imu, sim_state_traj    = mujoco.replay(start_state, action_chunk)

keep_episode = (real_success == sim_success) AND \
               (mean(|sim_state_traj - real_state_traj|) < 0.1 rad)
```

Filter retains cases A (both succeed) + D (both fail). Cases B (sim-only success) and C (real-only success) are discarded as sim-real mismatch.

### T2: Stage 3a/3b denoising loss — Conditional denoising + smoothness regularization

Training input: noisy observation (sim with injected noise).
Training target: A_clean (demo action).
Loss:

```
L_total = L1(predicted_action, A_clean) + λ × jerk_RMS(predicted_action)
```

λ sweep: {0.01, 0.1, 1.0} (initial), refine after first results.

### T3: RefineNet input — Context-aware, 4 inputs

**Positioning**: Context-aware residual refiner. Not a blind low-pass filter.

**Inputs** (all required, minimum context-aware set):
| Name | Shape | Source |
|---|---|---|
| `A_target` | (B, 100, 7) | ACT output (frozen ACT) |
| `A_prev` (history_action) | (B, 40, 7) | Previous episode action, patched 0.5s/token |
| `I_prev` (history_imu) | (B, 40, 6) | Previous episode IMU, patched 0.5s/token |
| `current_state` | (B, 7) | Current joint state |

**Excluded inputs**:
- Image (ACT already processes, redundant)
- ACT internal hidden state (requires ACT interface change)
- PhysicsModel-predicted IMU (out of scope for sim-based path)
- Short-term raw IMU (patched history sufficient for now; future work)

**Output**: `ΔA` (residual delta), `A_refined = ACT_output + ΔA`, chunk-level.

**Architecture**: Light transformer
- CNN tokenizer for each time-series input (0.5s patching, kernel/stride matching Phase A design)
- Linear projection for state token
- 2-layer encoder, 1-layer decoder, dim=256, 8 heads
- 100 learnable query embeddings (matching chunk_size)
- Output head: Linear(256, 7)
- ~2.8-5M parameters

**Ablation (S0-S3)** — optional, time permitting:
| Setting | Inputs | Purpose |
|---|---|---|
| S0 | ACT_output only | Blind smoother baseline |
| S1 | + IMU_history | Old design 1 (advisor sketch) |
| S2 | + history_action | + action context |
| S3 | + current_state | **Main method** |

If time-constrained, train S3 only with full data, and run S0/S1/S2 with reduced training steps as comparison.

### T4: A_label = A_clean + loss has smoothness reg

`A_label = A_clean` (from demo) — pure target.

Smoothness pressure comes from the **loss** (`+ λ × jerk_RMS(pred)`), not from preprocessing `A_clean`. This avoids destroying task-critical action details while still pushing for smoother outputs than the demo baseline.

### T5: Stage 6 deferred to post-Stage-5 decision

Decision tree after Stage 5:

```
if Stage 5 smoothness improvement > 30%:
    → Stage 6 → Future Work section, do not implement
    → Write paper

elif Stage 5 improvement in 10-30%:
    → Implement minimal Stage 6 (20-50 updates, α=0.001)
    → Validate complete supervised+RL paradigm

else (Stage 5 improvement < 10%):
    → Do NOT do Stage 6 (RL unlikely to help further)
    → Diagnose noise model / paradigm in paper Discussion
    → Write paper with honest negative result framing
```

---

## §5 Technical Specifications (Per Stage)

### Stage 0: Noise calibration from v3 demo data

**Goal**: Derive what v3 *can* calibrate vs what it *cannot*, with explicit honesty about data limits.

**Inputs**: v3 dataset (~30 episodes) at `~/.cache/huggingface/lerobot/local/piper_pick_place_imu_v3/`

**Status (2026-05-27)**: COMPLETE. Script `scripts/analyze_noise_from_v3.py`, outputs at `outputs/sim/`.

**Post-implementation findings (important)**: First-pass analysis revealed that v3 demo cannot directly calibrate several noise categories. `noise_params.json` is structured into four explicit categories:

1. **`calibrated_from_v3` — use directly**:
   - `gyro_sensor_noise_sigma` per channel (gravity-insensitive, robust): gx=0.071, gy=0.072, gz=0.033 rad/s
   - `demo_jerk_rms_baseline` = 113.7 (smoothness target for Phase 3 to beat)
   - `active_joints_indices` = [0,1,2,6] (joints 1,2,3, gripper — the rest are unused in v3 task)
   - `active_joint_velocity_p99` per active joint

2. **`measured_but_contaminated` — for reference, DO NOT use as noise injection**:
   - `accel_demo_residual_sigma` (ax=2.29, ay=0.56, az=1.00 m/s²): includes pose-dependent gravity projection across episodes + settling motion. Not true sensor noise.
   - `observable_low_freq_peaks_hz` (<15 Hz peaks in motion segments): reflects trajectory shape, not mechanical vibration.

3. **`not_calibrated_use_heuristic` — needs separate calibration or heuristic**:
   - True accel sensor noise → requires static calibration recording (IMU lifted off arm OR arm fully restrained, 1-3 min)
   - 30-100 Hz mechanical vibration → unobservable at fps=30 (Nyquist = 15 Hz). Either inject only ≤15 Hz heuristic vibration, or upgrade IMU logging rate.
   - j4/j5/j6 dynamics → use conservative copy from j2/j3 parameters

4. **`heuristic_defaults_for_stage1`**: conservative initial values that Stage 1 sim should refine by matching sim IMU PSD against real IMU PSD:
   - accel sensor σ = 0.1 m/s² (TSND151 datasheet typical)
   - Vibration sinusoid at 12 Hz (below Nyquist), amplitude_accel=0.05, amplitude_gyro=0.01
   - motor_noise_sigma_per_active_joint = 0.005 rad
   - overshoot coefficient = 0.02 (overshoot = c × velocity²)

**Stage 1 noise model statement**: "calibrated low-freq + bounded heuristic high-freq", not "pure data-driven noise".

**Outputs**:
- `outputs/sim/noise_params.json` — 4-category structured parameters
- `outputs/sim/action_stats.json` — demo jerk RMS baseline + action statistics
- `outputs/sim/noise_analysis.png` — diagnostic plots

**Future complementary calibration (out of Stage 0 scope, optional)**:
- Static IMU recording → true accel sensor noise σ
- Per-joint sinusoid sweep → per-joint motor delay, tracking α, overshoot, joint-specific IMU response (especially for j4/j5/j6)

### Stage 1: MuJoCo sim setup (no object)

**Goal**: Build a working MuJoCo simulation of the Piper arm with the calibrated noise model injected at both action and IMU layers.

**Components**:
1. Piper URDF/MJCF (already available per user confirmation).
2. Wrist-mounted IMU sensor (matches real TSND151 placement).
3. Camera placeholder (not used in Phase 1 sim, reserved for later).
4. Noise injection at three layers:
   - **Action layer**: `A_executed = A_commanded + motor_noise + overshoot(velocity) + control_delay`
   - **Sensor layer**: `IMU_measured = IMU_true + sensor_noise + bias_drift + vibration_coupling`
   - **Physical layer** (Phase 2 only, requires objects): contact force spikes

**Output**: `lerobot-piper/sim/piper_noisy_sim.py` — callable sim wrapper:
```python
sim = NoisyPiperSim(noise_params_path="outputs/sim/noise_params.json")
imu_trace, state_trace = sim.execute_action_chunk(start_state, action_chunk)
```

**Validation**: Run a demo action chunk in sim, compare IMU power spectrum to real IMU. Match should be qualitatively similar in 5-100 Hz band.

**Time**: ~1 week (sim setup + noise integration + validation).

### Stage 2: Generate Phase 1 sim paired data

**Goal**: Produce ~1000 episodes of 6-tuple paired data for Stage 3 training.

**Per-sample 6-tuple**:
| Field | Shape | Description |
|---|---|---|
| `A_clean` | (100, 7) | Demo or synthesized smooth action chunk |
| `IMU_clean` | (100, 6) | IMU from sim run without noise |
| `A_noisy` | (100, 7) | Action perturbed by noise model |
| `IMU_noisy` | (100, 6) | IMU from sim run with noise |
| `A_label` | (100, 7) | = A_clean (per T4 decision) |
| `labels` | dict | `{success: bool, phase: enum}` (Phase 1 sim mostly free motion) |

**Data sources**:
- 30 v3 demos as seed action sequences
- Random perturbations (start position, target offset) to expand diversity
- Aim for ~1000 episodes total

**Output**: `outputs/sim/phase1_paired_data.h5` (HDF5 for efficient loading).

**Time**: ~3-4 days (generation + serialization).

### Stage 3a: Denoising ACT pretraining (negative ablation)

**Goal**: Show that fine-tuning ACT itself on denoising objective compromises task ability, motivating Stage 3b's frozen-ACT + RefineNet design.

**Procedure**:
- Load C1-v3 ACT.
- Continue training on Stage 2 data with conditional denoising loss (T2 spec).
- Train for ~20K steps (much less than C1-v3's 70K, since we're fine-tuning).
- Output: `C2-denoising-act` ckpt.

**Validation**:
- Deploy on Phase 1 sim (5 ep), measure task success rate and smoothness.
- Compare against C1-v3 baseline.
- Expected: smoothness improves, task success drops noticeably.

**Output**: `outputs/train/C2_denoising_act/` ckpt + sim deploy results.

**Time**: ~3 days GPU.

### Stage 3b: Vanilla RefineNet training (main path)

**Goal**: Train RefineNet (T3 spec) on top of frozen C1-v3 ACT to produce smoother outputs without modifying ACT.

**Architecture**: T3 spec (light transformer, 4 inputs, residual output).

**Training procedure**:
```python
ACT = load_frozen(C1_v3_ckpt)
RefineNet = init(...)

for batch in dataloader(stage2_data):
    state         = batch["current_state"]
    action_demo   = batch["A_clean"]                # supervision target
    imu_history   = batch["history_imu"]
    action_history = batch["history_action"]

    with no_grad():
        act_output = ACT(state, image, history)     # frozen ACT
    
    delta = RefineNet(act_output, action_history, imu_history, state)
    refined = act_output + delta
    
    l_bc = L1(refined, action_demo)
    l_smooth = jerk_RMS(refined)
    loss = l_bc + λ × l_smooth
    
    loss.backward()
    optimizer.step()
```

**Validation**:
- Deploy on Phase 1 sim, compare against C1-v3 (no refine) and C2-denoising-act (Stage 3a).
- Measure smoothness metrics and task success.
- Expected: smoothness improved, task success preserved (since ACT frozen).

**Output**: `outputs/train/refine_net_phase1/` ckpt + sim deploy results.

**Time**: ~3 days GPU.

### Stage 4: Object-aware sim incremental（2026-05-30 重构：transfer-audit 先行）

**重构动机**（见 STAGE3B_RESULTS.md 真机 de-risk + 闭环 eval）：
论文指标是真机 IMU hf，但 RefineNet 至今只优化 action hf，两者 transfer 未知。
object-aware sim 对**训练**的增量取决于 action→IMU transfer 是否成立 → 先测，再定 loss 路线。

**Stage 4.0 — Object-aware transfer audit（先做，与 loss 路线无关）**
- 建 object-aware sim：cube + table + 接触参数。
- 接触标定：对齐真机抓取 IMU spike（**靶**：accel 峰值 ~7.5 / ratio 2.5×，gyro 峰值 ~0.8 / ratio 4–6×；
  baseline accel ~3.0、gyro ~0.1–0.24，来自 eval_C1_v3 + demo）。
- **2×2×2 IMU 审计**：{no-object, object} × {clean, noisy} × {ACT, refined+blend}，度量 hp_rms(IMU)。
  分离：接触贡献多少 IMU 高频 / action 平滑消掉多少 / 噪声层淹没多少。
- 关键判据：
  - `ΔIMU_clean = hp_rms_clean_IMU(ACT) − hp_rms_clean_IMU(refined+blend)`
  - `ΔIMU_noisy = hp_rms_noisy_IMU(ACT) − hp_rms_noisy_IMU(refined+blend)`
  - clean 明显降 → transfer 成立，L0/L1 可能够；
  - clean 降但 noisy 不降 → 被 noise floor 淹没，重点转 Stage 5；
  - clean 不降 → action 与 IMU 解耦，才需认真上 L2 surrogate。

**Stage 4.1 — 据 audit 定 loss 路线**：L0（action loss，IMU 仅验证）/ L1（可微运动学链 gyro≈J_rot(q)·q̇）/ L2（可微 action→IMU surrogate，sim 产训练对）。

**Stage 4.2 — fine-tune RefineNet**：从 λ_hf=0.10 ckpt fine-tune（~半步数、低 lr）；transfer 成立后把 L_boundary（跨 chunk 续接）加进来（在此之前部署继续用 deploy-blend）。

**已确定的物理/几何参数**：
- gripper：joint7/8 slide，每指行程 ctrlrange 0–0.0475m（关节限位 0.035m），最大开合 ~0.07–0.095m → cube 宽度须 < ~0.07m。
- **待实测**：cube 尺寸/质量（尺+秤）、table 高度（或在 sim 里调到 demo gripper 接触高度一致）。
- 接触初值：restitution 0.0–0.1，friction 0.8–1.2 起步，再用抓取 IMU spike 标定。

**Output**: `outputs/train/refine_net_phase2/` ckpt + 2×2×2 audit + sim deploy results。

**Time**: ~1-2 weeks。

### Stage 5: Real-world supervised fine-tune

**Goal**: Bridge sim-to-real gap using a small amount of real-robot data filtered for sim-real consistency.

**Sub-stages**:
- **5.1**: Record 30-50 real episodes using C1-v3 ACT + Stage 4 RefineNet (deterministic deploy).
- **5.2**: For each recorded episode, replay action sequence in MuJoCo from same start state. Apply T1 hybrid consistency filter (task success match + state trajectory distance threshold).
- **5.3**: Fine-tune RefineNet on filtered data with same loss as Stage 3b/4.
- **5.4**: Deploy final ckpt for 5 ep deterministic evaluation.

**Output**: `outputs/train/refine_net_final/` ckpt + real-robot deploy data + main result figures.

**Time**: ~1 week (real-robot time ~0.5 day, rest is data processing + training).

### Stage 6 (deferred): Minimal real-world RL fine-tune

Only implement if Stage 5 results indicate need (per T5 decision tree).

**If implemented**:
- Use Stage 5 final ckpt as starting point.
- ~50 PPO updates, α=0.001 (very small RL signal).
- Update RefineNet parameters only (ACT still frozen).
- ~3-5 days real-robot time.

---

## §6 Time Budget

| Stage | Time | Cumulative |
|---|---|---|
| Stage 0 | 0.5 day | 0.5 day |
| Stage 1 | 1 week | ~1.5 weeks |
| Stage 2 | 3-4 days | ~2 weeks |
| Stage 3a + 3b (parallel) | 1 week | ~3 weeks |
| Stage 4 | 1-2 weeks | ~5 weeks |
| Stage 5 | 1 week | ~6 weeks |
| Paper writing (parallel with Stage 5+) | 2-3 weeks | ~8-9 weeks |
| Stage 6 (if needed) | 3-5 days | ~9-10 weeks |

**Realistic total**: 8-10 weeks to submission-ready thesis.

---

## §7 Risk Register

| Risk | Probability | Mitigation |
|---|---|---|
| MuJoCo Piper URDF has joint dynamics mismatch | Medium | Calibrate against real motor response in Stage 1 validation |
| Noise calibration over-/under-estimates real noise | Medium | Compare sim IMU PSD vs real IMU PSD; iterate noise params |
| Stage 3a denoising ACT actually works well (kills negative ablation narrative) | Low | This would still be a positive result; pivot to dual contribution |
| Stage 3b RefineNet shows no smoothness improvement | Medium | Reflect on noise model, paradigm; if root cause unclear, write as diagnostic finding |
| Stage 4 object physics too inaccurate | Medium | Limit contact phase training; rely on Stage 5 real-world fine-tune to bridge |
| Stage 5 consistency filter retains too few samples | Medium | Loosen threshold; or record more real episodes |
| MuJoCo + Piper URDF setup takes longer than 1 week | High | Time-box Stage 1; if blocked, fall back to simpler 1-DOF or pre-built robot URDF for proof-of-concept |
| Real-robot hardware failure during Stage 5 recording | Low | Use existing eval_C1_v3 / eval_phase2_v3 data as fallback baseline |

---

## §8 Future Work

Items explicitly excluded from main thesis but mentioned in §7 of the paper:

- **Diffusion / flow-matching refinement** (replacing vanilla transformer RefineNet for stronger smoothness prior)
- **ACT internal hidden state as RefineNet input** (skipping 7-dim bottleneck; requires ACT architecture modification)
- **Sim-to-real with domain randomization** (training RefineNet on randomized sim params for stronger transfer)
- **Multi-task extension** (RefineNet on diverse manipulation tasks beyond pick-and-place)
- **Differential history representation** (velocity/jerk as inputs vs absolute values; some of this is implicit in CNN tokenizer learning)
- **Weight tying between ACT input/output projections**
- **Real-world RL fine-tuning at scale** (if Stage 6 is implemented as proof-of-concept, scaling up is future work)

---

## §9 Open TODOs (Updated After Each Stage)

### 下一步:恢复研究时优先 — RefineNet 输入消融 S0–S3（2026-06-03 登记）
**动机**：history_action(600×7)+history_imu(600×6) 是 RefineNet 最大、最重的输入，但间接证据显示其作用可能不大——闭环 eval 中「真机 history」vs「闭环 refined history」输出几乎一致（hf 0.0747 vs 0.0762，对 history 内容不敏感）；且 action↔IMU 耦合弱，history_imu 贡献存疑。验证是否可去掉/精简（更小网络、部署不依赖 history_buffer、论文更干净）。
**方法**：训练 4 个变体（各 ~3min，复用 `sim/train_refine_net.py` + paired data），用统一指标对比（chunk 内 hf_ratio + L_BC 护栏 + 闭环 hp_rms + 边界段）：
- [ ] **S0** 仅 ACT_output（盲精修 baseline）
- [ ] **S1** + current_state
- [ ] **S2** + 短近期 action history（只取最近 ~10–20 帧，针对 chunk 边界连续性）
- [ ] **S3** + 完整 history(action+imu 600)（= 现状，best λ_hf=0.10 ep68）
**判读**：若 S0/S1 已拿到大部分 chunk 内收益 → history 可砍；若 S2 改善边界 → 保留短 history；若只有 S3 好 → 完整 history 值这个复杂度。先验：完整 history(尤其 imu)可能不值复杂度。
**实现注**：需给 `RefineNet.__init__` 加输入开关(use_history_action/use_history_imu/use_state)+ tokenizer 条件构建；`train_refine_net.py` 加 `--variant S0..S3`。

### 其它候选下一步（Stage 5 之后，待定优先级）
- [ ] 诊断低层控制器跟踪误差（action vs state 残差谱）——传递链最大瓶颈在 action→state（丢约一半）
- [ ] 训练期加边界续接损失 `L_boundary`（真机 boundary 段 refined 仍略差）
- [ ] 真机 A/B 补录至 n≈15–20，使 full-episode IMU accel 也达显著（现仅 carry 段显著）
- [ ] 论文写作（结论候选见 STAGE5_RESULTS.md 末尾）

### 历史（Stage 0 之前，已完成）
- [x] Stage 0: 噪声标定脚本 + noise_params.json
- [x] Stage 1: MuJoCo 结构 + 噪声注入 + IMU 轴标定
- [x] Stage 2/2.5/2.6: 成对数据 + ACT_output/history 预计算
- [x] Stage 3a 负消融 / Stage 3b RefineNet + λ 扫 + 闭环 eval
- [x] Stage 4.0: object-aware(胶带圆环 weld proxy) + transfer audit
- [x] Stage 5: 真机部署 wrapper + A/B + transfer-chain 诊断
  （各阶段详见 STAGE3B_RESULTS.md / STAGE4_RESULTS.md / STAGE5_RESULTS.md）

---

## §10 Document History

| Version | Date | Change |
|---|---|---|
| v1 | 2026-04 | Original plan (Phase 0 → 1 → RL) |
| v2 | 2026-05-09 | Phase 2 RL experiments running |
| v3 | 2026-05-22 | After Phase 2 RL failures, proposed supervised pivot |
| **v4.3** | **2026-05-27** | **Sim-based denoising + RefineNet, T1-T5 finalized, 3 principles** |
| v4.3a | 2026-05-29 | Stage 3b Run #1 (λ=0.1 baseline) 跑完：负结果，无平滑增益、平滑项过拟合。详见 `STAGE3B_RESULTS.md` |
| v4.3b | 2026-05-30 | Stage 3b 收尾：指标改 hf_fraction/高通RMS，Run B(hf+delta) chunk内0.28×，闭环揭示边界瓶颈+deploy-blend修复，真机 de-risk(IMU hf +48.7%)。Stage 4 重构为 transfer-audit 先行 |
