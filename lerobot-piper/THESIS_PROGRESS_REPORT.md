# Piper ACT Smoothness Thesis — Progress Report

**Date**: 2026-05-01
**Status**: Phase 1 ✅ Complete (5/5 task success on real robot) | Phase 2 design 🔒 Locked, ready to implement

---

## 1. Project Goal

老师 brief: **改良 ACT 架构, 使其在 pick-and-place 任务上输出更平滑的动作**.

Smoothness measured by:
- IMU accel/gyro RMS during execution (real robot signal)
- IMU >5 Hz energy fraction (high-freq noise)
- Action chunk jerk RMS (3rd derivative)
- Without sacrificing task success

Hardware: **Agilex Piper 6-DoF + gripper**, **TSND151 IMU** mounted near end-effector.

---

## 2. Timeline Overview

```
═══════════════════════════════════════════════════════════════════
Phase            Status       Period             Outcome
═══════════════════════════════════════════════════════════════════
Phase 0           ✅           Apr 7-15           SAH/ΦSim diagnosis
Phase 1 (BC)      ✅           Apr 16-30          5/5 task success
Phase 2 (RL)      🔒 design   May 1-?            Spec locked, in impl
═══════════════════════════════════════════════════════════════════
```

---

## 3. Phase 0 — Diagnostic findings (abbreviated)

Key established facts that drove method choice:

- **GT demo action 95% bandwidth = 0.6 Hz** → very low-bandwidth task
- **ACT vanilla injects 13.6× more >5 Hz energy** than GT demos
- **Static IMU noise floor ≈ 6%** of motion RMS → ~94% headroom for smoothness improvement
- **Gripper bandwidth = 2.7 Hz** > arm joints' 0.6 Hz (different timescale per dim)

These motivated initial **SAH (Spectral Action Head)** approach using B-spline basis.

---

## 4. SAH approach — Initial attempt, validated then rejected

### 4.1 Method
Replace per-step Linear action head with K=5 B-spline basis function coefficients. Architecturally bounds chunk's bandwidth → eliminates HF noise injection.

### 4.2 Real-robot results (4-way comparison, P1-v2 dataset)

| Method | Task success | Accel RMS | Gyro >5Hz % | Action jerk |
|---|---|---|---|---|
| Vanilla | 5/5 | 4.375 | 17.88% | 601 |
| α-imu-only | 5/5 | 4.518 | 14.70% | 800 |
| SAH-only (3 seeds) | **1/15** | 3.957 ± 0.22 | 10.03 ± 1.4% | 1222 ± 140 |
| SAH+TE (3 seeds) | **0/15** | 3.406 ± 0.22 | 11.09 ± 2.7% | 1437 ± 224 |

### 4.3 Why rejected (老师 meeting decision)

```
SAH 强行限制 chunk 内带宽 → IMU smoothness 改善 22-45%
但 chunk 边界 jerk 暴涨 100-140% → task 完全崩
```

老师建议改方向: **autoregressive ACT + real-robot RL with IMU smoothness loss**.

---

## 5. Phase 1 — Autoregressive ACT (BC pre-training) ✅

### 5.1 Architectural change

ACT input augmented with previous episode's full action and IMU sequences:

```
Original ACT input:           Modified ACT input:
  - images_t                    - images_t                    (unchanged)
  - state_t                     - state_t                     (unchanged)
                                + a.seq(T-1)   [L_max=600, 7] (NEW)
                                + r.IMU(T-1)   [L_max=600, 6] (NEW)
                                + history_pad_mask [L_max]    (NEW)
```

Encoding:
- `a.seq(T-1)` → `Linear(7→512)` + learned positional embedding
- `r.IMU(T-1)` → `Linear(6→512)` + learned positional embedding
- Both fused at encoder input as additional tokens (no pooling, full sequence)
- Right-aligned padding: newest action always at index L-1

Total ACT params: 51.6M → 52.0M (+0.4M for history projections + pos embeds).

### 5.2 Training

| Item | Value |
|---|---|
| Dataset | `local/piper_pick_place_imu_v2` (30 episodes, IMU-clean) |
| Steps | 100,000 |
| Batch size | 16 |
| L_max | 600 (covers v2 ep length 574-589) |
| GPU | RTX 3090, peak 18.5 GB |
| Total time | ~12 hours |
| WandB project | `lerobot-piper-history` |
| Checkpoint | `outputs/train/2026-04-29/C1_v2_history_seed42/checkpoints/100000/` |

Final loss:
- l1_loss = 0.019 (vs vanilla v2 final ~0.034)
- kld_loss = 0.000 (VAE collapsed)
- grad_norm 102 → 2.5 (smooth descent)

### 5.3 Sanity checks (B4, B5)

- **B4** (5K steps, L=400): l1 dropped 0.32 → 0.075. Pipeline OK.
- **B4-v2** (5K steps, L=600 + right-align): l1 dropped 0.26 → 0.10. Confirmed L=600 + right-align fix.
- **B5** (5K steps, IMU-off ablation): l1 dropped to 0.124 (vs 0.100 with IMU). IMU has marginal contribution; kept on per teacher's plan.

### 5.4 Real-robot deployment (C3)

5 deployment episodes on real Piper, all task success.

```
C3 method                    Task   Action jerk   Action >5Hz   IMU >5Hz
Vanilla baseline             5/5    601           0.031%        15.78%
α-imu-only                   5/5    801 (worse)   0.070%        10.61%
SAH-only (3 seeds)           1/15   1222          0.118%        8.83%
SAH+TE (3 seeds)             0/15   1437          0.345%        8.63%
C1-history (this work)       5/5    568 ⭐ best   0.026% ⭐    16.16%
```

**C1-history is the only method that improves Action smoothness over Vanilla while keeping 5/5 task success.**
SAH variants achieved IMU smoothness gain at the cost of action jerk + task failure — fundamentally broken.

### 5.5 Engineering deliverables

- `lerobot/common/policies/act/configuration_act.py` — added history config fields
- `lerobot/common/policies/act/modeling_act.py` — encoder-input history token injection
- `lerobot/common/datasets/history_wrapper.py` — episode-pair sampling + IMU rolling-mean smoothing
- `lerobot/common/datasets/factory.py` — auto-wrap when `enable_history_input=True`
- `lerobot/common/robot_devices/history_buffer.py` — runtime history buffer for deployment
- `lerobot/common/robot_devices/control_utils.py` — control loop history injection
- `lerobot/scripts/control_robot.py` — buffer instantiation in main loop

All changes are backward-compatible (default `enable_history_input=False` reproduces vanilla ACT exactly).

---

## 6. Phase 2 — Real-robot RL fine-tuning 🔒 Spec locked

### 6.1 Goal

Reduce IMU >5 Hz (currently 16% accel / 20% gyro) while maintaining 5/5 task success.
Target: bring IMU >5 Hz down to SAH+TE level (~9-11%) without losing task.

### 6.2 Architecture

Reuse C1 ckpt as starting point. Add:
- **Value head** `Linear(512 → 1)` on encoder latent token (`detach`-ed to prevent gradient pollution of encoder)
- **Stochastic policy**: deterministic ACT output + fixed-σ Gaussian noise, no σ-head added
- **History buffer** (already implemented in Phase 1 deployment)

### 6.3 Reward (D2)

```
R_t = -||real_IMU_t - GT_IMU_t||²  + 𝟙[task_success at last step] × 10
```

GT_IMU comes from a state-NN against v2 demos:
- Build KDTree over normalized v2 state vectors (joint angles)
- At each chunk start, query NN → use that demo's next 100 IMU frames as chunk GT
- Threshold τ = max(v2_p95_intra-NN, C1-deploy_mean+3σ)
- If NN distance > τ → mark chunk as OOD, mask out from RL loss (BC still applies)

Reward post-processing:
- EMA standardize → unit variance
- Clip to [-3, +3] for PPO stability

### 6.4 Trajectory format (D3)

- **chunk-as-trajectory**: each ACT forward = one macro-action
- T = 6 chunks per 20s episode
- R_chunk = mean of per-step rewards × valid_mask
- log_prob factorized over (100, 7) Gaussian
- GAE: γ_inter=0.99, γ_intra=1.0, λ=0.95

### 6.5 RL algorithm (D4)

PPO selected over SAC.
- Clip ratio ε = 0.2, K_epochs = 4, K_episodes_per_update = 4
- Stochastic σ schedule: 0.10 → 0.05 → 0.02 (linear decay over 50K updates)
- Per-dim σ: joints = 0.05, gripper = 0.20 (normalized space)
- Critic warmup: 20 deterministic episodes + 2000 GPU steps with frozen actor

### 6.6 Loss coordination (D5)

```
L_total = L_BC + α × L_RL + 0.5 × L_V
```
- α schedule: Stage 1 (0-100 updates) = 0.01, Stage 2 (100-300) = 0.05, Stage 3 (300+) = 0.10
- L_V uses `encoder.detach()` so V loss doesn't pollute encoder
- BC data: 100% offline v2 (24 chunks per mini-batch)
- KL early-stop in PPO (target_KL = 0.05) instead of explicit KL penalty

### 6.7 Safety + monitoring (D6)

Four safety layers:
1. Action clamp: clip to v2 range × 1.2 margin
2. Velocity limit: |Δaction| ≤ 0.05 rad/step
3. Chunk-boundary smoothing: 5-step linear interpolation of μ across chunk boundaries (ε independent per chunk)
4. Emergency stop on: ESC key, hardware limit, IMU NaN, 3-consecutive failed episodes

`valid_mask` propagation (3 layers):
- Reward replacement at invalid steps (bootstrap value, makes Δ=0)
- Advantage explicit zero
- PPO + V loss use valid-step mean (not total-divided)

Eval gate every 25 updates, 5 deterministic episodes:
- 5/5 → continue
- 4/5 → warn, reduce α 25%, σ tighten
- 3/5 → auto rollback to previous ckpt, reduce α 50%
- ≤2/5 → terminate training, manual diagnosis

Monitoring (6 categories):
- **Drift detection** ⭐ leading indicator: L_BC absolute value (warn if > 2× C1 baseline = 0.038, alert if > 3× = 0.057)
- Performance: task success, episode length
- RL signals: mean_reward, mean_advantage, V_loss, V_explained_variance
- Stability: KL(new||old), ratio_clip_frac
- Safety: ood_skip_rate, emergency_stops, nan_actions
- Hyperparams: α, σ schedule trace

### 6.8 Budget

- Critic warmup: ~15 min real robot + 5 min GPU
- Main loop: 100 updates × 4 episodes = 400 episodes
- Eval gates (4×): 20 episodes
- Buffer: 30 episodes
- **Total: ~462 real-robot episodes ≈ 6 hours rollout, ~1 week part-time**

---

## 7. Risks & contingencies

### Identified failure modes (with response plan)

| Failure | Likely cause | Response |
|---|---|---|
| Task success drops below 4/5 | α too large or σ too aggressive | Halve α, tighten σ |
| ood_skip_rate > 50% | Exploration push policy OOD | Reduce σ, restart from earlier ckpt |
| L_BC drifts > 2× C1 | Actor diverging from demo distribution | Increase α weight on BC, add KL penalty β=0.01 |
| Reward stuck (no improvement 100 steps) | V function broken | Re-run critic warmup, check V_explained_variance |
| NaN reward / Inf advantage | IMU outlier or numeric instability | Tighten clip range, check IMU hardware |
| Gripper σ shakes object loose | σ_gripper too large | Halve σ_gripper |

### Fallback paths

If RL fundamentally fails:
1. **Increase α to make BC dominate** → degenerate to "BC + tiny RL" (still might marginal-improve)
2. **Train ΦSim DNN proxy** → switch to differentiable supervised approach (老师's stated future work)
3. **Inference-time IMU filter** → post-process action chunks at deployment, no RL

---

## 8. Future work / open items

### Hardware / data
- [ ] v3 dataset collection (50 episodes, planned but not started; v2 was sufficient for Phase 1)
- [ ] IMU mount fixture + photo documentation (for v3 reproducibility)

### Model / method
- [ ] Phase 2 implementation (codebase changes outlined in §6)
- [ ] Sim-to-real IMU calibration (deploy_mean drift correction if needed)
- [ ] Ablation: v3 retrain Phase 1 + Phase 2 to compare

### Paper writing
- [ ] Method section (autoregressive ACT + RL fine-tuning)
- [ ] Experiments: 5-way real-robot comparison (already collected)
- [ ] Ablations: with/without history input, with/without RL fine-tune

---

## 9. Key references (codebase)

| File | Purpose |
|---|---|
| `lerobot/common/policies/act/configuration_act.py` | ACT config (history fields added) |
| `lerobot/common/policies/act/modeling_act.py` | ACT model (history input branches) |
| `lerobot/common/datasets/history_wrapper.py` | Episode-history dataset wrapper |
| `lerobot/common/robot_devices/history_buffer.py` | Runtime history buffer |
| `scripts/c2_offline_eval.py` | Offline eval (C1 vs vanilla) |
| `scripts/c3_5way_realrobot_compare.py` | 5-way real-robot comparison |
| `outputs/eval/c3_5way/c3_5way_realrobot.png` | 5-way bar chart |
| `outputs/eval/c3_5way/c3_5way_psd.png` | 5-way PSD comparison |
| `outputs/train/2026-04-29/C1_v2_history_seed42/` | Phase 1 final ckpt (100K) |

---

## 10. Decision log (key turning points)

| Date | Decision | Reason |
|---|---|---|
| Apr 22 | Adopt SAH approach | Phase 0 showed 0.6 Hz GT bandwidth, K=5 sufficient |
| Apr 28 | Reject SAH | Real robot 1/15 task success — chunk-boundary jerk explosion |
| Apr 29 | Adopt 老师 plan: autoregressive ACT + RL | Decoupled smoothness from architectural bandwidth limits |
| Apr 29 | C1 trained on v2 (not v3) | v2 had IMU; saved 2 hours of data collection |
| Apr 30 | C3 5/5 task success | Phase 1 validated, unlock Phase 2 |
| May 1 | Phase 2 spec D2-D6 locked | Ready for implementation |

---

**Status (as of 2026-05-01)**: Phase 2 implementation begins next.
