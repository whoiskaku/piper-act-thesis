# Phase 2 Deployment Checklist

**Purpose**: pre-flight protocol before running `phase2_train.py` on real Piper hardware.

**Estimated total time**: ~30 min checklist + ~6 hours training.

---

## ◇ Section 1: Hardware sanity (10 min)

### 1.1 Robot

- [ ] Piper arm powered on, no error LEDs
- [ ] Arm at home position (joint_4 = 0 confirmed visually — gamepad bug zone)
- [ ] CAN interface up:
  ```bash
  bash piper_scripts/can_activate.sh
  ip link show can0   # should be UP
  ```
- [ ] Operator emergency stop button reachable (physical)

### 1.2 IMU (TSND151)

- [ ] TSND151 mounted at the SAME position as v2 dataset collection (photo reference)
- [ ] USB cable connected: `ls /dev/ttyACM0` exists
- [ ] User has read permission on `/dev/ttyACM0`:
  ```bash
  ls -la /dev/ttyACM0     # should be group-readable
  groups | grep dialout    # user should be in dialout
  ```
- [ ] IMU not loose, not knocked askew

### 1.3 Cameras

- [ ] `cam_two` (480×640) connected and visible:
  ```bash
  ls /dev/video*
  ```
- [ ] **Critical**: cam_one (1080×1920) is the one we exclude from training. Verify the model was loaded with `exclude_input_keys=["observation.images.one"]` in `train_config.json`.

### 1.4 Workspace

- [ ] Pick-and-place objects placed at training distribution positions
- [ ] No new objects in workspace that weren't in v2 demos
- [ ] Lighting matches v2 collection (cameras output exposure stable)

---

## ◇ Section 2: Software prerequisites (5 min)

### 2.1 Required artifacts

- [ ] **C1 ckpt** present and not corrupted:
  ```bash
  ls /home/enolab-guo/claude-test-2/outputs/train/2026-04-29/C1_v2_history_seed42/checkpoints/100000/pretrained_model/
  # should show: config.json, model.safetensors (~200 MB), train_config.json
  ```

- [ ] **IMURewardComputer pickle** built:
  ```bash
  ls -la outputs/eval/d2_imu_reward_computer.pkl     # should be ~36 MB
  ```
  
  If missing, regenerate:
  ```bash
  cd /home/enolab-guo/claude-test-2/lerobot-piper
  python scripts/d2_setup_imu_reward.py
  ```

- [ ] **v2 dataset cached**:
  ```bash
  ls ~/.cache/huggingface/lerobot/local/piper_pick_place_imu_v2/data/chunk-000/ | wc -l
  # should be 30
  ```

- [ ] **Phase 2 unit tests pass**:
  ```bash
  python scripts/test_modeling_act_phase2.py    # 7/7 PASS expected
  python scripts/test_phase2_actor_critic.py     # 12/12 PASS expected
  ```

### 2.2 Environment

- [ ] CUDA available:
  ```bash
  /home/enolab-guo/miniconda3/envs/lerobot/bin/python -c "import torch; print(torch.cuda.is_available())"
  # → True
  ```

- [ ] WandB logged in (if `--phase2.wandb_enable=true`):
  ```bash
  wandb login
  # or set WANDB_API_KEY env var
  ```

- [ ] Free disk space ≥ 10 GB (ckpts + logs):
  ```bash
  df -h /home/enolab-guo/claude-test-2
  ```

---

## ◇ Section 3: Smoke test before real run (5 min)

### 3.1 Code path dry-run (no hardware needed)

```bash
cd /home/enolab-guo/claude-test-2/lerobot-piper

python -m lerobot.scripts.phase2_train \
    --dry_run=true \
    --phase2.wandb_enable=false \
    --phase2.warmup_episodes=2 \
    --phase2.warmup_gpu_steps=10 \
    --phase2.k_episodes_per_update=1 \
    --phase2.n_updates=1 \
    --phase2.episode_time_s=10 \
    --phase2.minibatch_size_chunks=2
```

Expected: pipeline completes ~1.5 min, final ckpt saved. Press `n` (or just enter) at the "Task success?" prompts since MockRobot output is synthetic.

- [ ] Dry-run reaches "Phase 2 training complete" without traceback
- [ ] Output dir has `checkpoints/final/pretrained_model/model.safetensors`

If this fails, **DO NOT proceed to real robot**. Fix code first.

### 3.2 GPU memory headroom check

```bash
nvidia-smi
```

- [ ] At least 20 GB free on RTX 3090 before launching (Phase 2 peaks ~12 GB during forward+backward)
- [ ] No other CUDA processes consuming memory
- [ ] GPU temperature < 70°C at idle

---

## ◇ Section 4: First-run protocol (CAUTION mode)

**Run this BEFORE the full 100-update training to verify real-robot integration.**

### 4.1 Mini training run (10 episodes total, ~15 min)

```bash
python -m lerobot.scripts.phase2_train \
    --robot.type=piper \
    --robot.inference_time=true \
    --robot.enable_imu=true \
    --robot.tsnd_device=/dev/ttyACM0 \
    --robot.tsnd_interval_ms=10 \
    --phase2.output_dir=outputs/train/phase2_smoketest \
    --phase2.wandb_enable=true \
    --phase2.wandb_project=lerobot-piper-phase2-smoketest \
    --phase2.warmup_episodes=4 \
    --phase2.warmup_gpu_steps=500 \
    --phase2.n_updates=2 \
    --phase2.k_episodes_per_update=2 \
    --phase2.eval_freq=100
```

**During this run, watch for**:

- [ ] First few warmup rollouts: arm moves smoothly, no jitter (σ=0 means deterministic)
- [ ] After warmup, σ kicks in. Watch joint_4 specifically — gamepad bug history zone
- [ ] At task success prompt: type `y` if pick-and-place succeeded, `n` if failed
- [ ] No emergency stops triggered
- [ ] No NaN warnings in log

### 4.2 Verify Phase 2 output

After smoketest:

- [ ] Final ckpt saved: `outputs/train/phase2_smoketest/checkpoints/final/`
- [ ] WandB run shows reward / KL / drift curves (visit dashboard)
- [ ] L_BC / C1 baseline ratio < 1.5 (if > 2.0, something's wrong)

If smoketest passes ALL checks, proceed to full run. Else stop and debug.

---

## ◇ Section 5: Full Phase 2 run

### 5.1 Launch command

```bash
python -m lerobot.scripts.phase2_train \
    --robot.type=piper \
    --robot.inference_time=true \
    --robot.enable_imu=true \
    --robot.tsnd_device=/dev/ttyACM0 \
    --robot.tsnd_interval_ms=10 \
    --phase2.output_dir=outputs/train/phase2_v1 \
    --phase2.wandb_project=lerobot-piper-phase2 \
    --phase2.n_updates=100
```

Estimated time: ~6 hours of pure rollout, real wall-clock ~1 week part-time.

### 5.2 Per-update operator action

The script prompts after each episode: `Task success? [y/N]`. Operator enters `y` or `n` based on whether pick-and-place completed.

**Tip**: keep terminal in foreground; don't background the process (manual prompts).

---

## ◇ Section 6: Live monitoring (during training)

### 6.1 Five must-watch metrics on WandB

| Metric | Healthy range | Action if violated |
|---|---|---|
| `task_success_rate` | ≥ 0.6 | Track over 10 updates; if persistently low, rollback |
| `L_BC_ratio_to_C1` | < 2.0 (warn), < 3.0 (alert) | If > 3.0: pause, reduce α, restart |
| `kl_new_old` | < 0.10 (early stop kicks in at 0.05) | If consistently > 0.10: K_epoch may be too high |
| `ood_skip_rate` | < 0.30 | If > 0.50: σ exploration too aggressive, reduce |
| `ratio_clip_frac` | < 0.50 | If > 0.70: rollout-update gap too large, reduce K_episodes |

### 6.2 Eval gate (every 25 updates)

The script auto-runs 5 deterministic episodes every 25 updates. Watch the output:
- 5/5 → great, continue
- 4/5 → warn, no auto action; operator decides
- 3/5 → script logs "ROLLBACK" warning; operator should pause and reload prior ckpt
- ≤2/5 → script auto-terminates

### 6.3 Anomaly response cheat-sheet

```
KL > 0.5:                Pause; reduce α by half
Reward stuck 50 steps:   Verify GT IMU normalization, restart V head warmup
NaN action / IMU:        Stop, check TSND USB connection
Operator stop pressed:   Save current ckpt, address cause, resume from ckpt
3 consecutive failures:  Script auto-pauses; operator reviews
```

---

## ◇ Section 7: After training completes

### 7.1 Final eval

```bash
# Run real-robot deployment 5× with the Phase 2 final ckpt
python -m lerobot.scripts.control_robot \
    --robot.type=piper \
    --robot.inference_time=true \
    --robot.enable_imu=true \
    --robot.tsnd_device=/dev/ttyACM0 \
    --robot.tsnd_interval_ms=10 \
    --control.type=record \
    --control.repo_id=local/eval_phase2_final \
    --control.num_episodes=5 \
    --control.fps=30 \
    --control.episode_time_s=20 \
    --control.warmup_time_s=5 \
    --control.reset_time_s=10 \
    --control.policy.path=outputs/train/phase2_v1/checkpoints/final/pretrained_model \
    --control.single_task="pick_and_place" \
    --control.display_cameras=true
```

### 7.2 Compare against C1 baseline

Reuse the 5-way comparison script:

```bash
python scripts/c3_5way_realrobot_compare.py
```

Edit the script to add `eval_phase2_final` as a 6th method. Targets:

- [ ] Task success rate: ≥ 5/5 (no regression vs C1)
- [ ] IMU >5 Hz energy: ≤ 12% (was 16% on C1; SAH+TE was 9% but failed task)
- [ ] Action jerk RMS: ≤ C1's 568
- [ ] Action >5 Hz %: ≤ C1's 0.026%

### 7.3 Document results

- [ ] Update `THESIS_PROGRESS_REPORT.md` § 6 with Phase 2 final numbers
- [ ] Generate `outputs/eval/c3_6way/` plots
- [ ] Save final ckpt path + WandB run URL

---

## ◇ Section 8: Rollback / abort plan

### 8.1 Manual rollback (if eval shows degradation mid-training)

```bash
# Identify the last "good" checkpoint
ls outputs/train/phase2_v1/checkpoints/

# Resume training from that ckpt with reduced α/σ
# (current code doesn't support resume; for now, restart from C1 with reduced params)

python -m lerobot.scripts.phase2_train \
    --phase2.c1_ckpt_path=outputs/train/phase2_v1/checkpoints/step_0040/pretrained_model \
    --phase2.alpha_schedule_steps='[0, 200, 400]' \
    --phase2.alpha_schedule_values='[0.005, 0.02, 0.05]' \
    ...
```

### 8.2 Hard abort

If anything looks dangerous:
- **Operator**: press `ESC` (caught by lerobot keyboard listener) or `Ctrl+C`
- **Robot**: physical e-stop button
- After abort: don't immediately retry; investigate cause first

### 8.3 Failure modes & responses (refer to spec D6 §6.3)

| Symptom | Likely cause | Response |
|---|---|---|
| Task success drops below 4/5 | α too large / σ too aggressive | Halve α, tighten σ |
| ood_skip_rate > 50% | Exploration pushed policy OOD | Reduce σ, restart from earlier ckpt |
| L_BC drifts > 2× C1 | Actor diverging from demo distribution | Increase α weight on BC, add KL penalty β=0.01 |
| Reward stuck 100 steps | V function broken | Re-run critic warmup, check V_explained_variance |
| NaN reward / Inf advantage | IMU outlier or numeric instability | Tighten clip range, check IMU hardware |
| Gripper σ shakes object loose | σ_gripper too large | Halve σ_gripper |

---

## ◇ Quick-reference: command crib sheet

```
# Re-build IMU reward computer       (if needed)
python scripts/d2_setup_imu_reward.py

# Dry-run smoke (no hardware)
python -m lerobot.scripts.phase2_train --dry_run=true \
    --phase2.wandb_enable=false --phase2.warmup_episodes=2 \
    --phase2.warmup_gpu_steps=10 --phase2.k_episodes_per_update=1 \
    --phase2.n_updates=1 --phase2.episode_time_s=10

# Full training (real robot)
python -m lerobot.scripts.phase2_train \
    --robot.type=piper --robot.inference_time=true \
    --robot.enable_imu=true --robot.tsnd_device=/dev/ttyACM0 \
    --phase2.output_dir=outputs/train/phase2_v1 \
    --phase2.n_updates=100

# Real-robot eval with phase2 ckpt
python -m lerobot.scripts.control_robot \
    --robot.type=piper --robot.inference_time=true \
    --robot.enable_imu=true \
    --control.type=record --control.repo_id=local/eval_phase2_final \
    --control.num_episodes=5 \
    --control.policy.path=outputs/train/phase2_v1/checkpoints/final/pretrained_model

# Compare 5-way (rerun comparison)
python scripts/c3_5way_realrobot_compare.py
```

---

**Last updated**: 2026-05-05
**Spec reference**: `THESIS_PROGRESS_REPORT.md` § 6 (Phase 2 design)
