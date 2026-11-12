# RUNBOOK：FF 在 insert 上的实机验证（2026-08-04）

FF 为主线后的第一个实机实验。判据与预测已在 `PREREG_FF_insert.md` 写死（跑之前）。

**这个实验决定的事**：FF 是「方法」还是「pp2 上的参数修正」。
pp2 上 −33.0% 是靠逐步调参凑出来的；insert 是**不再调参、直接套用**的第一次检验。

---

## 0. 出发前（零机时，各 10 秒）

```bash
cd /home/enolab-guo/claude-test-2/lerobot-piper

# ① 检查点与标定文件在位
ls outputs/train/2026-07-01/ACT_insert_noimu_seed42/checkpoints/100000/pretrained_model
ls outputs/p2_labels/ins-deploy/manifest.json
mkdir -p outputs/ab_ins
```

**必须用 100000 这一档**：所有离线预测（L\*=2.84 帧、接缝 2.2×、离线可消减 40.2%）
都是在 `eval_insert_100k` 上算的。换档位则预测失效。

---

## 1. 三条命令

三臂全部用**同一个策略检查点**，只改 `BCORR_*`。
insert 单集 59 s（1182–1190 帧），所以 `episode_time_s=60`（pp2 用的是 30）。

### A —— 裸 ACT（基线）

```bash
BCORR_CKPT=outputs/p2_pi/pi_v1_s0.pt \
BCORR_CALIB=outputs/p2_labels/ins-deploy/manifest.json \
BCORR_MODE=off \
BCORR_LOG=outputs/ab_ins/A.json \
python lerobot/scripts/control_robot.py \
    --robot.type=piper --robot.inference_time=true --robot.enable_imu=true \
    --control.type=record --control.fps=20 --control.single_task="insert" \
    --control.repo_id=$USER/eval_abins_A \
    --control.num_episodes=10 --control.warmup_time_s=2 \
    --control.episode_time_s=60 --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/2026-07-01/ACT_insert_noimu_seed42/checkpoints/100000/pretrained_model
```

### B3 —— 只有 lag（有效 blend 2）

在上面的基础上改这三行：

```bash
BCORR_MODE=ff BCORR_FF_LAG=3 BCORR_FF_BLEND=5 \
BCORR_LOG=outputs/ab_ins/B3.json \
    --control.repo_id=$USER/eval_abins_B3
```

### B3b13 —— 定档（有效 blend 10）

```bash
BCORR_MODE=ff BCORR_FF_LAG=3 BCORR_FF_BLEND=13 \
BCORR_LOG=outputs/ab_ins/B3b13.json \
    --control.repo_id=$USER/eval_abins_B3b13
```

**两臂省事版**：只跑 A 与 B3b13（20 集）。够判 H1/H2，但 **H3 测不了**
（blend 的边际贡献分不出来）。三臂 30 集才能验「FF 两个分量各对应一个物理机制」。

---

## 2. 跑的时候盯什么

| 项 | 期望 | 不对时 |
|---|---|---|
| 启动日志的 `ff_lag` / `ff_blend_effective` | B3 → 3 / 2；B3b13 → 3 / 10 | 立刻停，env 没传进去 |
| 每集帧数 | ≈1188 | 明显偏少 = 中断集，补跑 |
| 成功率 | 记录，**不设门** | 见下 |

**成功率不设门的理由**：insert 的失败已知主因是策略指令本身
（corr=1.000×23 条，`finding_phase2_insert_bottleneck`），FF 不修策略错误；
10 集对成功率也没有检验力。只在「FF 臂明显低于 A」时回头查安全性。

臂内顺序照旧不交替（你之前定的）。时间趋势用臂内前后半差事后检查，与 pp2 同法。

---

## 3. 跑完之后（我来做，或你直接跑）

```bash
# 主判据：配对检验。insert 集长 ~1188 ⇒ min-frames 抬到 900
PYTHONPATH=.:sim python sim/analyze_ab.py \
    --root outputs/ab_ins --task ins --arms A,B3,B3b13 --min-frames 900

# 残余滞后（H2）：FF 臂应为 0 帧
PYTHONPATH=.:sim python sim/ff_transfer_offline.py --eps 10

# 波形四联图（与 pp2 同口径，A 面板跨图一致）
PYTHONPATH=.:sim python sim/plot_arm_compare.py \
    --other B3b13 --ep 6 --task ins --root outputs/ab_ins --min-frames 900
```

`analyze_ab.py` 与 `plot_arm_compare.py` 今天已改成任务/臂可配置，
并在 pp2 旧数据上**逐位复现**了已发布数字（0.05625 / 0.04360 / 0.03767，
五频段 0.565–0.925），可以直接用。

同时修掉两个会咬人的隐患：排序改用 ep 后的整数（字典序下 ep10 会排到 ep2 前面，
≥10 集必然配对错位）；中断集按 `--min-frames` 剔除。

---

## 4. 判据（重申，勿改）

| | 内容 | 判否 |
|---|---|---|
| **H1** | FF 在 insert 上显著降 gap | 降幅 <15% 或 p>0.05 ⇒ FF 降级为「pp2 上的参数修正」 |
| **H2** | FF 臂残余滞后 = 0 帧 | ≠0 ⇒ `ff_lag` 需逐任务标定 |
| **H3** | blend 边际贡献 < pp2 的 10.5 pp | ≥10.5 pp ⇒ 「接缝比例决定 blend 收益」被否 |

**点预测 −32%**（lag 28.0% + blend 4.0 pp），区间 **−20% ~ −40%**，
并已记下：我在 blend 上连续两次低估，所以真值更可能偏高一侧。

即便 H1 通过，**FF 相对 TE 的增值仍未测**（见 `TE_CONFOUND.md`）——
那是另一个实验，不在本次范围。
