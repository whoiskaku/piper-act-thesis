# RUNBOOK：开环激励采集（2026-08-04）

按指導教員 Step 1「G無し、実世界の実測を元に補正を学ぶ」。

## 为什么必须重采

**现有的 δa 数据全是闭环的** —— D/E/F 三条臂的 δa 都由 π 从状态算出，
于是辨识出的「响应」H 里混着控制器。实测后果（`STEP1_GATE.md`）：

```
E 与 F 同一天、同一台机、同一会话，只差 FF 开关
  用 F 自己的 H 求解、F 评估:   −86.7%
  用 E 的   H 求解、F 评估:  +187.8%
```

**只有与状态无关的外生激励，才能把被控对象与控制器分开。** 这也直接回答
指導教員追问的「G の性能どうなってんの」——用测量，不用辩解。

## 采什么

`BCORR_MODE=probe`：注入**由种子完全决定**的带限白噪声 δa（0.5–9 Hz，六关节独立），
**不读 IMU、不经 π、不渲染 ideal**（因此没有 E/F 那个 330 ms 阻塞），夹爪不动。

| 臂 | mode | seed | mrad | 集数 | 作用 |
|---|---|---|---|---|---|
| P0 | off | — | — | 5 | 同会话基线（δa=0） |
| P1 | probe | 1 | 1.5 | 8 | 主辨识集 |
| P2 | probe | **2** | 1.5 | 8 | **换种子** ← 决定性检查 |
| P3 | probe | 1 | **3.0** | 8 | **换幅值** ← 线性检验（时间不够可省） |

幅值依据：1.5 mrad 与部署时 π 的 δa（1.30–1.64）同量级、与最优解所需
（0.86–2.08）重合，且**触界 0.00%**；3.0 mrad 触界仅 0.11%。
（4 mrad 以上触界 >1%，削顶会破坏线性，不可用。）

## 命令

先 `mkdir -p outputs/ab4`。四条只改 `BCORR_MODE / BCORR_PROBE_* / BCORR_LOG / repo_id`。

```bash
# ---- P0 基线 ----
BCORR_CKPT=outputs/p2_pi/pi_direct_bare_s0.pt \
BCORR_CALIB=outputs/p2_labels/pp2a-deploy/manifest.json \
BCORR_MODE=off \
BCORR_LOG=outputs/ab4/P0.json \
python lerobot/scripts/control_robot.py \
    --robot.type=piper --robot.inference_time=true --robot.enable_imu=true \
    --control.type=record --control.fps=20 --control.single_task="pp2" \
    --control.repo_id=$USER/eval_ab4_P0 \
    --control.num_episodes=5 --control.warmup_time_s=2 \
    --control.episode_time_s=30 --control.reset_time_s=10 \
    --control.push_to_hub=false \
    --control.policy.path=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model

# ---- P1  seed=1, 1.5 mrad, 8 集 ----
BCORR_MODE=probe BCORR_PROBE_SEED=1 BCORR_PROBE_MRAD=1.5 \
BCORR_LOG=outputs/ab4/P1.json   --control.repo_id=$USER/eval_ab4_P1  --control.num_episodes=8

# ---- P2  seed=2, 1.5 mrad, 8 集 ----
BCORR_MODE=probe BCORR_PROBE_SEED=2 BCORR_PROBE_MRAD=1.5 \
BCORR_LOG=outputs/ab4/P2.json   --control.repo_id=$USER/eval_ab4_P2  --control.num_episodes=8

# ---- P3  seed=1, 3.0 mrad, 8 集（可省）----
BCORR_MODE=probe BCORR_PROBE_SEED=1 BCORR_PROBE_MRAD=3.0 \
BCORR_LOG=outputs/ab4/P3.json   --control.repo_id=$USER/eval_ab4_P3  --control.num_episodes=8
```

## 跑的时候盯什么

启动行会打印，**先确认再让它跑**：

```
P1/P2:  mode=probe ... π=off ff=off probe=seed1/1.5mrad(RMS 1.50, clip 0.0%)
P0:     mode=off ... π=off ff=off
```

| 项 | 期望 | 不对时 |
|---|---|---|
| `clip` | **0.0%** | 非 0 就停，幅值设错了 |
| `render_ms_max` | **0** | 非 0 说明误开了 π |
| 每集帧数 | ≈590（与 A2 同） | 明显偏少 = 控制环没跟上 |
| 成功率 | 记录 | 1.5 mrad 比 π 的 δa 还小，应不受影响 |

## 跑完之后要回答的三问

1. **H 能不能在同条件内泛化？** P1 前 5 集训 / 后 3 集测
2. **能不能跨种子？** P1 训 → **P2 测** ← 决定性，闭环数据这一关是崩的
3. **线性吗？** P1（1.5）训 → **P3（3.0）测**，看响应是否按幅值等比

三问都过 ⇒ Step 1 路径 A（显式求解 δa）复活，且拿到了 Ĝ 性能的硬数据。
任一不过 ⇒ 路径 A 判死，只剩路径 B（真机在环学习），受因果天花板 ≈30% 约束。

分析脚本：`sim/measure_da_response.py`、`sim/step1_gate.py`（改 `ARMS` 指向 `outputs/ab4`）。
