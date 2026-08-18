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

| 档 | mode | seed 表（`BCORR_PROBE_SEEDS`） | mrad | 集数 | 作用 |
|---|---|---|---|---|---|
| P0 | off | — | — | 5 | 同会话基线（δa=0） |
| P1 | probe | `1000,1001,…,1007`（**逐集不同**） | 1.5 | 8 | 主辨识集 |
| P2 | probe | `2000,2000,2001,2001,2002,2002` | 1.5 | 6 | **跨 seed 验证**：3 个未见 seed × 各 2 集 |
| P3 | probe | `1000,1001,1002,1003`（**复用 P1**） | **3.0** | 4 | 幅值线性检验 |

**逐集换 seed 是必需的**：旧实现整批只有 **1 个**噪声实现 —— probe 表在 `__init__`
一次性生成，而 `reset()` 把帧计数归零 ⇒ 每集都从波形第 0 帧重放。
现已改为按集号从 seed 表取（`parse_seed_schedule`），并把**当集实际 seed 与波形
hash** 写入日志。

**为什么必须是显式 seed 表**（而不是 `base_seed + episode_id`）：
P2 要「同 seed 重复 2 集」、P3 要「复用 P1 的 seed」，递增式生成表达不了这两种。

**P3 与 P1 的关系**：同 seed ⇒ 波形逐点相同（hash 相同），只有幅值不同；
实测未触界处比值**严格 = 2.000000**（触界 0.09%）。

幅值依据：1.5 mrad 与部署时 π 的 δa（1.30–1.64）同量级、与最优解所需
（0.86–2.08）重合，且**触界 0.00%**；3.0 mrad 触界仅 0.11%。
（4 mrad 以上触界 >1%，削顶会破坏线性，不可用。）

## 命令

先 `mkdir -p outputs/ab4`。各条只改 `BCORR_MODE / BCORR_PROBE_* / BCORR_LOG / repo_id / num_episodes`。

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

# ---- P1  逐集换 seed, 1.5 mrad, 8 集 ----
BCORR_MODE=probe BCORR_PROBE_MRAD=1.5 \
BCORR_PROBE_SEEDS=1000,1001,1002,1003,1004,1005,1006,1007 \
BCORR_LOG=outputs/ab4/P1.json   --control.repo_id=$USER/eval_ab4_P1  --control.num_episodes=8

# ---- P2  3 个未见 seed 各 2 集, 1.5 mrad, 6 集 ----
BCORR_MODE=probe BCORR_PROBE_MRAD=1.5 \
BCORR_PROBE_SEEDS=2000,2000,2001,2001,2002,2002 \
BCORR_LOG=outputs/ab4/P2.json   --control.repo_id=$USER/eval_ab4_P2  --control.num_episodes=6

# ---- P3  复用 P1 前 4 个 seed, **3.0** mrad, 4 集 ----
BCORR_MODE=probe BCORR_PROBE_MRAD=3.0 \
BCORR_PROBE_SEEDS=1000,1001,1002,1003 \
BCORR_LOG=outputs/ab4/P3.json   --control.repo_id=$USER/eval_ab4_P3  --control.num_episodes=4
```

## 跑的时候盯什么

启动行会打印，**先确认再让它跑**：

```
P1: mode=probe ... π=off ff=off probe: 8 seeds [1000, 1001, 1002, 1003]… @1.5mrad
P0: mode=off   ... π=off ff=off
```

每集开头会打印当集实际用的 seed 与波形 hash：

```
[b_corr] ep0: probe seed=1000 mrad=1.5 hash=ce7e464eddfff5d4 RMS=1.500mrad clip=0.00%
[b_corr] ep1: probe seed=1001 mrad=1.5 hash=311f35c31457ebdf RMS=1.500mrad clip=0.00%
```

| 项 | 期望 | 不对时 |
|---|---|---|
| **每集 hash 不同**（P1/P3） | 逐集不同 | 相同 ⇒ seed 表没生效，整批又只有一个实现 |
| **P2 的 hash 两两成对** | ep0=ep1, ep2=ep3, ep4=ep5 | 不成对 ⇒ seed 表写错 |
| **P3 的 hash = P1 前 4 集** | 完全相同 | 不同 ⇒ 没复用 P1 激励，线性检验失效 |
| `clip` | **0.0%**（1.5）／**≈0.09%**（3.0） | 明显更高就停，幅值设错了 |
| `probe_overrun` | **0** | 非 0 ⇒ 跑的集数超过 seed 表，属协议偏离 |
| `probe_wrapped` | **0** | 非 0 ⇒ 集长超过 100 s 波形，集内有重复 |
| `render_ms_max` | **0** | 非 0 说明误开了 π |
| 每集帧数 | ≈590（与 A2 同） | 明显偏少 = 控制环没跟上 |
| 成功率 | 记录 | 1.5 mrad 比 π 的 δa 还小，应不受影响 |

## 跑完之后要回答的三问

1. **H 能不能在同条件内泛化？** P1 前 5 集训 / 后 3 集测
   （现在这是**真正的**跨激励检验 —— 8 集是 8 个不同的噪声实现）
2. **能不能跨 seed？** P1 训 → **P2 测**（3 个未见 seed）← 决定性，闭环数据这一关是崩的
   统计时**先在 seed 内聚合**（各 2 集），以 **seed 为独立单位**
3. **线性吗？** P1（1.5）训 → **P3（3.0）测**，同波形、幅值严格 ×2，看响应是否等比

三问都过 ⇒ Step 1 路径 A（显式求解 δa）复活，且拿到了 H 性能的硬数据。
任一不过 ⇒ 路径 A 判死，只剩路径 B（真机在环学习），受因果天花板 ≈30% 约束
（`CAUSAL_CEILING.md`，其中测量侧经验值仅 3.7%）。

> 采集与判据的完整预注册见 **`reports/PREREG_STEP1.md`**。
> 本 runbook 只负责「怎么采」，「采了怎么判」以预注册为准。

分析脚本：`sim/measure_da_response.py`、`sim/step1_gate.py`（改 `ARMS` 指向 `outputs/ab4`）。
