# RUNBOOK：开环激励采集（2026-08-04，2026-08-19 更新）

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

## 命令与分阶段闸门

先在 repo 根目录执行下面整段。`run_probe` 每次都调用完整的采集命令，
不是只能粘在上一条后面的参数缩写。

```bash
set -e  # P2/P3/离线闸门返回非零时，整段立即停止
mkdir -p outputs/ab4 outputs/step1_candidate

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
CALIB=outputs/p2_labels/pp2a-deploy/manifest.json
PROBE_CKPT=outputs/p2_pi/pi_direct_bare_s0.pt

COMMON_RECORD_ARGS=(
  --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
  --control.type=record --control.fps=20 --control.single_task=pp2
  --control.warmup_time_s=2 --control.episode_time_s=30 --control.reset_time_s=10
  --control.push_to_hub=false --control.policy.path="$POLICY"
)

run_probe () {
  local tier="$1" mode="$2" mrad="$3" seeds="$4" episodes="$5"
  BCORR_CKPT="$PROBE_CKPT" \
  BCORR_CALIB="$CALIB" \
  BCORR_MODE="$mode" \
  BCORR_PROBE_MRAD="$mrad" \
  BCORR_PROBE_SEEDS="$seeds" \
  BCORR_LOG="outputs/ab4/${tier}.json" \
  python lerobot/scripts/control_robot.py \
    "${COMMON_RECORD_ARGS[@]}" \
    --control.repo_id="$USER/eval_ab4_${tier}" \
    --control.num_episodes="$episodes"
}

# 阶段 1：先只收 P0/P1/P2
run_probe P0 off   1.5 0                                             5
run_probe P1 probe 1.5 1000,1001,1002,1003,1004,1005,1006,1007       8
run_probe P2 probe 1.5 2000,2000,2001,2001,2002,2002                 6

# P2 跨 seed 闸门；PASS 才继续，FAIL（退出码 2）则停止
python sim/fit_step1_candidate.py \
  --data-root outputs/ab4 \
  --calib "$CALIB" \
  --output outputs/step1_candidate/step1_linear_v1.npz \
  --stage p2

# 阶段 2：仅在上一步明确 PASS 后采 P3
run_probe P3 probe 3.0 1000,1001,1002,1003                         4

# P3 线性 + P0 ep3–4 离线 8% 闸门；全过才写出 candidate NPZ
python sim/fit_step1_candidate.py \
  --data-root outputs/ab4 \
  --calib "$CALIB" \
  --output outputs/step1_candidate/step1_linear_v1.npz \
  --stage full
```

**不得用单集 `rerecord` 补采。** 它会让数据集集号与 wrapper 的 seed/日志集号错位。
发生纯技术故障时，废弃受影响的整个 P 档，用**原 seed 表**和新的 repo/log 名重开整批；
不得只重跑某一集后混回原批。

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

## Candidate 通过后：A/B/C 部署示例

A/B/C 都指向**同一个已冻结 candidate**；A 用 `off`，B 用 `step1_b`，
C 用 `step1_c`。下面是一个 `ABC` block 的完整可执行示例；正式 10 个 block
必须按 `PREREG_STEP1.md` 附录 A 的冻结顺序调用，不得全部固定为 `ABC`。

```bash
mkdir -p outputs/step1_confirm

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
CALIB=outputs/p2_labels/pp2a-deploy/manifest.json
CANDIDATE=outputs/step1_candidate/step1_linear_v1.npz

COMMON_CONFIRM_ARGS=(
  --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
  --control.type=record --control.fps=20 --control.single_task=pp2
  --control.num_episodes=1 --control.warmup_time_s=2
  --control.episode_time_s=30 --control.reset_time_s=10
  --control.push_to_hub=false --control.policy.path="$POLICY"
)

run_candidate () {
  local tag="$1" mode="$2"
  BCORR_CKPT="$CANDIDATE" \
  BCORR_CALIB="$CALIB" \
  BCORR_MODE="$mode" \
  BCORR_LOG="outputs/step1_confirm/${tag}.json" \
  python lerobot/scripts/control_robot.py \
    "${COMMON_CONFIRM_ARGS[@]}" \
    --control.repo_id="$USER/eval_step1_${tag}"
}

# 示例：会话 2，block 01 的顺序若为 ABC
run_candidate S2_block01_A off
run_candidate S2_block01_B step1_b
run_candidate S2_block01_C step1_c
```

开跑前核对 B/C 日志中 `artifact_sha256` 相同，且 B/C 分别记为
`step1_b` / `step1_c`。技术故障按预注册重跑**整个 triplet**，不单独补一臂。

## 跑完后看哪些结果

只以 `sim/fit_step1_candidate.py` 的统一输出为准：

1. 在 P0 ep0–2 + P1 上做 episode 分组 5 折 CV，每折重走 `H → W → Θ`，选 ridge；
2. P2 的 3 个未见 seed 验证 H 是否跨 seed（先在 seed 内聚合）；
3. P3 检查同波形从 1.5 mrad 到 3.0 mrad 的幅值线性；
4. P0 ep3–4 用完整因果卷积做独立离线闸门，要求预测中位降幅 ≥ 8%。

P2 不过就不收 P3；P3 或 P0 闸门不过就不写 candidate NPZ，不跑三臂。

> 采集与判据的完整预注册见 **`reports/PREREG_STEP1.md`**。
> 本 runbook 只负责「怎么采」，「采了怎么判」以预注册为准。

统一拟合/闸门脚本：`sim/fit_step1_candidate.py`。
