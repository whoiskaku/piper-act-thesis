# RUNBOOK：D′ 臂真机采集（`pi-distill-v2`）

**要回答的问题**：**把 C 臂那个已验证的控制器蒸馏进 π、从而消除它 3–4 帧的空档，
能不能把交付量拿回来？**

| | 交付量 | 来源 |
|---|---|---|
| C 臂（在线 G-2 + H 解析解，**有 3–4 帧空档**） | **−8.0%** | ab5 真机实测 |
| 教师（同一解析解，**零空档**） | **+12.25%** | H 模型算在真机实测墙上 |
| **`pi-distill-v2`（π 蒸馏，零空档已实测）** | **+11.23%** | 同法，α=0.90 |

---

## 0. 事前预测与判据（**跑之前写死**）

| | 点预测 |
|---|---|
| **`D′ − A`** | **−11.2%** |
| 对照 `C − A` | −8.0%（ab5，**另一会话**） |

**判否**：降幅 < 6% 或 p > 0.05 ⇒ 蒸馏路线不成立。
（6% 而非 8%：本次无同会话基线，见 §1 的偏差说明。）

**事前声明**：本次**做不出「π 优于 C」的统计结论** —— 两者相差约 3 pp，
需每臂 ~250 集。能立的只是 `D′−A` 本身。

---

## 1. ⚠ 本次的最大限制：**没有同会话的基线**

用户决定只收 D′ 臂。可用的基线是 **ab6 的 A×30**（今日 04:47–04:58 与 05:12 前后），
而 D′ 会整段排在其后。

已实测：**ab6 内部 A 从第一块到第二块变了 −26.5%**，而 ab5 是 **+27%**
—— **漂移方向都不固定**。所以 `D′ − A` 会带一个**未知大小、未知方向**的漂移偏差，
报告时必须写明。

**成本最低的补救（推荐，+10 集 ≈ 7 分钟）**：D′ 前后各插 5 集 A，
重新拿到漂移锚点，事后可做采集位协变量校正。

```
A′×5 → D′×30 → A″×5
```

不加也能跑，但结论要降格为「与另一时段的基线相比」。

---

## 2. 命令

```bash
cd /home/enolab-guo/claude-test-2/lerobot-piper
export PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
mkdir -p outputs/ab7

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
PINET=outputs/p2_pi/pi_distill_v2.pt
export CALIB=outputs/p2_labels/ab6-deploy/manifest.json   # 仅占位，两臂运行时都不用 R

COMMON=(
  --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
  --control.type=record --control.fps=20 --control.single_task=pp2
  --control.warmup_time_s=2 --control.episode_time_s=30 --control.reset_time_s=10
  --control.push_to_hub=false --control.policy.path="$POLICY"
)
run () {   # run <数据集后缀> <mode> <ckpt> <集数> <日志名>
  BCORR_CKPT="$3" BCORR_CALIB="$CALIB" BCORR_MODE="$2" BCORR_LOG="outputs/ab7/$5.json" \
  $PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
      --control.repo_id="$USER/eval_ab7_$1" --control.num_episodes="$4"
}
```

### 推荐（带漂移锚点，40 集 ≈ 33 分钟）

```bash
run Apre  off    "$PINET" 5  Apre
run D1    pi_net "$PINET" 15 D
run D2    pi_net "$PINET" 15 D2
run Apost off    "$PINET" 5  Apost
```

### 或者只跑 D′（30 集 ≈ 25 分钟）

```bash
run D1 pi_net "$PINET" 15 D
run D2 pi_net "$PINET" 15 D2
```

### 合并日志

```bash
for pair in "D2 D 15" "Apost Apre 5"; do set -- $pair
  n=$(ls outputs/ab7/$1_ep*.json 2>/dev/null | wc -l); [ "$n" = "0" ] && continue
  for k in $(seq $((n-1)) -1 0); do mv "outputs/ab7/$1_ep${k}.json" "outputs/ab7/$2_ep$(($3+k)).json"; done
done
ls outputs/ab7/ | sed 's/_ep[0-9]*\.json//' | sort | uniq -c
```

---

## 3. 跑的时候盯什么

启动必须看到：
`[pi_net] pi_distill_v2.pt ✓  68806 参数 · **纯前馈、不读 IMU** · sha b17c8ca3…`

| 项 | 期望（冒烟实测） | 对照：上次被判否的 `pi_g2env_v1` |
|---|---|---|
| **空档** | **0** | 0 |
| **δa RMS** | **0.61 mrad** | 2.14（**过量驱动，已判否**） |
| **触界** | **0.0%** | 0.9% |
| 接触门控 | 14–15% | 14.8% |
| π 推理 max | < 15 ms | 12.1 ms |
| stale / nan / 前向失败 | 0 | 0 |
| 帧数 | ≈593 | ≈593 |

**`δa RMS` 若明显超过 1.0 mrad 要立刻停** —— 那说明装错了 checkpoint
（旧的 `pi_g2env_v1` 是 2.14 mrad，在真机上 **+25%**）。
**`δa RMS ≈ 0` 也要立刻停** —— 等于没补正。

---

## 4. 跑完之后

```bash
# 标定（只给分析用；若跑了 Apre/Apost，用它们拟）
PYTHONPATH=.:sim $PY sim/imu_frame_calib.py --repo "$USER/eval_ab7_Apre" --eps 5 --lag 3
# …按 ab6 的做法生成 outputs/p2_labels/ab7-deploy/manifest.json

# 主判据（基线用 ab6 的 A）
PYTHONPATH=.:sim $PY sim/analyze_ab.py --root outputs/ab7 --task ab7 --arms A,D --n 30
```

⚠ 若基线取 ab6 的 A，必须把 A 的日志软链或复制进 `outputs/ab7/`，
并在报告里写明**基线来自另一时段**。

---

## 5. 口径

- `pi-distill-v2`：68,806 参数，**直接回归教师 δa\***（冻结 G-2+H 解析解，零空档）。
  留出集 **α = 0.90**、δa RMS **0.66 mrad**（教师 0.70）⇒ 落在 **H 被验证的区间**。
- **为什么不是「优化 H 的目标」**：那样训出的 π 是 α=1.97、δa 3.78 mrad，
  直接进入 **D 臂已证伪的大幅值区间**（`RESULT_pi_net_AD.md` §4）。**已判否，不上机。**
- 运行时**不跑 MuJoCo、不读 IMU、无 G-2/H/求解器**；ε=10 mrad 与 C 臂一致。
- **FF 不在本线**。
