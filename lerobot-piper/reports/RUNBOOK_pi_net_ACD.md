# RUNBOOK：三臂 A / C / D 真机实验（`pi-g2env-v1`）

**要回答的问题**：**在冻结的 G-2 环境里训出来的 π，能不能迁移到真机？**

| 臂 | 运行时组成 | 回答什么 |
|---|---|---|
| **A** | 裸 ACT | 基线 |
| **C** | ACT + 在线 G-2 + H 解析求解 | 模型式控制基线（已知 −8.0%） |
| **D** | **ACT + π 网络** | **G 训出的策略能否迁移** |

核心比较：**`D−A`** = π 真机是否有效 · **`D−C`** = π 能否追上在线模型式控制器 ·
**`C−A`** = 复现已有的 ~8%。

---

## 0. 事前预测与判据（**跑之前写死，不得事后改**）

| | 点预测 | 依据 |
|---|---|---|
| **`D − A`** | **−0.1%** | 用真机已验证的 `H`（β=0.955）把 π 的 δa 推到真机上算出来；逐集 −4.1/+1.9/+0.3/+1.7/−0.9% |
| `C − A` | −8.0% | ab5 实测复现 |
| `D − C` | **+7.9%**（D 明显更差） | 上两项之差 |

**成因预测**：π 的 δa **RMS 2.14 mrad = C 臂的 4.0×**，却什么也换不到 ——
方向学自 G-2 的增量响应，而它与真机的 `cos` 仅 **0.12**。

### 判读规则（事前）

| D−A 实测 | 结论 |
|---|---|
| **≈ −8%**（与预测严重不符） | G-2 的增量问题在闭环里被某种方式抵消了，**需要重新理解**，是重大发现 |
| **≈ −0.1%**（符合预测） | **定量确认**「G-2 绝对保真度够、增量保真度不够」，且给出了倍数 |
| **> 0**（补反了） | 方向也错，G-2 的增量通道**完全不可用** |

⚠ **无论哪一种，实验都成立** —— 因为已排除「训练没跑起来」：
π 在 G-2 环境里、**用真前向**在留出集上降 **+15.8%**（解析参照 +15.2%）。
所以失败只可能是**迁移**失败，不是学习失败。

---

## 1. 执行顺序（90 集，约 1 小时 20 分）

```
A×15 → [拟当日 R] → C×15 → D×15 → D×15 → C×15 → A×15
```

**为什么这个顺序**：三臂的平均采集位**完全相同**（A 45.5 / C 45.5 / D 45.5）
⇒ **线性漂移精确抵消**。上次实测会话漂移 **+27%**，比效应本身大 3 倍，
不做平衡会污染结论。**顺序不可调换。**

**为什么 A 必须在最前**：C 臂运行时需要 `R`，而 `R` 从当日 A 块拟。
（D **不需要** `R` —— π 是纯前馈、不做任何标定变换；但 wrapper 仍要求传入。）

**每臂 30 集**（= 两个 15 集的块）。依据：要分辨的是「D 接近 −8%」还是「D 接近 0」，
两者相距 8 pp；每臂 30 集的最小可检出降幅约 **8.7%**（用 ab2/ab5 的逐集离散度算，
CV 11.6%），刚好能把两个假设分开。15 集只到 ~13%，不够。

---

## 2. 命令

```bash
cd /home/enolab-guo/claude-test-2/lerobot-piper
export PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python
mkdir -p outputs/ab6

# libstdc++ 冲突修正（**必须**，否则 control_robot 起不来）
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
G2WALL=outputs/p2_pi/pi_g2wall_v1.json
PINET=outputs/p2_pi/pi_g2env_v1.pt

COMMON=(
  --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
  --control.type=record --control.fps=20 --control.single_task=pp2
  --control.warmup_time_s=2 --control.episode_time_s=30 --control.reset_time_s=10
  --control.push_to_hub=false --control.policy.path="$POLICY"
)
run () {   # run <臂名> <mode> <ckpt> <集数> <日志名>
  BCORR_CKPT="$3" BCORR_CALIB="$CALIB" BCORR_MODE="$2" BCORR_LOG="outputs/ab6/$5.json" \
  $PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
      --control.repo_id="$USER/eval_ab6_$1" --control.num_episodes="$4"
}
```

### ① A1 ×15（裸 ACT）

```bash
export CALIB=outputs/p2_labels/ab5-deploy/manifest.json   # 占位：A 臂 mode=off 不用 R
run A1 off "$G2WALL" 15 A
```

### ② 拟当日 R（不占机时）

```bash
PYTHONPATH=.:sim $PY sim/imu_frame_calib.py --repo "$USER/eval_ab6_A1" --eps 10 --lag 3
$PY - <<'PY'
import json, os
from pathlib import Path
src = json.loads(Path("outputs/p2_labels/pp2a-deploy/manifest.json").read_text())
fit = json.loads(Path(f"outputs/sim/ideal_imu/eval_ab6_A1/frame_calib.json").read_text())
src["R"], src["b"] = fit["R"], fit["b"]
src["repo"] = f"{os.environ['USER']}/eval_ab6_A1"
src["fitted_on"] = "当日 A1 块前 10 集; lag=3; **全程固定, 不重拟不扫描**"
out = Path("outputs/p2_labels/ab6-deploy"); out.mkdir(parents=True, exist_ok=True)
(out / "manifest.json").write_text(json.dumps(src, indent=1)); print("→", out)
PY
export CALIB=outputs/p2_labels/ab6-deploy/manifest.json
```

**⚠ 与冻结评价所用 R 的差角应 < 10°**（上次 2.93°）。超了就停，查绑带。

### ③–⑥ 其余四块

```bash
run C1 g2wall "$G2WALL" 15 C
run D1 pi_net "$PINET"  15 D
run D2 pi_net "$PINET"  15 D2
run C2 g2wall "$G2WALL" 15 C2
run A2 off    "$G2WALL" 15 A2
```

### ⑦ 合并日志（各臂 ep 号从 0 重开，必须接续）

```bash
# 第二块的 ep 号从 0 重开 ⇒ 整体后移 15，接到第一块后面
for pair in "A2 A" "C2 C" "D2 D"; do set -- $pair
  n=$(ls outputs/ab6/$1_ep*.json 2>/dev/null | wc -l)
  echo "$1: $n 集 → 接到 $2_ep15.."
  for k in $(seq $((n-1)) -1 0); do
    mv "outputs/ab6/$1_ep${k}.json" "outputs/ab6/$2_ep$((15+k)).json"
  done
done
for x in A C D; do echo "$x: $(ls outputs/ab6/${x}_ep*.json | wc -l) 集"; done   # 各应为 30
```

⚠ **先跑一次不带 `mv` 的 `ls` 确认块大小**再动文件 —— 若某块中途少录了几集，
偏移 15 会撞号或留空。

---

## 3. 跑的时候盯什么

**启动必须看到这几行**，缺任何一行就停：

| 臂 | 应出现 |
|---|---|
| C | `[g2wall] pi-g2wall-v1 ✓` + `进程卸载 **开**` + `标定差 …°` |
| **D** | `[pi_net] pi_g2env_v1.pt ✓  68806 参数 · **纯前馈、不读 IMU**` |

| 项 | C 臂期望 | **D 臂期望** |
|---|---|---|
| 空档 | 3–4 帧/chunk | **0**（π 同步一次前向） |
| δa RMS | 0.54 mrad | **2.14 mrad**（4×，预期如此） |
| 触界 | 0.00% | ~0.9% |
| 接触门控 | 14–15% | 14–15% |
| stale / nan / 失败 | 0 | 0 |
| 帧数 | ≈593 | ≈593 |

**`δa RMS ≈ 0` 要立刻停** —— 等于没补正（冒烟里踩过：`summary()` 有副作用导致
每 20 帧清空 π 状态，多数检验却平凡通过）。

---

## 4. 跑完之后

```bash
PYTHONPATH=.:sim $PY sim/analyze_ab.py \
    --root outputs/ab6 --task ab6 --arms A,C,D --n 30 --min-frames 500
```

事后还要各查一次：臂内时间趋势（各臂前半 vs 后半）、逐集空档与触界分布、成功率（记录不设门）。

---

## 5. 口径

- `pi-g2env-v1`：**68,806 参数**，训练于 176 集指令流的 G-2 环境，
  留出集上用**真 G-2** 复核 **+15.8%**。运行时**不跑 MuJoCo、不读 IMU、无求解器**。
- D 臂**不需要标定** —— π 直接出 δa，不做任何坐标变换。传入 `BCORR_CALIB` 只为满足接口。
- ε=10 mrad 三臂一致（保证可比）：π 内由 `tanh` 结构保证，外面再限幅兜底。
- 幅度安全性：2.14 mrad RMS / 10 mrad 峰值在已验证范围内
  （P3 开环探针跑过 3 mrad RMS / 10 mrad 峰值）。
- **FF 不在本线**。
