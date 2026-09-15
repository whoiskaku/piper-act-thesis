# RUNBOOK：`pi-g2wall-v1` 的真机 A/C（2026-09-15）

**判据 `C − A`**。两臂 = **A 裸 ACT** / **C 裸 ACT + π**。**FF 不在本线。**

零机时部分已全部完成并冻结：运行时逐帧一致性 5/5、端到端冒烟 12/12。
本文档只管上机，**不再做任何离线调整**。

---

## 0. 事前定死的预测与判据（跑之前写，不得事后改）

| | 内容 |
|---|---|
| **点预测** | `C − A` = **−7.6%**（离线逐 chunk 无空档 15.50%，扣掉实测 3–4 帧空档） |
| **区间** | −4% ~ −12%（把换会话风险算进去后的下沿） |
| **判否** | 降幅 < 4% 或 p > 0.05 ⇒ **G→π 这条路在真机上不成立** |

### 检出力（用 ab2 既有数据算，事前）

逐集 gap 能量变异系数 **11.6%**，按采集次序的配对相关 **r = 0.021**（等于没配对）。

| 每臂集数 | 最小可检出降幅（80%） |
|---|---|
| 10 | 16.3% |
| 20 | 10.8% |
| **40** | **7.5%** ← 本次采用 |

**⇒ 每臂必须 40 集**，否则「测不到」无法解释成「没效果」。
（对比：FF 的 −33% 在 10 集就 t=−6.30 / p=0.0001。）

### 两条必须随结果报告的已知风险

1. **换会话掉 25.8 pp**：ab4 P0 裸 **52.1%** → ab2 A 裸 **26.3%**。
   已做有边界诊断：在 ab2 上独立重拟 R 只把 26.3% 动到 **25.4%**（两 R 差 4.8°，
   属正常绑带漂移）⇒ **不是标定问题，是泛化问题，当天重算标定修不了。**
2. **离线指标从未预警过 π 的真机失败**（`finding_pi_realmachine_rejected`）。

---

## 1. 执行顺序（83 集，约 1 小时 15 分）

```
A1 ×20  →  [拟当日 R]  →  C_safe ×3(弃)  →  C ×40  →  A2 ×20
```

- **A 必须在前**：C 臂运行时需要 R，而 R 要从当日 A 块拟。
- **区块做成 A–C–A**：A 的平均采集位置 = (10.5+70.5)/2 = **40.5**，
  C = **40.5** ⇒ **线性漂移精确抵消**。顺序不可调换。
- `C_safe` 是真机安全块，**事前声明不进入分析**。

---

## 2. 命令

整段先在 repo 根目录执行一次：

```bash
cd /home/enolab-guo/claude-test-2/lerobot-piper
export PY=/home/enolab-guo/miniconda3/envs/lerobot/bin/python
mkdir -p outputs/ab5

# ── libstdc++ 冲突修正（**这行必须有**，否则 control_robot 起不来）──────────
# 症状: `import cv2` 报 /usr/lib/.../libstdc++.so.6: version `CXXABI_1.3.15' not found
# 成因: ROS noetic 的 setup 把 /usr/lib/x86_64-linux-gnu 顶到 LD_LIBRARY_PATH 最前,
#   lerobot 的依赖先把**系统**旧 libstdc++(只到 CXXABI_1.3.12)载入进程, 之后 cv2
#   复用已加载的那个 ⇒ 缺符号。单独 `import cv2` 不复现(那时系统库还没被载进来)。
# 修法: 预加载 conda env 自带的(CXXABI_1.3.15)让它先占位。**只动 libstdc++**,
#   不改 LD_LIBRARY_PATH ⇒ 不遮蔽 ROS 需要的其他系统库。
#   与 `scripts/collect_step1_stage1.sh` 里 ab4 采集用的是同一条, 已实测全链正常;
#   本次另测了 g2wall 的 **spawn 子进程**不受影响(往返 39.4 / 105.8 ms 不变)。
export LD_PRELOAD="/home/enolab-guo/miniconda3/envs/lerobot/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
MANIFEST=outputs/p2_pi/pi_g2wall_v1.json

COMMON=(
  --robot.type=piper --robot.inference_time=true --robot.enable_imu=true
  --control.type=record --control.fps=20 --control.single_task=pp2
  --control.warmup_time_s=2 --control.episode_time_s=30 --control.reset_time_s=10
  --control.push_to_hub=false --control.policy.path="$POLICY"
)
```

### ① A1 ×20（裸 ACT）

标定尚未拟出，A 臂 `mode=off` 不用 R，先借 `pp2a-deploy` 占位。

```bash
BCORR_CKPT="$MANIFEST" \
BCORR_CALIB=outputs/p2_labels/pp2a-deploy/manifest.json \
BCORR_MODE=off \
BCORR_LOG=outputs/ab5/A.json \
$PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
    --control.repo_id="$USER/eval_ab5_A1" --control.num_episodes=20
```

### ② 拟当日 R（不占机时，约 1 分钟）

```bash
PYTHONPATH=.:sim $PY sim/imu_frame_calib.py --repo "$USER/eval_ab5_A1" --eps 10 --lag 3

# 转成 wrapper 认的 manifest（复用 pp2a-deploy 的字段结构，只换 R/b）
$PY - <<'PY'
import json, os
from pathlib import Path
src = json.loads(Path("outputs/p2_labels/pp2a-deploy/manifest.json").read_text())
fit = json.loads(Path(f"outputs/sim/ideal_imu/eval_ab5_A1/frame_calib.json").read_text())
src["R"], src["b"] = fit["R"], fit["b"]
src["repo"] = f"{os.environ['USER']}/eval_ab5_A1"
out = Path("outputs/p2_labels/ab5-deploy"); out.mkdir(parents=True, exist_ok=True)
(out / "manifest.json").write_text(json.dumps(src, indent=1))
print("→", out / "manifest.json")
PY
```

**⚠ R 一旦拟出就全程固定**，C_safe / C / A2 都用它，**不重拟、不扫描**。
（自由旋转会把 cos 变成自由参数：同一份数据冻结 R 给 0.084、自由旋转给 0.412。）

### ③ C_safe ×3（真机安全块，**弃**）

```bash
BCORR_CKPT="$MANIFEST" \
BCORR_CALIB=outputs/p2_labels/ab5-deploy/manifest.json \
BCORR_MODE=g2wall \
BCORR_LOG=outputs/ab5/Csafe.json \
$PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
    --control.repo_id="$USER/eval_ab5_Csafe" --control.num_episodes=3
```

**跑完先看这一条，全绿才继续**：

```bash
$PY - <<'PY'
import json, glob
rows = [json.loads(open(p).read())["summary"] for p in sorted(glob.glob("outputs/ab5/Csafe_ep*.json"))]
gate = {"frame_ms p99 ≤60": None, "空档 3-5%": None, "sat ≈0.3%": None,
        "gated 15-17%": None, "δa RMS ≈0.6mrad": None, "stale/nan = 0": None,
        "进程卸载 on": None}
for i, s in enumerate(rows):
    g = s["g2wall"]
    print(f"ep{i}: 帧{s['frames']}  空档{g['blackout_pct']:.1f}%  sat{s['sat_pct']:.2f}%  "
          f"gated{s['gated_pct']:.1f}%  δa{s['da_mean_mrad']:.2f}mrad  "
          f"stale{s['stale']} nan{s['nan']}  卸载{g.get('process_offload')}  "
          f"推迟{g.get('defer_frames')}  标定差{g.get('calib_delta_deg')}°")
PY
```

| 项 | 冒烟实测（期望） | 不对时 |
|---|---|---|
| 空档 | **3.9%**（每 chunk 3–4 帧） | >8% ⇒ 后台比预期慢，先查 `process_offload` 是否为 `true` |
| `sat_pct` | ≈ **0.3%** | >5% ⇒ ε 被频繁顶满，停下来查 R |
| `gated_pct` | **15–17%** | 差很多 ⇒ 接触门控口径不对 |
| `da_mean_mrad` | ≈ **0.6** | ≈0 ⇒ **全程没补正**（查启动日志有没有 stage 失败） |
| `stale` / `nan` | **0** | 非 0 ⇒ IMU 链路有问题 |
| 帧数 | ≈ **585** | 明显偏少 = 中断集 |

真机的 `frame_ms` 从日志里没有单列，但**丢帧会直接表现为帧数偏少**；
另外启动时会打印 `[g2wall] 进程卸载 **开**`，没这行说明退回了线程池 ⇒ 帧时余量会变薄。

### ④ C ×40

```bash
BCORR_CKPT="$MANIFEST" \
BCORR_CALIB=outputs/p2_labels/ab5-deploy/manifest.json \
BCORR_MODE=g2wall \
BCORR_LOG=outputs/ab5/C.json \
$PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
    --control.repo_id="$USER/eval_ab5_C" --control.num_episodes=40
```

### ⑤ A2 ×20（裸 ACT，收尾）

```bash
BCORR_CKPT="$MANIFEST" \
BCORR_CALIB=outputs/p2_labels/ab5-deploy/manifest.json \
BCORR_MODE=off \
BCORR_LOG=outputs/ab5/A2.json \
$PY lerobot/scripts/control_robot.py "${COMMON[@]}" \
    --control.repo_id="$USER/eval_ab5_A2" --control.num_episodes=20

# A2 的 ep 号从 0 重新开始 ⇒ 接到 A 序列后面，否则会覆盖 A1
for k in $(seq 19 -1 0); do
  mv "outputs/ab5/A2_ep${k}.json" "outputs/ab5/A_ep$((20+k)).json"
done
ls outputs/ab5/A_ep*.json | wc -l    # 应为 40
ls outputs/ab5/C_ep*.json | wc -l    # 应为 40
```

---

## 3. 跑完之后

```bash
PYTHONPATH=.:sim $PY sim/analyze_ab.py \
    --root outputs/ab5 --task ab5 --arms A,C --n 40 --min-frames 500
```

`--task ab5` 会去读 `outputs/p2_labels/ab5-deploy/manifest.json`，即**当日的 R**。

事后还要各查一次（都不占机时）：

- **臂内时间趋势**：A 的前 20 集（A1）vs 后 20 集（A2）。两者若有显著差，
  说明 2 小时里确有漂移，A–C–A 的抵消是必要的，报告时要写明。
- **成功率**：记录但**不设门**（pp2 的成功率对 10–40 集没有检验力）。
- **空档与 sat 的逐集分布**：确认 C 臂全程健康，没有某几集退化。

---

## 4. 中途要停的情形

| 现象 | 处理 |
|---|---|
| 启动日志没有 `[g2wall] pi-g2wall-v1 ✓` | **立刻停**，冻结件校验没过 |
| 启动日志没有 `进程卸载 **开**` | 可以继续，但记下来；帧时余量变薄 |
| `da_mean_mrad ≈ 0` | **立刻停**，等于没补正（冒烟里踩过：全程 δa=0 而多数检验平凡通过） |
| 标定差 `calib_delta_deg` > 10° | **立刻停**，绑带可能装错了 |
| 任一集 `stale` 或 `nan` > 0 | 记下集号，该集事后剔除 |

---

## 5. 口径备忘

- 全部离线数字的被控对象是 **H（模型）**，**不是真机**。本次才是第一次真机验证。
- `pi-g2wall-v1` **自由参数 0 个**：G-2（原始冻结 v=0）+ H（`step1-H-v1`）+ 解析求解，
  **本次数据上不拟合任何东西**。
- C 臂**不读 IMU**（纯前馈，只吃当前 chunk 的 ACT 指令）。
  「实时 IMU 在环」是 `mode=full` 那条线的性质，**不是本臂的**。
