#!/usr/bin/env python3
"""#74 — Contact-window coherence gate: 决定 IMU-surrogate 路线是否该继续。

研究问题: 命令空间残差(RefineNet)能控制的真机 IMU 振动方差上限有多高?
  稳态段 #71 已封(线性 controller-aware 负结果, 透传带 coh~0.15)。
  唯一可能有余量的是 **抓取/释放接触瞬态** —— 它和 approach velocity / gripper timing /
  command roughness 有关, 而 #71 的线性 LTI 代理看不见瞬态。

口径(对齐讨论 R1-R4):
  R1  相干是平稳度量 → 只对长稳态段(motion/stationary)算 coherence;
      接触瞬态(窗太短)改用【跨集峰值回归】, 不算窗口内 coherence。
  R2  接触峰预测因子拆两类:
        · approach_vel / gross 轨迹 = 【行为类】→ 残差受 L_BC+|ΔA| 约束动不了(要降=改任务);
        · 接触前 command 高频粗糙度 = 【可达类】→ 残差能平滑、不改 demo 轨迹。
      PASS 仅当【可达类】能解释峰值方差(偏相关), 否则是速度/轨迹问题, 非残差问题。
  R3  接触事件用 gripper STATE 边沿检测(真机 eval 数据 gripper 活; cmd 尺度太小)。
  R4  30Hz, Nyquist=15Hz → 5-15Hz 是上半带近 Nyquist 噪声大; coh 受命令 15Hz Nyquist 封顶。

输入数据: 真机 LeRobot 数据集, observation.imu_real (6)=ax,ay,az,gx,gy,gz @30Hz, m/s²+rad/s。
判定:
  FAIL(封 IMU-surrogate 线): 各稳态窗 coh²<0.2 且 motion HF≈stationary floor 且 接触峰仅由 approach_vel 解释。
  PASS(转 contact-aware residual): 接触峰被 command_roughness 显著解释(偏相关) → 目标改为"降接触瞬态峰", 非全程平滑。

用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python scripts/contact_coherence_gate.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, csd, welch

LR = Path.home() / ".cache/huggingface/lerobot"
REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "outputs/eval/contact_coherence_gate.json"
OUT_PNG = REPO / "outputs/eval/contact_coherence_gate.png"
FS = 30.0
NYQ = FS / 2.0
MDEG_TO_RAD = np.pi / 180.0 / 1000.0
ARM = [0, 1, 2]                       # j1/j2/j3 (action↔state 同为 rad)
BANDS = {"2-4.5Hz": (2.0, 4.5), "5-15Hz": (5.0, 15.0)}
VEL_STILL = 0.05                       # rad/s, 低于则静止
NPERSEG = 64                           # df=0.47Hz; 2-4.5≈5bin, 5-15≈21bin

# 真机数据集: policy-rollout(ACT 抖动 regime, RefineNet 实际作用对象) + demo(干净基线/清晰事件)
POLICY = ["eval_s5_A_act", "eval_s5_C_full", "eval_s5tok_A", "eval_s5tok_1q",
          "eval_s5tok_Mq", "eval_s5tok_RLT", "eval_refine_A", "eval_refine_B",
          "eval_s5_B_blend"]
DEMO = ["local/piper_pick_place_imu_v3"]

_HP5 = butter(4, 5.0 / NYQ, "high")
_BP24 = butter(4, [2.0 / NYQ, 4.5 / NYQ], "band")


def load_ep(pq):
    df = pd.read_parquet(pq)
    act = np.stack(df["action"].to_numpy()).astype(np.float64)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
    imu = np.stack(df["observation.imu_real"].to_numpy()).astype(np.float64)
    st = st.copy(); st[:, :6] *= MDEG_TO_RAD
    return act, st, imu


def collect(dsets, base):
    eps = []
    for ds in dsets:
        for pq in sorted(glob.glob(str(LR / base / ds / "data/chunk-*/episode_*.parquet")
                                   if base else str(LR / ds / "data/chunk-*/episode_*.parquet"))):
            try:
                eps.append((ds, *load_ep(pq)))
            except (KeyError, ValueError):
                continue
    return eps


def band_mean(f, arr, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(arr[m].mean()) if m.any() else float("nan")


def pooled_coherence(seg_pairs):
    """seg_pairs: list of (x,y) 1-D segments(同一 input/output 通道). 池化 coh²=|ΣPxy|²/(ΣPxx·ΣPyy)。"""
    Pxx = Pyy = Pxy = None; f = None
    for x, y in seg_pairs:
        if len(x) < NPERSEG:
            continue
        f, pxx = welch(x, fs=FS, nperseg=NPERSEG)
        _, pyy = welch(y, fs=FS, nperseg=NPERSEG)
        _, pxy = csd(x, y, fs=FS, nperseg=NPERSEG)
        Pxx = pxx if Pxx is None else Pxx + pxx
        Pyy = pyy if Pyy is None else Pyy + pyy
        Pxy = pxy if Pxy is None else Pxy + pxy
    if f is None:
        return None, None
    coh2 = (np.abs(Pxy) ** 2) / (Pxx * Pyy + 1e-18)
    return f, coh2


def vel_of(act):
    return np.vstack([np.zeros((1, 6)), np.diff(act[:, :6], axis=0) * FS])  # (T,6) rad/s


def grip_events(st):
    """gripper STATE 边沿 → grasp(闭合上升沿) / release(张开下降沿) 帧索引。"""
    gs = st[:, 6]
    rng = gs.max() - gs.min()
    if rng < 1e-6:
        return [], []
    g = (gs - gs.min()) / rng
    closed = g > 0.5
    d = np.diff(closed.astype(int))
    grasp = list(np.where(d == 1)[0] + 1)     # 上升沿 = 闭合
    release = list(np.where(d == -1)[0] + 1)  # 下降沿 = 张开
    return grasp, release


def main():
    pol = collect(POLICY, "enolab-guo")
    dem = collect(DEMO, "")
    print(f"[data] policy-rollout eps={len(pol)}  demo eps={len(dem)}")
    alleps = pol + dem

    # ============ ① 稳态相干: motion 段 command/joint-vel → IMU ============
    # input 通道: cmd j1-3, vel j1-3 ; output: accel x/y/z, gyro x/y/z
    # 在 motion(|vel|>still) 与 stationary(|vel|<still) 两窗分别池化。
    in_names = [f"cmd_j{j+1}" for j in ARM] + [f"vel_j{j+1}" for j in ARM]
    out_names = ["ax", "ay", "az", "gx", "gy", "gz"]
    # 收集每窗每(in,out)对的 segments
    segs = {w: {(i, o): [] for i in in_names for o in out_names} for w in ["motion", "stationary"]}
    for ds, act, st, imu in alleps:
        v = vel_of(act); vn = np.linalg.norm(v[:, ARM], axis=1)
        motion = vn > VEL_STILL
        ins = {f"cmd_j{j+1}": act[:, j] for j in ARM}
        ins.update({f"vel_j{j+1}": v[:, j] for j in ARM})
        outs = {o: imu[:, k] for k, o in enumerate(out_names)}
        for w, mask in [("motion", motion), ("stationary", ~motion)]:
            if mask.sum() < NPERSEG:
                continue
            # 取最长连续段(避免拼接伪频)
            idx = np.where(mask)[0]
            splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            run = max(splits, key=len)
            if len(run) < NPERSEG:
                continue
            sl = slice(run[0], run[-1] + 1)
            for i in in_names:
                for o in out_names:
                    segs[w][(i, o)].append((ins[i][sl], outs[o][sl]))

    coh_report = {}
    for w in ["motion", "stationary"]:
        per_band = {b: [] for b in BANDS}
        for (i, o), sp in segs[w].items():
            f, coh2 = pooled_coherence(sp)
            if f is None:
                continue
            for b, (lo, hi) in BANDS.items():
                per_band[b].append(((i, o), band_mean(f, coh2, lo, hi)))
        coh_report[w] = {}
        for b in BANDS:
            vals = [(p, c) for p, c in per_band[b] if not np.isnan(c)]
            if not vals:
                continue
            cs = np.array([c for _, c in vals])
            top = max(vals, key=lambda t: t[1])
            coh_report[w][b] = {"mean": float(cs.mean()), "max": float(cs.max()),
                                "argmax": f"{top[0][0]}→{top[0][1]}",
                                "frac_gt_0.2": float((cs > 0.2).mean())}

    # ============ ② 静止地板 vs 运动段 HF 能量 ============
    floor = {b: {"motion": 0.0, "stationary": 0.0, "n_m": 0, "n_s": 0} for b in BANDS}
    # 用 IMU 振动幅值(高通/带通后), gyro+accel 合并(各通道能量和)
    for ds, act, st, imu in alleps:
        v = vel_of(act); vn = np.linalg.norm(v[:, ARM], axis=1)
        motion = vn > VEL_STILL
        for b, (lo, hi) in BANDS.items():
            bp = butter(4, [lo / NYQ, min(hi, NYQ - 0.01) / NYQ], "band")
            for k in range(6):
                y = filtfilt(*bp, imu[:, k])
                e = y ** 2
                if motion.sum() > 10:
                    floor[b]["motion"] += float(e[motion].sum()); floor[b]["n_m"] += int(motion.sum())
                if (~motion).sum() > 10:
                    floor[b]["stationary"] += float(e[~motion].sum()); floor[b]["n_s"] += int((~motion).sum())
    floor_ratio = {}
    for b in BANDS:
        pm = floor[b]["motion"] / max(floor[b]["n_m"], 1)   # 每帧能量(运动)
        ps = floor[b]["stationary"] / max(floor[b]["n_s"], 1)  # 每帧能量(静止=纯地板)
        floor_ratio[b] = {"motion_per_frame": pm, "stationary_per_frame": ps,
                          "motion/floor": float(pm / (ps + 1e-18))}

    # ============ ③ 接触峰跨集回归: peak ~ approach_vel(行为) + cmd_roughness(可达) ============
    rows = []  # 每个事件一行
    for ds, act, st, imu in alleps:
        v = vel_of(act); vn = np.linalg.norm(v[:, ARM], axis=1)
        grasp, release = grip_events(st)
        # IMU 振动(5Hz 高通) 幅值
        acc_hp = np.sqrt(sum(filtfilt(*_HP5, imu[:, k]) ** 2 for k in range(3)))   # accel mag
        gyr_hp = np.sqrt(sum(filtfilt(*_HP5, imu[:, k]) ** 2 for k in range(3, 6)))  # gyro mag
        # command 5Hz 高通能量(粗糙度), arm joints
        cmd_hp = np.vstack([filtfilt(*_HP5, act[:, j]) for j in ARM]).T  # (T,3)
        W = int(0.5 * FS)         # ±0.5s
        Wpre = slice(0, 0)
        for kind, evs in [("grasp", grasp), ("release", release)]:
            for e in evs:
                a, b2 = max(0, e - W), min(len(act), e + W)
                if b2 - a < 5:
                    continue
                pre_a, pre_b = max(0, e - int(1.0 * FS)), max(1, e - int(0.2 * FS))
                rows.append({
                    "ds": ds, "kind": kind,
                    "peak_accel": float(acc_hp[a:b2].max()),
                    "peak_gyro": float(gyr_hp[a:b2].max()),
                    "approach_vel": float(vn[pre_a:pre_b].mean()) if pre_b > pre_a else 0.0,
                    "cmd_rough": float((cmd_hp[a:b2] ** 2).sum()),
                    "grip_speed": float(np.abs(np.diff(st[a:b2, 6])).max()) if b2 - a > 1 else 0.0,
                })
    R = pd.DataFrame(rows)
    print(f"[events] {len(R)} contact events ({(R['kind']=='grasp').sum()} grasp / {(R['kind']=='release').sum()} release)")

    def corr(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    def partial_r2(y, x_keep, x_ctrl):
        """x_keep 在控制 x_ctrl 后对 y 的偏相关² (线性回归残差相关)。"""
        import numpy as _np
        X = _np.column_stack([_np.ones(len(y)), x_ctrl])
        def resid(t):
            beta, *_ = _np.linalg.lstsq(X, t, rcond=None)
            return t - X @ beta
        ry, rx = resid(_np.asarray(y, float)), resid(_np.asarray(x_keep, float))
        return corr(ry, rx)

    reg = {}
    if len(R) >= 8:
        for tgt in ["peak_accel", "peak_gyro"]:
            y = R[tgt].values
            reg[tgt] = {
                "corr_approach_vel": corr(y, R["approach_vel"]),     # 行为类(不可达)
                "corr_cmd_rough": corr(y, R["cmd_rough"]),           # 可达类
                "corr_grip_speed": corr(y, R["grip_speed"]),
                # 偏相关: 控制 approach_vel 后, cmd_rough 还能解释峰吗? (R2 关键)
                "partial_cmd_rough|vel": partial_r2(y, R["cmd_rough"], R[["approach_vel"]].values),
            }

    # ============ 判定 ============
    # ceiling 用 MOTION 段(RefineNet 实际作用 regime); stationary 段 |vel|≈0, 近零信号相干
    # 不可信(单段少Welch窗+低SNR上偏), 仅报告不计入判定。
    motion_coh = max([coh_report.get("motion", {}).get(b, {}).get("max", 0) for b in BANDS] or [0])
    stat_coh = max([coh_report.get("stationary", {}).get(b, {}).get("max", 0) for b in BANDS] or [0])
    floor_excess = max([floor_ratio[b]["motion/floor"] for b in BANDS])
    av = max([abs(reg.get(t, {}).get("corr_approach_vel", 0) or 0) for t in reg] or [0])
    pcr = max([abs(reg.get(t, {}).get("partial_cmd_rough|vel", 0) or 0) for t in reg] or [0])
    cr = max([abs(reg.get(t, {}).get("corr_cmd_rough", 0) or 0) for t in reg] or [0])
    verdict = {
        "motion_coh2_max": motion_coh,               # 相关regime: 命令/速度→IMU 可控上限
        "stationary_coh2_max_ARTIFACT": stat_coh,    # vel≈0 近零信号, 不计入判定
        "motion_over_floor_max": floor_excess,       # 运动诱发振动倍数(高≠可控, 须看相干)
        "contact_peak_corr_approach_vel": av,        # 行为类(残差不可达)
        "contact_peak_corr_cmd_rough": cr,           # 可达类(残差能碰)
        "contact_peak_partial_cmd_rough_given_vel": pcr,
    }
    # PASS 仅当接触峰被 command roughness 显著解释(R2: 残差可达); 否则若运动段低相干 + 峰非可达 → FAIL
    if cr >= 0.3 and pcr >= 0.3:
        verdict["decision"] = "PASS — 接触峰可由 command roughness 解释(残差可达) → 做 contact-aware residual(目标=降接触瞬态峰)"
    elif motion_coh < 0.2 and pcr < 0.2 and cr < 0.2:
        verdict["decision"] = (f"FAIL — 运动段 命令/速度→IMU 相干<0.2(即便运动HF是地板{floor_excess:.0f}×, "
                               f"振动与命令几乎不相干=运动诱发但命令不可控); 接触峰非roughness可达"
                               f"(corr≈{cr:.2f}/偏相关≈{pcr:.2f}), 仅approach_vel弱解释(≈{av:.2f})=轨迹问题非残差 → 封 IMU-surrogate 线")
    else:
        verdict["decision"] = "AMBIGUOUS — 见各项"

    report = {"n_policy_eps": len(pol), "n_demo_eps": len(dem), "n_events": len(R),
              "steady_coherence": coh_report, "hf_floor": floor_ratio,
              "contact_regression": reg, "verdict": verdict,
              "notes": "R4: 30Hz Nyquist15, 5-15Hz 近Nyquist噪声大. coherence 池化 |ΣPxy|²/(ΣPxxΣPyy)."}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # ---- 打印 ----
    print("\n================ ① 稳态相干 coh² (command/joint-vel → IMU) ================")
    for w in ["motion", "stationary"]:
        print(f"  [{w}]")
        for b in BANDS:
            r = coh_report[w].get(b)
            if r:
                print(f"    {b:>8}: mean={r['mean']:.3f}  max={r['max']:.3f} ({r['argmax']})  frac>0.2={r['frac_gt_0.2']:.2f}")
    print("\n================ ② 运动段 HF / 静止地板 (每帧能量比) ================")
    for b in BANDS:
        fr = floor_ratio[b]
        print(f"    {b:>8}: motion/floor = {fr['motion/floor']:.2f}  (motion={fr['motion_per_frame']:.4g}, floor={fr['stationary_per_frame']:.4g})")
    print("\n================ ③ 接触峰回归 (跨 {} 事件) ================".format(len(R)))
    for tgt, r in reg.items():
        print(f"  {tgt}: corr(approach_vel)={r['corr_approach_vel']:+.3f}  corr(cmd_rough)={r['corr_cmd_rough']:+.3f}  "
              f"corr(grip_speed)={r['corr_grip_speed']:+.3f}  | partial(cmd_rough|vel)={r['partial_cmd_rough|vel']:+.3f}")
    print("\n================ 判定 ================")
    for k, val in verdict.items():
        print(f"  {k}: {val}")
    print(f"\n[saved] {OUT_JSON}")

    # ---- 图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
        # (a) 稳态相干 max coh² 按带/窗
        ws = ["motion", "stationary"]; bs = list(BANDS)
        x = np.arange(len(bs)); wbar = 0.35
        for wi, w in enumerate(ws):
            vals = [coh_report[w].get(b, {}).get("max", 0) for b in bs]
            ax[0].bar(x + wi * wbar, vals, wbar, label=w)
        ax[0].axhline(0.2, color="r", ls="--", lw=1, label="0.2 阈")
        ax[0].set_xticks(x + wbar / 2); ax[0].set_xticklabels(bs)
        ax[0].set(ylabel="max coh² (over in×out pairs)", title="① 命令/速度→IMU 相干 (天花板)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        # (b) motion/floor
        vals = [floor_ratio[b]["motion/floor"] for b in bs]
        ax[1].bar(bs, vals, color="C3")
        ax[1].axhline(1.0, color="k", ls=":", lw=1, label="=floor(无命令驱动)")
        ax[1].set(ylabel="motion HF / stationary floor", title="② 运动段 HF 是否高于外生地板")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        # (c) 接触峰 scatter
        if len(R) >= 8:
            ax[2].scatter(R["cmd_rough"], R["peak_gyro"], s=14, alpha=.5, label="vs cmd_rough(可达)")
            ax2b = ax[2].twiny()
            ax2b.scatter(R["approach_vel"], R["peak_gyro"], s=14, alpha=.5, color="C1", label="vs approach_vel(行为)")
            ax[2].set(xlabel="cmd roughness (HP5 energy)", ylabel="contact peak |gyro| HP5",
                      title="③ 接触峰 ~ 可达 vs 行为")
            ax2b.set_xlabel("approach_vel (rad/s)", color="C1")
            ax[2].legend(fontsize=8, loc="upper left"); ax2b.legend(fontsize=8, loc="upper right")
            ax[2].grid(alpha=.3)
        fig.suptitle(f"#74 Contact-coherence gate — {verdict['decision'][:40]}", fontsize=11)
        fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)
        print(f"[saved] {OUT_PNG}")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
