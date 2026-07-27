#!/usr/bin/env python3
"""{REPO} 分析: 抓取+对准成功、插入失败 —— 区分两种机理。

执行 = observation.state × 1e-3 (毫度→度); 指令 = action × 57.3 (rad→度)。臂 j1/j2/j3 + 腕 j5。FS=20。
对照 = insert_train demo (人工成功插入)。

H1 指令能力: 执行忠实跟随(corr高/gap小, 无接触), 但指令本身没达成插入 → 失败在策略(欠训), 非偏离。
H2 执行偏离: 指令推向插入但执行落后/卡住(gap变大)+IMU接触尖峰 → 执行跟不进, 前馈/IMU校正相关=论文目标。

插入窗 = "活动段"末尾20%(排除 demo 插入后的静止保持段, 否则窗口取到不动的尾巴)。
用法: /home/enolab-guo/miniconda3/envs/lerobot/bin/python sim/eval_insert_analysis.py
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR = os.path.expanduser("~/.cache/huggingface/lerobot/enolab-guo")
REPO = sys.argv[1] if len(sys.argv) > 1 else "eval_insert_30k"
OUT = os.path.expanduser(f"~/claude-test-2/lerobot-piper/outputs/eval/{REPO}.png")
FS, R2D, MAXLAG = 20.0, 180 / np.pi, 8
ARM = [0, 1, 2, 4]
ARMN = ["j1", "j2", "j3", "j5"]
FRAC = 0.20


def load(repo):
    eps = []
    for p in sorted(glob.glob(f"{LR}/{repo}/data/chunk-*/episode_*.parquet"),
                    key=lambda q: int(q.split("_")[-1][:-8])):
        df = pd.read_parquet(p)
        act = np.stack(df["action"].to_numpy()).astype(float) * R2D
        st = np.stack(df["observation.state"].to_numpy()).astype(float) * 1e-3
        im = np.stack(df["observation.imu_real"].to_numpy()).astype(float)
        eps.append((act, st, im))
    return eps


def active_end(cmd_arm):
    """最后一个"还在动"的帧 (排除终态静止保持段)。cmd_arm: (T, |ARM|)。"""
    v = np.abs(np.diff(cmd_arm, axis=0)).sum(1)
    v = np.convolve(v, np.ones(9) / 9, "same")
    if v.max() < 1e-6:
        return len(cmd_arm)
    idx = np.where(v > 0.1 * np.percentile(v, 95))[0]
    return (idx[-1] + 2) if len(idx) else len(cmd_arm)


def insert_window(cmd_arm):
    ae = active_end(cmd_arm)
    return int(ae * (1 - FRAC)), ae


def best_lag(e, c):
    best_d, best_r = 0, -2.0
    for d in range(MAXLAG + 1):
        a, b = e[d:], c[:len(c) - d]
        if len(a) < 20 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r, best_d = r, d
    return best_d, best_r


def imu_mag(im, a, b):
    seg = im[a:b, :3]
    return float(np.linalg.norm(seg - seg.mean(0), axis=1).mean()) if b > a else float("nan")


def tail_motion(x, a, b):
    return float(x[a:b].max() - x[a:b].min()) if b > a else 0.0


def main():
    ev = load(REPO)
    demo = load("insert_train")
    print(f"\n{'='*72}\n{REPO} 分析 (N={len(ev)}, 无TE) vs demo(N={len(demo)})\n{'='*72}")

    # demo 插入signature: 活动段末尾20% 指令活动幅度
    dem = {j: [] for j in ARM}
    dem_corr = []
    for act, st, im in demo:
        a, b = insert_window(act[:, ARM])
        for k, j in enumerate(ARM):
            dem[j].append(tail_motion(act[:, j], a, b))
        dem_corr.append(np.mean([best_lag(st[:, j], act[:, j])[1] for j in ARM]))
    print("\ndemo 成功插入 — 活动段末尾20% 指令活动幅度(°,中位):")
    print("   " + "  ".join(f"{ARMN[k]}={np.median(dem[j]):.1f}" for k, j in enumerate(ARM)))
    print(f"demo 执行保真 corr(执行,指令) 中位 = {np.median(dem_corr):.3f}  (机器人本身跟随能力基线)")

    print(f"\n{'-'*72}\n逐条 eval:")
    summ = []
    for i, (act, st, im) in enumerate(ev):
        a, b = insert_window(act[:, ARM]); n = len(act)
        corr = [best_lag(st[:, j], act[:, j])[1] for j in ARM]
        lag = [best_lag(st[:, j], act[:, j])[0] for j in ARM]
        gap_all = [np.abs(st[:, j] - act[:, j]).mean() for j in ARM]
        gap_w = [np.abs(st[a:b, j] - act[a:b, j]).mean() for j in ARM]
        cmot = [tail_motion(act[:, j], a, b) for j in ARM]
        ie, iw = imu_mag(im, 0, a), imu_mag(im, a, b)
        summ.append((corr, gap_w, cmot, ie, iw))
        print(f"\n ep{i} ({n/FS:.0f}s, 插入窗 {a/FS:.0f}-{b/FS:.0f}s):")
        print(f"   corr臂均={np.mean(corr):.3f}  滞后={np.mean(lag)/FS*1e3:.0f}ms  整体|执行-指令|={np.mean(gap_all):.2f}°")
        print(f"   插入窗 指令活动幅度°: " + " ".join(f"{ARMN[k]}={cmot[k]:.1f}" for k in range(len(ARM))))
        print(f"   插入窗 |执行-指令|° : " + " ".join(f"{ARMN[k]}={gap_w[k]:.2f}" for k in range(len(ARM))))
        print(f"   IMU accel: 前段={ie:.2f} 插入窗={iw:.2f} (窗/前={iw/ie if ie>1e-9 else float('nan'):.2f}×)")

    corr_m = np.median([np.mean(s[0]) for s in summ])
    gapw_m = np.median([np.mean(s[1]) for s in summ])
    imu_r = np.median([s[4] / s[3] if s[3] > 1e-9 else np.nan for s in summ])
    cm_ev = np.median([[s[2][k] for k in range(len(ARM))] for s in summ], 0)
    cm_dm = np.array([np.median(dem[j]) for j in ARM])
    print(f"\n{'='*72}\n判读:")
    print(f"  执行保真: corr={corr_m:.3f}, 插入窗|执行-指令|={gapw_m:.2f}°, IMU窗/前={imu_r:.2f}×")
    print(f"  插入窗指令活动幅度  eval中位=[{' '.join(f'{x:.1f}' for x in cm_ev)}]°  demo中位=[{' '.join(f'{x:.1f}' for x in cm_dm)}]°")
    if corr_m > 0.95 and gapw_m < 2 and imu_r < 1.3:
        print("  → H1(指令能力): 执行近乎完美跟随指令(corr~1, gap<2°, 无接触尖峰)。")
        print("    插入失败 = 策略指令本身未达成插入(30K欠训), 不是执行偏离/过冲/卡阻。")
        print("    与Phase1一致: 这套真机执行保真度高, 瓶颈在策略而非执行偏离。等100K再评。")
    else:
        print("  → 存在执行偏离/接触迹象, 见逐条与图。")

    # ---- 图 (英文标签) ----
    nf = len(ev) + 1
    fig, ax = plt.subplots(nf, 2, figsize=(15, 3.0 * nf))
    jsel = ARM[int(np.argmax(cm_dm))]; jn = ARMN[ARM.index(jsel)]
    def panel(r, act, st, im, tag, col):
        n = len(act); t = np.arange(n) / FS; a, b = insert_window(act[:, ARM])
        ax[r, 0].plot(t, act[:, jsel], "C0", lw=1.3, label=f"command {jn}")
        ax[r, 0].plot(t, st[:, jsel], "C3", lw=1.1, label=f"executed {jn}")
        ax[r, 0].axvspan(a / FS, b / FS, color=col, alpha=.15, label="insertion window")
        ax[r, 0].set(title=f"{tag}: {jn} command vs executed", ylabel="deg"); ax[r, 0].legend(fontsize=7); ax[r, 0].grid(alpha=.3)
        am = np.linalg.norm(im[:, :3] - im[:, :3].mean(0), axis=1)
        ax[r, 1].plot(t, am, "C2", lw=.7); ax[r, 1].axvspan(a / FS, b / FS, color=col, alpha=.15)
        ax[r, 1].set(title=f"{tag}: IMU accel mag", ylabel="m/s^2"); ax[r, 1].grid(alpha=.3)
    for i, e in enumerate(ev):
        panel(i, *e, f"eval ep{i}", "orange")
    panel(len(ev), *demo[0], "demo ep0 (insert OK)", "green")
    ax[-1, 0].set_xlabel("s"); ax[-1, 1].set_xlabel("s")
    fig.suptitle(f"{REPO}: command vs executed + IMU (vs demo)", fontsize=12)
    fig.tight_layout(); os.makedirs(os.path.dirname(OUT), exist_ok=True); fig.savefig(OUT, dpi=130)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
