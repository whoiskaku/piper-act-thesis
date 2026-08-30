#!/usr/bin/env python3
"""**pick-and-place（pp2）任务域**的对比图 —— 给指导教师的主图。

**先说清楚**：在 pp2 上「旧 G」与「现冻结的 G-π-phys-v1」是**同一个模型**
（逐点差 0.00e+00），故**不存在新旧对比**。本图给出的是任务域的真实状况：
所有测过的候选 + 一条**上界**。

| 线 | 含义 |
|---|---|
| 黄 | 指令（理想，运动学渲染）—— 相对变化的 0% 基准 |
| 蓝 | 现实（真机 IMU） |
| **绿** | **旧 G ＝ 现冻结的 G-π-phys-v1** |
| 紫 | 旧 G + 加速度限幅器（**已判否**：6/17，中位 −13.7 pp） |
| 灰虚 | **喂真实关节角**（B 段上界：17/17，中位 **+97.2 pp**） |

灰虚线回答的是「若关节轨迹正确，还能好到哪」—— 这是本图最重要的一条。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))
from sim.step1_v2_loss import ALIGN_L, BAND, bandpass41, load_calib, render_ideal  # noqa
from sim.step2_gap_baseline import bare_logs                                        # noqa
from sim.step2_fidelity import CALIB, SEAM_W, apply_limiter, regions, fidelity      # noqa
from fair_matrix import run_mj                                                      # noqa
from ab_decompose import kin_gyro                                                   # noqa

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
C_IDEAL, C_REAL, C_DYN, C_NEW = "#eda100", "#2a78d6", "#008300", "#8b2fb0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
OUT = ROOT / "outputs/step2_g/pp2_compare"
REP = "S2s14_bare_ep0"          # 旧 G 主指标最接近中位的集（非择优）
FPS = 20.0


def traces(p, R):
    d = json.loads(p.read_text()); fr = d["frames"]
    act = np.asarray([f["a"] for f in fr], float)
    real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
    enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
    kin = render_ideal(d["chunks"], len(fr))
    _, g0 = run_mj(act, {0: 5000.}, len(fr))
    _, g1 = run_mj(apply_limiter(act, "a"), {0: 5000.}, len(fr))
    n = min(len(fr) - ALIGN_L, len(kin), len(g0), len(g1), len(enc) - ALIGN_L)
    ge = kin_gyro(enc[ALIGN_L:ALIGN_L + n])
    t = dict(ideal=bandpass41(kin[:n, 3:6] @ R.T),
             real=bandpass41(real[ALIGN_L:ALIGN_L + n]),
             old=bandpass41(g0[:n] @ R.T), lim=bandpass41(g1[:n] @ R.T),
             encf=bandpass41(ge[:n] @ R.T),
             seams=[c["start"] - ALIGN_L for c in d["chunks"][1:]
                    if 0 <= c["start"] - ALIGN_L < n], n=n, name=p.stem)
    return t


def rel(t, key, m):
    ei = float(np.square(t["real"][m] - t["ideal"][m]).sum())
    ed = float(np.square(t["real"][m] - t[key][m]).sum())
    return (1 - ed / ei) * 100 if ei > 0 else np.nan


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R, _ = load_calib(CALIB)
    logs = bare_logs()
    allm = []
    for p in logs:
        t = traces(p, R)
        m = np.zeros(t["n"], bool)
        for s in t["seams"]:
            m[s:min(t["n"], s + SEAM_W)] = True
        allm.append([rel(t, k, m) for k in ("ideal", "old", "lim", "encf")])
        if p.stem == REP:
            tr = t
    A = np.array(allm)

    fig = plt.figure(figsize=(14, 10.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.45, 1.15, 1.2], hspace=.60, wspace=.26)
    fig.suptitle("**pick-and-place（pp2 任务域）**　17 集裸 ACT\n"
                 f"{BAND[0]}–{BAND[1]} Hz 陀螺 ω_y　黄=指令(理想) / 蓝=现实 / "
                 "**绿=旧 G ＝ 现冻结的 G-π-phys-v1** / 紫=旧 G+限幅器(已判否) / "
                 "灰虚=**喂真实关节角(上界)**", fontsize=11.5, color=INK)

    n, x = tr["n"], np.arange(tr["n"]) / FPS
    ax = fig.add_subplot(gs[0, :])
    for s in tr["seams"]:
        ax.axvspan(s / FPS, min(n, s + SEAM_W) / FPS, color="#f0efe9", zorder=0)
    ax.plot(x, tr["encf"][:, 1], "--", color="#8a8a86", lw=1.6, zorder=2,
            label="喂真实关节角（上界）")
    ax.plot(x, tr["ideal"][:, 1], color=C_IDEAL, lw=1.8, zorder=3, label="指令（理想）")
    ax.plot(x, tr["real"][:, 1], color=C_REAL, lw=2.0, zorder=5, label="现实（真机 IMU）")
    ax.plot(x, tr["old"][:, 1], color=C_DYN, lw=1.6, zorder=4, label="旧 G（=现冻结底座）")
    ax.plot(x, tr["lim"][:, 1], color=C_NEW, lw=1.5, zorder=4, alpha=.9,
            label="旧 G+限幅器（已判否）")
    ax.set_title(f"① 全程　{REP}（旧 G 主指标最接近中位，**非择优**）　"
                 f"灰带 = 接缝后 {SEAM_W} 帧（难处）", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("时间 (s)", fontsize=8.5, color=INK2)
    ax.set_ylabel("ω_y (rad/s)", fontsize=8.5, color=INK2)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper right")
    ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8)

    e = np.square(tr["real"] - tr["ideal"]).sum(1)
    worst = sorted(tr["seams"], key=lambda s: -e[s:s + SEAM_W].sum())[:3]
    for i, s in enumerate(worst):
        az = fig.add_subplot(gs[1, i])
        lo, hi = max(0, s - 6), min(n, s + 26)
        z = slice(lo, hi)
        az.axvspan(s / FPS, min(n, s + SEAM_W) / FPS, color="#f0efe9", zorder=0)
        az.plot(x[z], tr["encf"][z, 1], "--", color="#8a8a86", lw=1.8, zorder=2)
        az.plot(x[z], tr["ideal"][z, 1], color=C_IDEAL, lw=2.0, zorder=3)
        az.plot(x[z], tr["real"][z, 1], color=C_REAL, lw=2.2, zorder=5)
        az.plot(x[z], tr["old"][z, 1], color=C_DYN, lw=1.8, zorder=4)
        az.plot(x[z], tr["lim"][z, 1], color=C_NEW, lw=1.6, zorder=4, alpha=.9)
        m = np.zeros(n, bool); m[s:min(n, s + SEAM_W)] = True
        az.set_title(f"② 难处 @{s/FPS:.1f}s　旧 G {rel(tr,'old',m):+.0f}% · "
                     f"限幅 {rel(tr,'lim',m):+.0f}% · **真实关节 {rel(tr,'encf',m):+.0f}%**",
                     fontsize=8.8, color=INK, loc="left")
        az.set_xlabel("时间 (s)", fontsize=8, color=INK2)
        az.grid(color=GRID, lw=.5); az.tick_params(labelsize=7.5)
        if i == 0:
            az.set_ylabel("ω_y (rad/s)", fontsize=8.5, color=INK2)

    axb = fig.add_subplot(gs[2, :])
    lab = ["指令\n(0% 基准)", "旧 G\n(=现冻结底座)", "旧 G+限幅器\n(已判否)",
           "喂真实关节角\n(上界)"]
    col = [C_IDEAL, C_DYN, C_NEW, "#8a8a86"]
    pos = np.arange(4)
    for k in range(4):
        axb.scatter(np.full(len(A), pos[k]) + np.linspace(-.16, .16, len(A)),
                    A[:, k], s=26, color=col[k], alpha=.65, zorder=3)
        axb.plot([pos[k]-.3, pos[k]+.3], [np.median(A[:, k])]*2, color=INK, lw=2.4, zorder=4)
        axb.text(pos[k], np.median(A[:, k]) + 5, f"中位 {np.median(A[:,k]):+.1f}%",
                 ha="center", fontsize=9.5, color=INK, fontweight="bold")
    axb.axhline(0, color=INK2, lw=1, ls=":")
    axb.set_xticks(pos); axb.set_xticklabels(lab, fontsize=9)
    axb.set_ylabel("主指标 相对变化 (%)", fontsize=8.5, color=INK2)
    axb.set_title("③ 逐集主指标（接缝后 10 帧）　每点一集，横线为中位　"
                  "**喂真实关节角 17/17 改善、中位 +97.2 pp**；"
                  "限幅器 6/17、中位 −13.7 pp（故判否）",
                  fontsize=9.8, color=INK, loc="left")
    axb.grid(axis="y", color=GRID, lw=.5); axb.tick_params(labelsize=8)
    fig.text(.5, .008, "**在 pp2 上「旧 G」与「现冻结的 G-π-phys-v1」是同一模型"
             "（逐点差 0.00e+00），故无新旧之分。** 灰虚线说明：刚体/传感器链本身没问题 —— "
             "缺口在**指令→关节轨迹**这一段。", ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .028, 1, .93])
    fig.savefig(OUT / "pp2_compare.png", dpi=150, facecolor="#fcfcfb")
    print("中位:", {l.split(chr(10))[0]: round(float(np.median(A[:, k])), 1)
                   for k, l in enumerate(lab)})
    print(f"→ {OUT/'pp2_compare.png'}")


if __name__ == "__main__":
    main()
