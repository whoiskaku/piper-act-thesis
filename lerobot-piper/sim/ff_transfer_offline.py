#!/usr/bin/env python3
"""FF 能否迁移到 insert？—— 零机时的离线前置判断（2026-07-29）。

pp2 上 FF(lag=3, blend=13) 实测 −33.0%。要判断这是"pp2 上的参数修正"还是"方法"，
决定性的两个量都可以在**已录的 eval 数据**上离线量出来，不用真机：

  ① 被控对象滞后 L*   —— 若 insert 的 L* ≠ 3，`ff_lag=3` 对 insert 就是错的，
                         FF 需要逐任务标定，"方法"的说法要减弱
  ② chunk 接缝瞬态    —— blend 的作用面。若 insert 没有接缝超出，blend 无从获益

③ 再给一个**有锚点的**外推：离线把比较窗口移位 L* 帧所消掉的 gap 能量，在 pp2 上
   与真机 lag 单项实测(−22.5%, B3 vs A)相除得到"传递率"，用同一传递率外推 insert。

**这不是预测真机结果**，只是把"离线可测的那一半"量出来。离线移位改的是**比较对齐**，
FF 改的是**发出的指令**，两者等价需要被控对象对时移指令给出时移响应(LTI、不限速)。
pp2 的传递率正是这条假设成立程度的直接测量 —— 所以它必须一起报出来。

口径与 A/B 表一致：gap = bandpass(real − ideal) 陀螺三轴，0.5–9 Hz 41 抽头 FIR，
ideal 由 `action` 逐 100 帧 chunk 运动学渲染后经 manifest 的 R 转到真机系。不剔接触帧。

用法: python sim/ff_transfer_offline.py [--eps 10]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import sys
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
FPS, CHUNK, SMAX = 20.0, 100, 6
TAGS = [("pp2-deploy", "pp2（已实测 FF）"), ("ins-deploy", "insert（未测）")]
# pp2 真机实测锚点（同口径，10 集配对）
PP2_A, PP2_B3 = 0.05625, 0.04360          # 裸 ACT / FF lag=3 有效blend2


def load_ep(repo, ep, R):
    """→ (ideal_gyro(n,3) 真机系, real_gyro(n,3))，**未做 LSHIFT 对齐**。"""
    from build_dagger_labels import kin_ideal_chunked
    kin, d, _ = kin_ideal_chunked(repo, ep)
    T = min(len(d["imu_real"]), len(kin))
    return (kin[:T, 3:] @ R.T).astype(np.float32), d["imu_real"][:T, 3:].astype(np.float32)


def shift_curve(ideal, real, bp):
    """gap 能量/帧 随比较移位 s 的变化（各 s 用等长窗口，可比）。"""
    n = len(ideal) - SMAX
    return np.array([float((bp(real[s:s + n] - ideal[:n]) ** 2).sum() / n)
                     for s in range(SMAX + 1)])


def parabolic_min(y):
    """整数最小点邻域抛物线插值 → 亚帧最优移位。"""
    k = int(np.argmin(y))
    if k in (0, len(y) - 1):
        return float(k), True                      # 落在边界，不可信
    a, b, c = y[k - 1], y[k], y[k + 1]
    den = a - 2 * b + c
    return (float(k) if abs(den) < 1e-12 else k + 0.5 * (a - c) / den), False


def chunk_profile(ideal, real, bp, s):
    """gap² 按 chunk 内位置(0..99)聚合 —— 接缝瞬态的形状。"""
    n = len(ideal) - SMAX
    g = bp(real[s:s + n] - ideal[:n])
    e = (g ** 2).sum(1)
    pos = np.arange(n) % CHUNK
    return np.array([e[pos == p].mean() if (pos == p).any() else np.nan
                     for p in range(CHUNK)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=int, default=10)
    a_ = ap.parse_args()
    from analyze_ab import _bp
    bp = _bp()

    res = {}
    for tag, label in TAGS:
        man = json.loads((ROOT / "outputs/p2_labels" / tag / "manifest.json").read_text())
        R = np.array(man["R"], np.float32)
        curves, profs = [], []
        for ep in range(a_.eps):
            ideal, real = load_ep(man["repo"], ep, R)
            curves.append(shift_curve(ideal, real, bp))
            profs.append((ideal, real))
        C = np.array(curves)                                  # (eps, SMAX+1)
        Lstar, edge = parabolic_min(C.mean(0))
        Li = int(round(Lstar))
        prof = np.nanmean([chunk_profile(i_, r_, bp, Li) for i_, r_ in profs], 0)
        res[tag] = {"label": label, "C": C, "Lstar": Lstar, "edge": edge, "Li": Li,
                    "prof": prof, "repo": man["repo"]}
        # 逐集 L*（离散度 → 这个滞后稳不稳）
        res[tag]["Lep"] = [parabolic_min(c)[0] for c in C]

    # ---------------- 打印 ----------------
    print("=" * 78)
    print("① 被控对象滞后 L*（离线移位扫描，10 集平均曲线的抛物线插值）")
    print(f"  {'任务':>16}{'L* (帧)':>10}{'= ms':>8}{'逐集 L* 中位':>13}"
          f"{'逐集 sd':>9}{'边界?':>7}")
    for tag, r in res.items():
        lep = np.array(r["Lep"])
        print(f"  {r['label']:>16}{r['Lstar']:>10.2f}{r['Lstar']*50:>8.0f}"
              f"{np.median(lep):>13.2f}{lep.std(ddof=1):>9.2f}"
              f"{('是' if r['edge'] else '否'):>7}")
    dL = res["ins-deploy"]["Lstar"] - res["pp2-deploy"]["Lstar"]
    print(f"\n  Δ(insert − pp2) = {dL:+.2f} 帧 = {dL*50:+.0f} ms")
    print(f"  ⇒ ff_lag=3 对 insert " +
          ("**合适**（同一整数帧）" if round(res['ins-deploy']['Lstar']) == 3
           else f"**不合适**，应为 {round(res['ins-deploy']['Lstar'])}"))

    print("\n" + "=" * 78)
    print("② chunk 接缝瞬态（gap² 按 chunk 内位置聚合，各任务用自己的 L*）")
    print(f"  {'任务':>16}{'开头10帧':>11}{'中段(30-90)':>13}{'倍率':>8}"
          f"{'末尾3帧':>10}{'倍率':>8}")
    for tag, r in res.items():
        p = r["prof"]
        head, mid, tail = p[:10].mean(), p[30:90].mean(), p[-3:].mean()
        r["ratio_head"] = head / mid
        print(f"  {r['label']:>16}{head:>11.5f}{mid:>13.5f}{head/mid:>8.2f}×"
              f"{tail:>10.5f}{tail/mid:>8.2f}×")

    print("\n" + "=" * 78)
    print("③ 离线可消减量 + pp2 传递率锚点")
    print(f"  {'任务':>16}{'s=0':>10}{'s=L*':>10}{'离线消减':>10}")
    for tag, r in res.items():
        m = r["C"].mean(0)
        r["off0"], r["offL"] = m[0], m.min()
        r["off_red"] = 1 - m.min() / m[0]
        print(f"  {r['label']:>16}{m[0]:>10.5f}{m.min():>10.5f}"
              f"{r['off_red']*100:>9.1f}%")
    real_red = 1 - PP2_B3 / PP2_A
    trans = real_red / res["pp2-deploy"]["off_red"]
    print(f"\n  pp2 真机 lag 单项实测:  {PP2_A:.5f} → {PP2_B3:.5f} = "
          f"−{real_red*100:.1f}%   (B3 vs A)")
    print(f"  pp2 离线移位可消减:                        "
          f"−{res['pp2-deploy']['off_red']*100:.1f}%")
    print(f"  ⇒ **传递率 = {trans:.2f}**  "
          f"（真机拿到离线上限的 {trans*100:.0f}%）")
    ins_pred = res["ins-deploy"]["off_red"] * trans
    print(f"\n  按同一传递率外推 insert 的 lag 单项:  −{ins_pred*100:.1f}%"
          f"   ← **外推，非预测；insert 的传递率未测**")

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.suptitle("FF 迁移到 insert 的离线前置判断（零机时；已录 eval 数据，10 集）",
                 fontsize=12)
    col = {"pp2-deploy": "tab:blue", "ins-deploy": "tab:red"}
    for tag, r in res.items():
        m, sd = r["C"].mean(0), r["C"].std(0, ddof=1) / np.sqrt(len(r["C"]))
        x = np.arange(SMAX + 1)
        ax[0].errorbar(x, m / m[0], yerr=sd / m[0], fmt="o-", capsize=4, lw=1.8,
                       color=col[tag], label=f"{r['label']}  L*={r['Lstar']:.2f}")
        ax[0].axvline(r["Lstar"], ls=":", color=col[tag], lw=1.3)
        ax[1].plot(r["prof"] / r["prof"][30:90].mean(), lw=1.2, color=col[tag],
                   label=f"{r['label']}  开头 {r['ratio_head']:.1f}×")
    ax[0].axvline(3, ls="--", c="k", lw=1.2, label="当前 ff_lag=3")
    ax[0].set_xlabel("比较移位 s（帧）"); ax[0].set_ylabel("gap 能量 / s=0")
    ax[0].set_title("① 滞后：移位扫描（各自归一到 s=0）", fontsize=10)
    ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3)

    ax[1].axhline(1, c="k", lw=.8)
    ax[1].axvspan(0, 10, color="gray", alpha=.15)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("chunk 内位置（帧）"); ax[1].set_ylabel("gap² / 中段均值")
    ax[1].set_title("② 接缝：chunk 内位置剖面（灰 = blend 作用面）", fontsize=10)
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)

    lab = ["pp2\n离线上限", "pp2\n真机实测", "insert\n离线上限", "insert\n按传递率外推"]
    val = [res["pp2-deploy"]["off_red"] * 100, real_red * 100,
           res["ins-deploy"]["off_red"] * 100, ins_pred * 100]
    cs = ["tab:blue", "tab:blue", "tab:red", "tab:red"]
    hat = ["///", "", "///", "xx"]
    for i, (v, c, h) in enumerate(zip(val, cs, hat)):
        ax[2].bar(i, v, color=c, alpha=.55 if h else .9, hatch=h, edgecolor="k", lw=.8)
        ax[2].annotate(f"{v:.1f}%", (i, v), ha="center",
                       textcoords="offset points", xytext=(0, 4), fontsize=9)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(lab, fontsize=8.5)
    ax[2].set_ylabel("lag 单项的 gap 消减 (%)")
    ax[2].set_title(f"③ lag 单项：离线上限 vs 真机\n传递率 {trans:.2f}"
                    f"（斜线=离线上限, ××=外推）", fontsize=10)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "outputs/ab2/FF_transfer_offline.png"
    fig.savefig(out, dpi=140)
    print(f"\n→ {out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
