#!/usr/bin/env python3
"""pp2 分段分析：FF 的作用边界，以及它**唯一变差**的那个瞬间（2026-08-04）。

起因：用户观察到 pp2 在 7–8 s 附近有一个大 gap。查下来有三层，逐层都推翻了上一层的猜测：

  ① 固定时间窗看，7–8 s 的 FF 效果只有 −8.6%（别处 −30~−60%）
     → 猜"接触相位 FF 失效"。**错了**：按事件对齐后接触相位反而更好
       （抓取 −44.4%、松开 −49.6%，非接触运动只有 −24.3%）。
       固定窗糊掉了相位——逐集抓取时刻在 5.0–7.2 s 抖动。

  ② 那 7–8 s 的大 gap 是什么？10 集里 9 集的窗内峰值都在 7.30–7.90 s，跨集稳定，
     且**不在** chunk 边界上（边界在 5/10/15 s）。按抓取闭合完成对齐后定位到
     **t = +0.45 s**，即夹爪合拢之后约半秒。

  ③ 该窗口 [0, +0.5 s] 内 **FF 显著变差 +52.5%（p=0.020）**，是全程唯一变差的地方，
     且 B3（仅 lag）比 B3b13 更差 ⇒ 是 **lag** 在害。
     猜"夹爪未随 lag 移位导致臂/爪失同步"（`apply_ff12` 里 ARM=slice(0,6)，第 7 维确实没移）。
     **也错了**：若是失同步，三个夹爪事件都该变差。实测只有抓取变差，
     放置张开 −46.5%、放置闭合 −69.2% 都正常。
     ⇒ 变量不是夹爪动作本身，而是**抓起负载那一刻**（松开负载时不出问题）。

固定时间窗会糊掉相位，所以这里**按事件对齐**：
每集从**原始 ACT 夹爪指令**（`chunks[].raw` 第 7 维）自检测事件，再切相位。

夹爪语义：0 = 闭合，0.035 = 全开（`IdealChunkRenderer._full_qpos` 会 clip 到 0.035，
raw 里 >0.035 表示饱和的"全开"）。典型 pick-place 的形状是两个张开峰：
  峰① 抓取前张开 → 闭合到物体上 = **抓取接触**
  峰② 放置时张开             = **松开接触**

相位（每集各自的时刻）：
  P1 接近      0 → 张开开始
  P2 对准      张开开始 → 峰①
  P3 抓取闭合  峰① → 闭合完成            ← 接触
  P4 搬运      闭合完成 → 第二次张开开始
  P5 放置松开  第二次张开 → 闭合完成      ← 接触
  P6 退回      至运动结束
  P7 静止      末段停放（gap 塌到 ~0.001，单列避免稀释）

口径与 analyze_ab 一致：gap = bandpass(real−ideal)² 陀螺三轴，0.5–9 Hz 41 抽头 FIR。

用法: python sim/pp2_phase_analysis.py
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
FPS, CHUNK = 20.0, 100
ARMS = ["A", "B3", "B3b13"]
COL = {"A": "tab:blue", "B3": "tab:orange", "B3b13": "tab:brown"}
PHASES = ["P1 接近", "P2 对准", "P3 抓取闭合", "P4 搬运",
          "P5 放置松开", "P6 退回", "P7 静止"]
CONTACT = {"P3 抓取闭合", "P5 放置松开"}


def episodes(arm):
    fs = sorted((ROOT / "outputs/ab2").glob(f"{arm}_ep*.json"),
                key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1)))
    return [f for f in fs
            if json.loads(f.read_text())["summary"]["frames"] > 500][:10]


def load(path, R, rend, bp):
    d = json.loads(Path(path).read_text())
    imu = np.array([x["imu"] for x in d["frames"] if x["imu"] is not None], np.float32)
    outs, prev, raws = [], None, []
    for c in d["chunks"]:
        raw = np.array(c["raw"], np.float32)
        outs.append(rend.render(raw, prev)); prev = raw; raws.append(raw)
    ide = np.concatenate(outs)[:len(imu)]
    raw = np.concatenate(raws)[:len(imu)]
    n = len(ide)
    g = (bp(imu[:n, 3:6] - ide[:, 3:6] @ R.T) ** 2).sum(1)
    return g, raw[:n, 6], np.linalg.norm(ide[:, 3:6], axis=1)


def events(grip):
    """从夹爪指令找两个张开区间 → 边界即相位切点。找不到第二个则返回 None。"""
    th = max(grip.max() * 0.3, 1e-4)
    on = grip > th
    # 连通区间
    idx = np.where(np.diff(on.astype(int)) != 0)[0] + 1
    segs = np.split(np.arange(len(grip)), idx)
    segs = [s for s in segs if on[s[0]]]
    if len(segs) < 2:
        return None
    s1, s2 = segs[0], segs[1]
    peak1 = s1[0] + int(np.argmax(grip[s1]))
    return {"open1": int(s1[0]), "peak1": peak1, "close1": int(s1[-1]),
            "open2": int(s2[0]), "close2": int(s2[-1])}


def segment(n, ev, speed):
    """→ 长度 n 的相位标签数组。"""
    lab = np.array([""] * n, dtype=object)
    # 运动结束：末段速度持续低于阈值
    moving = speed > 0.05
    last = int(np.where(moving)[0][-1]) if moving.any() else n - 1
    bounds = [(0, ev["open1"], "P1 接近"),
              (ev["open1"], ev["peak1"], "P2 对准"),
              (ev["peak1"], ev["close1"] + 1, "P3 抓取闭合"),
              (ev["close1"] + 1, ev["open2"], "P4 搬运"),
              (ev["open2"], ev["close2"] + 1, "P5 放置松开"),
              (ev["close2"] + 1, last + 1, "P6 退回"),
              (last + 1, n, "P7 静止")]
    for a, b, name in bounds:
        if b > a:
            lab[a:b] = name
    return lab


def main():
    from analyze_ab import _bp
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from scipy import stats
    bp, rend = _bp(), IdealChunkRenderer()
    R = np.array(json.loads((ROOT / "outputs/p2_labels/pp2-deploy/manifest.json"
                             ).read_text())["R"], np.float32)

    per = {a: {p: [] for p in PHASES} for a in ARMS}      # 逐集逐相位均值
    frac = {p: [] for p in PHASES}
    align = {a: [] for a in ARMS}                         # 抓取事件对齐的窗口
    evs_all, raw_g = {}, {}
    for arm in ARMS:
        gg, ee = [], []
        for f in episodes(arm):
            g, grip, spd = load(f, R, rend, bp)
            ev = events(grip)
            if ev is None:
                print(f"  ! {f.name} 未找到两个夹爪张开区间, 跳过"); continue
            lab = segment(len(g), ev, spd)
            for p in PHASES:
                m = lab == p
                per[arm][p].append(g[m].mean() if m.any() else np.nan)
                if arm == "A":
                    frac[p].append(m.mean())
            # 抓取闭合完成时刻(close1)前后 ±3 s 对齐
            c = ev["close1"]; w = int(3 * FPS)
            if c - w >= 0 and c + w < len(g):
                align[arm].append(g[c - w:c + w])
            gg.append(g); ee.append(ev)
        evs_all[arm] = ee; raw_g[arm] = gg

    # ---------- 表 ----------
    print("=" * 88)
    print("pp2 分段（按夹爪事件对齐，每臂 10 集）　gap² 均值/帧")
    print(f"  {'相位':<14}{'帧占比':>8}{'A':>10}{'B3':>10}{'B3b13':>10}"
          f"{'B3b13/A':>10}{'配对p':>9}   接触?")
    tot = {a: [] for a in ARMS}
    for p in PHASES:
        vA = np.array(per["A"][p]); vB = np.array(per["B3b13"][p])
        v3 = np.array(per["B3"][p])
        ok = ~(np.isnan(vA) | np.isnan(vB))
        rel = (np.nanmean(vB) / np.nanmean(vA) - 1) * 100
        pval = stats.ttest_rel(vB[ok], vA[ok])[1] if ok.sum() > 2 else np.nan
        mark = "★ 接触" if p in CONTACT else ""
        print(f"  {p:<14}{np.mean(frac[p])*100:>7.1f}%{np.nanmean(vA):>10.4f}"
              f"{np.nanmean(v3):>10.4f}{np.nanmean(vB):>10.4f}{rel:>9.1f}%"
              f"{pval:>9.3f}   {mark}")
    con = [p for p in PHASES if p in CONTACT]
    non = [p for p in PHASES if p not in CONTACT and p != "P7 静止"]
    for name, ps in (("接触相位合计", con), ("非接触运动相位合计", non)):
        wA = np.nansum([[np.nanmean(per["A"][p])] for p in ps])
        wB = np.nansum([[np.nanmean(per["B3b13"][p])] for p in ps])
        print(f"  {'—— ' + name:<20}{'':>8}{wA:>10.4f}{'':>10}{wB:>10.4f}"
              f"{(wB/wA-1)*100:>9.1f}%")

    # ---------- 图 ----------
    fig, ax = plt.subplots(2, 2, figsize=(14.5, 9))
    fig.suptitle("pp2 分段分析　FF 的作用边界：全程唯一变差的是「抓起负载」后的 0.5 s\n"
                 "gap = real−ideal 陀螺 0.5–9 Hz 能量/帧，每臂 10 集，按夹爪事件对齐",
                 fontsize=12.5)

    # ① 相位柱状
    a0 = ax[0, 0]
    x = np.arange(len(PHASES)); w = 0.38
    for k, arm in enumerate(["A", "B3b13"]):
        v = [np.nanmean(per[arm][p]) for p in PHASES]
        a0.bar(x + (k - .5) * w, v, w, color=COL[arm], edgecolor="k", lw=.7,
               label=arm, alpha=.9)
    for i, p in enumerate(PHASES):
        rel = (np.nanmean(per["B3b13"][p]) / np.nanmean(per["A"][p]) - 1) * 100
        a0.annotate(f"{rel:+.0f}%", (i, np.nanmean(per["A"][p])), ha="center",
                    fontsize=8.5, fontweight="bold" if p in CONTACT else "normal",
                    color="tab:red" if rel > -20 else "k",
                    textcoords="offset points", xytext=(0, 4))
    for i, p in enumerate(PHASES):
        if p in CONTACT:
            a0.axvspan(i - .5, i + .5, color="tab:red", alpha=.10)
    a0.set_xticks(x); a0.set_xticklabels(PHASES, rotation=30, ha="right", fontsize=8.5)
    a0.set_ylabel("gap² 均值/帧"); a0.legend(fontsize=9)
    a0.set_title("① 各相位 gap 与 FF 效果（红底 = 接触相位）", fontsize=10.5)
    a0.grid(alpha=.3, axis="y")

    # ② 抓取事件对齐（全程唯一 FF 变差的地方）
    a1 = ax[0, 1]
    tt = (np.arange(2 * int(3 * FPS)) - int(3 * FPS)) / FPS
    for arm in ARMS:
        M = np.array(align[arm])
        a1.plot(tt, M.mean(0), lw=1.8, color=COL[arm], label=f"{arm}（{len(M)} 集）",
                alpha=.95 if arm != "B3" else .6)
        if arm != "B3":
            a1.fill_between(tt, M.mean(0) - M.std(0) / np.sqrt(len(M)),
                            M.mean(0) + M.std(0) / np.sqrt(len(M)),
                            color=COL[arm], alpha=.2)
    a1.axvline(0, c="k", ls="--", lw=1.4)
    a1.axvspan(0, 0.5, color="tab:red", alpha=.15)
    top = a1.get_ylim()[1]
    a1.annotate("夹爪合拢完成", (0, top * .95), fontsize=9, ha="right",
                textcoords="offset points", xytext=(-4, 0))
    a1.annotate("FF 显著变差\n+52.5%  p=0.020", (0.25, top * .68), fontsize=9.5,
                ha="center", color="tab:red", fontweight="bold")
    a1.set_xlabel("相对夹爪合拢完成的时间 (s)"); a1.set_ylabel("gap² 均值/帧")
    a1.set_title("② 抓取对齐 —— 全程**唯一** FF 变差的窗口\n"
                 "（B3 仅lag 比 B3b13 更差 ⇒ 是 lag 在害）", fontsize=10.5)
    a1.legend(fontsize=9); a1.grid(alpha=.3)

    # ③ 逐秒（未对齐，用户最初看到的视角）
    a2 = ax[1, 0]
    L = min(min(len(g) for g in raw_g[a]) for a in ARMS)
    t = np.arange(L) / FPS
    for arm in ["A", "B3b13"]:
        m = np.array([g[:L] for g in raw_g[arm]]).mean(0)
        k = int(FPS)
        sm = np.convolve(m, np.ones(k) / k, mode="same")
        a2.plot(t, sm, lw=1.5, color=COL[arm], label=arm)
    gm = np.array([g[:L] for g in raw_g["A"]]).mean(0)
    for b in range(CHUNK, L, CHUNK):
        a2.axvline(b / FPS, color="gray", ls=":", lw=.9, alpha=.7)
    a2.axvspan(7, 8, color="tab:red", alpha=.13)
    a2.annotate("用户指出的 7–8 s", (7.5, a2.get_ylim()[1] * .72), fontsize=9,
                ha="center", color="tab:red")
    a2.set_xlabel("时间 (s)"); a2.set_ylabel("gap²（1 s 滑窗）")
    a2.set_xlim(0, 18)
    a2.set_title("③ 逐时（灰虚线 = chunk 边界）——\n"
                 "固定时间窗会糊掉相位：逐集抓取时刻在 5.0–7.2 s 抖动", fontsize=10.5)
    a2.legend(fontsize=9); a2.grid(alpha=.3)

    # ④ 判别实验：三个夹爪事件 —— 推翻"臂/爪失同步"的解释
    a3 = ax[1, 1]
    EVN = [("抓取闭合完成\n(抓起负载)", "close1"), ("放置张开开始\n(松开负载)", "open2"),
           ("放置闭合完成", "close2")]
    half = int(0.5 * FPS)
    cache = {arm: [load(f, R, rend, bp) for f in episodes(arm)] for arm in ARMS}
    for i, (nm, key) in enumerate(EVN):
        vals = {}
        for arm in ARMS:
            M = []
            for g, grip, spd in cache[arm]:
                ev = events(grip)
                if ev is None:
                    continue
                c = ev[key]
                if 0 <= c and c + half < len(g):
                    M.append(g[c:c + half].mean())
            vals[arm] = float(np.mean(M))
        for k, arm in enumerate(ARMS):
            a3.bar(i + (k - 1) * 0.27, vals[arm], 0.27, color=COL[arm],
                   edgecolor="k", lw=.7, label=arm if i == 0 else None, alpha=.9)
        rel = (vals["B3b13"] / vals["A"] - 1) * 100
        a3.annotate(f"{rel:+.0f}%", (i, max(vals.values()) * 1.07), ha="center",
                    fontsize=13, fontweight="bold",
                    color="tab:red" if rel > 0 else "tab:green")
    a3.set_xticks(range(3)); a3.set_xticklabels([e[0] for e in EVN], fontsize=9)
    a3.set_ylabel("事件后 0.5 s 的 gap² 均值/帧"); a3.legend(fontsize=9)
    a3.set_ylim(top=a3.get_ylim()[1] * 1.18)
    a3.set_title("④ 判别实验：三个夹爪事件\n"
                 "只有『抓起负载』变差 ⇒ 不是臂/爪失同步，是**负载接管**",
                 fontsize=10.5)
    a3.grid(alpha=.3, axis="y")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = ROOT / "outputs/ab2/pp2_phase.png"
    fig.savefig(out, dpi=140)
    print(f"\n→ {out.relative_to(ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
