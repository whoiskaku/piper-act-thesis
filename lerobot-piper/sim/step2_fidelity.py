#!/usr/bin/env python3
"""Step 2 `legacy_development` 保真度诊断 —— 当前手调 XML 基线。

本脚本复用已经查看过的 17 个 Step 1 裸 ACT 日志和手调伺服参数；它不是自动标定
G0，也不是 held-out/confirmatory 验收。输出只能用于定位当前 XML 的开发问题，
不能支持或否定未来 G0。

老师 2026-07-15 面談对 Ĝ 评价的三条要求, 本脚本逐条实现:

  ①「看平均值接近没有意义」「光是それっぽい指標好没有意义」
     ⇒ 全程平均**降为内部质检**, 不作主指标。
  ②「大部分地方波形本来就吻合 → 整体吻合度虚高」
     ⇒ 必须**分区**报告, 并把「本来就吻合」的区域单列, 让虚高可见。
     实测(S1s01, 582 帧): 能量**后 50% 的帧只含 3.3% 的能量** —— 老师说的就是这个。
  ③ 呈现用**波形叠图**(黄=理想 / 蓝=现实), 要看**难处/偏离真正发生处**准不准。
     本脚本加第三条(绿=当前手调 XML 预测)，只作 legacy 开发诊断。

难处的两种定义(都**不**由 XML 的表现决定, 只由 real-vs-ideal 的 gap 决定, 故不循环):
  · **机理定义**: 接缝后 10 帧 —— 可事前声明, 对应已知的 chunk 边界跳变
  · **能量定义**: real-vs-ideal gap 能量前 10% 的帧 —— 问题真正所在

配色经 dataviz 校验器全对比校验(light, surface #fcfcfb):
  CVD 最差 ΔE 16.2(protan) / 常视 29.0 —— 三条叠加仍可分。
  黄对表面对比 2.11 < 3:1 ⇒ 按规程以**直接标注 + 数值表**兜底。

用法:
    python sim/step2_fidelity.py                    # 全部裸 ACT 集 + 代表集波形
    python sim/step2_fidelity.py --ep S1s01_bare    # 指定波形集
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from sim.step1_v2_loss import ALIGN_L, BAND, bandpass41, load_calib, render_ideal  # noqa: E402
from sim.step2_gap_baseline import aa_down, bare_logs, dyn_replay_cmd  # noqa: E402

sys.path.insert(0, str(ROOT / "sim/step2_g"))
from accel_limit import accel_limited, pure_delay  # noqa: E402

# ── 固件 MOVE J 的运动学限幅（2026-08-26 加）────────────────────────────────
# 87→90 条激励套件测出: 指令与关节之间有一个**加速度受限的运动学轨迹生成器**,
# 它在 0.3–6 Hz 激励带内主导响应, 而当前 G(线性位置伺服)完全没有这一项。
# 逐关节参数在**大幅值真阶跃**上拟合(见 accel_limit.fit_ad / mechanism.py):
#   · 只有大幅值档能约束 a(小幅值时限幅几乎不触发, 留出验证 0.53–0.62° vs 0.05–0.19°)
#   · j4 的 a 弱可辨识(±60% 只让 RMS 变 1.3×, 其余 3.7–18.6×), 数值仅供占位
A_MAX = np.array([319.0, 319.0, 307.0, 346.0, 391.0, 307.0])      # °/s²
D_SEC = np.array([0.075, 0.075, 0.070, 0.065, 0.090, 0.075])      # s, 纯传输延迟


def apply_limiter(act: np.ndarray, mode: str) -> np.ndarray:
    """指令 → 固件 MOVE J 限幅 → 再喂 MuJoCo。这是「串联」而非「二选一」。

    mode: "none" 不改 · "a" 只加速度限 · "ad" 加速度限 + 纯延迟。
    "a" 与 "ad" 分开是因为 MuJoCo 伺服自身也有滞后(j1 的 kp=5000 当初正是为了
    凑出 2 帧滞后而手调的), 再叠一个 65–90 ms 纯延迟可能**重复计入**。
    """
    if mode == "none":
        return act
    out = np.array(act, float, copy=True)
    for j in range(6):
        y = accel_limited(np.degrees(act[:, j]), A_MAX[j])
        if mode == "ad":
            y = pure_delay(y, D_SEC[j])
        out[:, j] = np.radians(y)
    return out

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# dataviz 校验通过的三色(黄=理想 / 蓝=现实 / 绿=仿真)
C_IDEAL, C_REAL, C_DYN = "#eda100", "#2a78d6", "#008300"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
FPS = 20.0
SEAM_W = 10                 # 机理定义的难处宽度(帧)
TOP_FRAC = 0.10             # 能量定义的难处比例
CALIB = ROOT / "outputs/p2_labels/pp2a-deploy/manifest.json"
OUT = ROOT / "outputs/step2_g/legacy_fidelity"


def episode_traces(log: Path, R: np.ndarray, limiter: str = "none",
                   kpmul: float | None = None):
    """→ dict: 三条带通后的陀螺轨迹(已对齐) + 接缝位置。"""
    d = json.loads(log.read_text())
    fr = d["frames"]
    act = np.asarray([f["a"] for f in fr], float)
    real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
    kin = render_ideal(d["chunks"], len(fr))
    n = min(len(fr) - ALIGN_L, len(kin))
    import mujoco as _mj
    from sim.step2_gap_baseline import MODEL as _MODEL
    _m0 = _mj.MjModel.from_xml_path(str(_MODEL))
    jkp = ({j: float(_m0.actuator_gainprm[j, 0]) * kpmul for j in range(6)}
           if kpmul is not None else {0: 5000.0})
    hi, fs = dyn_replay_cmd(apply_limiter(act, limiter), jkp)
    dyn = aa_down(hi, fs, len(fr))[:, 3:6] @ R.T
    n = min(n, len(dyn))
    return {
        "ideal": bandpass41(kin[:n, 3:6] @ R.T),
        "real": bandpass41(real[ALIGN_L:ALIGN_L + n]),
        "dyn": bandpass41(dyn[:n]),
        "seams": [c["start"] - ALIGN_L for c in d["chunks"][1:]
                  if 0 <= c["start"] - ALIGN_L < n],
        "n": n, "name": log.stem,
    }


def regions(tr: dict) -> dict[str, np.ndarray]:
    """四个区域的布尔掩码。难处只由 real-vs-ideal 定义, 与 G 无关。"""
    n = tr["n"]
    e = np.square(tr["real"] - tr["ideal"]).sum(1)
    seam = np.zeros(n, bool)
    for s in tr["seams"]:
        seam[s:min(n, s + SEAM_W)] = True
    k = int(n * TOP_FRAC)
    top = np.zeros(n, bool)
    top[np.argsort(e)[::-1][:k]] = True
    bot = np.zeros(n, bool)
    bot[np.argsort(e)[: n // 2]] = True
    return {"全程": np.ones(n, bool), f"接缝后{SEAM_W}帧": seam,
            f"gap能量前{int(TOP_FRAC*100)}%": top, "gap能量后50%": bot}


def fidelity(tr: dict, m: np.ndarray) -> tuple[float, float, float]:
    """→ (运动学 gap 能量, 手调 XML 残差能量, 开发集相对变化)。"""
    ei = float(np.square(tr["real"][m] - tr["ideal"][m]).sum())
    ed = float(np.square(tr["real"][m] - tr["dyn"][m]).sum())
    return ei, ed, 1 - ed / ei if ei > 0 else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", help="画波形的集(stem)")
    ap.add_argument("--axis", type=int, default=1, help="陀螺轴 0/1/2")
    ap.add_argument("--n", type=int, help="只算前 N 集")
    ap.add_argument("--limiter", choices=("none", "a", "ad"), default="none",
                    help="在 MuJoCo 之前插入固件 MOVE J 的加速度限幅")
    ap.add_argument("--kpmul", type=float, default=None,
                    help="把**全部** kp 乘以该倍数(覆盖 j1 的 5000 手调值)。"
                         "真机静态刚度 ≥1.5e5 N·m/rad(无重力垂落), 而 XML 的 "
                         "kp=200 预测 1.51° 垂落 ⇒ 限幅器在前之后, 后级伺服应更硬。")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R, _ = load_calib(CALIB)
    logs = bare_logs()[: a.n]

    # ── 分区保真度(全部裸 ACT 集汇总) ───────────────────────────────
    print("⚠ legacy_development：当前手调 XML 基线，不是自动标定 G0 或最终验收")
    print(f"分区保真度开发诊断 —— {len(logs)} 集已查看的裸 ACT")
    print(f"口径 {BAND[0]}–{BAND[1]} Hz 陀螺三轴;  黄=理想 蓝=现实 绿=手调 XML")
    print(f"限幅器: {{'none':'无(当前 G)','a':'加速度限','ad':'加速度限+纯延迟'}}[a.limiter]"
          .replace('{a.limiter}', a.limiter) if False else
          f"限幅器: {'无(当前 G)' if a.limiter=='none' else ('加速度限' if a.limiter=='a' else '加速度限+纯延迟')}\n")
    acc: dict[str, list] = {}
    traces = []
    for p in logs:
        tr = episode_traces(p, R, a.limiter, a.kpmul)
        traces.append(tr)
        for nm, m in regions(tr).items():
            acc.setdefault(nm, []).append(fidelity(tr, m))
    print(f"{'区域':<18}{'占帧':>7}{'占能量':>8}{'待解释gap':>11}{'XML剩下':>10}{'相对变化':>9}")
    agg = {}
    for nm in acc:
        A = np.array(acc[nm])
        ei, ed = A[:, 0].sum(), A[:, 1].sum()
        tot = sum(fidelity(t, regions(t)["全程"])[0] for t in traces)
        fr_ = np.mean([regions(t)[nm].mean() for t in traces])
        agg[nm] = 1 - ed / ei
        print(f"{nm:<18}{fr_*100:>6.1f}%{ei/tot*100:>7.1f}%{ei:>11.2f}{ed:>10.2f}"
              f"{agg[nm]*100:>+8.1f}%")

    print(f"\n【老师②的验证】「大部分地方本来就吻合」:")
    print(f"  gap能量后50%的帧 —— 占一半帧数, 却只含 "
          f"{np.mean([np.square(t['real'][regions(t)['gap能量后50%']]-t['ideal'][regions(t)['gap能量后50%']]).sum()/np.square(t['real']-t['ideal']).sum() for t in traces])*100:.1f}% 的能量。")
    print(f"  ⇒ 任何全程平均都被这些「没什么要解释」的帧稀释 —— 老师的批评成立。")
    print(f"\n【开发诊断；不是 G0 主结论】")
    hard = f"接缝后{SEAM_W}帧"
    print(f"  难处(机理) 相对变化 = **{agg[hard]*100:+.1f}%**")
    print(f"  难处(能量) 相对变化 = **{agg[f'gap能量前{int(TOP_FRAC*100)}%']*100:+.1f}%**")
    print(f"  (全程 {agg['全程']*100:+.1f}% —— **内部质检用, 不作主指标**)")
    print(f"\n  以历史 42% 参考线看，当前手调 XML 的难处相对变化为负；"
          f"这只说明该开发基线比运动学基线差，不能支持或否定 G0。")
    print(f"\n  注: 相对变化 0% **就是**运动学基线(令 dyn≡ideal 则恒等于 0), "
          f"故负值 = 比零参数运动学假设更差。")

    # ── 内部质检: 逐集最优滞后补偿后的同一批数字 ──────────────────
    print(f"\n【内部质检, 不作汇报】给 dyn 逐集补最优滞后后的同一批数字:")
    print(f"  给模型对齐滞后是一种**美化** —— 标定正确时本不该需要。两者都列以免误导。")
    hdr = True
    lagged: dict[str, list] = {}
    lags = []
    for tr_ in traces:
        best, bl = np.inf, 0
        for L in range(9):
            nn = tr_["n"] - L
            v = float(np.square(tr_["real"][L:L + nn] - tr_["dyn"][:nn]).sum())
            if v < best:
                best, bl = v, L
        lags.append(bl)
        sh = dict(tr_)
        nn = tr_["n"] - bl
        sh["n"] = nn
        sh["real"] = tr_["real"][bl:bl + nn]
        sh["ideal"] = tr_["ideal"][bl:bl + nn]
        sh["dyn"] = tr_["dyn"][:nn]
        sh["seams"] = [s - bl for s in tr_["seams"] if 0 <= s - bl < nn]
        for nm, m in regions(sh).items():
            lagged.setdefault(nm, []).append(fidelity(sh, m))
    for nm in lagged:
        A_ = np.array(lagged[nm])
        print(f"    {nm:<18}{(1 - A_[:, 1].sum() / A_[:, 0].sum())*100:>+8.1f}%"
              + ("   ← 主指标的补滞后版" if nm.startswith("接缝") else ""))
    print(f"    逐集最优滞后中位 {np.median(lags):.1f} 帧 —— 标定正确时应为 0")

    # ── 波形叠图 ────────────────────────────────────────────────
    tr = next((t for t in traces if t["name"].startswith(a.ep)), traces[0]) if a.ep else traces[0]
    J = a.axis
    n, t = tr["n"], np.arange(tr["n"]) / FPS
    e = np.square(tr["real"] - tr["ideal"]).sum(1)
    worst = sorted(tr["seams"], key=lambda s: -e[s:s + SEAM_W].sum())[:3]

    fig = plt.figure(figsize=(13, 11))
    gs = fig.gridspec = fig.add_gridspec(4, 3, height_ratios=[2.4, 1.6, 1.6, 0.05],
                                         hspace=0.42, wspace=0.22)
    fig.suptitle(f"Step 2 · 手调 XML legacy 诊断（波形叠图 · 看难处 · 不看平均）\n"
                 f"{tr['name']}　{BAND[0]}–{BAND[1]} Hz 陀螺 {['ω_x','ω_y','ω_z'][J]}"
                 f"　黄=理想 / 蓝=现实 / 绿=手调 XML（legacy）",
                 fontsize=12, color=INK)

    ax = fig.add_subplot(gs[0, :])
    for s in tr["seams"]:
        ax.axvspan(s / FPS, min(n, s + SEAM_W) / FPS, color="#f0efe9", zorder=0)
    ax.plot(t, tr["ideal"][:, J], color=C_IDEAL, lw=2.4, label="理想（运动学）", zorder=3)
    ax.plot(t, tr["real"][:, J], color=C_REAL, lw=2.0, label="现实（真机 IMU）", zorder=4)
    ax.plot(t, tr["dyn"][:, J], color=C_DYN, lw=1.6, label="手调 XML 预测", zorder=5)
    for nm, yv, c, fx in (("理想", tr["ideal"], C_IDEAL, 0.045),
                          ("现实", tr["real"], C_REAL, 0.085),
                          ("手调XML", tr["dyn"], C_DYN, 0.125)):
        k = int(n * fx)
        ax.annotate(nm, (t[k], yv[k, J]), color=c, fontsize=9.5, fontweight="bold",
                    xytext=(2, 7), textcoords="offset points")
    ax.set_title("① 全程　灰带 = 接缝后 10 帧（难处）", fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("时间 (s)", color=INK2)
    ax.set_ylabel("角速度 (rad/s)", color=INK2)
    ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper right")

    for i, s in enumerate(worst):
        axz = fig.add_subplot(gs[1, i])
        lo, hi_ = max(0, s - 6), min(n, s + 26)
        z = slice(lo, hi_)
        axz.axvspan(s / FPS, min(n, s + SEAM_W) / FPS, color="#f0efe9", zorder=0)
        axz.plot(t[z], tr["ideal"][z, J], color=C_IDEAL, lw=2.6, zorder=3)
        axz.plot(t[z], tr["real"][z, J], color=C_REAL, lw=2.2, zorder=4)
        axz.plot(t[z], tr["dyn"][z, J], color=C_DYN, lw=1.8, zorder=5)
        m = np.zeros(n, bool); m[s:min(n, s + SEAM_W)] = True
        _, _, f = fidelity(tr, m)
        axz.set_title(f"② 难处放大 · 接缝 @{s/FPS:.1f}s　相对变化 {f*100:+.0f}%",
                      fontsize=9.5, color=INK, loc="left")
        axz.set_xlabel("时间 (s)", color=INK2, fontsize=9)
        if i == 0:
            axz.set_ylabel("角速度 (rad/s)", color=INK2)

    axb = fig.add_subplot(gs[2, :])
    names = list(agg)
    vals = [agg[k] * 100 for k in names]
    cols = [C_DYN if v > 0 else "#c0392b" for v in vals]
    y = np.arange(len(names))
    axb.barh(y, vals, color=cols, height=0.55)
    axb.axvline(0, color=INK, lw=1.2)
    # 参照线: 只有两个有意义的锚 —— 天花板与「不建模」。
    # 42% 是**旧 Ĝ 达到并被老师否决**的成绩, 不是及格线, 故画成「已被否决」而非门槛。
    axb.axvline(42, color="#c0392b", lw=1.3, ls=":")
    axb.annotate("旧 Ĝ 曾达 42%\n—— 老师判「使いようがない」\n（是被否决的成绩，不是及格线）",
                 (42, len(names) - 0.42), fontsize=8, color="#c0392b",
                 xytext=(6, 0), textcoords="offset points", va="center")
    axb.axvline(96.9, color=INK2, lw=1.6, ls="--")
    axb.annotate("天花板 96.9%\n（真机自身可重复性，难处）", (96.9, 0.35),
                 fontsize=8, color=INK2, xytext=(-6, 0), textcoords="offset points",
                 va="center", ha="right")
    for yy, v in zip(y, vals):
        xa = max(v, -118.0)
        axb.annotate(f"{v:+.1f}%", (xa, yy), fontsize=9.5, color=INK, fontweight="bold",
                     xytext=(8 if v >= 0 else 22, 0), textcoords="offset points",
                     va="center", ha="left")
    axb.set_yticks(y); axb.set_yticklabels(names, fontsize=10, color=INK)
    # 离群条(能量后50%可到 -285%)会把其余三条压成一线 ⇒ 裁剪并标注
    lo_lim = max(min(vals) * 1.35 - 5, -120.0)
    axb.set_xlim(lo_lim, 108)
    for yy, v in zip(y, vals):
        if v < lo_lim:
            axb.annotate("◀ 裁剪", (lo_lim, yy), fontsize=8, color="#c0392b",
                         xytext=(3, 0), textcoords="offset points", va="center")
    axb.set_xlabel("手调 XML 的开发集相对变化 (%)　1 − XML残差/gap", color=INK2)
    axb.set_title("③ 分区相对变化　——「全程」与难处均只作 legacy 开发诊断",
                  fontsize=10.5, color=INK, loc="left")
    axb.invert_yaxis()

    for x in fig.axes:
        x.grid(True, color=GRID, lw=0.7, alpha=0.9)
        x.set_axisbelow(True)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            x.spines[sp].set_color(GRID)
        x.tick_params(colors=INK2, labelsize=9)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.905, bottom=0.055)
    f = OUT / "legacy_fidelity.png"
    fig.savefig(f, dpi=140, facecolor="#fcfcfb")
    print(f"\n→ {f.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
