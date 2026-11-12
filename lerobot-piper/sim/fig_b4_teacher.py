"""B4 教师验收的关节对比图：Shift 与 Shift＋教师补正（各 3 轮，交替跑）→ outputs/b4/teacher/fig_teacher_vs_shift.png。

  python -u sim/fig_b4_teacher.py          # 所有验收通过的教师
  python -u sim/fig_b4_teacher.py p47 p34  # 只画这几条
  python -u sim/fig_b4_teacher.py --ja     # 日语版 → fig_teacher_vs_shift_ja.png

上：第一条教师里教师相对 Shift 改善最大的 1 s 的关节角（挑选的片段，第 1 轮）；
下：每条教师的关节偏差（相对原始策略指令）随时间，3 轮逐点均方根、0.25 s 平滑，同 ilc_route1.plot_compare。
时间从开头 1 s 保持段结束时算起。保持段（不计入评价，那里的「偏差」只是 Shift 第一帧与原始指令第一帧之差）
和之后 0.5 s 的起步段（计入评价，但两个条件相同：窗从 0 升起，教师基本不补）不画。
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_teacher as B  # noqa: E402
import cmd_tracking as T  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402
from step4_gvnext import data as D  # noqa: E402

CONDS = (("as", T.MUTED), ("at", T.BLUE))   # as = Shift，at = Shift＋教师补正
T0 = RR.HOLD * RR.DT                  # 保持段结束的名义时刻（s）；图上的时间从这里算起
SKIP = 0.5                            # 保持段之后不画的起步段（s）
TEXT = {
    "zh": {"as": "Shift", "at": "Shift＋教师补正", "cmd": "原始策略指令", "act": "实际：",
           "split": {"train": "训练", "val": "验证"},
           "panel": "{tid}（{split}）· Shift {s:.2f} → 教师 {t:.2f} mrad（{r:+.1f}%）",
           "zoom": "{tid} · j{j} 关节角：放大到教师相对 Shift 改善最大的 1 s（挑选的片段，验收第 1 轮；全程见下方）",
           "deg": "关节角（°）", "dev": "关节偏差（mrad）", "time": "时间（s，保持段之后）",
           "foot": "下方：验收 3 轮（与 Shift 交替）逐点均方根、0.25 s 平滑；偏差 = 编码器 200 Hz 关节角 − 原始策略指令（五个活动关节）；"
                   "标题数字 = 3 轮中位（按评价定义，含起步段）\n开头 1 s 保持段（不计入评价）和之后 0.5 s 起步段（计入评价、两条件相同）不画；"
                   "灰色 = 不计入评价"},
    "ja": {"as": "Shift", "at": "Shift＋教師補正", "cmd": "元の方策指令", "act": "実測：",
           "split": {"train": "学習", "val": "検証"},
           "panel": "{tid}（{split}）・Shift {s:.2f} → 教師 {t:.2f} mrad（{r:+.1f}%）",
           "zoom": "{tid}・j{j} 関節角：教師補正による Shift からの改善が最大の 1 s を拡大（抽出区間・受入れ試験 第 1 回；全体は下段）",
           "deg": "関節角（°）", "dev": "関節偏差（mrad）", "time": "時間（s，保持区間の終了から）",
           "foot": "下段：受入れ試験 3 回（Shift と交互）の点ごとの二乗平均平方根、0.25 s 平滑化；偏差 = エンコーダ 200 Hz の関節角 − 元の方策指令"
                   "（補正対象の 5 関節）；タイトルの数値 = 3 回の中央値（評価の定義どおり、起動区間を含む）\n"
                   "冒頭 1 s の保持区間（評価対象外）とその後 0.5 s の起動区間（評価対象・両条件で同一）は描画しない；灰色 = 評価対象外"},
}


def episodes(tid: str, sess: Path, cond: str) -> list[dict]:
    return [T.from_record(h, np.asarray(h["cmd"], float)) for h in
            (json.loads((sess / f"replay_{tid}_{cond}{r:02d}_r1.json").read_text()) for r in (1, 2, 3))]


def dev_joints(eps: list[dict], a6: np.ndarray, n: int) -> np.ndarray:
    """各遍逐关节的偏差平方按名义帧时间对齐（起步段之后）→ 逐点对各遍取均值、0.25 s 平滑（网格 × 5，rad²）。
    平滑除以窗内有效点数，两端不被补零拉低。"""
    grid = np.arange(T0 + SKIP, (n - 1) * RR.DT, 1 / D.FS_Q)
    sq = [np.stack([np.interp(grid, np.interp(ep["tq"], ep["t_send"], RR.DT * np.arange(n)), e) for e in
                    ((S.zoh(ep["t_send"], a6, ep["tq"]) - ep["q"])[:, T.ACTIVE] ** 2).T], 1) for ep in eps]
    k = np.ones(int(0.25 * D.FS_Q))
    norm = np.convolve(np.ones(len(grid)), k, "same")
    return np.stack([np.convolve(c, k, "same") / norm for c in np.mean(sq, 0).T], 1)


def excluded(q: dict, n: int) -> np.ndarray:
    """不计入评价的帧：开头保持、疑似接触、末尾留白（同 ilc_route1.plot_compare）。"""
    off = q["marked"].copy()
    off[:RR.HOLD] = True
    off[np.arange(n) * RR.DT > (n - 1) * RR.DT - C.MARGIN] = True
    return off


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    args = sys.argv[1:] if argv is None else list(argv)
    lang = "ja" if "--ja" in args else "zh"
    L = TEXT[lang]
    split = {p["id"]: p["split"] for p in P.load()["pool"]}
    acc = {f.stem[len("accept_"):]: json.loads(f.read_text()) for f in (B.TEACH / "sessions").glob("*/accept_*.json")}
    ids = [i for i in ([a for a in args if a != "--ja"] or P.load()["order"]) if acc.get(i, {}).get("passed")]
    if not ids:
        raise SystemExit("⛔ 没有验收通过的教师")
    rows = (len(ids) + 1) // 2
    fig = plt.figure(figsize=(12, 3.2 + 2.4 * rows))
    gs = fig.add_gridspec(1 + rows, 2, height_ratios=[1.3] + [1] * rows)
    axes = [fig.add_subplot(gs[1 + k // 2, k % 2]) for k in range(len(ids))]
    for ax in axes[1:]:
        ax.sharey(axes[0])
    top = 0.0
    for k, tid in enumerate(ids):
        q, sess = B.load_seq(tid), B.TEACH / "sessions" / acc[tid]["session"]
        n, a6 = len(q["a7"]), q["a7"][:, :6].astype(float)
        eps = {c: episodes(tid, sess, c) for c, _ in CONDS}
        A, off = axes[k], excluded(q, n)
        for k0, k1 in zip(np.flatnonzero(off & ~np.r_[False, off[:-1]]), np.flatnonzero(off & ~np.r_[off[1:], False])):
            if (k1 + 1) * RR.DT > T0 + SKIP:           # 保持段、起步段那一块不画
                A.axvspan(max(k0 * RR.DT, T0 + SKIP) - T0, (k1 + 1) * RR.DT - T0, color=T.SHADE, alpha=0.5, lw=0)
        sqj = {c: dev_joints(eps[c], a6, n) for c, _ in CONDS}
        tabs = T0 + SKIP + np.arange(len(sqj["as"])) / D.FS_Q
        free = ~off[np.clip((tabs / RR.DT).astype(int), 0, n - 1)]
        for c, color in CONDS:
            y = np.sqrt(sqj[c].mean(1)) * 1e3
            top = max(top, y[free].max())
            A.plot(tabs - T0, y, color=color, lw=1.4, label=L[c])
        A.set_xlim(SKIP, tabs[-1] - T0)
        med = {c: np.median([v["joint_raw"] for key, v in acc[tid]["accept"].items() if key.startswith(c)]) for c, _ in CONDS}
        A.set_title(L["panel"].format(tid=tid, split=L["split"].get(split[tid], split[tid]), s=med["as"], t=med["at"],
                                      r=(med["at"] / med["as"] - 1) * 100), loc="left", color=T.INK)
        A.grid(axis="y", color=T.SHADE, lw=0.6)
        if k % 2 == 0:
            A.set_ylabel(L["dev"])
        if k == 0:                                     # 上方放大图：第一条教师里教师相对 Shift 改善最大的关节与时刻（画出来的、计入评价的时段）
            gain = np.sqrt(sqj["as"]) - np.sqrt(sqj["at"])
            gain[~free] = -np.inf
            i, j = np.unravel_index(np.argmax(gain), gain.shape)
            t0, jt = tabs[i], T.ACTIVE[j]
            Z = fig.add_subplot(gs[0, :])
            tg = np.arange(max(t0 - 0.5, T0), min(t0 + 0.5, (n - 1) * RR.DT), 1 / D.FS_Q)
            Z.step(tg - T0, np.degrees(a6[np.clip((tg / RR.DT).astype(int), 0, n - 1), jt]), where="post",
                   color=T.INK, lw=1.0, ls="--", label=L["cmd"])
            for c, color in CONDS:
                e = eps[c][0]
                f = np.interp(e["tq"], e["t_send"], RR.DT * np.arange(n))
                Z.plot(tg - T0, np.degrees(np.interp(tg, f, e["q"][:, jt])), color=color, lw=1.6, label=L["act"] + L[c])
            Z.set_title(L["zoom"].format(tid=tid, j=jt + 1), loc="left", color=T.INK)
            Z.set_ylabel(L["deg"])
            Z.set_xlabel(L["time"])
            Z.grid(axis="y", color=T.SHADE, lw=0.6)
            Z.legend(frameon=False, labelcolor=T.INK, loc="best")
            for side in ("top", "right"):
                Z.spines[side].set_visible(False)
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
    for ax in axes[-2:]:
        ax.set_xlabel(L["time"])
    axes[0].set_ylim(0, 1.3 * top)                    # 按计入评价的时段定纵轴；灰色段的尖峰可以出界
    axes[0].legend(frameon=False, labelcolor=T.INK, loc="upper right")
    fig.text(0.01, 0.005, L["foot"], color=T.MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = B.TEACH / ("fig_teacher_vs_shift.png" if lang == "zh" else "fig_teacher_vs_shift_ja.png")
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print(f"{len(ids)} 条教师：{' '.join(ids)} → {out}")
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
