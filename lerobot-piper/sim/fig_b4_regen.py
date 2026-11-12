"""B4（重新生成网络 ②，冻结的 outputs/b4/regen_t24）汇报用的两张图，中文、日语各一版 → outputs/b4/report/（日语版带 _ja）。

  python -u sim/fig_b4_regen.py

  fig_regen_val  真机验证集（会话 20261112n，shift84 / B4 各 2 轮，主指标 = 相对老师定义的理想的关节偏差）：
                 上 = 改善最接近中位数的那条轨迹第 1 轮、改善最多的关节的「实测 − 理想」随时间（不计入主指标的区间打灰底）；
                 下 = 每条轨迹两种条件的关节偏差（两轮平均）与变化，判定结果取自 b4val_check.json。
  fig_regen_net  网络结构：输入、网络内部的相对化与标准化、嵌入与位置编码、Transformer、输出与发令。
配色同 sim/fig_b4_report.py：蓝 = 网络（B4），灰 = 基线（shift84）；结构图里红框 = 训练的部分。
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_teacher as B  # noqa: E402
import cmd_tracking as T  # noqa: E402
import ideal_motion as IM  # noqa: E402
import shift_replay as S  # noqa: E402

OUT = HERE.parent / "outputs/b4/report"
SESS = B.TEACH / "sessions/20261112n"
GRAY, SURF = "#898781", "#fcfcfb"
RED, RED_BG, BLUE_BG = "#d6405f", "#fbe9ee", "#def0f7"
JN = ("j1", "j2", "j3", "j5", "j6")
TEXT = {
    "zh": {"title": "B4 在真机上比 shift84 低 16.4%（10 条验证轨迹 × 2 轮，相对老师定义的理想）",
           "a": "例：{tid} 第 1 轮 {j} 的「实测 − 理想」（改善最接近中位数的轨迹，{c}）",
           "ylab": "实测 − 理想（mrad）", "xlab": "时间（s）", "shade": "不计入主指标的区间",
           "s84": "shift84（基线）", "b4": "B4（网络）",
           "b": "每条轨迹：相对理想的关节偏差（两轮平均，5 个关节的 RMS）　效应 {e}［95% 区间 {lo}, {hi}］",
           "bx": "相对理想的关节偏差（mrad）", "chg": "变化",
           "foot": "会话 20261112n。上图灰底 = 起步、疑似接触、末尾。\n变化 = 每轮配对的相对变化（两轮的中位）；效应 = 各轨迹的平均，区间用两级自助法（先抽轨迹、再抽轮次）。",
           "n_title": "B4 网络的结构",
           "past": ("过去 8 帧的 ACT 指令", "8 帧 × 6 关节，rad"),
           "fut": ("之后 10 帧的 ACT 指令", "10 帧 × 6 关节，rad（TE 融合后当时已知的值）"),
           "prep": ("网络内部的相对化与标准化", "各帧减去当前帧的 ACT 指令 r，之后段再附上 r；按训练集的均值、标准差标准化"),
           "learn": "训练的部分", "tok": ("过去 8 个单元", "之后 10 个单元"),
           "emb": "线性嵌入 → 128 维 + 可学的位置编码（18 个单元）",
           "tf": ("Transformer 编码器", "4 层 · 4 头 · 128 维 · 前馈 512"),
           "head": "之后 10 个单元的输出 → 线性头 128 → 6 → 反标准化",
           "out": ("输出：之后 10 帧 × 6 关节的指令", "绝对角（rad）；没有把 ACT 或 Shift 指令加回输出的通路"),
           "send": ("发令（每帧 50 ms）", "只发第一帧，下一帧重新生成；窗为 0 的关节和夹爪发 shift84"),
           "train": "训练：标签 = teacher24（ILC 教师整体提前 24 ms）；损失 = 输出与标签之差的加权平方和（mrad，第一帧 × 9）",
           "n_foot": "网络还留有已发指令、编码器、IMU 的输入位置，B4 里全部置 0（只看 ACT 指令）；误差输入放到 B3。"},
    "ja": {"title": "B4 は実機で shift84 より 16.4% 小さい（検証 10 軌道 × 2 回、先生の定義した理想に対して）",
           "a": "例：{tid} 第 1 回 {j} の「実測 − 理想」（改善が中央値に最も近い軌道、{c}）",
           "ylab": "実測 − 理想（mrad）", "xlab": "時間（s）", "shade": "主指標に入れない区間",
           "s84": "shift84（基線）", "b4": "B4（網）",
           "b": "軌道ごと：理想に対する関節偏差（2 回の平均、5 関節の RMS）　効果 {e}［95% 区間 {lo}, {hi}］",
           "bx": "理想に対する関節偏差（mrad）", "chg": "変化",
           "foot": "セッション 20261112n。上図の灰色の帯 = 開始・接触の疑い・末尾。\n変化 = 各回の対応のある相対変化（2 回の中央値）。効果 = 軌道の平均、区間は 2 段ブートストラップ（軌道、回の順に抽出）。",
           "n_title": "B4 の網の構造",
           "past": ("過去 8 フレームの ACT 指令", "8 フレーム × 6 関節、rad"),
           "fut": ("先の 10 フレームの ACT 指令", "10 フレーム × 6 関節、rad（TE で融合した、その時点で分かっている値）"),
           "prep": ("網の中での相対化と標準化", "各フレームから現在の ACT 指令 r を引き、先の分には r も付ける。訓練データの平均・標準偏差で標準化"),
           "learn": "学習する部分", "tok": ("過去 8 単位", "先の 10 単位"),
           "emb": "線形埋め込み → 128 次元 + 学習する位置埋め込み（18 単位）",
           "tf": ("Transformer エンコーダ", "4 層 · 4 ヘッド · 128 次元 · FFN 512"),
           "head": "先の 10 単位の出力 → 線形ヘッド 128 → 6 → 標準化を戻す",
           "out": ("出力：先の 10 フレーム × 6 関節の指令", "絶対角（rad）。ACT や Shift の指令を出力に足し戻す経路はない"),
           "send": ("送信（各フレーム 50 ms）", "先頭フレームだけ送り、次のフレームで作り直す。窓が 0 の関節とグリッパーは shift84"),
           "train": "学習：ラベル = teacher24（ILC 教師を 24 ms 前倒し）。損失 = 出力とラベルの差の重み付き二乗和（mrad、先頭フレーム × 9）",
           "n_foot": "網には送信済み指令・エンコーダ・IMU の入力の枠もあるが、B4 ではすべて 0（ACT 指令だけを見る）。誤差の入力は B3 で使う。"},
}


def pct(v: float) -> str:
    return f"{v:+.1f}%".replace("-", "\u2212")


def deviation(tid: str) -> dict:
    """会话里这条轨迹第 1 轮两种条件的「实测 − 理想」(mrad)：{条件: (名义时刻 s, (n, 5)，不计入主指标的采样为 NaN)}。"""
    q = B.load_seq(tid)
    cmd = IM.IdealCmd(IM.ideal(q["a7"]))
    out = {}
    for cond in ("shift84", "regen"):
        h = json.loads((SESS / f"replay_{tid}_n{cond}1_r1.json").read_text())
        ep = C.T.from_record(h, np.asarray(h["cmd"], float))
        fq, _ = S.masks(ep, np.asarray(q["marked"], bool), C.MARGIN)
        tn = IM.nominal(ep["tq"], ep["t_send"])
        e = (np.asarray(ep["q"]) - cmd(tn)[:, :6])[:, IM.ACT] * 1e3
        out[cond] = (tn, np.where(fq[:, None], e, np.nan))
    return out


def fig_val(plt, L: dict, suf: str) -> None:
    d = json.loads((SESS / "b4val_check.json").read_text())
    cmp_ = d["compare"]["regen"]["hier2"]["joint_ideal"]
    chg = {t: v["median"] for t, v in cmp_["per_group"].items()}
    tid = min(chg, key=lambda t: abs(chg[t] - np.median(list(chg.values()))))
    dev = deviation(tid)
    rms = {c: np.sqrt(np.nanmean(dev[c][1] ** 2, 0)) for c in dev}
    j = int(np.argmax(rms["shift84"] - rms["regen"]))
    print(f"上图：{tid}（{chg[tid]:+.1f}%），{JN[j]}；各关节 RMS shift84 {np.round(rms['shift84'], 2)} → B4 {np.round(rms['regen'], 2)} mrad")

    fig = plt.figure(figsize=(9.6, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.55], hspace=0.5, left=0.11, right=0.9, top=0.9, bottom=0.115)
    A = fig.add_subplot(gs[0])
    t, e = dev["shift84"]
    off = np.isnan(e[:, j])
    edges = np.flatnonzero(np.diff(np.r_[0, off.astype(int), 0]))
    for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
        A.axvspan(t[a], t[min(b, len(t) - 1)], color=T.SHADE, alpha=0.6, lw=0, label=L["shade"] if k == 0 else None)
    A.axhline(0, color=T.MUTED, lw=0.8)
    A.plot(t, e[:, j], color=GRAY, lw=1.0, label=L["s84"])
    A.plot(dev["regen"][0], dev["regen"][1][:, j], color=T.BLUE, lw=1.0, label=L["b4"])
    A.set_title(L["a"].format(tid=tid, j=JN[j], c=pct(chg[tid])), loc="left", color=T.INK)
    A.set_ylabel(L["ylab"], color=T.INK)
    A.set_xlabel(L["xlab"], color=T.INK)
    A.set_xlim(t[0], t[-1])
    A.grid(axis="y", color=T.SHADE, lw=0.6)
    A.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=T.INK)        # 右下角没有曲线

    A = fig.add_subplot(gs[1])
    ids = sorted(chg, key=lambda t: chg[t], reverse=True)          # 改善最多的在最上面
    mean = lambda t, c: np.mean([d["res"][f"{t} {c} {r}"]["joint_ideal"] for r in (1, 2)])   # noqa: E731
    for y, t_ in enumerate(ids):
        x0, x1 = mean(t_, "shift84"), mean(t_, "regen")
        A.plot([x0, x1], [y, y], color=T.SHADE, lw=2.5, zorder=1, solid_capstyle="round")
        A.scatter([x0], [y], s=64, color=GRAY, edgecolor=SURF, linewidth=2, zorder=3, label=L["s84"] if y == 0 else None)
        A.scatter([x1], [y], s=64, color=T.BLUE, edgecolor=SURF, linewidth=2, zorder=3, label=L["b4"] if y == 0 else None)
        A.text(1.02, y, pct(chg[t_]), transform=A.get_yaxis_transform(), va="center", ha="left", fontsize=9, color=T.INK)
    A.text(1.02, len(ids) - 0.3, L["chg"], transform=A.get_yaxis_transform(), va="bottom", ha="left", fontsize=9, color=T.MUTED)
    A.set_yticks(range(len(ids)), ids)
    A.set_ylim(-0.6, len(ids) - 0.4)
    A.set_xlabel(L["bx"], color=T.INK)
    A.set_title(L["b"].format(e=pct(cmp_["effect"]), lo=pct(cmp_["ci95"][0])[:-1], hi=pct(cmp_["ci95"][1])[:-1]), loc="left", color=T.INK)
    A.grid(axis="x", color=T.SHADE, lw=0.6)
    A.legend(frameon=False, loc="lower right", labelcolor=T.INK)
    fig.suptitle(L["title"], x=0.01, ha="left", color=T.INK, fontsize=11.5)
    fig.text(0.01, 0.012, L["foot"], color=T.MUTED, fontsize=7.8, linespacing=1.5)
    fig.savefig(OUT / f"fig_regen_val{suf}.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def fig_net(plt, L: dict, suf: str) -> None:
    from matplotlib.patches import FancyBboxPatch, Rectangle
    fig = plt.figure(figsize=(9.6, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 82)
    ax.axis("off")

    def box(x0, x1, y0, y1, lines, edge=T.BLUE, face=BLUE_BG, bold=True):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=face, edgecolor=edge, lw=1.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (1.2 if len(lines) > 1 else 0), lines[0], ha="center", va="center",
                fontsize=10.5, color=T.INK, weight="bold" if bold else "normal")
        if len(lines) > 1:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 - 1.7, lines[1], ha="center", va="center", fontsize=8.6, color=T.MUTED)

    def arrow(x, y0, y1, color=T.BLUE):
        ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=0, shrinkB=0))

    ax.text(2, 79.5, L["n_title"], fontsize=12, color=T.INK, weight="bold", va="center")
    box(4, 40, 68, 75, L["past"])
    box(46, 96, 68, 75, L["fut"])
    arrow(22, 68, 63.5)
    arrow(71, 68, 63.5)
    box(10, 90, 57, 63.5, L["prep"])
    arrow(50, 57, 52.5)

    ax.add_patch(FancyBboxPatch((4, 22.5), 92, 30, boxstyle="round,pad=0,rounding_size=1.5", facecolor="none",
                                edgecolor=RED, lw=1.6, ls=(0, (5, 3))))
    ax.text(5.5, 50.6, L["learn"], fontsize=9, color=RED, va="center")
    w, gap, x0 = 3.6, 0.7, 50 - (18 * 3.6 + 17 * 0.7 + 2.0) / 2    # 18 个单元排成一行，过去 8 个和之后 10 个之间多留 2
    xs = x0 + np.arange(18) * (w + gap) + np.where(np.arange(18) < 8, 0.0, 2.0)
    for x in xs:
        ax.add_patch(Rectangle((x, 45.5), w, 2.6, facecolor=BLUE_BG, edgecolor=T.BLUE, lw=1))
    ax.text(xs[:8].mean() + w / 2, 49.4, L["tok"][0], ha="center", fontsize=8.6, color=T.MUTED)
    ax.text(xs[8:].mean() + w / 2, 49.4, L["tok"][1], ha="center", fontsize=8.6, color=T.MUTED)
    ax.text(50, 43.4, L["emb"], ha="center", va="center", fontsize=9, color=T.INK)
    box(12, 88, 34, 41.5, L["tf"], edge=RED, face=RED_BG)
    for x in xs[8:]:
        ax.add_patch(Rectangle((x, 29.6), w, 2.2, facecolor=RED_BG, edgecolor=RED, lw=1))
        arrow(x + w / 2, 34, 31.8, color=RED)
    ax.text(50, 26.2, L["head"], ha="center", va="center", fontsize=9, color=T.INK)
    arrow(50, 22.5, 19.5)
    box(10, 90, 13, 19.5, L["out"])
    arrow(50, 13, 10.5)
    box(10, 90, 4.2, 10.5, L["send"])
    ax.text(2, 1.8, L["train"] + "\n" + L["n_foot"], fontsize=7.8, color=T.MUTED, va="center", linespacing=1.6)
    fig.savefig(OUT / f"fig_regen_net{suf}.png", dpi=160, facecolor=SURF)
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9.5, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, suf in (("zh", ""), ("ja", "_ja")):
        fig_val(plt, TEXT[lang], suf)
        fig_net(plt, TEXT[lang], suf)
    print(f"→ {OUT}/fig_regen_val(_ja).png、fig_regen_net(_ja).png")
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
