"""B4 整段版补正网络 ①（v0.3，冻结版，见 reports/PLAN_b4_seq.md 与 outputs/b4/net_seq/frozen.json）的示意图，中文、日语各一版
→ outputs/b4/fig_b4_net.png、fig_b4_net_ja.png。一帧版 ① 的旧图另存为 fig_b4_net_1frame(_ja).png。

  python -u sim/fig_b4_net.py

上：离线训练（标签 = ILC 教师实发的整段指令 − ACT 参考）；
下：闭环部署（每帧 50 ms，只发第一帧；网络指令直接发给机械臂；接触、限位附近的安全规则同一帧版，只在图注里说明）；
右：网络本身（输入排布、网络内部的相对化、MLP 各层）。配色与框的画法同 reports/fig_ours_overview（蓝 = 数据与固定模块，红 = 要学习的网络）。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs/b4"
BLUE, BLUE_BG, RED, RED_BG = "#1d8fbf", "#def0f7", "#d6405f", "#fbe9ee"
INK, NOTE, GRID = "#1f2d4d", "#5b6b80", "#b8c4d0"
P, F = 8, 16                                   # 冻结的整段版 ①：过去 P 帧、未来 F 帧（h = 128，N = 10）
TEXT = {
    "zh": {"side": "B4：整段版补正网络", "train": "① 离线训练", "rec": ("采教师时的录制记录", "每帧 ACT 整段预测"),
           "net_train": ("整段网络（训练）", "结构见右"), "teacher": ("ILC 教师（真机，含 IMU）", "实发整段指令（含提前量）"),
           "loss": ("损失", "与教师的均方误差（MSE）"), "backprop": "反向传播，更新网络参数",
           "freeze": "训练完冻结，\n搬到闭环里", "deploy": "② 闭环部署（每帧 50 ms）",
           "act": ("ACT 策略 + TE", "每帧预测 100 步，加权平均"),
           "net_frozen": ("整段网络（冻结）", "输出 10 帧，只发第一帧"), "arm": ("Piper 机械臂", "控制器跟踪 u(k)"),
           "u0note": "网络直接给出的指令",
           "cmp": "与一帧版 ① 的区别：一帧版 u = Shift + Δ（网络只给 5 个补正量，提前量来自 Shift）｜"
                  "整段版 u = 网络输出的指令（不加 Shift，提前量由网络自己学）",
           "inset": "整段网络的结构（MLP）", "dim7": "7 维", "past": "过去 8 帧\n已定的指令", "future": "未来 16 帧\nTE 部分融合值",
           "input": "网络内部：\n各帧 − a_π(k)\n再附 a_π(k)", "bars": ["输入\n175", "隐层\n128", "隐层\n128", "输出 50\n= 10 帧 × 5"],
           "fc": ["全连接\n+ ReLU", "全连接\n+ ReLU", "全连接"], "note": "冻结版 v0.3：P8、F16、h128、N10",
           "symbols": r"x(k)：网络输入　$\hat{y}(k)$：网络输出的 10 帧指令（相对 ACT 参考）　"
                      "y*(k)：教师实发的整段指令 − ACT 参考（标签）　u(k)：实发指令",
           "foot": "已冻结（11-06）。每帧只发第一帧，后 9 帧只用于训练；接触、限位附近按一帧版的安全规则处理；"
                   "IMU 只在 ILC 教师里用到，部署时网络不用 IMU 和编码器。"},
    "ja": {"side": "B4：系列出力版の補正ネットワーク", "train": "① オフライン学習", "rec": ("教師収録時の記録", "各フレームの ACT チャンク予測"),
           "net_train": ("系列ネットワーク（学習）", "構造は右図"), "teacher": ("ILC 教師（実機・IMU 込み）", "実送信の指令列（先行量込み）"),
           "loss": ("損失", "教師との平均二乗誤差（MSE）"), "backprop": "誤差逆伝播でパラメータを更新",
           "freeze": "学習後に凍結し、\n閉ループへ", "deploy": "② 閉ループ実行（1 フレーム 50 ms）",
           "act": ("ACT 方策 + TE", "100 ステップ予測を加重平均"),
           "net_frozen": ("系列ネットワーク（凍結）", "10 フレーム出力、先頭のみ送信"), "arm": ("Piper アーム", "コントローラが u(k) に追従"),
           "u0note": "ネットワークが直接出す指令",
           "cmp": "1 フレーム版との違い：1 フレーム版 u = Shift + Δ（ネットワークは補正量 5 個のみ、先行量は Shift）｜"
                  "系列出力版 u = ネットワーク出力の指令（Shift は加えず、先行量もネットワークが学習）",
           "inset": "系列ネットワークの構造（MLP）", "dim7": "7 次元", "past": "過去 8 フレーム\n確定した指令",
           "future": "未来 16 フレーム\nTE の部分融合値", "input": "ネットワーク内で\n各フレーム − a_π(k)\n＋ a_π(k)",
           "bars": ["入力\n175", "隠れ層\n128", "隠れ層\n128", "出力 50\n= 10×5"], "fc": ["全結合\n+ ReLU", "全結合\n+ ReLU", "全結合"],
           "note": "凍結版 v0.3：P8・F16・h128・N10",
           "symbols": r"x(k)：入力　$\hat{y}(k)$：出力の 10 フレーム指令（ACT 参照との差）　"
                      "y*(k)：教師の実送信指令列 − ACT 参照（ラベル）　u(k)：実送信指令",
           "foot": "学習済み・凍結（11-06）。毎フレーム先頭のみ送信し、残り 9 フレームは学習にのみ使用。"
                   "接触・関節限界付近は 1 フレーム版と同じ安全規則。IMU は ILC 教師でのみ使い、実行時は IMU もエンコーダも使わない。"},
}


def box(ax, x, y, w, h, title, sub, red=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                fc=RED_BG if red else BLUE_BG, ec=RED if red else BLUE, lw=2.2))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=10.5, color=INK)


def arrow(ax, pts, color=BLUE, ls="-", label=None, at=0, dx=0, dy=10):
    """折线箭头：pts 为途经点，最后一段带箭头；label 标在第 at 段的中点（偏移 dx、dy）。"""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=2.2, ls=ls, solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=color, lw=2.2, shrinkA=0, shrinkB=0))
    if label:
        (x0, y0), (x1, y1) = pts[at], pts[at + 1]
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va="bottom", fontsize=11, color=INK, style="italic")


def node(ax, x, y, sign):
    ax.add_patch(Circle((x, y), 18, fc="white", ec=INK, lw=2))
    ax.text(x, y + 1, sign, ha="center", va="center", fontsize=17, weight="bold", color=INK)


def main() -> int:
    plt.rcParams.update({"font.family": "Noto Sans CJK JP"})
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, suf in (("zh", ""), ("ja", "_ja")):
        L = TEXT[lang]
        fig = plt.figure(figsize=(16, 7.2))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1600)
        ax.set_ylim(0, 720)
        ax.axis("off")
        ax.text(48, 400, L["side"], rotation=90, ha="center", va="center", fontsize=20, weight="bold", color=INK)

        # ① 离线训练
        ax.text(150, 683, L["train"], fontsize=12.5, weight="bold", color=INK, va="center")
        box(ax, 150, 565, 250, 70, *L["rec"])
        arrow(ax, [(400, 600), (470, 600)], label="x(k)")
        box(ax, 470, 565, 220, 70, *L["net_train"], red=True)
        arrow(ax, [(690, 600), (772, 600)], label=r"$\hat{y}(k)$")
        node(ax, 790, 600, "−")
        box(ax, 665, 455, 250, 60, *L["teacher"])
        arrow(ax, [(790, 515), (790, 582)], label="y*(k)", dx=24, dy=-8)
        arrow(ax, [(808, 600), (870, 600)])
        box(ax, 870, 565, 230, 70, *L["loss"])
        arrow(ax, [(985, 635), (985, 680), (580, 680), (580, 637)], color=RED, ls=(0, (6, 4)))
        ax.text(782, 686, L["backprop"], ha="center", va="bottom", fontsize=10.5, color=RED, style="italic")

        # 训练完冻结 → 部署
        ax.add_patch(FancyArrow(580, 558, 0, -262, width=34, head_width=68, head_length=26, length_includes_head=True,
                                fc="#d6ecf6", ec=BLUE, lw=2))
        ax.text(545, 440, L["freeze"], ha="right", va="center", fontsize=10.5, color=INK)

        # ② 闭环部署：网络指令直接发给机械臂（只发第一帧）
        ax.text(150, 360, L["deploy"], fontsize=12.5, weight="bold", color=INK, va="center")
        box(ax, 150, 215, 230, 70, *L["act"])
        arrow(ax, [(380, 250), (470, 250)], label="x(k)")
        box(ax, 470, 215, 220, 70, *L["net_frozen"], red=True)
        arrow(ax, [(690, 250), (900, 250)], label="u(k)")
        ax.text(795, 238, L["u0note"], ha="center", va="top", fontsize=9.5, color=INK)
        box(ax, 900, 215, 210, 70, *L["arm"])

        # 右：网络结构
        ax.add_patch(FancyBboxPatch((1150, 310), 435, 395, boxstyle="round,pad=0,rounding_size=12",
                                    fc="white", ec=BLUE, lw=1.8, ls=(0, (7, 4))))
        ax.text(1168, 686, L["inset"], fontsize=12.5, weight="bold", color=INK, va="center")
        cw, ch, gx, gy = 9.5, 10, 1200, 572              # 输入排布：7 行（j1–j6、夹爪）× (P + F) 列
        for r in range(7):
            for c in range(P + F):
                ax.add_patch(plt.Rectangle((gx + c * cw, gy + r * ch), cw, ch, fc="#9dc3dc" if c < P else "#e3eff7",
                                           ec="white", lw=0.8))
        ax.plot([gx + P * cw] * 2, [gy - 4, gy + 7 * ch + 4], color=INK, lw=1.4)
        ax.text(gx + P * cw, gy + 7 * ch + 6, "k", ha="center", va="bottom", fontsize=10, color=INK, style="italic")
        ax.text(gx - 6, gy + 3.5 * ch, L["dim7"], ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(gx + P * cw / 2, gy - 6, L["past"], ha="center", va="top", fontsize=9.5, color=INK)
        ax.text(gx + (P + F / 2) * cw, gy - 6, L["future"], ha="center", va="top", fontsize=9.5, color=INK)
        ax.text(gx + (P + F) * cw + 12, gy + 3.5 * ch, L["input"], ha="left", va="center", fontsize=10, color=INK)
        bars = [(1215, 84), (1320, 64), (1425, 64), (1530, 50)]
        yc = 445
        for (x0, h0), (x1, h1) in zip(bars, bars[1:]):
            for a in range(5):
                for b in range(5):
                    ax.plot([x0 + 7, x1 - 7], [yc - h0 / 2 + h0 * a / 4, yc - h1 / 2 + h1 * b / 4], color=GRID, lw=0.6, zorder=1)
        for (x, h), lab in zip(bars, L["bars"]):
            ax.add_patch(FancyBboxPatch((x - 7, yc - h / 2), 14, h, boxstyle="round,pad=0,rounding_size=4",
                                        fc=RED_BG, ec=RED, lw=1.8, zorder=2))
            ax.text(x, yc - 50, lab, ha="center", va="top", fontsize=9.5, color=INK)
        for (x0, _), (x1, _), lab in zip(bars, bars[1:], L["fc"]):
            ax.text((x0 + x1) / 2, yc + 46, lab, ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.text(1530, yc + 30, r"$\hat{y}(k)$", ha="center", va="bottom", fontsize=11, color=INK)
        ax.text(1168, 330, L["note"], fontsize=9.5, color=NOTE, va="center")

        # 与一帧版的区别、符号与口径
        ax.text(150, 76, L["cmp"], fontsize=10.5, color=INK, va="center")
        ax.text(150, 48, L["symbols"], fontsize=10.5, color=INK, va="center")
        ax.text(150, 22, L["foot"], fontsize=10.5, color=NOTE, va="center")
        fig.savefig(OUT / f"fig_b4_net{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)
    print(f"→ {OUT}/fig_b4_net.png、fig_b4_net_ja.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
