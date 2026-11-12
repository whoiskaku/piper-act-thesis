"""B4 汇报用的图，中文、日语各一版 → outputs/b4/report/（日语版文件名带 _ja）。

  python -u sim/fig_b4_report.py

  fig_teachers    全部教师验收：关节偏差、末端角速度偏差相对 Shift 的变化（通过 / 没通过）
  fig_net1_val    ①（一帧版）验证集真机回放：每条轨迹 ① 与该轨迹教师的两项变化
  fig_ee_why      末端角速度偏差（只报告）的拆解：① 与教师按成分、频段、腕部关节比较（验证集 10 条中位）
  fig_floor_p47   p47 的关节偏差曲线：Shift 与 ILC 教师
  fig_curve       ① 的学习曲线（训练轨迹 10 / 20 / 30 条时验证集 R²）
"""
from __future__ import annotations

import gc
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import b4_common as C  # noqa: E402
import b4_eval as EV  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_teacher as B  # noqa: E402
import cmd_tracking as T  # noqa: E402
import fig_b4_teacher as FT  # noqa: E402
import ilc_route1 as I  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402
from step4_gvnext import data as D  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

OUT = HERE.parent / "outputs/b4/report"
VAL = B.TEACH / "sessions/20261105v"                     # ① 验证集回放的会话
LOOSE = {"20261104b", "20261104d"}                       # 采集时 IMU 固定松动的会话（b 收尾核对转 6.87°，d 逐遍朝向一路上升）
TEXT = {
    "zh": {"groups": ["训练（含备用）", "验证", "没通过"], "joint": "关节偏差（相对原始策略指令）", "ee": "末端角速度偏差（IMU 测，只报告）",
           "chg": "相对 Shift 的变化（%）", "med": "中位", "teacher": "该轨迹的教师（验收）", "net": "①（验证集回放）",
           "a_title": "ILC 教师：验收时 Shift＋教师 相对 Shift（每条 3 轮平均）",
           "a_foot": "没通过的 5 条都因路径比 Shift 长 5% 以上（护栏）；会话 b、d 的教师采集时 IMU 固定松动，末端数字仅供参考",
           "b_title": "补正网络 ①（一帧版）在没见过的验证集轨迹上（真机，每条 2 轮）",
           "b_foot": "教师的数字来自各自轨迹的验收（3 轮）；* 教师验收没通过（路径护栏）；† 教师采集时 IMU 固定松动，末端仅供参考",
           "items": ["末端角速度偏差（总）", "编码器能解释的部分", "只有 IMU 看得到的部分", "0.5–5 Hz", "5–15 Hz",
                     "j5 关节角速度偏差", "j6 关节角速度偏差"],
           "c_title": "末端角速度偏差的拆解：①（验证集回放）与教师（验收）", "c_x": "相对 Shift 的变化（%，验证集 10 条轨迹中位）",
           "c_foot": "编码器能解释的部分 = 编码器推算的末端角速度 − 指令对应的末端角速度；只有 IMU 看得到的部分 = IMU 实测 − 编码器推算",
           "phases": ["接近", "张开夹爪下降", "夹持搬运", "松开", "撤回"], "shift5": "Shift（5 遍）", "teacher3": "Shift＋教师补正（3 遍）",
           "d_title": "p47：关节偏差（相对原始策略指令）——Shift 与 ILC 教师（数字 = 大峰编号）",
           "d_y": "关节偏差（mrad）", "time": "时间（s，保持段之后）",
           "d_foot": "逐点均方根、0.25 s 平滑；灰色 = 不计入评价",
           "e_title": "补正网络 ① 的学习曲线", "e_x": "训练轨迹条数", "e_y": "验证集 R²（五关节平均）"},
    "ja": {"groups": ["学習（予備を含む）", "検証", "不合格"], "joint": "関節偏差（元の方策指令との差）", "ee": "手先角速度偏差（IMU 計測・参考値）",
           "chg": "Shift に対する変化（%）", "med": "中央値", "teacher": "教師（受入れ試験）", "net": "①（検証リプレイ）",
           "a_title": "ILC 教師：受入れ試験での Shift＋教師 の Shift に対する変化（各 3 回の平均）",
           "a_foot": "不合格の 5 本はすべて経路長が Shift より 5% 以上長いため（ガード）。セッション b、d の教師は IMU の固定が緩んでいたため手先は参考値",
           "b_title": "補正ネットワーク ①（1 フレーム版）の未学習の検証軌道での結果（実機、各 2 回）",
           "b_foot": "教師の数値は各軌道の受入れ試験（3 回）。* 教師は不合格（経路長ガード）。† 教師の収録中 IMU の固定が緩んでいたため手先は参考値",
           "items": ["手先角速度偏差（全体）", "エンコーダで説明できる部分", "IMU でしか見えない部分", "0.5–5 Hz", "5–15 Hz",
                     "j5 関節角速度偏差", "j6 関節角速度偏差"],
           "c_title": "手先角速度偏差の内訳：①（検証リプレイ）と教師（受入れ試験）", "c_x": "Shift に対する変化（%、検証 10 軌道の中央値）",
           "c_foot": "エンコーダで説明できる部分 = エンコーダからの推定 − 指令に対応する手先角速度。IMU でしか見えない部分 = IMU 実測 − エンコーダからの推定",
           "phases": ["接近", "グリッパを開いて下降", "把持・搬送", "解放", "退避"], "shift5": "Shift（5 回）", "teacher3": "Shift＋教師補正（3 回）",
           "d_title": "p47：関節偏差（元の方策指令との差）——Shift と ILC 教師（数字 = 大きな山の番号）",
           "d_y": "関節偏差（mrad）", "time": "時間（s、保持区間の後）",
           "d_foot": "点ごとの二乗平均平方根、0.25 s 平滑化。灰色 = 評価対象外",
           "e_title": "補正ネットワーク ① の学習曲線", "e_x": "学習軌道の本数", "e_y": "検証 R²（5 関節の平均）"},
}


def ep_of(sess: Path, rid: str) -> dict:
    h = json.loads((sess / f"replay_{rid}_r1.json").read_text())
    return T.from_record(h, np.asarray(h["cmd"], float))


def mid_ref(ep: dict, a6: np.ndarray) -> np.ndarray:
    """穿过每级台阶中点的平滑参考，取在编码器时刻上。"""
    ts = ep["t_send"]
    mid = np.r_[(ts[:-1] + ts[1:]) / 2, ts[-1] + 0.025]
    return np.stack([np.interp(ep["tq"], mid, a6[:, j]) for j in range(6)], 1)


def ee_parts(sess: Path, rid: str, q: dict, ctx) -> np.ndarray:
    """一遍的：末端角速度偏差（总、编码器可解释、IMU 独有、0.5–5 Hz、5–15 Hz，rad/s）与 j5、j6 角速度偏差（rad/s）。"""
    ep, a6 = ep_of(sess, rid), q["a7"][:, :6].astype(float)
    fq, fi = S.masks(ep, q["marked"], C.MARGIN)
    tgt = S.raw_gyro(ep, a6, ctx.calib, ctx.model, ctx.sid)
    meas, enc = ep["imu"][:, 3:], I.enc_gyro(ep, ctx.calib, ctx.model, ctx.sid)
    lp = lambda x, hi=T.LP_VEL: D.lp(x, hi, D.FS_I)                       # noqa: E731
    r = lambda x: np.sqrt(np.mean(np.sum(x[fi] ** 2, 1)))                  # noqa: E731
    e = meas - tgt
    dv = (T.lp_deriv(ep["q"], T.LP_VEL, 1) - T.lp_deriv(mid_ref(ep, a6), T.LP_VEL, 1))[fq]
    return np.array([r(lp(e)), r(lp(enc - tgt)), r(lp(meas - enc)), r(lp(e, 5.0) - lp(e, 0.5)), r(lp(e) - lp(e, 5.0)),
                     np.sqrt(np.mean(dv[:, 4] ** 2)), np.sqrt(np.mean(dv[:, 5] ** 2))])


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9.5, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    OUT.mkdir(parents=True, exist_ok=True)
    split = {p["id"]: p["split"] for p in P.load()["pool"]}
    acc = {Path(f).stem[len("accept_"):]: (json.loads(Path(f).read_text())["passed"], Path(f).parent)
           for f in glob.glob(str(B.TEACH / "sessions/*/accept_*.json"))}

    # 数据：教师验收（res 缓存里的指标）
    tch = {}
    for tid, (_, s) in acc.items():
        m = lambda k: json.loads(str(np.load(s / f"res_{tid}_{k}.npz")["met"]))   # noqa: E731
        tch[tid] = [np.mean([m(f"at0{r}")[k] / m(f"as0{r}")[k] - 1 for r in (1, 2, 3)]) * 100 for k in ("joint_raw", "ee_raw")]
    grp = {0: [t for t in acc if acc[t][0] and split[t] in ("train", "spare")], 1: [t for t in acc if acc[t][0] and split[t] == "val"],
           2: [t for t in acc if not acc[t][0]]}
    # 数据：① 验证集回放（analyze 的结果）
    ev = json.loads((C.OUT / "eval/val/summary.json").read_text())
    ids = sorted(EV._ids("val"), key=lambda t: ev["joint_raw"]["per_group"][t]["median"])
    # 数据：末端成分（重算；验证集回放 + 同一批轨迹的教师验收）
    ctx_v, rel_net, rel_tch = B.session_ctx(VAL), [], []
    for tid in ids:
        q, s = B.load_seq(tid), acc[tid][1]
        ctx_t = B.session_ctx(s)
        mean = lambda sess, rids, ctx: np.mean([ee_parts(sess, r, q, ctx) for r in rids], 0)   # noqa: E731
        rel_net.append(mean(VAL, [f"{tid}_enet{r}" for r in (1, 2)], ctx_v) / mean(VAL, [f"{tid}_eshift{r}" for r in (1, 2)], ctx_v) - 1)
        rel_tch.append(mean(s, [f"{tid}_at0{r}" for r in (1, 2, 3)], ctx_t) / mean(s, [f"{tid}_as0{r}" for r in (1, 2, 3)], ctx_t) - 1)
        print(f"{tid} 末端成分已算", flush=True)
    med_net, med_tch = np.median(rel_net, 0) * 100, np.median(rel_tch, 0) * 100
    # 数据：p47 的偏差曲线、抓放阶段、大峰
    tid, s = "p47", acc["p47"][1]
    q = B.load_seq(tid)
    n, a6 = len(q["a7"]), q["a7"][:, :6].astype(float)
    sh = [ep_of(s, f"{tid}_{k}") for k in ("b01", "b02", "as01", "as02", "as03")]
    cur = {k: np.sqrt(FT.dev_joints(v, a6, n).mean(1)) * 1e3 for k, v in
           (("as", sh), ("at", [ep_of(s, f"{tid}_at0{r}") for r in (1, 2, 3)]))}
    tabs = FT.T0 + FT.SKIP + np.arange(len(cur["as"])) / D.FS_Q
    off = FT.excluded(q, n)
    free = ~off[np.clip((tabs / RR.DT).astype(int), 0, n - 1)]
    from scipy.signal import find_peaks
    yf = np.where(free, cur["as"], 0.0)
    pk, _ = find_peaks(yf, distance=int(D.FS_Q))
    pk = pk[np.argsort(yf[pk])[::-1][:5]]
    tr = np.flatnonzero(np.diff((q["a7"][:, 6] > O.load_cfg(C.CFG).grip_open).astype(int))) + 1
    edges = [RR.HOLD, *tr, n]
    curve = json.loads((C.OUT / "net/curve.json").read_text())

    for lang, suf in (("zh", ""), ("ja", "_ja")):
        L = TEXT[lang]
        # 图 1：全部教师
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
        rng = np.random.default_rng(0)
        for col, (A, title) in enumerate(zip(ax, (L["joint"], L["ee"]))):
            for g, tids in grp.items():
                x = [tch[t][col] for t in tids]
                A.scatter(x, g + rng.uniform(-0.18, 0.18, len(x)), s=26, color=T.BLUE if g < 2 else "white",
                          edgecolor=T.BLUE, linewidth=1.2, zorder=3)
                if g < 2:
                    m = np.median(x)
                    A.plot([m, m], [g - 0.3, g + 0.3], color=T.INK, lw=1.6, zorder=4)
                    A.text(m, g - 0.33, f"{L['med']} {m:+.1f}%", ha="center", va="bottom", fontsize=8.5, color=T.INK)
            A.axvline(0, color=T.MUTED, lw=0.8)
            A.set_title(title, loc="left", color=T.INK)
            A.set_xlabel(L["chg"])
            A.grid(axis="x", color=T.SHADE, lw=0.6)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[0].set_yticks([0, 1, 2], [f"{name}（{len(grp[g])}）" for g, name in enumerate(L["groups"])])
        ax[0].set_ylim(2.4, -0.6)                        # 上下翻转，并给中位标注留出位置
        fig.suptitle(L["a_title"], x=0.01, ha="left", color=T.INK)
        fig.text(0.01, 0.01, L["a_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"fig_teachers{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 2：① 验证集回放，每条轨迹与该轨迹的教师
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        y = np.arange(len(ids))
        for col, (A, key) in enumerate(zip(ax, ("joint_raw", "ee_raw"))):
            A.scatter([tch[t][col] for t in ids], y, s=34, color=T.BLUE, label=L["teacher"], zorder=3)
            A.scatter([ev[key]["per_group"][t]["median"] for t in ids], y, s=34, color=T.ORANGE, label=L["net"], zorder=3)
            lo, hi = ev[key]["ci95"]
            A.axvspan(lo, hi, color=T.ORANGE, alpha=0.12, lw=0)
            A.axvline(ev[key]["effect"], color=T.ORANGE, lw=1.2, ls="--")
            A.axvline(0, color=T.MUTED, lw=0.8)
            A.set_title(f"{L['joint'] if col == 0 else L['ee']}\n① {ev[key]['effect']:+.1f}% [{lo:+.1f}, {hi:+.1f}]", loc="left", color=T.INK)
            A.set_xlabel(L["chg"])
            A.grid(axis="x", color=T.SHADE, lw=0.6)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[0].set_yticks(y, [t + ("" if acc[t][0] else "*") + ("†" if acc[t][1].name in LOOSE else "") for t in ids])
        fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                   ncol=2, labelcolor=T.INK)                # 和总标题同一行，不挤右图的小标题
        fig.suptitle(L["b_title"], x=0.01, ha="left", color=T.INK)
        fig.text(0.01, 0.01, L["b_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"fig_net1_val{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 3：末端变差在哪
        fig, A = plt.subplots(figsize=(8.5, 4.2))
        y = np.arange(len(L["items"]))
        for off_, vals, color, label in ((-0.19, med_tch, T.BLUE, L["teacher"]), (0.19, med_net, T.ORANGE, L["net"])):
            A.barh(y + off_, vals, height=0.36, color=color, label=label, zorder=3)
            for yy, v in zip(y + off_, vals):
                A.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                       fontsize=8.5, color=T.INK)
        A.axvline(0, color=T.MUTED, lw=0.8)
        A.set_yticks(y, L["items"])
        A.invert_yaxis()
        A.set_xlim(min(med_tch.min(), med_net.min()) - 6, max(med_tch.max(), med_net.max()) + 7)
        A.set_xlabel(L["c_x"])
        A.set_title(L["c_title"], loc="left", color=T.INK)
        A.grid(axis="x", color=T.SHADE, lw=0.6)
        A.legend(frameon=False, loc="upper right", labelcolor=T.INK)
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
        fig.text(0.01, 0.01, L["c_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"fig_ee_why{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 4：p47 的 Shift 与 ILC 教师
        fig, A = plt.subplots(figsize=(12, 4.3))
        for k0, k1 in zip(np.flatnonzero(off & ~np.r_[False, off[:-1]]), np.flatnonzero(off & ~np.r_[off[1:], False])):
            if (k1 + 1) * RR.DT > FT.T0 + FT.SKIP:
                A.axvspan(max(k0 * RR.DT, FT.T0 + FT.SKIP) - FT.T0, (k1 + 1) * RR.DT - FT.T0, color=T.SHADE, alpha=0.5, lw=0)
        t = tabs - FT.T0
        A.plot(t, cur["as"], color=T.MUTED, lw=1.4, label=L["shift5"])
        A.plot(t, cur["at"], color=T.BLUE, lw=1.4, label=L["teacher3"])
        for i, p in enumerate(sorted(pk, key=lambda p: -yf[p]), 1):
            A.text(t[p], cur["as"][p] + 0.15, str(i), ha="center", va="bottom", fontsize=10, weight="bold", color=T.INK)
        for (f0, f1), name in zip(zip(edges[:-1], edges[1:]), L["phases"]):
            A.axvline(f0 * RR.DT - FT.T0, color=T.MUTED, lw=0.6, ls=":")
            A.text(((f0 + f1) / 2) * RR.DT - FT.T0, 1.01, name, transform=A.get_xaxis_transform(), ha="center", va="bottom",
                   fontsize=9, color=T.INK)                # 阶段名放在图框上沿外，不压曲线
        A.set_xlim(FT.SKIP, t[-1])
        A.set_ylim(0, 1.25 * cur["as"][free].max())
        A.set_xlabel(L["time"])
        A.set_ylabel(L["d_y"])
        A.set_title(L["d_title"], loc="left", color=T.INK, pad=18)
        A.grid(axis="y", color=T.SHADE, lw=0.6)
        A.legend(frameon=False, loc="upper right", labelcolor=T.INK)
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
        fig.text(0.01, 0.01, L["d_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(OUT / f"fig_floor_p47{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 5：学习曲线
        fig, A = plt.subplots(figsize=(4.8, 3.3))
        xs, ys = [p["n"] for p in curve], [p["val_r2_mean"] for p in curve]
        A.plot(xs, ys, "o-", color=T.BLUE, lw=1.6)
        for x_, y_ in zip(xs, ys):
            A.text(x_, y_ + 0.03, f"{y_:.2f}".replace("-0.00", "0.00"), ha="center", va="bottom", fontsize=9, color=T.INK)
        A.axhline(0, color=T.MUTED, lw=0.8)
        A.set_xticks(xs)
        A.set_ylim(min(ys) - 0.1, max(ys) + 0.15)
        A.set_xlabel(L["e_x"])
        A.set_ylabel(L["e_y"])
        A.set_title(L["e_title"], loc="left", color=T.INK)
        A.grid(axis="y", color=T.SHADE, lw=0.6)
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_curve{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)
    print(f"→ {OUT}（5 张 × 中、日）")
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
