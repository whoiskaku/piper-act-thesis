"""整段版 ①（方案 reports/PLAN_b4_seq.md）验证集真机结果：图与末端拆解，中文、日语各一版 → outputs/b4/report/（日语版带 _ja）。

  python -u sim/fig_b4_seq.py
  python -u sim/fig_b4_seq.py --teacher-time  # 只另画 p44 中日时间曲线，加入教师验收，保留原图

  fig_seq_val     每条验证轨迹：整段版 ①（会话 20261106q）相对 Shift 的关节、末端变化；一帧版 ①（11-05 会话 20261105v）只作跨会话粗比
  fig_seq_breakdown  左：主指标（关节位置偏差）拆成各关节；右：末端角速度偏差（只报告）按成分、频段、腕部关节拆开；
                     整段版、一帧版、教师（10 条中位）。末端角速度偏差增大只说明这个量增大，不等于「末端变差」。
  fig_seq_time       表现最好的那条轨迹（整段版关节偏差下降最多，不代表典型）的关节偏差随时间：五关节合计、j1、j5；
                     同会话的 Shift 与整段版。
  fig_seq_time_nocontact  同上，换成抓取时不碰桌的那条（NOCONTACT，用户看录制视频确认）。
拆解表另打印出来，并写 outputs/b4/report/seq_breakdown.json。
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
import b4_eval as EV  # noqa: E402
import b4_teacher as B  # noqa: E402
import cmd_tracking as T  # noqa: E402
import fig_b4_report as R  # noqa: E402

SEQ = B.TEACH / "sessions/20261106q"                     # 整段版 ① 验证集回放（shift / seq）
AQUA = "#1baf7a"                                          # 分类色第 3 位（dataviz 参考色板；教师蓝、一帧版橙沿用）
# 抓取时不碰桌的那条：用户 11-06 看 10-29 录制视频（腕部相机）确认 p23 不碰桌。
# 参考：按「末端最低点后 IMU 尖峰」排序时 p23 次小（0.334 rad/s；最小 p28 0.235，最大 p31 0.745）
NOCONTACT = "p23"
TEXT = {
    "zh": {"joint": "关节偏差（相对原始策略指令）", "ee": "末端角速度偏差（IMU 测，只报告）", "chg": "相对 Shift 的变化（%）",
           "seq": "整段版 ①（20261106q）", "one": "一帧版 ①（11-05，另一会话）", "tch": "教师（验收）", "seq_short": "整段版 ①",
           "a_title": "整段版 ① 在验证集轨迹上（真机，每条 2 轮）",
           "a_foot": "一帧版来自 11-05 另一会话，只作粗比；离线门槛未过、用户看结果后决定上机；这 10 条未参与训练、参与过模型选择",
           "items": ["末端角速度偏差（总）", "编码器能解释的部分", "只有 IMU 看得到的部分", "0.5–5 Hz", "5–15 Hz",
                     "j5 关节角速度偏差", "j6 关节角速度偏差"],
           "pos": "位置偏差", "b_ee": "末端角速度偏差拆开看（只报告）", "b_pos": "主指标拆成各关节（位置偏差）",
           "b_title": "主指标的各关节拆分与末端角速度偏差的拆解", "chg_med": "相对 Shift 的变化（%，验证集 10 条中位）",
           "b_foot": "编码器能解释 = 编码器推算 − 指令对应值；只有 IMU 看得到 = IMU 实测 − 编码器推算；"
                     "位置偏差取 ē_q（5 Hz 低通、按帧，与主指标算法不同，只看方向和相对大小）；一帧版、教师来自其他会话",
           "c_title": "{tid}：关节偏差随时间——Shift 与整段版 ①", "shift2": "Shift", "seq_t": "整段版 ①",
           "rows": ["五关节合计 / mrad", "j1 / mrad", "j5 / mrad"], "t_x": "时间 / s",
           "c_foot": "真机验证集；关节偏差相对原始策略指令；{tid} = 10 条里整段版关节偏差下降最多的那条，不代表典型；"
                     "同会话各 2 遍，逐点均方根、0.25 s 平滑；灰色 = 不计入评价",
           "t_title": "{tid}：关节偏差随时间——Shift、整段版 ①、ILC 教师", "teacher_t": "ILC 教师（另一会话，3 遍）",
           "t_foot": "真机验证集；偏差相对原始策略指令。{tid} 是整段版关节偏差下降最多的轨迹，不代表典型。\n"
                     "Shift、整段版：11-06 同会话各 2 遍；教师：11-04 另一会话、空桌验收 3 遍。逐点 RMS、0.25 s 平滑；灰色 = 不计入评价",
           "n_title": "{tid}：关节偏差随时间——Shift 与整段版 ①，抓取时不碰桌",
           "n_foot": "真机验证集；关节偏差相对原始策略指令；{tid} = 看录制视频确认抓取时不碰桌的那条；"
                     "同会话各 2 遍，逐点均方根、0.25 s 平滑；灰色 = 不计入评价"},
    "ja": {"joint": "関節偏差（元の方策指令との差）", "ee": "手先角速度偏差（IMU 計測・参考値）", "chg": "Shift に対する変化（%）",
           "seq": "系列出力版 ①（20261106q）", "one": "1 フレーム版 ①（11-05、別セッション）", "tch": "教師（受入れ試験）",
           "seq_short": "系列出力版 ①", "a_title": "系列出力版 ① の検証軌道での結果（実機、各 2 回）",
           "a_foot": "1 フレーム版は 11-05 の別セッションの結果で参考比較のみ。オフライン基準は未達だが結果を見て実機試行を決定。"
                     "この 10 本は学習には未使用、モデル選択には使用",
           "items": ["手先角速度偏差（全体）", "エンコーダで説明できる部分", "IMU でしか見えない部分", "0.5–5 Hz", "5–15 Hz",
                     "j5 関節角速度偏差", "j6 関節角速度偏差"],
           "pos": "位置偏差", "b_ee": "手先角速度偏差の内訳（参考値）", "b_pos": "主指標の関節別内訳（位置偏差）",
           "b_title": "主指標の関節別内訳と手先角速度偏差の内訳", "chg_med": "Shift に対する変化（%、検証 10 軌道の中央値）",
           "b_foot": "エンコーダで説明できる = エンコーダ推定 − 指令対応値。IMU でしか見えない = 実測 − 推定。"
                     "位置偏差は ē_q（5 Hz 低域通過、主指標とは算出法が異なる）。1 フレーム版・教師は別セッション",
           "c_title": "{tid}：関節偏差の時間変化——Shift と系列出力版 ①", "shift2": "Shift", "seq_t": "系列出力版 ①",
           "rows": ["5 関節合計 / mrad", "j1 / mrad", "j5 / mrad"], "t_x": "時間 / s",
           "c_foot": "実機・検証データ。関節偏差は元の方策指令との差。{tid} = 10 本中で系列出力版の関節偏差の減少が最大の軌道で、典型例ではない。"
                     "同セッション各 2 回、点ごとの二乗平均平方根、0.25 s 平滑化。灰色 = 評価対象外",
           "t_title": "{tid}：関節偏差の時間変化——Shift・系列出力版 ①・ILC 教師", "teacher_t": "ILC 教師（別セッション・3 回）",
           "t_foot": "実機・検証データ。偏差は元の方策指令との差。{tid} は系列出力版の関節偏差減少が最大の軌道で、典型例ではない。\n"
                     "Shift・系列出力版：11-06 同セッション各 2 回。教師：11-04 別セッション・空卓での受入れ試験 3 回。"
                     "点ごとの RMS、0.25 s 平滑化。灰色 = 評価対象外",
           "n_title": "{tid}：関節偏差の時間変化——Shift と系列出力版 ①、把持時にテーブル非接触",
           "n_foot": "実機・検証データ。関節偏差は元の方策指令との差。{tid} = 収録動画で把持時にテーブルに当たらないことを確認した軌道。"
                     "同セッション各 2 回、点ごとの二乗平均平方根、0.25 s 平滑化。灰色 = 評価対象外"},
}


def rel(sess: Path, ctx, tid: str, q: dict, a: str, b: str, rounds: tuple) -> np.ndarray:
    """一条轨迹：条件 a 相对 b 的相对变化（12 项）= ee_parts 的 7 项 + 五个活动关节的位置偏差（ē_q 的 RMS）。"""
    f = lambda rid: np.r_[R.ee_parts(sess, rid, q, ctx),                                      # noqa: E731
                          np.sqrt(np.mean(B.replay_result(rid, sess, q, ctx)[1][0] ** 2, 0))]
    m = lambda c: np.mean([f(f"{tid}_{c}{r}") for r in rounds], 0)                           # noqa: E731
    return m(a) / m(b) - 1


def time_data(tid: str) -> dict:
    """一条轨迹的关节偏差随时间（同会话 Shift、整段版各 2 遍），以及抓放阶段的分界、不计入评价的帧。"""
    q = B.load_seq(tid)
    n, a6 = len(q["a7"]), q["a7"][:, :6].astype(float)
    sh = [R.ep_of(SEQ, f"{tid}_qshift{r}") for r in (1, 2)]
    dev = {"shift": R.FT.dev_joints(sh, a6, n),
           "seq": R.FT.dev_joints([R.ep_of(SEQ, f"{tid}_qseq{r}") for r in (1, 2)], a6, n)}
    tr = np.flatnonzero(np.diff((q["a7"][:, 6] > R.O.load_cfg(R.C.CFG).grip_open).astype(int))) + 1
    return {"tid": tid, "dev": dev, "t": R.FT.SKIP + np.arange(len(dev["shift"])) / R.D.FS_Q,
            "excl": R.FT.excluded(q, n), "edges": [R.RR.HOLD, *tr, n]}


def draw_time(plt, d: dict, L: dict, lang: str, key: str, path: Path) -> None:
    """一条轨迹的关节偏差随时间：五关节合计、j1、j5 三层；标题、图注取 L[key + "_title"]、L[key + "_foot"]。"""
    dev, t, excl, edges = d["dev"], d["t"], d["excl"], d["edges"]
    with_teacher = "teacher" in dev
    panels = ((L["rows"][0], lambda x: np.sqrt(x.mean(1))), (L["rows"][1], lambda x: np.sqrt(x[:, 0])),
              (L["rows"][2], lambda x: np.sqrt(x[:, 3])))
    fig, ax = plt.subplots(3, 1, figsize=(12, 8.2), sharex=True)
    for A, (name, f) in zip(ax, panels):
        for k0, k1 in zip(np.flatnonzero(excl & ~np.r_[False, excl[:-1]]), np.flatnonzero(excl & ~np.r_[excl[1:], False])):
            if (k1 + 1) * R.RR.DT > R.FT.T0 + R.FT.SKIP:
                A.axvspan(max(k0 * R.RR.DT, R.FT.T0 + R.FT.SKIP) - R.FT.T0, (k1 + 1) * R.RR.DT - R.FT.T0,
                          color=T.SHADE, alpha=0.5, lw=0)
        A.plot(t, 1e3 * f(dev["shift"]), color=T.MUTED, lw=1.3, label=L["shift2"])
        A.plot(t, 1e3 * f(dev["seq"]), color=AQUA, lw=1.5, label=L["seq_t"])
        if with_teacher:
            A.plot(t, 1e3 * f(dev["teacher"]), color=T.BLUE, lw=1.4, label=L["teacher_t"])
        for f0 in edges[1:-1]:
            A.axvline(f0 * R.RR.DT - R.FT.T0, color=T.MUTED, lw=0.6, ls=":")
        A.set_ylabel(name)
        A.set_ylim(0, None)
        A.grid(axis="y", color=T.SHADE, lw=0.6)
        for side in ("top", "right"):
            A.spines[side].set_visible(False)
    for (f0, f1), name in zip(zip(edges[:-1], edges[1:]), R.TEXT[lang]["phases"]):
        ax[0].text(((f0 + f1) / 2) * R.RR.DT - R.FT.T0, 1.01, name, transform=ax[0].get_xaxis_transform(),
                   ha="center", va="bottom", fontsize=9, color=T.INK)              # 抓放阶段名放在图框上沿外
    ax[-1].set_xlim(R.FT.SKIP, t[-1])
    ax[-1].set_xlabel(L["t_x"])
    ax[0].set_title(L[f"{key}_title"].format(tid=d["tid"]), loc="left", color=T.INK, pad=18)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="lower right", bbox_to_anchor=(0.995, 0.035),
               ncol=3 if with_teacher else 2, labelcolor=T.INK)
    fig.text(0.01, 0.01, L[f"{key}_foot"].format(tid=d["tid"]), color=T.MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.09 if with_teacher else 0.07, 1, 1))
    fig.savefig(path, dpi=160, facecolor="#fcfcfb")
    plt.close(fig)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9.5, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    if "--teacher-time" in sys.argv:
        tid = "p44"
        row = json.loads((B.TEACH / "summary.json").read_text())[tid]
        teacher_sess = B.TEACH / "sessions" / row["session"]
        d, q = time_data(tid), B.load_seq(tid)
        d["dev"]["teacher"] = R.FT.dev_joints(R.FT.episodes(tid, teacher_sess, "at"),
                                              q["a7"][:, :6].astype(float), len(q["a7"]))
        for lang, suf in (("zh", ""), ("ja", "_ja")):
            out = R.OUT / f"fig_seq_time_teacher{suf}.png"
            draw_time(plt, d, TEXT[lang], lang, "t", out)
            print(f"{tid} 教师：{teacher_sess.name}，验收 3 遍 → {out}", flush=True)
        return 0
    acc = {Path(f).stem[len("accept_"):]: Path(f).parent for f in glob.glob(str(B.TEACH / "sessions/*/accept_*.json"))}
    ev = {k: json.loads((EV.EVAL / k / "summary.json").read_text()) for k in ("valseq", "val")}
    ids = sorted(EV._ids("valseq"), key=lambda t: ev["valseq"]["joint_raw"]["per_group"][t]["median"])

    # 数据：逐条轨迹的拆解（整段版、一帧版各自同会话相对 Shift；教师相对它自己验收时的 Shift）
    ctx = {s: B.session_ctx(s) for s in (SEQ, R.VAL)}
    rows = {"seq": [], "one": [], "tch": []}
    for tid in ids:
        q = B.load_seq(tid)
        rows["seq"].append(rel(SEQ, ctx[SEQ], tid, q, "qseq", "qshift", (1, 2)))
        rows["one"].append(rel(R.VAL, ctx[R.VAL], tid, q, "enet", "eshift", (1, 2)))
        rows["tch"].append(rel(acc[tid], B.session_ctx(acc[tid]), tid, q, "at0", "as0", (1, 2, 3)))
        print(f"{tid} 已算", flush=True)
    med = {k: np.median(v, 0) * 100 for k, v in rows.items()}
    names = TEXT["zh"]["items"] + [f"{j} 位置偏差" for j in ("j1", "j2", "j3", "j5", "j6")]
    print(f"\n{'（10 条中位，%）':<18}{'整段版':>9}{'一帧版':>9}{'教师':>9}")
    for i, n in enumerate(names):
        print(f"{n:<18}{med['seq'][i]:+9.1f}{med['one'][i]:+9.1f}{med['tch'][i]:+9.1f}")
    R.OUT.mkdir(parents=True, exist_ok=True)
    (R.OUT / "seq_breakdown.json").write_text(json.dumps({"items": names, "ids": ids, **{k: v.tolist() for k, v in med.items()}},
                                                      ensure_ascii=False, indent=1))

    # 数据：两条轨迹的关节偏差随时间——表现最好的（ids 已按整段版关节变化升序排好）与抓取时不碰桌的
    times = {"fig_seq_time": (time_data(ids[0]), "c"), "fig_seq_time_nocontact": (time_data(NOCONTACT), "n")}

    for lang, suf in (("zh", ""), ("ja", "_ja")):
        L = TEXT[lang]
        # 图 1：每条轨迹，整段版与一帧版（跨会话）
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
        y = np.arange(len(ids))
        for col, (A, key) in enumerate(zip(ax, ("joint_raw", "ee_raw"))):
            s = ev["valseq"][key]
            A.scatter([ev["val"][key]["per_group"][t]["median"] for t in ids], y, s=34, facecolor="white",
                      edgecolor=T.ORANGE, linewidth=1.4, label=L["one"], zorder=3)
            A.scatter([s["per_group"][t]["median"] for t in ids], y, s=34, color=AQUA, label=L["seq"], zorder=3)
            A.axvspan(*s["ci95"], color=AQUA, alpha=0.15, lw=0)
            A.axvline(s["effect"], color=AQUA, lw=1.2, ls="--")
            A.axvline(0, color=T.MUTED, lw=0.8)
            A.set_title(f"{L['joint'] if col == 0 else L['ee']}\n{L['seq_short']} {s['effect']:+.1f}% "
                        f"[{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}]", loc="left", color=T.INK)
            A.set_xlabel(L["chg"])
            A.grid(axis="x", color=T.SHADE, lw=0.6)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[0].set_yticks(y, ids)
        fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                   ncol=2, labelcolor=T.INK)
        fig.suptitle(L["a_title"], x=0.01, ha="left", color=T.INK)
        fig.text(0.01, 0.01, L["a_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(R.OUT / f"fig_seq_val{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 2：左：主指标拆成各关节的位置偏差（5 项）；右：末端角速度偏差的拆解（7 项，只报告）
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.0), gridspec_kw={"width_ratios": (5, 7)})
        groups = ((ax[0], slice(7, 12), [f"{j} {L['pos']}" for j in ("j1", "j2", "j3", "j5", "j6")], L["b_pos"]),
                  (ax[1], slice(0, 7), L["items"], L["b_ee"]))
        for A, sl, labels, title in groups:
            yy = np.arange(len(labels))
            for off, k, color, label in ((-0.27, "tch", T.BLUE, L["tch"]), (0.0, "one", T.ORANGE, L["one"]),
                                         (0.27, "seq", AQUA, L["seq"])):
                vals = med[k][sl]
                A.barh(yy + off, vals, height=0.25, color=color, label=label, zorder=3)
                for v_, y_ in zip(vals, yy + off):
                    A.text(v_ + (0.8 if v_ >= 0 else -0.8), y_, f"{v_:+.1f}", va="center", ha="left" if v_ >= 0 else "right",
                           fontsize=7.5, color=T.INK)
            A.axvline(0, color=T.MUTED, lw=0.8)
            A.set_yticks(yy, labels)
            A.invert_yaxis()
            lo, hi = min(med[k][sl].min() for k in med), max(med[k][sl].max() for k in med)
            A.set_xlim(min(lo, 0) - 8, max(hi, 0) + 9)
            A.set_title(title, loc="left", color=T.INK)
            A.set_xlabel(L["chg_med"])
            A.grid(axis="x", color=T.SHADE, lw=0.6)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="upper right", bbox_to_anchor=(0.995, 0.995),
                   ncol=3, labelcolor=T.INK)
        fig.suptitle(L["b_title"], x=0.01, ha="left", color=T.INK)
        fig.text(0.01, 0.01, L["b_foot"], color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(R.OUT / f"fig_seq_breakdown{suf}.png", dpi=160, facecolor="#fcfcfb")
        plt.close(fig)

        # 图 3、4：关节偏差随时间（表现最好的、抓取时不碰桌的）
        for name, (d, key) in times.items():
            draw_time(plt, d, L, lang, key, R.OUT / f"{name}{suf}.png")
    print(f"→ {R.OUT}（4 张 × 中、日；随时间：{ids[0]} 表现最好、{NOCONTACT} 不碰桌）")
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
