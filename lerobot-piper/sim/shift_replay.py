#!/usr/bin/env python3
"""Shift 回放对照（方案 reports/PLAN_shift_replay.md v1）：实测「提前发指令」让实际运动离原始策略指令近多少。

  seq      由可重复性检验冻结的 A、B 生成三个条件的序列：基线、提前 15 ms、提前 60 ms（第 k 帧发原序列在 k·DT + Δ 处的值）。
  replay   每轮 A、B 各跑三个条件（顺序按轮轮换），共 6 轮 36 遍；每遍一个子进程调 replay_probe，之间等回车，开始过的不再跑。
  analyze  误差一律相对原始策略指令（按本遍 t_send 对齐第 k 帧）：关节偏差、末端角速度偏差各两种参考；配对「条件 − 基线」，
           按评价标准判据 b 判；护栏；与离线预测（基线回放的实际运动整体提前 Δ）对照；曲线图。

用法：
    python -u sim/shift_replay.py seq
    python -u sim/shift_replay.py replay --dry-run     # 六个序列过一遍回放管线的安全检查，不连机械臂
    python -u sim/shift_replay.py replay               # 真机；中断后重新运行即从下一遍接着跑
    python -u sim/shift_replay.py analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import cmd_tracking as T  # noqa: E402  读取、理想渲染、标定后的 IMU 渲染
import repeat_reversal as RR  # noqa: E402  冻结序列、接触标记、回放子进程、「开始过」检查
from step4_gvnext import data as D  # noqa: E402  低通、采样率

OUT = HERE.parent / "outputs/shift_replay"
SEQS = ("A", "B")
CONDS = {"base": 0.0, "d15": 0.015, "d60": 0.060}          # 条件 → 提前量（s）
LABEL = {"base": "基线", "d15": "提前 15 ms", "d60": "提前 60 ms"}
N_REP = 6
MIN_PAIRS = 5                    # 每条序列每个条件至少 5 对完整配对（6 遍的 ≥75%，同可重复性检验 6/8）才判定，否则只报告
TH_DROP, TH_WORSE, TH_GUARD = 5.0, 5.0, 5.0               # %：判据 b 的门槛草稿（评价标准 §4）
FAMILIES = {"raw": ("joint_raw", "ee_raw", "原始指令一族"), "ideal": ("joint_ideal", "ee_ideal", "理想渲染一族")}


# ── 序列与排程 ───────────────────────────────────────────────────────────
def shifted(cmd7, delta, dt=RR.DT):
    """第 k 帧发原序列在 k·dt + delta 处的值（帧间线性插值，末尾保持最后一帧）；整条动作流（含夹爪）一起提前。"""
    t = dt * np.arange(len(cmd7))
    return np.stack([np.interp(t + delta, t, cmd7[:, j]) for j in range(cmd7.shape[1])], 1).astype(np.float32)


def schedule(n_rep=N_REP):
    """[(序列, 条件, 轮次)]：每轮 A、B 各跑三个条件，条件顺序按轮轮换（拉丁方），各条件在每个位置出现的次数相同。"""
    conds = list(CONDS)
    return [(s, conds[(i + r) % len(conds)], r + 1) for r in range(n_rep) for s in SEQS for i in range(len(conds))]


def cmd_seq():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in SEQS:
        cmd7 = np.load(RR.OUT / f"seq_{s}.npz")["cmd7"].astype(float)
        for c, d in CONDS.items():
            x = shifted(cmd7, d)
            src = {"base_seq": str(RR.OUT / f"seq_{s}.npz"), "cond": c, "delta_s": d,
                   "cmd_hash": hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]}   # 同 replay_probe 的算法
            np.savez(OUT / f"seq_{s}_{c}.npz", cmd7=x, da=np.zeros((len(x), 6), np.float32), source=json.dumps(src))
            print(f"{s} · {LABEL[c]}：{len(x)} 帧，与原指令的最大差 {np.degrees(np.abs(x[:, :6] - cmd7[:, :6]).max()):.2f}°，指令哈希 {src['cmd_hash']}")
    return 0


# ── 回放 ─────────────────────────────────────────────────────────────────
def run_one(s, c, r):
    """子进程跑一遍（本脚本的序列与命名），见 guarded_run。"""
    return guarded_run(OUT / f"seq_{s}_{c}.npz", f"{s}_{c}_", f"r{r}", OUT)


def guarded_run(seqfile, tag, arm, outdir):
    """子进程跑一遍，接住 replay_probe 非 pilot 路径里冒出的异常。预移动、静置阶段的异常（Ctrl-C、到位或稳定断言）
    在那里直接抛出，不落盘也不归位；任其冒出，PiperRobot 被释放时 __del__ 会发一次无插值归位。这里：
    ① 还没有回放记录 → 落一份 stopped 记录（不补跑、分析排除），再走 replay_probe 自己的分段插值归位（受阻即停）；
    ② 已有记录 → 管线已落盘并归位过，不再动，只在 stopped 为空时补上原因；
    两种情况最后都把 is_connected 置 False（__del__ 只在连接时归位），返回非零，由 main 的 os._exit 退出。"""
    import importlib
    import traceback
    import replay_probe as RP
    U = importlib.import_module("lerobot.common.robot_devices.robots.utils")
    made, make = [], U.make_robot
    U.make_robot = lambda *a, **k: made.append(make(*a, **k)) or made[-1]     # 留住 robot，异常时才拿得到
    p = outdir / f"replay_{tag}{arm}.json"
    try:
        return RP.run(arm, False, seqfile=str(seqfile), tag=tag, outdir=outdir)
    except BaseException as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"\n  ✗ 这一遍异常退出：{reason}")
        if p.exists():
            rec = json.loads(p.read_text())
            rec["stopped"] = rec.get("stopped") or reason
            p.write_text(json.dumps(rec))
            print("  已有回放记录（管线已落盘并归位过），不再归位。")
        else:
            rec = {"arm": arm, "seqfile": str(seqfile), "stopped": reason,
                   "traceback": traceback.format_exc()}
            p.write_text(json.dumps(rec))
            if made and made[0].is_connected:
                try:
                    import torch
                    grip = float(np.clip(np.load(seqfile)["cmd7"][0, 6], 0, 0.035))
                    if not RP.staged_return(made[0], torch, grip):
                        rec["return_blocked"] = True
                except BaseException as e2:                       # 归位中再按 Ctrl-C 也走到这里
                    rec["return_error"] = f"{type(e2).__name__}: {e2}"
                p.write_text(json.dumps(rec))
        return 1
    finally:
        U.make_robot = make
        for robot in made:
            robot.is_connected = False


def cmd_replay(dry_run, one=None):
    import replay_probe as RP
    if one:                          # 子进程：跑一遍，如 "A,d15,3"
        return run_one(*one.split(","))
    if dry_run:
        return max([RP.run("r0", True, seqfile=str(OUT / f"seq_{s}_{c}.npz"), tag=f"{s}_{c}_", outdir=OUT)
                    for s in SEQS for c in CONDS])
    sha = hashlib.sha256((HERE / "replay_probe.py").read_bytes()).hexdigest()
    plan = schedule()
    for k, (s, c, r) in enumerate(plan, 1):
        if RR.attempted(f"{s}_{c}", r, OUT):
            continue
        if input(f"\n[{k}/{len(plan)}] 序列 {s} · {LABEL[c]} · 第 {r} 轮：桌面已清空、可以运动就按回车（输入 q 退出）").strip().lower() == "q":
            return 0
        utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        code = RR.run_child([sys.executable, "-u", __file__, "replay", "--one", f"{s},{c},{r}"])
        with open(OUT / "replay_log.jsonl", "a") as f:
            f.write(json.dumps({"k": k, "seq": s, "cond": c, "rep": r, "utc": utc, "code": code, "replay_probe_sha256": sha}) + "\n")
        if code:
            print(f"  ⚠ 第 {k} 次返回码 {code}：这一遍不补跑。检查机械臂后再继续。")
    print(f"{len(plan)} 遍都已开始过。下一步：python -u sim/shift_replay.py analyze")
    return 0


# ── 分析 ─────────────────────────────────────────────────────────────────
class LinCmd:
    """原始指令的线性插值（在 t_send 处正好等于指令），按 `sp(t, nu)` 接口取值：位置线性插值，速度 = 帧间斜率，加速度 0。"""

    def __init__(self, ts, cmd):
        self.ts, self.cmd, self.slope = ts, cmd, np.diff(cmd, axis=0) / np.diff(ts)[:, None]

    def __call__(self, t, nu=0):
        t = np.atleast_1d(t)
        if nu == 0:
            y = np.stack([np.interp(t, self.ts, self.cmd[:, j]) for j in range(self.cmd.shape[1])], 1)
        elif nu == 1:
            y = self.slope[np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.slope) - 1)]
        else:
            y = np.zeros((len(t), self.cmd.shape[1]))
        return np.column_stack([y, np.zeros(len(t))])


def zoh(ts, cmd, t):
    """原始指令按写总线时刻零阶保持：时刻 t 生效的是第 k 帧（t_send[k] ≤ t 的最后一帧）。"""
    return cmd[np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)]


def joint_dev(tq, q, ts, cmd, free):
    """关节偏差（相对原始指令零阶保持）：五个活动关节 RMS，mrad；cmd 是原始策略指令，第 k 帧在本遍的 t_send[k] 生效。"""
    return float(np.sqrt(np.mean((zoh(ts, cmd, tq) - q)[free][:, T.ACTIVE] ** 2)) * 1e3)


def correction(sent, cmd):
    """补正量 |a_r − a_π|（发出的指令 − 原始指令），五个活动关节：{rms, max}，mrad。"""
    d = (np.asarray(sent)[:, :6] - np.asarray(cmd)[:, :6])[:, T.ACTIVE]
    return {"rms": float(np.sqrt(np.mean(d ** 2)) * 1e3), "max": float(np.abs(d).max() * 1e3)}


def raw_gyro(ep, cmd, calib, model, sid):
    """原始指令线性插值经运动学渲染、按标定换到 IMU 坐标的陀螺读数（IMU 时刻上）：末端角速度偏差的目标（原始指令一族）。"""
    import mujoco
    from ideal_imu_kinematic import ideal_imu_traj
    return (ideal_imu_traj(model, mujoco.MjData(model), LinCmd(ep["t_send"], cmd), ep["ti"] - calib["delay_s"], sid)[1][:, 3:]
            @ np.asarray(calib["R"]).T + np.asarray(calib["gyro_bias"]))


def sq_dev(ep, cmd, calib, model, sid):
    """一遍相对原始策略指令的逐样本平方偏差：关节两种参考（编码器网格上，五个活动关节平方的均值）、
    末端两种参考（IMU 时刻上，陀螺 <15 Hz 三轴平方和）。参考：原始指令一族 = 零阶保持（关节）/ 线性插值经运动学渲染（末端）；
    理想渲染一族 = 原指令经 a_max 规划、从实测起点出发。"""
    tq, q, ti, g, ts = ep["tq"], ep["q"], ep["ti"], ep["imu"][:, 3:], ep["t_send"]
    traj = T.PlanTrajectory(cmd, ts, q0=q[0])
    g_ideal = T.ideal_imu(traj, ti - calib["delay_s"], calib, model, sid)[1]
    g_raw = raw_gyro(ep, cmd, calib, model, sid)
    return {"joint_raw": np.mean((zoh(ts, cmd, tq) - q)[:, T.ACTIVE] ** 2, axis=1),
            "joint_ideal": np.mean((traj(tq)[:, :6] - q)[:, T.ACTIVE] ** 2, axis=1),
            "ee_raw": np.sum(D.lp(g - g_raw, T.LP_VEL, D.FS_I) ** 2, axis=1),
            "ee_ideal": np.sum(D.lp(g - g_ideal, T.LP_VEL, D.FS_I) ** 2, axis=1)}


def metrics(ep, cmd, free_q, free_i, calib, model, sid):
    """一遍的主指标与护栏量（非接触段、保持段之后）：关节偏差两种参考（mrad）、末端角速度偏差两种参考（rad/s）、
    关节路径长度（rad）、关节速度 RMS（rad/s）。"""
    sq = sq_dev(ep, cmd, calib, model, sid)
    out = {k: float(np.sqrt(np.mean(v[free_q]))) * 1e3 if k.startswith("joint") else float(np.sqrt(np.mean(v[free_i])))
           for k, v in sq.items()}
    v = T.lp_deriv(ep["q"], T.LP_VEL, 1)[:, T.ACTIVE]
    out["path"] = float(np.sum(np.abs(np.diff(ep["q"][:, T.ACTIVE], axis=0))[free_q[1:] & free_q[:-1]]))
    out["speed"] = float(np.sqrt(np.mean(v[free_q] ** 2)))
    return out


def masks(ep, marked, margin):
    """非接触段（marked：疑似接触帧）、保持段之后、末尾留 margin 秒：编码器网格与 IMU 时刻各一份。"""
    ts = ep["t_send"]

    def m(t):
        frame = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(marked) - 1)
        return ~marked[frame] & (t >= ts[RR.HOLD]) & (t <= ts[-1] - margin)

    return m(ep["tq"]), m(ep["ti"])


def shift_actual(ep, delta):
    """离线预测用：把实际运动（编码器、IMU）整体提前 delta 秒（参考不变）。"""
    out = dict(ep)
    out["q"] = np.stack([np.interp(ep["tq"] + delta, ep["tq"], ep["q"][:, j]) for j in range(6)], 1)
    out["imu"] = np.stack([np.interp(ep["ti"] + delta, ep["ti"], ep["imu"][:, k]) for k in range(6)], 1)
    return out


def load_cond(s, c):
    """一个条件的完整各遍 → {轮次: (from_record 结果, 发出的指令)}。完整 = 管线没有提前终止、指令哈希与帧数对得上、有编码器记录。"""
    z = np.load(OUT / f"seq_{s}_{c}.npz")
    src = json.loads(str(z["source"]))
    out = {}
    for r in range(1, N_REP + 1):
        p = OUT / f"replay_{s}_{c}_r{r}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        if (h.get("stopped") is None and h.get("cmd_hash") == src["cmd_hash"] and len(h["t_send"]) == len(z["cmd7"])
                and len(h.get("can_q200", [])) > 0):
            sent = np.asarray(h["cmd"], float)
            out[r] = (T.from_record(h, sent), sent)
    return out


def paired(rows, cond, key):
    """同序列同轮的「条件 − 基线」相对变化（%）：中位、自助法 95% 置信区间、Wilcoxon 符号秩 p、配对数。"""
    from scipy.stats import wilcoxon
    rel = np.array([(rows[(s, cond, r)][key] - rows[(s, "base", r)][key]) / rows[(s, "base", r)][key] * 100
                    for (s, c, r) in rows if c == cond and (s, "base", r) in rows])
    if len(rel) < 2:
        return None
    boot = np.median(np.random.default_rng(0).choice(rel, (10000, len(rel))), axis=1)
    return {"median": float(np.median(rel)), "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "p": float(wilcoxon(rel).pvalue) if np.any(rel != 0) else 1.0, "n": int(len(rel))}


def verdict_b(joint, ee, path, speed):
    """判据 b：两个主指标都不变差（置信区间上界 ≤ +5%），至少一个显著下降（上界 < 0、中位降幅 ≥ 5%）；护栏：路径长度、速度变化在 ±5% 内。"""
    if any(x is None for x in (joint, ee, path, speed)):
        return "无法判定（配对不足）"
    if not all(abs(x["median"]) <= TH_GUARD for x in (path, speed)):
        return "护栏未过（运动量或速度的变化超过 ±5%）"
    drops = [n for n, x in (("关节偏差", joint), ("末端角速度偏差", ee)) if x["ci95"][1] < 0 and x["median"] <= -TH_DROP]
    if joint["ci95"][1] <= TH_WORSE and ee["ci95"][1] <= TH_WORSE and drops:
        return f"有效（{'、'.join(drops)}显著下降）"
    return "无效"


def plot(curves):
    """每条序列一行；列 = 关节偏差（两种参考）、末端角速度偏差（两种参考）随时间（0.25 s 滑动 RMS）；三个条件各取中位那一遍。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    cols = (("joint_raw", "关节偏差 相对原始指令（mrad）"), ("joint_ideal", "关节偏差 相对理想渲染（mrad）"),
            ("ee_raw", "末端角速度偏差 相对原始指令（rad/s）"), ("ee_ideal", "末端角速度偏差 相对理想渲染（rad/s）"))
    colors = {"base": T.INK, "d15": T.BLUE, "d60": T.ORANGE}
    fig, ax = plt.subplots(len(SEQS), len(cols), figsize=(18, 3.2 * len(SEQS)), squeeze=False)
    for r, s in enumerate(SEQS):
        for c, (key, title) in enumerate(cols):
            A = ax[r, c]
            for cond in CONDS:
                if (s, cond) in curves:
                    A.plot(*curves[(s, cond)][key], color=colors[cond], lw=1.0, label=LABEL[cond])
            if r == 0:
                A.set_title(title, loc="left", color=T.INK)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[r, 0].set_ylabel(f"序列 {s}")
    for c in range(len(cols)):
        ax[-1, c].set_xlabel("时间（s，从第一帧起）")
    ax[0, -1].legend(frameon=False, labelcolor=T.INK)
    fig.suptitle("Shift 回放对照：各条件中位那一遍的偏差（相对原始策略指令，0.25 s 滑动 RMS，含接触段）", x=0.01, ha="left", color=T.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "curves.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def cmd_analyze():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    calib = json.loads((HERE.parent / "outputs/imu_calib/imu_calib.json").read_text())
    center = T.target_center("pickcube_raw", model)
    keys = ("joint_raw", "joint_ideal", "ee_raw", "ee_ideal")
    rows, corr, pred, curves = {}, {}, {}, {}
    for s in SEQS:
        cmd7 = np.load(OUT / f"seq_{s}_base.npz")["cmd7"].astype(float)
        cmd = cmd7[:, :6]
        marked = np.any([RR.contact_frames(np.load(OUT / f"seq_{s}_{c}.npz")["cmd7"].astype(float), model, center)
                         for c in CONDS], axis=0)                     # 三个条件的疑似接触帧取并集：各条件比的是同一段
        for c in CONDS:
            done = load_cond(s, c)
            for r, (ep, sent) in done.items():
                fq, fi = masks(ep, marked, max(CONDS.values()) + 0.05)
                rows[(s, c, r)] = metrics(ep, cmd, fq, fi, calib, model, sid)
                corr[(s, c, r)] = correction(sent, cmd7)
                if c == "base":
                    for cc, d in CONDS.items():
                        if d > 0:
                            pred[(s, cc, r)] = metrics(shift_actual(ep, d), cmd, fq, fi, calib, model, sid)
            ks = sorted(done, key=lambda r: rows[(s, c, r)]["joint_raw"])
            if ks:                                                    # 画图：这个条件中位那一遍
                ep = done[ks[len(ks) // 2]][0]
                curves[(s, c)] = {}
                for k, v in sq_dev(ep, cmd, calib, model, sid).items():
                    t, fs, unit = (ep["tq"], D.FS_Q, 1e3) if k.startswith("joint") else (ep["ti"], D.FS_I, 1.0)
                    w = int(0.25 * fs)                                # 0.25 s 滑动 RMS
                    curves[(s, c)][k] = (t - ep["t_send"][0], np.sqrt(np.convolve(v, np.ones(w) / w, "same")) * unit)
    n = {c: sum(1 for k in rows if k[1] == c) for c in CONDS}

    def med(d, c, k):
        x = [v[k] for (s, cc, r), v in d.items() if cc == c]
        return float(np.median(x)) if x else float("nan")

    res = {}
    for c in (c for c in CONDS if c != "base"):
        res[c] = {k: paired(rows, c, k) for k in (*keys, "path", "speed")}
        short = [sq for sq in SEQS if sum((sq, c, r) in rows and (sq, "base", r) in rows for r in range(1, N_REP + 1)) < MIN_PAIRS]
        res[c]["verdict"] = {f: f"只报告、不判定（序列 {'、'.join(short)} 完整配对少于 {MIN_PAIRS} 对）" if short
                             else verdict_b(res[c][j], res[c][e], res[c]["path"], res[c]["speed"]) for f, (j, e, _) in FAMILIES.items()}
    summary = {"plan": "reports/PLAN_shift_replay.md v1", "n_complete": n, "paired": res,
               "median": {c: {k: med(rows, c, k) for k in keys} for c in CONDS},
               "offline_prediction": {c: {k: med(pred, c, k) for k in keys} for c in CONDS if c != "base"},
               "correction_mrad": {c: {k: med(corr, c, k) for k in ("rms", "max")} for c in CONDS}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("完整回放：" + " · ".join(f"{LABEL[c]} {n[c]}/{N_REP * len(SEQS)}" for c in CONDS))
    m = summary["median"]["base"]
    print(f"基线（中位）：关节 相对原始指令 {m['joint_raw']:.2f}、相对理想渲染 {m['joint_ideal']:.2f} mrad；"
          f"末端 相对原始指令 {m['ee_raw']:.4f}、相对理想渲染 {m['ee_ideal']:.4f} rad/s")
    for c, x in res.items():
        print(f"\n=== {LABEL[c]} − 基线（配对 {x['joint_raw']['n'] if x['joint_raw'] else 0} 对；相对变化中位 [95% 置信区间]，Wilcoxon p）")
        for key, name in (("joint_raw", "关节偏差 相对原始指令"), ("joint_ideal", "关节偏差 相对理想渲染"),
                          ("ee_raw", "末端角速度偏差 相对原始指令"), ("ee_ideal", "末端角速度偏差 相对理想渲染"),
                          ("path", "护栏：关节路径长度"), ("speed", "护栏：关节速度 RMS")):
            v = x[key]
            print(f"  {name:22s}" + (f"{v['median']:+6.1f}% [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]  p={v['p']:.3g}" if v else "N/A"))
        for f, (_, _, label) in FAMILIES.items():
            print(f"  判定（{label}）：{x['verdict'][f]}")
        mm, p = summary["median"][c], summary["offline_prediction"][c]
        print(f"  实测 / 离线预测（中位）：关节 相对原始指令 {mm['joint_raw']:.2f} / {p['joint_raw']:.2f} mrad，相对理想渲染 {mm['joint_ideal']:.2f} / "
              f"{p['joint_ideal']:.2f}；末端 相对原始指令 {mm['ee_raw']:.4f} / {p['ee_raw']:.4f}，相对理想渲染 {mm['ee_ideal']:.4f} / {p['ee_ideal']:.4f} rad/s")
        print(f"  补正量 |a_r − a_π|：RMS {summary['correction_mrad'][c]['rms']:.1f} mrad、最大 {summary['correction_mrad'][c]['max']:.1f} mrad")
    if curves:
        plot(curves)
    print(f"\n→ {OUT}/summary.json" + ("、curves.png" if curves else ""))
    return 0


def main(argv=None) -> int:
    gc.disable()        # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seq", help="生成三个条件的序列（基线、提前 15 ms、提前 60 ms）")
    r = sub.add_parser("replay", help="交替回放（加 --dry-run 只做安全检查，不连机械臂）")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--one", help=argparse.SUPPRESS)        # 内部用：子进程跑一遍，如 "A,d15,3"
    sub.add_parser("analyze", help="相对原始策略指令的配对比较、判据 b、离线预测对照、曲线图")
    a = ap.parse_args(argv)
    if a.cmd == "seq":
        return cmd_seq()
    if a.cmd == "replay":
        return cmd_replay(a.dry_run, a.one)
    return cmd_analyze()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # 跳过解释器收尾（关 GC 的进程在退出阶段也会崩）；回放子进程因此同标定回放，不触发机器人析构里的无插值归位
