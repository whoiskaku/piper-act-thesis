#!/usr/bin/env python3
"""换向 / 近静止事件附近动态跟踪偏差的可重复性检验（方案与判据见 reports/PLAN_repeatability_reversal.md v2）。

同一条 ACT 实发指令重放 8 遍，看相对理想渲染的残差在事件附近能否以同样的方向、时刻、大小复现。
  seq      按事先定的规则从抓放正式基线选 3 条指令（A、B 开 TE，C 不开 TE），开头加 1 s 首帧保持；
           在名义参考（50 ms 帧长、从首帧出发）里固定事件（帧 f、细分步 s、关节 j），连同指令存进序列文件。
  replay   A B C 交替 8 轮，每遍一个子进程调 sim/replay_probe.py（同 IMU 标定回放），之间等人按回车；
           开始过的那一遍不再跑（不补跑）。
  analyze  各遍按自己的 t_send 换算事件时刻、取残差窗口；留一法 R²；三项分别只按 A、B 判定，C 只报告；
           窗口碰到疑似接触段（抓取时碰桌）的换向事件单列：用法 A 判定只用不碰的，用法 B 用全部，两种都给。

用法：
    python -u sim/repeat_reversal.py seq
    python -u sim/repeat_reversal.py replay --dry-run     # 三条序列过一遍回放管线的安全检查，不连机械臂
    python -u sim/repeat_reversal.py replay               # 真机；中断后重新运行即从下一遍接着跑
    python -u sim/repeat_reversal.py analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import cmd_tracking as T  # noqa: E402  理想渲染、标定后的理想 IMU、记录读取
from step4_gvnext import data as D  # noqa: E402  低通、采样率

OUT = HERE.parent / "outputs/repeat_reversal"
SEQS = {"A": "TE", "B": "TE", "C": "noTE"}       # 序列名 → 从哪一组基线选（同组按挑选顺序依次取）
REPO = {"TE": "eval_pickcube_base_TE", "noTE": "eval_pickcube_base_noTE"}
JUDGED = ("A", "B")              # 判定只看两条开 TE 序列；C 只报告
N_REP, MIN_OK = 8, 6
DT, HOLD = 0.05, 20              # 名义帧长；开头保持首帧指令的帧数（1 s，噪声底用）
MIN_PEAK = 0.1                   # rad/s：换向前同向运动的峰速、近静止前 1 s 内的峰速下限
STILL_S, STILL_DEG, PRE_S = 0.5, 0.1, 1.0
WIN = (-0.2, 0.6)                # s：事件窗口
GQ = np.arange(round(WIN[0] * D.FS_Q), round(WIN[1] * D.FS_Q) + 1) / D.FS_Q   # 编码器窗口网格（相对事件时刻）
GI = np.arange(round(WIN[0] * D.FS_I), round(WIN[1] * D.FS_I) + 1) / D.FS_I   # 陀螺窗口网格
SUPPORT, STOP = 0.6, 0.3
ITEMS = (("reversal", "enc", "换向·编码器"), ("reversal", "gyro", "换向·陀螺"), ("still", "gyro", "近静止·陀螺"))


# ── 事件与指标 ───────────────────────────────────────────────────────────
def find_events(cmd):
    """名义参考（帧长 DT、从首帧出发）里的事件 → [{kind, k, j, sign, t}]，k = 帧号 × SUB + 细分步，t = 名义时刻。
    换向：活动关节参考位置的转折点（T.turning_points，往回走 ≥ T.TURN_DEG 才确认），且从上一个转折点到这里的峰速 ≥ MIN_PEAK；
          k 取极值所在的细分步，sign = 之前的方向。参考在一帧内降到零又同向继续、到位时的数值小过冲（速度也会短暂变号）都不算。
    近静止：之后 STILL_S 内所有活动关节的参考位置变化范围 < STILL_DEG，且之前 PRE_S 内有关节峰速 ≥ MIN_PEAK；k 取起点。
    只留窗口完整落在序列内的事件（末尾再留一帧余量：各遍帧长有抖动）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    tr = T.PlanTrajectory(cmd, DT * np.arange(len(cmd)))
    Q, V, h = np.degrees(tr.Q), tr.V, DT / T.SUB
    n_still, n_pre = round(STILL_S / h), round(PRE_S / h)
    ev = []
    for j in T.ACTIVE:
        prev = 0
        for k, d in T.turning_points(Q[:, j], T.TURN_DEG):
            if np.abs(V[prev:k + 1, j]).max() >= MIN_PEAK:
                ev.append({"kind": "reversal", "k": int(k), "j": int(j), "sign": int(d)})
            prev = k
    W = sliding_window_view(Q[:, T.ACTIVE], n_still, axis=0)
    still = np.r_[(W.max(-1) - W.min(-1)).max(-1) < STILL_DEG, np.zeros(n_still - 1, bool)]
    for k in np.flatnonzero(still & ~np.r_[False, still[:-1]]):
        if k >= n_pre and np.abs(V[k - n_pre:k][:, T.ACTIVE]).max() >= MIN_PEAK:
            ev.append({"kind": "still", "k": int(k), "j": -1, "sign": 0})
    t_end = DT * (len(cmd) - 2)
    return sorted((e | {"t": float(tr.t[e["k"]])} for e in ev
                   if tr.t[e["k"]] + WIN[0] >= 0 and tr.t[e["k"]] + WIN[1] <= t_end), key=lambda e: e["k"])


def event_time(t_send, k):
    """事件（第 f 帧、第 s 个细分步）在一遍回放里的时刻：t_send[f] + (s+1)/SUB × (t_send[f+1] − t_send[f])，即本遍参考的细分步网格。"""
    f, s = divmod(int(k), T.SUB)
    return float(t_send[f] + (s + 1) / T.SUB * (t_send[f + 1] - t_send[f]))


def over_swing(t, q_act, e_q, q_turn, sign):
    """换向事件窗口（t 相对事件时刻）→ (多走, 回摆, 多走峰时刻)。多走 = max s·(q_实测 − q_参考(t_e))，负 = 没走到；
    回摆 = 多走峰之后 max(−s·e_q)，e_q 对的是**当时**的参考：跟着参考正常返回时 −s·e_q ≤ 0，不算；≤ 0 记 0。"""
    d = sign * (q_act - q_turn)
    p = int(np.argmax(d))
    return float(d[p]), max(0.0, float(np.max(-sign * e_q[p:]))), float(t[p])


def loo_r2(W):
    """同一事件各遍的残差波形 W（遍数 × …）：用其余各遍的平均预测每一遍，1 − Σ‖w_r − 其余平均‖² / Σ‖w_r‖²。
    各遍相同为 1；纯噪声约 −1/(n−1)。"""
    W = np.asarray(W, float).reshape(len(W), -1)
    others = (W.sum(0) - W) / (len(W) - 1)
    return float(1.0 - np.sum((W - others) ** 2) / np.sum(W ** 2))


def contact_flags(cmd7, ev, model, center):
    """每个事件的窗口是否碰到疑似接触段（同 cmd_tracking 口径：夹爪切换前后 CONTACT_S、指令末端在目标区 ZONE_M 内），
    在冻结序列的名义时间上按帧标。回放中目视确认：抓取下降到最低点时碰到桌面。"""
    tn = DT * np.arange(len(cmd7))
    marked = (T.near(tn, tn[T.grip_events(cmd7[:, 6])], T.CONTACT_S)
              | (np.linalg.norm(T.tcp(cmd7[:, :6], model) - center, axis=1) <= T.ZONE_M))
    return [bool(marked[(tn >= e["t"] + WIN[0]) & (tn <= e["t"] + WIN[1])].any()) for e in ev]


def turn_bounds(cmd7, ev):
    """每个换向事件在同一关节上前后相邻的转折点（细分步序号，没有则 None），包括没通过峰速筛选的转折；
    在冻结序列的名义参考上算，同 find_events。多走、回摆、多走峰时刻只在这两个转折之间取。"""
    tr = T.PlanTrajectory(cmd7[:, :6], DT * np.arange(len(cmd7)))
    Q = np.degrees(tr.Q)
    tps = {j: [k for k, _ in T.turning_points(Q[:, j], T.TURN_DEG)] for j in T.ACTIVE}
    out = []
    for e in ev:
        if e["kind"] != "reversal":
            out.append((None, None))
            continue
        ks = tps[e["j"]]
        n = ks.index(e["k"])                     # 换向事件本身就是其中一个转折点（找不到会报错）
        out.append((ks[n - 1] if n > 0 else None, ks[n + 1] if n + 1 < len(ks) else None))
    return out


def verdict(vals):
    """一项的判定。vals = {序列名: 该项 R²_LOO 中位数，None = 完整回放不足 MIN_OK 遍或该项没有事件}；只看 A、B。"""
    v = [vals.get(s) for s in JUDGED]
    if any(x is None for x in v):
        return "无法判定"
    if all(x >= SUPPORT for x in v):
        return "支持针对这一项继续尝试补正"
    if all(x < STOP for x in v):
        return "这一项暂不推进"
    return "无法判定"


# ── seq ──────────────────────────────────────────────────────────────────
def episode_cmd(repo, i):
    """基线一条的实发指令（7 列 float32，即回放管线发出的值），开头加 HOLD 帧首帧保持。"""
    import pandas as pd
    a = np.stack(pd.read_parquet(T.LR / repo / f"data/chunk-000/episode_{i:06d}.parquet")["action"].to_numpy())
    return np.concatenate([np.repeat(a[:1], HOLD, 0), a]).astype(np.float32)


def select():
    """事先定的挑选规则：同一组里按陀螺 <15 Hz 残差离组中位数由近到远（距离相同取序号小的），依次取有近静止事件的条。
    → {序列名: (组, 集号, cmd7, 事件)}。"""
    import pandas as pd
    df = pd.read_csv(T.OUT / "metrics.csv")
    out = {}
    for cond in ("TE", "noTE"):
        names = [n for n, c in SEQS.items() if c == cond]
        g = df[(df.task == "pickcube") & (df.cond == cond)]
        g = g.assign(d=(g.imu_gyro_lt15_all - g.imu_gyro_lt15_all.median()).abs().round(9))   # 去掉浮点尾数再比
        for i in g.sort_values(["d", "episode"]).episode:
            if not names:
                break
            cmd7 = episode_cmd(REPO[cond], int(i))
            ev = find_events(cmd7[:, :6].astype(float))
            if any(e["kind"] == "still" for e in ev):
                out[names.pop(0)] = (cond, int(i), cmd7, ev)
    return out


def cmd_seq():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (cond, i, cmd7, ev) in select().items():
        src = {"repo": REPO[cond], "episode": i, "hold_frames": HOLD, "plan": "reports/PLAN_repeatability_reversal.md v2",
               "cmd_hash": hashlib.sha256(np.ascontiguousarray(cmd7).tobytes()).hexdigest()[:16]}   # 同 replay_probe 的算法
        np.savez(OUT / f"seq_{name}.npz", cmd7=cmd7, da=np.zeros((len(cmd7), 6), np.float32),
                 events=json.dumps(ev), source=json.dumps(src, ensure_ascii=False))
        still = [round(e["t"], 2) for e in ev if e["kind"] == "still"]
        print(f"{name}：{REPO[cond]} 第 {i} 条 · {len(cmd7)} 帧（含开头保持 {HOLD}）· 换向 "
              f"{sum(e['kind'] == 'reversal' for e in ev)} · 近静止 {len(still)} @ {still} s · 指令哈希 {src['cmd_hash']}")
    return 0


# ── replay ───────────────────────────────────────────────────────────────
def attempted(name, r):
    """这一遍是否开始过：留下了回放记录，或预移动后的静置记录（臂已动过）。连接失败不算。"""
    return (OUT / f"replay_{name}_r{r}.json").exists() or any(OUT.glob(f"settle*_{name}_r{r}_*.json"))


def cmd_replay(dry_run, one=None):
    import replay_probe as RP
    if one:                          # 子进程：跑一遍，如 "A3"
        return RP.run(f"r{one[1:]}", False, seqfile=str(OUT / f"seq_{one[0]}.npz"), tag=f"{one[0]}_", outdir=OUT)
    if dry_run:
        return max([RP.run("r0", True, seqfile=str(OUT / f"seq_{s}.npz"), tag=f"{s}_", outdir=OUT) for s in SEQS])
    sha = hashlib.sha256((HERE / "replay_probe.py").read_bytes()).hexdigest()
    plan = [(s, r) for r in range(1, N_REP + 1) for s in SEQS]           # A B C 交替 8 轮
    for k, (s, r) in enumerate(plan, 1):
        if attempted(s, r):
            continue
        if input(f"\n[{k}/{len(plan)}] 序列 {s} 第 {r} 遍：桌面已清空、可以运动就按回车（输入 q 退出）").strip().lower() == "q":
            return 0
        utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        p = subprocess.Popen([sys.executable, "-u", __file__, "replay", "--one", f"{s}{r}"])
        prev = signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C 只交给子进程（管线先落盘再分段归位），这里继续等它结束
        try:
            code = p.wait()
        finally:
            signal.signal(signal.SIGINT, prev)
        with open(OUT / "replay_log.jsonl", "a") as f:
            f.write(json.dumps({"k": k, "seq": s, "rep": r, "utc": utc, "code": code, "replay_probe_sha256": sha}) + "\n")
        if code:
            print(f"  ⚠ 第 {k} 次返回码 {code}：这一遍不补跑。检查机械臂后再继续。")
    print(f"{len(plan)} 遍都已开始过。下一步：python -u sim/repeat_reversal.py analyze")
    return 0


# ── analyze ──────────────────────────────────────────────────────────────
def load_reps(name):
    """→ (cmd7, 事件, 来源, [(遍号, from_record 结果或 None, 墙钟起点)])。完整 = 管线没有提前终止、指令哈希和帧数与序列一致、有编码器记录。"""
    z = np.load(OUT / f"seq_{name}.npz")
    ev, src = json.loads(str(z["events"])), json.loads(str(z["source"]))
    reps = []
    for r in range(1, N_REP + 1):
        p = OUT / f"replay_{name}_r{r}.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text())
        ok = (h.get("stopped") is None and h.get("cmd_hash") == src["cmd_hash"] and len(h["t_send"]) == len(z["cmd7"])
              and len(h.get("can_q200", [])) > 0)
        reps.append((r, T.from_record(h, np.asarray(h["cmd"], float)) if ok else None,
                     h["t_send"][0] + h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"] if ok else None))
    return z["cmd7"].astype(float), ev, src, reps


def residuals(ep, calib, model, sid):
    """一遍 → 参考、编码器残差 e_q（tq 上，rad，不滤波）、陀螺残差（ti 上，<15 Hz，3 轴）。参考同 cmd_tracking：按本遍 t_send、从实测起始位置出发。"""
    traj = T.PlanTrajectory(ep["cmd"], ep["t_send"], q0=ep["q"][0])
    eq = ep["q"] - traj(ep["tq"])[:, :6]
    eg = D.lp(ep["imu"][:, 3:] - T.ideal_imu(traj, ep["ti"] - calib["delay_s"], calib, model, sid)[1], T.LP_VEL, D.FS_I)
    return traj, eq, eg


def windows(ep, traj, eq, eg, ev):
    """一遍里每个事件的窗口量：陀螺残差波形（GI × 3）；换向另有事件关节的编码器残差波形（GQ，°）与多走、回摆、多走峰时刻；
    近静止另有余振（起点后 STILL_S 内陀螺残差三轴模的 RMS）。评分波形用整个 WIN；多走、回摆、多走峰时刻只在
    e["turns"]（同关节前后相邻的转折，见 turn_bounds）之间取——跨过去以后参考自己的下一段运动可能超过旧转折点。"""
    out = []
    for e in ev:
        te = event_time(ep["t_send"], e["k"])
        w = {"gyro": np.stack([np.interp(te + GI, ep["ti"], eg[:, a]) for a in range(3)], 1)}
        if e["kind"] == "reversal":
            j = e["j"]
            w["enc"] = np.degrees(np.interp(te + GQ, ep["tq"], eq[:, j]))
            qa = np.degrees(np.interp(te + GQ, ep["tq"], ep["q"][:, j]))
            kp, kn = e["turns"]                  # 换算到本遍时刻：本遍参考的细分步网格 traj.t（同 event_time）
            m = (GQ >= (traj.t[kp] - te if kp is not None else -np.inf)) & (GQ <= (traj.t[kn] - te if kn is not None else np.inf))
            w["over"], w["swing"], w["t_peak"] = over_swing(GQ[m], qa[m], w["enc"][m], np.degrees(traj(te)[0, j]), e["sign"])
        else:
            w["ring"] = T.rms(np.linalg.norm(w["gyro"][(GI >= 0) & (GI <= STILL_S)], axis=1))
        out.append(w)
    return out


def rep_summary(ep, eq, eg):
    """一遍的噪声底（开头保持段 0.4 s 起、到第一条运动指令前 0.1 s，去均值）与保持段之后的整体残差（看漂移）。"""
    t0, t1, ta = ep["t_send"][0] + 0.4, ep["t_send"][HOLD] - 0.1, ep["t_send"][HOLD]
    mq, mi = (ep["tq"] >= t0) & (ep["tq"] < t1), (ep["ti"] >= t0) & (ep["ti"] < t1)
    deg = 180 / np.pi
    return {"noise_gyro": T.rms(np.linalg.norm(eg[mi] - eg[mi].mean(0), axis=1)),
            "noise_enc_deg": T.rms((eq[mq] - eq[mq].mean(0))[:, T.ACTIVE]) * deg,
            "gyro_rms": T.rms(np.linalg.norm(eg[ep["ti"] >= ta], axis=1)),
            "enc_rms_deg": T.rms(eq[ep["tq"] >= ta][:, T.ACTIVE]) * deg}


def analyze_sequence(ev, eps, calib, model, sid):
    """一条序列的完整各遍 → ({每个事件的 R²_LOO 与标量统计、各项的 R²_LOO 中位数、每遍汇总}, 各遍窗口（画图用）)。"""
    waves, reps = [], []
    for ep in eps:
        traj, eq, eg = residuals(ep, calib, model, sid)
        waves.append(windows(ep, traj, eq, eg, ev))
        reps.append(rep_summary(ep, eq, eg))
    rows = []
    for i, e in enumerate(ev):
        ws = [w[i] for w in waves]
        row = dict(e)
        for sig in ("enc", "gyro"):
            if sig in ws[0]:
                X = np.array([w[sig] for w in ws])
                row[f"r2_{sig}"] = loo_r2(X)
                row[f"r2_{sig}_demeaned"] = loo_r2(X - X.mean(1, keepdims=True))     # 各遍窗口各自去均值：只看动态部分
                m = X.mean(0)
                row[f"size_{sig}"] = [float(np.sqrt(np.sum(m ** 2) / len(m))),                # 可复现部分：各遍平均波形的 RMS
                                      float(np.sqrt(np.sum((X - m) ** 2) / (len(X) * len(m))))]   # 不可复现部分：各遍相对平均的 RMS
        for key in ("over", "swing", "t_peak", "ring"):
            if key in ws[0]:
                x = np.array([w[key] for w in ws])
                row[key] = [float(x.mean()), float(x.std(ddof=1))]
        if "over" in ws[0]:
            pos = float(np.mean([w["over"] > 0 for w in ws]))
            row["over_sign_agree"] = max(pos, 1 - pos)
        rows.append(row)
    items = {}
    for kind, sig, label in ITEMS:
        sel = [r for r in rows if r["kind"] == kind]
        free = [r for r in sel if not r["contact"]]

        def med(rs, key=f"r2_{sig}"):
            return float(np.median([r[key] for r in rs])) if rs else None

        items[label] = {"n_free": len(free), "n_contact": len(sel) - len(free),
                        "median_r2": med(free), "median_r2_all": med(sel),          # 用法 A / 用法 B
                        "median_r2_contact": med([r for r in sel if r["contact"]]),
                        "median_r2_demeaned": med(free, f"r2_{sig}_demeaned"),
                        "median_size": [float(np.median([r[f"size_{sig}"][k] for r in free])) for k in (0, 1)] if free else None}
    return {"events": rows, "items": items, "reps": reps}, waves


def report(res, verdicts):
    print("完整回放：" + " · ".join(f"{s} {len(res[s]['complete'])}/{N_REP}" for s in SEQS)
          + f"（A 或 B 少于 {MIN_OK} 遍 → 三项都无法判定）")

    def f(x):
        return f"{x:8.2f}" if x is not None else f"{'N/A':>8s}"

    for title, vk, dk, ck in (("用法 A（推荐）：两项换向只用窗口不碰疑似接触段的事件", "values", "verdict", "median_r2"),
                              ("用法 B：全部事件", "values_all", "verdict_all", "median_r2_all")):
        print(f"\n{title}\n{'项目':8s}{'A':>8s}{'B':>8s}   {'判定（只看 A、B）':22s}辅助 C")
        for _, _, label in ITEMS:
            v = verdicts[label]
            c = res["C"].get("items", {}).get(label, {}).get(ck)
            print(f"{label:8s}{f(v[vk]['A'])}{f(v[vk]['B'])}   {v[dk]:22s}{f(c)}")
    print("（值 = 该项各事件留一法 R² 的中位数；通过只支持尝试 ILC，不证明能修好；各项互不否定）\n\n只报告：")
    for s in SEQS:
        if "events" not in res[s]:
            continue
        it = res[s]["items"]
        enc, gyr = (it[label] for _, _, label in ITEMS[:2])
        rv = [e for e in res[s]["events"] if e["kind"] == "reversal" and not e["contact"]]
        dm = " / ".join(f"{label} {x['median_r2_demeaned']:.2f}" for label, x in it.items() if x["median_r2_demeaned"] is not None)
        print(f"  {s}（{res[s]['source']['repo']} 第 {res[s]['source']['episode']} 条，完整 {len(res[s]['complete'])} 遍）"
              f"换向 自由 {enc['n_free']} + 接触 {enc['n_contact']} 个；接触事件 R² 中位 编码器 {f(enc['median_r2_contact']).strip()}"
              f" / 陀螺 {f(gyr['median_r2_contact']).strip()}；去均值后 R² 中位（用法 A）：{dm}")
        print("    可复现 / 不可复现部分的 RMS 中位（用法 A；编码器 °，陀螺 rad/s）："
              + " · ".join(f"{label} {x['median_size'][0]:.3f} / {x['median_size'][1]:.3f}" for label, x in it.items()
                           if x["median_size"] is not None))
        if rv:
            def med(key, k):
                return float(np.median([e[key][k] for e in rv]))
            print(f"    自由换向：多走 遍间均值的中位 {med('over', 0):+.3f}°、遍间标准差的中位 {med('over', 1):.3f}°、"
                  f"符号一致率中位 {np.median([e['over_sign_agree'] for e in rv]):.2f}；回摆 {med('swing', 0):.3f}° / "
                  f"{med('swing', 1):.3f}°；多走峰时刻遍间标准差的中位 {med('t_peak', 1) * 1e3:.0f} ms")
        for e in res[s]["events"]:
            if e["kind"] == "still":
                print(f"    近静止 @{e['t']:.2f} s：余振 {e['ring'][0]:.4f} ± {e['ring'][1]:.4f} rad/s（遍间均值 ± 标准差）")
    reps = sorted(({"seq": s, **x} for s in SEQS for x in res[s].get("reps", [])), key=lambda x: x["wall"])
    if reps:
        print("\n  每遍按实际时间：分钟 · 序列遍 · 整体陀螺残差 rad/s · 整体编码器残差 ° · 噪声底 陀螺 rad/s / 编码器 °")
        for x in reps:
            print(f"    {(x['wall'] - reps[0]['wall']) / 60:5.1f} · {x['seq']}{x['rep']} · {x['gyro_rms']:.4f} · "
                  f"{x['enc_rms_deg']:.3f} · {x['noise_gyro']:.4f} / {x['noise_enc_deg']:.4f}")


def plot(res, waves):
    """每条序列一行；列 = 换向·编码器（中位事件、R² 最低的事件）、换向·陀螺（同）、近静止·陀螺（R² 最低的事件）。
    细线 = 各遍，粗线 = 各遍平均，虚线 = 事件时刻；陀螺只画残差能量最大的轴。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    cols = (("reversal", "enc", 0.5, "换向·编码器残差（°）中位事件"), ("reversal", "enc", 0.0, "换向·编码器 R² 最低的事件"),
            ("reversal", "gyro", 0.5, "换向·陀螺残差（rad/s）中位事件"), ("reversal", "gyro", 0.0, "换向·陀螺 R² 最低的事件"),
            ("still", "gyro", 0.0, "近静止·陀螺残差（rad/s）"))
    fig, ax = plt.subplots(len(waves), len(cols), figsize=(18, 3.0 * len(waves)), squeeze=False)
    for r, name in enumerate(waves):
        ev = res[name]["events"]
        for c, (kind, sig, q, title) in enumerate(cols):
            A = ax[r, c]
            idx = sorted((i for i, e in enumerate(ev) if e["kind"] == kind and not e["contact"]),
                         key=lambda i: ev[i][f"r2_{sig}"])
            if r == 0:
                A.set_title(title + "\n", loc="left", color=T.INK)   # 列名占上一行，下一行留给事件标签
            if not idx:
                A.set_axis_off()
                continue
            i = idx[int(q * (len(idx) - 1))]
            X = np.array([w[i][sig] for w in waves[name]])
            label = f"j{ev[i]['j'] + 1} " if kind == "reversal" else ""
            if sig == "gyro":
                a = int(np.argmax(np.mean(X ** 2, axis=(0, 1))))
                X, label = X[:, :, a], label + f"{'xyz'[a]} 轴 "
            t = GQ if sig == "enc" else GI
            for x in X:
                A.plot(t, x, color=T.BLUE, lw=0.7, alpha=0.5)
            A.plot(t, X.mean(0), color=T.INK, lw=1.5)
            A.axvline(0, color=T.MUTED, ls="--", lw=0.8)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
            A.set_title(f"{label}@{ev[i]['t']:.1f} s · R² {ev[i][f'r2_{sig}']:.2f}", loc="right", color=T.MUTED, fontsize=8)
        src = res[name]["source"]
        ax[r, 0].set_ylabel(f"{name}：{'不开' if src['repo'].endswith('noTE') else '开'} TE 第 {src['episode']} 条")
    for c in range(len(cols)):
        ax[-1, c].set_xlabel("相对事件时刻（s）")
    fig.suptitle("同一指令重放各遍的残差（细线 = 各遍，粗线 = 平均；只取不碰疑似接触段的事件；"
                 "参考 = a_max 5 rad/s² 参考假设、按各遍 t_send）", x=0.01, ha="left", color=T.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "overlay.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def cmd_analyze():
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    calib = json.loads((HERE.parent / "outputs/imu_calib/imu_calib.json").read_text())
    center = T.target_center("pickcube_raw", model)         # 目标区中心（示教放下点中位），同 cmd_tracking
    res, waves = {}, {}
    for name in SEQS:
        cmd7, ev, src, reps = load_reps(name)
        ev = [e | {"contact": c, "turns": b}
              for e, c, b in zip(ev, contact_flags(cmd7, ev, model, center), turn_bounds(cmd7, ev))]
        good = [(r, ep, wall) for r, ep, wall in reps if ep is not None]
        res[name] = {"source": src, "complete": [r for r, _, _ in good], "files": [r for r, _, _ in reps]}
        if len(good) >= 2:
            out, waves[name] = analyze_sequence(ev, [ep for _, ep, _ in good], calib, model, sid)
            for (r, _, wall), x in zip(good, out["reps"]):
                x.update(rep=r, wall=wall)
            res[name].update(out)
    verdicts = {}
    for _, _, label in ITEMS:
        ok = {s: len(res[s]["complete"]) >= MIN_OK for s in SEQS}
        v = {s: res[s]["items"][label]["median_r2"] if ok[s] else None for s in SEQS}          # 用法 A
        va = {s: res[s]["items"][label]["median_r2_all"] if ok[s] else None for s in SEQS}     # 用法 B
        verdicts[label] = {"values": v, "verdict": verdict(v), "values_all": va, "verdict_all": verdict(va)}
    (OUT / "analysis.json").write_text(json.dumps({"plan": "reports/PLAN_repeatability_reversal.md v2",
                                                   "verdicts": verdicts, "sequences": res}, ensure_ascii=False, indent=1))
    report(res, verdicts)
    if waves:
        plot(res, waves)
    print(f"\n→ {OUT}/analysis.json" + ("、overlay.png" if waves else ""))
    return 0


def main(argv=None) -> int:
    gc.disable()        # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seq", help="按事先定的规则选 3 条指令、固定事件、存序列文件")
    r = sub.add_parser("replay", help="A B C 交替 8 轮真机回放（加 --dry-run 只做安全检查，不连机械臂）")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--one", help=argparse.SUPPRESS)        # 内部用：子进程跑一遍
    sub.add_parser("analyze", help="留一法 R²、分项判定、叠图")
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
