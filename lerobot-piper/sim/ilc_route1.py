#!/usr/bin/env python3
"""路线 1：不训练前向动态 G 的迭代学习补正（方案 reports/PLAN_route_compare.md v0.6 §2、§4、§5）。

  run      按事先排定的顺序跑完一个会话：Shift 基准 → 共同关节阶段 → 两支（只关节 / 加 IMU）→ 最终评价，共 68 遍（标定另做）。
           每遍之后按 §2.5 判定，并由已接受补正的回放误差生成下一个候选；中断后重新运行，从已有回放记录重建状态接着跑。
  analyze  最终评价：同序列同轮配对、按序列汇总、分层自助法置信区间、四组判定；诊断（纯时间平移分量、经验滞后、迭代曲线）。

用法（lerobot 环境；会话开始前先做当天的 IMU 标定：sim/imu_calib.py replay、fit）：
    python -u sim/ilc_route1.py run --dry-run     # 不连机械臂：只对下一遍要发的序列做回放管线的安全检查
    python -u sim/ilc_route1.py run               # 真机；中断后重新运行即接着跑
    python -u sim/ilc_route1.py analyze
"""
from __future__ import annotations

import argparse
import datetime as dt
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
import cmd_tracking as T  # noqa: E402  读取、接触目标区、配色
import repeat_reversal as RR  # noqa: E402  接触标记、回放子进程、「开始过」检查
import shift_replay as S  # noqa: E402  零阶保持、线性插值渲染、指标、带保护的回放
from step4_gvnext import data as D  # noqa: E402  低通、采样率

OUT = HERE.parent / "outputs/ilc_route1"
SR = HERE.parent / "outputs/shift_replay"          # 原始指令与提前 60 ms 的序列
CALIB = HERE.parent / "outputs/imu_calib/imu_calib.json"
SEQS = ("A", "B")
TAU, GAMMA0, LAM, CLIP, GUARD, QC_MAX = 0.060, 0.5, 0.05, 0.015, 1.05, 2.0   # s（初始假设）、步长、阻尼、rad、护栏、°
F_Q, BAND_W, TAPER = 5.0, (0.5, 5.0), 5            # Hz：Q 与关节项低通；IMU 项带通；窗的过渡帧数（0.25 s）
BUDGET = {"b": 2, "c": 4, "j": 6, "i": 6}           # 每条序列的回放上限（§4）
N_EVAL, EVAL = 4, ("shift", "bp", "j", "i")
MARGIN = max(S.CONDS.values()) + 0.05              # 非接触段末尾留白，同 Shift 回放分析
HARD_SLACK = 0.035                                 # rad：回放管线的硬界 = 模型限位 ± 2°（同 replay_probe.safety_checks 的 mlo2 / mhi2）
BOUND_MARGIN = np.radians(0.2)                     # Shift + Δ 离硬界至少 0.2°


# ── 补正的更新（§2.2–2.4）────────────────────────────────────────────────
def frozen(marked, sat, hold=RR.HOLD):
    """不更新的帧（帧 × 5）：保持段、最后一帧、疑似接触帧（所有关节），加上逐关节的限位饱和帧 sat。"""
    z = np.repeat(np.asarray(marked, bool)[:, None], sat.shape[1], 1) | sat
    z[:hold] = True
    z[-1] = True
    return z


def window(zero, taper=TAPER):
    """新增量的窗 W（帧 × 关节）：zero 为 True 的帧为 0，向外 taper 帧余弦过渡到 1。"""
    zero = np.asarray(zero, bool)
    k = np.arange(len(zero))
    d = np.stack([np.abs(k[:, None] - np.flatnonzero(z)[None, :]).min(1) for z in zero.T], 1)
    return np.where(d >= taper, 1.0, 0.5 - 0.5 * np.cos(np.pi * d / taper))


def saturated(a6, s6, lo, hi):
    """限位饱和帧（帧 × 5）：原始帧越界、实际发令的 Shift 帧越界、前瞻误差对应的原始帧（t_send[k] + τ 时生效的那帧）越界，取并集。"""
    out_a, out_s = (a6 < lo) | (a6 > hi), (s6 < lo) | (s6 > hi)
    look = np.minimum(np.arange(len(a6)) + int(TAU // RR.DT), len(a6) - 1)
    return (out_a | out_s | out_a[look])[:, T.ACTIVE]


def qfilter(x):
    """Q：20 Hz 帧序列上的零相位 5 Hz 低通。"""
    return D.lp(x, F_Q, 1.0 / RR.DT)


def step(acc, ebar, gamma, W, lo=-CLIP, hi=CLIP):
    """Δ_next = clip(Δ_acc + W ⊙ Q[γ·ē], lo, hi)：窗只乘新增量，零更新保持 Δ_acc 不变；
    lo、hi 已含 ±15 mrad 与「Shift + Δ 离管线硬界至少 0.2°」两个限制（§2.4）。"""
    return np.clip(acc + W * qfilter(gamma * ebar), lo, hi)


def frame_of(ts, t):
    """时刻 t 生效的帧号（t_send[k] ≤ t 的最后一帧）。"""
    return np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)


def sample(t, x, at):
    """x(t) 在 at 处的线性插值；超出记录范围置 0。"""
    out = np.stack([np.interp(at, t, x[:, j]) for j in range(x.shape[1])], 1)
    out[(at < t[0]) | (at > t[-1])] = 0.0
    return out


def joint_err(ep, a6, marked):
    """ē_q（帧 × 5，rad）：e_q = 原始指令（零阶保持）− 编码器，疑似接触帧置 0 → 零相位 5 Hz 低通 → 取 t_send[k] + τ。"""
    ts, tq = ep["t_send"], ep["tq"]
    e = (S.zoh(ts, a6, tq) - ep["q"])[:, T.ACTIVE]
    e[marked[frame_of(ts, tq)]] = 0.0
    return sample(tq, D.lp(e, F_Q, D.FS_Q), ts + TAU)


def jacobians(model, sid, q6):
    """各位形下安装点角速度对 5 个活动关节的雅可比（3×5，安装点坐标）。"""
    import mujoco
    from ideal_imu_kinematic import full_qpos
    d, jacr = mujoco.MjData(model), np.zeros((3, model.nv))
    out = np.empty((len(q6), 3, len(T.ACTIVE)))
    for i, q in enumerate(q6):
        d.qpos[:] = full_qpos(np.r_[q, 0.0])
        mujoco.mj_fwdPosition(model, d)
        mujoco.mj_jacSite(model, d, None, jacr, sid)
        out[i] = (d.site_xmat[sid].reshape(3, 3).T @ jacr)[:, T.ACTIVE]
    return out


def ang_to_joint(t, eps, J):
    """角速度误差 → 关节位置修改：δq̇ = J⁺ε（J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹，rad/s）→ 梯形积分 → 零相位高通 0.5 Hz 去积分漂移（rad）。"""
    Jt = np.transpose(J, (0, 2, 1))
    dqd = np.einsum("nij,nj->ni", Jt @ np.linalg.inv(J @ Jt + LAM ** 2 * np.eye(3)), eps)
    dq = np.vstack([np.zeros((1, dqd.shape[1])), np.cumsum(0.5 * (dqd[1:] + dqd[:-1]) * np.diff(t)[:, None], 0)])
    return dq - D.lp(dq, BAND_W[0], D.FS_I)


def imu_err(ep, a6, marked, calib, model, sid):
    """ē_ω（帧 × 5，rad）：e_ω = 目标（原始指令运动学渲染）− 实测陀螺，疑似接触帧置 0 → 带通 0.5–5 Hz、转到安装点坐标
    → 原始指令位形下的雅可比逆 → 积分、高通 → 取 t_send[k] + τ + d（IMU 记下的是 d 秒前的运动）。"""
    ts, ti, d = ep["t_send"], ep["ti"], calib["delay_s"]
    e = S.raw_gyro(ep, a6, calib, model, sid) - ep["imu"][:, 3:]
    e[marked[frame_of(ts, ti - d)]] = 0.0
    eps = (D.lp(e, BAND_W[1], D.FS_I) - D.lp(e, BAND_W[0], D.FS_I)) @ np.asarray(calib["R"])   # 逐行 Rᵀ·e
    J = jacobians(model, sid, S.LinCmd(ts, a6)(ti - d)[:, :6])
    return sample(ti, ang_to_joint(ti, eps, J), ts + TAU + d)


# ── 一支的迭代（§2.4–2.6）────────────────────────────────────────────────
def mean_res(rs):
    """几遍回放结果 [(指标, (ē_q, ē_ω))] 的平均：指标逐项平均，误差逐帧平均。"""
    return ({k: float(np.mean([r[0][k] for r in rs])) for k in rs[0][0]},
            tuple(np.mean([r[1][i] for r in rs], 0) for i in range(2)))


class Learner:
    """一条序列上的一支：c 共同关节阶段、j 只关节支、i 加 IMU 支。acc / err / met = 已接受的补正及其回放误差、指标
    （经确认的取两遍平均）；base = 护栏与归一化的基准（c：两遍 Shift 基准；j、i：分叉点）。"""

    def __init__(self, kind, budget, acc, err, met, box, rids):
        self.kind, self.left, self.box = kind, budget, box   # box = (W, lo, hi)：窗与 Δ 的逐帧上下限
        self.base = (met["joint_raw"], met["ee_raw"])
        self.acc, self.err, self.met = acc, err, met
        self.gamma, self.c, self.noimp, self.probing = GAMMA0, 1.0, 0, False
        self.cands, self.hist = [], [(rids, acc)]           # hist：每次接受的（回放编号, Δ），供经验滞后

    def J(self, m):
        return m["joint_raw"] / self.base[0] + m["ee_raw"] / self.base[1]

    def ok(self, m):
        """护栏：关节项都看；末端项只在加 IMU 支看。"""
        return m["joint_raw"] <= GUARD * self.base[0] and (self.kind != "i" or m["ee_raw"] <= GUARD * self.base[1])

    def cand(self, sign, label):
        eq, ew = self.err
        e = eq if self.kind == "c" else 0.5 * eq if self.kind == "j" else 0.5 * eq + sign * self.c * 0.5 * ew
        return {"label": label, "delta": step(self.acc, e, self.gamma, *self.box), "res": [], "rids": [], "confirm": False}

    def next(self):
        """下一遍要回放的候选：先跑这一步里还没跑的、再补确认；这一步完了就开新的一步。
        试探阶段只剩 1 遍时，复测已接受的补正，不再更新。"""
        for c in self.cands:
            if not c["res"] or (c["confirm"] and len(c["res"]) == 1):
                return c
        if self.probing:
            self.cands = ([self.cand(1, "试探+"), self.cand(-1, "试探−")] if self.left >= 2
                          else [{"label": "复测", "delta": self.acc, "res": [], "rids": [], "confirm": False}])
        else:
            self.cands = [self.cand(1, "候选")]
        return self.cands[0]

    def void(self):
        """无效的一遍（提前终止等）：占预算，候选不变。"""
        self.left -= 1

    def add(self, cand, met, err):
        """一遍回放的结果；返回给人看的一句。"""
        self.left -= 1
        cand["res"].append((met, err))
        unrun = sum(1 for c in self.cands if not c["res"])
        if len(cand["res"]) == 1 and not self.ok(met) and self.left - unrun >= 1:
            cand["confirm"] = True
            return "触发护栏 → 下一遍确认"
        if any(not c["res"] or (c["confirm"] and len(c["res"]) == 1) for c in self.cands):
            return "等这一步的另一个候选"
        return self.decide()

    def decide(self):
        rs = [(c, *mean_res(c["res"])) for c in self.cands]
        self.cands = []
        if rs[0][0]["label"] == "复测":
            return "复测已接受的补正（不更新）"
        if len(rs) == 1:
            c, m, e = rs[0]
            if not self.ok(m):
                self.gamma /= 2
                self.noimp += 1
                msg = f"退回，γ → {self.gamma:g}"
            else:
                better = self.J(m) < self.J(self.met)
                self.acc, self.err, self.met = c["delta"], e, m
                self.hist.append((c["rids"], c["delta"]))
                self.noimp = 0 if better else self.noimp + 1
                msg = "接受" + ("" if better else "（目标值没有下降）")
        else:
            good = [(self.J(m), i, c, m, e) for i, (c, m, e) in enumerate(rs) if self.ok(m) and self.J(m) < self.J(self.met)]
            if good:
                _, _, c, m, e = min(good)
                self.acc, self.err, self.met = c["delta"], e, m
                self.hist.append((c["rids"], c["delta"]))
                msg = f"接受{c['label']}"
            else:
                self.c /= 2
                msg = f"两个试探都没有改善，c → {self.c:g}"
        if self.kind == "i" and not self.probing and self.noimp >= 2:
            self.probing = True
            msg += "；连续两次没有改善 → 改为试探"
        return msg


# ── 会话 ─────────────────────────────────────────────────────────────────
def schedule():
    """[(序列, 环节, 第几遍)]，共 68 遍：Shift 基准 2、共同关节阶段 4、两支各 6（交错，各占一半先手）、最终评价 16（拉丁方）。"""
    out = [(s, "b", n) for n in range(1, BUDGET["b"] + 1) for s in SEQS]
    out += [(s, "c", n) for n in range(1, BUDGET["c"] + 1) for s in SEQS]
    for n in range(1, BUDGET["j"] + 1):
        first, second = ("j", "i") if n % 2 else ("i", "j")
        out += [("A", first, n), ("B", second, n), ("A", second, n), ("B", first, n)]
    out += [(s, "e_" + EVAL[(i + r) % len(EVAL)], r + 1) for r in range(N_EVAL) for s in SEQS for i in range(len(EVAL))]
    return out


def bounds(s6, lo_hard, hi_hard):
    """Δ 的逐帧上下限（帧 × 5）：±15 mrad，且 Shift + Δ 离管线硬界至少 0.2°。"""
    return (np.maximum(-CLIP, (lo_hard + BOUND_MARGIN - s6)[:, T.ACTIVE]),
            np.minimum(CLIP, (hi_hard - BOUND_MARGIN - s6)[:, T.ACTIVE]))


class Ctx:
    """会话共用：模型、标定；每条序列的原始指令 a7、提前 60 ms 的指令 s7、疑似接触帧（两份指令取并集）、
    限位饱和帧、新增量的窗 W 与 Δ 的上下限 lo、hi。"""

    def __init__(self):
        import mujoco
        from ideal_imu_kinematic import MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
        self.calib = json.loads(CALIB.read_text())
        center = T.target_center("pickcube_raw", self.model)
        lo_m, hi_m = self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1]
        self.seq = {}
        for s in SEQS:
            a7 = np.load(SR / f"seq_{s}_base.npz")["cmd7"].astype(float)
            s7 = np.load(SR / f"seq_{s}_d60.npz")["cmd7"]
            marked = RR.contact_frames(a7, self.model, center) | RR.contact_frames(s7.astype(float), self.model, center)
            sat = saturated(a7[:, :6], s7[:, :6].astype(float), lo_m, hi_m)
            lo, hi = bounds(s7[:, :6].astype(float), lo_m - HARD_SLACK, hi_m + HARD_SLACK)
            if np.any(lo > 0) or np.any(hi < 0):
                raise SystemExit(f"⛔ 序列 {s} 的 Shift 指令离管线硬界不足 0.2°：Δ = 0 都不满足限制")
            self.seq[s] = {"a7": a7, "s7": s7, "marked": marked, "sat": sat, "box": (window(frozen(marked, sat)), lo, hi)}


def write_seq(rid, s7, delta, label):
    """发出的指令 = 提前 60 ms 的指令 + Δ（5 个活动关节，float32）。这一遍已开始过时，核对重建出的是同一份（不同就停）。
    单帧跳变按回放管线同一公式检查（Δ 写在序列里，管线自己的这项检查拿序列和它自己比，起不到作用）：以 Shift 为裸序列。"""
    cmd7 = s7.copy()
    cmd7[:, T.ACTIVE] += delta.astype(np.float32)
    jump, base = (np.abs(np.diff(x[:, :6].astype(float), axis=0)).max() for x in (cmd7, s7))
    if not jump < base * 1.5 + 0.006:
        raise SystemExit(f"⛔ {rid}：单帧跳变 {jump * 1e3:.1f} mrad 超过 Shift 的 1.5 倍 + 6 mrad（{base * 1e3:.1f} mrad）")
    h = hashlib.sha256(np.ascontiguousarray(cmd7, np.float32).tobytes()).hexdigest()[:16]   # 同 replay_probe
    p = OUT / f"seq_{rid}.npz"
    if RR.attempted(rid, 1, OUT):
        old = json.loads(str(np.load(p)["source"]))["cmd_hash"]
        if old != h:
            raise SystemExit(f"⛔ {rid}：重建出的序列与回放时的不同（{old} → {h}），代码或输入变了，不能接着跑")
    else:
        np.savez(p, cmd7=cmd7, da=np.zeros((len(cmd7), 6), np.float32),
                 source=json.dumps({"rid": rid, "label": label, "cmd_hash": h}, ensure_ascii=False))
    return p


def orientation_deg(ep, calib):
    """朝向质量检查：<3 Hz 编码器刚体运动学角速度与陀螺之间的最佳旋转，相对标定 R 的夹角（同 cmd_tracking 的检查）。"""
    C = T.C
    w = D.lp(C.predict_site(ep["tq"], ep["q"])[1], 3.0, D.FS_Q)
    g = D.lp(ep["imu"][:, 3:], 3.0, D.FS_I) - np.asarray(calib["gyro_bias"])
    wi = np.stack([np.interp(ep["ti"] - calib["delay_s"], ep["tq"], w[:, k]) for k in range(3)], 1)
    mv = np.linalg.norm(g, axis=1) > C.GYRO_MIN
    return float(C._deg(C._kabsch(wi[mv], g[mv]) @ np.asarray(calib["R"]).T))


def result(rid, ctx, s):
    """一遍回放 → (指标, (ē_q, ē_ω), 朝向偏差°)，缓存在 res_{rid}.npz；无效遍（提前终止、哈希或帧数不符、没有编码器）返回 None。"""
    cache = OUT / f"res_{rid}.npz"
    if cache.exists():
        z = np.load(cache)
        return json.loads(str(z["met"])), (z["eq"], z["ew"]), float(z["qc"])
    h = json.loads((OUT / f"replay_{rid}_r1.json").read_text())
    z = np.load(OUT / f"seq_{rid}.npz")
    if not (h.get("stopped") is None and h.get("cmd_hash") == json.loads(str(z["source"]))["cmd_hash"]
            and len(h.get("t_send", [])) == len(z["cmd7"]) and h.get("can_q200")):
        return None
    ep = T.from_record(h, np.asarray(h["cmd"], float))
    q, a6 = ctx.seq[s], ctx.seq[s]["a7"][:, :6]
    fq, fi = S.masks(ep, q["marked"], MARGIN)
    met = S.metrics(ep, a6, fq, fi, ctx.calib, ctx.model, ctx.sid)
    err = (joint_err(ep, a6, q["marked"]), imu_err(ep, a6, q["marked"], ctx.calib, ctx.model, ctx.sid))
    qc = orientation_deg(ep, ctx.calib)
    np.savez(cache, met=json.dumps(met), eq=err[0], ew=err[1], qc=qc)
    return met, err, qc


def learner(st, lk, box):
    """需要时建这一环节的学习器：共同关节阶段从两遍 Shift 基准出发（Δ = 0）；两支从分叉点（共同阶段最后接受的补正）出发。"""
    if lk == "b" or lk.startswith("e_"):
        return None
    if lk not in st:
        if lk == "c":
            if not st["b"]:
                raise SystemExit("⛔ 没有有效的 Shift 基准回放，不能开始共同关节阶段")
            m, e = mean_res([r for _, r in st["b"]])
            st["c"] = Learner("c", BUDGET["c"], np.zeros_like(e[0]), e, m, box, [rid for rid, _ in st["b"]])
        else:
            c = st["c"]
            st[lk] = Learner(lk, BUDGET[lk], c.acc, c.err, c.met, box, c.hist[-1][0])
    return st[lk]


def fixed(st, lk, n):
    """不迭代的那几遍要发的 Δ（n 帧）：Shift 基准与评价里的 Shift 为 0，其余取各环节最后接受的补正。"""
    if lk in ("b", "e_shift"):
        return np.zeros((n, len(T.ACTIVE)))
    return st["c" if lk == "e_bp" else lk[2:]].acc


def walk(ctx, execute=None):
    """按排程把已有回放喂给各环节；遇到还没开始过的那一遍：execute 为 None 就停下，否则调用 execute(序号, 总数, rid, 序列文件, 学习器)，
    它返回 True 表示跑完了、接着往下，返回整数则以此为退出码停下。
    返回 (各序列状态, 退出码)。状态：b = [(rid, 结果)]、e = {(轮, 条件): 指标}、trace = [(序号, 环节, 指标)]、c / j / i = Learner。"""
    st = {s: {"b": [], "e": {}, "trace": []} for s in SEQS}
    plan = schedule()
    for k, (s, lk, n) in enumerate(plan, 1):
        rid, q = f"{s}_{lk}{n:02d}", ctx.seq[s]
        L = learner(st[s], lk, q["box"])
        cand = L.next() if L else None
        label = cand["label"] if cand else ("Shift 基准" if lk == "b" else f"评价 {lk[2:]}")
        seq = write_seq(rid, q["s7"], cand["delta"] if cand else fixed(st[s], lk, len(q["s7"])), label)
        if not RR.attempted(rid, 1, OUT):
            if execute is None:
                return st, 0
            rc = execute(k, len(plan), rid, seq, L)
            if rc is not True:
                return st, rc
        res = result(rid, ctx, s)
        if res is None:
            msg = "无效的一遍（提前终止或记录不全）：占预算、不补跑"
            if L:
                L.void()
        elif res[2] > QC_MAX:
            print(f"⛔ {rid}：IMU 朝向偏差 {res[2]:.1f}° > {QC_MAX}°，IMU 可能被碰动；停下重新标定，本会话不能接着用")
            return st, 3
        else:
            st[s]["trace"].append((k, lk, res[0]))
            if L:
                cand["rids"].append(rid)
                msg = L.add(cand, res[0], res[1]) + f"（余 {L.left} 遍）"
            elif lk == "b":
                st[s]["b"].append((rid, res[:2]))
                msg = "记下"
            else:
                st[s]["e"][(n, lk[2:])] = res[0]
                msg = "记下"
        if execute is not None:
            vals = f"关节 {res[0]['joint_raw']:.2f} mrad · 末端 {res[0]['ee_raw']:.4f} rad/s · 朝向 {res[2]:.1f}° " if res else ""
            print(f"  [{k}/{len(plan)}] {rid} {label}：{vals}→ {msg}")
    return st, 0


def cmd_run(dry_run, one=None):
    if one:                                       # 子进程：跑一遍（序列文件, rid）
        return S.guarded_run(Path(one[0]), f"{one[1]}_", "r1", OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(CALIB.read_bytes()).hexdigest()[:16]
    if dt.date.fromtimestamp(CALIB.stat().st_mtime) != dt.date.today():
        print("⚠ IMU 标定不是今天做的：上机前先跑 sim/imu_calib.py replay 和 fit（方案 §1 IMU 前提）")
        if not dry_run:
            return 2
    sess = OUT / "session.json"
    if sess.exists() and json.loads(sess.read_text())["calib_sha"] != sha:
        print("⛔ 标定文件和本会话开始时不同：同一会话不能混用两次标定")
        return 2
    ctx = Ctx()
    if dry_run:
        import replay_probe as RP
        _, rc = walk(ctx, lambda k, n, rid, seq, _L: print(f"下一遍：[{k}/{n}] {rid}") or RP.run("r1", True, seqfile=str(seq), tag=f"{rid}_", outdir=OUT))
        return rc
    if not sess.exists():
        sess.write_text(json.dumps({"calib_sha": sha, "plan": "reports/PLAN_route_compare.md v0.6"}))
    rp_sha = hashlib.sha256((HERE / "replay_probe.py").read_bytes()).hexdigest()

    def execute(k, n, rid, seq, _L):
        if input(f"\n[{k}/{n}] {rid}：桌面已清空、可以运动就按回车（输入 q 退出）").strip().lower() == "q":
            return 0
        utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        code = RR.run_child([sys.executable, "-u", __file__, "run", "--one", str(seq), rid])
        with open(OUT / "run_log.jsonl", "a") as f:
            f.write(json.dumps({"k": k, "rid": rid, "utc": utc, "code": code, "replay_probe_sha256": rp_sha}) + "\n")
        if code:
            print(f"  ⚠ {rid} 返回码 {code}：这一遍不补跑。检查机械臂后再运行（会从下一遍接着跑）。")
            return code
        return True

    _, rc = walk(ctx, execute)
    if rc == 0 and all(RR.attempted(f"{s}_{lk}{n:02d}", 1, OUT) for s, lk, n in schedule()):
        print("68 遍都已开始过。下一步：python -u sim/ilc_route1.py analyze")
    return rc


# ── 分析（§5）────────────────────────────────────────────────────────────
def hier(rel, n_boot=10000, seed=0):
    """rel = {序列: 各轮的相对变化（%），按轮次对齐}：总体效应 = 各序列中位数的平均；95% 区间 = 对完整轮次有放回重抽，
    A、B 用同一组轮次（保留同轮关联），序列不重抽。"""
    R = np.array(list(rel.values()))                       # 序列 × 轮
    idx = np.random.default_rng(seed).integers(0, R.shape[1], (n_boot, R.shape[1]))
    boot = np.median(R[:, idx], axis=2).mean(0)
    return {"effect": float(np.mean([np.median(v) for v in rel.values()])),
            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
            "per_seq": {s: {"median": float(np.median(v)), "range": [float(min(v)), float(max(v))], "n": len(v)} for s, v in rel.items()}}


def compare(e, x, y, keys=("joint_raw", "ee_raw", "path", "speed")):
    """条件 x 相对 y：只用两条序列都有 x、y 的完整轮次，按轮配对算相对变化（%）。规定的 4 轮都完整才判定，否则只报告、记未完成。
    e = {序列: {(轮, 条件): 指标}}。"""
    rounds = [n for n in range(1, N_EVAL + 1) if all((n, x) in e[s] and (n, y) in e[s] for s in e)]
    r = {key: hier({s: [(e[s][(n, x)][key] - e[s][(n, y)][key]) / e[s][(n, y)][key] * 100 for n in rounds] for s in e})
         if rounds else None for key in keys}
    r["rounds"] = rounds
    r["verdict"] = verdict(r) if len(rounds) == N_EVAL else f"只报告、未完成（完整轮次 {len(rounds)}/{N_EVAL}）"
    return r


def verdict(r):
    """判据 b：两个主指标的上界都 ≤ +5%（不变差），至少一个上界 < 0（改善，不设最小降幅）；护栏：路径长度、关节速度 ±5%。"""
    if any(abs(r[k]["effect"]) > 5 for k in ("path", "speed")):
        return "护栏未过（运动量或速度的变化超过 ±5%）"
    better = [name for k, name in (("joint_raw", "关节偏差"), ("ee_raw", "末端角速度偏差")) if r[k]["ci95"][1] < 0]
    if all(r[k]["ci95"][1] <= 5 for k in ("joint_raw", "ee_raw")) and better:
        return f"有改善（{'、'.join(better)}）"
    return "没有改善"


def time_shift_ms(delta, a6, W):
    """Δ 的纯时间平移分量 δt = ⟨Δ, ȧ⟩ / ⟨ȧ, ȧ⟩（ȧ = 原始指令速度；5 个活动关节、窗 W > 0 的帧与关节），ms；正 = 更早。"""
    ad = np.gradient(a6[:, T.ACTIVE], RR.DT, axis=0)[W > 0]
    return float(np.sum(delta[W > 0] * ad) / np.sum(ad ** 2) * 1e3)


def empirical_lag_ms(pairs, W, lags=np.arange(0, 0.205, 0.005)):
    """新增 Δ 的经验滞后：相邻两份接受的补正之差 dΔ，与对应回放的编码器之差 dq（按名义帧时间对齐）互相关最大的滞后（ms）。"""
    def q_nominal(rid, grid):
        h = json.loads((OUT / f"replay_{rid}_r1.json").read_text())
        ep = T.from_record(h, np.asarray(h["cmd"], float))
        fr = np.interp(ep["tq"], ep["t_send"], RR.DT * np.arange(len(ep["t_send"])))
        return np.stack([np.interp(grid, fr, ep["q"][:, j]) for j in T.ACTIVE], 1)

    out, tk, m = [], RR.DT * np.arange(len(W)), W > 0
    for (r0, d0), (r1, d1) in pairs:
        dd = (d1 - d0)[m]
        if not r0 or not r1 or not np.any(dd):
            continue
        cs = [np.sum(dd * (q_nominal(r1[0], tk + lag) - q_nominal(r0[0], tk + lag))[m]) for lag in lags]
        out.append(float(lags[int(np.argmax(cs))] * 1e3))
    return out


def plot(st):
    """每条序列一列：关节偏差、末端角速度偏差随回放序号（Shift 基准、共同关节阶段、两支、最终评价）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    color = {"b": T.MUTED, "c": T.INK, "j": T.BLUE, "i": T.ORANGE}
    name = {"b": "Shift 基准", "c": "共同关节阶段", "j": "只关节支", "i": "加 IMU 支"}
    fig, ax = plt.subplots(2, len(SEQS), figsize=(12, 6), squeeze=False)
    for c, s in enumerate(SEQS):
        for r, (key, title) in enumerate((("joint_raw", "关节偏差（mrad）"), ("ee_raw", "末端角速度偏差（rad/s）"))):
            A = ax[r, c]
            for lk in color:
                pts = [(k, m[key]) for k, l, m in st[s]["trace"] if l == lk]
                if pts:
                    A.plot(*zip(*pts), "o-", color=color[lk], ms=3, lw=1, label=name[lk])
            ev = [(k, m[key]) for k, l, m in st[s]["trace"] if l.startswith("e_")]
            if ev:
                A.plot(*zip(*ev), "o", color=T.SHADE, ms=3, label="最终评价")
            A.set_title(f"序列 {s} · {title}", loc="left", color=T.INK)
            for side in ("top", "right"):
                A.spines[side].set_visible(False)
        ax[-1, c].set_xlabel("回放序号")
    ax[0, -1].legend(frameon=False, labelcolor=T.INK)
    fig.tight_layout()
    fig.savefig(OUT / "iterations.png", dpi=120, facecolor="#fcfcfb")
    plt.close(fig)


def plot_compare(st, ctx):
    """最终评价的偏差随时间：Shift、只关节支、加 IMU 支，每个条件的各遍按名义帧时间对齐后逐点均方根、0.25 s 平滑。
    两张图：相对原始指令（主指标）、相对理想渲染（另报）。灰色 = 不计入评价的时段（开头保持、疑似接触段、末尾）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 9, "axes.edgecolor": T.MUTED,
                         "xtick.color": T.MUTED, "ytick.color": T.MUTED})
    conds = (("shift", "Shift（提前 60 ms）", T.MUTED, "-"), ("j", "ILC 只关节支", T.BLUE, "-"), ("i", "ILC 加 IMU 支", T.ORANGE, "--"))
    keys = {"joint_raw": D.FS_Q, "joint_ideal": D.FS_Q, "ee_raw": D.FS_I, "ee_ideal": D.FS_I}
    curves = {}
    for s in SEQS:
        n, a6 = len(ctx.seq[s]["a7"]), ctx.seq[s]["a7"][:, :6]
        for cond, *_ in conds:
            acc = {k: [] for k in keys}
            for r in sorted(r for r, c in st[s]["e"] if c == cond):     # 只用有效的那几遍
                h = json.loads((OUT / f"replay_{s}_e_{cond}{r:02d}_r1.json").read_text())
                ep = T.from_record(h, np.asarray(h["cmd"], float))
                sq = S.sq_dev(ep, a6, ctx.calib, ctx.model, ctx.sid)
                for k, fs in keys.items():
                    t = ep["tq"] if k.startswith("joint") else ep["ti"]
                    fr = np.interp(t, ep["t_send"], RR.DT * np.arange(n))
                    acc[k].append(np.interp(np.arange(0, (n - 1) * RR.DT, 1 / fs), fr, sq[k]))
            for k, fs in keys.items():
                w = int(0.25 * fs)
                curves[(s, cond, k)] = np.sqrt(np.convolve(np.mean(acc[k], 0), np.ones(w) / w, "same"))
    for fam, title in (("raw", "ILC 前后：相对原始策略指令的偏差（主指标）"),
                       ("ideal", "ILC 前后：相对理想渲染的偏差（另报；理想 = 加速度受限、不含延迟）")):
        fig, ax = plt.subplots(2, len(SEQS), figsize=(12, 5.6), squeeze=False, sharex=True)
        for c, s in enumerate(SEQS):
            n = len(ctx.seq[s]["a7"])
            off = ctx.seq[s]["marked"].copy()
            off[:RR.HOLD] = True
            off[np.arange(n) * RR.DT > (n - 1) * RR.DT - MARGIN] = True
            for row, (k, name, scale, unit) in enumerate(((f"joint_{fam}", "关节偏差", 1e3, "mrad"), (f"ee_{fam}", "末端角速度偏差", 1.0, "rad/s"))):
                A = ax[row, c]
                for k0, k1 in zip(np.flatnonzero(off & ~np.r_[False, off[:-1]]), np.flatnonzero(off & ~np.r_[off[1:], False])):
                    A.axvspan(k0 * RR.DT, (k1 + 1) * RR.DT, color=T.SHADE, alpha=0.5, lw=0)
                for cond, label, color, ls in conds:
                    y = curves[(s, cond, k)] * scale
                    med = np.median([st[s]["e"][(r, cc)][k] for r, cc in st[s]["e"] if cc == cond])
                    A.plot(np.arange(len(y)) / keys[k], y, color=color, ls=ls, lw=1.4,
                           label=f"{label}  {med:.2f} {unit}" if unit == "mrad" else f"{label}  {med:.3f} {unit}")
                A.set_ylim(0, 1.45 * max(curves[(s, cc, k)].max() for cc, *_ in conds) * scale)   # 上方留白放图例，不压曲线
                A.grid(axis="y", color=T.SHADE, lw=0.6)
                A.set_title(f"序列 {s} · {name}（{unit}）", loc="left", color=T.INK)
                A.legend(frameon=False, labelcolor=T.INK, fontsize=8, loc="upper right")
                for side in ("top", "right"):
                    A.spines[side].set_visible(False)
            ax[-1, c].set_xlabel("时间（s，名义帧时间）")
        fig.suptitle(title, x=0.01, ha="left", color=T.INK)
        fig.text(0.01, 0.005, "每个条件 4 遍逐点均方根、0.25 s 平滑；灰色 = 不计入评价；图例数字 = 非接触段 4 轮中位", color=T.MUTED, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(OUT / f"compare_{fam}.png", dpi=130, facecolor="#fcfcfb")
        plt.close(fig)


def cmd_analyze():
    sess = json.loads((OUT / "session.json").read_text())
    if sess["calib_sha"] != hashlib.sha256(CALIB.read_bytes()).hexdigest()[:16]:
        print("⛔ 当前标定文件不是本会话用的那一份")
        return 2
    ctx = Ctx()
    st, _ = walk(ctx)
    keys = ("joint_raw", "ee_raw", "path", "speed")
    names = {"joint_raw": "关节偏差", "ee_raw": "末端角速度偏差", "path": "护栏：关节路径长度", "speed": "护栏：关节速度 RMS"}
    e = {s: st[s]["e"] for s in SEQS}
    res = {f"{x} vs {y}": compare(e, x, y) for x, y in (("bp", "shift"), ("j", "shift"), ("i", "shift"), ("i", "j"))}
    diag = {}
    for s in SEQS:
        W, a6 = ctx.seq[s]["box"][0], ctx.seq[s]["a7"][:, :6]
        hist = [h for lk in ("c", "j", "i") if lk in st[s] for h in zip(st[s][lk].hist, st[s][lk].hist[1:])]
        diag[s] = {"time_shift_ms": {b: time_shift_ms(fixed(st[s], "e_" + b, len(W)), a6, W) for b in ("bp", "j", "i")},
                   "empirical_lag_ms": empirical_lag_ms(hist, W),
                   "final": {lk: {"gamma": st[s][lk].gamma, "c": st[s][lk].c, "probing": st[s][lk].probing,
                                  "accepted": len(st[s][lk].hist) - 1} for lk in ("c", "j", "i") if lk in st[s]}}
    (OUT / "summary.json").write_text(json.dumps({"plan": sess["plan"], "paired": res, "diagnostics": diag}, ensure_ascii=False, indent=1))
    label = {"bp": "分叉点", "shift": "Shift", "j": "只关节支", "i": "加 IMU 支"}
    for comp, r in res.items():
        x, y = comp.split(" vs ")
        print(f"\n=== {label[x]} vs {label[y]}（相对变化 %：总体 = 两条序列中位数的平均 [分层自助法 95% 区间]；逐序列中位）")
        for key in keys:
            v = r[key]
            print(f"  {names[key]:14s}" + (f"{v['effect']:+6.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]  "
                                           + " · ".join(f"{s} {p['median']:+.1f}（{p['n']} 对）" for s, p in v["per_seq"].items()) if v else "N/A"))
        print(f"  判定：{r['verdict']}")
    for s, d in diag.items():
        print(f"\n序列 {s}：纯时间平移分量 " + " · ".join(f"{label[b]} {v:+.1f} ms" for b, v in d["time_shift_ms"].items())
              + f"；新增 Δ 的经验滞后 {d['empirical_lag_ms']} ms（τ 的初始假设 {TAU * 1e3:.0f} ms）")
        print("  " + " · ".join(f"{lk}：接受 {f['accepted']} 次、γ {f['gamma']:g}、c {f['c']:g}{'、试探' if f['probing'] else ''}" for lk, f in d["final"].items()))
    plot(st)
    plot_compare(st, ctx)
    print(f"\n→ {OUT}/summary.json、iterations.png、compare_raw.png、compare_ideal.png")
    return 0


def main(argv=None) -> int:
    gc.disable()        # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="跑会话（加 --dry-run 只检查下一遍要发的序列，不连机械臂）")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--one", nargs=2, help=argparse.SUPPRESS)      # 内部用：子进程跑一遍（序列文件, rid）
    sub.add_parser("analyze", help="最终评价的配对比较、四组判定与诊断")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a.dry_run, a.one)
    return cmd_analyze()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # 跳过解释器收尾（关 GC 的进程在退出阶段也会崩）；回放子进程因此不触发机器人析构里的无插值归位
