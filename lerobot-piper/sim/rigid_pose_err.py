#!/usr/bin/env python3
"""补充分析：编码器推算的刚体末端位姿误差（用户 2026-10-22 定：补充分析，不替换已冻结的判据；评价标准 §7）。

编码器 6 个关节角经刚体正运动学，得到末端站点 ee_imu_site（link6 上、关节 6 轴向 13 cm，夹爪根部附近）的位置与姿态；
参考关节轨迹经同一正运动学得到参考位姿。逐样本取位置误差（欧氏距离）与姿态误差（相对旋转角），在关节偏差所用的同一批帧上取 RMS：
位置 mm、姿态 mrad。参考同关节偏差：原始指令一族 = 原始策略指令零阶保持；理想渲染一族 = 原指令经 a_max 规划、从实测起点出发。
只含编码器看得到的刚体部分，不含连杆、传动、支架的变形，也不用 IMU。配对与区间沿用各会话已冻结的做法
（Shift 回放：每个条件 12 对合并；路线 1 各轮：A、B 共同重抽完整轮次）。

用法（lerobot 环境）：python -u sim/rigid_pose_err.py   → outputs/rigid_pose_err/summary.json
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
import cmd_tracking as T  # noqa: E402
import ilc_route1 as I  # noqa: E402
import shift_replay as S  # noqa: E402

OUT = HERE.parent / "outputs/rigid_pose_err"
KEYS = ("pos_raw", "rot_raw", "pos_ideal", "rot_ideal")
NAMES = {"pos_raw": "位置 相对原始指令", "rot_raw": "姿态 相对原始指令", "pos_ideal": "位置 相对理想渲染", "rot_ideal": "姿态 相对理想渲染"}
ILC = (("ilc_route1", "10-19 路线 1（v0.6）"), ("ilc_route1_v2", "10-21 v0.7"), ("ilc_route1_v3", "10-22 v0.8"), ("ilc_route1_v4", "v0.9"))
LABEL = {"shift": "Shift", "bp": "分叉点", "j": "只关节支", "i": "加 IMU 支", "j07": "v0.7 只关节支（0.25 s 窗）"}


def fk(model, data, sid, q6):
    """关节角（N × 6）→ 末端站点的位置（N × 3，m）与姿态（N × 3 × 3）：刚体正运动学。"""
    import mujoco
    p, R = np.empty((len(q6), 3)), np.empty((len(q6), 3, 3))
    for i, q in enumerate(q6):
        data.qpos[:6] = q
        mujoco.mj_kinematics(model, data)
        p[i], R[i] = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)
    return p, R


def pose_err(p, R, pr, Rr):
    """逐样本位置误差 ‖p − p_ref‖（m）与姿态误差 = R_refᵀR 的旋转角（rad；2·arcsin(‖R_refᵀR − I‖_F / 2√2)，小角度数值稳定）。"""
    dR = np.einsum("nji,njk->nik", Rr, R) - np.eye(3)
    return np.linalg.norm(p - pr, axis=1), 2 * np.arcsin(np.clip(np.linalg.norm(dR, axis=(1, 2)) / (2 * np.sqrt(2)), 0, 1))


def metrics(ep, cmd, free_q, model, data, sid):
    """一遍的编码器推算的刚体末端位姿误差：两种参考各一份位置（mm）与姿态（mrad），关节偏差同一批帧（free_q）上的 RMS。"""
    tq = ep["tq"]
    p, R = fk(model, data, sid, ep["q"][:, :6])
    refs = {"raw": S.zoh(ep["t_send"], cmd, tq), "ideal": T.PlanTrajectory(cmd, ep["t_send"], q0=ep["q"][0])(tq)[:, :6]}
    out = {}
    for fam, ref in refs.items():
        dp, dr = pose_err(p, R, *fk(model, data, sid, ref))
        out[f"pos_{fam}"] = float(np.sqrt(np.mean(dp[free_q] ** 2))) * 1e3
        out[f"rot_{fam}"] = float(np.sqrt(np.mean(dr[free_q] ** 2))) * 1e3
    return out


def shift_session(model, data, sid):
    """10-19 Shift 回放：各遍的位姿误差 → 「条件 − 基线」配对（同 shift_replay.paired）与各条件中位。帧与 shift_replay 分析相同。"""
    center = T.target_center("pickcube_raw", model)
    rows = {}
    for s in S.SEQS:
        cmd = np.load(S.OUT / f"seq_{s}_base.npz")["cmd7"].astype(float)[:, :6]
        marked = np.any([S.RR.contact_frames(np.load(S.OUT / f"seq_{s}_{c}.npz")["cmd7"].astype(float), model, center)
                         for c in S.CONDS], axis=0)
        for c in S.CONDS:
            for r, (ep, _) in S.load_cond(s, c).items():
                rows[(s, c, r)] = metrics(ep, cmd, S.masks(ep, marked, max(S.CONDS.values()) + 0.05)[0], model, data, sid)
    med = {c: {k: float(np.median([v[k] for (_, cc, _), v in rows.items() if cc == c])) for k in KEYS} for c in S.CONDS}
    return {"median": med, "paired": {c: {k: S.paired(rows, c, k) for k in KEYS} for c in S.CONDS if c != "base"}}


def ilc_session(name):
    """路线 1 一轮会话：最终评价有效各遍（同 analyze）的位姿误差 → 与 analyze 相同的比较组与各条件中位。帧与 analyze 相同。"""
    import mujoco
    I.OUT = HERE.parent / "outputs" / name
    ctx = I.Ctx()
    st, _ = I.walk(ctx)                       # 只读：已开始过的遍只核对序列哈希、读缓存
    data = mujoco.MjData(ctx.model)
    e = {s: {} for s in I.SEQS}
    for s in I.SEQS:
        for n, c in st[s]["e"]:
            h = json.loads((I.OUT / f"replay_{s}_e_{c}{n:02d}_r1.json").read_text())
            ep = T.from_record(h, np.asarray(h["cmd"], float))
            e[s][(n, c)] = metrics(ep, ctx.seq[s]["a7"][:, :6], S.masks(ep, ctx.seq[s]["marked"], I.MARGIN)[0], ctx.model, data, ctx.sid)
    pairs = [(x, "shift") for x in ("bp", "j", "i") if x in ctx.evals] + [("i", "j")] + ([("j", "j07"), ("j07", "shift")] if "j07" in ctx.evals else [])
    paired = {}
    for x, y in pairs:
        rounds = [n for n in range(1, I.N_EVAL + 1) if all((n, x) in e[s] and (n, y) in e[s] for s in e)]
        paired[f"{x} vs {y}"] = {k: I.hier({s: [(e[s][(n, x)][k] - e[s][(n, y)][k]) / e[s][(n, y)][k] * 100 for n in rounds] for s in e})
                                 for k in KEYS} | {"rounds": rounds}
    med = {s: {c: {k: float(np.median([v[k] for (_, cc), v in e[s].items() if cc == c])) for k in KEYS} for c in ctx.evals} for s in I.SEQS}
    return {"median": med, "paired": paired}


def report(res):
    print("编码器推算的刚体末端位姿误差（补充分析，不替换已冻结的判据）：位置 mm、姿态 mrad；相对变化 %：中位或总体 [95% 区间]")
    sh = res["shift_replay"]
    print("\n=== 10-19 Shift 回放（每个条件 12 对合并）")
    for c in S.CONDS:
        print(f"  {S.LABEL[c]} 中位：" + " · ".join(f"{NAMES[k]} {sh['median'][c][k]:.3f}" for k in KEYS))
    for c, d in sh["paired"].items():
        print(f"  {S.LABEL[c]} − 基线：" + " · ".join(f"{NAMES[k]} {v['median']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]"
                                                  for k, v in d.items() if v))
    for name, label in ((n, l) for n, l in ILC if n in res):
        r = res[name]
        print(f"\n=== {label}（A、B 共同重抽完整轮次）")
        for s, m in r["median"].items():
            print(f"  序列 {s} 中位（相对原始指令）：" + " · ".join(f"{LABEL[c]} {v['pos_raw']:.3f} mm / {v['rot_raw']:.2f} mrad" for c, v in m.items()))
        for comp, d in r["paired"].items():
            x, y = comp.split(" vs ")
            print(f"  {LABEL[x]} vs {LABEL[y]}（完整轮次 {len(d['rounds'])}）：" + " · ".join(
                f"{NAMES[k]} {d[k]['effect']:+.1f} [{d[k]['ci95'][0]:+.1f}, {d[k]['ci95'][1]:+.1f}]" for k in KEYS))
    print(f"\n→ {OUT}/summary.json")


def main() -> int:
    gc.disable()        # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    res = {"shift_replay": shift_session(model, mujoco.MjData(model), sid)}
    res.update({name: ilc_session(name) for name, _ in ILC if (HERE.parent / "outputs" / name / "summary.json").exists()})   # 已分析过的会话
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    report(res)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # 跳过解释器收尾（关 GC 的进程在退出阶段也会崩）
