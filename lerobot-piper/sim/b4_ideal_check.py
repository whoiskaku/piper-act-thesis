"""按老师定义的理想动作做的真机评价（用户 11-12）。两个评价集，都走标准回放管线（b4_teacher.py run）：
  ideal   训练轨迹 p01、p03、p05：shift60（录制时的 Shift）、shift84（同一算法提前 84 ms）、teacher24（ILC 教师整体提前 24 ms）。
          依据：已有回放离线算出「实际运动 ≈ 理想整体晚约 83 ms + 形状误差」，Shift 60 ms 还剩约 24 ms 滞后。
  b4val   10 条验证轨迹：shift84（新基线）对 regen（冻结的 ②：teacher24 标签、只看 ACT 指令）——B4 的真机评价。
          ② 只看 ACT 指令，输出与机械臂状态无关：按使用方式逐帧预演（R.preview）算出的就是在线会发的整条指令，预先算好再回放，
          两种条件都在帧开始时发令。
  fbval   同 b4val 的 10 条验证轨迹：b4（冻结的 ②）对 b4fb（② + 写死的比例反馈，设计 `reports/PLAN_b3.md` §9）。
          两种条件都走在线回放（b4_regen_replay.py run --mode regen，b4fb 另加 --fb），发令时刻相同；不写回放序列，在线回放自己落盘。
夹爪都用 Shift 的（regen 用 shift84 的，同使用时）。每条轨迹的各遍连着跑，第 2 轮顺序倒过来；编号 `{轨迹}_{前缀}{条件}{轮}`。

  python -u sim/b4_ideal_check.py replay --set b4val --session outputs/b4/teacher/sessions/<会话>   # 写回放序列，逐遍提示摆方块，子进程跑
  python -u sim/b4_ideal_check.py analyze --set b4val --session outputs/b4/teacher/sessions/<会话>  # 新主指标（理想）、旧主指标与判定
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_regen as R  # noqa: E402
import b4_stats as ST  # noqa: E402
import b4_teacher as B  # noqa: E402
import ideal_motion as IM  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

SETS = {"ideal": (("p01", "p03", "p05"), ("shift60", "shift84", "teacher24"), "v"),          # (轨迹, 条件（第一个是基线）, 编号前缀)
        "b4val": (("p07", "p17", "p23", "p25", "p27", "p28", "p30", "p31", "p39", "p44"), ("shift84", "regen"), "n")}
SETS["fbval"] = (SETS["b4val"][0], ("b4", "b4fb"), "f")
ONLINE = {"fbval": {"b4": [], "b4fb": ["--fb"]}}     # 走在线回放的评价集：条件 → b4_regen_replay.py run 另加的参数
ROUNDS = 2
KEYS = ("joint_ideal", "lag_ms", "shape", "joint_raw", "ee_ideal", "qc", "path", "speed")
NAMES = {"joint_ideal": "理想 mrad", "lag_ms": "滞后 ms", "shape": "去掉滞后 mrad", "joint_raw": "零阶保持 mrad",
         "ee_ideal": "末端角速度 rad/s", "qc": "朝向°", "path": "路径", "speed": "速度"}


def regen_cmd(tid: str, fn) -> np.ndarray:
    """冻结的 ② 在这条轨迹上会发出的整条指令：按使用方式逐帧预演（窗为 0 的关节和夹爪发 shift84）。只看 ACT 指令时与实测无关，
    实测窗口借会话 20261106q 的 qshift1 回放凑输入单元。预演触发逐帧检查就停（这样的指令不能上机）。"""
    sess = B.TEACH / "sessions" / "20261106q"
    base = {**B.load_seq(tid), "s7": R.shift_cmd(tid, IM.LEAD_S)}
    h = dict(R.valid_replays(sess, tid, len(base["a7"])))[f"{tid}_qshift1"]
    bd = json.loads((R.REGEN / "bounds.json").read_text())
    pv = R.preview(fn, h, base, R._log(tid), R._calib(sess), np.asarray(bd["lo"]), np.asarray(bd["hi"]))
    if pv["stops"]:
        raise SystemExit(f"⛔ {tid}：② 预演触发逐帧检查 {pv['stops'][:2]}，不能上机")
    return pv["u7"].astype(float)


def commands(tid: str, conds: tuple, rows: dict, fn=None) -> dict:
    """要发的整条指令。shift84 由 R.shift_cmd 生成（同录制时的算法，先用 60 ms 重建核对）；regen 见 regen_cmd。"""
    make = {"shift60": lambda: np.asarray(B.load_seq(tid)["s7"], float),
            "shift84": lambda: R.shift_cmd(tid, IM.LEAD_S),
            "teacher24": lambda: IM.earlier(np.load(B.TEACH / "sessions" / rows[tid]["session"] / f"teacher_{tid}.npz")["cmd7"], IM.EARLY_S),
            "regen": lambda: regen_cmd(tid, fn)}
    return {c: make[c]() for c in conds}


def plan(ids, conds) -> list[tuple[str, str, int]]:
    """每条轨迹的各遍连着跑：第 1 轮按 conds 的顺序，第 2 轮倒过来（抵消会话内的漂移）。"""
    return [(t, c, r) for t in ids for r in range(1, ROUNDS + 1) for c in (conds if r % 2 else conds[::-1])]


def _rid(tid: str, prefix: str, cond: str, r: int) -> str:
    return f"{tid}_{prefix}{cond}{r}"


def cmd_replay(set_: str, sess: Path) -> int:
    ids, conds, prefix = SETS[set_]
    B.check_session(sess, C.CALIB)
    sess.mkdir(parents=True, exist_ok=True)
    if not (sess / "session.json").exists():
        (sess / "session.json").write_text(json.dumps({"calib_sha": B._sha(C.CALIB), "rule": B.RULE, "qc_max": B.QC_MAX}))
    rows, pos = json.loads((B.TEACH / "summary.json").read_text()), {p["id"]: p["cm"] for p in P.load()["pool"]}
    fn, sha = O.load_regen(str(R.REGEN / "frozen.json")) if "regen" in conds else (None, None)
    online = ONLINE.get(set_)
    cmds, todo = ({} if online else {t: commands(t, conds, rows, fn) for t in ids}), plan(ids, conds)
    for k, (tid, cond, r) in enumerate(todo, 1):
        rid = _rid(tid, prefix, cond, r)
        if online:
            args = ["b4_regen_replay.py", "run", "--session", str(sess), "--tid", tid, "--rid", rid, "--mode", "regen",
                    "--net", str(R.REGEN / "frozen.json"), *online[cond]]
        else:
            seq = B.write_cmd(sess, rid, B.load_seq(tid)["s7"], cmds[tid][cond], f"{set_} {cond}", {"net_sha": sha} if cond == "regen" else None)
            args = ["b4_teacher.py", "run", "--session", str(sess), "--one", str(seq), rid]
        if RR.attempted(rid, 1, sess):
            continue
        if input(f"\n[{k}/{len(todo)}] {rid}：方块摆在 {P.mark_desc(pos[tid])}（{pos[tid]:+g} cm），可以运动就按回车（输入 q 退出）"
                 ).strip().lower() == "q":
            return 0
        code = RR.run_child([sys.executable, "-u", str(HERE / args[0]), *args[1:]])
        if code:
            print(f"  ⚠ {rid} 返回码 {code}：这一遍不补跑。检查机械臂后再运行（会从下一遍接着跑）。")
            return code
    print(f"=== {len(todo)} 遍都跑完了。下一步：python -u sim/b4_ideal_check.py analyze --set {set_} --session {sess}")
    return 0


def cmd_analyze(set_: str, sess: Path) -> dict:
    """每遍：相对理想的关节偏差（新主指标）、整体滞后、去掉滞后后的形状误差、零阶保持（旧主指标）、末端角速度相对理想的偏差、IMU 朝向、
    路径长度、关节速度。表：各条件「每条轨迹先平均各轮」的中位。比较：每个非基线条件相对基线（conds 的第一个），按轮配对的相对变化
    → 两级自助（同 b4_eval）；轨迹都跑齐时按 ST.judge(main="joint_ideal") 判定。→ 会话目录 {评价集}_check.json。"""
    ids, conds, prefix = SETS[set_]
    ctx = B.session_ctx(sess)
    res = {}
    for tid in ids:
        q = B.load_seq(tid)
        x = IM.ideal(q["a7"])
        for cond in conds:
            for r in range(1, ROUNDS + 1):
                rid = _rid(tid, prefix, cond, r)
                got = B.replay_result(rid, sess, q, ctx) if (sess / f"replay_{rid}_r1.json").exists() else None
                if got is None:
                    print(f"  {rid}：没跑或无效，不计")
                    continue
                h = json.loads((sess / f"replay_{rid}_r1.json").read_text())
                ep = C.T.from_record(h, np.asarray(h["cmd"], float))
                fq, fi = S.masks(ep, np.asarray(q["marked"], bool), C.MARGIN)
                ji = IM.joint_ideal(ep, x, fq)
                g = IM.gyro_ideal(ep, x, ctx.calib, ctx.model, ctx.sid)
                res[tid, cond, r] = {"joint_ideal": ji["dev"], "lag_ms": ji["lag_ms"], "shape": ji["shape"],
                                     "joint_raw": got[0]["joint_raw"], "qc": got[2], "path": got[0]["path"], "speed": got[0]["speed"],
                                     "ee_ideal": float(np.sqrt(np.mean((np.asarray(ep["imu"])[:, 3:] - g)[fi] ** 2)))}
    full = [t for t in ids if all((t, c, r) in res for c in conds for r in range(1, ROUNDS + 1))]
    mean = lambda t, c: np.mean([[res[t, c, r][k] for k in KEYS] for r in range(1, ROUNDS + 1) if (t, c, r) in res], 0)   # noqa: E731
    print(f"{'':<12}" + "".join(f"{NAMES[k]:>16}" for k in KEYS))
    out = {"res": {f"{t} {c} {r}": v for (t, c, r), v in res.items()}, "median": {}, "compare": {}, "complete": full}
    for cond in conds:
        a = np.array([mean(t, cond) for t in ids if any((t, cond, r) in res for r in range(1, ROUNDS + 1))])
        if len(a):
            out["median"][cond] = dict(zip(KEYS, np.median(a, 0).tolist()))
            print(f"{cond:<12}" + "".join(f"{v:16.3f}" for v in np.median(a, 0)) + f"   （{len(a)} 条）")
    base = conds[0]
    for cond in conds[1:]:
        rel = {k: {t: [ST.rel_change(res[t, cond, r][k], res[t, base, r][k]) for r in range(1, ROUNDS + 1)] for t in full}
               for k in ("joint_ideal", "joint_raw", "ee_ideal", "path", "speed")}
        r2 = {k: ST.hier2(v) for k, v in rel.items()} if full else {}
        verdict = (ST.judge(r2, main="joint_ideal") if len(full) == len(ids)
                   else f"只报告、未完成（完整轨迹 {len(full)}/{len(ids)}）")
        out["compare"][cond] = {"vs": base, "hier2": r2, "verdict": verdict}
        print(f"\n{cond} 相对 {base}（{len(full)} 条完整轨迹，按轮配对，两级自助 95% 区间）：")
        for k, v in r2.items():
            print(f"  {NAMES[k]:<14} {v['effect']:+7.1f}%  [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]")
        print(f"  判定：{verdict}")
    (Path(sess) / f"{set_}_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("replay", "analyze"):
        sp = sub.add_parser(name)
        sp.add_argument("--set", required=True, choices=tuple(SETS))
        sp.add_argument("--session", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        return cmd_replay(a.set, Path(a.session))
    cmd_analyze(a.set, Path(a.session))
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
