"""B4 的回放评价（方案 `reports/PLAN_b4.md` §5.3 冻结前的验证集检查、§5.4 最终测试；实现计划任务 10）。

  python -u sim/b4_eval.py replay --set val  [--net 选中的权重或冻结文件] [--session 目录] [id …]   # 默认该集全部
  python -u sim/b4_eval.py replay --set test [--session 目录] id …                                     # 测试日：先跑这条的教师
  python -u sim/b4_eval.py analyze --set val|test [--net 同 replay]   # 只读这个网络跑的、整条轨迹在同一会话里跑齐的结果

条件：shift = 冻结的在线 Shift；net = Shift＋网络（指令由录制时记下的在线 Shift、窗与网络输入离线算好，与部署时逐位相同）；
teacher = 这一条自己的教师补正（测试集，只作参考）。回放时桌上放方块，摆在这一条录制时的位置，和闭环同条件。
判定只看关节偏差（两级自助：先抽轨迹、再抽轮次）加运动护栏；末端角速度等只报告。
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_stats as ST  # noqa: E402
import b4_teacher as B  # noqa: E402
import repeat_reversal as RR  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

EVAL = C.OUT / "eval"
CONDS = {"val": ("shift", "net"), "test": ("shift", "net", "teacher")}
ROUNDS = {"val": 2, "test": 4}
KEYS = ("joint_raw", "path", "speed", "ee_raw")


def net_command(log: dict, seq: dict, net, P: int, F: int, cfg) -> np.ndarray:
    """Shift＋网络的回放指令 (帧, 7) float32：每帧用录制时记下的在线 Shift 和窗，网络输入在 P、F 与记录相同时用记下的、
    否则由整段预测复算，再经 apply_correction（同部署）；开头补 HOLD 帧首帧保持。"""
    if P == int(log["P"]) and F == int(log["F"]):
        X = np.asarray(log["x"], np.float32)
    else:
        _, ens = O.te_states(log["chunk"])
        X = np.stack([O.online_input(np.asarray(log["a_pi"]), ens[k], k, P, F) for k in range(len(ens))])
    sh = np.asarray(log["shift"], np.float32)
    u = np.stack([O.apply_correction(sh[k], log["w"][k], net(X[k]), cfg) for k in range(len(sh))])
    return np.concatenate([np.repeat(u[:1], RR.HOLD, 0), u])


def plan(set_: str, ids: list[str]) -> list[tuple[str, str, int]]:
    """(id, 条件, 轮)：val 每条 2 轮 × {shift, net}，轮内先后交替；test 每条 4 轮 × 3 个条件，第 r 轮从第 (r − 1) mod 3 个条件开始。"""
    c = CONDS[set_]
    return [(i, x, r) for i in ids for r in range(1, ROUNDS[set_] + 1)
            for x in (c[(r - 1) % len(c):] + c[:(r - 1) % len(c)] if set_ == "test" else (c if r % 2 else c[::-1]))]


def analyze_results(res: dict, conds: tuple, n_rounds: int) -> dict:
    """res = {轨迹: {(条件, 轮): 指标 | None}}。规定轮次齐全的轨迹才进判定；各键「网络相对 Shift」逐轮相对变化 → 两级自助 → 只用关节偏差判定。
    另报：网络 / ILC 收益比 =（Shift − 网络）/（Shift − 教师）（关节偏差，各条中位数后平均）、教师相对 Shift 的变化。"""
    full = {t: d for t, d in res.items()
            if all(d.get((c, r)) is not None for c in conds for r in range(1, n_rounds + 1))}
    per = {t: {"status": "完整" if t in full else "未完成"} for t in res}
    rounds = range(1, n_rounds + 1)
    rel = lambda a, k: {t: [ST.rel_change(d[(a, r)][k], d[("shift", r)][k]) for r in rounds] for t, d in full.items()}   # noqa: E731
    out = {k: ST.hier2(rel("net", k)) if full else None for k in KEYS}
    out["verdict"] = (ST.judge(out) if len(full) == len(res)
                      else f"只报告、未完成（完整轨迹 {len(full)}/{len(res)}）：已有数据照常报告，不下判定")
    if "teacher" in conds and full:
        out["teacher_joint_raw"] = ST.hier2(rel("teacher", "joint_raw"))
        med = lambda t, c: np.median([full[t][(c, r)]["joint_raw"] for r in rounds])   # noqa: E731
        gains = [(med(t, "shift") - med(t, "net")) / (med(t, "shift") - med(t, "teacher")) for t in full]
        out["gain_ratio"] = float(np.mean(gains))
    out["per_traj"] = per
    return out


def _rid(tid: str, cond: str, r: int) -> str:
    return f"{tid}_e{cond}{r}"


def _ids(set_: str) -> list[str]:
    split = {p["id"]: p["split"] for p in P.load()["pool"]}
    return [i for i in P.load()["order"] if split[i] == set_]


def _net_sha(net_path: str) -> str:
    """评价用的网络的权重哈希：冻结文件（.json）里记的，或训练权重（.pt）本身的。"""
    if net_path.endswith(".json"):
        return json.loads(Path(net_path).read_text())["weights_sha"]
    return hashlib.sha256(Path(net_path).read_bytes()).hexdigest()[:16]


def pick_session(tid: str, sessions: list[Path], conds: tuple, n_rounds: int, net_sha: str) -> Path | None:
    """这条轨迹用哪个会话的结果：全部条件、全部轮次都在同一会话里跑过，且网络那几遍用的是 net_sha 这个网络；
    有几个就取最新的。没有 → None（不跨会话拼轮次，也不读别的网络的结果）。"""
    rids = [_rid(tid, c, r) for c in conds for r in range(1, n_rounds + 1)]
    for s in reversed(sessions):
        if all((s / f"replay_{rid}_r1.json").exists() for rid in rids) and all(
                json.loads(str(np.load(s / f"seq_{rid}.npz")["source"])).get("net_sha") == net_sha
                for rid in rids if rid.startswith(f"{tid}_enet")):
            return s
    return None


def cmd_replay(set_: str, sess: Path, ids: list[str], net_path: str) -> int:
    cfg, pos = O.load_cfg(C.CFG), {p["id"]: p["cm"] for p in P.load()["pool"]}
    rec = json.loads(P.REC.read_text())
    B.check_session(sess, C.CALIB)
    sess.mkdir(parents=True, exist_ok=True)
    if not (sess / "session.json").exists():
        (sess / "session.json").write_text(json.dumps({"calib_sha": B._sha(C.CALIB), "rule": B.RULE, "qc_max": B.QC_MAX}))
    if net_path and net_path.endswith(".json"):
        net, _, Pn, Fn = O.load_frozen(net_path)
    else:                                             # 冻结前（验证集检查）：选中的那次训练的权重
        import torch
        m = torch.load(net_path, map_location="cpu")
        mod = O.B4Net(m["n_in"], m["hidden"])
        mod.load_state_dict(m["state_dict"])
        mod.eval()
        net = lambda x: mod(torch.from_numpy(np.asarray(x, np.float32))[None])[0].detach().numpy()   # noqa: E731
        Pn, Fn = m["P"], m["F"]
    todo = plan(set_, ids or _ids(set_))
    for k, (tid, cond, r) in enumerate(todo, 1):
        q, rid = B.load_seq(tid), _rid(tid, cond, r)
        if cond == "shift":
            cmd7 = q["s7"]
        elif cond == "net":
            r_ = rec[tid]
            log = dict(np.load(C.T.LR / r_["repo"] / "high_rate" / f"episode_{r_['episode']:06d}_b4.npz"))
            cmd7 = net_command(log, q, net, Pn, Fn, cfg)
        else:
            cmd7 = np.load(sess / f"teacher_{tid}.npz")["cmd7"]
        seq = B.write_cmd(sess, rid, q["s7"], cmd7, f"评价 {cond}", {"net_sha": _net_sha(net_path)} if cond == "net" else None)
        if RR.attempted(rid, 1, sess):
            continue
        if input(f"\n[{k}/{len(todo)}] {rid}：方块摆在 {P.mark_desc(pos[tid])}（{pos[tid]:+g} cm），可以运动就按回车（输入 q 退出）"
                 ).strip().lower() == "q":
            return 0
        code = RR.run_child([sys.executable, "-u", str(HERE / "b4_teacher.py"), "run", "--session", str(sess), "--one", str(seq), rid])
        if code:
            print(f"  ⚠ {rid} 返回码 {code}：这一遍不补跑。检查机械臂后再运行（会从下一遍接着跑）。")
            return code
    print(f"=== {set_}：这一轮的回放都跑完了。下一步：python -u sim/b4_eval.py analyze --set {set_}")
    return 0


def cmd_analyze(set_: str, net_path: str) -> int:
    conds, ctxs, sha = CONDS[set_], {}, _net_sha(net_path)
    res = {}
    for tid in _ids(set_):
        q, d = B.load_seq(tid), {}
        s = pick_session(tid, B._sessions(), conds, ROUNDS[set_], sha)
        for c in conds:
            for r in range(1, ROUNDS[set_] + 1):
                rid = _rid(tid, c, r)
                if s is not None and s not in ctxs:
                    ctxs[s] = B.session_ctx(s)            # 每个会话用它自己那天的标定
                v = B.replay_result(rid, s, q, ctxs[s]) if s else None
                d[(c, r)] = None if v is None or v[2] > B.QC_MAX else v[0]
        res[tid] = d                                  # 预定的每一条都算，一遍没跑也算「未完成」
    out = analyze_results(res, conds, ROUNDS[set_])
    p = EVAL / set_ / "summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    j = out["joint_raw"]
    if j:
        print(f"关节偏差（相对原始策略指令）Shift＋网络 相对 Shift：{j['effect']:+.1f}% [{j['ci95'][0]:+.1f}, {j['ci95'][1]:+.1f}]，"
              f"完整轨迹 {j['n_groups']}/{len(res)}")
        print(f"护栏：路径长度 {out['path']['effect']:+.1f}%、关节速度 {out['speed']['effect']:+.1f}%；"
              f"只报告：末端角速度偏差 {out['ee_raw']['effect']:+.1f}%")
    if "gain_ratio" in out:
        print(f"Shift＋教师 相对 Shift：{out['teacher_joint_raw']['effect']:+.1f}%；网络拿到 ILC 收益的 {out['gain_ratio']:.0%}")
    print(f"判定：{out['verdict']} → {p}")
    return 0


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser(description="B4 的回放评价")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--set", required=True, choices=("val", "test"))
    r.add_argument("--session", default=None)
    r.add_argument("--net", default=None, help="val 默认 selection.json 选中的权重；test 用 outputs/b4/net/frozen.json")
    r.add_argument("ids", nargs="*")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--set", required=True, choices=("val", "test"))
    a_.add_argument("--net", default=None, help="同 replay：只读用这个网络跑的结果")
    a = ap.parse_args(argv)
    net = a.net or (str(C.OUT / "net/frozen.json") if a.set == "test" else
                    json.loads((C.OUT / "net/selection.json").read_text())["chosen"]["path"])
    if a.cmd == "analyze":
        return cmd_analyze(a.set, net)
    import datetime as dt
    sess = Path(a.session or B.TEACH / "sessions" / dt.date.today().strftime("%Y%m%d")).resolve()
    return cmd_replay(a.set, sess, a.ids, net)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
