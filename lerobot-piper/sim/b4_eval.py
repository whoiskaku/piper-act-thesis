"""B4 的回放评价（方案 `reports/PLAN_b4.md` §5.3 冻结前的验证集检查、§5.4 最终测试；实现计划任务 10）。

  python -u sim/b4_eval.py replay --set val  [--net 选中的权重或冻结文件] [--session 目录] [id …]   # 默认该集全部
  python -u sim/b4_eval.py replay --set test [--session 目录] id …                                     # 测试日：先跑这条的教师
  python -u sim/b4_eval.py analyze --set val|test [--net 同 replay]   # 只读这个网络跑的、整条轨迹在同一会话里跑齐的结果
  python -u sim/b4_eval.py replay --set valseq --session 目录 [--seq-net …] [id …]   # 整段版 ①（reports/PLAN_b4_seq.md §4.5）
  python -u sim/b4_eval.py analyze --set valseq [--seq-net …]

条件：shift = 冻结的在线 Shift；net = Shift＋网络（指令由录制时记下的在线 Shift、窗与网络输入离线算好，与部署时逐位相同）；
teacher = 这一条自己的教师补正（测试集，只作参考）；seq = 整段版 ①（网络直接输出整段指令、只发第一帧，同样经 apply_correction）。
回放时桌上放方块，摆在这一条录制时的位置，和闭环同条件。valseq 只有 shift、seq 两个条件（用户 11-06），回放编号用前缀 q，
与 val、test 的 e 分开；与一帧版的比较用 val 的已有结果（跨会话，只作粗比）。
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
CONDS = {"val": ("shift", "net"), "test": ("shift", "net", "teacher"), "valseq": ("shift", "seq")}
ROUNDS = {"val": 2, "test": 4, "valseq": 2}
PREFIX = {"val": "e", "test": "e", "valseq": "q"}  # 回放编号前缀：valseq 另起，旧的 val 分析不会读到新会话
KEYS = ("joint_raw", "path", "speed", "ee_raw")


def _inputs(log: dict, P: int, F: int) -> np.ndarray:
    """逐帧网络输入：P、F 与记录相同时用记下的，否则由整段预测复算（只用第 k 帧当时已有的 chunk）。"""
    if P == int(log["P"]) and F == int(log["F"]):
        return np.asarray(log["x"], np.float32)
    _, ens = O.te_states(log["chunk"])
    return np.stack([O.online_input(np.asarray(log["a_pi"]), ens[k], k, P, F) for k in range(len(ens))])


def net_command(log: dict, seq: dict, net, P: int, F: int, cfg) -> np.ndarray:
    """Shift＋网络的回放指令 (帧, 7) float32：每帧用录制时记下的在线 Shift 和窗，网络输入见 `_inputs`，
    再经 apply_correction（同部署）；开头补 HOLD 帧首帧保持。"""
    X = _inputs(log, P, F)
    sh = np.asarray(log["shift"], np.float32)
    u = np.stack([O.apply_correction(sh[k], log["w"][k], net(X[k]), cfg) for k in range(len(sh))])
    return np.concatenate([np.repeat(u[:1], RR.HOLD, 0), u])


def seq_delta(log: dict, net, P: int, F: int, N: int) -> np.ndarray:
    """整段版 ① 每帧实发的等效补正 (帧, 5)：网络输出的第一帧指令（a_π(k) + 第一帧输出）− 录制时的在线 Shift。"""
    X, act = _inputs(log, P, F), list(O.ACTIVE)
    a_pi, sh = np.asarray(log["a_pi"], float), np.asarray(log["shift"], float)
    return np.stack([a_pi[k, act] + np.asarray(net(X[k])).reshape(N, -1)[0] - sh[k, act] for k in range(len(X))])


def seq_command(log: dict, net, P: int, F: int, N: int, cfg) -> np.ndarray:
    """整段版 ① 的回放指令 (帧, 7) float32：每帧只发第一帧，等效补正经 apply_correction（窗门、限幅、离硬界，同一帧版）；
    开头补 HOLD 帧首帧保持。"""
    d, sh = seq_delta(log, net, P, F, N), np.asarray(log["shift"], np.float32)
    u = np.stack([O.apply_correction(sh[k], log["w"][k], d[k], cfg) for k in range(len(sh))])
    return np.concatenate([np.repeat(u[:1], RR.HOLD, 0), u])


def plan(set_: str, ids: list[str]) -> list[tuple[str, str, int]]:
    """(id, 条件, 轮)：val 每条 2 轮 × {shift, net}，轮内先后交替；test 每条 4 轮 × 3 个条件，第 r 轮从第 (r − 1) mod 3 个条件开始。"""
    c = CONDS[set_]
    return [(i, x, r) for i in ids for r in range(1, ROUNDS[set_] + 1)
            for x in (c[(r - 1) % len(c):] + c[:(r - 1) % len(c)] if set_ == "test" else (c if r % 2 else c[::-1]))]


def analyze_results(res: dict, conds: tuple, n_rounds: int, judged: str = "net") -> dict:
    """res = {轨迹: {(条件, 轮): 指标 | None}}。规定轮次齐全的轨迹才进判定；各键「judged 相对 Shift」逐轮相对变化 → 两级自助 → 只用关节偏差判定。
    另报：网络 / ILC 收益比 =（Shift − 网络）/（Shift − 教师）（关节偏差，各条中位数后平均）、教师相对 Shift 的变化。"""
    full = {t: d for t, d in res.items()
            if all(d.get((c, r)) is not None for c in conds for r in range(1, n_rounds + 1))}
    per = {t: {"status": "完整" if t in full else "未完成"} for t in res}
    rounds = range(1, n_rounds + 1)
    rel = lambda a, k: {t: [ST.rel_change(d[(a, r)][k], d[("shift", r)][k]) for r in rounds] for t, d in full.items()}   # noqa: E731
    out = {k: ST.hier2(rel(judged, k)) if full else None for k in KEYS}
    out["verdict"] = (ST.judge(out) if len(full) == len(res)
                      else f"只报告、未完成（完整轨迹 {len(full)}/{len(res)}）：已有数据照常报告，不下判定")
    if "teacher" in conds and full:
        out["teacher_joint_raw"] = ST.hier2(rel("teacher", "joint_raw"))
        med = lambda t, c: np.median([full[t][(c, r)]["joint_raw"] for r in rounds])   # noqa: E731
        gains = [(med(t, "shift") - med(t, "net")) / (med(t, "shift") - med(t, "teacher")) for t in full]
        out["gain_ratio"] = float(np.mean(gains))
    out["per_traj"] = per
    return out


def _rid(tid: str, cond: str, r: int, prefix: str = "e") -> str:
    return f"{tid}_{prefix}{cond}{r}"


def _ids(set_: str) -> list[str]:
    split = {p["id"]: p["split"] for p in P.load()["pool"]}
    return [i for i in P.load()["order"] if split[i] == {"valseq": "val"}.get(set_, set_)]


def _log(tid: str, rec: dict) -> dict:
    """这一条录制时的每集记录（整段预测、在线 Shift、窗、网络输入）。"""
    r = rec[tid]
    return dict(np.load(C.T.LR / r["repo"] / "high_rate" / f"episode_{r['episode']:06d}_b4.npz"))


def _load_run(path: str):
    """训练权重（.pt，冻结前评价用）→ (CPU 上的推理函数 np → np, P, F, n_out)。"""
    import torch
    m = torch.load(path, map_location="cpu")
    mod = O.B4Net(m["n_in"], m["hidden"], m.get("n_out", len(O.ACTIVE)), m.get("rel_P"))
    mod.load_state_dict(m["state_dict"])
    mod.eval()
    fn = lambda x: mod(torch.from_numpy(np.asarray(x, np.float32))[None])[0].detach().numpy()   # noqa: E731
    return fn, m["P"], m["F"], m.get("n_out", len(O.ACTIVE))


def _net_sha(net_path: str) -> str:
    """评价用的网络的权重哈希：冻结文件（.json）里记的，或训练权重（.pt）本身的。"""
    if net_path.endswith(".json"):
        return json.loads(Path(net_path).read_text())["weights_sha"]
    return hashlib.sha256(Path(net_path).read_bytes()).hexdigest()[:16]


def pick_session(tid: str, sessions: list[Path], conds: tuple, n_rounds: int, shas: dict[str, str],
                 prefix: str = "e") -> Path | None:
    """这条轨迹用哪个会话的结果：全部条件、全部轮次都在同一会话里跑过，且有网络的条件（shas 的键）那几遍用的都是指定的网络；
    有几个就取最新的。没有 → None（不跨会话拼轮次，也不读别的网络的结果）。"""
    rids = {(c, r): _rid(tid, c, r, prefix) for c in conds for r in range(1, n_rounds + 1)}
    for s in reversed(sessions):
        if all((s / f"replay_{rid}_r1.json").exists() for rid in rids.values()) and all(
                json.loads(str(np.load(s / f"seq_{rid}.npz")["source"])).get("net_sha") == shas[c]
                for (c, _), rid in rids.items() if c in shas):
            return s
    return None


def cmd_replay(set_: str, sess: Path, ids: list[str], net_path: str, seq_path: str | None = None) -> int:
    cfg, pos = O.load_cfg(C.CFG), {p["id"]: p["cm"] for p in P.load()["pool"]}
    rec = json.loads(P.REC.read_text())
    B.check_session(sess, C.CALIB)
    sess.mkdir(parents=True, exist_ok=True)
    if not (sess / "session.json").exists():
        (sess / "session.json").write_text(json.dumps({"calib_sha": B._sha(C.CALIB), "rule": B.RULE, "qc_max": B.QC_MAX}))
    shas = {c: _net_sha(p) for c, p in (("net", net_path), ("seq", seq_path)) if c in CONDS[set_]}
    if "net" in shas and net_path.endswith(".json"):
        net, _, Pn, Fn = O.load_frozen(net_path)
    elif "net" in shas:                               # 冻结前（验证集检查）：选中的那次训练的权重
        net, Pn, Fn, _ = _load_run(net_path)
    if "seq" in shas:                                 # valseq：整段版 ①（冻结前，选中的那次训练的权重）
        seq_net, Ps, Fs, n_out = _load_run(seq_path)
    todo = plan(set_, ids or _ids(set_))
    for k, (tid, cond, r) in enumerate(todo, 1):
        q, rid = B.load_seq(tid), _rid(tid, cond, r, PREFIX[set_])
        if cond == "shift":
            cmd7 = q["s7"]
        elif cond == "net":
            cmd7 = net_command(_log(tid, rec), q, net, Pn, Fn, cfg)
        elif cond == "seq":
            cmd7 = seq_command(_log(tid, rec), seq_net, Ps, Fs, n_out // len(O.ACTIVE), cfg)
        else:
            cmd7 = np.load(sess / f"teacher_{tid}.npz")["cmd7"]
        seq = B.write_cmd(sess, rid, q["s7"], cmd7, f"评价 {cond}", {"net_sha": shas[cond]} if cond in shas else None)
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


def cmd_analyze(set_: str, net_path: str, seq_path: str | None = None) -> int:
    conds, ctxs = CONDS[set_], {}
    shas = {c: _net_sha(p) for c, p in (("net", net_path), ("seq", seq_path)) if c in conds}
    res = {}
    for tid in _ids(set_):
        q, d = B.load_seq(tid), {}
        s = pick_session(tid, B._sessions(), conds, ROUNDS[set_], shas, PREFIX[set_])
        for c in conds:
            for r in range(1, ROUNDS[set_] + 1):
                rid = _rid(tid, c, r, PREFIX[set_])
                if s is not None and s not in ctxs:
                    ctxs[s] = B.session_ctx(s)            # 每个会话用它自己那天的标定
                v = B.replay_result(rid, s, q, ctxs[s]) if s else None
                d[(c, r)] = None if v is None or v[2] > B.QC_MAX else v[0]
        res[tid] = d                                  # 预定的每一条都算，一遍没跑也算「未完成」
    out = analyze_results(res, conds, ROUNDS[set_], "seq" if "seq" in conds else "net")
    if "seq" in conds:                                # 实发里限幅触发了多少：窗 > 0 的帧，逐关节，各条合计（只报告）
        seq_net, Ps, Fs, n_out = _load_run(seq_path)
        rec, hit, on = json.loads(P.REC.read_text()), 0, 0
        for tid in _ids(set_):
            log = _log(tid, rec)
            w = np.asarray(log["w"]) > 0
            hit = hit + ((np.abs(seq_delta(log, seq_net, Ps, Fs, n_out // len(O.ACTIVE))) > O.CLIP) & w).sum(0)
            on = on + w.sum(0)
        out["seq_clip_rate"] = (hit / on).tolist()
    p = EVAL / set_ / "summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    j = out["joint_raw"]
    if j:
        name = "整段版 ① 相对 Shift" if "seq" in conds else "Shift＋网络 相对 Shift"
        print(f"关节偏差（相对原始策略指令）{name}：{j['effect']:+.1f}% [{j['ci95'][0]:+.1f}, {j['ci95'][1]:+.1f}]，"
              f"完整轨迹 {j['n_groups']}/{len(res)}")
        print(f"护栏：路径长度 {out['path']['effect']:+.1f}%、关节速度 {out['speed']['effect']:+.1f}%；"
              f"只报告：末端角速度偏差 {out['ee_raw']['effect']:+.1f}%")
    if "seq_clip_rate" in out:
        print("只报告：整段版实发限幅触发率（逐关节）" + " ".join(f"{x:.2%}" for x in out["seq_clip_rate"]))
    if "gain_ratio" in out:
        print(f"Shift＋教师 相对 Shift：{out['teacher_joint_raw']['effect']:+.1f}%；网络拿到 ILC 收益的 {out['gain_ratio']:.0%}")
    print(f"判定：{out['verdict']} → {p}")
    return 0


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser(description="B4 的回放评价")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--set", required=True, choices=("val", "test", "valseq"))
    r.add_argument("--session", default=None)
    r.add_argument("--net", default=None, help="val 默认 selection.json 选中的权重；test 用 outputs/b4/net/frozen.json")
    r.add_argument("--seq-net", default=None, help="valseq：整段版 ① 的权重（默认 outputs/b4/net_seq/selection.json 选中的）")
    r.add_argument("ids", nargs="*")
    a_ = sub.add_parser("analyze")
    a_.add_argument("--set", required=True, choices=("val", "test", "valseq"))
    a_.add_argument("--net", default=None, help="同 replay：只读用这个网络跑的结果")
    a_.add_argument("--seq-net", default=None, help="同 replay")
    a = ap.parse_args(argv)
    net = a.net or (str(C.OUT / "net/frozen.json") if a.set == "test" else
                    json.loads((C.OUT / "net/selection.json").read_text())["chosen"]["path"])
    seq = (a.seq_net or json.loads((C.OUT / "net_seq/selection.json").read_text())["chosen"]["path"]
           if a.set == "valseq" else None)
    if a.cmd == "analyze":
        return cmd_analyze(a.set, net, seq)
    import datetime as dt
    sess = Path(a.session or B.TEACH / "sessions" / dt.date.today().strftime("%Y%m%d")).resolve()
    return cmd_replay(a.set, sess, a.ids, net, seq)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
