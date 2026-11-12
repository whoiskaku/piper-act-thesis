"""B4 的教师（方案 `reports/PLAN_b4.md` §5.2；实现计划任务 6、7）。

  python -u sim/b4_teacher.py seq [id …]                 # 第二步录的闭环 Shift 轨迹 → 教师的回放序列（默认 recordings.json 里的全部）
  python -u sim/b4_teacher.py run [--session 目录] id …  # 真机：逐条跑教师（学习 12 遍 + 独立验收 6 遍），一条跑完再下一条
  python -u sim/b4_teacher.py run --dry-run id …         # 只打印排程、检查第一遍要发的序列，不连机械臂
  python -u sim/b4_teacher.py status                     # 各条教师的进度与验收结论 → outputs/b4/teacher/summary.json

教师 = 固定的 v0.10 学习流程（用户 10-29）：Shift 基准 2 遍 + 共同关节阶段 4 遍 + 加 IMU 支 6 遍（IMU 独有部分权重 ½），
试探、确认、退回都占这 12 遍；用完冻结最后接受的补正 Δ*，不要求收敛、不追加；再独立验收 3 轮（Shift / Shift＋Δ* 交替，
6 遍另计），验收数据不进任何更新。一条教师的 18 遍在同一会话（同一天、同一次 IMU 标定）里跑完。

回放序列 = 开头补 HOLD 帧首帧保持 + 录制的每一帧（同 repeat_reversal.episode_cmd）：
- a7 = 原始策略指令 a_π（评价与 ILC 误差的参考）；
- s7 = 录制时实发的在线 Shift，逐字节取自记录（教师围绕这个冻结的基准学习，约束 2）；
- w = 录制时在线算出并记下的窗（补正的形状，同部署时的窗，约束 1、2）；
- marked = 疑似接触帧（v0.10 口径，在最终指令上算），只用于评价口径与 ILC 误差，不进补正本身。
"""
from __future__ import annotations

import argparse
import datetime as dt
import functools
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import ilc_route1 as I  # noqa: E402
import repeat_reversal as RR  # noqa: E402
import shift_replay as S  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

TEACH = C.OUT / "teacher"
QC_MAX, RULE = 2.5, "b4-v1"                      # 逐遍朝向检查的停机门槛（°，同 v0.10）；规则版本
BUDGET = {"c": 4, "h": 6}
SCHEDULE = [("b", n) for n in (1, 2)] + [("c", n) for n in range(1, 5)] + [("h", n) for n in range(1, 7)]
ACCEPT = [("as", 1), ("at", 1), ("at", 2), ("as", 2), ("as", 3), ("at", 3)]   # as = Shift，at = Shift＋Δ*；先后交替


def build_seq(log: dict, act: np.ndarray, cfg: O.OnlineCfg, model) -> dict:
    """一集闭环 Shift 的记录（_b4.npz）+ 数据集动作 → 教师的回放序列 {"a7", "s7", "w", "lo", "hi", "marked", "n_unsafe"}。
    记录不能当冻结基准时停下：不是完整的 shift 模式；数据集动作与实发指令不同；由整段预测复算的 a_π、在线 Shift 与记录不符；
    Δ = 0 都不满足上下限。n_unsafe = 窗 > 0 但属于疑似接触帧的帧数（在线窗没盖住的接触，只报告）。"""
    if str(log["mode"]) != "shift" or not bool(log["complete"]):
        raise SystemExit(f"⛔ 记录不能用：模式 {log['mode']}、完整 {bool(log['complete'])}（教师要闭环 Shift 的完整一集）")
    u = np.asarray(log["u"], np.float32)
    gap = float(np.abs(np.asarray(act, np.float32) - u).max())
    if gap > 1e-6:
        raise SystemExit(f"⛔ 数据集动作与包装实发的指令不一致（最大差 {gap:.2e}）：发送时被改过，不能当冻结的 Shift 基准")
    a_pi, ens = O.te_states(log["chunk"])
    err = max(float(np.abs(a_pi - log["a_pi"]).max()),
              float(np.abs(np.stack([O.online_shift(e) for e in ens]) - u).max()))
    if err > 1e-5:
        raise SystemExit(f"⛔ 由整段预测复算的 a_π / 在线 Shift 与记录不符（最大差 {err:.2e}）")
    H = RR.HOLD
    a7 = np.concatenate([np.repeat(np.asarray(log["a_pi"], np.float32)[:1], H, 0), np.asarray(log["a_pi"], np.float32)])
    s7 = np.concatenate([np.repeat(u[:1], H, 0), u])
    w = np.concatenate([np.zeros((H, 5)), np.asarray(log["w"], float)])
    lo, hi = I.bounds(s7[:, :6].astype(float), np.asarray(cfg.jnt_lo) - O.HARD_SLACK, np.asarray(cfg.jnt_hi) + O.HARD_SLACK)
    if np.any(lo > 0) or np.any(hi < 0):
        raise SystemExit("⛔ Shift 指令离管线硬界不足 0.2°：Δ = 0 都不满足上下限")
    center = np.asarray(cfg.center)
    marked = RR.contact_frames(a7.astype(float), model, center) | RR.contact_frames(s7.astype(float), model, center)
    return {"a7": a7, "s7": s7, "w": w, "lo": lo, "hi": hi, "marked": marked,
            "n_unsafe": int(((w > 0).any(1) & marked).sum())}


def save_seq(path: Path, q: dict, source: dict) -> str:
    """写序列文件，返回内容哈希（a7、s7、w）；已存在且内容相同就跳过，不同就停（不覆盖）。"""
    h = hashlib.sha256(b"".join(np.ascontiguousarray(q[k]).tobytes() for k in ("a7", "s7", "w"))).hexdigest()[:16]
    path = Path(path)
    if path.exists():
        old = json.loads(str(np.load(path)["source"]))["hash"]
        if old != h:
            raise SystemExit(f"⛔ {path} 已存在且内容不同（{old} → {h}），不覆盖")
        return h
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: q[k] for k in ("a7", "s7", "w", "lo", "hi", "marked")},
             source=json.dumps({**source, "hash": h, "n_unsafe": q["n_unsafe"]}, ensure_ascii=False))
    return h


def cmd_seq(ids: list[str]) -> int:
    import mujoco
    import pandas as pd
    from ideal_imu_kinematic import MODEL_PATH
    rec = json.loads(P.REC.read_text())
    cfg, model = O.load_cfg(C.CFG), mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    rejected = []
    for tid in ids or sorted(rec):
        r = rec[tid]
        root, i = C.T.LR / r["repo"], r["episode"]
        logp = root / "high_rate" / f"episode_{i:06d}_b4.npz"
        try:
            act = np.stack(pd.read_parquet(root / f"data/chunk-000/episode_{i:06d}.parquet")["action"].to_numpy())
            q = build_seq(dict(np.load(logp)), act, cfg, model)
            h = save_seq(TEACH / tid / "seq_base.npz", q, {"id": tid, "repo": r["repo"], "episode": i, "cfg_sha": O.cfg_sha(C.CFG),
                                                            "log_sha": hashlib.sha256(logp.read_bytes()).hexdigest()[:16]})
        except SystemExit as e:
            rejected.append(tid)
            print(f"{tid}：{e}")
            continue
        print(f"{tid}：{r['repo']} 第 {i} 集，{len(q['s7'])} 帧（含开头保持 {RR.HOLD}），序列哈希 {h}，在线窗没盖住的接触帧 {q['n_unsafe']}")
    print(f"拒绝 {len(rejected)} 条：{' '.join(rejected) or '无'}")
    return 1 if rejected else 0


# ── 任务 7：教师运行与独立验收 ─────────────────────────────────────────────
def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def session_ctx(sess: Path, calib_dir: Path = C.CALIB.parent):
    """会话用的那份标定：按 session.json 记下的哈希，在 imu_calib*.json（含备份）里找；找不到就停（同 ilc_route1.calib_path）。"""
    want = json.loads((Path(sess) / "session.json").read_text())["calib_sha"]
    for p in sorted(Path(calib_dir).glob("imu_calib*.json")):
        if _sha(p) == want:
            return C.make_ctx(p)
    raise SystemExit(f"⛔ {calib_dir} 里没有会话 {Path(sess).name} 用的标定（哈希 {want}）")


def load_seq(tid: str) -> dict:
    z = np.load(TEACH / tid / "seq_base.npz")
    return {k: z[k] for k in ("a7", "s7", "w", "lo", "hi", "marked")}


def check_session(sess: Path, calib: Path) -> None:
    """IMU 标定必须是今天做的；会话已开始过的话，标定文件必须和开始时是同一份。"""
    if dt.date.fromtimestamp(Path(calib).stat().st_mtime) != dt.date.today():
        raise SystemExit("⛔ IMU 标定不是今天做的：先跑 python -u sim/imu_calib.py replay 和 fit")
    f = Path(sess) / "session.json"
    if f.exists() and json.loads(f.read_text())["calib_sha"] != _sha(calib):
        raise SystemExit("⛔ 标定文件和本会话开始时不同：同一会话不能混用两次标定")


def write_seq(sess: Path, rid: str, s7: np.ndarray, delta: np.ndarray, label: str) -> Path:
    """发出的指令 = 冻结的在线 Shift + Δ（同 ilc_route1.write_seq，写在会话目录里）。"""
    cmd7 = np.array(s7, np.float32)
    cmd7[:, list(O.ACTIVE)] += np.asarray(delta).astype(np.float32)
    return write_cmd(sess, rid, s7, cmd7, label)


def write_cmd(sess: Path, rid: str, s7: np.ndarray, cmd7: np.ndarray, label: str, meta: dict | None = None) -> Path:
    """把要发的整条指令写成回放序列：单帧跳变按回放管线同一公式、以 Shift 为裸序列检查；
    这一遍已开始过时，核对重建出的是同一份（不同就停）。meta 另记进来源（如评价时用的网络哈希）。"""
    cmd7 = np.asarray(cmd7, np.float32)
    jump, base = (np.abs(np.diff(x[:, :6].astype(float), axis=0)).max() for x in (cmd7, s7))
    if not jump < base * 1.5 + 0.006:
        raise SystemExit(f"⛔ {rid}：单帧跳变 {jump * 1e3:.1f} mrad 超过 Shift 的 1.5 倍 + 6 mrad（{base * 1e3:.1f} mrad）")
    h = hashlib.sha256(np.ascontiguousarray(cmd7, np.float32).tobytes()).hexdigest()[:16]   # 同 replay_probe
    p = Path(sess) / f"seq_{rid}.npz"
    if RR.attempted(rid, 1, Path(sess)):
        old = json.loads(str(np.load(p)["source"]))["cmd_hash"]
        if old != h:
            raise SystemExit(f"⛔ {rid}：重建出的序列与回放时的不同（{old} → {h}），代码或输入变了，不能接着跑")
    else:
        np.savez(p, cmd7=cmd7, da=np.zeros((len(cmd7), 6), np.float32),
                 source=json.dumps({"rid": rid, "label": label, "cmd_hash": h, **(meta or {})}, ensure_ascii=False))
    return p


def replay_result(rid: str, sess: Path, q: dict, ctx):
    """一遍回放 → (指标, (ē_q, ē_ω, ē_enc), 朝向偏差°)，同 ilc_route1.result（参考 = a7，疑似接触帧 = seq 的 marked）；
    无效遍（提前终止、哈希或帧数不符、没有编码器）返回 None。结果缓存在 res_{rid}.npz。"""
    cache = Path(sess) / f"res_{rid}.npz"
    if cache.exists():
        z = np.load(cache)
        if "calib_sha" in z.files and str(z["calib_sha"]) == ctx.calib_sha:      # 缓存只在同一份标定下复用
            return json.loads(str(z["met"])), (z["eq"], z["ew"], z["ee"]), float(z["qc"])
    h = json.loads((Path(sess) / f"replay_{rid}_r1.json").read_text())
    z = np.load(Path(sess) / f"seq_{rid}.npz")
    if not (h.get("stopped") is None and h.get("cmd_hash") == json.loads(str(z["source"]))["cmd_hash"]
            and len(h.get("t_send", [])) == len(z["cmd7"]) and h.get("can_q200")):
        return None
    ep = C.T.from_record(h, np.asarray(h["cmd"], float))
    a6, mk = q["a7"][:, :6].astype(float), q["marked"]
    fq, fi = S.masks(ep, mk, C.MARGIN)
    met = S.metrics(ep, a6, fq, fi, ctx.calib, ctx.model, ctx.sid)
    target = S.raw_gyro(ep, a6, ctx.calib, ctx.model, ctx.sid)
    M = lambda x: I.ang_err(ep, x, a6, mk, ctx.calib, ctx.model, ctx.sid)   # noqa: E731
    err = (I.joint_err(ep, a6, mk), M(target - ep["imu"][:, 3:]), M(target - I.enc_gyro(ep, ctx.calib, ctx.model, ctx.sid)))
    qc = I.orientation_deg(ep, ctx.calib)
    np.savez(cache, met=json.dumps(met), eq=err[0], ew=err[1], ee=err[2], qc=qc, calib_sha=ctx.calib_sha)
    return met, err, qc


def _learner(st: dict, lk: str, box: tuple):
    """需要时建学习器：共同关节阶段从两遍 Shift 基准出发（Δ = 0）；加 IMU 支（权重 ½）从共同阶段最后接受的补正出发。"""
    if lk not in BUDGET:
        return None
    if lk not in st:
        if lk == "c":
            if not st["b"]:
                raise SystemExit("⛔ 没有有效的 Shift 基准回放，不能开始共同关节阶段")
            m, e = I.mean_res([r for _, r in st["b"]])
            st["c"] = I.Learner("c", BUDGET["c"], np.zeros_like(e[0]), e, m, box, [rid for rid, _ in st["b"]], enc=False, wr=0.5)
        else:
            c = st["c"]
            st["h"] = I.Learner("h", BUDGET["h"], c.acc, c.err, c.met, box, c.hist[-1][0], enc=False, wr=0.5)
    return st[lk]


def accept_verdict(acc: dict, qc_max: float = QC_MAX) -> tuple[bool, list[str]]:
    """acc = {(as|at, 轮): (指标, 朝向偏差°) | None}。通过 = 6 遍都有效、朝向 ≤ qc_max；每轮 Shift＋Δ* 的关节偏差都比同轮 Shift 小；
    路径长度、关节速度相对同轮 Shift 都在 ±5% 内。"""
    bad = [f"{lk}{n} 无效（提前终止或记录不全）" for lk, n in ACCEPT if acc.get((lk, n)) is None]
    bad += [f"{lk}{n} 朝向偏差 {acc[(lk, n)][1]:.1f}°" for lk, n in ACCEPT if acc.get((lk, n)) and acc[(lk, n)][1] > qc_max]
    for r in (1, 2, 3):
        a, s = acc.get(("at", r)), acc.get(("as", r))
        if a is None or s is None:
            continue
        if not a[0]["joint_raw"] < s[0]["joint_raw"]:
            bad.append(f"第 {r} 轮关节偏差没有比 Shift 小（{a[0]['joint_raw']:.2f} / {s[0]['joint_raw']:.2f} mrad）")
        for key, name in (("path", "路径长度"), ("speed", "关节速度")):
            if abs(a[0][key] / s[0][key] - 1) > 0.05:
                bad.append(f"第 {r} 轮{name}变化超过 ±5%")
    return not bad, bad


def walk_teacher(tid: str, sess: Path, execute, result, q: dict | None = None) -> dict:
    """按排程把已有回放喂给学习器（同 ilc_route1.walk 的确定性重建）；遇到还没跑过的一遍就调 execute(序号, 总数, rid, 序列, 学习器)，
    它返回 True 表示跑完、接着往下，返回别的就以此为退出码停下（st["stopped"]）。第 12 遍之后冻结最后接受的补正（写 teacher_{tid}.npz），
    再跑验收：验收结果只进 st["accept"]，不调用任何学习器。result(rid) → (指标, 误差, 朝向) | None。"""
    q = load_seq(tid) if q is None else q
    sess = Path(sess)
    box, zero = (q["w"], q["lo"], q["hi"]), np.zeros((len(q["s7"]), len(O.ACTIVE)))
    st = {"tid": tid, "b": [], "learning": [], "accept": {}, "stopped": None}
    total = len(SCHEDULE) + len(ACCEPT)

    def run_one(k, rid, delta, label, L):
        seq = write_seq(sess, rid, q["s7"], delta, label)
        if not RR.attempted(rid, 1, sess):
            rc = 0 if execute is None else execute(k, total, rid, seq, L)
            if rc is not True:
                st["stopped"] = rc
                return "stop"
        return result(rid)

    for k, (lk, n) in enumerate(SCHEDULE, 1):
        rid = f"{tid}_{lk}{n:02d}"
        L = _learner(st, lk, box)
        cand = L.next() if L else None
        label = cand["label"] if cand else "Shift 基准"
        res = run_one(k, rid, cand["delta"] if cand else zero, label, L)
        if isinstance(res, str):
            return st
        st["learning"].append((rid, None if res is None else res[0]))
        if res is None:
            msg = "无效的一遍（提前终止或记录不全）：占预算、不补跑"
            if L:
                L.void()
        elif res[2] > QC_MAX:
            print(f"⛔ {rid}：IMU 朝向偏差 {res[2]:.1f}° > {QC_MAX}°：停下检查 IMU 有没有动（这条教师在本会话作废）")
            st["stopped"] = 3
            return st
        elif L:
            cand["rids"].append(rid)
            msg = L.add(cand, res[0], res[1]) + f"（余 {L.left} 遍）"
        else:
            st["b"].append((rid, res[:2]))
            msg = "记下"
        if execute is not None:
            vals = f"关节 {res[0]['joint_raw']:.2f} mrad · 朝向 {res[2]:.1f}° " if res else ""
            print(f"  [{k}/{total}] {rid} {label}：{vals}→ {msg}")

    st["delta"] = np.array(st["h"].acc)
    cmd7 = np.array(q["s7"], np.float32)
    cmd7[:, list(O.ACTIVE)] += st["delta"].astype(np.float32)
    h = hashlib.sha256(np.ascontiguousarray(cmd7).tobytes()).hexdigest()[:16]
    tp = sess / f"teacher_{tid}.npz"
    if tp.exists() and json.loads(str(np.load(tp)["source"]))["cmd_hash"] != h:
        raise SystemExit(f"⛔ {tid}：重建出的冻结补正和之前写下的不同，不能接着跑")
    if not tp.exists():
        np.savez(tp, delta=st["delta"], cmd7=cmd7, source=json.dumps({"tid": tid, "session": sess.name, "rule": RULE, "cmd_hash": h}))

    for k, (lk, n) in enumerate(ACCEPT, len(SCHEDULE) + 1):
        rid = f"{tid}_{lk}{n:02d}"
        res = run_one(k, rid, st["delta"] if lk == "at" else zero, "验收 Shift＋Δ*" if lk == "at" else "验收 Shift", None)
        if isinstance(res, str):
            return st
        st["accept"][(lk, n)] = None if res is None else (res[0], res[2])
        if execute is not None and res is not None:
            print(f"  [{k}/{total}] {rid}：关节 {res[0]['joint_raw']:.2f} mrad · 朝向 {res[2]:.1f}°（验收，不进学习）")
    st["verdict"] = accept_verdict(st["accept"])
    (sess / f"accept_{tid}.json").write_text(json.dumps(
        {"passed": st["verdict"][0], "reasons": st["verdict"][1], "session": sess.name,
         "accept": {f"{lk}{n}": None if v is None else {**{k: v[0][k] for k in ("joint_raw", "path", "speed")}, "qc": v[1]}
                    for (lk, n), v in st["accept"].items()}}, ensure_ascii=False, indent=1))
    return st


def _sessions() -> list[Path]:
    return sorted(p for p in (TEACH / "sessions").glob("*") if p.is_dir())


def next_ids(n: int, order: list[str], split: dict[str, str], done: set[str]) -> list[str]:
    """按录制顺序，训练、验证里还没跑完（没有验收结论）的前 n 条。测试集的教师在网络冻结后再跑；备用的只在需要补时手动指定。"""
    return [i for i in order if split[i] in ("train", "val") and i not in done][:n]


def cmd_run(sess: Path, ids: list[str], dry_run: bool, one=None) -> int:
    if one:                                           # 子进程：跑一遍（序列文件, rid）
        return S.guarded_run(Path(one[0]), f"{one[1]}_", "r1", sess)
    import replay_probe as RP
    if dry_run:
        for tid in ids:
            q = load_seq(tid)
            print(f"{tid}：{len(SCHEDULE)} 遍学习 + {len(ACCEPT)} 遍验收："
                  + " ".join(f"{lk}{n:02d}" for lk, n in SCHEDULE + ACCEPT))
            sess.mkdir(parents=True, exist_ok=True)
            seq = write_seq(sess, f"{tid}_b01", q["s7"], np.zeros((len(q["s7"]), 5)), "Shift 基准")
            print(f"  第一遍要发的序列 {seq.name}：回放管线的安全检查（不连机械臂）")
            RP.run("r1", True, seqfile=str(seq), tag=f"{tid}_b01_", outdir=sess)
        return 0
    check_session(sess, C.CALIB)
    sess.mkdir(parents=True, exist_ok=True)
    if not (sess / "session.json").exists():
        (sess / "session.json").write_text(json.dumps({"calib_sha": _sha(C.CALIB), "rule": RULE, "qc_max": QC_MAX,
                                                       "date": str(dt.date.today())}))
    ctx, rp_sha = C.make_ctx(), _sha(HERE / "replay_probe.py")

    def execute(k, n, rid, seq, _L):
        if input(f"\n[{k}/{n}] {rid}：桌面已清空、可以运动就按回车（输入 q 退出）").strip().lower() == "q":
            return 0
        utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        code = RR.run_child([sys.executable, "-u", __file__, "run", "--session", str(sess), "--one", str(seq), rid])
        with open(sess / "run_log.jsonl", "a") as f:
            f.write(json.dumps({"k": k, "rid": rid, "utc": utc, "code": code, "replay_probe_sha": rp_sha}) + "\n")
        if code:
            print(f"  ⚠ {rid} 返回码 {code}：这一遍不补跑。检查机械臂后再运行（会从下一遍接着跑）。")
            return code
        return True

    for tid in ids:
        done = [s for s in _sessions() if s != sess and (s / f"accept_{tid}.json").exists()]
        if done:
            print(f"{tid}：已在会话 {done[0].name} 跑完，跳过")
            continue
        q = load_seq(tid)
        st = walk_teacher(tid, sess, execute, functools.partial(replay_result, sess=sess, q=q, ctx=ctx), q)
        if st["stopped"] is not None:
            return st["stopped"] or 0
        ok, why = st["verdict"]
        print(f"=== {tid}：验收{'通过' if ok else '没通过：' + '；'.join(why)}")
    return 0


def cmd_status() -> int:
    """各条教师的进度与验收结论（扫描各会话目录）→ outputs/b4/teacher/summary.json。"""
    split = {p["id"]: p["split"] for p in P.load()["pool"]}
    rows = {}
    for s in _sessions():
        for f in s.glob("accept_*.json"):
            d = json.loads(f.read_text())
            rows[f.stem[len("accept_"):]] = {"split": split.get(f.stem[len("accept_"):]), **d}
        for f in s.glob("replay_*_b01_r1.json"):                  # 真跑过第一遍才算开始（dry-run 只写序列文件）
            tid = f.stem[len("replay_"):-len("_b01_r1")]
            rows.setdefault(tid, {"split": split.get(tid), "session": s.name, "passed": None, "reasons": ["进行中或中断"]})
    (TEACH / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for tid in sorted(rows):
        r = rows[tid]
        state = "通过" if r["passed"] else ("进行中或中断" if r["passed"] is None else "没通过：" + "；".join(r["reasons"]))
        print(f"{tid}（{r['split']}，会话 {r['session']}）：{state}")
    n = {k: sum(1 for r in rows.values() if r["passed"] is k) for k in (True, False, None)}
    print(f"通过 {n[True]}、没通过 {n[False]}、进行中或中断 {n[None]} → {TEACH / 'summary.json'}")
    return 0


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser(description="B4 的教师")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seq", help="生成教师的回放序列")
    s.add_argument("ids", nargs="*")
    r = sub.add_parser("run", help="真机跑教师（--dry-run 只检查）")
    r.add_argument("--session", default=str(TEACH / "sessions" / dt.date.today().strftime("%Y%m%d")))
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--one", nargs=2, help=argparse.SUPPRESS)      # 内部用：子进程跑一遍（序列文件, rid）
    r.add_argument("--next", type=int, default=0, help="自动挑下 N 条还没跑完的训练、验证教师")
    r.add_argument("ids", nargs="*")
    sub.add_parser("status", help="各条教师的进度与验收结论")
    a = ap.parse_args(argv)
    if a.cmd == "seq":
        return cmd_seq(a.ids)
    if a.cmd == "run":
        ids = list(a.ids)
        if a.next:
            d = P.load()
            done = {f.stem[len("accept_"):] for s in _sessions() for f in s.glob("accept_*.json")}
            ids += next_ids(a.next, d["order"], {p["id"]: p["split"] for p in d["pool"]}, done)
            print(f"这次跑：{' '.join(ids) or '（训练、验证都跑完了）'}")
        return cmd_run(Path(a.session).resolve(), ids, a.dry_run, a.one)
    return cmd_status()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)      # 跳过解释器收尾（关 GC 的进程在退出阶段也会崩）；回放子进程因此不触发机器人析构里的无插值归位
