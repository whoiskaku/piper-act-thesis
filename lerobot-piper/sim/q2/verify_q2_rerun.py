"""阶段二正式运行的同种子复跑核对与驱动（用户 2026-10-04 决定：降并行、每个作业跑两遍核对；出错就重跑到成功）。

背景：本机（i9-9900X，X299，非 ECC 内存）高负载下出现内存破坏 —— 崩溃，以及一次静默漏读数据（被名单核对拦下）。
已跑完的作业也可能被悄悄改了数值，而原有核验只核对哈希、查不出。作业是确定性的（CPU 单线程、同种子逐位复现），
所以用同一份冻结代码、同样的上游产物把每个正式作业再跑一遍，逐项比较确定性内容，只采用两遍一致的结果。
不改冻结代码；本文件不在代码指纹里。

驱动（`run`）按阶段依次：
1. 补齐该阶段的正式作业（失败就重跑；每次尝试的日志与退出码留在 `正式目录/logs/attempt_NNN/`）；
2. 在核对目录 `VER/<阶段>/`（先 prepare，再从正式目录复制该阶段的上游产物）里把每个正式作业再跑一遍；
3. 比较确定性内容的摘要：
   - 正式 = 核对 ⇒ 一致；
   - 否则看全部已有的各遍（含存档的）：某个结果已出现两遍 ⇒ 以它为准 ——
     是正式那遍的结果 ⇒ 一致（本遍核对出错，存档）；不是 ⇒ 正式那遍出错：正式产物移到 `正式目录/_corrupted/`，
     回到 1 重跑正式作业（下游会因输入指纹变化自动重排）；
   - 还没有两遍一致 ⇒ 存档本遍核对，再跑一遍；超过 max_runs 遍仍对不上 ⇒ 停止并报告（不是偶发错误）。
全部阶段通过后跑正式汇总与作图。退出码 2（未冻结）、3（截止时间 / 未完成）、4（核验失败）不重跑，立即停止。
比较的确定性内容：参照与终训的逐集分数；内层表每个检查点的 ρ 与权重哈希；终训的 V、step*、边界标记、权重哈希与预测数组。
不比耗时、时间戳与 .pt / .npz 的文件字节哈希（两者本身不逐字节确定）。比较前先核对两边的输入指纹相同（同样的上游产物）。

    python -u sim/q2/verify_q2_rerun.py run --out outputs/q2_stage2 --ver outputs/q2_stage2_verify --jobs 4
    python -u sim/q2/verify_q2_rerun.py status --out outputs/q2_stage2 --ver outputs/q2_stage2_verify
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent / "q1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import q2_report as REP        # noqa: E402  （导入时不加载 torch / mujoco / scipy）
import q2_run as RUN           # noqa: E402

PHASES = RUN.PHASES
FINALS = ("nom-final", "inc-final")
RUN_KINDS = ("prepare",) + PHASES
STOP_RCS = (2, 3, 4)
MAX_RUNS = 6
UPSTREAM = {"refs": (), "nom-inner": (), "nom-final": ("nom/inner/seed*/*.json",),
            "inc-inner": ("nom/final/seed*/*.json", "nom/final/seed*/*.pt"),
            "inc-final": ("nom/final/seed*/*.json", "nom/final/seed*/*.pt", "inc/*/inner/seed*/*.json")}
FIELDS = {"refs": ("episodes",), "nom-inner": ("scores",), "inc-inner": ("scores",),
          "nom-final": ("V", "step_star", "boundary", "state_sha256", "q1", "zero", "H", "aux_zero", "aux_H",
                        "aux_H_beta"),
          "inc-final": ("V", "step_star", "boundary", "state_sha256", "cand", "aux_cand", "aux_beta")}


# ── 作业与摘要 ────────────────────────────────────────────────────────────────
def keystr(key) -> str:
    return "|".join(str(x) for x in key)


def args_of(key) -> list:
    return [str(x) for x in key[1:]]


def phase_keys(out, phase) -> list:
    return [k for k in REP.artifacts(out) if k[0] == phase]


def files_of(out, key) -> list:
    j = REP.artifacts(out)[key]
    return [j] + ([j.with_suffix(".pt"), j.with_suffix(".npz")] if key[0] in FINALS else [])


def digest(out, key):
    """确定性内容的摘要；文件缺失或无法解析 ⇒ None。"""
    j = RUN._load(REP.artifacts(out)[key])
    if j is None:
        return None
    h = hashlib.sha256(json.dumps({k: j.get(k) for k in FIELDS[key[0]]}, sort_keys=True).encode())
    if key[0] in FINALS:
        try:
            with np.load(REP.artifacts(out)[key].with_suffix(".npz"), allow_pickle=False) as z:
                for name in sorted(z.files):
                    a = np.ascontiguousarray(z[name])
                    for part in (name, str(a.dtype), str(a.shape)):
                        h.update(part.encode())
                    h.update(a.tobytes())
        except (OSError, ValueError, zipfile.BadZipFile):
            return None
    return h.hexdigest()


def formal_done(out, phase) -> set:
    todo = {tuple(a) for a in RUN.jobs(out, phase)}
    return {k for k in phase_keys(out, phase) if tuple(args_of(k)) not in todo}


# ── 执行：每次尝试一个日志与退出码；失败重跑 ─────────────────────────────────────
class Done:
    """已结束的作业句柄（进程内启动器用）。"""

    def __init__(self, rc: int):
        self.rc = rc

    def poll(self):
        return self.rc


class _Proc:
    def __init__(self, p, log, status: Path, task, k: int):
        self.p, self.log, self.status, self.task, self.k, self.done = p, log, status, task, k, False

    def poll(self):
        rc = self.p.poll()
        if rc is not None and not self.done:
            self.done = True
            self.log.close()
            kind, _, args = self.task
            with open(self.status, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{kind}\t{' '.join(args)}\t{rc}\t{self.k}\n")
        return rc


class Launcher:
    """子进程启动器：每次尝试写独立日志 `<kind>_<参数>_try<k>.log`，结束时在 status.tsv 记一行（时间、阶段、参数、退出码、第几次）。
    核对目录（ver_root 之下）的作业写进 verify_logdir，其余写进 logdir。"""

    def __init__(self, logdir, scripts=None, verify_logdir=None, ver_root=None):
        self.logdir, self.verify_logdir = Path(logdir), Path(verify_logdir) if verify_logdir else None
        self.ver_root = Path(ver_root).resolve() if ver_root else None
        self.scripts = {**{k: HERE / "q2_run.py" for k in RUN_KINDS}, "report": HERE / "q2_report.py",
                        "fig": HERE / "fig_q2.py", **(scripts or {})}

    def _dir(self, out) -> Path:
        if self.ver_root is not None and self.verify_logdir is not None:
            try:
                Path(out).resolve().relative_to(self.ver_root)
                return self.verify_logdir
            except ValueError:
                pass
        return self.logdir

    def start(self, task, k: int):
        kind, out, args = task
        d = self._dir(out)
        d.mkdir(parents=True, exist_ok=True)
        name = kind + ("_" + "_".join(args) if args else "") + f"_try{k}.log"
        log = open(d / name, "w")
        cmd = [sys.executable, "-u", str(self.scripts[kind])] + ([kind] if kind in RUN_KINDS else []) + \
              ["--out", str(out), *args]
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(RUN.ROOT))
        return _Proc(p, log, d / "status.tsv", task, k)


def run_jobs(tasks, start, jobs: int = 4, tries: int = 6, poll: float = 1.0) -> list:
    """并行执行 tasks = [(kind, out, args)]；退出码非 0 就重跑，最多 tries 次；退出码 2/3/4 不重跑，且不再启动新任务。
    → 仍失败的 [(task, 最后退出码)]（空 = 全部成功）。"""
    pending, running, failed, stop = [(t, 1) for t in tasks], [], [], False
    while (pending and not stop) or running:
        while pending and not stop and len(running) < jobs:
            t, k = pending.pop(0)
            running.append((t, k, start(t, k)))
        still = []
        for t, k, h in running:
            rc = h.poll()
            if rc is None:
                still.append((t, k, h))
            elif rc != 0:
                if rc in STOP_RCS:
                    failed.append((t, rc))
                    stop = True
                elif k < tries:
                    pending.append((t, k + 1))
                else:
                    failed.append((t, rc))
        running = still
        if running:
            time.sleep(poll)
    return failed


def new_attempt(logs: Path) -> Path:
    logs.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        d = logs / f"attempt_{n:03d}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            n += 1


# ── 核对目录与判定 ────────────────────────────────────────────────────────────
def load_state(ver) -> dict:
    j = RUN._load(Path(ver) / "state.json")
    return j if isinstance(j, dict) else {}


def save_state(ver, state: dict) -> None:
    p = Path(ver) / "state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + ".tmp")
    t.write_text(json.dumps(state, indent=1, ensure_ascii=False))
    os.replace(t, p)


def sync_upstream(out, d, phase) -> None:
    """把正式目录里该阶段的上游产物同步到核对目录（内容不同才复制；正式目录里已不存在的副本删掉）。"""
    out, d = Path(out), Path(d)
    want = {s.relative_to(out) for pat in UPSTREAM[phase] for s in out.glob(pat)}
    have = {s.relative_to(d) for pat in UPSTREAM[phase] for s in d.glob(pat)}
    for rel in have - want:
        (d / rel).unlink()
    for rel in want:
        src, dst = out / rel, d / rel
        if not dst.exists() or RUN.file_sha(dst) != RUN.file_sha(src):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _archive(root, key, dest: Path) -> None:
    files = [p for p in files_of(root, key) if p.exists()]
    if not files:
        return
    n = 1
    while (dest / str(n)).exists():
        n += 1
    (dest / str(n)).mkdir(parents=True)
    for p in files:
        shutil.move(str(p), str(dest / str(n) / p.name))


def _record(st: dict, who: str, at, d: str) -> None:
    if not any(r["who"] == who and r["at"] == at for r in st["runs"]):
        st["runs"].append({"who": who, "at": at, "digest": d})


def assess(out, ver, phase, state: dict, max_runs: int = MAX_RUNS) -> dict:
    """比较该阶段每个已完成的正式作业与其核对副本，按模块说明的规则更新 state。
    → {"need": 需再跑核对的作业, "rerun_formal": 正式产物已移走待重跑的作业, "unstable": 停止的作业}"""
    out, d = Path(out), Path(ver) / phase
    done = formal_done(out, phase)
    res = {"need": [], "rerun_formal": [], "unstable": []}
    for key in phase_keys(out, phase):
        ks = keystr(key)
        st = state.setdefault(ks, {"runs": [], "status": "pending"})
        if st["status"] == "unstable":
            res["unstable"].append(key)
            continue
        if key not in done:
            st["status"] = "formal-pending"
            continue
        fj, fd = RUN._load(REP.artifacts(out)[key]), digest(out, key)
        _record(st, "formal", fj.get("finished_at"), fd)
        vj = RUN._load(REP.artifacts(d)[key])
        vd = digest(d, key) if vj is not None else None
        if vj is None or vd is None or vj.get("inputs") != fj.get("inputs"):
            st["status"] = "pending"                                  # 还没核对，或上游不同（陈旧）⇒ 再跑核对
            res["need"].append(key)
            continue
        _record(st, "verify", vj.get("finished_at"), vd)
        if fd == vd:
            st["status"] = "match"
            continue
        counts = Counter(r["digest"] for r in st["runs"])
        agreed = [x for x, c in counts.items() if c >= 2]
        n_formal = sum(r["who"] == "formal" for r in st["runs"])
        if len(agreed) > 1:                                           # 两种结果各有两遍：不是偶发错误
            st["status"] = "unstable"
        elif fd in agreed:                                            # 正式结果已有两遍印证；本遍核对出错
            _archive(d, key, Path(ver) / "_archive" / ks.replace("|", "__"))
            st["status"] = "match"
        elif agreed:                                                  # 另有两遍一致的结果 ⇒ 正式那遍出错（最多重跑正式两次）
            if n_formal >= 3:
                st["status"] = "unstable"
            else:
                _archive(out, key, out / "_corrupted" / ks.replace("|", "__"))
                st["status"] = "rerun-formal"
                res["rerun_formal"].append(key)
        elif len(st["runs"]) >= max_runs:                             # 跑满 max_runs 遍仍没有两遍一致
            st["status"] = "unstable"
        else:
            _archive(d, key, Path(ver) / "_archive" / ks.replace("|", "__"))
            st["status"] = "pending"
            res["need"].append(key)
        if st["status"] == "unstable":
            res["unstable"].append(key)
    return res


# ── 驱动 ───────────────────────────────────────────────────────────────────
def run(out, ver, launcher, jobs: int = 4, tries: int = 6, max_runs: int = MAX_RUNS, max_rounds: int = 8,
        poll: float = 1.0, say=print) -> int:
    """→ 0 全部两遍一致且汇总完成；1 某作业重跑 tries 次仍失败；2 不稳定 / 未冻结 / 轮数用尽；3 截止时间 / 未完成；4 核验失败。"""
    out, ver = Path(out), Path(ver)
    state = load_state(ver)

    def go(tasks) -> int:
        failed = run_jobs(tasks, launcher.start, jobs=jobs, tries=tries, poll=poll)
        for (kind, o, args), rc in failed:
            say(f"  ✗ {kind} {' '.join(args)}（{o}）退出码 {rc}", flush=True)
        rcs = {rc for _, rc in failed}
        return 0 if not failed else next((r for r in (3, 4, 2) if r in rcs), 1)

    rc = go([("prepare", out, [])])
    if rc:
        return rc
    for phase in PHASES:
        for _round in range(max_rounds):
            todo = RUN.jobs(out, phase)
            if todo:
                say(f"[{phase}] 正式作业 {len(todo)} 个", flush=True)
                rc = go([(phase, out, a) for a in todo])
                if rc:
                    return rc
            if RUN.jobs(out, phase):
                say(f"[{phase}] 正式作业成功退出却仍未完成 —— 停止", flush=True)
                return 2
            d = ver / phase
            d.mkdir(parents=True, exist_ok=True)
            if not (RUN.inputs_file(d).exists() and RUN.deadline_file(d).exists()):
                rc = go([("prepare", d, [])])
                if rc:
                    return rc
            sync_upstream(out, d, phase)
            for _ in range(max_runs + 2):
                res = assess(out, ver, phase, state, max_runs)
                save_state(ver, state)
                if res["unstable"]:
                    say(f"[{phase}] 多遍结果始终对不上，停止：{[keystr(k) for k in res['unstable']]}", flush=True)
                    return 2
                if res["rerun_formal"] or not res["need"]:
                    break
                say(f"[{phase}] 核对作业 {len(res['need'])} 个", flush=True)
                rc = go([(phase, d, args_of(k)) for k in res["need"]])
                if rc:
                    return rc
            if res["rerun_formal"]:
                say(f"[{phase}] 正式结果被两遍核对否定，已移到 _corrupted/ 并重跑：{[keystr(k) for k in res['rerun_formal']]}",
                    flush=True)
                continue
            if all(state[keystr(k)]["status"] == "match" for k in phase_keys(out, phase)):
                say(f"[{phase}] {len(phase_keys(out, phase))} 个作业两遍一致", flush=True)
                break
        else:
            say(f"[{phase}] 轮数用尽 —— 停止", flush=True)
            return 2
    for kind in ("report", "fig"):
        rc = go([(kind, out, [])])
        if rc:
            return rc
    summary = {ph: Counter(state[keystr(k)]["status"] for k in phase_keys(out, ph)) for ph in PHASES}
    extra = {k: s["runs"] for k, s in state.items() if len(s["runs"]) > 2}
    save_state(ver, state)
    (ver / "verify_summary.json").write_text(json.dumps({"phases": summary, "more_than_two_runs": extra}, indent=1,
                                                        ensure_ascii=False))
    say(f"完成：全部作业两遍一致；多于两遍的作业 {len(extra)} 个（见 {ver / 'verify_summary.json'}）", flush=True)
    return 0


def status(out, ver) -> dict:
    state = load_state(ver)
    return {ph: dict(Counter(state.get(keystr(k), {}).get("status", "未核对") for k in phase_keys(out, ph)))
            for ph in PHASES}


def main(argv=None) -> int:
    gc.disable()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--ver", required=True)
        if name == "run":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--tries", type=int, default=6)
            p.add_argument("--max-runs", type=int, default=MAX_RUNS)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(status(a.out, a.ver), ensure_ascii=False, indent=1))
        return 0
    out, ver = Path(a.out), Path(a.ver)
    L = Launcher(new_attempt(out / "logs"), verify_logdir=new_attempt(ver / "logs"), ver_root=ver)
    print(f"正式日志 {L.logdir} · 核对日志 {L.verify_logdir}", flush=True)
    return run(out, ver, L, jobs=a.jobs, tries=a.tries, max_runs=a.max_runs)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
