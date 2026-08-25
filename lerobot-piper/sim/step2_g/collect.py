#!/usr/bin/env python3
"""Step 2 激励套件的正式采集编排器（90 条）。

**为什么需要它**：pilot 是 16 条、手工逐条可控；90 条不行。而手工跑 90 条的
风险不是累，是**漏**——漏掉门禁、漏掉断点、把中断前后的数据混在一起。

**全部机制都从已验证过的地方复用，不重写**（每一条都对应一次真实踩坑）：

| 机制 | 出处 | 踩过的坑 |
|---|---|---|
| 崩溃分类**按帧数不按退出码** | `spsa_step1_v2` ⑥-5b | 录满 584 帧、写完 parquet、正常 disable 之后才在 Qt teardown abort(退出码 −6)，被误判为「部分录制」并**删掉了日志** |
| 原子落盘 + 断点续跑 | `collect_step1_v3` | 编排器被本机偶发 SIGSEGV 杀掉，靠原子写才没丢账 |
| **孤儿清理** | `confirm_step1_v3` | 中断后用同一 tag 重跑会**跨中断配对**；实测两种配法给出 r = −11.4% 与 +13.5%，相差 25pp |
| 同名**移开不覆盖** | `replay_probe` | 判据修订后重跑会覆盖原始数据 |
| 逐条门禁 | `pilot_j4` | 固定 2 帧滞后 / 稳定位置带 / 到位 / 限位 / 速度 |
| **落盘前核对签字** | 本文件 | 没签过字的序列**不得上机** |

用法:
    python sim/step2_g/collect.py --plan            # 只看计划
    python sim/step2_g/collect.py --dry-run         # 全流程逻辑校验
    python sim/step2_g/collect.py --n 10            # 只跑接下来 10 条
    python sim/step2_g/collect.py --status
    python sim/step2_g/collect.py                   # 从断点continue
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SUITE = ROOT / "outputs/step2_g"
OUT = ROOT / "outputs/step2_g/collect"
STATE = OUT / "state.json"
SIGNOFF = ROOT / "reports/step2_g_signoff.json"
PY = sys.executable
MAX_VOIDS = 3


def load_plan() -> list[dict]:
    """读 manifest 并**逐条核对签字**。没签过字的序列不得上机。"""
    man = json.loads((SUITE / "manifest.json").read_text())
    sign = {x["name"]: x["sha"]
            for x in json.loads(SIGNOFF.read_text())["sequences"]}
    bad = [f"{k}: manifest sha {v['sha']} ≠ 签字 {sign.get(k, '未签字')}"
           for k, v in man.items() if sign.get(k) != v["sha"]]
    if bad:
        raise SystemExit("**签字核对失败，不得上机**:\n  " + "\n  ".join(bad))
    return [{"name": k, **v} for k, v in sorted(man.items())]


class State:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.d = (json.loads(STATE.read_text()) if STATE.exists()
                  else {"done": [], "voided": [], "used": 0, "wasted": 0})

    def save(self) -> None:
        t = STATE.with_suffix(".tmp")
        t.write_text(json.dumps(self.d, ensure_ascii=False, indent=1))
        t.replace(STATE)                              # 原子

    def is_done(self, n: str) -> bool:
        return any(r["name"] == n for r in self.d["done"])

    def voids(self, n: str) -> int:
        return sum(1 for r in self.d["voided"] if r["name"] == n)


def sweep_orphans(st: State, plan: list[dict]) -> None:
    """把上次中断留在盘上、属于**下一条未完成序列**的结果正式作废并移开。"""
    nxt = next((p for p in plan if not st.is_done(p["name"])), None)
    if nxt is None:
        return
    nm = nxt["name"]
    found = [f for f in OUT.glob(f"replay_{nm}.json")]
    found += [f for f in OUT.glob(f"settle_{nm}_*.json")]
    if not found:
        return
    ts = int(time.time())
    # 报出**为什么**判为孤儿 —— 「上次被中断」不等于「数据一定坏」。
    # 判据仍是**帧数**(同 ⑥-5b): 录满就有效, 只是记账没跟上。
    why = []
    for f in found:
        if f.name.startswith("replay_"):
            try:
                d = json.loads(f.read_text())
                why.append(f"{len(d.get('imu', []))} 帧"
                           + (f", stopped={d['stopped']}" if d.get("stopped") else ""))
            except Exception:                                     # noqa: BLE001
                why.append("无法解析")
    print(f"⚠ 发现 {nm} 的孤儿结果 {len(found)} 个（上次运行被中断）"
          + (f" —— {'; '.join(why)}" if why else ""))
    for f in found:
        f.rename(f.with_name(f"{f.stem}.orphan_{ts}.json"))
        print(f"    保留为 {f.stem}.orphan_{ts}.json")
    st.d["voided"].append({"name": nm, "reason": "中断留下的孤儿结果",
                           "n_files": len(found), "ts": ts})
    st.d["wasted"] += 1
    st.save()


def run_one(st: State, p: dict, dry: bool) -> None:
    nm = p["name"]
    while True:
        nv = st.voids(nm)
        if nv >= MAX_VOIDS:
            raise SystemExit(f"{nm}: 已作废 {nv} 次（上限 {MAX_VOIDS}）—— 请先查环境")
        print(f"\n── {nm}  {p['kind']}  {p['frames']} 帧"
              + (f"  [第 {nv+1} 次]" if nv else ""))
        if dry:
            st.d["done"].append({"name": nm, "sha": p["sha"], "dry": True,
                                 "ts": time.time()})
            st.d["used"] += 1
            st.save()
            return
        t0 = time.time()
        rc = subprocess.run(
            [PY, str(ROOT / "sim/replay_probe.py"),
             "--seqfile", str(SUITE / f"{nm}.npz"), "--arm", nm,
             "--outdir", str(OUT)], cwd=ROOT, check=False).returncode
        log = OUT / f"replay_{nm}.json"
        n = 0
        if log.exists():
            try:
                n = len(json.loads(log.read_text()).get("imu", []))
            except Exception:                                     # noqa: BLE001
                n = 0
        # **判据是帧数, 不是退出码** —— 本机会在录完、写完、disable 之后才在
        # Qt/glibc teardown 处 abort(退出码 −6)。见 spsa_step1_v2 ⑥-5b。
        if n >= min(100, p["frames"] // 2):
            if rc != 0:
                print(f"   ⚠ 退出期崩溃(退出码 {rc}) —— {n} 帧完整, 本条**有效**")
            st.d["done"].append({"name": nm, "sha": p["sha"], "frames": n,
                                 "exit_code": rc,
                                 "wall_s": round(time.time() - t0, 1),
                                 "ts": time.time()})
            st.d["used"] += 1
            st.save()
            print(f"   ✓ {n} 帧  {time.time()-t0:.0f}s")
            return
        print(f"   ✗ 只录到 {n} 帧(退出码 {rc}) ⇒ 作废重跑")
        if log.exists():                              # 移开保留, **不删**
            log.rename(log.with_name(f"{log.stem}.partial_{int(time.time())}.json"))
        st.d["voided"].append({"name": nm, "reason": f"frames={n}, rc={rc}",
                               "ts": time.time()})
        st.d["wasted"] += 1
        st.save()


def status(st: State, plan: list[dict]) -> None:
    d = st.d
    print(f"\n{'═'*56}\n进度 {len(d['done'])}/{len(plan)} 条"
          f"   在册 {d['used']} + 作废 {d['wasted']}")
    todo = [p["name"] for p in plan if not st.is_done(p["name"])]
    if todo:
        rem = sum(p["frames"] for p in plan if not st.is_done(p["name"]))
        print(f"  待跑 {len(todo)} 条 ≈ {rem/20/60:.1f} 分钟指令时长")
        print(f"  下一条: {todo[0]}")
    else:
        print("  ✅ 全部完成")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--n", type=int, help="只跑接下来 N 条")
    a = ap.parse_args()
    plan = load_plan()
    print(f"套件 {len(plan)} 条 · 签字核对 ✅ 全部一致")
    if a.plan:
        for i, p in enumerate(plan, 1):
            print(f"  {i:>3} {p['name']:<26}{p['kind']:<10}{p['frames']:>5} 帧")
        print(f"\n合计 {sum(p['frames'] for p in plan)/20/60:.1f} 分钟指令时长")
        return 0
    st = State()
    if a.status:
        status(st, plan)
        return 0
    if not a.dry_run:
        sweep_orphans(st, plan)
    todo = [p for p in plan if not st.is_done(p["name"])]
    if a.n:
        todo = todo[: a.n]
    if not todo:
        print("没有待跑的序列。")
        status(st, plan)
        return 0
    print(f"待跑 {len(todo)} 条")
    for p in todo:
        run_one(st, p, a.dry_run)
    status(st, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
