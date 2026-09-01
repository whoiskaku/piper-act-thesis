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
# 不占用作废额度的类别: env=运动前的环境失败(机时 0、无数据);
# artifact=编排器自身的记账错误产生的假记录(非真实事件)。
NOT_COUNTED = ("env", "artifact")

# ── 暂缓执行的序列（2026-08-26，用户决定）────────────────────────────────
# 第 54 条 S_j4_p2_rise10_a8.2 录完后, 机械臂在**断开归位**时发生碰撞:
# `PiperMotorsBus.safe_disconnect()` 是**一次性写** safe_disable_position
# [0,0,0,0,29.8°,0]、无插值; 从该条末帧 [−4.8, 51.5, −25.2, −4.2, 61.9, −113.9]
# 出发, 等于让腕部在低头 62° 的姿态下横滚 114°、同时 j2 下扫 51°。
# 事后臂停在 j4/j5 双双卡在 19.9° 的瘫态。
#
# **注意: 这条路径从未被任何检查覆盖过。** 套件的限位/工作空间/碰撞检查用的是
# `full_timeline`, 它建模的回停是「2s 余弦插值回 HOME_DEG」—— 终点、路径、
# 有无插值三者都与真机不同。事后补做的碰撞检查(真实 mesh, 全 90 条真实回停
# 路径)报告**零接触**, 即模型看不见这次碰撞 ⇒ 撞到的东西不在模型内
# (IMU 与支架未建模)。故没有任何模型侧依据可以放行重试。
#
# 下列 3 条的**末帧位姿与出事那条完全相同**(j5=61.9°, j6=−113.8°), 断开时会
# 执行一模一样的动作; 该位姿组首次上机即碰撞(1/1)。⇒ 暂缓, 不作废、不删除,
# 待「断开前分段插值归位」实现并验证后再单独补收。
# 2026-08-26 **解除暂缓**: `replay_probe.staged_return()` 已实现 ——
# 断开前由我们自己分段插值归位(j6 单独先走 → j4/j5 → j1–j3), 每段 3s 余弦、
# 每段带 1.5° 到位门禁, 不到位即停在原地如实报告并写入 return_blocked。
# 同时 `sequences.full_timeline` 的两端已按真机改正(起点/终点均为 safe_disable,
# 归位改为分段), 用修正后的时间线重查 90 条: 越模型限位 0.0000°、真实 mesh
# 碰撞 0/90。⇒ 这 3 条恢复执行, **同时充当 staged_return 的真机首验**
# (它们的末帧位姿正是出事那条的位姿, 是最坏情况)。
# 事故证据已归档为 ACCIDENT_20260826_return_collision_settle_*.json,
# 名字不再匹配孤儿通配, 不会被 sweep_orphans 扫走。
DEFERRED: dict[str, str] = {}


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
        """计入 MAX_VOIDS 的作废数。**环境失败(cls="env")不计** ——
        它发生在任何运动之前, 机时为 0、无数据产生, 对「这条序列能不能跑」
        不提供信息, 不该占用额度。"""
        return sum(1 for r in self.d["voided"]
                   if r["name"] == n and r.get("cls") not in NOT_COUNTED)


def sweep_orphans(st: State, plan: list[dict]) -> None:
    """把上次中断留在盘上、属于**下一条未完成序列**的结果正式作废并移开。"""
    # **暂缓的序列不参与孤儿清理** —— 它们不会再跑, 而盘上留下的结果往往正是
    # 事故证据(如 S_j4_p2_rise5_a4.1 的稳定段轨迹记录了断开碰撞后的瘫态位姿)。
    # 当成「下一条的孤儿」扫走 = 把证据改名归档并记一条假作废。
    nxt = next((p for p in plan
                if not st.is_done(p["name"]) and p["name"] not in DEFERRED), None)
    if nxt is None:
        return
    nm = nxt["name"]
    # **必须排除已扫过的孤儿** —— settle 的通配 `settle_{nm}_*.json` 会匹配
    # 重命名后的 `settle_{nm}_….orphan_….json`, 于是每跑一次就把同一个文件再改名
    # 一次、再记一条作废。2026-08-26 实际发生: C_j1_p2 因此累积 5 层 .orphan_
    # 后缀和 4 条假作废记录, 在环境完全正常的情况下把 MAX_VOIDS 撞满并锁死。
    found = [f for f in OUT.glob(f"replay_{nm}.json") if ".orphan_" not in f.name]
    found += [f for f in OUT.glob(f"settle_{nm}_*.json") if ".orphan_" not in f.name]
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
        if rc == 3:            # 运动前的环境失败, 见 replay_probe.py 的冻结约定
            st.d["voided"].append({"name": nm, "reason": "环境失败(运动前)",
                                   "cls": "env", "rc": rc, "ts": time.time()})
            st.save()             # wasted 不加: 没有机时被浪费
            raise SystemExit(
                f"\n{nm}: **环境失败** —— 机器未动、无数据产生, 不计入作废额度。\n"
                "  修好环境后重跑同一条命令, 会从断点接上。")
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
    n_def = sum(1 for p in plan if p["name"] in DEFERRED and not st.is_done(p["name"]))
    print(f"\n{'═'*56}\n进度 {len(d['done'])}/{len(plan)} 条"
          f"   在册 {d['used']} + 作废 {d['wasted']}"
          + (f" + **暂缓 {n_def}**" if n_def else ""))
    todo = [p["name"] for p in plan
            if not st.is_done(p["name"]) and p["name"] not in DEFERRED]
    if todo:
        rem = sum(p["frames"] for p in plan
                  if not st.is_done(p["name"]) and p["name"] not in DEFERRED)
        print(f"  待跑 {len(todo)} 条 ≈ {rem/20/60:.1f} 分钟指令时长")
        print(f"  下一条: {todo[0]}")
    elif n_def:
        # **不得报「全部完成」** —— 暂缓的 3 条没跑, 最终数据集是 87/90。
        print(f"  ⏸ 可跑的已跑完; 仍有 **{n_def} 条暂缓未收** ⇒ 数据集为 "
              f"{len(d['done'])}/{len(plan)}, 不是全套")
        for nm, why in DEFERRED.items():
            if not st.is_done(nm):
                print(f"      {nm:24s} —— {why}")
    else:
        print("  ✅ 全部完成")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--n", type=int, help="只跑接下来 N 条")
    # ── 路径参数化(2026-09-02) ────────────────────────────────────────
    # G-vNext 的复放采集要用**同一套机制**(崩溃重启/原子记账/孤儿清理/签字核对),
    # 只是换套件与落盘目录。故把常量改成可覆盖, **机制一行不动**。
    ap.add_argument("--suite", type=Path, default=None,
                    help="套件目录(含 manifest.json 与 {name}.npz); 默认 outputs/step2_g")
    ap.add_argument("--out", type=Path, default=None, help="落盘目录; 默认 SUITE/collect")
    ap.add_argument("--signoff", type=Path, default=None, help="签字文件")
    ap.add_argument("--session", type=int, default=None,
                    help="只跑 manifest 里 session 等于该值的序列")
    ap.add_argument("--only", nargs="+", default=None,
                    help="只跑指定名字的序列(用于跨会话重复采样)")
    a = ap.parse_args()
    global SUITE, OUT, STATE, SIGNOFF
    if a.suite:
        SUITE = a.suite if a.suite.is_absolute() else ROOT / a.suite
        OUT = SUITE / "collect"
        SIGNOFF = ROOT / "reports/step2_g_signoff.json"
    if a.out:
        OUT = a.out if a.out.is_absolute() else ROOT / a.out
    if a.signoff:
        SIGNOFF = a.signoff if a.signoff.is_absolute() else ROOT / a.signoff
    STATE = OUT / "state.json"
    plan = load_plan()
    if a.session is not None:
        plan = [p for p in plan if p.get("session") == a.session]
        print(f"**只跑会话 {a.session}** ⇒ {len(plan)} 条")
    if a.only:
        miss = sorted(set(a.only) - {p["name"] for p in plan})
        if miss:
            raise SystemExit(f"--only 里这些名字不在计划中: {miss}")
        plan = [p for p in plan if p["name"] in set(a.only)]
        print(f"**只跑指定 {len(plan)} 条**: {[p['name'] for p in plan]}")
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
    skipped = [p for p in todo if p["name"] in DEFERRED]
    todo = [p for p in todo if p["name"] not in DEFERRED]
    if skipped:
        print(f"⏸ 暂缓 {len(skipped)} 条（**不作废、不删除**，待归位流程修好后补收）:")
        for p in skipped:
            print(f"    {p['name']:24s} —— {DEFERRED[p['name']]}")
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
