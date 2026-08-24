#!/usr/bin/env python3
"""Step 1 v3 确认阶段 —— 候选 (-++) 的 30 组 matched A/B。

预注册 `reports/PREREG_STEP1_V3.md` §5：判据、口径、matched 定义、安全门
**全部沿用 v2 §5/§6 原样不改**。日程 `reports/step1_v3_confirm_schedule.json`。

  A 臂 = 裸 ACT               B 臂 = 候选 (-++) A_prox -2 / A_wrist +2 mrad / τ 10 帧
  r = 1 - L_B / L_A           30 block, 会话1 = 1-15, 会话2 = 16-30

**筛选阶段的 40 集不得进入本阶段的统计**（§5）。本脚本只读自己的 state。

两条由冻结文本推出的实现级读法（不是新规则）:
  · **A 类故障 ⇒ 整个 block 作废重跑**（两集都重跑）。依据: 附录 B 明写
    「初始机器人状态超出则该 triplet 按 A 类技术故障重跑」, 单位是 block;
    且 block 是统计的独立单位, 只重跑一集会破坏 matched。
  · 附录 B 的 `z_t` 不得继承一条在 v3 **不适用** —— 接缝脉冲无跨集状态
    (v2 §2.5「每集状态清零」), 结构上已保证。

用法:
    python sim/confirm_step1_v3.py --plan
    python sim/confirm_step1_v3.py --dry-run
    python sim/confirm_step1_v3.py --session 1        # block 1-15 (30 集)
    python sim/confirm_step1_v3.py --blocks 3         # 只跑接下来 3 个 block
    python sim/confirm_step1_v3.py --status
    python sim/confirm_step1_v3.py --analyze          # H1 五项判据
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.spsa_step1_v2 import (                      # noqa: E402
    CALIB, MAX_LAUNCH_ATTEMPTS, MIN_FRAMES, POLICY,
    PartialEpisode, _frames_in, loss_of, safety_check,
)

OUT = ROOT / "outputs/step1_v3_confirm"
SCHEDULE = ROOT / "reports/step1_v3_confirm_schedule.json"
STATE = OUT / "state.json"
MAX_BLOCK_VOIDS = 3
MATCH_TOL_RAD = np.deg2rad(0.5)      # 附录 B: 两臂各关节初始编码器差 ≤ 0.5°
H1_BAR = 0.15                        # §5 判据 1
SIGN_MIN = 20                        # §5 判据 3: ≥20/30 (H0 下 α=0.0494)


def sched() -> dict:
    return json.loads(SCHEDULE.read_text())


class State:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        self.d = (json.loads(STATE.read_text()) if STATE.exists()
                  else {"blocks": [], "voided": [], "episodes_used": 0,
                        "episodes_wasted": 0})

    def save(self) -> None:
        t = STATE.with_suffix(".tmp")
        t.write_text(json.dumps(self.d, ensure_ascii=False, indent=1))
        t.replace(STATE)

    def done(self, b: int) -> bool:
        return any(r["block"] == b for r in self.d["blocks"])

    def voids(self, b: int) -> int:
        return sum(1 for r in self.d["voided"] if r["block"] == b)


def theta_path(c: dict) -> Path:
    p = OUT / "theta_candidate.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"theta": {"a_prox": c["a_prox"] * 1e-3,
                                           "a_wrist": c["a_wrist"] * 1e-3,
                                           "tau": c["tau"]},
                                 "corner": c["corner"],
                                 "source": "v3 筛选阶段候选; PREREG_STEP1_V3 §4.2"},
                                indent=1))
    return p


def _launch(tag: str, tf: Path | None, attempt: int) -> int:
    env = dict(os.environ)
    env["BCORR_CALIB"] = CALIB
    env["BCORR_LOG"] = str(OUT / f"{tag}.json")
    if tf is None:
        env["BCORR_MODE"] = "off"
        env["BCORR_CKPT"] = str(ROOT / "outputs/p2_pi/pi_direct_bare_s0.pt")
    else:
        env["BCORR_MODE"] = "step1_v2"
        env["BCORR_THETA"] = str(tf)
        env["BCORR_CKPT"] = str(tf)
    lib = Path(env.get("CONDA_PREFIX", "")) / "lib/libstdc++.so.6"
    if lib.exists():
        env["LD_PRELOAD"] = f"{lib}:{env.get('LD_PRELOAD','')}".rstrip(":")
    return subprocess.run(
        [sys.executable, "lerobot/scripts/control_robot.py",
         "--robot.type=piper", "--robot.inference_time=true", "--robot.enable_imu=true",
         "--control.type=record", "--control.fps=20", "--control.single_task=pp2",
         "--control.warmup_time_s=2", "--control.episode_time_s=30",
         "--control.reset_time_s=10", "--control.push_to_hub=false",
         f"--control.policy.path={POLICY}",
         f"--control.repo_id={env.get('USER','u')}/eval_v3c_{tag}_a{attempt}",
         "--control.num_episodes=1"], cwd=ROOT, env=env, check=False).returncode


def run_episode(tag: str, tf: Path | None, dry: bool) -> tuple[Path, int, int]:
    ep = OUT / f"{tag}_ep0.json"
    if dry:
        return ep, 1, 0
    for a in range(1, MAX_LAUNCH_ATTEMPTS + 1):
        if ep.exists():
            ep.rename(ep.with_name(f"{tag}_ep0.stale_{int(time.time())}.json"))
        rc = _launch(tag, tf, a)
        n = _frames_in(ep)
        if n >= MIN_FRAMES:
            if rc != 0:
                print(f"     ⚠ 退出期崩溃(退出码 {rc}) —— {n} 帧完整, 本集**有效**")
            if a > 1:
                print(f"     (第 {a} 次启动成功)")
            return ep, a, rc
        if n == 0:
            if a < MAX_LAUNCH_ATTEMPTS:
                print(f"     ⚠ 启动期崩溃(退出码 {rc}, 0 帧) —— 3s 后第 {a+1} 次")
                time.sleep(3)
                continue
            break
        aside = ep.with_name(f"{tag}_ep0.partial_{int(time.time())}.json")
        ep.rename(aside)
        raise PartialEpisode(f"{tag}: 只录到 {n} 帧(<{MIN_FRAMES}), 退出码 {rc} "
                             f"⇒ A 类, **整个 block 作废重跑**; 残件 {aside.name}")
    raise RuntimeError(f"{tag}: 连续 {MAX_LAUNCH_ATTEMPTS} 次启动期崩溃 —— 请先查环境")


def init_state(log: Path, dry: bool) -> np.ndarray:
    """frames[0].state 的前 6 维（关节初始编码器）。"""
    if dry:
        return np.zeros(6)
    d = json.loads(log.read_text())
    return np.asarray(d["frames"][0]["state"], float)[:6]


class BlockVoid(RuntimeError):
    pass


def one_block(st: State, blk: dict, cand: dict, dry: bool) -> None:
    b, s, order = blk["block"], blk["session"], blk["order"]
    while True:
        nv = st.voids(b)
        if nv >= MAX_BLOCK_VOIDS:
            raise RuntimeError(f"block {b}: 已作废 {nv} 次(上限 {MAX_BLOCK_VOIDS}) —— 请先查环境")
        print(f"\n{'─'*58}\nblock {b:>2}/30  会话{s}  顺序 {order}"
              + (f"  [第 {nv+1} 次]" if nv else ""))
        print("  ⚠ matched 要求: 两集**同一物体摆放、同一 reset target**（附录 B）")
        eps, ran = {}, 0
        try:
            for arm in order:
                tag = f"B{b:02d}{arm}" + (f"_v{nv}" if nv else "")
                tf = None if arm == "A" else theta_path(cand)
                print(f"  ── {arm} 臂 ({'裸 ACT' if tf is None else cand['corner']})")
                t0 = time.time()
                log, att, rc = run_episode(tag, tf, dry)
                ran += 1
                d = loss_of(log, dry, dry_theta=[0.0, 0.0, 0.0] if tf is None
                            else [cand["a_prox"] / 2, cand["a_wrist"] / 2,
                                  (cand["tau"] - 6) / 4])
                bad = safety_check(d)
                if [x for x in bad if "A 类" in x]:
                    raise BlockVoid(f"{arm} 臂 A 类: {'; '.join(bad)}")
                eps[arm] = {"arm": arm, "tag": tag, "L": d["L"], "log_L": d["log_L"],
                            "frames": d["frames"], "sat_pct": d["sat_pct"],
                            "stale": d["stale"], "nan": d["nan"],
                            "launch_attempts": att, "exit_code": rc,
                            "wall_s": round(time.time() - t0, 1),
                            "b_class": bool([x for x in bad if "B 类" in x]),
                            "b_class_detail": [x for x in bad if "B 类" in x],
                            "init_state": init_state(log, dry).tolist()}
                print(f"     ✓ L={d['L']:.6f}  帧={d['frames']}  sat={d['sat_pct']:.2f}%")
            # 附录 B: 两臂初始编码器差 ≤ 0.5°
            dv = np.abs(np.array(eps["A"]["init_state"]) - np.array(eps["B"]["init_state"]))
            if dv.max() > MATCH_TOL_RAD:
                raise BlockVoid(f"matched 违规: 初始编码器最大差 "
                                f"{np.rad2deg(dv.max()):.2f}° > 0.5° (附录 B)")
        except (PartialEpisode, BlockVoid) as ex:
            print(f"     ✗ {ex}\n     ⇒ **整个 block 作废重跑**（附录 B / ⑥-5）")
            st.d["voided"].append({"block": b, "session": s, "attempt": nv,
                                   "reason": str(ex), "episodes_ran": ran})
            st.d["episodes_wasted"] += ran
            st.save()
            continue
        LA, LB = eps["A"]["L"], eps["B"]["L"]
        r = 1.0 - LB / LA
        bcl = eps["A"]["b_class"] or eps["B"]["b_class"]
        rec = {"block": b, "session": s, "order": order, "L_A": LA, "L_B": LB,
               "r": r, "b_class": bcl,
               "init_diff_deg": float(np.rad2deg(dv.max())),
               "A": eps["A"], "B": eps["B"], "ts": time.time()}
        st.d["blocks"].append(rec)
        st.d["episodes_used"] += 2
        st.save()
        print(f"  ⇒ L_A={LA:.6f}  L_B={LB:.6f}   **r = {r*100:+.1f}%**"
              f"   初始差 {rec['init_diff_deg']:.2f}°"
              + ("   ⚠ B 类 ⇒ 判否" if bcl else ""))
        return


def collect(st: State, dry: bool, session: int | None, nb: int | None) -> None:
    sc = sched()
    todo = [b for b in sc["blocks"] if not st.done(b["block"])
            and (session is None or b["session"] == session)]
    if nb:
        todo = todo[:nb]
    if not todo:
        print("没有待跑的 block。")
        return
    print(f"待跑 {len(todo)} 个 block ({len(todo)*2} 集); 已完成 {len(st.d['blocks'])}/30")
    for b in todo:
        one_block(st, b, sc["candidate"], dry)
    status(st)


def status(st: State) -> None:
    B = st.d["blocks"]
    print(f"\n{'═'*58}\n进度 {len(B)}/30 block"
          f"   在册 {st.d['episodes_used']} + 作废 {st.d['episodes_wasted']} 集")
    for s in (1, 2):
        bs = [r for r in B if r["session"] == s]
        rr = [r["r"] for r in bs]
        print(f"  会话{s}: {len(bs)}/15" + (f"   r 中位 {np.median(rr)*100:+.1f}%"
                                           f"   正向 {sum(1 for x in rr if x>0)}/{len(rr)}"
                                           if rr else ""))
    if B:
        rr = np.array([r["r"] for r in B])
        print(f"  全体: r 中位 {np.median(rr)*100:+.1f}%  均值 {rr.mean()*100:+.1f}%"
              f"  正向 {int((rr>0).sum())}/{len(rr)}")
    nb = [r["block"] for r in B if r["b_class"]]
    if nb:
        print(f"  ⚠ B 类失败 block: {nb} ⇒ 判否")


def analyze(st: State) -> int:
    from scipy import stats
    B = sorted(st.d["blocks"], key=lambda r: r["block"])
    if not B:
        print("还没有数据。")
        return 1
    r = np.array([x["r"] for x in B])
    n = len(r)
    final = n == 30
    print(f"{'═'*66}")
    print(f"H1 五项判据 —— 候选 (-++)   n = {n}/30 block"
          f"{'' if final else '  ⚠ **未采集完, 以下为描述性中期数, 不构成判定**'}")
    print(f"{'═'*66}\n")
    print(f"{'block':>6}{'会话':>5}{'顺序':>5}{'L_A':>10}{'L_B':>10}{'r':>9}{'初始差':>8}")
    for x in B:
        print(f"{x['block']:>6}{x['session']:>5}{x['order']:>5}{x['L_A']:>10.5f}"
              f"{x['L_B']:>10.5f}{x['r']*100:>+8.1f}%{x['init_diff_deg']:>7.2f}°"
              + ("  ⚠B类" if x["b_class"] else ""))
    med = float(np.median(r))
    tt = stats.ttest_1samp(r, 0.0)
    npos = int((r > 0).sum())
    m1 = [x["r"] for x in B if x["session"] == 1]
    m2 = [x["r"] for x in B if x["session"] == 2]
    gates_bad = [x["block"] for x in B if x["b_class"]]
    c1 = med >= H1_BAR
    c2 = tt.pvalue < 0.05 and r.mean() > 0
    c3 = npos >= SIGN_MIN
    c4 = bool(m1) and bool(m2) and np.median(m1) > 0 and np.median(m2) > 0
    c5 = not gates_bad
    print(f"\n{'判据':<44}{'结果':<22}{'通过'}")
    print(f"{'① median(r) ≥ 15%':<44}{f'{med*100:+.2f}%':<22}{'✓' if c1 else '✗'}")
    print(f"{'② 单样本 t 检验 p<0.05 双侧, 方向为改善':<44}"
          f"{f'p={tt.pvalue:.4f}, 均值 {r.mean()*100:+.2f}%':<22}{'✓' if c2 else '✗'}")
    print(f"{f'③ ≥ {SIGN_MIN}/30 个 block 正向':<44}{f'{npos}/{n}':<22}{'✓' if c3 else '✗'}")
    print(f"{'④ 两会话各自中位数 > 0':<44}"
          f"{(f'{np.median(m1)*100:+.1f}% / {np.median(m2)*100:+.1f}%' if m1 and m2 else '—'):<22}"
          f"{'✓' if c4 else '✗'}")
    print(f"{'⑤ 安全门通过':<44}{('全通过' if c5 else f'B类 block {gates_bad}'):<22}"
          f"{'✓' if c5 else '✗'}")
    ok = all((c1, c2, c3, c4, c5))
    se = r.std(ddof=1) / np.sqrt(n)
    lo, hi = r.mean() - 1.96 * se, r.mean() + 1.96 * se
    print(f"\n效应量: 均值 {r.mean()*100:+.2f}%  95%CI [{lo*100:+.2f}%, {hi*100:+.2f}%]"
          f"   SD(r)={r.std(ddof=1)*100:.1f}%")
    if not final:
        print("\n⇒ **未采集完, 不作判定。**")
        return 0
    print(f"\n{'═'*66}")
    if ok:
        print("⇒ **满足完整 H1** ⇒ Step 1 完成, 可开始研究通用 G（指導教員第 4、5 条）")
    elif c2 and med > 0:
        print("⇒ **显著 > 0 但 < 15%** ⇒ 证明「真机 IMU 学出的补正确实缩小 gap」，")
        print("   但未达实用门槛 ⇒ **不**进入 Step 2")
    else:
        print("⇒ **不显著或反向** ⇒ 这个**三参数接缝补正族**不够；")
        print("   **不得**推出「SPSA / model-free 整体无效」")
    lo1 = r.mean() - stats.t.ppf(0.95, n - 1) * se
    print(f"\n单侧 95% 界: 真实改善 ≥ {lo1*100:+.1f}%"
          f"   ⇒ {'可以' if hi < H1_BAR else '不能'}排除「达到 15%」"
          f"（上界 {hi*100:+.1f}%）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--session", type=int, choices=(1, 2))
    ap.add_argument("--blocks", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    sc = sched()
    if a.plan:
        c = sc["candidate"]
        print(f"预注册 {sc['prereg']}")
        print(f"候选 B 臂 = ({c['corner']})  A_prox {c['a_prox']:+.0f} / "
              f"A_wrist {c['a_wrist']:+.0f} mrad / τ {c['tau']:.0f} 帧")
        print(f"{sc['order_rule']}\nmatched: {sc['matched']}\n")
        for b in sc["blocks"]:
            print(f"  block {b['block']:>2}  会话{b['session']}  {b['order']}", end="")
            if b["block"] % 5 == 0:
                print()
        return 0
    st = State()
    if a.status:
        status(st)
        return 0
    if a.analyze:
        return analyze(st)
    collect(st, a.dry_run, a.session, a.blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
