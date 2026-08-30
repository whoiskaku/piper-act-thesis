#!/usr/bin/env python3
"""Step 1 v3 采集与分析 —— 箱体 8 顶点全因子候选筛选。

**行政状态（2026-08-24）**：本阶段已完成，整个 Step 1 随后在确认 9/30 block
处永久停止，见 `reports/STEP1_CLOSEOUT.md`。本脚本仅保留计划、状态及离线分析
用于审计；继续采集、dry-run 写状态和 reset 均被关闭。

预注册: `reports/PREREG_STEP1_V3.md`（2026-08-24 冻结）
日程:   `reports/step1_v3_schedule.json`（40 集, seed 20260824）

设计: 8 顶点 × 4 + 8 集裸 ACT, 分两会话各 20 集。
      裸 ACT 在合并序列的 slot 1/7/14/20 ⇒ 每个顶点都夹在两个裸之间。

**无 G、无 H、无求逆。** 运行时只用已记录的指令算接缝方向, 不读 IMU。

用法:
    python sim/collect_step1_v3.py --plan              # 只看日程, 不动机器
    python sim/collect_step1_v3.py --dry-run           # 全流程逻辑校验(合成 loss)
    python sim/collect_step1_v3.py --session 1         # 跑会话 1 的 20 集
    python sim/collect_step1_v3.py --slots 5           # 只跑接下来 5 集
    python sim/collect_step1_v3.py --status            # 进度
    python sim/collect_step1_v3.py --analyze           # §4 筛选分析
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.spsa_step1_v2 import (                      # noqa: E402  复用 v2 的冻结机制
    CALIB, MAX_LAUNCH_ATTEMPTS, MIN_FRAMES, POLICY,
    PartialEpisode, _frames_in, loss_of, safety_check,
)

OUT = ROOT / "outputs/step1_v3"
SCHEDULE = ROOT / "reports/step1_v3_schedule.json"
STATE = OUT / "state.json"
STOP_RECORD = ROOT / "reports/STEP1_CLOSEOUT.md"
COLLECTION_CLOSED = True                  # 行政终止硬门；不依赖报告文件是否被移动
MAX_SLOT_VOIDS = 3          # 同一 slot 的 A 类作废上限, 超过则停下查环境
CORNERS = ["".join(s) for s in itertools.product("-+", repeat=3)]   # 枚举顺序 = 平局规则


# ══════════════════════════════════════════════════════════════════════════
def load_schedule() -> list[dict]:
    return json.loads(SCHEDULE.read_text())["schedule"]


def theta_file(corner: str, a_prox: float, a_wrist: float, tau: float) -> Path:
    """顶点的 θ 文件（幅值以 mrad 给出, 落盘转 rad）。首次调用时写。"""
    p = OUT / "theta" / f"{corner}.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"theta": {"a_prox": a_prox * 1e-3,
                                           "a_wrist": a_wrist * 1e-3,
                                           "tau": tau},
                                 "corner": corner, "source": "PREREG_STEP1_V3 §2.1"},
                                indent=1))
    return p


class State:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        if STATE.exists():
            d = json.loads(STATE.read_text())
        else:
            d = {"done": [], "voided": [], "episodes_used": 0, "episodes_wasted": 0}
        self.d = d

    def save(self) -> None:
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, ensure_ascii=False, indent=1))
        tmp.replace(STATE)

    def is_done(self, session: int, slot: int) -> bool:
        return any(r["session"] == session and r["slot"] == slot for r in self.d["done"])

    def voids_at(self, session: int, slot: int) -> int:
        return sum(1 for r in self.d["voided"]
                   if r["session"] == session and r["slot"] == slot)


# ══════════════════════════════════════════════════════════════════════════
def _launch(tag: str, tf: Path | None, attempt: int) -> int:
    env = dict(os.environ)
    env["BCORR_CALIB"] = CALIB
    env["BCORR_LOG"] = str(OUT / f"{tag}.json")
    if tf is None:                                   # 裸 ACT = 对照
        env["BCORR_MODE"] = "off"
        env["BCORR_CKPT"] = str(ROOT / "outputs/p2_pi/pi_direct_bare_s0.pt")
    else:
        env["BCORR_MODE"] = "step1_v2"               # v2 的接缝脉冲运行时, 只换 θ
        env["BCORR_THETA"] = str(tf)
        env["BCORR_CKPT"] = str(tf)
    lib = Path(env.get("CONDA_PREFIX", "")) / "lib/libstdc++.so.6"
    if lib.exists():                                 # cv2/libstdc++ CXXABI 冲突
        env["LD_PRELOAD"] = f"{lib}:{env.get('LD_PRELOAD','')}".rstrip(":")
    cmd = [sys.executable, "lerobot/scripts/control_robot.py",
           "--robot.type=piper", "--robot.inference_time=true", "--robot.enable_imu=true",
           "--control.type=record", "--control.fps=20", "--control.single_task=pp2",
           "--control.warmup_time_s=2", "--control.episode_time_s=30",
           "--control.reset_time_s=10", "--control.push_to_hub=false",
           f"--control.policy.path={POLICY}",
           f"--control.repo_id={env.get('USER','u')}/eval_v3_{tag}_a{attempt}",
           "--control.num_episodes=1"]
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def run_episode(tag: str, tf: Path | None, dry: bool) -> tuple[Path, int, int]:
    """→ (日志路径, 启动次数, 退出码)。判据是**帧数**, 不是退出码（§6 ⑥-5b）。"""
    ep = OUT / f"{tag}_ep0.json"
    if dry:
        return ep, 1, 0
    for attempt in range(1, MAX_LAUNCH_ATTEMPTS + 1):
        if ep.exists():                              # 残留移开, **不删**
            ep.rename(ep.with_name(f"{tag}_ep0.stale_{int(time.time())}.json"))
        rc = _launch(tag, tf, attempt)
        n = _frames_in(ep)
        if n >= MIN_FRAMES:
            if rc != 0:
                print(f"     ⚠ 退出期崩溃(退出码 {rc}) —— {n} 帧完整, 本集**有效**")
            if attempt > 1:
                print(f"     (第 {attempt} 次启动成功)")
            return ep, attempt, rc
        if n == 0:                                   # ⑥-5b: 未产生 episode, 原地重启
            if attempt < MAX_LAUNCH_ATTEMPTS:
                print(f"     ⚠ 启动期崩溃(退出码 {rc}, 0 帧) —— "
                      f"3s 后第 {attempt+1}/{MAX_LAUNCH_ATTEMPTS} 次")
                time.sleep(3)
                continue
            break
        aside = ep.with_name(f"{tag}_ep0.partial_{int(time.time())}.json")
        ep.rename(aside)                             # 保留证据
        raise PartialEpisode(f"{tag}: 只录到 {n} 帧(< {MIN_FRAMES}), 退出码 {rc} "
                             f"⇒ A 类, 同 slot 原位重跑; 残件 {aside.name}")
    raise RuntimeError(f"{tag}: 连续 {MAX_LAUNCH_ATTEMPTS} 次启动期崩溃 —— 请先查环境")


# ══════════════════════════════════════════════════════════════════════════
def one_slot(st: State, e: dict, dry: bool) -> None:
    """跑一个 slot。A 类失败 ⇒ 同 slot 原位重跑（§3.4）; B 类 ⇒ 保留并取消候选资格。"""
    s, slot, arm = e["session"], e["slot"], e["arm"]
    corner = e["corner"]
    while True:
        nv = st.voids_at(s, slot)
        if nv >= MAX_SLOT_VOIDS:
            raise RuntimeError(f"S{s}/slot{slot}: 已作废 {nv} 次(上限 {MAX_SLOT_VOIDS}) —— 请先查环境")
        tag = f"S{s}s{slot:02d}_{corner if corner else 'bare'}" + (f"_v{nv}" if nv else "")
        tf = None if arm == "bare" else theta_file(corner, e["a_prox"], e["a_wrist"], e["tau"])
        print(f"\n── 会话{s} slot {slot:>2}/20  {corner or '裸 ACT'}"
              + ("" if tf is None else f"  A=({e['a_prox']:+.0f},{e['a_wrist']:+.0f})mrad τ={e['tau']:.0f}")
              + (f"  [第 {nv+1} 次]" if nv else ""))
        t0 = time.time()
        try:
            log, att, rc = run_episode(tag, tf, dry)
        except PartialEpisode as ex:
            print(f"     ✗ {ex}")
            st.d["voided"].append({"session": s, "slot": slot, "tag": tag,
                                   "reason": "partial", "detail": str(ex)})
            st.d["episodes_wasted"] += 1
            st.save()
            continue
        d = loss_of(log, dry, dry_theta=[0.0, 0.0, 0.0] if tf is None
                    else [e["a_prox"] / 2, e["a_wrist"] / 2, (e["tau"] - 6) / 4])
        bad = safety_check(d)
        a_cls = [b for b in bad if "A 类" in b]
        b_cls = [b for b in bad if "B 类" in b]
        if a_cls:                                    # ⑥-5: 作废 + 原位重跑
            print(f"     ✗ A 类技术故障: {'; '.join(a_cls)} ⇒ 作废, 同 slot 原位重跑")
            st.d["voided"].append({"session": s, "slot": slot, "tag": tag,
                                   "reason": "A_class", "detail": a_cls})
            st.d["episodes_wasted"] += 1
            st.save()
            continue
        rec = {"session": s, "slot": slot, "arm": arm, "corner": corner, "tag": tag,
               "L": d["L"], "log_L": d["log_L"], "frames": d["frames"],
               "sat_pct": d["sat_pct"], "stale": d["stale"], "nan": d["nan"],
               "launch_attempts": att, "exit_code": rc,
               "wall_s": round(time.time() - t0, 1), "ts": time.time(),
               "b_class": bool(b_cls), "b_class_detail": b_cls}
        if b_cls:                                    # ⑥-6: 保留记录, 取消候选资格
            print(f"     ⚠ B 类候选失败: {'; '.join(b_cls)}")
            print(f"       ⇒ 记录保留进描述模型, 但顶点 ({corner}) **取消候选资格**")
        st.d["done"].append(rec)
        st.d["episodes_used"] += 1
        st.save()
        print(f"     ✓ L={d['L']:.6f}  帧={d['frames']}  sat={d['sat_pct']:.2f}%  "
              f"{rec['wall_s']:.0f}s")
        return


def collect(st: State, sched: list[dict], dry: bool,
            session: int | None, n_slots: int | None) -> None:
    if COLLECTION_CLOSED:
        raise RuntimeError(
            "Step 1 已永久关闭；"
            f"权威记录：{STOP_RECORD}. 只允许 --plan/--status/--analyze/--sensitivity。"
        )
    todo = [e for e in sched
            if not st.is_done(e["session"], e["slot"])
            and (session is None or e["session"] == session)]
    if n_slots:
        todo = todo[:n_slots]
    if not todo:
        print("没有待跑的 slot。")
        return
    print(f"待跑 {len(todo)} 集"
          + (f"（会话 {session}）" if session else "")
          + f"；已完成 {len(st.d['done'])}/40，作废 {st.d['episodes_wasted']}")
    for e in todo:
        one_slot(st, e, dry)
    status(st, sched)


# ══════════════════════════════════════════════════════════════════════════
def status(st: State, sched: list[dict]) -> None:
    done = st.d["done"]
    print(f"\n{'═'*62}\n进度 {len(done)}/40 集"
          f"   在册 {st.d['episodes_used']} + 作废 {st.d['episodes_wasted']}"
          f" = 机时 {st.d['episodes_used'] + st.d['episodes_wasted']}")
    for s in (1, 2):
        ds = [r for r in done if r["session"] == s]
        print(f"  会话{s}: {len(ds)}/20", end="")
        if ds:
            b = [r["L"] for r in ds if r["arm"] == "bare"]
            c = [r["L"] for r in ds if r["arm"] == "corner"]
            print(f"   裸 n={len(b)}" + (f" 中位 {np.median(b):.5f}" if b else "")
                  + f"   顶点 n={len(c)}" + (f" 中位 {np.median(c):.5f}" if c else ""))
        else:
            print()
    dq = sorted({r["corner"] for r in done if r.get("b_class")})
    if dq:
        print(f"  ⚠ B 类失败 ⇒ 取消候选资格的顶点: {', '.join('('+c+')' for c in dq)}")
    cnt = {c: sum(1 for r in done if r["corner"] == c) for c in CORNERS}
    print("  每顶点集数: " + "  ".join(f"({c}){cnt[c]}" for c in CORNERS))


# ══════════════════════════════════════════════════════════════════════════
def analyze(st: State) -> int:
    """§4 筛选分析。饱和模型 + Dunnett 单侧 + §4.3 的负结论界。"""
    done = st.d["done"]
    if len(done) < 12:
        print(f"只有 {len(done)} 集，太少，先采集。")
        return 1
    y = np.array([r["log_L"] for r in done])
    sess = np.array([r["session"] - 1.5 for r in done], float)     # 中心化
    slot = np.array([(r["slot"] - 10.5) / 9.5 for r in done], float)
    Dm = np.array([[1.0 if r["corner"] == c else 0.0 for c in CORNERS] for r in done])
    keep = [j for j in range(8) if Dm[:, j].sum() > 0]
    # 中期分析(只有一个会话)时 sess 是常数 ⇒ 与截距共线, inv(AᵀA) 会炸。
    # **主分析(40 集, 两会话)不受影响**; 这里只是让中期也能看。
    cov_cols = [("session", sess)] if len({r["session"] for r in done}) > 1 else []
    if not cov_cols:
        print("⚠ 只有一个会话 ⇒ 已剔除 session 项（中期分析；主分析需两会话）\n")
    cov_cols.append(("slot", slot))
    A = np.column_stack([np.ones(len(y))] + [c for _, c in cov_cols] + [Dm[:, keep]])
    n_cov = len(cov_cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(y) - A.shape[1]
    res = y - A @ beta
    s2 = float(res @ res) / dof
    C = s2 * np.linalg.inv(A.T @ A)
    off = 1 + n_cov                               # 截距 + 协变量
    eff = {CORNERS[j]: (beta[off + i], float(np.sqrt(C[off + i, off + i])))
           for i, j in enumerate(keep)}
    dq = {r["corner"] for r in done if r.get("b_class")}

    print(f"{'═'*70}\n§4.1 筛选 —— 顶点 vs 裸 ACT"
          f"（调整项: {', '.join(nm for nm, _ in cov_cols)}; dof={dof}; n={len(y)}）\n")
    print(f"{'顶点':<8}{'n':>3}{'ΔlogL':>9}{'SE':>7}{'L 变化':>10}{'t':>7}  资格")
    for c in CORNERS:
        if c not in eff:
            continue
        b, se = eff[c]
        n = sum(1 for r in done if r["corner"] == c)
        print(f"({c:<5}){n:>3}{b:>+9.4f}{se:>7.4f}{(np.exp(b)-1)*100:>+9.1f}%"
              f"{b/se:>+7.2f}  {'✗ B类失格' if c in dq else '✓'}")

    # ── Dunnett 单侧临界值（MC, 用实际拟合协方差）──
    # 共同残差方差由本批数据估计，因此零假设下是**多元 t(df=dof)**，不是
    # 多元正态。旧实现漏掉共享的 χ²/dof 分母，给出 2.441；那会把
    # 小样本临界值和校正 p 值都算得过于乐观。
    idx = [off + i for i in range(len(keep))]
    Cs = C[np.ix_(idx, idx)]
    se_v = np.sqrt(np.diag(Cs))
    corr = Cs / np.outer(se_v, se_v)
    rng = np.random.default_rng(20260824)
    Z = rng.multivariate_normal(np.zeros(len(idx)), corr, 400000)
    scale = np.sqrt(rng.chisquare(dof, len(Z)) / dof)
    max_t = (Z / scale[:, None]).max(1)
    crit = float(np.quantile(max_t, 0.95))
    print(f"\nDunnett 单侧临界值（k={len(idx)}, α=0.05, df={dof} 多元 t，"
          f"MC 于实际协方差）= {crit:.3f}")

    elig = [c for c in eff if c not in dq]
    if not elig:
        print("\n⚠ 所有顶点均被 B 类失格 —— 无候选。")
        return 1
    best = min(elig, key=lambda c: (eff[c][0], CORNERS.index(c)))   # 严格最小 + 枚举平局
    b, se = eff[best]
    adj_p = float(np.mean(max_t >= -b / se))
    print(f"\n§4.2 候选 = **({best})**   ΔlogL {b:+.4f} ± {se:.4f}"
          f"   ⇒ L {(np.exp(b)-1)*100:+.1f}%   单侧 p={_p1(b/se, dof):.4f}")
    print(f"     Dunnett 校正后单侧 p≈{adj_p:.4f}")
    print("     （严格取最小；无平局带、无入围门槛 —— 预注册 §4.2）")

    # ── §4.3 负结论: 同时单侧下界 ──
    lo = min(eff[c][0] - crit * eff[c][1] for c in eff)
    B = 1 - np.exp(lo)
    print(f"\n§4.3 负结论\n  8 顶点同时单侧 95% 下界 ⇒ 「所有顶点真实改善 < {B*100:.1f}%」")
    print(f"  ✓ 只能写: 「排除这 {len(eff)} 个顶点达到 {B*100:.0f}%」")
    print("  ✗ 不得写: 「排除整个箱体」/「排除达到 15%」/「接缝脉冲族无效」")
    print(f"  15% 级否定只能来自确认阶段（30 对 matched A/B, 无多重性）")

    # ── 因子分解（仅供解读）──
    if len(eff) == 8:
        v = np.array([eff[c][0] for c in CORNERS])
        sg = np.array([[1 if ch == '+' else -1 for ch in c] for c in CORNERS], float)
        print(f"\n因子分解（仅解读用；x=±1 编码, 单位 ΔlogL）")
        print(f"  常数项 b0 = {v.mean():+.4f}  ← **顶点均值 vs 中心(裸 ACT)** = 总体曲率对比")
        print(f"           （这是中心重复能给的**唯一**曲率信息, 1 个自由度；"
              f"不能证明箱体线性）")
        for nm, col in (("x1 (A_prox)", sg[:, 0]), ("x2 (A_wrist)", sg[:, 1]),
                        ("x3 (τ)", sg[:, 2]),
                        ("x1:x2", sg[:, 0]*sg[:, 1]), ("x1:x3", sg[:, 0]*sg[:, 2]),
                        ("x2:x3", sg[:, 1]*sg[:, 2]),
                        ("x1:x2:x3", sg[:, 0]*sg[:, 1]*sg[:, 2])):
            print(f"  {nm:<13} {float((col*v).mean()):+.4f}")
    return 0


def sensitivity(st: State) -> int:
    """留一法敏感性分析 —— 主分析**不变**, 这是并列报告的附加信息。

    预注册没有离群剔除条款, 所以任何一集都**不会**被剔出主分析。此处只回答
    「结论有多依赖单独某一集」。对照臂只有 8 集(中期 4 集), 单点杠杆很大,
    所以这个检查是必要的, 而不是可选的。
    """
    done = st.d["done"]
    base = _corner_effects(done)
    if base is None:
        print("集数太少。")
        return 1
    gm = float(np.mean([v[0] for v in base.values()]))
    print(f"{'═'*70}\n留一法敏感性 —— 逐集剔除后「顶点均值 vs 对照」的变化")
    print(f"主分析（协议，全部在册）: {(np.exp(gm)-1)*100:+.1f}%\n")
    print(f"{'剔除':<22}{'顶点均值':>10}{'变化':>10}")
    rows = []
    for i, r in enumerate(done):
        e = _corner_effects([x for j, x in enumerate(done) if j != i])
        if e is None or len(e) < len(base):
            continue                              # 剔掉后某顶点没数据了, 跳过
        g = float(np.mean([v[0] for v in e.values()]))
        rows.append((abs(g - gm), r, g))
    rows.sort(reverse=True)
    for dv, r, g in rows[:5]:
        print(f"{(r['corner'] or '裸')+' slot'+str(r['slot']):<22}"
              f"{(np.exp(g)-1)*100:>+9.1f}%{(np.exp(g)-np.exp(gm))*100:>+9.1f}pp")
    print(f"  …（共 {len(rows)} 集，只列影响最大的 5 集）")
    dv, r, g = rows[0]
    share = 1 - (1 - np.exp(g)) / (1 - np.exp(gm)) if abs(1 - np.exp(gm)) > 1e-9 else 0.0
    print(f"\n最有影响的一集: {r['corner'] or '裸 ACT'} slot {r['slot']} "
          f"(L={r['L']:.5f}, {'对照臂' if r['arm']=='bare' else '顶点臂'})")
    print(f"  剔除它 ⇒ 顶点均值 {(np.exp(gm)-1)*100:+.1f}% → {(np.exp(g)-1)*100:+.1f}%"
          f"   ⇒ 表观效应的 **{share*100:.0f}%** 来自这一集")
    if r["arm"] == "bare" and share > 0.5:
        print("  ⚠ 它在**对照臂**上 —— 对照被抬高会让**所有**顶点一起显得偏好。")
        print("    对照臂集数: 中期 4 集 ⇒ 单点杠杆 1/4；全 40 集为 8 集 ⇒ 杠杆减半。")
    return 0


def _corner_effects(rows: list[dict]) -> dict | None:
    """顶点 vs 裸 ACT 的效应（含 slot 调整, 两会话时含 session）。"""
    if len(rows) < 12:
        return None
    y = np.array([r["log_L"] for r in rows])
    cols = [np.ones(len(y))]
    if len({r["session"] for r in rows}) > 1:
        cols.append(np.array([r["session"] - 1.5 for r in rows], float))
    cols.append(np.array([(r["slot"] - 10.5) / 9.5 for r in rows], float))
    Dm = np.array([[1.0 if r["corner"] == c else 0.0 for c in CORNERS] for r in rows])
    keep = [j for j in range(8) if Dm[:, j].sum() > 0]
    A = np.column_stack(cols + [Dm[:, keep]])
    if np.linalg.matrix_rank(A) < A.shape[1] or len(y) <= A.shape[1]:
        return None
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    s2 = float(res @ res) / (len(y) - A.shape[1])
    C = s2 * np.linalg.inv(A.T @ A)
    off = len(cols)
    return {CORNERS[j]: (b[off + i], float(np.sqrt(C[off + i, off + i])))
            for i, j in enumerate(keep)}


def _p1(t: float, dof: int) -> float:
    from scipy import stats
    return float(stats.t.cdf(t, dof))            # 单侧(改善方向 = t 为负)


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="只打印日程")
    ap.add_argument("--dry-run", action="store_true", help="合成 loss, 校验流程")
    ap.add_argument("--session", type=int, choices=(1, 2))
    ap.add_argument("--slots", type=int, help="只跑接下来 N 集")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="留一法：结论有多依赖单独某一集（主分析不变）")
    ap.add_argument("--reset", action="store_true", help="清空状态(需二次确认)")
    a = ap.parse_args()
    sched = load_schedule()

    if a.plan:
        print(f"预注册 reports/PREREG_STEP1_V3.md（冻结）  日程 {len(sched)} 集")
        for e in sched:
            t = "—" if e["tau"] is None else f"{e['tau']:.0f}"
            print(f"  会话{e['session']} slot {e['slot']:>2}  "
                  f"{('('+e['corner']+')') if e['corner'] else '裸 ACT':<8}"
                  f"A=({e['a_prox']:+.0f},{e['a_wrist']:+.0f})mrad  τ={t}")
            if e["session"] == 1 and e["slot"] == 20:
                print("  ─── 会话边界 ───")
        return 0

    audit_only = a.status or a.sensitivity or a.analyze
    if COLLECTION_CLOSED and (a.reset or not audit_only):
        print("拒绝采集或重置：Step 1 已永久关闭。", file=sys.stderr)
        print(f"权威记录：{STOP_RECORD}", file=sys.stderr)
        print("仍可使用 --plan、--status、--analyze 或 --sensitivity。", file=sys.stderr)
        return 2

    st = State()
    if a.reset:
        if input("清空 v3 状态? 输入 yes: ").strip() != "yes":
            print("取消。")
            return 1
        STATE.unlink(missing_ok=True)
        print("已清空。")
        return 0
    if a.status:
        status(st, sched)
        return 0
    if a.sensitivity:
        return sensitivity(st)
    if a.analyze:
        return analyze(st)
    collect(st, sched, a.dry_run, a.session, a.slots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
