#!/usr/bin/env python3
"""Step 1 v2 —— episode 级 SPSA 编排器。**无 G、无 H、无求逆。**

**行政状态（2026-08-24）**：整个 Step 1 已永久停止，见
`reports/STEP1_CLOSEOUT.md`。本模块仍供冻结分析代码导入；直接执行时只允许
`--plan` 审计，不再允许续跑、reset、final-check 或改写最终参数。

预注册 `reports/PREREG_STEP1_V2.md` §3–§4 的唯一实现。一轮 = 两集真机：

    采样 Δ  →  跑 θ⁺=θ+c_kΔ 一集  →  跑 θ⁻=θ−c_kΔ 一集
            →  各自用**实测 IMU** 算 L  →  ĝ=(y⁺−y⁻)/(2c_k)·Δ  →  θ ← θ−a_kĝ

状态全程落盘, **可中断续跑**（真机采集随时可能因技术故障停下）。

用法（先 conda activate lerobot）:
    python sim/spsa_step1_v2.py --dry-run           # 不碰机器人, 校验全流程逻辑
    python sim/spsa_step1_v2.py --plan              # 只打印本轮计划就退出
    python sim/spsa_step1_v2.py                     # 真机跑, 从状态文件续
    python sim/spsa_step1_v2.py --finalize          # 输出最终 θ(最后10个中心平均)
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 预注册 §3.3–3.4 冻结的 SPSA 常数 ─────────────────────────────────────────
N_ROUNDS = 40
PHASE1_ROUNDS = 10           # 前 10 轮固定 τ=6, 只学两个幅值
TAU_FIXED = 6.0
C0, C_GAMMA = 0.20, 0.101    # c_k = C0/(k+1)^C_GAMMA
A0, A_SHIFT, A_ALPHA = 0.35, 6.0, 0.602   # a_k = A0/(k+A_SHIFT)^A_ALPHA
TRUST = 0.12                 # 每个归一化参数单轮最大变化
BASELINE_EVERY = 5           # 每 5 轮插一集裸 ACT
FAILED_LOSS_MULT = 2.0       # 任务失败集的 L = 2 × 最近裸 ACT L
FINAL_AVG = 10               # 最终 θ = 最后 10 个中心的平均
SEED = 20260819              # Δ 的采样种子(可复现)
MIN_FRAMES = 550             # 预注册 §6 ⑥-3
MAX_LAUNCH_ATTEMPTS = 8      # 本机偶发启动期 SIGSEGV 的重试上限
MAX_PAIR_VOIDS = 3           # 同一轮因中途崩被作废重跑的上限(⑥-5 的自动化)

OUT = ROOT / "outputs/spsa_v2"
STATE = OUT / "state.json"
STOP_RECORD = ROOT / "reports/STEP1_CLOSEOUT.md"
COLLECTION_CLOSED = True                  # 行政终止硬门；不依赖报告文件是否被移动
POLICY = "outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model"
CALIB = "outputs/p2_labels/pp2a-deploy/manifest.json"


def _sp():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_sp_orch", ROOT / "lerobot/common/policies/act/step1_seam_pulse.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_sp_orch", m)
    spec.loader.exec_module(m)
    return m


def c_k(k: int) -> float:
    return C0 / (k + 1) ** C_GAMMA


def a_k(k: int) -> float:
    return A0 / (k + A_SHIFT) ** A_ALPHA


def delta_for(k: int) -> np.ndarray:
    """第 k 轮的 Δ（各分量独立 ±1）。由固定种子派生 ⇒ 可复现、可续跑。

    前 PHASE1_ROUNDS 轮把 τ 的分量置零 ⇒ τ 不被扰动也不被更新（§3.3）。
    """
    d = np.where(np.random.default_rng(SEED + k).random(3) < 0.5, -1.0, 1.0)
    if k < PHASE1_ROUNDS:
        d[2] = 0.0
    return d


# ══════════════════════════════════════════════════════════════════════════
class State:
    """落盘的优化状态。字段固定, 便于事后审计与复现。"""

    def __init__(self, d: dict | None = None):
        d = d or {}
        self.x = np.asarray(d.get("x", [0.0, 0.0, 0.0]), float)   # θ 的归一化坐标
        self.round = int(d.get("round", 0))                       # 下一个要跑的轮次 k
        self.centers: list[list[float]] = [list(map(float, c)) for c in d.get("centers", [])]
        self.history: list[dict] = list(d.get("history", []))
        self.baselines: list[dict] = list(d.get("baselines", []))
        self.episodes_used = int(d.get("episodes_used", 0))
        # 作废掉的集也消耗了机时, 单独计账以便对齐 §4 的 100 集预算
        self.episodes_wasted = int(d.get("episodes_wasted", 0))
        self.voided: list[dict] = list(d.get("voided", []))

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "round": self.round, "centers": self.centers,
                "history": self.history, "baselines": self.baselines,
                "episodes_used": self.episodes_used,
                "episodes_wasted": self.episodes_wasted, "voided": self.voided,
                "frozen": {"N_ROUNDS": N_ROUNDS, "PHASE1_ROUNDS": PHASE1_ROUNDS,
                           "C0": C0, "C_GAMMA": C_GAMMA, "A0": A0,
                           "A_SHIFT": A_SHIFT, "A_ALPHA": A_ALPHA, "TRUST": TRUST,
                           "BASELINE_EVERY": BASELINE_EVERY,
                           "FAILED_LOSS_MULT": FAILED_LOSS_MULT,
                           "FINAL_AVG": FINAL_AVG, "SEED": SEED}}

    def save(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(STATE)          # 原子替换, 中途断电不会留半个文件

    @classmethod
    def load(cls) -> "State":
        return cls(json.loads(STATE.read_text()) if STATE.exists() else None)

    def last_baseline_L(self) -> float | None:
        ok = [b["L"] for b in self.baselines if b.get("L") is not None]
        return float(ok[-1]) if ok else None


# ══════════════════════════════════════════════════════════════════════════
class PartialEpisode(RuntimeError):
    """录到一部分才崩 —— 预注册 §6 ⑥-5 的 A 类技术故障, **整对作废重跑**。

    `sides_ran`: 本轮已成功跑完但**数据将被丢弃**的 SPSA 侧集数(基线不算, 它会被保留)。
    """

    def __init__(self, msg: str, sides_ran: int = 0):
        super().__init__(msg)
        self.sides_ran = sides_ran


def _frames_in(log: Path) -> int:
    try:
        return len(json.loads(log.read_text()).get("frames", []))
    except Exception:                                            # noqa: BLE001
        return 0


def run_episode(tag: str, theta_file: Path | None, dry: bool,
                max_attempts: int = MAX_LAUNCH_ATTEMPTS) -> tuple[Path, int, int]:
    """跑一集真机, 返回 (日志路径, 启动次数, 退出码)。theta_file=None ⇒ 裸 ACT。

    **本机已知**: 偶发的启动期 SIGSEGV(退出码 −11), 随机, 重试即可
    (见 `scripts/train_resilient.sh` 的同一绕法)。故区分两种失败:

      · 一帧都没录到就崩 ⇒ 这一次**根本没产生 episode**, 原地重启, 不算协议事件
      · 录到一部分才崩   ⇒ 抛 PartialEpisode ⇒ 按 ⑥-5 整对作废重跑

    每次尝试用**独立的 repo_id**(带 _aN 后缀), 避免"数据集已存在"并且不删任何东西;
    失败尝试留下的数据集约 16 MB/集, 可事后自行清理。
    """
    ep_log = OUT / f"{tag}_ep0.json"          # wrapper 落盘的实际文件名
    if dry:
        return ep_log, 1, 0
    for attempt in range(1, max_attempts + 1):
        if ep_log.exists():                   # 上一次尝试的残留: **移开而非删除**
            ep_log.rename(ep_log.with_name(f"{tag}_ep0.stale_{int(time.time())}.json"))
        rc = _launch(tag, theta_file, attempt)
        n = _frames_in(ep_log)
        if n >= MIN_FRAMES:
            # **数据完整就接受, 与退出码无关。**
            # 本机会在**退出期**崩(Qt teardown + glibc "free(): invalid pointer",
            # 退出码 −6/SIGABRT), 发生在录完、写完 parquet、机器人 disable、
            # 打完 "Exiting" **之后** —— 与数据无关。
            # (最终 flush 由 control_robot 在 record 循环后显式调 policy.summary() 完成,
            #  所以此时日志已是完整的。)
            if rc != 0:
                print(f"     ⚠ 退出期崩溃(退出码 {rc}) —— 但 {n} 帧已完整录入,"
                      f" 本集**有效**, 仅记录退出码")
            if attempt > 1:
                print(f"     (第 {attempt} 次启动成功; 前 {attempt-1} 次为启动期崩溃)")
            return ep_log, attempt, rc
        if n == 0:
            if attempt < max_attempts:
                print(f"     ⚠ 启动期崩溃(退出码 {rc}, 0 帧) —— 本次未产生 episode, "
                      f"3s 后第 {attempt+1}/{max_attempts} 次尝试")
                time.sleep(3)
                continue
            print(f"     ⚠ 启动期崩溃(退出码 {rc}, 0 帧) —— 已是第 {attempt}/{max_attempts} 次")
            break
        # **不删证据** —— 移开保留, 供事后判断到底断在哪
        aside = ep_log.with_name(f"{tag}_ep0.partial_{int(time.time())}.json")
        ep_log.rename(aside)
        raise PartialEpisode(
            f"{tag}: 只录到 {n} 帧(< {MIN_FRAMES})后崩溃(退出码 {rc}) "
            f"⇒ 按 §6 ⑥-5 整对作废重跑; 残件留在 {aside.name}")
    raise RuntimeError(f"{tag}: 连续 {max_attempts} 次启动期崩溃 —— 不是偶发, 请先查环境")


def _launch(tag: str, theta_file: Path | None, attempt: int) -> int:
    log = OUT / f"{tag}.json"
    env = dict(os.environ)
    env["BCORR_CALIB"] = CALIB
    if theta_file is None:
        env["BCORR_MODE"] = "off"
        env["BCORR_CKPT"] = str(ROOT / "outputs/p2_pi/pi_direct_bare_s0.pt")
    else:
        env["BCORR_MODE"] = "step1_v2"
        env["BCORR_THETA"] = str(theta_file)
        env["BCORR_CKPT"] = str(theta_file)     # wrapper 用它当"有补正"的开关
    env["BCORR_LOG"] = str(log)
    cp = env.get("CONDA_PREFIX", "")
    lib = Path(cp) / "lib/libstdc++.so.6"
    if lib.exists():                            # cv2/libstdc++ CXXABI 冲突, 见 runbook
        env["LD_PRELOAD"] = f"{lib}:{env.get('LD_PRELOAD','')}".rstrip(":")
    cmd = [sys.executable, "lerobot/scripts/control_robot.py",
           "--robot.type=piper", "--robot.inference_time=true", "--robot.enable_imu=true",
           "--control.type=record", "--control.fps=20", "--control.single_task=pp2",
           "--control.warmup_time_s=2", "--control.episode_time_s=30",
           "--control.reset_time_s=10", "--control.push_to_hub=false",
           f"--control.policy.path={POLICY}",
           f"--control.repo_id={env.get('USER','u')}/eval_spsa_{tag}_a{attempt}",
           "--control.num_episodes=1"]
    print(f"    → {tag} 第 {attempt} 次启动")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


def loss_of(log: Path, dry: bool, dry_theta=None) -> dict:
    """实测 IMU 的 L。dry-run 时用一个合成曲面 + 噪声代替, 只为校验流程逻辑。"""
    if dry:
        x = np.asarray(dry_theta, float)
        true = 0.072 * (1.0 - 0.22 * np.exp(-((x[0] - 0.55) ** 2 + (x[1] + 0.40) ** 2) / 0.5))
        L = float(true * np.exp(np.random.default_rng(abs(hash(str(log))) % 2**32)
                                .normal(0, 0.123)))
        return {"L": L, "log_L": float(np.log(L)), "sat_pct": 0.0, "stale": 0, "nan": 0,
                "frames": 590, "failed": False}
    from sim.step1_v2_loss import episode_loss, load_calib
    R, b = load_calib(ROOT / CALIB)
    d = episode_loss(log, R, b)
    d["failed"] = False
    return d


def safety_check(d: dict) -> list[str]:
    """预注册 §6 的安全门。返回违规项（空 = 通过）。"""
    bad = []
    if d.get("sat_pct", 0.0) > 1.0:
        bad.append(f"sat_pct={d['sat_pct']:.2f}% > 1.0% (B 类候选失败)")
    if d.get("nan", 0):
        bad.append(f"nan={d['nan']} ≠ 0 (B 类候选失败)")
    if d.get("stale", 0):
        bad.append(f"stale={d['stale']} ≠ 0 (A 类技术故障)")
    if d.get("frames", 0) < 550:
        bad.append(f"frames={d['frames']} < 550 (A 类技术故障)")
    return bad


# ══════════════════════════════════════════════════════════════════════════
def one_round(st: State, sp, dry: bool, plan_only: bool) -> bool:
    """跑一轮。返回 False 表示应停止（已满 40 轮）。"""
    k = st.round
    if k >= N_ROUNDS:
        return False
    ck, ak = c_k(k), a_k(k)
    D = delta_for(k)
    xp = np.clip(st.x + ck * D, -1, 1)
    xm = np.clip(st.x - ck * D, -1, 1)
    if k < PHASE1_ROUNDS:                      # τ 固定 6 ⇒ 归一化坐标 0
        xp[2] = xm[2] = 0.0
    thp, thm = sp.SeamPulseTheta.from_x(xp), sp.SeamPulseTheta.from_x(xm)
    order = ["p", "m"] if k % 2 == 0 else ["m", "p"]     # 执行顺序逐轮交替

    print(f"\n════ 轮 {k+1}/{N_ROUNDS} ({'阶段一 τ 固定' if k < PHASE1_ROUNDS else '阶段二'}) ════")
    print(f"  中心 x = [{st.x[0]:+.4f} {st.x[1]:+.4f} {st.x[2]:+.4f}]  "
          f"c_k={ck:.4f}  a_k={ak:.4f}  Δ={D.astype(int).tolist()}")
    for nm, th in (("θ⁺", thp), ("θ⁻", thm)):
        print(f"  {nm}: A_prox={th.a_prox*1e3:+.3f} A_wrist={th.a_wrist*1e3:+.3f} mrad "
              f"τ={th.tau:.2f}")
    print(f"  执行顺序: {order}   已用集数 {st.episodes_used}")
    if plan_only:
        return False

    # 每 BASELINE_EVERY 轮（含第 0 轮）先跑一集裸 ACT 控漂移
    if k % BASELINE_EVERY == 0 and not any(b["tag"] == f"R{k:02d}base" for b in st.baselines):
        run_baseline(st, k, f"R{k:02d}base", dry)   # 已跑过就不重复(作废重跑时复用)

    res: dict[str, dict] = {}
    sides_ran = 0            # 本轮已跑完的侧集; 只有整轮提交时才计入 episodes_used
    for side in order:
        th = thp if side == "p" else thm
        tag = f"R{k:02d}{side}"
        tf = OUT / f"{tag}_theta.json"
        th.save(tf, round=k, side=side, x=(xp if side == "p" else xm).tolist())
        print(f"  ── {tag}")
        try:
            lg, tries, rc = run_episode(tag, tf, dry)
        except PartialEpisode as exc:
            raise PartialEpisode(str(exc), sides_ran=sides_ran) from None
        d = loss_of(lg, dry, xp if side == "p" else xm)
        sides_ran += 1
        viol = safety_check(d)
        if viol:
            print(f"     ⚠ 安全门: {'; '.join(viol)}")
            if any("A 类" in v for v in viol):
                raise RuntimeError(f"{tag}: A 类技术故障 ⇒ 整对作废重跑（不改 θ）")
            d["failed"] = True                 # B 类: 保留记录并按 §3.5 赋值
        d["launch_attempts"] = tries
        d["exit_code"] = rc
        res[side] = d
        print(f"     L={d['L']:.6f}  log L={d['log_L']:+.4f}"
              + ("  [B类失败, 将按 2×基线赋值]" if d["failed"] else ""))
        st.save()

    # §3.5 任务失败/候选失败 ⇒ L = 2 × 最近裸 ACT L（避免"少动"作弊）
    base = st.last_baseline_L()
    for side, d in res.items():
        if d["failed"]:
            if base is None:
                raise RuntimeError("尚无裸 ACT 基线, 无法按 §3.5 赋值")
            d["L_used"] = FAILED_LOSS_MULT * base
            d["log_L_used"] = float(np.log(d["L_used"]))
        else:
            d["L_used"], d["log_L_used"] = d["L"], d["log_L"]

    g = (res["p"]["log_L_used"] - res["m"]["log_L_used"]) / (2.0 * ck) * D
    step = np.clip(-ak * g, -TRUST, TRUST)      # §3.4 信赖域
    x_new = np.clip(st.x + step, -1, 1)
    if k < PHASE1_ROUNDS:
        x_new[2] = 0.0
    print(f"  ĝ = [{g[0]:+.3f} {g[1]:+.3f} {g[2]:+.3f}]   "
          f"step = [{step[0]:+.4f} {step[1]:+.4f} {step[2]:+.4f}]"
          + ("  ← 信赖域生效" if np.any(np.abs(-ak * g) > TRUST + 1e-12) else ""))
    print(f"  新中心 x = [{x_new[0]:+.4f} {x_new[1]:+.4f} {x_new[2]:+.4f}]")

    st.history.append({"round": k, "c_k": ck, "a_k": ak, "delta": D.tolist(),
                       "x_center": st.x.tolist(), "x_plus": xp.tolist(),
                       "x_minus": xm.tolist(), "order": order,
                       "L_plus": res["p"]["L"], "L_minus": res["m"]["L"],
                       "L_plus_used": res["p"]["L_used"],
                       "L_minus_used": res["m"]["L_used"],
                       "failed_plus": res["p"]["failed"], "failed_minus": res["m"]["failed"],
                       "exit_plus": res["p"]["exit_code"], "exit_minus": res["m"]["exit_code"],
                       "g": g.tolist(), "step": step.tolist(), "x_new": x_new.tolist()})
    st.x = x_new
    st.centers.append(x_new.tolist())
    st.round = k + 1
    st.episodes_used += sides_ran        # 整轮提交, 侧集此刻才入账
    st.save()
    return True


def run_baseline(st: State, k: int, tag: str, dry: bool) -> None:
    """一集裸 ACT, 用于漂移对照与 §3.5 的失败赋值基准。"""
    print(f"  ── 裸 ACT 基线 {tag}")
    lg, tries, rc = run_episode(tag, None, dry)
    bd = loss_of(lg, dry, [0, 0, 0])
    st.episodes_used += 1
    st.baselines.append({"round": k, "tag": tag, "launch_attempts": tries,
                         "exit_code": rc,
                         **{kk: bd[kk] for kk in ("L", "log_L", "frames")}})
    print(f"     L={bd['L']:.6f}")
    st.save()


def final_check(st: State, sp, dry: bool, n: int = 5) -> None:
    """预注册 §4 的"最终冻结参数检查 5 集" —— 只验安全与量级, **不是** H1。"""
    f = OUT / "theta_final.json"
    if not f.exists():
        print("先跑 --finalize 生成 theta_final.json"); return
    th = sp.SeamPulseTheta.load(f)
    print(f"\n最终参数检查 {n} 集  A_prox={th.a_prox*1e3:+.3f} "
          f"A_wrist={th.a_wrist*1e3:+.3f} mrad τ={th.tau:.2f}")
    rows = []
    for i in range(n):
        tag = f"FINAL{i}"
        lg, tries, rc = run_episode(tag, f, dry)
        d = loss_of(lg, dry, th.to_x())
        st.episodes_used += 1
        viol = safety_check(d)
        rows.append({"tag": tag, "exit_code": rc,
                     **{k: d[k] for k in ("L", "frames", "sat_pct")},
                     "violations": viol})
        print(f"  {tag}: L={d['L']:.6f}  sat={d['sat_pct']:.2f}%"
              + (f"  ⚠ {'; '.join(viol)}" if viol else "  ✓"))
        st.save()
    (OUT / "final_check.json").write_text(json.dumps(
        {"theta": th.to_dict(), "episodes": rows}, indent=1))
    Ls = np.array([r["L"] for r in rows])
    base = np.array([b["L"] for b in st.baselines]) if st.baselines else None
    print(f"\n  最终 θ: n={n} 中位 {np.median(Ls):.6f}")
    if base is not None and len(base):
        print(f"  裸 ACT 基线: n={len(base)} 中位 {np.median(base):.6f}")
        print(f"  ⇒ 粗略降幅 {(1-np.median(Ls)/np.median(base))*100:+.2f}%"
              f"   ← **不是** H1; H1 须用全新的 10 个 matched block(§5)")
    if any(r["violations"] for r in rows):
        print("  ⚠ 有集违反安全门 —— 不得进入 H1 A/B, 先查原因")


def finalize(st: State, sp) -> None:
    """最终 θ = 最后 FINAL_AVG 个中心的平均（§3.6, **不**取 loss 最低那轮）。"""
    if not st.centers:
        print("还没有任何中心, 无法定稿"); return
    tail = np.asarray(st.centers[-FINAL_AVG:], float)
    x = tail.mean(axis=0)
    th = sp.SeamPulseTheta.from_x(x)
    f = OUT / "theta_final.json"
    th.save(f, source=f"mean of last {len(tail)} centers", n_centers=len(st.centers),
            rounds_done=st.round, episodes_used=st.episodes_used, x=x.tolist())
    print(f"\n最终 θ（最后 {len(tail)} 个中心的平均, 非 loss 最低轮）")
    print(f"  A_prox = {th.a_prox*1e3:+.4f} mrad")
    print(f"  A_wrist= {th.a_wrist*1e3:+.4f} mrad")
    print(f"  τ      = {th.tau:.3f} 帧")
    print(f"  → {f.relative_to(ROOT)}")
    if st.baselines:
        Ls = np.array([b["L"] for b in st.baselines])
        print(f"\n  裸 ACT 基线 n={len(Ls)}: 均值 {Ls.mean():.6f}  "
              f"CV {Ls.std(ddof=1)/Ls.mean()*100:.1f}%" if len(Ls) > 1 else
              f"\n  裸 ACT 基线 n=1: {Ls[0]:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="不碰机器人, 用合成曲面校验逻辑")
    ap.add_argument("--plan", action="store_true", help="只打印本轮计划")
    ap.add_argument("--rounds", type=int, default=None, help="本次最多跑几轮")
    ap.add_argument("--finalize", action="store_true", help="输出最终 θ 并退出")
    ap.add_argument("--final-check", action="store_true",
                    help="用 theta_final 跑 5 集冻结检查(§4), 只验安全与量级")
    ap.add_argument("--reset", action="store_true", help="清空状态重新开始（会先备份）")
    a = ap.parse_args()

    plan_only = (a.plan and not a.dry_run and not a.finalize
                 and not a.final_check and not a.reset)
    if COLLECTION_CLOSED:
        if not plan_only:
            print("拒绝执行：Step 1 已永久关闭；v2 只保留 --plan 审计。", file=sys.stderr)
            print(f"权威记录：{STOP_RECORD}", file=sys.stderr)
            return 2
        # 原流程在 round=40 时会自动重写 theta_final；关闭后 `--plan` 必须严格只读。
        sp = _sp()
        st = State.load()
        print(f"Step 1 v2 已关闭 —— {st.round}/{N_ROUNDS} 轮，"
              f"在册 {st.episodes_used} + 作废 {st.episodes_wasted} 集")
        if st.round < N_ROUNDS:
            one_round(st, sp, dry=False, plan_only=True)
        else:
            print("无待执行轮次；冻结结果保持不变。")
        return 0
    sp = _sp()

    if a.reset and STATE.exists():
        bak = STATE.with_name(f"state_backup_{int(time.time())}.json")
        STATE.replace(bak); print(f"旧状态已备份到 {bak.name}")

    st = State.load()
    if a.finalize:
        finalize(st, sp); return 0
    if a.final_check:
        final_check(st, sp, a.dry_run); return 0

    print(f"Step 1 v2 SPSA  —— 预注册 reports/PREREG_STEP1_V2.md")
    print(f"状态: {STATE.relative_to(ROOT)}  已完成 {st.round}/{N_ROUNDS} 轮, "
          f"在册 {st.episodes_used} 集 + 作废 {st.episodes_wasted} 集 "
          f"= 机时 {st.episodes_used + st.episodes_wasted}/100" + ("   [DRY-RUN]" if a.dry_run else ""))

    budget = a.rounds if a.rounds is not None else N_ROUNDS
    done = 0
    while done < budget:
        voids = 0
        while True:
            try:
                ok = one_round(st, sp, a.dry_run, a.plan)
                break
            except PartialEpisode as exc:
                # ⑥-5: 整对作废重跑。本轮未写入 state, 所以"重跑"就是再进一次。
                # 只把**数据被丢弃**的集记作废: 本轮已跑完的 SPSA 侧集 + 崩掉那一集。
                # 基线**不算** —— 它已提交进 st.baselines 且重跑时去重复用, 数据有效。
                voids += 1
                ran_ok = int(getattr(exc, "sides_ran", 0))
                st.episodes_wasted += ran_ok + 1
                st.voided.append({"round": st.round, "voids_this_round": voids,
                                  "episodes_discarded": ran_ok + 1,
                                  "reason": str(exc)})
                st.save()
                print(f"\n  ⚠ {exc}")
                print(f"  ⇒ 本轮第 {voids}/{MAX_PAIR_VOIDS} 次作废重跑; 本次弃掉 "
                      f"{ran_ok + 1} 集（{ran_ok} 侧集已跑完但轮次未提交 + 1 集崩掉）"
                      f"{'; 基线保留复用' if st.round % BASELINE_EVERY == 0 else ''}")
                print(f"     累计: 在册 {st.episodes_used} + 作废 {st.episodes_wasted} "
                      f"= 机时 {st.episodes_used + st.episodes_wasted}/100")
                if voids >= MAX_PAIR_VOIDS:
                    print(f"  ✗ 同一轮已作废 {voids} 次 —— 不再自动重试, 先查环境")
                    return 2
                time.sleep(5)
        if not ok:
            break
        done += 1
    if st.round >= N_ROUNDS:
        # 收尾再插一集裸 ACT ⇒ 基线共 9 集, 与预注册 §4 的 "开始 1 + 每 5 轮" 对齐,
        # 且给出优化全程首尾的漂移量
        if not any(b["tag"] == "ENDbase" for b in st.baselines):
            run_baseline(st, N_ROUNDS, "ENDbase", a.dry_run)
        print(f"\n{N_ROUNDS} 轮跑完（共用 {st.episodes_used} 集）。")
        finalize(st, sp)
        print(f"\n下一步: python sim/spsa_step1_v2.py --final-check"
              f"   (预注册 §4 的 5 集冻结检查)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
