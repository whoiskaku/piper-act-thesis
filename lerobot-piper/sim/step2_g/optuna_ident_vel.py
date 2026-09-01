#!/usr/bin/env python3
"""**用关节速度目标重拟 23 个参数** —— 见 `reports/PREREG_VELOCITY_OBJECTIVE.md`。

结构、搜索空间、切分、`kp6` gauge、限幅器、前向回放**全部复用
`optuna_ident_enc`**，**只换目标函数**：

```
q̇ = 因果后向差分（编码器与仿真同一算法）
L_vel(band) = mean_i mean_{j∈激励关节} RMS(bp(q̇̂−q̇_enc, band)) / RMS(bp(q̇_enc, band))
L = 0.5·L_vel(0.5–3 Hz) + 0.5·L_vel(3–9 Hz)
```

**目标里只用编码器**（保持编码器域，不回 IMU 域）；IMU 只在评估阶段出现。

用法:
    python sim/step2_g/optuna_ident_vel.py --baseline
    python sim/step2_g/optuna_ident_vel.py --run --stage 1
    python sim/step2_g/optuna_ident_vel.py --report
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
W = 0.5                       # 两带权重（预注册定死）
BANDS = ((0.5, 3.0), (3.0, 9.0))
SEEDS = (11235813, 27182818, 16180339)
N_TRIALS = 800
PENALTY = 10.0
OUT = ROOT / "outputs/step2_g/optuna_vel"


def bp(x, lo, hi):
    from scipy.signal import firwin
    k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
    x = np.asarray(x, float)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def dvel(q):
    """因果后向差分（编码器与仿真用同一算法）。"""
    v = np.zeros_like(q)
    v[1:] = (q[1:] - q[:-1]) * FPS
    return v


def prep_vel(names, mj_dt):
    """复用 `optuna_ident_enc.prep`，再加上与 θ 无关的编码器速度分带能量。"""
    from optuna_ident_enc import prep
    items = prep(names, mj_dt)
    for it in items:
        ve = dvel(it["enc"])
        it["ve"] = {b: bp(ve, *b) for b in BANDS}
        it["ven"] = {b: {j: max(float(np.sqrt(np.mean(it["ve"][b][:, j] ** 2))), 1e-9)
                         for j in it["target"]} for b in BANDS}
    return items


def eval_vel(m, items, theta):
    """→ (L, L_lo, L_hi)。发散 ⇒ PENALTY。"""
    from optuna_ident_enc import roll
    import params as P
    P.apply(m, np.asarray(theta, float))
    per = {b: [] for b in BANDS}
    for it in items:
        try:
            q, _, _ = roll(m, it)
        except Exception:                                    # noqa: BLE001
            return PENALTY, PENALTY, PENALTY
        if not np.isfinite(q).all():
            return PENALTY, PENALTY, PENALTY
        vh = dvel(q)
        for b in BANDS:
            e = bp(vh, *b) - it["ve"][b]          # 带通线性 ⇒ 等价 bp(vh−ve)
            per[b].append(float(np.mean([
                min(np.sqrt(np.mean(e[:, j] ** 2)) / it["ven"][b][j], PENALTY)
                for j in it["target"]])))
    lo = float(np.mean(per[BANDS[0]])); hi = float(np.mean(per[BANDS[1]]))
    return W * lo + (1 - W) * hi, lo, hi


# ── Optuna ──────────────────────────────────────────────────────────────
def storage(seed):
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    OUT.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(OUT / f"journal_s{seed}.log")))


def study_name(seed):
    return f"g_ident_vel_s{seed}"


def make_study(seed, create, sampler_seed=None):
    import optuna
    from optuna_ident_enc import BASE_XML, BASE_PHYS_V1, flat
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(
        study_name=study_name(seed), storage=storage(seed), direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=seed if sampler_seed is None else sampler_seed,
            multivariate=True, group=True, n_startup_trials=40))
    if create and not s.trials:
        from g_q_v1 import THETA_STAR
        s.enqueue_trial(flat(BASE_XML))
        s.enqueue_trial(flat(THETA_STAR))
    return s


def _worker(seed, rank, n_trials):
    import gc as _gc, mujoco
    _gc.disable()
    from optuna_ident_enc import split, suggest
    from sim.step2_gap_baseline import MODEL
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    items = prep_vel(split()[0], m.opt.timestep)
    st = make_study(seed, False, seed + 1009 * rank)
    t0 = time.time()

    def obj(trial):
        L, lo, hi = eval_vel(m, items, suggest(trial))
        trial.set_user_attr("L_lo", lo); trial.set_user_attr("L_hi", hi)
        return L

    def cb(study, trial):
        if rank == 0:
            b = study.best_trial
            print(f"  seed {seed} [{len(study.trials):4d}] 本 {trial.value:.4f}"
                  f"   最优 {b.value:.4f} (lo {b.user_attrs.get('L_lo',0):.3f} /"
                  f" hi {b.user_attrs.get('L_hi',0):.3f})   {time.time()-t0:.0f}s", flush=True)

    st.optimize(obj, n_trials=n_trials, callbacks=[cb], gc_after_trial=True)
    sys.stdout.flush(); sys.stderr.flush()
    import os; os._exit(0)


def n_done(seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        return sum(1 for t in optuna.load_study(study_name=study_name(seed),
                   storage=storage(seed)).trials if t.value is not None)
    except Exception:                                        # noqa: BLE001
        return 0


def run_seed(seed, target, workers):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    make_study(seed, True)
    for att in range(1, 16):
        n = n_done(seed)
        if n >= target:
            break
        per = int(np.ceil((target - n) / workers))
        ps = [ctx.Process(target=_worker, args=(seed, i, per)) for i in range(workers)]
        for p in ps: p.start()
        for p in ps: p.join()
        n2 = n_done(seed)
        print(f"  ── seed {seed} 第 {att} 轮: {n} → {n2}/{target} ──", flush=True)
        if n2 <= n:
            print(f"  **零进展，停止重试**", flush=True); break


def best_of(seed):
    import optuna
    from optuna_ident_enc import theta_of
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tri = [t for t in optuna.load_study(study_name=study_name(seed),
           storage=storage(seed)).trials if t.value is not None]
    b = min(tri, key=lambda t: t.value)
    return b.value, theta_of(b), b.user_attrs.get("L_lo"), b.user_attrs.get("L_hi"), len(tri)


# ── 评估 ────────────────────────────────────────────────────────────────
def pp2_metrics(theta):
    """→ (全带, 0.5-3, 3-9 的 IMU 残差/实测; 关节角 RMS; 逐集全带数组)。L=1。"""
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB, A_MAX
    from sim.step2_gap_baseline import MODEL, bare_logs
    from optuna_ident_enc import accel_limited_dt, roll
    import params as P
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL)); P.apply(m, np.asarray(theta, float))
    acc = {b: [0.0, 0.0] for b in (BANDS[0], BANDS[1], (0.5, 9.0))}
    per, jr = [], []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        n = len(act); cmd = np.degrees(act[:, :6])
        ref = np.empty((n, 6))
        for j in range(6):
            ref[:, j] = accel_limited_dt(cmd[:, j], float(A_MAX[j]), np.full(n, 1 / FPS))
        ss = np.rint(np.arange(n) / FPS / m.opt.timestep).astype(int)
        it = {"ref": np.radians(ref), "enc0": enc[0],
              "g0": float(np.clip(act[0, 6], 0, 0.035)) if act.shape[1] > 6 else 0.0,
              "send_step": ss, "read_step": ss + 1}
        q, g, _ = roll(m, it)
        k = min(len(g), len(real) - 1, len(enc))
        A = real[1:1 + k]; S = g[:k] @ R.T
        for b in acc:
            acc[b][0] += float((bp(A - S, *b) ** 2).sum())
            acc[b][1] += float((bp(A, *b) ** 2).sum())
        e0 = float((bp(A - S, 0.5, 9.0) ** 2).sum()) / max(float((bp(A, 0.5, 9.0) ** 2).sum()), 1e-30)
        per.append(e0)
        jr.append(float(np.sqrt(np.mean((q[:k] - enc[:k]) ** 2))))
    r = {b: acc[b][0] / max(acc[b][1], 1e-30) for b in acc}
    return r, float(np.mean(jr)), np.array(per)


def suite_metrics(theta):
    """→ {集: (关节L, IMU, L_vel)}  套件留出 A/B。"""
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    from optuna_ident_enc import split, evaluate
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    _, ha, hb = split()
    out = {}
    for nm, names in (("留出A", ha), ("留出B", hb)):
        it = prep_vel(names, m.opt.timestep)
        r = evaluate(m, it, np.asarray(theta, float), R)
        L, lo, hi = eval_vel(m, it, theta)
        out[nm] = (r["loss"], r["imu"], L, lo, hi)
    return out


def cmd_baseline(_a):
    import mujoco
    from sim.step2_gap_baseline import MODEL
    from optuna_ident_enc import split, BASE_XML, BASE_PHYS_V1
    from g_q_v1 import THETA_STAR as TH
    gc.disable()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr = prep_vel(split()[0], m.opt.timestep)
    print("**速度目标的基线**  L = 0.5·L_vel(0.5–3) + 0.5·L_vel(3–9)\n")
    print(f"  {'配置':16s}{'训练 L':>10s}{'L_lo':>9s}{'L_hi':>9s}")
    for nm, th in (("现 XML", BASE_XML), ("G-q-v1 θ*", TH)):
        L, lo, hi = eval_vel(m, tr, th)
        print(f"  {nm:16s}{L:>10.4f}{lo:>9.4f}{hi:>9.4f}", flush=True)
    print("\n  (1.0 = 「预测速度全零」的水平)")
    return 0


def cmd_run(a):
    gc.disable()
    import mujoco
    from sim.step2_gap_baseline import MODEL
    from optuna_ident_enc import split
    from g_q_v1 import THETA_STAR as TH
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr = prep_vel(split()[0], m.opt.timestep)
    L0, lo0, hi0 = eval_vel(m, tr, TH)
    print(f"**速度目标搜索**  G-q-v1 θ* 的训练 L = {L0:.4f} (lo {lo0:.4f} / hi {hi0:.4f})")
    print(f"  预算: 第 1 段 seed {SEEDS[0]} × {a.trials}；续跑规则见预注册 §2\n", flush=True)
    run_seed(SEEDS[0], a.trials, a.workers)
    Lb, thb, blo, bhi, ntr = best_of(SEEDS[0])
    dL = 1 - Lb / L0
    print(f"\n  第 1 段最优 L = {Lb:.4f} (lo {blo:.4f} / hi {bhi:.4f})  {ntr} trial")
    print(f"  训练 L 改善 **{dL*100:+.1f}%**（续跑门槛 ≥15%）", flush=True)
    r, jr, per = pp2_metrics(thb)
    r0, jr0, per0 = pp2_metrics(TH)
    d_imu = 1 - r[(0.5, 9.0)] / r0[(0.5, 9.0)]
    print(f"  pp2 全带 IMU {r0[(0.5,9.0)]:.4f} → {r[(0.5,9.0)]:.4f}"
          f"  **{d_imu*100:+.1f}%**（续跑门槛 ≥5%）", flush=True)
    go = dL >= 0.15 and d_imu >= 0.05
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1.json").write_text(json.dumps(
        {"L0": L0, "L_best": Lb, "dL": dL, "pp2_imu": [r0[(0.5,9.0)], r[(0.5,9.0)]],
         "d_imu": d_imu, "continue": bool(go), "theta": thb.tolist()},
        ensure_ascii=False, indent=2))
    print(f"\n⇒ " + ("**满足续跑条件 ⇒ 进入第 2 段（seed B/C）**" if go else
          "**不满足续跑条件 ⇒ 不跑第 2 段，直接按门禁判否**"))
    if go and not a.stage1_only:
        for sd in SEEDS[1:]:
            run_seed(sd, a.trials, a.workers)
    return cmd_report(a)


def cmd_report(_a):
    gc.disable()
    from g_q_v1 import THETA_STAR as TH
    import mujoco
    from sim.step2_gap_baseline import MODEL
    from optuna_ident_enc import split
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    tr = prep_vel(split()[0], m.opt.timestep)
    L0, lo0, hi0 = eval_vel(m, tr, TH)
    got = [(sd,) + best_of(sd) for sd in SEEDS if n_done(sd) > 0]
    print(f"\n{'='*82}\n**结果**  基线 G-q-v1 θ*: 训练 L {L0:.4f} (lo {lo0:.4f}/hi {hi0:.4f})\n")
    for sd, Lb, th, blo, bhi, ntr in got:
        print(f"  seed {sd}: {ntr} trial  最优 L {Lb:.4f} "
              f"(lo {blo:.4f} / hi {bhi:.4f})  改善 {(1-Lb/L0)*100:+.1f}%")
    sd, Lb, TB, blo, bhi, _ = min(got, key=lambda x: x[1])
    print(f"\n  选中 seed {sd}（训练 L 最低）")
    r0, jr0, p0 = pp2_metrics(TH)
    r1, jr1, p1 = pp2_metrics(TB)
    s0, s1 = suite_metrics(TH), suite_metrics(TB)
    print(f"\n  {'指标':26s}{'G-q-v1':>11s}{'速度目标':>11s}{'变化':>9s}")
    rows = [("pp2 IMU 全带", r0[(0.5,9.0)], r1[(0.5,9.0)]),
            ("pp2 IMU 0.5–3 Hz", r0[BANDS[0]], r1[BANDS[0]]),
            ("pp2 IMU 3–9 Hz", r0[BANDS[1]], r1[BANDS[1]]),
            ("pp2 关节角 RMS (度)", jr0, jr1),
            ("留出A IMU", s0["留出A"][1], s1["留出A"][1]),
            ("留出B IMU", s0["留出B"][1], s1["留出B"][1]),
            ("留出A 关节角 L", s0["留出A"][0], s1["留出A"][0]),
            ("留出A 关节速度 L", s0["留出A"][2], s1["留出A"][2])]
    for nm, a_, b_ in rows:
        print(f"  {nm:26s}{a_:>11.4f}{b_:>11.4f}{(b_/a_-1)*100:>8.1f}%")
    d = (p0 - p1) / p0
    w, med = int((d > 0).sum()), float(np.median(d))
    print(f"  {'pp2 逐集改善':26s}{'':>11s}{w:>8d}/{len(d):<3d}{med*100:>8.1f}%(中位)")
    ok = {
        1: r1[(0.5,9.0)] <= r0[(0.5,9.0)] * 0.90,
        2: r1[BANDS[0]] <= r0[BANDS[0]] * 0.90,
        3: w >= 9 and med > 0,
        4: s1["留出A"][1] <= s0["留出A"][1] * 1.05 and s1["留出B"][1] <= s0["留出B"][1] * 1.05,
        5: jr1 <= jr0 * 1.10,
        6: s1["留出A"][2] <= s0["留出A"][2] * 0.95,
        7: r1[BANDS[1]] <= r0[BANDS[1]] * 1.05}
    if len(got) >= 3:
        Ls = [g[1] for g in got]
        ok[8] = (max(Ls) / min(Ls) - 1) <= 0.10
    txt = {1: "pp2 全带 IMU 改善 ≥10%", 2: "pp2 0.5–3 Hz 改善 ≥10%",
           3: "pp2 逐集 ≥9/17 且中位>0", 4: "留出 A/B IMU 不恶化 >5%",
           5: "pp2 关节角 RMS 不恶化 >10%", 6: "留出A 关节速度改善 ≥5%",
           7: "3–9 Hz 不恶化 >5%", 8: "跨 seed 最优 L 极差 ≤10%"}
    print(f"\n{'='*82}\n**门禁**（预注册）")
    for k in sorted(ok):
        print(f"  {k} {txt[k]:34s}{'✓' if ok[k] else '**✗**'}")
    good = all(ok.values())
    print(f"\n⇒ " + ("**通过** —— 冻结为新候选并出 pp2 波形图" if good else
          "**未通过** ⇒ 记录「参数层在角度与速度两种目标下均已到顶」，"
          "43% 的可达空间不能由参数调优取得；本线收口。"))
    (OUT / "gate.json").write_text(json.dumps(
        {"L0": L0, "seeds": [{"seed": g[0], "L": g[1], "n": g[5]} for g in got],
         "rows": {n: [a_, b_] for n, a_, b_ in rows},
         "pp2_wins": w, "pp2_median": med,
         "gate": {str(k): bool(v) for k, v in ok.items()}, "passed": bool(good),
         "theta_best": TB.tolist()}, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'gate.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stage1-only", action="store_true")
    a = ap.parse_args()
    gc.disable()
    if a.baseline: return cmd_baseline(a)
    if a.run: return cmd_run(a)
    if a.report: return cmd_report(a)
    print(__doc__.split("用法:")[0]); return 0


if __name__ == "__main__":
    _c = main(); sys.stdout.flush(); import os; os._exit(_c)
