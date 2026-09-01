#!/usr/bin/env python3
"""**带权目标重辨识**（G-q-v2 候选）—— 预注册见 `PREREG_BAND_WEIGHTED_IDENT.md`。

唯一的改动：损失里加一个**同权的 3–9 Hz 带内项**。

    L = 0.5 · L_broad + 0.5 · L_39

    L_broad = mean_j [ RMS(err_j) / travel_j ]                      # 与冻结版逐字相同
    L_39    = mean_j [ RMS(bp(err_j,3–9)) / RMS(bp(enc_j,3–9)) ]    # 1.0 = 带内预测全零

为什么要加
----------
冻结版 `optuna_ident_enc.evaluate` 的损失是**宽带位置 RMS，没有任何带通**，
而 3–9 Hz 只占关节位置能量的 0.003%–0.12%(pp2) / 中位 0.66%(套件)
⇒ 权重是万分之一到千分之几量级，**23 个参数从没被要求拟合这个带**。
0.5–3 Hz 位置 ρ²=0.999–1.000 正是目标优化出来的；3–9 Hz 的 0.73–0.93 是顺带掉出来的。

其余一个字节不改
----------------
本模块**不复制**管线：`split` / `prep` / `roll` / 搜索空间 / 事件网格 / 读数惯例 /
实际 `t_send`/`t` / kp6 gauge / 崩溃重启，全部直接用 `optuna_ident_enc` 的实现，
在**导入时**打补丁。补丁必须在导入时生效 —— worker 是 spawn 出去的，
父进程的猴补丁不会传到子进程（子进程 import 本模块时才拿到）。

`evaluate_band` 为了不把 `roll` 跑两遍而重写了聚合部分，
故 `--selftest` **断言它的 L_broad 与冻结版 `evaluate` 的 loss 逐位相等**。

用法:
    python -u sim/step2_g/optuna_ident_band.py --selftest
    python -u sim/step2_g/optuna_ident_band.py --run
    python -u sim/step2_g/optuna_ident_band.py --report
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import optuna_ident_enc as E                                        # noqa: E402

BAND = (3.0, 9.0)
W_BAND = 0.5                    # **事前定死，不调**（两项已各自无量纲化）
PAD = 20                        # 丢掉带通的边缘瞬态（41 抽头 FIR 的半宽）
RMS_FLOOR_DEG = 0.005           # 带内 RMS 低于此的关节不计入 L_39（5×编码器 LSB）
NEW_OUT = ROOT / "outputs/step2_g/optuna_band"

_PREP, _EVAL, _NAME = E.prep, E.evaluate, E.study_name


def _bp(x: np.ndarray) -> np.ndarray:
    from scipy.signal import firwin
    k = firwin(41, list(BAND), fs=E.FPS, pass_zero=False)
    return np.convolve(x, k, mode="same")


def prep_band(names, mj_dt: float = 0.002):
    """在冻结版 `prep` 的产物上追加带内参考量。原字段一个不动。"""
    items = _PREP(names, mj_dt)
    for it in items:
        enc = it["enc"]
        bpe = np.stack([_bp(enc[:, j]) for j in range(6)], 1)
        sl = slice(PAD, -PAD) if len(enc) > 2 * PAD + 10 else slice(None)
        it["bp_slice"] = sl
        it["bp_rms"] = np.array([float(np.sqrt(np.mean(bpe[sl, j] ** 2)))
                                 for j in range(6)])
        it["bp_enc"] = bpe
    return items


def evaluate_band(m, items, theta, R, want_diag: bool = False) -> dict:
    """→ {loss, per, imu, broad, band, (diag)}。发散 ⇒ 全 PENALTY。

    `loss` 是加权后的量；`broad` 是与冻结版 `evaluate` **逐位相同**的宽带损失
    （`--selftest` 会断言这一点）。
    """
    import params as P
    from sim.step1_v2_loss import bandpass41
    theta = np.asarray(theta, float)
    if theta.shape != (P.N_P,):
        raise TypeError(f"theta 形状应为 ({P.N_P},)，收到 {theta.shape}")
    if np.shape(R) != (3, 3):
        raise TypeError(f"R 形状应为 (3,3)，收到 {np.shape(R)}")
    P.apply(m, theta)                                # **kp6 由 params 压回 gauge**
    per, band, imu_per = [], [], []
    for it in items:
        try:
            q, g, q_eq = E.roll(m, it)
        except Exception:                            # noqa: BLE001
            return {"loss": E.PENALTY, "per": np.full(len(items), E.PENALTY),
                    "imu": E.PENALTY, "broad": E.PENALTY, "band": E.PENALTY,
                    "diag": []}
        if not (np.isfinite(q).all() and np.isfinite(g).all()):
            return {"loss": E.PENALTY, "per": np.full(len(items), E.PENALTY),
                    "imu": E.PENALTY, "broad": E.PENALTY, "band": E.PENALTY,
                    "diag": []}
        err = q - it["enc"]
        # ① 宽带项 —— 与 optuna_ident_enc.evaluate 逐字相同
        ej = [float(np.sqrt(np.mean(err[:, j] ** 2))) / max(it["travel"][j], 1e-9)
              for j in it["target"]]
        per.append(min(float(np.mean(ej)), E.PENALTY))
        # ② 带内项 —— 按**带内实测 RMS** 归一 ⇒ 1.0 = 带内预测全零
        sl = it["bp_slice"]
        bj = []
        for j in it["target"]:
            if it["bp_rms"][j] < RMS_FLOOR_DEG:      # 带内几乎无信号, 不拟合噪声
                continue
            e = _bp(err[:, j])[sl]
            bj.append(float(np.sqrt(np.mean(e ** 2))) / it["bp_rms"][j])
        band.append(min(float(np.mean(bj)), E.PENALTY) if bj else np.nan)
        b = bandpass41(g @ R.T)
        imu_per.append(min(float(np.square(it["abp"] - b).sum())
                           / max(it["ten"], 1e-12), E.PENALTY))
    per = np.array(per)
    band = np.array(band)
    Lb = float(per.mean())
    L9 = float(np.nanmean(band)) if np.isfinite(band).any() else E.PENALTY
    return {"loss": (1 - W_BAND) * Lb + W_BAND * L9, "per": per,
            "imu": float(np.mean(imu_per)), "broad": Lb, "band": L9, "diag": []}


# ── 导入时打补丁（spawn 的子进程 import 本模块时同样生效）────────────────
E.prep = prep_band
E.evaluate = evaluate_band
E.OUT = NEW_OUT
E.study_name = lambda seed: f"g_ident_band_s{seed}"


def _worker_band(seed: int, rank: int, n_trials: int) -> None:
    """spawn 入口。**必须在本模块里**，这样子进程 import 时补丁才生效。"""
    E._worker(seed, rank, n_trials)


def cmd_selftest() -> int:
    """断言：新损失的宽带项与冻结版 `evaluate` 的 loss **逐位相等**。"""
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    from g_q_v1 import THETA_STAR
    R = load_calib(CALIB)[0]
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    names = E.split()[0][:6]
    items = prep_band(names, m.opt.timestep)
    a = _EVAL(m, items, THETA_STAR, R)                # 冻结版
    b = evaluate_band(m, items, THETA_STAR, R)        # 本模块
    print(f"  冻结版 loss      {a['loss']!r}")
    print(f"  本模块 broad     {b['broad']!r}")
    assert a["loss"] == b["broad"], "**宽带项与冻结版不一致 —— 管线已分叉，停止**"
    assert np.array_equal(a["per"], b["per"]), "**逐条宽带损失不一致**"
    print(f"  ✓ 宽带项逐位相等（{len(names)} 条）")
    print(f"  带内项 L_39      {b['band']:.4f}   (1.0 = 带内预测全零)")
    print(f"  加权 loss        {b['loss']:.4f}  = 0.5×{b['broad']:.4f} + 0.5×{b['band']:.4f}")
    print(f"  IMU(不参与选参)  {b['imu']:.4f}")
    nb = sum(1 for it in items for j in it["target"] if it["bp_rms"][j] >= RMS_FLOOR_DEG)
    nt = sum(len(it["target"]) for it in items)
    print(f"  带内有效关节     {nb}/{nt}（RMS ≥ {RMS_FLOOR_DEG} 度）")
    return 0


def cmd_run(trials: int, workers: int) -> int:
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tr, ha, hb = E.split()
    print("**带权目标重辨识**  L = 0.5·L_broad + 0.5·L_39(3–9 Hz)")
    print(f"  自由量 {len(E.free_names())} 个（kp6 = {E.__dict__.get('P', None) and ''}50.0 固定为 gauge）")
    print(f"  训练 {len(tr)} · 留出A {len(ha)} · 留出B {len(hb)}   ← 留出集本阶段**不评估**")
    print(f"  预算**固定**: {len(E.SEEDS)} seed × {trials} trial，结束后不追加")
    print(f"  权重 W_BAND = {W_BAND} **事前定死**，不用留出集挑\n")
    for seed in E.SEEDS:
        _run_seed(seed, trials, workers, ctx)
    return 0


def _run_seed(seed: int, target: int, workers: int, ctx) -> None:
    """与 `E._run_seed` 同逻辑，只是 spawn 本模块的 `_worker_band`。"""
    E.make_study(seed, create=True)
    for att in range(1, 21):
        n = E._n_done(seed)
        if n >= target:
            break
        per = int(np.ceil((target - n) / workers))
        ps = [ctx.Process(target=_worker_band, args=(seed, i, per))
              for i in range(workers)]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        bad = [p.exitcode for p in ps if p.exitcode != 0]
        n2 = E._n_done(seed)
        print(f"  ── seed {seed} 第 {att} 轮: {n} → {n2}/{target}"
              f"   非零退出 {len(bad)}/{workers} {sorted(set(bad))[:4]} ──", flush=True)
        if n2 <= n:
            print(f"  **seed {seed} 一轮零进展，停止重试**", flush=True)
            break
    print(f"  ── seed {seed} 收尾: {E._n_done(seed)} trial ──\n", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--trials", type=int, default=E.N_TRIALS)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.selftest:
        return cmd_selftest()
    if a.status:
        for seed in E.SEEDS:
            print(f"  seed {seed}: {E._n_done(seed)} trial")
        return 0
    if a.run:
        return cmd_run(a.trials, a.workers)
    ap.print_help()
    return 0


if __name__ == "__main__":
    import gc
    gc.disable()                                     # mujoco+scipy 的 GC 段错误
    _c = main()
    sys.stdout.flush(); sys.stderr.flush()
    import os
    os._exit(_c)
