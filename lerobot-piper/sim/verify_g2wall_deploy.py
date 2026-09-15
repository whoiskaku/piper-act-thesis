#!/usr/bin/env python3
"""**部署现实性三问**：`pi-g2wall-v1` 从离线口径搬到真机上要付多少代价？（零机时）

离线冻结的 **16.92%** 是在**整集指令流**上一次跑 G-2 得到的。真机上做不到两件事：

1. **逐 chunk 前向** —— 只有当前 chunk 的 100 帧指令，G-2 的伺服 rollout 与 FIR
   历史在 chunk 边界被截断 ⇒ 需要**热身前缀** `P`（把上一 chunk 尾部的指令接上）。
   ⚠ 这**不是因果性问题**：chunk 内的 100 帧指令在 t₀ 全部已知，
   `np.gradient` 的 ±1 帧前瞻与 ideal 渲染同理**合法**。问题只在**跨 chunk 的状态延续**。
2. **后台算完才能用** —— G-2 前向实测 ~210–290 ms、ideal 渲染 ~200 ms，
   合计约 **8–10 帧 @20 Hz**。这期间 δa 只能置零 ⇒ 每个 chunk **头部有空档**，
   而 chunk 头恰是接缝瞬态最糟的地方。

另有一处口径差：`ê₀[k]` 需要 `ω̂[k+ALIGN_L]`，chunk 末尾 3 帧越界 ⇒ **钳位到 T−1**。
（那几帧求解器的 horizon `K = min(M, T−1−k) ≤ 2`，本就近乎不动。）

本脚本把三者叠加，给出**真机上真正能指望的数字**。

口径
----
- 数据 = `ab4 P0` 5 集（无探针 ⇒ `e₀ = gap`，H 不进 `e₀`），与冻结时同一批
- G-2 用**原始冻结件 v=0**（`G2v0`，与 `pi-g2wall-v1` manifest 一致）
- 求解器、ε=10 mrad 峰值、chunk 钳位、接触门控、`ALIGN_L=3` 全部沿用 `delivery_g2_predictor`
- P0 的 `a` 即原始 ACT 指令（δa≡0）⇒ 喂给 G-2 的指令流与部署时的 `_chunk_raw` 同义

用法:
    python -u sim/verify_g2wall_deploy.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                    # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402
import delivery_g2_predictor as DG                                  # noqa: E402
from freeze_pi_g2wall import G2v0                                   # noqa: E402

FPS = 20.0
PREFIXES = (0, 10, 20, 40, 80)
BLACKOUTS = (0, 4, 8, 12, 16, 20)
OUT = ROOT / "outputs/p2_pi/g2wall_deploy_verify.json"


def sub_dd(cmd: np.ndarray) -> dict:
    """指令片段 → G-2 前向所需的 `dd`（均匀 20 Hz；`gyro100` 不进计算路径）。"""
    n = len(cmd)
    t = np.arange(n) / FPS
    return {"name": "sub", "cmd": np.asarray(cmd, float), "t_send": t, "t_read": t,
            "tg_i": np.arange(n * 5) / (FPS * 5), "gyro100": np.zeros((n * 5, 3))}


def chunk_spans(cap: np.ndarray) -> list[tuple[int, int]]:
    """`cap[t]` = 本 chunk 末帧的全局下标 → [(start, end_exclusive), ...]。"""
    spans, s = [], 0
    n = len(cap)
    while s < n:
        e = int(cap[s]) + 1
        spans.append((s, min(e, n)))
        s = e
    return spans


def omega_whole(g2, cmd: np.ndarray) -> np.ndarray:
    return g2.omega(sub_dd(cmd))


def omega_perchunk(g2, cmd: np.ndarray, spans, prefix: int) -> np.ndarray:
    """逐 chunk 前向 + 热身前缀 P，拼回整集长度。"""
    out = np.zeros((len(cmd), 3))
    for s, e in spans:
        a = max(0, s - prefix)
        w = g2.omega(sub_dd(cmd[a:e]))
        take = w[s - a: s - a + (e - s)]
        out[s:s + len(take)] = take
    return out


def e0hat_from_omega(w: np.ndarray, ideal: np.ndarray, spans, R32, b32,
                     per_chunk_tail: bool) -> np.ndarray:
    """ω̂(机体) → 标定 → `ê₀[k] = calib(ω̂)[k+3] − ideal[k]`。

    `per_chunk_tail=True` 时，`k+3` **在 chunk 内钳位**（部署时拿不到下一 chunk 的 ω̂）。
    """
    w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
    ws = S1.calibrate_ideal_imu(w6, R32, b32)[:, 3:].astype(float)
    n = len(ideal)
    out = np.zeros((n, 3))
    if per_chunk_tail:
        for s, e in spans:
            for k in range(s, min(e, n)):
                out[k] = ws[min(k + S1.ALIGN_L, e - 1)] - ideal[k]
    else:
        for k in range(n):
            out[k] = ws[min(k + S1.ALIGN_L, len(ws) - 1)] - ideal[k]
    return out


def roll_blackout(e, H, pred, spans, nblack: int):
    """`DG.roll` + 每个 chunk 头 `nblack` 帧强制 δa=0（后台未算完）。"""
    n, M1 = e["n"], S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = DG.KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    black = np.zeros(n, bool)
    for s, _e in spans:
        black[s:min(s + nblack, n)] = True
    da = np.zeros((n, 6)); resp = np.zeros((n + M1, 3)); nsat = 0
    for t in range(n):
        K = max(int(min(S1.M, e["cap"][t] - t)), 0)
        tgt = np.stack([pred[min(t + k, n - 1)] + resp[t + k] for k in range(K + 1)])
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        nsat += int(np.any(np.abs(d) > DG.EPS))
        d = np.clip(d, -DG.EPS, DG.EPS)
        if e["contact"][t] or black[t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da, nsat / max(n, 1)


def predictability(E, preds):
    n_ = d_ = 0.0
    for e, p in zip(E, preds):
        a, b = S1.bandpass41(e["e0"]), S1.bandpass41(p)
        n_ += float(np.square(a - b).sum()); d_ += float(np.square(a).sum())
    return 1 - n_ / max(d_, 1e-30)


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R32 = np.asarray(calib["R"], np.float32)
    b32 = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    g2 = G2v0()

    E = []
    for p in S1.discover_logs(ROOT / "outputs/ab4", ("P0",))["P0"]:
        ep = S1.load_episode(p, "P0", np.asarray(R32, np.float64),
                             np.asarray(b32, np.float64), rend, 550)
        pay = json.loads(p.read_text())
        cmdlog, ideal_sim, cap = S1._render_logged_chunks(
            pay.get("chunks", []), rend, len(pay["frames"]))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R32, b32)[:, 3:6]
        n = min(len(ep.gap), len(ideal))
        dd = G.ab4_to_dd(p, p.stem)
        E.append({"name": p.stem, "n": n, "e0": np.asarray(ep.gap, float)[:n],
                  "ideal": ideal[:n].astype(float),
                  "contact": np.asarray(ep.contact_mask, bool)[:n],
                  "cap": np.asarray(cap[:n], int),
                  "cmd": dd["cmd"], "spans": chunk_spans(np.asarray(cap[:n], int))})

    print(f"ab4 P0 {len(E)} 集 · chunk = {E[0]['spans'][1][0] - E[0]['spans'][0][0]} 帧 · "
          f"G-2 = **原始冻结 v=0**（与 manifest 一致）")
    print(f"⚠ chunk 内的 ±1 帧前瞻**合法**（整 chunk 指令在 t₀ 已知）；"
          f"本检验查的是**跨 chunk 状态延续**与**后台延迟**\n")

    # ── 参考：整集前向（= 冻结 16.92% 的口径）──────────────────────────
    ref = [e0hat_from_omega(omega_whole(g2, e["cmd"]), e["ideal"], e["spans"],
                            R32, b32, per_chunk_tail=False) for e in E]
    pr_ref = predictability(E, ref)
    r_ref, rms_ref, sa_ref = DG.score(
        E, H, [DG.roll(e, H, p) for e, p in zip(E, ref)])
    print(f"  **参考（整集前向，冻结口径）**  可预测性 {pr_ref*100:.1f}%  "
          f"交付 **{r_ref*100:.2f}%**  δa RMS {rms_ref:.2f} mrad  触界 {sa_ref:.2f}%\n")

    # ── ① 热身前缀 P：逐 chunk 前向能否复现整集前向？ ──────────────────
    print(f"  ── ① 热身前缀（逐 chunk 前向 vs 整集前向）──")
    print(f"  {'P (帧)':<9}{'ê₀ 最大绝对差':>15}{'ê₀ 相对误差':>14}"
          f"{'可预测性':>10}{'交付':>10}{'前向 ms/chunk':>15}")
    best_P, res_P = None, {}
    for P in PREFIXES:
        t0 = time.perf_counter()
        dep = [e0hat_from_omega(omega_perchunk(g2, e["cmd"], e["spans"], P),
                                e["ideal"], e["spans"], R32, b32, per_chunk_tail=True)
               for e in E]
        ms = (time.perf_counter() - t0) * 1e3 / sum(len(e["spans"]) for e in E)
        mx = max(float(np.abs(a[:e["n"]] - b[:e["n"]]).max())
                 for a, b, e in zip(dep, ref, E))
        num = sum(float(np.square(a[:e["n"]] - b[:e["n"]]).sum())
                  for a, b, e in zip(dep, ref, E))
        den = sum(float(np.square(b[:e["n"]]).sum()) for b, e in zip(ref, E))
        rel = np.sqrt(num / max(den, 1e-30))
        pr = predictability(E, dep)
        r, _, _ = DG.score(E, H, [DG.roll(e, H, p) for e, p in zip(E, dep)])
        res_P[P] = {"max_abs": mx, "rel_err": rel, "predictability": pr,
                    "reduction": r, "ms_per_chunk": ms}
        print(f"  {P:<9}{mx:>15.3e}{rel*100:>13.2f}%{pr*100:>9.1f}%{r*100:>9.2f}%{ms:>15.1f}")
        if best_P is None or rel < res_P[best_P]["rel_err"]:
            best_P = P
    print(f"  ⇒ 选 **P = {best_P}**（相对误差 {res_P[best_P]['rel_err']*100:.2f}%）\n")

    # ── ② 头部空档：后台算完前 δa 必须置零 ────────────────────────────
    dep_best = [e0hat_from_omega(omega_perchunk(g2, e["cmd"], e["spans"], best_P),
                                 e["ideal"], e["spans"], R32, b32, per_chunk_tail=True)
                for e in E]
    print(f"  ── ② chunk 头部空档（后台 G-2 + ideal 渲染的延迟代价）──")
    print(f"  {'空档 (帧)':<11}{'= ms':>8}{'交付':>10}{'相对参考':>11}{'δa RMS':>9}{'触界%':>8}")
    res_B = {}
    for nb in BLACKOUTS:
        das = [roll_blackout(e, H, p, e["spans"], nb) for e, p in zip(E, dep_best)]
        r, rms, sa = DG.score(E, H, das)
        res_B[nb] = {"reduction": r, "da_rms_mrad": rms, "sat_pct": sa}
        print(f"  {nb:<11}{nb*50:>8}{r*100:>9.2f}%{r/max(r_ref,1e-12)*100:>10.0f}%"
              f"{rms:>9.2f}{sa:>8.2f}")

    print(f"\n  ── 判读 ──")
    r10 = res_B[8]["reduction"] if 8 in res_B else None
    print(f"  参考（离线冻结口径）            **{r_ref*100:.2f}%**")
    print(f"  逐 chunk 前向（P={best_P}，无空档）  {res_P[best_P]['reduction']*100:.2f}%"
          f"（{(res_P[best_P]['reduction']-r_ref)*100:+.2f} pp）")
    if r10 is not None:
        print(f"  **+ 8 帧空档（≈400 ms 后台）**   **{r10*100:.2f}%**"
              f"（{(r10-r_ref)*100:+.2f} pp，占参考 {r10/max(r_ref,1e-12)*100:.0f}%）")
    print(f"\n  ⚠ 仍是**离线**数字，被控对象是 H（模型），**非真机验证**。")
    print(f"  ⚠ **换会话风险独立于此**：ab4 P0 裸 52.1% → ab2 A 裸 26.3%，"
          f"且已证**不是标定问题**（重拟 R 只从 26.3% 动到 25.4%）。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"reference": {"predictability": pr_ref, "reduction": r_ref,
                       "da_rms_mrad": rms_ref, "sat_pct": sa_ref},
         "prefix_sweep": {str(k): v for k, v in res_P.items()},
         "chosen_prefix": best_P,
         "blackout_sweep": {str(k): v for k, v in res_B.items()},
         "caliber": ("ab4 P0 5 集(e0=gap); G-2=**原始冻结 v=0**; "
                     "求解器/ε=10mrad 峰值/chunk 钳位/接触门控/ALIGN_L=3 沿用 "
                     "delivery_g2_predictor; chunk 内 ±1 帧前瞻**合法**(整 chunk 指令 t0 已知), "
                     "本检验查**跨 chunk 状态延续**与**后台延迟**; "
                     "chunk 末 3 帧 ê0 的 k+3 **在 chunk 内钳位**(部署拿不到下一 chunk 的 omega); "
                     "**离线数字, 被控对象是 H 模型, 非真机验证**"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
