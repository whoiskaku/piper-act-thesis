#!/usr/bin/env python3
"""**训练—部署逐帧一致性**核验：`G2WallRuntime` 是否与离线口径给出同一条 δa？（零机时）

查五件事（任一不过就**不能上机**）
----------------------------------
1. **模型缓存无副作用**：`install_model_cache()` 后 G-2 前向与原路径**逐位一致**
   （缓存是发深拷贝，因为 `_sim` 会把冻结 θ 写进模型）
2. **前向路径一致**：运行时内联的 `_omega` 与冻结件 `G2v0.omega` 一致
3. **ideal 必须整 chunk 渲染**：`IdealChunkRenderer` 用 `filtfilt`（零相位、非因果），
   渲染前缀 ≠ 整渲染的前缀 ⇒ stage A 只能截断 **G-2 的前向**，ideal 必须整渲染
4. **stage A == stage B**（在**求解器读得到的整个范围** `[0, W_A+M]` 上）
   —— 两段式不引入额外损失。⚠ 只比 `[0, W_A)` 会漏掉中心差分的**尾部污染**
5. **逐帧 δa 与离线一致**，且交付量复现 `verify_g2wall_deploy` 的 15.50%。
   ⚠ **只比完整 chunk**：末尾被集末截断的那个 chunk 上，离线 (a) 把 G-2 前向也截断
   ⇒ ω̂ 尾部被中心差分边界污染，(b) 把 horizon 钳在集末。两处都是**离线重放的产物**，
   运行时跑的是完整 chunk，**运行时为准**

口径
----
- 数据 = `ab4 P0` 5 集（`e₀ = gap`），与冻结、与部署现实性检验同一批
- 参考 = `verify_g2wall_deploy`：**P=20 逐 chunk 前向 + 无空档**（15.50%）
- P0 的 `chunks[].raw` 即原始 ACT 指令（δa≡0）⇒ 与部署时的 `_chunk_raw` 同义

用法:
    python -u sim/verify_g2wall_runtime.py
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
import delivery_g2_predictor as DG                                  # noqa: E402
import verify_g2wall_deploy as V                                    # noqa: E402

OUT = ROOT / "outputs/p2_pi/g2wall_runtime_verify.json"
TOL_BIT = 0.0          # 模型缓存必须逐位一致
TOL_DA = 1e-9          # δa 一致性（纯 float64 重算，允许求解方式差异）


def main() -> int:
    import gc
    gc.disable()
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    from lerobot.common.policies.act.g2wall_runtime import (
        G2WallRuntime, N_A, W_A, PREFIX, M, TAIL_GUARD, install_model_cache)
    from lerobot.common.policies.act.step1_correction_runtime import calibrate_ideal_imu
    from freeze_pi_g2wall import G2v0

    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R32 = np.asarray(calib["R"], np.float32)
    b32 = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    rend = IdealChunkRenderer()
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    ok = {}

    # ── 检 1：模型缓存逐位一致 ────────────────────────────────────────
    g2 = G2v0()
    probe = np.zeros((PREFIX + N_A, 7))
    probe[:, :6] = 0.02 * np.sin(np.arange(PREFIX + N_A)[:, None] / 7.0
                                 + np.arange(6)[None, :])
    w_before = g2.omega(V.sub_dd(probe))
    t0 = time.perf_counter(); g2.omega(V.sub_dd(probe)); ms_nocache = (time.perf_counter() - t0) * 1e3
    install_model_cache()
    w_after = g2.omega(V.sub_dd(probe))
    t0 = time.perf_counter(); g2.omega(V.sub_dd(probe)); ms_cache = (time.perf_counter() - t0) * 1e3
    d1 = float(np.abs(w_before - w_after).max())
    ok["1_model_cache_bitwise"] = d1 <= TOL_BIT
    print(f"  检 1 模型缓存逐位一致          最大差 {d1:.3e}   "
          f"{'✓' if ok['1_model_cache_bitwise'] else '✗'}")
    print(f"        前向耗时 {ms_nocache:.0f} ms → **{ms_cache:.0f} ms**"
          f"（{PREFIX + N_A} 帧，{ms_nocache/max(ms_cache,1e-9):.1f}× 提速）")

    # ── 检 2：运行时 _omega == 冻结件 omega ───────────────────────────
    rt = G2WallRuntime(ROOT / "outputs/p2_pi/pi_g2wall_v1.json", R32, b32, verbose=True)
    d2 = float(np.abs(rt._omega(probe) - g2.omega(V.sub_dd(probe))).max())
    ok["2_omega_path"] = d2 <= 1e-12
    print(f"  检 2 运行时前向 == 冻结件前向  最大差 {d2:.3e}   "
          f"{'✓' if ok['2_omega_path'] else '✗'}")

    # ── 载入 5 集 ────────────────────────────────────────────────────
    E = []
    for p in S1.discover_logs(ROOT / "outputs/ab4", ("P0",))["P0"]:
        ep = S1.load_episode(p, "P0", np.asarray(R32, np.float64),
                             np.asarray(b32, np.float64), rend, 550)
        pay = json.loads(p.read_text())
        chunks = sorted(pay["chunks"], key=lambda c: int(c["start"]))
        cmdlog, ideal_sim, cap = S1._render_logged_chunks(chunks, rend, len(pay["frames"]))
        ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R32, b32)[:, 3:6]
        n = min(len(ep.gap), len(ideal))
        E.append({"name": p.stem, "n": n, "e0": np.asarray(ep.gap, float)[:n],
                  "ideal": ideal[:n].astype(float),
                  "contact": np.asarray(ep.contact_mask, bool)[:n],
                  "cap": np.asarray(cap[:n], int),
                  "cmd": cmdlog, "chunks": chunks,
                  "spans": V.chunk_spans(np.asarray(cap[:n], int))})

    # ── 检 3：ideal **必须整 chunk 渲染**（不能只渲头部）─────────────
    # `IdealChunkRenderer` 用 CubicSpline + **filtfilt（零相位、非因果）**，
    # 所以渲染前缀 ≠ 整渲染的前缀。这正是冻结口径（`_render_logged_chunks` 逐 chunk
    # 整渲染），故 stage A 也必须拿**整 chunk 的 ideal**，只截断 G-2 的前向。
    # 本检验证实该约束真实存在，并核对整渲染的耗时在一帧预算内。
    d3 = 0.0
    for e in E[:2]:
        prev = None
        for c in e["chunks"]:
            raw = np.asarray(c["raw"], np.float64)
            full = np.asarray(rend.render(raw, prev), np.float64)
            head = np.asarray(rend.render(raw[:N_A], prev), np.float64)
            d3 = max(d3, float(np.abs(full[:N_A] - head).max()))
            prev = raw
    t0 = time.perf_counter()
    for _ in range(5):
        rend.render(np.asarray(E[0]["chunks"][1]["raw"], np.float32), None)
    ms_render = (time.perf_counter() - t0) * 1e3 / 5
    ok["3_full_chunk_ideal"] = d3 > 1e-6 and ms_render <= 40.0
    print(f"  检 3 ideal 须整 chunk 渲染     部分渲染差 {d3:.2e}（**确实不可截断**）"
          f" · 整渲染 {ms_render:.1f} ms   {'✓' if ok['3_full_chunk_ideal'] else '✗'}")

    # ── 检 4+5：逐帧重放 ─────────────────────────────────────────────
    ref = [V.e0hat_from_omega(V.omega_perchunk(g2, e["cmd"], e["spans"], rt.prefix),
                              e["ideal"], e["spans"], R32, b32, per_chunk_tail=True)
           for e in E]
    ref_da = [V.roll_blackout(e, H, p, e["spans"], 0)[0] for e, p in zip(E, ref)]
    r_ref, rms_ref, sa_ref = DG.score(E, H, [(d, 0.0) for d in ref_da])

    d4 = d5 = 0.0
    run_da, sums = [], []
    for e in E:
        rt.reset()
        da = np.zeros((e["n"], 6))
        prev = None
        for c in e["chunks"]:
            s = int(c["start"])
            if s >= e["n"]:
                break
            raw = np.asarray(c["raw"], np.float64)
            ideal_full = calibrate_ideal_imu(
                np.asarray(rend.render(raw, prev), np.float32), R32, b32)[:, 3:6].astype(float)
            # stage A：**整 chunk 的 ideal**（filtfilt 非因果，不可截断），
            # 只截断 G-2 的前向（头 N_A 帧 + 热身前缀）
            rt.begin_chunk(raw)
            rt.fill_stage_a(ideal_full)
            # 比到**求解器读得到的最远处** k+M（k 最大 = W_A−1）——
            # 只比 [:W_A] 会漏掉尾部污染（第一版就是这么漏的）
            look = min(rt.w_a + M, rt._plan.valid_to)
            a_head = rt._plan.e0hat[:look].copy()
            # 先跑 W_A 帧（stage A 供给），再上 stage B，验证两段在重叠处相同
            for k in range(min(rt.w_a, e["n"] - s)):
                da[s + k] = rt.step().delta
            rt.fill_stage_b(ideal_full)
            d4 = max(d4, float(np.abs(rt._plan.e0hat[:look] - a_head).max()))
            for k in range(rt.w_a, min(len(raw), e["n"] - s)):
                da[s + k] = rt.step().delta
            prev = raw
        run_da.append(da)
        sums.append(rt.summary())
    # ⚠ **最后一个 chunk（被集末截断的那个）要排除**，理由是两处口径差，
    # 两处都是「离线重放的产物」而非部署行为：
    #   (a) 离线 `chunk_spans` 把末 chunk 截到集末才跑 G-2 ⇒ 它的 ω̂ 尾部被
    #       中心差分边界污染；运行时跑的是**完整 100 帧 chunk**，没有这个问题。
    #   (b) 离线 `roll` 把 horizon 钳在集末 (`pred[min(t+k, n−1)]`)；运行时用
    #       chunk 里真实存在的 ê₀ —— 真机上那些帧确实在计划里，只是机器人先停了。
    # 两处**都是运行时为准**。完整 chunk 上必须逐帧一致。
    d5_tail = 0.0
    for a, b, e in zip(run_da, ref_da, E):
        cut = 0
        for s_, e_ in e["spans"]:
            if int(e["cap"][s_]) + 1 <= e["n"]:
                cut = e_
        d5 = max(d5, float(np.abs(a[:cut] - b[:cut]).max()))
        if cut < e["n"]:
            d5_tail = max(d5_tail, float(np.abs(a[cut:] - b[cut:]).max()))
    ok["4_stageA_eq_stageB"] = d4 <= 1e-12
    ok["5_delta_parity"] = d5 <= TOL_DA
    print(f"  检 4 stage A == stage B([0,W_A+M]) 最大差 {d4:.3e}   "
          f"{'✓' if ok['4_stageA_eq_stageB'] else '✗'}")
    print(f"  检 5 逐帧 δa == 离线参考        最大差 {d5:.3e} rad "
          f"({d5*1e3:.2e} mrad)   {'✓' if ok['5_delta_parity'] else '✗'}")
    print(f"        （比的是**完整 chunk**；末尾被截断的那个 chunk 另计 "
          f"{d5_tail:.2e} rad —— 离线重放的产物，**运行时为准**，见脚本注释）")

    r_run, rms_run, sa_run = DG.score(E, H, [(d, 0.0) for d in run_da])
    print(f"\n  ── 交付量 ──")
    print(f"  离线参考（P={rt.prefix} 逐 chunk，无空档）  {r_ref*100:.2f}%  "
          f"δa RMS {rms_ref:.2f} mrad")
    print(f"  **运行时重放**                        **{r_run*100:.2f}%**  "
          f"δa RMS {rms_run:.2f} mrad")

    sa = [s["stage_a_ms_p95"] for s in sums]; sb = [s["stage_b_ms_p95"] for s in sums]
    sv = [s["solve_ms_p95"] for s in sums]
    print(f"\n  ── 耗时（p95，模型已缓存）──")
    print(f"  stage A（**同步**，chunk 边界那一帧）  {np.mean(sa):6.1f} ms  "
          f"最大 {max(s['stage_a_ms_max'] for s in sums):.1f} ms")
    print(f"  stage B（后台，余裕 {rt.w_a} 帧 = {rt.w_a*50} ms）    {np.mean(sb):6.1f} ms  "
          f"最大 {max(s['stage_b_ms_max'] for s in sums):.1f} ms")
    print(f"  逐帧求解                              {np.mean(sv):6.3f} ms")
    print(f"  无计划帧占比 {np.mean([s['no_plan_pct'] for s in sums]):.2f}%  ·  "
          f"接触门控 {np.mean([s['gated_pct'] for s in sums]):.2f}%  ·  "
          f"触界 {np.mean([s['sat_pct'] for s in sums]):.2f}%")

    allok = all(ok.values())
    print(f"\n  ── {'**五项全过 ⇒ 可以接进部署 wrapper**' if allok else '**未全过 ⇒ 不能上机**'} ──")
    for k, v in ok.items():
        print(f"    {'✓' if v else '✗'} {k}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"checks": ok, "all_pass": allok,
         "max_diffs": {"model_cache": d1, "omega_path": d2, "partial_render": d3,
                       "stageA_vs_B": d4, "delta_rad": d5,
                       "delta_rad_truncated_last_chunk": d5_tail},
         "delivery": {"reference": r_ref, "runtime": r_run,
                      "da_rms_mrad_reference": rms_ref, "da_rms_mrad_runtime": rms_run},
         "config": {"prefix": rt.prefix, "n_a": rt.n_a, "w_a": rt.w_a},
         "timing_ms": {"stage_a_p95_mean": float(np.mean(sa)),
                       "stage_a_max": float(max(s["stage_a_ms_max"] for s in sums)),
                       "stage_b_p95_mean": float(np.mean(sb)),
                       "solve_p95_mean": float(np.mean(sv)),
                       "omega_nocache_ms": ms_nocache, "omega_cache_ms": ms_cache},
         "per_episode_summary": sums,
         "caliber": ("ab4 P0 5 集(e0=gap); 参考 = verify_g2wall_deploy 的 "
                     "**P=20 逐 chunk + 无空档**; **离线数字, 被控对象是 H 模型, 非真机验证**"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0 if allok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
