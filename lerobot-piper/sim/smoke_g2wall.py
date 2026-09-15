#!/usr/bin/env python3
"""`BCORR_MODE=g2wall` 的**端到端冒烟**（零机时）—— 驱动**真实** wrapper，真 20 Hz 墙钟。

与已有检验的分工
----------------
| 脚本 | 覆盖 |
|---|---|
| `verify_g2wall_runtime.py` | 运行时**数值**：前向/缓存/两段一致/逐帧 δa 与离线一致 |
| `verify_async_render.py` | 旧 `_pending` 路径的**编排**（假渲染器） |
| **本脚本** | `BCorrectionDeployWrapper` 的 **g2wall 编排 + 实时性 + 安全**，真时钟 |

查八条
------
1. 启动期校验：manifest/H sha/G-2 W sha 对得上；**篡改 manifest 必须抛**
2. `step()` **每帧恰好一次**，chunk 内帧号 `k` 与真实位置同步（空档也要推进）
3. 空档期 δa **严格为零**，且空档帧数被计数
4. 非空档帧的 δa 与**同一空档掩码下**的离线参考逐点一致
5. `frame_ms` p99 ≤ 60 ms、**无丢帧**（帧间隔 ≤ 60 ms）
6. ε=10 mrad never exceeded；**夹爪通道原样透传**
7. 旧模式不受影响（`off` 仍能构造并跑）
8. 运行中的异常输入 → δa 置零并计数，**不抛**

⚠ 本机的 ACT 推理用 `--act-ms`（默认 20 ms）模拟 —— 真机上它与 stage A 抢同一个
50 ms 预算，直接决定空档帧数。真实值上机后从 `infer_ms` 读。

用法:
    python -u sim/smoke_g2wall.py [--act-ms 20] [--episodes 1]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MANIFEST = ROOT / "outputs/p2_pi/pi_g2wall_v1.json"
OUT = ROOT / "outputs/p2_pi/smoke_g2wall.json"
FRAME_DT = 0.050
FRAME_MS_GATE = 60.0


class _Cfg:
    n_action_steps = 100
    image_features = None


class _FakeACT:
    """回放已记录的 chunk，并模拟 ACT 推理耗时。"""

    def __init__(self, chunks, act_ms):
        self.chunks = [np.asarray(c["raw"], np.float32) for c in chunks]
        self.i = 0
        self.act_ms = act_ms / 1e3
        self.config = _Cfg()

    def normalize_inputs(self, batch):
        return batch

    def unnormalize_outputs(self, d):
        return d

    def model(self, batch):
        import torch
        time.sleep(self.act_ms)                 # 模拟推理占用帧预算
        c = self.chunks[min(self.i, len(self.chunks) - 1)]
        self.i += 1
        return (torch.from_numpy(c).unsqueeze(0),)

    def reset(self):
        self.i = 0

    def to(self, d):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])


def main() -> int:
    import gc
    gc.disable()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-ms", type=float, default=20.0)
    ap.add_argument("--episodes", type=int, default=1)
    a = ap.parse_args()

    import fit_step1_candidate as S1
    import delivery_g2_predictor as DG
    import verify_g2wall_deploy as V
    from freeze_pi_g2wall import G2v0
    from lerobot.common.policies.act.b_correction_deploy import (
        BCorrectionDeployWrapper as BCorr)
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer, EPS_B
    from lerobot.common.policies.act.g2wall_runtime import G2WallError

    ok = {}

    def chk(name, cond, detail=""):
        ok[name] = bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('   ' + detail) if detail else ''}")

    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float32)
    b = np.asarray(calib.get("b", [0, 0, 0]), np.float32)
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)
    rend = IdealChunkRenderer()

    # ── 1. 启动期校验 ────────────────────────────────────────────────
    paths = S1.discover_logs(ROOT / "outputs/ab4", ("P0",))["P0"][:a.episodes]
    pay0 = json.loads(paths[0].read_text())
    act = _FakeACT(sorted(pay0["chunks"], key=lambda c: int(c["start"])), a.act_ms)
    pol = BCorr(act, str(MANIFEST), "cpu", R, b, mode="g2wall",
                log_path=str(ROOT / "outputs/p2_pi/smoke_log.json"), verbose=True)
    chk("1a 启动期校验通过（H sha / G-2 W sha / 求解器常量）", pol.g2w is not None)
    bad = ROOT / "outputs/p2_pi/_bad_manifest.json"
    m = json.loads(MANIFEST.read_text())
    m["components"]["H"]["sha256"] = "0" * 64
    bad.write_text(json.dumps(m))
    try:
        BCorr(_FakeACT(pay0["chunks"], 0), str(bad), "cpu", R, b, mode="g2wall",
              verbose=False)
        raised = False
    except G2WallError:
        raised = True
    except Exception:
        raised = False
    bad.unlink(missing_ok=True)
    chk("1b 篡改 manifest 的 H sha **必须抛**", raised)

    # ── 跑一集（真 20 Hz 墙钟）────────────────────────────────────────
    print(f"\n  回放 {paths[0].stem}（ACT 推理模拟 {a.act_ms:.0f} ms/chunk · 真 20 Hz）")
    n_frames = len(pay0["frames"])
    das, stages, ks, grip_out, contacts = [], [], [], [], []
    t_next = time.perf_counter()
    for t in range(n_frames):
        t_next += FRAME_DT
        batch = {"observation.state": torch.zeros(1, 7),
                 "observation.imu_real": torch.zeros(1, 6)}
        out = pol.select_action(batch)
        fr = pol._last_g2w_frame
        das.append(np.asarray(fr.delta, float))
        stages.append(fr.stage)
        ks.append(fr.k)
        contacts.append(bool(fr.contact_gated))
        grip_out.append(float(out[0, 6]))
        sl = t_next - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
    das = np.asarray(das)
    summ = pol.summary()
    g = summ["g2wall"]

    # ── 2. 帧号同步 ──────────────────────────────────────────────────
    T = act.chunks[0].shape[0]
    exp_k = [t % T for t in range(n_frames)]
    chk("2 chunk 内帧号 k 与真实位置同步（空档也推进）",
        ks == exp_k, f"首个不符 @ {next((i for i,(x,y) in enumerate(zip(ks,exp_k)) if x!=y), None)}")
    chk("2b step() 每帧恰好一次", g["frames"] == n_frames, f"{g['frames']} == {n_frames}")

    # ── 3. 空档 ──────────────────────────────────────────────────────
    black = np.asarray([s == "none" for s in stages])
    chk("3a 空档期 δa 严格为零", not np.any(das[black]),
        f"空档 {int(black.sum())} 帧 ({g['blackout_pct']:.1f}%)")
    # ⚠ 3a/4 在**全盘失效**时会平凡通过（处处 δa=0）。下面两条堵住这个洞 ——
    #   第一版接线就是全程 δa=0 而 3a/4 照过，靠 6c 才抓出来。
    nz = float(np.mean(np.any(das != 0, axis=1)))
    chk("3c δa 确实在动（非全盘失效）", black.mean() < 0.2 and nz > 0.5,
        f"空档 {black.mean()*100:.1f}% · 非零 δa 占 {nz*100:.1f}% · "
        f"RMS {np.sqrt((das**2).mean())*1e3:.2f} mrad")
    per_chunk_black = [int(black[s:s + T].sum()) for s in range(0, n_frames, T)]
    chk("3b 空档只在 chunk 头部",
        all(np.all(black[s:s + nb]) and not np.any(black[s + nb:s + T])
            for s, nb in zip(range(0, n_frames, T), per_chunk_black)),
        f"逐 chunk 空档帧数 {per_chunk_black}")

    # ── 4. 与离线参考一致（同一空档掩码）──────────────────────────────
    ep = S1.load_episode(paths[0], "P0", np.asarray(R, np.float64),
                         np.asarray(b, np.float64), rend, 550)
    cmdlog, ideal_sim, cap = S1._render_logged_chunks(
        sorted(pay0["chunks"], key=lambda c: int(c["start"])), rend, n_frames)
    ideal = S1.calibrate_ideal_imu(ideal_sim.astype(np.float32), R, b)[:, 3:6]
    nn = min(len(ep.gap), len(ideal))
    e = {"n": nn, "e0": np.asarray(ep.gap, float)[:nn], "ideal": ideal[:nn].astype(float),
         "contact": np.asarray(ep.contact_mask, bool)[:nn],
         "cap": np.asarray(cap[:nn], int), "cmd": cmdlog}
    e["spans"] = V.chunk_spans(e["cap"])
    g2 = G2v0()
    ref_e0 = V.e0hat_from_omega(
        V.omega_perchunk(g2, e["cmd"], e["spans"], pol.g2w.prefix),
        e["ideal"], e["spans"], R, b, per_chunk_tail=True)
    ref_da = _roll_mask(e, H, ref_e0, black[:nn])
    # 只比完整 chunk（末 chunk 被集末截断，是离线重放的产物 —— 见 verify_g2wall_runtime）
    cut = max((en for s_, en in e["spans"] if int(e["cap"][s_]) + 1 <= nn), default=0)
    d4 = float(np.abs(das[:cut] - ref_da[:cut]).max()) if cut else float("inf")
    chk("4 非空档帧 δa == 同掩码离线参考", d4 <= 1e-9,
        f"最大差 {d4:.2e} rad（比的是完整 chunk，{cut} 帧）")

    # ── 5. 实时性 ────────────────────────────────────────────────────
    fm = np.asarray(pol.stats.get("frame_ms", []), float)
    p99 = float(np.percentile(fm, 99)) if len(fm) else 0.0
    drops = int(np.sum(fm > FRAME_MS_GATE))
    chk(f"5a frame_ms p99 ≤ {FRAME_MS_GATE:.0f} ms", p99 <= FRAME_MS_GATE,
        f"p99 {p99:.1f} ms · 中位 {np.median(fm):.1f} ms · 最大 {fm.max():.1f} ms")
    chk("5b 无丢帧", drops == 0, f"{drops} 帧超 {FRAME_MS_GATE:.0f} ms")
    # ⚠ 进程卸载时 stage_a/b_ms 记在**子进程**里，父进程看不到 ⇒ 用父进程侧的
    #   真实往返延迟(`render_ms`)。这才是决定空档帧数的量。
    rt_ms = np.asarray(pol.stats.get("render_ms", []), float)
    print(f"      stage A 往返(父进程实测) p95 {np.percentile(rt_ms, 95):.1f} ms / "
          f"max {rt_ms.max():.1f} ms  ·  逐帧求解 {g['solve_ms_p95']:.3f} ms  ·  "
          f"进程卸载 {'**开**' if g.get('process_offload') else '关(线程)'}"
          f" · 推迟提交 {g.get('defer_frames')} 帧")

    # ── 6. 安全 ──────────────────────────────────────────────────────
    chk("6a |δa| ≤ ε=10 mrad", float(np.abs(das).max()) <= EPS_B + 1e-12,
        f"max {np.abs(das).max()*1e3:.2f} mrad")
    grip_ref = np.concatenate([c[:, 6] for c in act.chunks])[:n_frames]
    chk("6b 夹爪通道原样透传",
        np.allclose(np.asarray(grip_out), grip_ref, atol=1e-6), "")
    # **逐帧精确**比对，而不是比总比例 —— 总比例会被空档帧（不计门控）稀释
    served = (~black[:nn]) & (np.arange(nn) < cut)
    cm = np.asarray(contacts[:nn], bool)
    mism = int(np.sum(cm[served] != e["contact"][served]))
    chk("6c 接触门控**逐帧**与离线一致（服务帧上）", mism == 0,
        f"不符 {mism}/{int(served.sum())} 帧 · 运行时 {g['gated_pct']:.1f}% "
        f"vs 离线 {100*e['contact'].mean():.1f}%（差值是空档帧不计门控）")

    # ── 7. 旧模式不受影响 ────────────────────────────────────────────
    off = BCorr(_FakeACT(sorted(pay0["chunks"], key=lambda c: int(c["start"])), 0),
                None, "cpu", R, b, mode="off", verbose=False)
    o = off.select_action({"observation.state": torch.zeros(1, 7),
                           "observation.imu_real": torch.zeros(1, 6)})
    chk("7 旧模式 off 仍可构造并输出裸指令",
        o.shape == (1, 7) and off.g2w is None and float(np.abs(
            o[0, :6].numpy() - act.chunks[0][0, :6]).max()) < 1e-6)

    # ── 8. 运行中异常不抛 ────────────────────────────────────────────
    pol.g2w._plan = None                       # 模拟计划缺失
    try:
        pol.select_action({"observation.state": torch.zeros(1, 7)})
        noraise = True
    except Exception as exc:
        noraise = False
        print(f"      抛了: {type(exc).__name__}: {exc}")
    chk("8 运行中缺计划 → δa 置零且**不抛**",
        noraise and pol.g2w.stats["no_plan"] > 0)

    allok = all(ok.values())
    print(f"\n  ── 判决 ──")
    print(f"  {'✅ PASS — g2wall 可以上机冒烟' if allok else '❌ FAIL — 不得上机'}")
    print(f"\n  实测空档 **{g['blackout_pct']:.1f}%**（逐 chunk {per_chunk_black}）"
          f" ⇒ 按离线空档曲线，交付量约 "
          f"**{_expect(int(np.median(per_chunk_black))):.1f}%**（离线无空档 15.50%）")
    print(f"  ⚠ 这是**离线/回放**数字，被控对象是 H（模型），**非真机验证**。")
    print(f"  ⚠ **换会话风险独立**：ab4 P0 裸 52.1% → ab2 A 裸 26.3%，"
          f"已证**不是标定**问题。")
    OUT.write_text(json.dumps(
        {"checks": ok, "all_pass": allok, "act_ms": a.act_ms,
         "frame_ms": {"p50": float(np.median(fm)), "p99": p99, "max": float(fm.max()),
                      "drops_over_gate": drops, "gate": FRAME_MS_GATE},
         "blackout": {"pct": g["blackout_pct"], "per_chunk": per_chunk_black},
         "stage_a_roundtrip_ms": {"p95": float(np.percentile(rt_ms, 95)),
                                  "max": float(rt_ms.max())},
         "process_offload": bool(g.get("process_offload")),
         "max_delta_diff_rad": d4, "summary": summ,
         "caliber": ("真 20 Hz 墙钟回放 ab4 P0; ACT 推理用 --act-ms 模拟(真机从 infer_ms 读); "
                     "δa 只在**完整 chunk** 上与离线比(末 chunk 被集末截断是离线重放的产物); "
                     "**离线数字, 被控对象是 H 模型, 非真机验证**"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0 if allok else 1


def _roll_mask(e, H, pred, black):
    """`V.roll_blackout` 的显式掩码版 —— 用**运行时实测**的空档掩码。"""
    import fit_step1_candidate as S1
    import delivery_g2_predictor as DG
    n, M1 = e["n"], S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = DG.KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    da = np.zeros((n, 6)); resp = np.zeros((n + M1, 3))
    for t in range(n):
        K = max(int(min(S1.M, e["cap"][t] - t)), 0)
        tgt = np.stack([pred[min(t + k, n - 1)] + resp[t + k] for k in range(K + 1)])
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        d = np.clip(d, -DG.EPS, DG.EPS)
        if e["contact"][t] or black[t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da


def _expect(nb: int) -> float:
    """离线实测的空档→交付曲线（`verify_g2wall_deploy`）。"""
    curve = {0: 15.50, 1: 11.88, 2: 9.24, 3: 8.04, 4: 7.63, 6: 7.42}
    ks = sorted(curve)
    return curve[min(ks, key=lambda k: abs(k - nb))]


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
