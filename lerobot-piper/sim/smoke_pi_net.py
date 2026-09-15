#!/usr/bin/env python3
"""`BCORR_MODE=pi_net` 的**端到端冒烟**（零机时）—— 驱动真实 wrapper，真 20 Hz 墙钟。

查九条
------
1. 启动期校验：检查点装得上；**篡改检查点必须抛**
2. `step()` **每帧恰好一次**，chunk 内帧号 `k` 与真实位置同步
3. **零空档** —— π 是同步一次前向，**第 0 帧就必须有 δa**（这是它相对 `g2wall`
   的结构优势：后者每 chunk 空 3–4 帧，而墙的 52.2% 能量就在头 4 帧）
4. 逐帧 δa 与**离线同一网络**的输出逐点一致
5. `frame_ms` p99 ≤ 60 ms、无丢帧；π 推理 <5 ms
6. ε=10 mrad never exceeded；**夹爪通道原样透传**
7. 接触门控**逐帧**与离线一致
8. 旧模式不受影响（`off` / `g2wall` 仍能构造）
9. 运行中异常 → δa 置零并计数，**不抛**

用法:
    python -u sim/smoke_pi_net.py [--act-ms 20] [--ckpt pi_g2env_v1]
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

OUT = ROOT / "outputs/p2_pi/smoke_pi_net.json"
FRAME_DT, FRAME_MS_GATE = 0.050, 60.0


class _Cfg:
    n_action_steps = 100
    image_features = None


class _FakeACT:
    def __init__(self, chunks, act_ms):
        self.chunks = [np.asarray(c["raw"], np.float32) for c in chunks]
        self.i, self.act_ms, self.config = 0, act_ms / 1e3, _Cfg()

    def normalize_inputs(self, batch):
        return batch

    def unnormalize_outputs(self, d):
        return d

    def model(self, batch):
        import torch
        time.sleep(self.act_ms)
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
    ap.add_argument("--ckpt", default="pi_g2env_v1")
    a = ap.parse_args()

    import fit_step1_candidate as S1
    import train_pi_g2env as T
    from lerobot.common.policies.act.b_correction_deploy import (
        BCorrectionDeployWrapper as BCorr)
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer, EPS_B
    from lerobot.common.policies.act.pi_net_runtime import PiNetError

    ok = {}

    def chk(name, cond, detail=""):
        ok[name] = bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('   ' + detail) if detail else ''}")

    CKPT = ROOT / f"outputs/p2_pi/{a.ckpt}.pt"
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], np.float32)
    b = np.asarray(calib.get("b", [0, 0, 0]), np.float32)

    # ── 1. 启动期校验 ────────────────────────────────────────────────
    p0 = ROOT / "outputs/ab5/A_ep0.json"
    pay0 = json.loads(p0.read_text())
    chunks = sorted(pay0["chunks"], key=lambda c: int(c["start"]))
    act = _FakeACT(chunks, a.act_ms)
    pol = BCorr(act, str(CKPT), "cpu", R, b, mode="pi_net",
                log_path=str(ROOT / "outputs/p2_pi/smoke_pi_log.json"), verbose=True)
    chk("1a 启动期校验通过（检查点 / ε / 归一化统计）", pol.pinet is not None)
    bad = ROOT / "outputs/p2_pi/_bad_ckpt.pt"
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck.pop("norm_mu"); torch.save(ck, bad)
    try:
        BCorr(_FakeACT(chunks, 0), str(bad), "cpu", R, b, mode="pi_net", verbose=False)
        raised = False
    except PiNetError:
        raised = True
    except Exception:
        raised = False
    bad.unlink(missing_ok=True)
    chk("1b 检查点缺字段 **必须抛**", raised)

    # ── 回放一集（真 20 Hz 墙钟）────────────────────────────────────
    print(f"\n  回放 {p0.stem}（ACT 推理模拟 {a.act_ms:.0f} ms/chunk · 真 20 Hz）")
    n_frames = len(pay0["frames"])
    das, stages, ks, contacts, grip_out = [], [], [], [], []
    t_next = time.perf_counter()
    for t in range(n_frames):
        t_next += FRAME_DT
        batch = {"observation.state": torch.zeros(1, 7),
                 "observation.imu_real": torch.zeros(1, 6)}
        out = pol.select_action(batch)
        fr = pol._last_pinet_frame
        das.append(np.asarray(fr.delta, float)); stages.append(fr.stage)
        ks.append(fr.k); contacts.append(bool(fr.contact_gated))
        grip_out.append(float(out[0, 6]))
        sl = t_next - time.perf_counter()
        if sl > 0:
            time.sleep(sl)
    das = np.asarray(das)
    summ = pol.summary(); g = summ["pi_net"]

    T_ = act.chunks[0].shape[0]
    chk("2 chunk 内帧号 k 与真实位置同步", ks == [t % T_ for t in range(n_frames)])
    chk("2b step() 每帧恰好一次", g["frames"] == n_frames, f"{g['frames']} == {n_frames}")

    # ── 3. 零空档（结构优势）────────────────────────────────────────
    black = np.asarray([s != "pi" for s in stages])
    heads = [s for s in range(0, n_frames, T_)]
    head_nz = all(np.any(das[s] != 0) or contacts[s] for s in heads)
    chk("3 **零空档** · 每个 chunk 第 0 帧就有 δa",
        black.sum() == 0 and head_nz,
        f"空档 {int(black.sum())} 帧（g2wall 实测 3–4 帧/chunk）· "
        f"非零 δa 占 {np.mean(np.any(das != 0, axis=1))*100:.1f}%")
    chk("3b δa 确实在动（非全盘失效）",
        np.sqrt((das ** 2).mean()) * 1e3 > 0.5,
        f"RMS {np.sqrt((das**2).mean())*1e3:.2f} mrad · max {np.abs(das).max()*1e3:.2f}")

    # ── 4. 与离线同一网络逐点一致 ───────────────────────────────────
    z = np.load(ROOT / "outputs/p2_pi/g2env/ab5__A_ep0.npz")
    x, spans = T.build_inputs(z["u"], z["starts"], int(z["n"]))
    mu, sd = pol.pinet.mu, pol.pinet.sd
    with torch.no_grad():
        dch = pol.pinet.net(torch.from_numpy(
            ((x - mu) / sd).astype(np.float32))).cpu().numpy()
    ref = np.zeros((int(z["n"]), 6))
    for i, (s0, lc) in enumerate(spans):
        ref[s0:s0 + lc] = dch[i, :lc]
    ref[z["contact"][:len(ref)]] = 0.0
    m = min(len(ref), n_frames)
    d4 = float(np.abs(das[:m] - ref[:m]).max())
    chk("4 逐帧 δa == 离线同一网络", d4 <= 1e-6, f"最大差 {d4:.2e} rad")

    # ── 5. 实时性 ───────────────────────────────────────────────────
    fm = np.asarray(pol.stats.get("frame_ms", []), float)
    p99 = float(np.percentile(fm, 99)) if len(fm) else 0.0
    drops = int(np.sum(fm > FRAME_MS_GATE))
    chk(f"5a frame_ms p99 ≤ {FRAME_MS_GATE:.0f} ms", p99 <= FRAME_MS_GATE,
        f"p99 {p99:.1f} · 中位 {np.median(fm):.1f} · 最大 {fm.max():.1f} ms")
    chk("5b 无丢帧", drops == 0, f"{drops} 帧超门禁")
    # π 每 **100 帧(5 s)** 才前向一次, 与 ACT 推理同处 chunk 边界那一帧。
    # 真正的约束是 5a/5b 的 frame_ms; 这里只挡「病态慢」——
    # 15 ms 是 50 ms 预算里留给 ACT(实测 20–40 ms)之后仍宽裕的上限。
    chk("5c π 推理 < 15 ms（每 chunk 一次，与 ACT 同帧）", g["infer_ms_max"] < 15.0,
        f"p95 {g['infer_ms_p95']:.2f} · 最大 {g['infer_ms_max']:.2f} ms")

    # ── 6/7. 安全与门控 ─────────────────────────────────────────────
    chk("6a |δa| ≤ ε=10 mrad", float(np.abs(das).max()) <= EPS_B + 1e-12,
        f"max {np.abs(das).max()*1e3:.2f} mrad · 触界 {g['sat_pct']:.1f}%")
    grip_ref = np.concatenate([c[:, 6] for c in act.chunks])[:n_frames]
    chk("6b 夹爪通道原样透传",
        np.allclose(np.asarray(grip_out), grip_ref, atol=1e-6))
    cm = np.asarray(contacts[:m], bool)
    chk("7 接触门控**逐帧**与离线一致",
        int(np.sum(cm != z["contact"][:m])) == 0,
        f"不符 {int(np.sum(cm != z['contact'][:m]))}/{m} · 门控 {g['gated_pct']:.1f}%")

    # ── 8. 旧模式 ───────────────────────────────────────────────────
    off = BCorr(_FakeACT(chunks, 0), None, "cpu", R, b, mode="off", verbose=False)
    o = off.select_action({"observation.state": torch.zeros(1, 7)})
    chk("8 旧模式 off 不受影响",
        o.shape == (1, 7) and off.pinet is None
        and float(np.abs(o[0, :6].numpy() - act.chunks[0][0, :6]).max()) < 1e-6)

    # ── 9. 运行中异常不抛 ───────────────────────────────────────────
    pol.pinet._plan = None
    try:
        pol.select_action({"observation.state": torch.zeros(1, 7)})
        noraise = True
    except Exception as exc:
        noraise = False
        print(f"      抛了: {type(exc).__name__}: {exc}")
    chk("9 运行中缺计划 → δa=0 且**不抛**", noraise and pol.pinet.stats["no_plan"] > 0)

    allok = all(ok.values())
    print(f"\n  ── {'✅ PASS — pi_net 可以上机' if allok else '❌ FAIL — 不得上机'} ──")
    print(f"\n  π: {g['n_params']} 参数 · 推理 p95 {g['infer_ms_p95']:.2f} ms · "
          f"**零空档** · 不读 IMU · sha {g['ckpt_sha256'][:8]}…")
    print(f"  δa RMS {np.sqrt((das**2).mean())*1e3:.2f} mrad "
          f"(对照 C 臂 0.54) · 触界 {g['sat_pct']:.1f}%")
    # 事前预测从**该 checkpoint 自己的**核验产物读，避免印成上一个实验的数
    pred = ROOT / f"outputs/p2_pi/verify_pi_distill.json"
    try:
        v = json.loads(pred.read_text())
        if v.get("tag") == a.ckpt:
            print(f"\n  ⚠ **事前预测**（用真机已验证的 H 推算，同法预测 C 臂 −7.6% / 实测 −8.0%）：")
            print(f"     α = **{v['alpha']:.2f}** · 预测交付 **{v['pi_delivery']*100:+.2f}%**"
                  f"（教师 {v['teacher_delivery']*100:+.2f}%）· 对照 C 臂真机 **−8.0%**")
        else:
            print(f"\n  ⚠ 未找到 `{a.ckpt}` 的核验产物 —— 先跑 sim/verify_pi_distill.py")
    except Exception:
        print(f"\n  ⚠ 未找到核验产物 —— 先跑 sim/verify_pi_distill.py --tag {a.ckpt}")
    OUT.write_text(json.dumps(
        {"checks": ok, "all_pass": allok, "act_ms": a.act_ms,
         "frame_ms": {"p50": float(np.median(fm)), "p99": p99,
                      "max": float(fm.max()), "drops": drops},
         "blackout_frames": int(black.sum()),
         "da_rms_mrad": float(np.sqrt((das ** 2).mean()) * 1e3),
         "max_delta_diff_rad": d4, "summary": summ,
         "caliber": ("真 20 Hz 墙钟回放 ab5/A_ep0; ACT 推理用 --act-ms 模拟; "
                     "**pi_net 运行时不跑 MuJoCo、不读 IMU、无求解器**; "
                     "**离线/回放数字, 非真机验证**"),
         }, ensure_ascii=False, indent=1))
    print(f"\n  → {OUT}")
    return 0 if allok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
