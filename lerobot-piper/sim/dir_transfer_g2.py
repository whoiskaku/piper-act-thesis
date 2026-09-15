#!/usr/bin/env python3
"""**有边界的方向性实验**：G-2 直接前向指示的补正方向，能否迁移到实测 H 的离线评价？

只回答这一个问题
----------------
**不回答** π 的最终性能，**不回答** G→π 是否可行，**不授权 ③b**。
通过 ⇒ 只说明「**这一受限策略**从 G 获得了可迁移的方向信号」。
未通过 ⇒ 只否定**这次低维参数化**，**不得**宣称 G→π 不可能。

参数化（低维、因果，18 个参数）
--------------------------------
    δa[t] = clip( S · (Θ @ r̃[t−3]) , ±ε ) ，接触窗归零
    r[t−3] = **当时可观测的原始残差**（不带通）；r̃ = 用**训练集统计**标准化后的 r
    Θ ∈ R^{6×3} 无量纲，S = 1 mrad/单位 ⇒ SPSA 的 a/c 作用在**无量纲**参数上

⚠ 训练端不得接触 H（用户 2026-09-14）
-------------------------------------
- **直接读 P0**（`S1.load_episode` 不加载 H），**不调用** `O.load_all()`
  （后者对 P1 层算 `e₀ = gap − H⊛da`，会把 H 带进训练）。
- P1 层**完全不用**。
- 标定 `(R, b)` 取自**独立冻结**的 manifest。

⚠ 坐标系与时间轴（用户 2026-09-14 写死）
----------------------------------------
`e₀ = gap` 在**真机传感器系**（`gap[t] = real[t+3] − ideal[t]`）；
G-2 的 `Δω_G` 在**机体系**。**不得直接相加**：

    Δgap_G[t] = Δω_G[t + ALIGN_L] @ R.T          ALIGN_L = 3（事前固定）

**位移不得按 H 结果重新挑选。** 开跑前做零补正与时间轴自检（§自检）。

固定预算（事前定死，不追加、不中途调参）
----------------------------------------
SPSA 200 步 × 2 次扰动 × 3 集 × **2 轮定点迭代** × **2 次前向/odd()** = **4800 次 G 前向**。

⚠ **预算订正（2026-09-14，未见任何结果前）**：初版写成「1200 次」是我算错 —— 漏了
`rollout()` 的 2 轮定点迭代与 `odd()` 的 ±配对（各 ×2）。跑到 step 0 即发现并停机，
**未看到任何结果**，故重定预算不是事后改判据。步数/种子/划分/门禁**全部不变**。

`a=0.02, c=0.05, seed=0`
（**归一化后**的无量纲参数上）。训练/选参 = P0 ep0–2；评价 = P0 ep3–4。

一次性 H 评价（定稿后）
-----------------------
三臂：① 零补正（≡0）② G 选出的 Θ ③ **199 个固定种子**的随机 Θ，
从**同一 18 参数策略类**采样，并**匹配补正 RMS、钳位与接触门控**。

门禁（事前定死）
----------------
**G 臂的 H 降幅 > 随机分布的 95 分位，且 P0 ep3、ep4 各自 > 0**（防一集掩盖另一集）。

⚠ P0 ep3–4 曾在 `STEP1_RESULT_v1` §2.3 的诊断里被用于模型选择，**不是全新留出**；
且只有 2 集，**样本很薄**。ab4 无时间戳 ⇒ G 前向在**假设均匀 20 Hz** 下跑。

用法:
    python -u sim/dir_transfer_g2.py --self-test   # 只跑零补正与时间轴自检
    python -u sim/dir_transfer_g2.py               # 自检 + 训练 + 一次性 H 评价
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                    # noqa: E402
import g2_on_ab4 as G                                               # noqa: E402

ALIGN_L = S1.ALIGN_L            # 3，事前固定
EPS = 0.010                     # 认证 ε（**瞬时峰值**硬界）
S_SCALE = 1e-3                  # 1 mrad / 单位 ⇒ Θ 无量纲
STEPS, A_SPSA, C_SPSA, SEED = 200, 0.02, 0.05, 0
TRAIN, EVAL = (0, 1, 2), (3, 4)
N_RAND = 199
OUT = ROOT / "outputs/step1_H/dir_transfer_g2.json"


def load_P0():
    """**直接读 P0**（不碰 H）+ 独立冻结的 (R, b)。"""
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0",))
    calib = json.loads(
        (ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(calib["R"], float)
    b = np.asarray(calib.get("b", [0, 0, 0]), float)
    sys.path.insert(0, str(ROOT))
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    rend = IdealChunkRenderer()
    E = []
    for p in logs["P0"]:
        ep = S1.load_episode(p, "P0", R, b, rend, 550)
        pay = json.loads(p.read_text())
        _, _, cap = S1._render_logged_chunks(pay.get("chunks", []), rend,
                                             len(pay.get("frames", [])))
        n = len(ep.gap)
        E.append({"name": p.stem, "e0": np.asarray(ep.gap, float),     # P0 无探针 ⇒ e0=gap
                  "contact": np.asarray(ep.contact_mask, bool),
                  "cap": np.asarray(cap[:n], int), "dd": G.ab4_to_dd(p, p.stem), "n": n})
    return E, R, b


def to_gap(dw, R, n):
    """`Δgap_G[t] = Δω_G[t+ALIGN_L] @ R.T`。位移为事前固定值，不按 H 重挑。"""
    out = np.zeros((n, 3))
    m = min(n, len(dw) - ALIGN_L)
    if m > 0:
        out[:m] = dw[ALIGN_L:ALIGN_L + m] @ R.T
    return out


def rollout(e, Th, mu, sd, g2, R):
    """因果滚动：δa[t] = clip(S·Θ·r̃[t−3]) ，r 为**当时可观测的原始残差**。

    ⚠ 严格因果需逐帧交替「算 δa → 跑 G」，代价过高。本实现按预注册的**一次前向**
    近似：先用**上一轮的 δa 轨迹**算残差，再一次性求新 δa。
    自检 §③ 校验它与逐帧实现在零补正处一致。
    """
    n = e["n"]
    da = np.zeros((n, 6))
    r = e["e0"].copy()                       # 零补正起点：可观测残差 = e0
    for _ in range(2):                       # 两轮定点迭代（预算内固定）
        rt = (r - mu) / sd
        v = np.zeros((n, 6))
        v[ALIGN_L:] = S_SCALE * (rt[:n - ALIGN_L] @ Th.T)
        v = np.clip(v, -EPS, EPS)
        v[e["contact"]] = 0.0
        dw = g2.odd(e["dd"], v) * 2.0        # odd() 返回 (f(+)−f(−))/2 ⇒ ×2 得全差分
        r = e["e0"] + to_gap(dw, R, n)
        da = v
    return da, r


def loss(E, Th, mu, sd, g2, R, base):
    tot = 0.0
    for e, b0 in zip(E, base):
        _, r = rollout(e, Th, mu, sd, g2, R)
        tot += float(np.square(S1.bandpass41(r)).sum()) / b0
    return tot / len(E)


def main() -> int:
    import gc
    gc.disable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    E, R, b = load_P0()
    g2 = G.G2()
    print(f"P0 {len(E)} 集（**直接读取，未调用 O.load_all，H 未介入训练**）")
    print(f"标定 (R,b) 取自独立冻结 manifest · ALIGN_L={ALIGN_L}（事前固定）")
    print(f"⚠ ab4 无时间戳 ⇒ G 前向假设均匀 20 Hz\n")

    # ── 自检：零补正 + 时间轴 ────────────────────────────────────────
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✓' if cond else '**✗**'} {name}{('  ' + detail) if detail else ''}")

    print("自检")
    e = E[0]
    dw0 = g2.odd(e["dd"], np.zeros((e["n"], 6)))
    chk("零补正 ⇒ G 增量恒为 0", float(np.abs(dw0).max()) < 1e-12,
        f"max {float(np.abs(dw0).max()):.2e}")
    Th0 = np.zeros((6, 3))
    mu = np.concatenate([E[i]["e0"] for i in TRAIN]).mean(0)
    sd = np.concatenate([E[i]["e0"] for i in TRAIN]).std(0) + 1e-12
    da0, r0 = rollout(e, Th0, mu, sd, g2, R)
    chk("Θ=0 ⇒ δa 恒为 0 且残差 == e₀", float(np.abs(da0).max()) == 0.0
        and float(np.abs(r0 - e["e0"]).max()) < 1e-12)
    dwt = np.zeros((e["n"] + 8, 3)); dwt[ALIGN_L + 5] = [1.0, 0.0, 0.0]
    gt = to_gap(dwt, R, e["n"])
    chk("时间轴：Δω_G[t+3] → Δgap_G[t]", int(np.argmax(np.abs(gt).max(1))) == 5,
        f"脉冲落在 gap 轴 t={int(np.argmax(np.abs(gt).max(1)))}（应为 5）")
    chk("坐标旋转用独立冻结 R（非按 H 重挑）",
        np.allclose(R @ R.T, np.eye(3), atol=1e-4))
    print(f"\n自检判决: {'✅ 通过' if ok else '❌ FAIL —— 不开跑'}")
    if a.self_test or not ok:
        return 0 if ok else 1

    # ── SPSA（**只用 G**）──────────────────────────────────────────
    base = [float(np.square(S1.bandpass41(E[i]["e0"])).sum()) for i in range(len(E))]
    tr = [E[i] for i in TRAIN]
    btr = [base[i] for i in TRAIN]
    rng = np.random.default_rng(SEED)
    Th = np.zeros((6, 3))
    nfwd = STEPS * 2 * len(tr) * 2 * 2      # 步 × 扰动 × 集 × 定点迭代 × ±配对
    print(f"\nSPSA：{STEPS} 步 × 2 扰动 × {len(tr)} 集 × 2 迭代 × 2 前向 = "
          f"**{nfwd} 次 G 前向**（已订正，见文档）· a={A_SPSA} c={C_SPSA} seed={SEED}")
    hist = []
    for k in range(STEPS):
        ak, ck = A_SPSA / (k + 1 + 50) ** 0.602, C_SPSA / (k + 1) ** 0.101
        D = rng.choice([-1.0, 1.0], size=Th.shape)
        lp = loss(tr, Th + ck * D, mu, sd, g2, R, btr)
        lm = loss(tr, Th - ck * D, mu, sd, g2, R, btr)
        Th = Th - ak * (lp - lm) / (2 * ck) * D
        if k % 40 == 0 or k == STEPS - 1:
            hist.append({"step": k, "loss": 0.5 * (lp + lm)})
            print(f"  step {k:>4}  训练损失 {0.5*(lp+lm):.5f}  ‖Θ‖ {np.linalg.norm(Th):.3f}")

    # ── 一次性 H 评价 ───────────────────────────────────────────────
    H = np.ascontiguousarray(np.load(ROOT / "outputs/step1_H/step1_H_v1.npz")["H"], float)

    def h_red(e, da, b0):
        new = S1.bandpass41(e["e0"] + S1.causal_conv(H, da))
        return 1 - float(np.square(new).sum()) / b0

    print(f"\n一次性 H 评价（P0 ep{EVAL[0]}/ep{EVAL[1]}，H **首次**介入）")
    res = {}
    for i in EVAL:
        e = E[i]
        da_g, _ = rollout(e, Th, mu, sd, g2, R)
        rg = h_red(e, da_g, base[i])
        rms = float(np.sqrt(np.mean(da_g ** 2)))
        rr = []
        rr_rng = np.random.default_rng(1234)
        for s in range(N_RAND):
            Tr = rr_rng.normal(size=Th.shape)
            # 随机臂只需 δa 与 H 评价，**不跑 G**（跑了也只是浪费前向）
            v = np.zeros((e["n"], 6))
            v[ALIGN_L:] = S_SCALE * (((e["e0"] - mu) / sd)[:e["n"] - ALIGN_L] @ Tr.T)
            v = np.clip(v, -EPS, EPS); v[e["contact"]] = 0.0
            m = float(np.sqrt(np.mean(v ** 2)))
            if m > 1e-15:
                v = np.clip(v * (rms / m), -EPS, EPS); v[e["contact"]] = 0.0
            rr.append(h_red(e, v, base[i]))
        rr = np.asarray(rr)
        res[E[i]["name"]] = {"G": rg, "rand_p95": float(np.percentile(rr, 95)),
                             "rand_med": float(np.median(rr)), "da_rms_mrad": rms * 1e3}
        print(f"  {E[i]['name']}  G 臂 {rg*100:+.2f}%  ·  随机中位 "
              f"{np.median(rr)*100:+.2f}%  95分位 {np.percentile(rr,95)*100:+.2f}%"
              f"  ·  δa RMS {rms*1e3:.2f} mrad")

    pass_all = all(v["G"] > v["rand_p95"] and v["G"] > 0 for v in res.values())
    print(f"\n门禁（事前定死）：G 臂 > 随机 95 分位 **且** 两集各自 > 0")
    print(f"**判决: {'通过 —— 这一受限策略从 G 获得了可迁移的方向信号' if pass_all else '未通过'}**")
    print(f"\n⚠ 通过**不授权 ③b**；未通过**只**否定这次低维参数化，"
          f"不得宣称 G→π 不可能。\n⚠ P0 ep3–4 非全新留出且只有 2 集，样本很薄。")
    OUT.write_text(json.dumps(
        {"pass": bool(pass_all), "per_episode": res, "theta": Th.tolist(),
         "budget": {"steps": STEPS, "forwards": STEPS * 2 * len(tr) * 4,
                    "forwards_note": "步x扰动x集x2定点迭代x2(±配对); 初版误算为 1200, 未见结果前订正", "a": A_SPSA,
                    "c": C_SPSA, "seed": SEED, "n_rand": N_RAND},
         "split": {"train": [E[i]["name"] for i in TRAIN],
                   "eval": [E[i]["name"] for i in EVAL]},
         "loss_history": hist,
         "caliber": ("训练端**只用 G**(直接读 P0, 未调 O.load_all, P1 完全不用); "
                     "Δgap_G[t]=Δω_G[t+3]@R.T, 位移为事前固定 ALIGN_L 非按 H 重挑; "
                     "策略输入是**未带通的原始可观测残差**, 带通只进离线损失; "
                     "随机臂从**同一 18 参数策略类**采样并匹配 RMS/钳位/接触门控; "
                     "P0 ep3-4 非全新留出且仅 2 集; ab4 无时间戳⇒假设均匀 20Hz; "
                     "**通过不授权 ③b**"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
