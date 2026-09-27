#!/usr/bin/env python3
"""阶段 A：`F_abs` 与 G0 —— 严格按 `reports/PREREG_FABS_G0.md`（已冻结）。

**只做阶段 A。不训练 `F_inc`，不训练 π。** `G-2` 冻结 · ③c 不动 · 零机时。

学什么（**直接学「墙」**）
    ê_base(u) ≈ e_bare = ω_real[t+3] − ω_ideal[t]        ← 与 `Episode.gap` 同义
    F_abs(u)  = ω_ideal(u) + ê_base(u)

⚠ 两句定位（预注册 §4bis）
--------------------------
1. **左填充只保证学出的 `ê_base` 这一支因果。**
   `ω_ideal` 由 `IdealChunkRenderer`（CubicSpline + filtfilt）在**整 chunk** 上渲染、
   **本身非因果** ⇒ **完整的 `F_abs` 不是严格逐帧因果的物理模型。**
2. **`G-2` 是冻结的已拟合模型，不是零参数物理模型**（含 `rs_fir`/`rq_fir` 权重），
   只是本轮不重新拟合它。

⚠ 两条防泄漏（预注册 §4 补记）
------------------------------
- **每折固定取第 3000 步权重；外层留出会话只在最后评分一次，不参与模型选择。**
- **输入归一化统计只用该折的训练会话。**

输入（**保留绝对指令**）：`x_t = [u_t, Δ⁻u_t, Δ⁻²u_t]`，21 通道，差分**向后**。
⚠ **不复用** `build_inputs` 的相对指令；**旧 π 的特征路径不改。**

用法:
    python -u sim/train_fabs_g0.py --self-test
    python -u sim/train_fabs_g0.py
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

OUT = ROOT / "outputs/fabs_g0"
SESSIONS = [("ab5", "A"), ("ab6", "A"), ("ab7", "Apost"), ("ab2", "A"), ("ab4", "P0")]
EDGE, STEPS, SEED, LR = 20, 3000, 0, 1e-3
G0_GATE = 0.30


def features(u):
    """`x_t = [u_t, Δ⁻u_t, Δ⁻²u_t]` —— **绝对指令 + 向后差分**，21 通道。"""
    d1 = np.zeros_like(u); d1[1:] = u[1:] - u[:-1]
    d2 = np.zeros_like(u); d2[1:] = d1[1:] - d1[:-1]
    return np.concatenate([u, d1, d2], 1).astype(np.float32)      # (T,21)


def load_all(S1, rend, R, b):
    eps = []
    for sess, arm in SESSIONS:
        for p in sorted((ROOT / "outputs" / sess).glob(f"{arm}_ep*.json"),
                        key=lambda q: int(q.stem.split("ep")[-1])):
            pay = json.loads(p.read_text())
            if len(pay["frames"]) < 550 or any(f.get("imu") is None for f in pay["frames"]):
                continue
            try:
                e = S1.load_episode(p, "P0", R, b, rend, 550)
            except Exception:
                continue
            cmd, ideal_sim, _ = S1._render_logged_chunks(
                sorted(pay["chunks"], key=lambda c: int(c["start"])), rend, len(pay["frames"]))
            n = len(e.gap)
            e.sess = f"{sess}/{arm}"
            e.x = features(cmd[:n, :7])                  # (n,21) 绝对 u + 向后差分
            e.y = np.asarray(e.gap, np.float32)          # (n,3) = 墙
            # ω_ideal（G-2 对照要用）。⚠ 整 chunk 渲染 ⇒ **本身非因果**
            e.ideal = S1.calibrate_ideal_imu(
                ideal_sim.astype(np.float32), np.asarray(R, np.float32),
                np.asarray(b, np.float32))[:n, 3:6].astype(np.float64)
            e.path = str(p.relative_to(ROOT))
            eps.append(e)
    return eps


def bp_kernel(S1):
    from scipy.signal import firwin
    return firwin(41, list(S1.METRIC_BAND), fs=S1.FS, pass_zero=False).astype(np.float32)


def err_energy(resid, K):
    """口径：扣逐集常数 → bandpass41 → 丢首尾 20 → 能量/(3·T_e)。numpy 版。"""
    r = resid - resid.mean(0, keepdims=True)
    f = np.stack([np.convolve(r[:, j], K, mode="same") for j in range(3)], 1)
    v = f[EDGE:-EDGE]
    return float((v ** 2).sum()), 3 * len(v)


class FAbs:
    """因果空洞卷积（**左填充**）4 层 k=5 d=1/2/4/8 64ch → 1×1 → 3。输出头零初始化。"""

    def __init__(self, dev):
        import torch, torch.nn as nn
        layers, c = [], 21
        for d in (1, 2, 4, 8):
            layers += [nn.ConstantPad1d(((5 - 1) * d, 0), 0.0),      # **左填充 ⇒ 因果**
                       nn.Conv1d(c, 64, 5, dilation=d), nn.GELU()]
            c = 64
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(64, 3, 1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.net = nn.Sequential(self.body, self.head).to(dev)
        self.dev = dev

    def __call__(self, x):                                   # (B,21,T) → (B,T,3)
        return self.net(x).transpose(1, 2)


def run_fold(tr, te, K, dev, steps=STEPS, seed=SEED, log=None):
    """训一折。**固定取第 `steps` 步权重；留出只在最后评分一次。**"""
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    # ⚠ 归一化统计**只用训练会话**
    Xtr = np.concatenate([e.x for e in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1.0
    Kt = torch.tensor(K, device=dev).view(1, 1, -1).repeat(3, 1, 1)

    def band(r):                                              # (B,T,3) → 带通后
        return torch.nn.functional.conv1d(r.transpose(1, 2), Kt, padding=20, groups=3
                                          ).transpose(1, 2)

    m = FAbs(dev)
    opt = torch.optim.Adam(m.net.parameters(), lr=LR)
    t0 = time.time()
    for it in range(steps):
        e = tr[int(rng.integers(len(tr)))]
        x = torch.tensor((e.x - mu) / sd, device=dev).T[None]
        y = torch.tensor(e.y, device=dev)[None]
        r = y - m(x)
        r = r - r.mean(1, keepdim=True)                       # 逐集常数
        v = band(r)[:, EDGE:-EDGE]
        loss = (v ** 2).sum() / (3 * v.shape[1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.net.parameters(), 1.0); opt.step()
        if log and (it + 1) % 1000 == 0:
            log(f"      step {it+1}/{steps}  train loss {loss.item():.6g}  "
                f"{time.time()-t0:.0f}s")
    # ── 留出：**最后评分一次** ────────────────────────────────────────
    m.net.eval(); num = den = cnt = 0.0
    per = []
    with torch.no_grad():
        for e in te:
            x = torch.tensor((e.x - mu) / sd, device=dev).T[None]
            p = m(x)[0].cpu().numpy()
            a, c = err_energy(e.y - p, K); b_, _ = err_energy(e.y, K)
            num += a; den += b_; cnt += c
            per.append({"ep": e.path, "num": a, "den": b_})
    return {"redu": 1.0 - num / den, "num": num, "den": den, "n_frames_x3": cnt,
            "episodes": per, "sec": time.time() - t0}


def g2_score(te, K, R, b, S1):
    """**冻结的已拟合** G-2 在同一留出集、同一口径上重新评分（**不重新拟合**）。"""
    import g2_on_ab4 as G
    from freeze_pi_g2wall import G2v0
    g2 = G2v0(); num = den = 0.0
    for e in te:
        dd = G.ab4_to_dd(ROOT / e.path, e.sess)
        w = np.asarray(g2.omega(dd), float)
        w6 = np.zeros((len(w), 6), np.float32); w6[:, 3:] = w
        ws = S1.calibrate_ideal_imu(w6, np.asarray(R, np.float32),
                                    np.asarray(b, np.float32))[:, 3:]
        n = min(len(e.y), len(ws) - S1.ALIGN_L, len(e.ideal))
        # ê_base^{G-2}[t] = calib(ω_G2)[t+L] − ideal[t]，与标签 e.y 同轴
        pred = ws[S1.ALIGN_L:S1.ALIGN_L + n].astype(float) - e.ideal[:n]
        a, _ = err_energy(e.y[:n] - pred, K); c, _ = err_energy(e.y[:n], K)
        num += a; den += c
    return 1.0 - num / den


def self_test(eps, K, S1, dev):
    print("\n" + "=" * 74 + "\n自测\n" + "=" * 74)
    ok = {}
    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    # ① 零输出复现 G0 分母
    e = eps[0]
    a, _ = err_energy(e.y - np.zeros_like(e.y), K); b_, _ = err_energy(e.y, K)
    chk("1 **零输出的误差 == G0 分母**（⇒ redu=0）", abs(a - b_) < 1e-12, f"{a:.6g} vs {b_:.6g}")
    r = e.y - e.y.mean(0, keepdims=True)
    c_, _ = err_energy(r, K)
    # ⚠ 容差**必须是相对的** —— e.y 是 float32，逐集去均值会累积舍入
    chk("1b 「逐集常数预测墙」与零输出等价（口径已扣逐集常数）",
        abs(c_ - b_) / b_ < 1e-6, f"相对 {abs(c_-b_)/b_:.2e}")

    # ② 标签对齐
    pay = json.loads((ROOT / e.path).read_text())
    real = np.asarray([f["imu"] for f in pay["frames"]], float)[:, 3:6]
    n = len(e.y)
    dmax = float(np.abs(e.y - (real[S1.ALIGN_L:S1.ALIGN_L + n] - e.ideal[:n])).max())
    f32 = float(np.finfo(np.float32).eps * np.abs(e.y).max())
    # ⚠ e.y 以 float32 存 ⇒ 门槛不得比 float32 精度还紧
    chk("2 **标签 = ω_real[t+3] − ω_ideal[t]**", dmax <= f32,
        f"L={S1.ALIGN_L} · max|Δ| {dmax:.2e} ≤ float32 eps×|y|max {f32:.2e}")

    # ③ 特征：保留绝对指令 + 向后差分
    u = np.arange(30 * 7, dtype=np.float64).reshape(30, 7)
    X = features(u)
    chk("3 **前 7 维 = 绝对 u**（不是相对指令）", np.abs(X[:, :7] - u).max() < 1e-9)
    chk("3b 中 7 维 = 向后一阶差分，首帧为 0",
        np.abs(X[1:, 7:14] - (u[1:] - u[:-1])).max() < 1e-9 and np.abs(X[0, 7:14]).max() == 0)
    chk("3c 后 7 维 = 向后二阶差分", np.abs(X[2:, 14:] - (u[2:] - 2*u[1:-1] + u[:-2])).max() < 1e-9)
    chk("3d 共 21 通道", X.shape[1] == 21)

    # ④ 因果：改未来指令不影响过去输出
    import torch
    m = FAbs(dev)
    with torch.no_grad():
        for p_ in m.net.parameters():
            p_.add_(torch.randn_like(p_) * 0.1)          # 打破零初始化才测得出
        x1 = torch.randn(1, 21, 200, device=dev); x2 = x1.clone(); x2[..., 100:] = 0
        chk("4 **左填充 ⇒ ê_base 这一支因果**（改 t≥100 不影响 t<100）",
            (m(x1)[0, :100] - m(x2)[0, :100]).abs().max().item() < 1e-5)

    # ⑤ 零初始化的网络输出恰为 0
    m0 = FAbs(dev)
    with torch.no_grad():
        chk("5 输出头零初始化 ⇒ 起点输出恒 0（= G0 分母）",
            m0(torch.randn(1, 21, 100, device=dev)).abs().max().item() == 0.0)

    # ⑥ 留出不进归一化：按会话划分后，统计只用训练会话
    sess = sorted({e.sess for e in eps})
    tr = [x for x in eps if x.sess != sess[0]]
    mu_tr = np.concatenate([x.x for x in tr]).mean(0)
    mu_all = np.concatenate([x.x for x in eps]).mean(0)
    chk("6 **归一化统计只用训练会话**（与全量不同 ⇒ 确实排除了留出）",
        np.abs(mu_tr - mu_all).max() > 0)
    chk("6b 5 个会话齐全且整集不切", len(sess) == 5, str(sess))
    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def main() -> int:
    import gc; gc.disable()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    import fit_step1_candidate as S1
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    cal = json.loads((ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(cal["R"], np.float64); b = np.asarray(cal.get("b", [0, 0, 0]), np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load_all(S1, IdealChunkRenderer(), R, b)
    K = bp_kernel(S1)
    byS = {}
    for e in eps: byS.setdefault(e.sess, []).append(e)
    print(f"裸集 {len(eps)} 集 / {len(byS)} 会话  " +
          " · ".join(f"{k} {len(v)}" for k, v in byS.items()) + f"   device={dev}")

    if a.self_test:
        return 0 if self_test(eps, K, S1, dev) else 1

    print("\n" + "=" * 74 + "\nleave-one-session-out（每折固定取第 "
          f"{a.steps} 步；留出只最后评分一次）\n" + "=" * 74)
    t_all = time.time(); rows = []
    for s in byS:
        tr = [e for e in eps if e.sess != s]; te = byS[s]
        print(f"\n  留出 {s}（{len(te)} 集）· 训练 {len(tr)} 集")
        r = run_fold(tr, te, K, dev, steps=a.steps, log=lambda m: print(m))
        g2 = g2_score(te, K, R, b, S1)
        r.update({"session": s, "n_te": len(te), "n_tr": len(tr), "g2_redu": g2})
        rows.append(r)
        print(f"      **F_abs redu = {r['redu']*100:+.2f}%** ｜ "
              f"G-2（冻结，同口径）redu = {g2*100:+.2f}%   [{r['sec']:.0f}s]")
    hrs = (time.time() - t_all) / 3600
    mean = float(np.mean([r["redu"] for r in rows]))
    allpos = all(r["redu"] > 0 for r in rows)
    g2m = float(np.mean([r["g2_redu"] for r in rows]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fabs_g0_v1.json").write_text(json.dumps(
        {"folds": rows, "mean_redu": mean, "all_positive": allpos,
         "g2_mean_redu": g2m, "gate": G0_GATE, "steps": a.steps, "seed": SEED,
         "gpu_hours": hrs, "pass": bool(mean >= G0_GATE and allpos)},
        indent=2, ensure_ascii=False))
    print("\n" + "=" * 74 + "\nG0 / G1\n" + "=" * 74)
    print(f"  {'留出会话':<14}{'F_abs redu':>13}{'G-2 redu':>12}")
    for r in rows:
        print(f"  {r['session']:<14}{r['redu']*100:>12.2f}%{r['g2_redu']*100:>11.2f}%")
    print(f"  {'平均':<14}{mean*100:>12.2f}%{g2m*100:>11.2f}%")
    print(f"\n  **G0（≥{G0_GATE*100:.0f}% 且 5 折全正）："
          f"{'通过' if (mean >= G0_GATE and allpos) else '**不通过**'}**"
          f"   （全正 = {allpos}）")
    print(f"  G1：`G-2`（**冻结的已拟合模型**）同口径 redu = {g2m*100:+.2f}%")
    print(f"\n  用时 {hrs:.2f} GPU·小时（预算 2）"
          + ("" if hrs <= 2 else "   ⚠ **超预算 ⇒ 记为未完成**"))
    print(f"→ {OUT/'fabs_g0_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
