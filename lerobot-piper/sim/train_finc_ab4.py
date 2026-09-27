#!/usr/bin/env python3
"""`F_inc` 限定 `ab4` 工况 —— 严格按 `reports/PREREG_FINC_AB4.md`（**已冻结**）。

⚠ **范围变更**：本实验只问「在 `ab4` 这一工况内，冻结名义预测后，神经增量分支
能否比固定 `H` 更好地预测观测结果」。**不能视为全会话 `G0` 已通过**；
即使成功也**只**支持该工况的模型预测收益，**不**代表跨会话泛化或真机迁移。
**全会话 `G0` 仍不通过；`ab2` 保留。不上机、不训练 π。**

四项预测（同轴 `gap[t] = ω_real[t+3] − ω_ideal[t]`）
    (a)  ê_base
    (b)  ê_base + H ⊛ δa                 冻结 H，**原 H 不修改**
    (b′) ê_base + β·(H ⊛ δa)             β 拟合**扣除名义预测后的残差**，只用 P1
    (c)  ê_base + F_inc(u, δa)           本轮唯一被训练的东西

用法:
    python -u sim/train_finc_ab4.py --self-test
    python -u sim/train_finc_ab4.py
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_fabs_g0 as A          # features / FAbs / err_energy / bp_kernel / load_all / EDGE …
import train_fabs_balance as B     # run_fold（return_model=True 取名义分支权重）

OUT = ROOT / "outputs/finc_ab4"
AB4 = ROOT / "outputs/ab4"
H_PATH = ROOT / "outputs/step1_H/step1_H_v1.npz"
TRAIN_SEEDS = (0, 1, 2)            # **训练种子**（≠ 探针种子），事前固定
GATE_D = 0.05                      # 推进阈值 d̄ ≥ 5%（对 (b) 与 (b′) 都要求）
NOMINAL = dict(arm="episode", seed=0, hold="ab4/P0")


# ── 误差泛函 P 与 E ──────────────────────────────────────────────────────
def Pfilt(v, K):
    """`P(·)` = 逐集去均值 → bandpass41 → 丢首尾 `EDGE`。与 `A.err_energy` 同口径。"""
    r = np.asarray(v, np.float64)
    r = r - r.mean(0, keepdims=True)
    f = np.stack([np.convolve(r[:, j], K, mode="same") for j in range(3)], 1)
    return f[A.EDGE:-A.EDGE]


def Eerr(resid, K):
    """`E = ‖P(resid)‖² / (3 T_e)`，`T_e` = 有效帧数。"""
    p = Pfilt(resid, K)
    return float((p ** 2).sum()) / (3 * len(p)), len(p)


# ── 数据 ────────────────────────────────────────────────────────────────
def load_tier(S1, rend, R, b, tier):
    """载入 `outputs/ab4/{tier}_ep*.json`，并按 `A.load_all` 的方式覆写 21 维特征。"""
    eps = []
    for p in sorted(AB4.glob(f"{tier}_ep*.json"), key=lambda q: int(q.stem.split("ep")[-1])):
        pay = json.loads(p.read_text())
        e = S1.load_episode(p, tier, R, b, rend, 550)
        cmd, _, _ = S1._render_logged_chunks(
            sorted(pay["chunks"], key=lambda c: int(c["start"])), rend, len(pay["frames"]))
        n = len(e.gap)
        e.x = A.features(cmd[:n, :7])                 # (n,21) 绝对 u + 向后差分
        e.y = np.asarray(e.gap, np.float32)           # (n,3) 实测 gap
        e.dq = np.ascontiguousarray(e.da[:n], np.float64)   # (n,6) 实际施加的 δa
        e.path = str(p.relative_to(ROOT))
        eps.append(e)
    return eps


# ── ① 恢复并核验名义分支 ─────────────────────────────────────────────────
def restore_nominal(eps_bare, K, dev, log=print):
    """重跑「臂 E · seed 0 · 留出 ab4/P0」，用**完整精度逐集 `num`/`den`** 核验。"""
    byS = {}
    for e in eps_bare:
        byS.setdefault(e.sess, []).append(e)
    te = byS[NOMINAL["hold"]]
    tr = [e for e in eps_bare if e.sess != NOMINAL["hold"]]
    r, m, mu, sd = B.run_fold(tr, te, K, dev, NOMINAL["arm"], NOMINAL["seed"],
                              return_model=True)
    got = {d["ep"]: (d["num"], d["den"]) for d in r["episodes"]}
    checks = []
    for tag, fn in (("fabs_g0_v1", ROOT / "outputs/fabs_g0/fabs_g0_v1.json"),
                    ("fabs_balance_v1", ROOT / "outputs/fabs_balance/fabs_balance_v1.json")):
        j = json.loads(fn.read_text())
        folds = j["folds"] if "folds" in j else [
            x for x in j["runs"] if x["arm"] == NOMINAL["arm"] and x["seed"] == NOMINAL["seed"]]
        f = [x for x in folds if x["session"] == NOMINAL["hold"]]
        if len(f) != 1:
            checks.append((tag, False, f"归档里找到 {len(f)} 条匹配折，应为 1")); continue
        ref = {d["ep"]: (d["num"], d["den"]) for d in f[0]["episodes"]}
        same = set(ref) == set(got) and all(
            ref[k][0] == got[k][0] and ref[k][1] == got[k][1] for k in ref)
        worst = max((max(abs(ref[k][0] - got[k][0]), abs(ref[k][1] - got[k][1]))
                     for k in ref if k in got), default=float("inf"))
        checks.append((tag, same, f"{len(ref)} 集逐位比较 · 最大绝对差 {worst:.3e}"))
    for tag, okk, d in checks:
        log(f"    {'✓' if okk else '**✗**'} 完整精度逐集 num/den == {tag}   {d}")
    return r, m, mu, sd, all(c[1] for c in checks)


def ebase_predict(m, mu, sd, eps, dev):
    """冻结名义分支的逐集预测（**用它自己的归一化统计**，不是 P1 的）。"""
    import torch
    m.net.eval(); out = []
    with torch.no_grad():
        for e in eps:
            x = torch.tensor((e.x - mu) / sd, device=dev).T[None]
            out.append(m(x)[0].cpu().numpy().astype(np.float64))
    return out


# ── ④ β：拟合**扣除名义预测后的残差** ────────────────────────────────────
def fit_beta(eps, base, H, K, S1):
    """β = Σ w_e⟨r_e,h_e⟩ / Σ w_e‖h_e‖²,  w_e = 1/(3T_e),  r_e = P(gap−ê_base), h_e = P(H⊛δa)。

    **只拟合一个全局标量，仅用 P1；不扫描、不按轴分开。允许为负。**
    """
    num = den = 0.0
    for e, p0 in zip(eps, base):
        r = Pfilt(np.asarray(e.y, np.float64) - p0, K)
        h = Pfilt(S1.causal_conv(H, e.dq), K)
        w = 1.0 / (3 * len(h))
        num += w * float((r * h).sum())
        den += w * float((h * h).sum())
    return num / den


# ── ② F_inc ─────────────────────────────────────────────────────────────
class FInc:
    """与 `F_abs` **同一主干**（`A.FAbs`，`in_ch=27`）。差分式保证 `F_inc(u,0) ≡ 0`。"""

    def __init__(self, dev, seed):
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        self.m = A.FAbs(dev, in_ch=27)
        self.dev = dev

    def __call__(self, xu, xd, xd0):
        """`N([x(u), 𝒩(δa)]) − N([x(u), 𝒩(0)])` —— 零扰动分支喂**原始 0 归一化后**的值。"""
        import torch
        return self.m(torch.cat([xu, xd], 1)) - self.m(torch.cat([xu, xd0], 1))


def train_finc(P1, base1, K, dev, seed, steps=A.STEPS, log=None):
    """只用 P1 训练；目标是**实测总 gap** 的预测误差，**不用 H 造标签**。"""
    import torch
    net = FInc(dev, seed)
    rng = np.random.default_rng(seed)
    # 归一化统计**仅用 P1**
    mu_u, sd_u = (lambda X: (X.mean(0), np.where(X.std(0) < 1e-8, 1.0, X.std(0))))(
        np.concatenate([e.x for e in P1]))
    mu_d, sd_d = (lambda X: (X.mean(0), np.where(X.std(0) < 1e-8, 1.0, X.std(0))))(
        np.concatenate([e.dq for e in P1]))
    Kt = torch.tensor(K, device=dev).view(1, 1, -1).repeat(3, 1, 1)

    def band(r):
        return torch.nn.functional.conv1d(r.transpose(1, 2), Kt, padding=20, groups=3
                                          ).transpose(1, 2)

    def inputs(e):
        xu = torch.tensor((e.x - mu_u) / sd_u, device=dev, dtype=torch.float32).T[None]
        xd = torch.tensor((e.dq - mu_d) / sd_d, device=dev, dtype=torch.float32).T[None]
        # **先把原始 δa 置 0，再过同一归一化**
        xd0 = torch.tensor((np.zeros_like(e.dq) - mu_d) / sd_d,
                           device=dev, dtype=torch.float32).T[None]
        return xu, xd, xd0

    pre = [inputs(e) for e in P1]
    ys = [torch.tensor(np.asarray(e.y, np.float32), device=dev)[None] for e in P1]
    b0 = [torch.tensor(p.astype(np.float32), device=dev)[None] for p in base1]
    opt = torch.optim.Adam(net.m.net.parameters(), lr=A.LR)      # 无 weight decay
    hist, t0 = [], time.time()
    for it in range(steps):
        i = int(rng.integers(len(P1)))
        xu, xd, xd0 = pre[i]
        r = ys[i] - (b0[i] + net(xu, xd, xd0))
        r = r - r.mean(1, keepdim=True)
        v = band(r)[:, A.EDGE:-A.EDGE]
        loss = (v ** 2).sum() / (3 * v.shape[1])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.m.net.parameters(), 1.0); opt.step()
        if (it + 1) % 250 == 0:
            hist.append({"step": it + 1, "loss": float(loss.item())})
        if log and (it + 1) % 1000 == 0:
            log(f"      step {it+1}/{steps}  loss {loss.item():.6g}  {time.time()-t0:.0f}s")
    return net, (mu_u, sd_u, mu_d, sd_d), hist, time.time() - t0


def finc_predict(net, stats, eps, dev):
    import torch
    mu_u, sd_u, mu_d, sd_d = stats
    net.m.net.eval(); out = []
    with torch.no_grad():
        for e in eps:
            xu = torch.tensor((e.x - mu_u) / sd_u, device=dev, dtype=torch.float32).T[None]
            xd = torch.tensor((e.dq - mu_d) / sd_d, device=dev, dtype=torch.float32).T[None]
            xd0 = torch.tensor((np.zeros_like(e.dq) - mu_d) / sd_d,
                               device=dev, dtype=torch.float32).T[None]
            out.append(net(xu, xd, xd0)[0].cpu().numpy().astype(np.float64))
    return out


# ── ② 聚合（**先误差后平均，不是先平均预测**）──────────────────────────
def aggregate(eps, err_by_seed):
    """`err_by_seed[k][i]` = 训练种子 k 下第 i 集的 E。

    ① 同一**探针种子**的两集**误差**取平均 → ② 三个**训练种子**的评分取平均。
    """
    seeds = sorted({int(e.seed) for e in eps})
    out = {}
    for s in seeds:
        idx = [i for i, e in enumerate(eps) if int(e.seed) == s]
        per_k = [float(np.mean([err_by_seed[k][i] for i in idx])) for k in sorted(err_by_seed)]
        out[s] = float(np.mean(per_k))
    return out


def dbar(Ey, Ec):
    """`d_s = (E_y(s) − E_c(s)) / E_y(s)`，再对探针种子**等权**平均。"""
    ss = sorted(Ey)
    ds = {s: (Ey[s] - Ec[s]) / Ey[s] for s in ss}
    return ds, float(np.mean([ds[s] for s in ss]))


# ── 自测 ─────────────────────────────────────────────────────────────────
def self_test(P1, P2, P3, K, dev, S1, H):
    print("\n" + "=" * 78 + "\n自测\n" + "=" * 78)
    ok = {}

    def chk(k, c, d=""):
        ok[k] = bool(c); print(f"  {'✓' if c else '**✗**'} {k}{('  ' + d) if d else ''}")

    e = P1[0]
    # ① P 与 A.err_energy 同口径
    resid = np.asarray(e.y, np.float64) * 0.37 + 1.0
    s1, c1 = A.err_energy(resid.astype(np.float32), K)
    s2, t2 = Eerr(resid, K)
    chk("1 **`Eerr` 与归档 `A.err_energy` 同口径**",
        abs(s2 - s1 / c1) / abs(s1 / c1) < 1e-6 and c1 == 3 * t2,
        f"{s2:.8g} vs {s1/c1:.8g} · 3T={c1}")

    # ② 结构：F_inc(u,0) 逐位为 0，且参数非零时也成立
    import torch
    net = FInc(dev, 0)
    with torch.no_grad():
        for p_ in net.m.net.parameters():
            p_.add_(torch.randn_like(p_) * 0.1)          # 打破零初始化
        xu = torch.randn(1, 21, 300, device=dev)
        mu_d = np.random.randn(6) * 3; sd_d = np.abs(np.random.randn(6)) + 0.5
        z = torch.tensor(((np.zeros((300, 6)) - mu_d) / sd_d), device=dev,
                         dtype=torch.float32).T[None]
        chk("2 **`F_inc(u, 0) ≡ 0` 逐位成立**（参数已随机化，非零初始化的假象）",
            net(xu, z, z).abs().max().item() == 0.0)
        nz = torch.randn(1, 6, 300, device=dev)
        chk("2b 非零扰动时输出确实非零（上一条不是恒真）",
            net(xu, nz, z).abs().max().item() > 0)
    chk("2c 零扰动分支 = **原始 δa 置 0 后再归一化**（不是喂归一化后的 0）",
        abs(float(((np.zeros(6) - mu_d) / sd_d)[0]) - float(-mu_d[0] / sd_d[0])) < 1e-12)

    # ③ 主干与 F_abs 一致，仅输入通道不同
    n21 = sum(p.numel() for p in A.FAbs(dev).net.parameters())
    n27 = sum(p.numel() for p in A.FAbs(dev, in_ch=27).net.parameters())
    chk("3 **复用 `A.FAbs` 主干**，仅首层输入通道 21→27", n27 - n21 == 6 * 64 * 5,
        f"参数 {n21} → {n27}（差 {n27-n21} = 6×64×5）")

    # ④ 聚合顺序：先误差后平均 ≠ 先平均预测
    idx2 = {int(x.seed) for x in P2}
    chk("4 **P2 = 3 个探针种子 × 2 集**", len(idx2) == 3 and len(P2) == 6, str(sorted(idx2)))
    chk("4b P1 = 8 个互不相同的探针种子", len({int(x.seed) for x in P1}) == 8 == len(P1))
    chk("4c P3 复用 P1 的探针种子（跨幅值，非独立折）",
        {int(x.seed) for x in P3} <= {int(x.seed) for x in P1}, str(sorted(int(x.seed) for x in P3)))
    fake = {0: [1.0, 3.0, 1.0, 1.0, 1.0, 1.0], 1: [5.0, 3.0, 1.0, 1.0, 1.0, 1.0],
            2: [3.0, 3.0, 1.0, 1.0, 1.0, 1.0]}
    agg = aggregate(P2, fake)
    s0 = sorted(idx2)[0]
    chk("4d **聚合 = 先两集误差平均、再三训练种子平均**",
        abs(agg[s0] - np.mean([np.mean([fake[k][0], fake[k][1]]) for k in (0, 1, 2)])) < 1e-12,
        f"种子 {s0} → {agg[s0]:.6f}（= mean_k mean_e，非 mean over 12 混合）")
    # 先平均预测会给出不同答案 —— 用能量的凸性构造反例
    chk("4e 与「先平均预测再评分」确实不同（凸性 ⇒ 后者更小）",
        np.mean([1.0, 25.0]) > (np.mean([1.0, 5.0])) ** 2,
        "mean(x²)=13 > (mean x)²=9")

    # ⑤ β 公式：拟合残差而非总 gap；且解析最优
    rng = np.random.default_rng(0)
    fakeb = [rng.standard_normal(np.asarray(x.y).shape) for x in P1]
    bet = fit_beta(P1, fakeb, H, K, S1)
    num = den = 0.0
    for x, p0 in zip(P1, fakeb):
        r = Pfilt(np.asarray(x.y, np.float64) - p0, K); h = Pfilt(S1.causal_conv(H, x.dq), K)
        w = 1.0 / (3 * len(h)); num += w * (r * h).sum(); den += w * (h * h).sum()
    chk("5 **β 用扣除名义预测后的残差**，w_e = 1/(3T_e)，全局单标量",
        abs(bet - num / den) < 1e-12, f"β(随机名义) = {bet:+.6f}")
    betg = fit_beta(P1, [np.zeros_like(np.asarray(x.y, np.float64)) for x in P1], H, K, S1)
    chk("5b 与「拟合总 gap」确实不同（ê_base=0 才退化成总 gap）",
        abs(bet - betg) > 1e-6, f"残差版 {bet:+.5f} vs 总 gap 版 {betg:+.5f}")

    # ⑥ 常量与范围
    chk("6 训练种子 0/1/2、门槛 5%、`H` 形状 (16,3,6)",
        TRAIN_SEEDS == (0, 1, 2) and GATE_D == 0.05 and H.shape == (16, 3, 6))
    chk("6b `H` 哈希 = 冻结的 step1_H_v1",
        hashlib.sha256(np.ascontiguousarray(H, np.float64).tobytes()).hexdigest()
        == "89d118d4b69837f05636c1a73f3aefbe8d8c2a2fe3f35e37ba647a6c0c4164c2")
    print(f"\n  ⇒ 自测 {sum(ok.values())}/{len(ok)} 通过")
    return all(ok.values())


def main() -> int:
    import gc; gc.disable()
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--steps", type=int, default=A.STEPS)
    a = ap.parse_args()
    import fit_step1_candidate as S1
    from lerobot.common.policies.act.b_runtime import IdealChunkRenderer
    cal = json.loads((ROOT / "outputs/step1_H/step1_H_v1.json").read_text())["calibration"]
    R = np.asarray(cal["R"], np.float64); b = np.asarray(cal.get("b", [0, 0, 0]), np.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rend = IdealChunkRenderer()
    K = A.bp_kernel(S1)
    H = np.ascontiguousarray(np.load(H_PATH)["H"], np.float64)
    P1, P2, P3 = (load_tier(S1, rend, R, b, t) for t in ("P1", "P2", "P3"))
    print(f"P1 {len(P1)} 集（种子 {sorted(int(e.seed) for e in P1)}）· "
          f"P2 {len(P2)} 集（{sorted(int(e.seed) for e in P2)}）· "
          f"P3 {len(P3)} 集（{sorted(int(e.seed) for e in P3)}）   device={dev}")
    if a.self_test:
        return 0 if self_test(P1, P2, P3, K, dev, S1, H) else 1

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    # ── ① 恢复名义分支 ────────────────────────────────────────────────
    print("\n" + "=" * 78 + "\n① 恢复名义分支（臂 E · seed 0 · 留出 ab4/P0）\n" + "=" * 78)
    eps_bare = A.load_all(S1, rend, R, b)
    rN, mN, muN, sdN = None, None, None, None
    rN, mN, muN, sdN, okN = restore_nominal(eps_bare, K, dev)
    print(f"    留出 ab4/P0 的 redu = {rN['redu']*100:.4f}%")
    if not okN:
        print("\n  ⛔ **完整性检查不通过 ⇒ 停止**（不得继续）"); return 3
    ck = OUT / "fabs_nominal_ab4.pt"
    torch.save({"state_dict": mN.net.state_dict(), "mu": muN, "sd": sdN,
                "spec": NOMINAL, "redu_ab4": rN["redu"]}, ck)
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    print(f"    ✓ 已保存权重 + 归一化统计 → {ck.name}  sha256 {sha[:32]}…")

    # ── 名义分支在各层的预测（冻结）──────────────────────────────────
    base = {t: ebase_predict(mN, muN, sdN, E, dev) for t, E in (("P1", P1), ("P2", P2), ("P3", P3))}

    def redu_of(E, preds):
        num = den = 0.0
        for e, p in zip(E, preds):
            n_, _ = A.err_energy(np.asarray(e.y, np.float64) - p, K)
            d_, _ = A.err_energy(np.asarray(e.y, np.float64), K)
            num += n_; den += d_
        return 1.0 - num / den

    # ── 事前停止条件：基线迁移前提 ────────────────────────────────────
    print("\n" + "=" * 78 + "\n事前停止条件：基线迁移前提\n" + "=" * 78)
    ra = {t: redu_of(E, base[t]) for t, E in (("P1", P1), ("P2", P2), ("P3", P3))}
    for t in ("P1", "P2", "P3"):
        print(f"    (a) ê_base 在 {t} 的 redu = {ra[t]*100:+.2f}%")
    if not (ra["P1"] > 0 and ra["P2"] > 0):
        print("\n  ⛔ **基线迁移前提未满足 ⇒ 停止。**")
        print("     ⚠ 这**不是**「三项比较不可解读」，**也不是**「神经增量分支失败」"
              "—— `F_inc` 在该情形下根本没被检验。")
        (OUT / "finc_ab4_v1.json").write_text(json.dumps(
            {"stopped": "基线迁移前提未满足", "redu_a": ra}, indent=2, ensure_ascii=False))
        return 4
    print("    ✓ 基线迁移前提满足（P1、P2 上 (a) 均为正）")

    # ── ④ β（只用 P1）─────────────────────────────────────────────────
    beta = fit_beta(P1, base["P1"], H, K, S1)
    print(f"\n④ β = **{beta:+.6f}**（只用 P1，全局单标量，拟合扣除名义预测后的残差）"
          + ("\n    ⚠ **β 为负 ⇒ 含方向反转，不能只称「幅度校准」**" if beta < 0 else ""))

    hconv = {t: [S1.causal_conv(H, e.dq) for e in E]
             for t, E in (("P1", P1), ("P2", P2), ("P3", P3))}

    # ── ② 训练 F_inc（3 个训练种子）───────────────────────────────────
    print("\n" + "=" * 78 + f"\n② 训练 F_inc（只用 P1 {len(P1)} 集；训练种子 {TRAIN_SEEDS}）\n"
          + "=" * 78)
    errs = {t: {} for t in ("P2", "P3")}
    hists, secs = {}, {}
    for k in TRAIN_SEEDS:
        print(f"  训练种子 {k}")
        net, stats, hist, sec = train_finc(P1, base["P1"], K, dev, k, steps=a.steps,
                                           log=lambda m: print(m))
        hists[k], secs[k] = hist, sec
        for t, E in (("P2", P2), ("P3", P3)):
            pf = finc_predict(net, stats, E, dev)
            errs[t][k] = [Eerr(np.asarray(e.y, np.float64) - (b0 + f), K)[0]
                          for e, b0, f in zip(E, base[t], pf)]
        pv = redu_of(P2, [b0 + f for b0, f in zip(base["P2"], finc_predict(net, stats, P2, dev))])
        print(f"      → (c) 在 P2 的 redu = {pv*100:+.2f}%   [{sec:.0f}s]")

    # ── 聚合与判据 ────────────────────────────────────────────────────
    res = {}
    for t, E in (("P2", P2), ("P3", P3)):
        Ea = aggregate(E, {0: [Eerr(np.asarray(e.y, np.float64) - p, K)[0]
                               for e, p in zip(E, base[t])]})
        Eb = aggregate(E, {0: [Eerr(np.asarray(e.y, np.float64) - (p + h), K)[0]
                               for e, p, h in zip(E, base[t], hconv[t])]})
        Eb2 = aggregate(E, {0: [Eerr(np.asarray(e.y, np.float64) - (p + beta * h), K)[0]
                                for e, p, h in zip(E, base[t], hconv[t])]})
        Ec = aggregate(E, errs[t])
        ds_b, db_b = dbar(Eb, Ec); ds_b2, db_b2 = dbar(Eb2, Ec)
        res[t] = {"E_a": Ea, "E_b": Eb, "E_bp": Eb2, "E_c": Ec,
                  "d_b": ds_b, "dbar_b": db_b, "d_bp": ds_b2, "dbar_bp": db_b2,
                  "mean_E_a": float(np.mean(list(Ea.values()))),
                  "mean_E_c": float(np.mean(list(Ec.values())))}

    p2 = res["P2"]
    gate_b = all(v > 0 for v in p2["d_b"].values()) and p2["dbar_b"] >= GATE_D
    gate_bp = all(v > 0 for v in p2["d_bp"].values()) and p2["dbar_bp"] >= GATE_D
    gate_a = p2["mean_E_c"] < p2["mean_E_a"]
    advance = bool(gate_b and gate_bp and gate_a)
    hrs = (time.time() - t_all) / 3600

    print("\n" + "=" * 78 + "\n判据（P2 主；P3 次要压力）\n" + "=" * 78)
    for t in ("P2", "P3"):
        r_ = res[t]; ss = sorted(r_["E_a"])
        print(f"\n  【{t}】按探针种子等权的误差 E（越小越好）")
        print(f"    {'探针种子':<10}{'(a) ê_base':>14}{'(b) H':>12}{'(b′) βH':>12}{'(c) F_inc':>12}"
              f"{'d vs b':>10}{'d vs b′':>10}")
        for s in ss:
            print(f"    {s:<10}{r_['E_a'][s]:>14.6g}{r_['E_b'][s]:>12.6g}{r_['E_bp'][s]:>12.6g}"
                  f"{r_['E_c'][s]:>12.6g}{r_['d_b'][s]*100:>+9.2f}%{r_['d_bp'][s]*100:>+9.2f}%")
        print(f"    {'等权平均':<8}{r_['mean_E_a']:>14.6g}{'':>12}{'':>12}{r_['mean_E_c']:>12.6g}"
              f"{r_['dbar_b']*100:>+9.2f}%{r_['dbar_bp']*100:>+9.2f}%")

    print("\n  推进三条（**全部满足才算推进**）：")
    print(f"    1. vs (b)  冻结 H ：三种子全正 = {all(v>0 for v in p2['d_b'].values())} · "
          f"d̄ = {p2['dbar_b']*100:+.2f}% ≥ 5% = {p2['dbar_b'] >= GATE_D}  ⇒ "
          f"{'**满足**' if gate_b else '**不满足**'}")
    print(f"    2. vs (b′) 标量校准 H：三种子全正 = {all(v>0 for v in p2['d_bp'].values())} · "
          f"d̄ = {p2['dbar_bp']*100:+.2f}% ≥ 5% = {p2['dbar_bp'] >= GATE_D}  ⇒ "
          f"{'**满足**' if gate_bp else '**不满足**'}")
    print(f"    3. (c) 误差低于 (a)：{p2['mean_E_c']:.6g} < {p2['mean_E_a']:.6g}  ⇒ "
          f"{'**满足**' if gate_a else '**不满足**'}")
    print(f"\n  **推进：{'**是**' if advance else '**否**'}**")
    if gate_b and not gate_bp:
        print("    ⚠ **只赢 (b) 而输给 (b′) ⇒ 必须写成「未能优于标量校准后的 H」，不算通过。**")
    print(f"\n  P3（**次要压力，不挽救 P2**）：d̄ vs b = {res['P3']['dbar_b']*100:+.2f}% · "
          f"vs b′ = {res['P3']['dbar_bp']*100:+.2f}%")
    print("    ⚠ P3 均值 > 0 **只能记作平均改善**，不能据此宣称已覆盖 3 mrad RMS。")
    print(f"\n  诊断：(b) 在 P2 的 redu = "
          f"{redu_of(P2, [p+h for p,h in zip(base['P2'], hconv['P2'])])*100:+.2f}% "
          f"vs (a) {ra['P2']*100:+.2f}%")
    print(f"\n  用时 {hrs:.3f} GPU·小时（预算 0.5，**含名义分支恢复**）"
          + ("" if hrs <= 0.5 else "   ⚠ **超预算 ⇒ 记为未完成**"))

    (OUT / "finc_ab4_v1.json").write_text(json.dumps(
        {"nominal": {**NOMINAL, "redu_ab4": rN["redu"], "ckpt_sha256": sha,
                     "integrity": "完整精度逐集 num/den 与两份归档逐位相等"},
         "redu_a": ra, "beta": beta, "results": res,
         "gates": {"vs_b": gate_b, "vs_bp": gate_bp, "vs_a": gate_a, "advance": advance},
         "train_seeds": list(TRAIN_SEEDS), "gate_d": GATE_D, "steps": a.steps,
         "loss_history": hists, "sec": secs, "gpu_hours": hrs,
         "scope": "**限定 ab4 工况的开发实验；不能视为全会话 G0 已通过；"
                  "不代表跨会话泛化或真机迁移；ab2 保留。**"},
        indent=2, ensure_ascii=False, default=float))
    print(f"→ {OUT/'finc_ab4_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
