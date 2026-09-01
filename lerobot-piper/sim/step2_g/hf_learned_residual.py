#!/usr/bin/env python3
"""**物理底座 + 学习残差**的零机时门 —— 残差是否**可因果预测**？

```
q̂ = G-q-v1(u)                       ← 冻结，不改关节轨迹
ω̂ = ω_rigid(q̂) + r_φ(x_{t−L:t})     ← 只学三轴 IMU 高频残差
```

**第一步不是训练更大的网络，是用现有数据验证残差可不可以被因果预测。**
若仍输给相位模板 ⇒ **立即停止**；那说明缺的是**可观测输入或传感器信息**，
不是模型容量。

## 数据（全部现有，零机时）

| 族 | 集数 | 来源 repo | 用途 |
|---|---|---|---|
| `pp2` | 10 | `eval_pickplace2_100k` | 训练 |
| `pp2a` | 10 | （同族 A 臂） | 训练 |
| `pp2f` | 7 | `eval_pp2_fork_100k` | **留出（跨族）** |
| `pp2f60` | 3 | `eval_pp2_fork_60k` | **整族域外**（轨迹相关仅约 0.70–0.73） |
| step1_v3 裸 ACT | 17 | 另一会话/管线 | **现有 pp2 开发集**（门禁条件之一） |

均为**裸 ACT 评估回放**（补正管线之前录的）。**都不是最终密封 holdout。**
（另有 `pp2ff` 10 集，不在本方案内，不动。）

## 输入边界（严格执行）

**允许**（部署时可得的历史量，全部来自 G-q-v1 的预测或指令侧）：
`q̂` · `q̇̂` · `q̈̂`（**因果后向差分**）· jerk 有符号值与 |jerk| 包络 ·
刚体模型预测的陀螺 `ω_rigid` · 位形 · 末端高度 · 夹爪命令变化与状态 · 实际时间间隔。

**禁止**：已判死的 `cmd−ref` 高频差（§24）· 帧号 · 任务阶段（npz 里的 `phase`）·
episode ID · 未来指令 · **真实编码器或真实 IMU 作为推理输入** ·
人工标注的 oracle 接触窗（npz 里的 `mask_contact`）。

> 相位模板与零输入网络**故意违反**上述边界（用帧号/无输入）——
> 它们是**对照**，不是候选。这正是它们的作用。

## 六个模型（同时比较）

1. **底座** `r ≡ 0`
2. **线性 FIR/NARX**（因果滞后窗 → 3 轴，ridge 闭式）
3. **相位模板**（只用帧号；**对照**）
4. **事件对齐相位模板**（按夹爪命令变化对齐后平均；夹爪事件是**允许**输入）
5. **零输入 TCN**（输入置零，只剩递归/时钟；**对照**）
6. **小型 causal TCN**（复用 `bakeoff_backbones.TCN` 的 3 层空洞卷积，
   感受野 15 帧 = 0.75 s，只换 head 为 3 轴无界输出）

**旧 GRU 的目标 / 缓存 / LOO 划分一律不沿用。**

## 划分（跨族，非留一）

训练 `pp2 + pp2a`（20）→ 留出 `pp2f`（7）→ 整族域外 `pp2f60`（3）。
**归一化与模型选择只用训练侧。**

## 门禁（预注册，跑前定死）

现有数据上可评的条件：

| # | 条件 | 阈值 |
|---|---|---|
| 1 | 留出 `pp2f` 总体（0.5–9 Hz）改善 | **≥ 10%** |
| 2 | **3–9 Hz** 残差明显下降 | ≥ 10% |
| 3 | **0.5–3 Hz** 不恶化 | ≤ +5% |
| 4 | 比**相位模板**再好 | **≥ 5%** |
| 5 | 比**事件对齐模板**再好 | **≥ 5%** |
| 6 | 现有 pp2 开发集（17 集）改善数 / 中位 | **≥ 9/17** 且中位 > 0 |
| 7 | 整族域外 `pp2f60` 不恶化 | ≤ +5% |

**若不过 ⇒ 立即停止，不采 pilot、不加模型容量。**
（「≥2/3 轨迹族、4/6 唯一轨迹」等条件属于**12 集 pilot 之后**的阶段，本轮不评。）

用法:
    python sim/step2_g/hf_learned_residual.py --run
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
LROOT = ROOT / "outputs/p2_labels"
OUT = ROOT / "outputs/step2_g/hf_learned"
TRAIN = ("pp2", "pp2a")
DEV = ("pp2f",)
OOD = ("pp2f60",)
NLAG = 16                     # 因果滞后窗（0.8 s），与 TCN 感受野同量级
SEED = 20260830


def band(x, lo, hi):
    from scipy.signal import firwin
    k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
    x = np.asarray(x, float)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def load_family(tag):
    """→ [{name, act7, real_gyro}]，只取**允许**的原始量。"""
    d = LROOT / f"{tag}-deploy"
    man = json.loads((d / "manifest.json").read_text())
    R = np.asarray(man["R"], float)
    out = []
    for f in sorted(d.glob("ep*.npz")):
        z = np.load(f)
        a = np.asarray(z["a_arm"], float)
        g = np.asarray(z["grip"], float).reshape(-1, 1)
        out.append({"name": f"{tag}/{f.stem}", "act": np.hstack([a, g]),
                    "real": np.asarray(z["real"], float)[:, 3:6], "R": R})
    return out


def load_step1v3():
    from sim.step2_gap_baseline import bare_logs
    cal = json.loads((LROOT / "pp2a-deploy/manifest.json").read_text())
    R = np.asarray(cal["R"], float)
    out = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        out.append({"name": f"v3/{p.stem}",
                    "act": np.asarray([f["a"] for f in fr], float),
                    "real": np.asarray([f["imu"] for f in fr], float)[:, 3:6], "R": R})
    return out


def featurize(e):
    """→ (X 因果特征, base 刚体陀螺, tgt 带通残差, ev 夹爪事件帧)。

    **所有特征都来自 G-q-v1 的预测或指令侧**；真实编码器/IMU 只用于构造目标。
    """
    from ab_decompose import kin_gyro
    import g_q_v1 as GQ
    act, R = e["act"], e["R"]
    q = GQ.predict_qpos(act)                       # (T,6) 度 —— 冻结底座
    n = min(len(q), len(e["real"]) - 1)            # L_eval = 1
    q = q[:n]
    base = kin_gyro(q) @ R.T                       # ω_rigid(q̂)
    # 因果后向差分
    dq = np.vstack([np.zeros((1, 6)), np.diff(q, axis=0)]) * FPS
    ddq = np.vstack([np.zeros((1, 6)), np.diff(dq, axis=0)]) * FPS
    jrk = np.vstack([np.zeros((1, 6)), np.diff(ddq, axis=0)]) * FPS
    env = np.abs(jrk)
    for k in range(1, n):                          # 因果指数包络
        env[k] = np.maximum(env[k], 0.75 * env[k - 1])
    grip = act[:n, 6:7] if act.shape[1] > 6 else np.zeros((n, 1))
    dgrip = np.vstack([np.zeros((1, 1)), np.diff(grip, axis=0)])
    eez = np.zeros((n, 1))                         # 末端高度（由 q̂ 正运动学）
    import mujoco
    from sim.step2_gap_baseline import MODEL
    m = mujoco.MjModel.from_xml_path(str(MODEL)); d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    for t in range(n):
        d.qpos[:6] = np.radians(q[t]); mujoco.mj_forward(m, d)
        eez[t, 0] = d.site_xpos[sid][2]
    dt_col = np.full((n, 1), 1.0 / FPS)
    X = np.hstack([q / 100.0, dq / 100.0, ddq / 1000.0, jrk / 1e4,
                   env / 1e4, base, eez * 10.0, grip * 20.0, dgrip * 200.0, dt_col])
    tgt = band(e["real"][1:1 + n], 0.5, 9.0) - band(base, 0.5, 9.0)
    ev = np.where(np.abs(dgrip[:, 0]) > 0.5 * max(np.abs(dgrip).max(), 1e-9))[0]
    return X.astype(np.float32), base, tgt.astype(np.float32), ev


def lagwin(X, nlag=NLAG):
    """因果滞后窗 (T, nlag*d)，起始处零填充。"""
    T, dd = X.shape
    out = np.zeros((T, nlag * dd), np.float32)
    for k in range(nlag):
        out[k:, k * dd:(k + 1) * dd] = X[:T - k]
    return out


# ── 六个模型 ──
def m_zero(tr, ep):
    return np.zeros_like(ep["tgt"])


def m_linear(tr, ep, lam=1.0):
    A = np.vstack([lagwin(e["X"]) for e in tr]); Y = np.vstack([e["tgt"] for e in tr])
    mu, sd = A.mean(0), A.std(0) + 1e-6
    An = (A - mu) / sd
    G = An.T @ An + lam * len(An) * np.eye(An.shape[1], dtype=np.float32)
    W = np.linalg.solve(G, An.T @ Y)
    return ((lagwin(ep["X"]) - mu) / sd) @ W


def m_phase(tr, ep):
    """**对照**：只用帧号。"""
    T = max(len(e["tgt"]) for e in tr)
    S = np.zeros((T, 3)); C = np.zeros((T, 1))
    for e in tr:
        k = len(e["tgt"]); S[:k] += e["tgt"]; C[:k] += 1
    tpl = S / np.maximum(C, 1)
    n = len(ep["tgt"])
    return tpl[:n] if n <= len(tpl) else np.vstack([tpl, np.zeros((n - len(tpl), 3))])


def m_event(tr, ep, W=20):
    """按**夹爪命令变化**对齐后平均（夹爪事件是允许输入）。"""
    S = np.zeros((2 * W + 1, 3)); C = np.zeros((2 * W + 1, 1))
    for e in tr:
        for i in e["ev"]:
            a, b = max(0, i - W), min(len(e["tgt"]), i + W + 1)
            S[a - i + W:b - i + W] += e["tgt"][a:b]; C[a - i + W:b - i + W] += 1
    tpl = S / np.maximum(C, 1)
    out = np.zeros_like(ep["tgt"])
    for i in ep["ev"]:
        a, b = max(0, i - W), min(len(out), i + W + 1)
        out[a:b] += tpl[a - i + W:b - i + W]
    return out


def _tcn(tr, ep_list, zero_input, epochs=300):
    import torch, torch.nn as nn
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = tr[0]["X"].shape[1]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.c = nn.ModuleList([nn.Conv1d(d, 32, 3, dilation=1),
                                 nn.Conv1d(32, 32, 3, dilation=2),
                                 nn.Conv1d(32, 32, 3, dilation=4)])
            s.p = [2, 4, 8]
            s.h = nn.Linear(32, 3)

        def forward(s, u):
            x = u.transpose(1, 2)
            for c, p in zip(s.c, s.p):
                x = torch.nn.functional.gelu(c(torch.nn.functional.pad(x, (p, 0))))
            return s.h(x.transpose(1, 2))

    net = Net().to(dev)
    A = np.vstack([e["X"] for e in tr]); mu, sd = A.mean(0), A.std(0) + 1e-6
    def prep(e):
        f = (e["X"] - mu) / sd
        if zero_input:
            f = np.zeros_like(f)
        return (torch.tensor(f, device=dev)[None],
                torch.tensor(e["tgt"], device=dev)[None])
    XY = [prep(e) for e in tr]
    w = torch.tensor(np.array([len(e["X"]) for e in tr], np.float32), device=dev)
    w = w / w.sum()
    opt = torch.optim.Adam(net.parameters(), 2e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(wi * torch.nn.functional.mse_loss(net(x), y)
                   for wi, (x, y) in zip(w, XY))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    outs = []
    with torch.no_grad():
        for e in ep_list:
            outs.append(net(prep(e)[0])[0].cpu().numpy())
    return outs


def metric(ep, pred, lo, hi):
    """→ (残差能量, 实测能量) 在子带 [lo,hi]。r/m = num/den。"""
    err = ep["tgt"] - pred
    num = float(np.square(band(err, lo, hi)).sum())
    den = float(np.square(band(ep["real1"], lo, hi)).sum())
    return num, max(den, 1e-30)


def agg(eps, preds, lo, hi):
    n = d = 0.0
    per = []
    for e, p in zip(eps, preds):
        a, b = metric(e, p, lo, hi)
        n += a; d += b; per.append(a / b)
    return n / d, np.array(per)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    a_ = ap.parse_args()
    gc.disable()
    if not a_.run:
        print(__doc__.split("用法:")[0]); return 0

    print("**物理底座 + 学习残差：残差是否可因果预测？**（零机时）\n")
    fam = {}
    for tag in TRAIN + DEV + OOD:
        fam[tag] = load_family(tag)
        print(f"  载入 {tag}: {len(fam[tag])} 集", flush=True)
    v3 = load_step1v3()
    print(f"  载入 step1_v3 裸 ACT: {len(v3)} 集", flush=True)

    def prep(lst):
        out = []
        for e in lst:
            X, base, tgt, ev = featurize(e)
            n = len(tgt)
            out.append({"name": e["name"], "X": X, "base": base, "tgt": tgt,
                        "ev": ev, "real1": e["real"][1:1 + n]})
        return out
    tr = prep([e for t in TRAIN for e in fam[t]])
    dv = prep([e for t in DEV for e in fam[t]])
    od = prep([e for t in OOD for e in fam[t]])
    vv = prep(v3)
    print(f"\n  训练 {len(tr)} · 留出(pp2f) {len(dv)} · 整族域外(pp2f60) {len(od)}"
          f" · 现有 pp2 开发集 {len(vv)}   特征维 {tr[0]['X'].shape[1]}", flush=True)

    P = {}
    P["① 底座 r≡0"] = {k: [m_zero(tr, e) for e in s] for k, s in
                      (("dev", dv), ("ood", od), ("v3", vv))}
    print("  ② 线性 FIR/NARX ...", flush=True)
    P["② 线性 FIR/NARX"] = {k: [m_linear(tr, e) for e in s] for k, s in
                            (("dev", dv), ("ood", od), ("v3", vv))}
    P["③ 相位模板(对照)"] = {k: [m_phase(tr, e) for e in s] for k, s in
                            (("dev", dv), ("ood", od), ("v3", vv))}
    P["④ 事件对齐模板"] = {k: [m_event(tr, e) for e in s] for k, s in
                          (("dev", dv), ("ood", od), ("v3", vv))}
    print("  ⑤ 零输入 TCN(对照) ...", flush=True)
    P["⑤ 零输入TCN(对照)"] = {k: _tcn(tr, s, True, a_.epochs) for k, s in
                             (("dev", dv), ("ood", od), ("v3", vv))}
    print("  ⑥ causal TCN ...", flush=True)
    P["⑥ causal TCN"] = {k: _tcn(tr, s, False, a_.epochs) for k, s in
                         (("dev", dv), ("ood", od), ("v3", vv))}

    S = {"dev": dv, "ood": od, "v3": vv}
    print(f"\n  {'模型':20s}{'留出 0.5-9':>12s}{'留出 3-9':>11s}{'留出 0.5-3':>12s}"
          f"{'域外 0.5-9':>12s}{'v3 0.5-9':>11s}{'v3 改善数':>10s}")
    T = {}
    for nm, pr in P.items():
        row = {}
        for k in ("dev", "ood", "v3"):
            for lo, hi in ((0.5, 9.0), (3.0, 9.0), (0.5, 3.0)):
                row[(k, lo, hi)] = agg(S[k], pr[k], lo, hi)
        bv3 = agg(S["v3"], P["① 底座 r≡0"]["v3"], 0.5, 9.0)[1]
        mv3 = row[("v3", 0.5, 9.0)][1]
        w = int((bv3 > mv3).sum())
        T[nm] = (row, w, float(np.median((bv3 - mv3) / bv3)))
        print(f"  {nm:20s}{row[('dev',0.5,9.0)][0]:>12.4f}{row[('dev',3.0,9.0)][0]:>11.4f}"
              f"{row[('dev',0.5,3.0)][0]:>12.4f}"
              f"{row[('ood',0.5,9.0)][0]:>12.4f}{row[('v3',0.5,9.0)][0]:>11.4f}"
              f"{w:>7d}/{len(vv):<3d}", flush=True)

    b = T["① 底座 r≡0"][0]; t = T["⑥ causal TCN"][0]
    ph = T["③ 相位模板(对照)"][0]; evm = T["④ 事件对齐模板"][0]
    K9, K39, K3 = ("dev", 0.5, 9.0), ("dev", 3.0, 9.0), ("dev", 0.5, 3.0)
    d1 = 1 - t[K9][0] / b[K9][0]
    d2 = 1 - t[K39][0] / b[K39][0]
    d3 = t[K3][0] / b[K3][0] - 1
    d4 = 1 - t[K9][0] / ph[K9][0]
    d5 = 1 - t[K9][0] / evm[K9][0]
    d7 = t[("ood", 0.5, 9.0)][0] / b[("ood", 0.5, 9.0)][0] - 1
    w6, m6 = T["⑥ causal TCN"][1], T["⑥ causal TCN"][2]
    ok = [d1 >= 0.10, d2 >= 0.10, d3 <= 0.05, d4 >= 0.05, d5 >= 0.05,
          w6 >= 9 and m6 > 0, d7 <= 0.05]
    print(f"\n{'='*90}\n**门禁**（预注册）")
    for i, (txt, v, o) in enumerate(zip(
        ["1 留出 0.5–9 Hz 改善 ≥10%", "2 留出 3–9 Hz 改善 ≥10%",
         "3 留出 0.5–3 Hz 不恶化 >5%", "4 比相位模板再好 ≥5%",
         "5 比事件模板再好 ≥5%", f"6 现有 pp2 开发集 ≥9/17 且中位>0",
         "7 整族域外 pp2f60 不恶化 >5%"],
        [f"{d1*100:+.1f}%", f"{d2*100:+.1f}%", f"{d3*100:+.1f}%", f"{d4*100:+.1f}%",
         f"{d5*100:+.1f}%", f"{w6}/{len(vv)} 中位 {m6*100:+.1f}%", f"{d7*100:+.1f}%"], ok)):
        print(f"  {txt:34s}{v:>22s}   {'✓' if o else '**✗**'}")
    good = all(ok)
    print(f"\n⇒ " + ("**通过** —— 残差可因果预测，进入 12 集 pilot" if good else
          "**未通过 ⇒ 立即停止**：不采 pilot、不加模型容量。"
          "\n   缺的是**可观测输入或传感器信息**，不是模型容量。"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate.json").write_text(json.dumps(
        {"models": {k: {f"{s}_{lo}-{hi}": v[0][(s, lo, hi)][0]
                        for s in ("dev", "ood", "v3")
                        for lo, hi in ((0.5, 9.0), (3.0, 9.0), (0.5, 3.0))}
                    for k, v in T.items()},
         "deltas": {"dev_0.5-9": d1, "dev_3-9": d2, "dev_0.5-3": d3,
                    "vs_phase": d4, "vs_event": d5, "ood": d7,
                    "v3_wins": w6, "v3_median": m6},
         "passed": bool(good)}, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'gate.json'}")
    return 0


if __name__ == "__main__":
    _c = main()
    sys.stdout.flush()
    import os
    os._exit(_c)
