#!/usr/bin/env python3
"""**训练 π 的第一步：确定 π 能看什么** —— 预测器特征消融（零机时）。

为什么先做这个
--------------
已测数字把上界框死了（同一 δa 求解 + 同一部署约束，ε=10 mrad）：

    完美预测（真实 e₀）                      **22.07%**
    线性 W(指令) + 线性 AR(IMU 残差)          **4.63%**

π 要超过 4.63%，只能靠**更好的预测**，不是更好的求解 —— 求解那半边已经是最优的。
而 `CAUSAL_CEILING.md` 记着：**指令侧**非线性 TCN(51 维) 只有 26.1%，**低于**线性
`W` 的 31.6% ⇒ 指令侧加非线性没用。所以增益只可能来自**没用过的观测**。

此前的反馈通道**只用了 IMU 残差**（lead 3 处折外 R²=0.018）。
**真机关节编码器从未进过反馈** —— 而它 200 Hz 可得，部署时就在 `observation.state` 里。
本脚本量化它值多少。

口径
----
- 全部**严格因果**：命令时刻 t 只用 `s ≤ t − ALIGN_L` 的实测量（`gap[t]=real[t+3]−ideal[t]`）。
- δa 求解、chunk 钳位、接触门控、ε=10 mrad 全部与 `oracle_constraint_ceiling.py` 相同。
- 逐集留一折外拟合预测器；被控对象 = 已验证的 H。
- **完美预测那一行用到未来 e₀，是诊断上界，不可部署、不得用于生成标签。**

用法:
    python -u sim/pi_feature_ablation.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import oracle_causal_feedback as O                                   # noqa: E402
import oracle_constraint_ceiling as CC                               # noqa: E402
import fit_step1_candidate as S1                                     # noqa: E402

EPS, KMULT = 0.010, 1.0          # 认证 ε；冻结的 λ_K
JLAG = 8                         # 反馈历史长度（帧）
LAM_FB = 1e-3


def add_state(E):
    """把真机编码器（`observation.state`，逐帧可得）挂到每集上。"""
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0", "P1"))
    paths = [p for t in ("P0", "P1") for p in logs[t]]
    by = {f"{p.parent.name}/{p.stem}": p for p in paths}
    for e, p in zip(E, paths):
        fr = json.loads(p.read_text())["frames"]
        # ⚠ `observation.state` 的单位是 **mdeg**，而命令是 **rad**（实测比值 57268
        # ≈ 1000/(π/180)）。不换算的话"跟踪误差"就等于 state 本身 —— 而 corr(state,cmd)
        # 高达 0.997–0.999，跟踪误差是**两个大数之差**，单位错就被完全淹没。
        st = np.radians(np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0)
        n = len(e["e0"])
        e["state"] = st[:n] if len(st) >= n else np.vstack(
            [st, np.repeat(st[-1:], n - len(st), 0)])
        # 编码器跟踪误差：实测关节 − 名义命令（两者都逐帧可得）
        # x[:,18:24] = action[j]（见 build_command_features 的 36 维布局）
        cmd = e["x"][:, 18:24]
        e["terr"] = np.concatenate([e["state"] - cmd,
                                    np.gradient(e["state"], axis=0)], 1)   # 误差 + 实测速度
    return E


def hist(src, t, j0=S1.ALIGN_L, n=JLAG):
    idx = [t - j0 - k for k in range(n + 1)]
    return np.concatenate([src[i] if i >= 0 else np.zeros(src.shape[1]) for i in idx])


def fit_fb(train, W, use_imu, use_enc):
    """按前瞻 ℓ 拟合反馈修正项；特征可含 IMU 残差历史与/或编码器历史。"""
    Cs = {}
    for lead in range(S1.ALIGN_L, S1.ALIGN_L + S1.N_LAGS):
        X, Y = [], []
        for e in train:
            r = e["e0"] - e["x"] @ W.T
            T = len(r)
            for s in range(JLAG + S1.ALIGN_L, T - lead):
                f = []
                if use_imu:
                    f.append(hist(r, s + S1.ALIGN_L))
                if use_enc:
                    f.append(hist(e["terr"], s + S1.ALIGN_L))
                X.append(np.concatenate(f)); Y.append(r[s + lead])
        coef, _ = S1._ridge_solve(np.asarray(X), np.asarray(Y), LAM_FB)
        Cs[lead] = coef.T
    return Cs


def roll(e, H, W, Cs, use_imu, use_enc, oracle=False):
    T = len(e["e0"])
    M1 = S1.N_LAGS
    hs_full = H.reshape(M1 * 3, 6)
    lam = KMULT * max(float(np.trace(hs_full.T @ hs_full)) / 6.0, np.finfo(float).tiny)
    da = np.zeros((T, 6)); resp = np.zeros((T + M1, 3)); r = np.zeros((T, 3))
    e0 = e["e0"]
    ff = e["x"] @ W.T if W is not None else np.zeros_like(e0)
    nsat = 0
    for t in range(T):
        s = t - S1.ALIGN_L
        if s >= 0:
            r[s] = e0[s] - ff[s] - resp[s]
        K = max(int(min(S1.M, e["cap"][t] - t)), 0)
        tgt = np.empty((K + 1, 3))
        f = None
        if Cs is not None and s >= 0:
            parts = []
            if use_imu:
                parts.append(hist(r, t))
            if use_enc:
                parts.append(hist(e["terr"], t))
            f = np.concatenate(parts)
        for k in range(K + 1):
            j = min(t + k, T - 1)
            p = e0[j] if oracle else ff[j].copy()
            if f is not None:
                p = p + Cs[min(k + S1.ALIGN_L, S1.ALIGN_L + S1.M)] @ f
            tgt[k] = p + resp[t + k]
        hs = H[:K + 1].reshape((K + 1) * 3, 6)
        d = -np.linalg.solve(hs.T @ hs + lam * np.eye(6), hs.T @ tgt.reshape(-1))
        nsat += int(np.any(np.abs(d) > EPS))
        d = np.clip(d, -EPS, EPS)
        if e["contact"][t]:
            d[:] = 0.0
        da[t] = d
        for k in range(M1):
            resp[t + k] += H[k] @ d
    return da, nsat / max(T, 1)


def main() -> int:
    import gc
    gc.disable()
    from train_rq import KFOLD
    H, E = O.load_all()
    E = add_state(E)
    print(f"ab4 {len(E)} 集 · 被控对象 = 已验证 H · ε = {EPS*1e3:.0f} mrad（认证）· "
          f"chunk 钳位 + 接触门控 · 严格因果\n")
    print(f"  {'预测器':<34}{'降幅':>9}{'δa RMS':>10}{'触界%':>8}")
    out = {}
    for name, use_imu, use_enc in (("指令 only（线性 W）", False, False),
                                   ("指令 + IMU 反馈（此前）", True, False),
                                   ("指令 + **编码器反馈**", False, True),
                                   ("指令 + IMU + 编码器", True, True)):
        das = []
        for i, e in enumerate(E):
            tr = [E[j] for j in range(len(E)) if j != i]
            W = O.fit_w(tr)
            Cs = fit_fb(tr, W, use_imu, use_enc) if (use_imu or use_enc) else None
            das.append(roll(e, H, W, Cs, use_imu, use_enc))
        r, rm, sa = CC.score(E, H, das)
        out[name] = {"reduction": r, "da_rms_mrad": rm, "sat_pct": sa}
        print(f"  {name:<34}{r*100:>8.2f}%{rm:>10.2f}{sa:>8.1f}")
    das = [roll(e, H, None, None, False, False, oracle=True) for e in E]
    r, rm, sa = CC.score(E, H, das)
    out["完美预测（诊断上界）"] = {"reduction": r, "da_rms_mrad": rm, "sat_pct": sa}
    print(f"  {'完美预测（诊断上界，不可部署）':<34}{r*100:>8.2f}%{rm:>10.2f}{sa:>8.1f}")

    best = max((k for k in out if "完美" not in k), key=lambda k: out[k]["reduction"])
    b = out[best]["reduction"] * 100
    base = out["指令 + IMU 反馈（此前）"]["reduction"] * 100
    print(f"\n  ── 判读 ──")
    print(f"  最好的可部署预测器 = **{best}** → {b:.2f}%")
    print(f"  相对此前（指令+IMU）{base:.2f}%  增量 {b-base:+.2f} pp")
    print(f"  距完美预测 {r*100:.2f}% 还差 {r*100-b:.2f} pp")
    if b - base < 1.0:
        print(f"  ⇒ **新观测没带来实质增益**：π 的信息集已到顶，训练 π 最多回到 ~{b:.1f}%。")
    else:
        print(f"  ⇒ 新观测有增益 ⇒ π 应把它纳入输入。")
    (ROOT / "outputs/step1_H/pi_feature_ablation.json").write_text(json.dumps(
        {"arms": out, "eps_rad": EPS, "jlag": JLAG,
         "caliber": ("严格因果(命令时刻 t 只用 s<=t-3 的实测量); 逐集留一折外; "
                     "被控对象=已验证 H; ε=10mrad 是**瞬时峰值**硬界(不是 RMS); "
                     "完美预测行用到未来 e0, 是诊断上界, 不可部署、不得用于生成标签"),
         }, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
