#!/usr/bin/env python3
"""从冻结的 `g2-incr-v1` 抽出**逐时刻局部增量算子** `J[t,k] ∈ R^{3×6}`，并验证其忠实度。

    Δω̂[t] ≈ Σ_{k=0}^{M} J[t,k] · δa[t−k]

为什么需要它
------------
训练 π 的损失必须用**冻结的 g2-incr-v1**（用户 2026-09-02）。但该模型的增量是
**MuJoCo 前向差分**：不可微、且每次评估要重跑 rollout —— 无法进训练内循环。

而**不能**用一个全局 LTI 核代替：实测同一关节的逐发响应范数随位形差
**4×(j2) – 27×(j3)** ⇒ 增量算子是**时变**的。

抽法（每集 12 次前向）
----------------------
对每个关节 j，注入**间隔 `M+1` 帧**的 ± 配对小脉冲（响应窗互不重叠），
取奇分量 `(ω̂⁺−ω̂⁻)/2` 消掉偶次项与任务基线：

    J[t₀, k, :, j] = odd[t₀+k] / AMP          脉冲落在 t₀

再沿 t 线性插值到全时刻（位形变化远慢于 0.8 s 窗）。
得到的算子**关于 δa 线性、可微**，且**保留位形依赖**。

幅度 1 mrad ⇒ 加速度需求 `2A/dt² = 46°/s²`，远低于 `A_MAX`(307–391) ⇒ 限幅器基本不介入。

验证（必须过，否则训练建在沙子上）
----------------------------------
用**未参与抽取**的波形（8 帧 5 Hz 小波，即 ③a 的探针）做留出检验：
比较 `Σ J·δa` 与**真实前向差分** `[ω̂(u+δa) − ω̂(u−δa)]/2`。
报 cos / 范数比 / 相对误差。cos < 0.9 或范数比偏离 1 超过 ±20% ⇒ 不可用于训练。

用法:
    python -u sim/step4_gvnext/incr_operator.py            # 抽取 + 验证（训练族）
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g", ROOT / "sim/step3_canobs", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data import load, bp, valid_files, FPS                          # noqa: E402
from train_rq import features as rq_features, fir_design             # noqa: E402
from train_rs import rs_features                                     # noqa: E402
from eval_gate import base_q                                         # noqa: E402
from ceiling_200hz import kin_gyro_fs                                # noqa: E402

S1, S2 = ROOT / "outputs/step4_gvnext/s1", ROOT / "outputs/step4_gvnext/s2"
RQ = ROOT / "outputs/step4_gvnext/rq"
ART = ROOT / "outputs/step4_gvnext/g2_incr_v1.npz"
OUT = ROOT / "outputs/step4_gvnext/incr_operator"
M = 15                       # 核长 M+1 = 16，与 H 同
AMP = 0.001                  # 1 mrad 探测脉冲
BAND = (0.5, 9.0)
COS_GATE, NR_TOL = 0.90, 0.20


class Model:
    """冻结的 g2-incr-v1 前向。"""

    def __init__(self):
        z = np.load(ART, allow_pickle=False)
        self.man = json.loads(str(z["manifest_json"]))
        self.W = np.ascontiguousarray(z["W"], float)
        assert hashlib.sha256(self.W.tobytes()).hexdigest() == self.man["W_sha256"]
        self.nlag = int(z["nlag"])
        self.zq = np.load(RQ / "rq_fir.npz")
        self.sq = json.loads((RQ / "select.json").read_text())

    def omega(self, dd, t20):
        qh = base_q(dd, t20) + fir_design(rq_features(dd)[0], self.sq["nlag"]) @ self.zq["W"]
        n = len(t20)
        return kin_gyro_fs(qh, FPS)[:n] + fir_design(rs_features(dd, t20, qh)[:n],
                                                     self.nlag) @ self.W

    def odd(self, dd, t20, da_loop):
        """± 配对奇分量：`[ω̂(u+δa) − ω̂(u−δa)]/2`。"""
        out = []
        for s in (+1.0, -1.0):
            c = np.array(dd["cmd"], float, copy=True)
            m = min(len(c), len(da_loop))
            c[:m, :6] += s * da_loop[:m]
            out.append(self.omega({**dd, "cmd": c}, t20))
        m = min(len(out[0]), len(out[1]))
        return 0.5 * (out[0][:m] - out[1][:m])


def extract(mdl, dd, t20, tr):
    """→ J（T, M+1, 3, 6），逐时刻局部增量核。每集 12 次前向。"""
    T = len(tr)
    step = M + 1
    t0s = np.arange(step, T - 2 * step, step)          # 脉冲位置，响应窗不重叠
    J = np.zeros((len(t0s), M + 1, 3, 6))
    for j in range(6):
        da = np.zeros((T, 6))
        da[t0s, j] = AMP
        od = mdl.odd(dd, t20, da)
        od_loop = np.stack([np.interp(tr, np.asarray(t20, float)[:len(od)], od[:, c])
                            for c in range(3)], 1)
        for i, t0 in enumerate(t0s):
            w = od_loop[t0:t0 + M + 1]
            J[i, :len(w), :, j] = w / AMP
    return t0s, J


def apply_J(t0s, J, da, T):
    """`Δω̂[t] = Σ_k J[t−k, k] δa[t−k]`，J 沿 t 线性插值到全时刻。"""
    Jf = np.stack([np.stack([np.stack([np.interp(np.arange(T), t0s, J[:, k, c, j])
                                       for j in range(6)], -1)
                             for c in range(3)], -2) for k in range(M + 1)], 1)  # (T,M+1,3,6)
    out = np.zeros((T, 3))
    for k in range(M + 1):
        if k:
            out[k:] += np.einsum("tcj,tj->tc", Jf[:T - k, k], da[:T - k])
        else:
            out += np.einsum("tcj,tj->tc", Jf[:, 0], da)
    return out


def main() -> int:
    import gc
    gc.disable()
    mdl = Model()
    print(f"模型 = {mdl.man['artifact']}（v={mdl.man['v']}, W sha "
          f"{mdl.man['W_sha256'][:16]}…）· 核长 {M+1} · 探测幅度 {AMP*1e3:.0f} mrad")
    print(f"验证波形 = 8 帧 5 Hz 小波（③a 探针，**未参与抽取**）\n")
    wav = np.array([1, 1, -1, -1, 1, 1, -1, -1], float)

    print(f"  {'集':<16}{'cos':>9}{'范数比':>9}{'相对误差':>10}{'判':>5}")
    rows, nok = [], 0
    files = list(valid_files(S1))[:6] + list(valid_files(S2, "replay_pp2f*.json"))[:3]
    for f in files:
        dd = load(f)
        t20 = dd["tg_i"][::5]
        tr = np.asarray(dd["t_read"], float)
        T = len(tr)
        t0s, J = extract(mdl, dd, t20, tr)
        # 留出波形：与抽取用的脉冲**位置和形状都不同**
        rng = np.random.default_rng(int.from_bytes(
            hashlib.sha256(dd["name"].encode()).digest()[:4], "little"))
        da = np.zeros((T, 6))
        for t in range(30, T - 40, 37):                 # 37 ≠ 16，错开
            da[t:t + 8, rng.integers(0, 6)] = 0.005 * wav
        true = mdl.odd(dd, t20, da)
        true_l = np.stack([np.interp(tr, np.asarray(t20, float)[:len(true)], true[:, c])
                           for c in range(3)], 1)
        appr = apply_J(t0s, J, da, T)
        a, b = bp(true_l, *BAND), bp(appr, *BAND)
        cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        nr = float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-30))
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
        ok = cos > COS_GATE and abs(nr - 1) < NR_TOL
        nok += int(ok)
        rows.append({"name": dd["name"], "cos": cos, "norm_ratio": nr, "rel_err": rel,
                     "ok": ok})
        print(f"  {dd['name']:<16}{cos:>9.3f}{nr:>9.3f}{rel:>10.3f}{'  ✓' if ok else ' ✗':>5}")
        np.savez_compressed(OUT / f"J_{dd['name']}.npz", t0s=t0s, J=J.astype(np.float32))
    med = lambda k: float(np.median([r[k] for r in rows]))
    print(f"\n  中位: cos {med('cos'):.3f} · 范数比 {med('norm_ratio'):.3f} · "
          f"相对误差 {med('rel_err'):.3f}   通过 {nok}/{len(rows)}")
    good = med("cos") > COS_GATE and abs(med("norm_ratio") - 1) < NR_TOL
    print(f"  门槛 cos>{COS_GATE} 且 |范数比−1|<{NR_TOL}  ⇒ "
          f"{'**忠实，可用于训练 π**' if good else '**不忠实，不得用于训练**'}")
    (OUT / "verify.json").write_text(json.dumps(
        {"rows": rows, "median": {k: med(k) for k in ("cos", "norm_ratio", "rel_err")},
         "pass": bool(good), "M": M, "amp_rad": AMP,
         "model": mdl.man["artifact"], "W_sha256": mdl.man["W_sha256"],
         "caliber": ("J 由 ±配对小脉冲(1mrad, 间隔 M+1 帧)从**冻结模型**抽出, 沿 t 线性插值; "
                     "验证波形(8帧小波, 位置间隔 37 帧)**未参与抽取**; 带 0.5-9Hz"),
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0 if good else 2


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rc = main()
    sys.stdout.flush()
    import os
    os._exit(rc)
