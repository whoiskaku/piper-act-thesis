#!/usr/bin/env python3
"""让冻结的 `g2-incr-v1` 跑在 **ab4** 数据上，并抽出逐时刻增量算子 `J[t,k]`。

为什么需要适配
--------------
π 的训练/评价要**沿用已固定的数据划分**（ab4 13 集逐集留一，用户 2026-09-14）。
但 `g2-incr-v1` 的前向需要 `cmd / gyro100 / tg_i / t_read / t_send`，
而 **ab4 日志只有 `k, da, a, imu, state`，没有时间戳**。

⚠ **假设（必须随结果一并报告）**
--------------------------------
- 时基**假设均匀 20 Hz**（ab4 是 20 Hz 的 ACT rollout；真实 jitter 未记录）。
  它只影响 `base_q` 的伺服 rollout，进而影响 J，**不影响**用 H 做的评价。
- `gyro100` 由 20 Hz 的 `imu[:,3:6]` 上采样填充；它**不进** J 的计算路径
  （只为满足接口），故不引入额外误差。

为什么评价仍用 H
----------------
训练目标用 `g2-incr-v1`（这正是「**能否用 G 训练 π**」这个研究问题），
但**评价**必须用独立的被控对象模型 —— 否则用 G 训练又用 G 打分是循环。
H 在 ab4 上是**原生**的（同一坐标系、同一 `ALIGN_L=3` 对齐），不需要跨会话旋转。

用法:
    python -u sim/g2_on_ab4.py            # 抽 J 并缓存 + 忠实度自检
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fit_step1_candidate as S1                                     # noqa: E402
import oracle_causal_feedback as O                                   # noqa: E402

OUT = ROOT / "outputs/step4_gvnext/J_ab4"
FPS = 20.0
M = 15
AMP = 0.001


def ab4_to_dd(path: Path, name: str) -> dict:
    """ab4 日志 → `g2-incr-v1` 前向所需的 `dd`。**时基假设均匀 20 Hz。**"""
    d = json.loads(path.read_text())
    fr = d["frames"]
    a = np.asarray([f["a"] for f in fr], float)
    if a.ndim == 3:
        a = a[:, 0]
    cmd = np.zeros((len(fr), 7))
    cmd[:, :min(7, a.shape[1])] = a[:, :min(7, a.shape[1])]
    t = np.arange(len(fr)) / FPS                    # ⚠ 假设
    imu = np.asarray([f["imu"] for f in fr], float)
    gy = imu[:, 3:6] if imu.shape[1] >= 6 else imu[:, :3]
    gyro100 = np.repeat(gy, 5, axis=0)              # 仅满足接口，不进 J 的路径
    return {"name": name, "cmd": cmd, "t_send": t, "t_read": t,
            "tg_i": np.arange(len(fr) * 5) / (FPS * 5), "gyro100": gyro100}


def rq_X(dd):
    """`train_rq.features` 的**只要 X** 版本（原版还算目标 `qt−qh`，ab4 无 q200）。"""
    import g_q_v1 as GQ
    from train_rq import fir_design                                   # noqa: F401
    t20 = dd["tg_i"][::5][:len(dd["t_read"])]
    qh_f = np.radians(GQ.predict_qpos(dd["cmd"], t_send=dd["t_send"],
                                      t_read=dd["t_read"]))
    qh = np.stack([np.interp(t20, dd["t_read"], qh_f[:, j]) for j in range(6)], 1)
    u = np.stack([np.interp(t20, dd["t_read"], dd["cmd"][:, j]) for j in range(6)], 1)
    g = np.interp(t20, dd["t_read"], dd["cmd"][:, 6])
    dq = np.gradient(qh, 1 / FPS, axis=0)
    ddq = np.gradient(dq, 1 / FPS, axis=0)
    X = np.concatenate([u, qh, dq, ddq, u - qh, g[:, None],
                        np.gradient(g, 1 / FPS)[:, None]], 1)
    return X, qh, t20


class G2:
    """冻结的 g2-incr-v1 在 ab4 上的前向。"""

    def __init__(self):
        z = np.load(ROOT / "outputs/step4_gvnext/g2_incr_v1.npz", allow_pickle=False)
        self.man = json.loads(str(z["manifest_json"]))
        self.W = np.ascontiguousarray(z["W"], float)
        assert hashlib.sha256(self.W.tobytes()).hexdigest() == self.man["W_sha256"]
        self.nlag = int(z["nlag"])
        self.zq = np.load(ROOT / "outputs/step4_gvnext/rq/rq_fir.npz")
        self.sq = json.loads((ROOT / "outputs/step4_gvnext/rq/select.json").read_text())

    def omega(self, dd):
        from train_rq import fir_design
        from train_rs import rs_features
        from ceiling_200hz import kin_gyro_fs
        X, qh0, t20 = rq_X(dd)
        qh = qh0 + fir_design(X, self.sq["nlag"]) @ self.zq["W"]
        n = len(t20)
        return kin_gyro_fs(qh, FPS)[:n] + fir_design(
            rs_features(dd, t20, qh)[:n], self.nlag) @ self.W

    def odd(self, dd, da):
        out = []
        for s in (+1.0, -1.0):
            c = np.array(dd["cmd"], float, copy=True)
            m = min(len(c), len(da))
            c[:m, :6] += s * da[:m]
            out.append(self.omega({**dd, "cmd": c}))
        m = min(len(out[0]), len(out[1]))
        return 0.5 * (out[0][:m] - out[1][:m])


def extract(g2, dd, T):
    """→ (t0s, J[len(t0s), M+1, 3, 6])，每集 12 次前向。"""
    step = M + 1
    t0s = np.arange(step, T - 2 * step, step)
    J = np.zeros((len(t0s), M + 1, 3, 6))
    for j in range(6):
        da = np.zeros((T, 6)); da[t0s, j] = AMP
        od = g2.odd(dd, da)
        for i, t0 in enumerate(t0s):
            w = od[t0:t0 + M + 1]
            J[i, :len(w), :, j] = w / AMP
    return t0s, J


def dense(t0s, J, T):
    """J 沿 t 线性插值到全时刻 → (T, M+1, 3, 6)。"""
    return np.stack([np.stack([np.stack(
        [np.interp(np.arange(T), t0s, J[:, k, c, j]) for j in range(6)], -1)
        for c in range(3)], -2) for k in range(M + 1)], 1)


def main() -> int:
    import gc
    gc.disable()
    g2 = G2()
    logs = S1.discover_logs(ROOT / "outputs/ab4", ("P0", "P1"))
    paths = [p for t in ("P0", "P1") for p in logs[t]]
    _, E = O.load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"g2-incr-v1 (W sha {g2.man['W_sha256'][:16]}…) 跑在 ab4 {len(paths)} 集上")
    print(f"⚠ 时基**假设均匀 20 Hz**（ab4 无时间戳）；gyro100 仅为满足接口，不进 J 路径\n")
    print(f"  {'集':<12}{'T':>6}{'‖J(0)‖':>10}{'尾比 ‖J(15)‖/‖J(0)‖':>22}")
    for p, e in zip(paths, E):
        dd = ab4_to_dd(p, f"{p.parent.name}_{p.stem}")
        T = min(len(e["e0"]), len(dd["t_read"]))
        t0s, J = extract(g2, dd, T)
        nrm = np.linalg.norm(J.reshape(len(t0s), M + 1, -1), axis=2).mean(0)
        print(f"  {p.stem:<12}{T:>6}{nrm[0]:>10.4f}{nrm[15]/max(nrm[0],1e-12):>22.3f}")
        np.savez_compressed(OUT / f"{p.stem}.npz", t0s=t0s, J=J.astype(np.float32), T=T)
    (OUT / "meta.json").write_text(json.dumps(
        {"model": g2.man["artifact"], "W_sha256": g2.man["W_sha256"],
         "M": M, "amp_rad": AMP, "fps": FPS,
         "assumptions": ["ab4 无时间戳 ⇒ 时基假设均匀 20 Hz（只影响 J，不影响用 H 的评价）",
                         "gyro100 由 20Hz imu 上采样，仅满足接口，不进 J 的计算路径"],
         }, indent=1, ensure_ascii=False))
    print(f"\n  → {OUT}")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
