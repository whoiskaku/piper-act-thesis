"""阶段一事后诊断：两个采集时期的输入与输出差异 —— **只描述，不改阶段一的判决**。

时期（按文件写入时间，2026-10-03 核实）：
- 旧：ab2/A（07-29）、ab4/P0（08-19）；
- 近期：ab5/A（09-15 22:44 → 09-16 00:09）、ab6/A（09-16）、ab7/Apost（09-16），同一个约 9 小时的时段。

检查一（指令分布）：ACT 指令的关节位置中位数与范围、速度、加速度、chunk 边界跳变；
  以及按阶段一的折，留出会话的指令落在训练会话 1–99 百分位之外的比例（越界率）。
检查二（gap 与传感器 / 动力学）：
  - 墙相对理想信号的能量、实测/理想能量比、gap 分频占比；
  - 逐会话 IMU 旋转：沿用仓库惯例（`sim/imu_frame_calib.py`）—— 未滤波陀螺、|理想| > 0.15 rad/s 的样本、
    滞后 3 帧做 Kabsch，得到相对全局标定 R（由 ab2/A 的数据拟合）的残余旋转 Δ；
    再算「纯旋转误差 (Δ − I)·理想」最多能占带通 gap 能量的多少；
  - 关节跟踪（编码器）：指令速度与编码器速度的互相关，取峰值处的延迟 d 与幅值比 a。

⚠ 只有 2 个旧会话、3 个近期会话：任何差异都可能是会话本身的，而不是时期的；
   发现差异也不等于它导致了阶段一的失败（排除 ≠ 证明）。
不在 `q1_train.CODE_FILES` 里 ⇒ 不改变冻结实验的 code_sha；产物写到 `outputs/q1_diag_era/`。

    python -u sim/q1/diag_q1_era.py
"""
from __future__ import annotations
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import q1_data as D            # noqa: E402
import q1_windows as W         # noqa: E402

OLD = ("ab2/A", "ab4/P0")
RECENT = ("ab5/A", "ab6/A", "ab7/Apost")
ORDER = OLD + RECENT
DATE = {"ab2/A": "07-29", "ab4/P0": "08-19", "ab5/A": "09-15/16", "ab6/A": "09-16", "ab7/Apost": "09-16"}
JOINTS = (0, 1, 2, 4, 5)                  # j4（下标 3）的指令在 100 集里恒为 0，不计
JNAME = {j: f"j{j + 1}" for j in JOINTS}
DT = 1.0 / D.FS
SUBBANDS = ((0.5, 3.0), (3.0, 6.0), (6.0, 9.0))
OUT = D.ROOT / "outputs/q1_diag_era"


# ── 检查一：指令分布 ──────────────────────────────────────────────────────
def positions(ep) -> np.ndarray:
    """已执行帧的关节指令（rad）→ (T, 5)。"""
    return np.asarray(ep.cmd[:ep.T])[:, JOINTS]


def speeds(ep) -> np.ndarray:
    return np.diff(positions(ep), axis=0) / DT


def cmd_stats(ep) -> dict:
    """一集的指令统计：逐关节位置分位数、速度/加速度 RMS、chunk 边界跳变。"""
    u = positions(ep)
    v = np.diff(u, axis=0) / DT
    a = np.diff(u, n=2, axis=0) / DT ** 2
    out = {}
    for i, j in enumerate(JOINTS):
        n = JNAME[j]
        out[f"pos_p05_{n}"] = float(np.percentile(u[:, i], 5))
        out[f"pos_p50_{n}"] = float(np.median(u[:, i]))
        out[f"pos_p95_{n}"] = float(np.percentile(u[:, i], 95))
        out[f"speed_{n}"] = float(np.sqrt(np.mean(v[:, i] ** 2)))
        out[f"acc_{n}"] = float(np.sqrt(np.mean(a[:, i] ** 2)))
    out["speed_norm"] = float(np.sqrt(np.mean((v ** 2).sum(1))))
    out["acc_norm"] = float(np.sqrt(np.mean((a ** 2).sum(1))))
    steps = np.linalg.norm(np.diff(u, axis=0), axis=1)            # steps[k] = |u[k+1] − u[k]|
    idx = [w.t0 - 1 for w in W.chunk_windows(ep) if 0 < w.t0 < ep.T]
    jump = float(np.mean(steps[idx]))
    inner = float(np.median(np.delete(steps, idx)))
    out["boundary_jump"] = jump
    out["boundary_jump_ratio"] = jump / max(inner, 1e-12)
    return out


def ood_rate(held, train, fn, lo: float = 1.0, hi: float = 99.0) -> float:
    """held 的逐帧特征中，有任一维落在 train 的 [lo, hi] 百分位之外的帧所占比例。"""
    Xt = np.concatenate([fn(e) for e in train])
    a, b = np.percentile(Xt, lo, axis=0), np.percentile(Xt, hi, axis=0)
    Xh = np.concatenate([fn(e) for e in held])
    return float(np.mean(((Xh < a) | (Xh > b)).any(axis=1)))


# ── 检查二：gap、IMU 旋转、关节跟踪 ──────────────────────────────────────
GYRO_MIN = 0.15                           # rad/s，同 imu_frame_calib.calibrate 的 gyro_min


def kabsch_R(A, B) -> np.ndarray:
    """`A @ Δ ≈ B` 的最优真旋转 Δ（det = +1；行向量约定）。A、B：(N, 3)。"""
    U, _, Vt = np.linalg.svd(np.asarray(A).T @ np.asarray(B))
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def angle_deg(Rd) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))))


def kabsch(A, B) -> tuple[float, float]:
    """→ (Δ 的旋转角 deg, 旋转后的最小二乘增益 g)。"""
    Rd = kabsch_R(A, B)
    AR = np.asarray(A) @ Rd
    return angle_deg(Rd), float((AR * B).sum() / (AR * AR).sum())


def rotation_calib(eps):
    """仓库惯例的逐会话标定：未滤波陀螺，|理想| > GYRO_MIN 的样本，ω_real[t+3] ≈ ideal[t] @ Δ。
    ideal 已按全局 R 旋到传感器系 ⇒ Δ 是相对全局 R 的残余旋转。→ (角 deg, 增益, Δ, 样本数)。"""
    A, B = [], []
    for e in eps:
        a, b = e.ideal[:e.n], e.w[D.ALIGN_L:D.ALIGN_L + e.n]
        m = np.linalg.norm(a, axis=1) > GYRO_MIN
        A.append(a[m]); B.append(b[m])
    A, B = np.concatenate(A), np.concatenate(B)
    Rd = kabsch_R(A, B)
    AR = A @ Rd
    return angle_deg(Rd), float((AR * B).sum() / (AR * AR).sum()), Rd, len(A)


def rotation_share(eps, Rd, K) -> float:
    """纯旋转误差 `ideal @ (Δ − I)` 的带通能量占带通 gap 能量的比例（旋转能解释的上限）。"""
    num = sum(float((W.P(e.ideal[:e.n] @ (np.asarray(Rd) - np.eye(3)), K) ** 2).sum()) for e in eps)
    den = sum(float((W.P(e.gap, K) ** 2).sum()) for e in eps)
    return num / den


def _band(x, lo, hi) -> np.ndarray:
    from scipy.signal import firwin
    k = firwin(41, [lo, hi], fs=D.FS, pass_zero=False)
    x = np.asarray(x, np.float64) - np.mean(x, axis=0, keepdims=True)
    y = np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)
    return y[D.EDGE:-D.EDGE]


def gap_stats(eps, K) -> dict:
    """会话池化：墙/理想、实测/理想（带通能量比），以及 gap 在三个分频段的能量占比。"""
    g = i = r = 0.0
    bands = np.zeros(len(SUBBANDS))
    for e in eps:
        g += float((W.P(e.gap, K) ** 2).sum())
        i += float((W.P(e.ideal[:e.n], K) ** 2).sum())
        r += float((W.P(e.w[D.ALIGN_L:D.ALIGN_L + e.n], K) ** 2).sum())
        for b, (lo, hi) in enumerate(SUBBANDS):
            bands[b] += float((_band(e.gap, lo, hi) ** 2).sum())
    out = {"wall_rel": g / i, "real_rel": r / i}
    for (lo, hi), s in zip(SUBBANDS, bands / bands.sum()):
        out[f"gap_share_{lo:g}-{hi:g}Hz"] = float(s)
    return out


def tracking_fit(eps, j: int, dmax: int = 10) -> tuple[int, float, float]:
    """关节 j：编码器速度 Δq[k] 与指令速度 Δu[k−d] 的池化互相关，d = 0…dmax 取峰值
    → (延迟 d 帧, 幅值比 a = Σ Δq·Δu / Σ Δu², 相关系数 r)。只用 q[k]、q[k−1] 都有效的帧。"""
    best = None
    for d in range(dmax + 1):
        X, Y = [], []
        for e in eps:
            k = np.arange(max(D.Q_VALID_FROM + 1, d + 1), e.T)
            k = k[e.q_valid[k] & e.q_valid[k - 1]]
            X.append(e.cmd[k - d, j] - e.cmd[k - d - 1, j])
            Y.append(e.q[k, j] - e.q[k - 1, j])
        X, Y = np.concatenate(X), np.concatenate(Y)
        r = float(np.corrcoef(X, Y)[0, 1])
        if best is None or r > best[2]:
            best = (d, float(X @ Y / (X @ X)), r)
    return best


# ── 汇总 ──────────────────────────────────────────────────────────────────
def separated(old: dict, recent: dict) -> bool:
    """两个旧会话的值都落在近期三会话的 [min, max] 之外（哪一侧都算）。"""
    lo, hi = min(recent.values()), max(recent.values())
    return all(v < lo or v > hi for v in old.values())


def same_side(old: dict, recent: dict) -> bool:
    lo, hi = min(recent.values()), max(recent.values())
    return all(v > hi for v in old.values()) or all(v < lo for v in old.values())


def _row(name, vals: dict, fmt) -> str:
    o = {s: vals[s] for s in OLD}
    r = {s: vals[s] for s in RECENT}
    flag = ("是（同侧）" if same_side(o, r) else "是（两侧）") if separated(o, r) else "否"
    return f"| {name} | " + " | ".join(fmt(vals[s]) for s in ORDER) + f" | {flag} |"


def main() -> int:
    gc.disable()
    eps = D.load_bare()
    D.check_roster(eps)
    by = D.by_session(eps)
    K = W.kernel()
    head = "| 统计 | " + " | ".join(f"{s} {DATE[s]}" for s in ORDER) + " | 两个旧会话都在近期范围外 |"
    sep = "|" + "---|" * (len(ORDER) + 2)
    res = {"cmd": {}, "ood": {}, "gap": {}, "rotation": {}, "tracking": {}}

    # 检查一
    per = {s: [cmd_stats(e) for e in by[s]] for s in ORDER}
    med = {s: {k: float(np.median([x[k] for x in per[s]])) for k in per[s][0]} for s in ORDER}
    res["cmd"] = med
    print("## 检查一：ACT 指令分布（各会话逐集中位数）\n\n" + head + "\n" + sep)
    print(_row("速度 RMS（关节空间范数, rad/s）", {s: med[s]["speed_norm"] for s in ORDER}, lambda v: f"{v:.3f}"))
    print(_row("加速度 RMS（rad/s²）", {s: med[s]["acc_norm"] for s in ORDER}, lambda v: f"{v:.2f}"))
    print(_row("chunk 边界跳变（rad）", {s: med[s]["boundary_jump"] for s in ORDER}, lambda v: f"{v:.4f}"))
    print(_row("边界跳变 / 块内步长中位", {s: med[s]["boundary_jump_ratio"] for s in ORDER}, lambda v: f"{v:.1f}"))
    for j in JOINTS:
        n = JNAME[j]
        print(_row(f"{n} 位置中位（rad）", {s: med[s][f"pos_p50_{n}"] for s in ORDER}, lambda v: f"{v:+.3f}"))
        print(_row(f"{n} 位置范围 p95−p05（rad）",
                   {s: med[s][f"pos_p95_{n}"] - med[s][f"pos_p05_{n}"] for s in ORDER}, lambda v: f"{v:.3f}"))
        print(_row(f"{n} 速度 RMS（rad/s）", {s: med[s][f"speed_{n}"] for s in ORDER}, lambda v: f"{v:.3f}"))

    print("\n### 越界率：留出会话的指令落在其余四个训练会话 1–99 百分位之外的帧（任一关节）\n")
    print("| 留出会话 | 位置越界率 | 速度越界率 | 参照：训练会话自身 位置 / 速度 |\n|---|---|---|---|")
    for s in ORDER:
        tr = [e for e in eps if e.sess != s]
        o = {"pos": ood_rate(by[s], tr, positions), "speed": ood_rate(by[s], tr, speeds),
             "pos_ref": ood_rate(tr, tr, positions), "speed_ref": ood_rate(tr, tr, speeds)}
        res["ood"][s] = o
        print(f"| {s} {DATE[s]} | {o['pos'] * 100:.1f}% | {o['speed'] * 100:.1f}% | "
              f"{o['pos_ref'] * 100:.1f}% / {o['speed_ref'] * 100:.1f}% |")

    # 检查二
    gs = {s: gap_stats(by[s], K) for s in ORDER}
    rot = {s: rotation_calib(by[s]) for s in ORDER}
    share = {s: rotation_share(by[s], rot[s][2], K) for s in ORDER}
    res["gap"] = gs
    res["rotation"] = {s: {"angle_deg": a, "gain": g, "delta": Rd.tolist(), "n": n, "gap_share": share[s]}
                       for s, (a, g, Rd, n) in rot.items()}
    print("\n## 检查二：gap 与传感器 / 动力学\n\n" + head + "\n" + sep)
    print(_row("墙 / 理想（带通能量比）", {s: gs[s]["wall_rel"] for s in ORDER}, lambda v: f"{v:.3f}"))
    print(_row("实测 / 理想（带通能量比）", {s: gs[s]["real_rel"] for s in ORDER}, lambda v: f"{v:.3f}"))
    for lo, hi in SUBBANDS:
        k = f"gap_share_{lo:g}-{hi:g}Hz"
        print(_row(f"gap 能量占比 {lo:g}–{hi:g} Hz", {s: gs[s][k] for s in ORDER}, lambda v: f"{v * 100:.1f}%"))
    print(_row("IMU 残余旋转（相对全局 R，°）", {s: rot[s][0] for s in ORDER}, lambda v: f"{v:.2f}"))
    print(_row("标定增益（应约为 1）", {s: rot[s][1] for s in ORDER}, lambda v: f"{v:.3f}"))
    print(_row("纯旋转误差占带通 gap 能量（上限）", share, lambda v: f"{v * 100:.2f}%"))
    for j in (0, 1, 2, 4):
        n = JNAME[j]
        fits = {s: tracking_fit(by[s], j) for s in ORDER}
        res["tracking"] = res.get("tracking", {})
        res["tracking"][n] = {s: {"delay": d, "amp": a, "r": r} for s, (d, a, r) in fits.items()}
        print(_row(f"{n} 跟踪：延迟（帧）", {s: fits[s][0] for s in ORDER}, lambda v: f"{v:d}"))
        print(_row(f"{n} 跟踪：幅值比", {s: fits[s][1] for s in ORDER}, lambda v: f"{v:.3f}"))
        print(_row(f"{n} 跟踪：相关系数", {s: fits[s][2] for s in ORDER}, lambda v: f"{v:.3f}"))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag_q1_era.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"\n→ {OUT / 'diag_q1_era.json'}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
