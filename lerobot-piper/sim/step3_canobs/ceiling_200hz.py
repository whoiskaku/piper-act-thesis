#!/usr/bin/env python3
"""**200 Hz 关节角 → 腕部陀螺：抗混叠口径统一前后的天花板**。

§28 的 `kin(实测编码器) 3–9 Hz ρ²=0.487` 是这么算的：
    真机 IMU  ← TSND 100 Hz 被 `get_latest()` **无抗混叠**抽到 20 Hz
    关节角    ← `GetArmJointMsgs()` 被 **无抗混叠**抽到 20 Hz
2026-09-01 实测发现两者在总线上分别是 100 Hz 和 **200 Hz**，「20 Hz」纯粹是落盘限制。

为什么不用落盘的 20 Hz 字段做对照
--------------------------------
这两条诊断集的 20 Hz 控制环**本身就是坏的**：帧间隔中位 51.2 ms、p99 84.5 ms、
最大 103 ms（同序列 07-27/28 无 CAN 记录器时 7 条全部 19.91–19.92 Hz、p99 ≤ 55 ms）。
拿它做「旧口径」会把回路抖动算进抗混叠的账上。

但**原始数据是完好的**：
  · TSND 设备端 ms 步长恒为 10、**零丢样**（退化的只是主机到达时刻）
  · CAN 帧是内核时间戳，实测 200.0 Hz
故本脚本全程用**设备/内核时间戳**，两个口径都从同一份干净流生成：
    旧 = 两侧朴素抽到 20 Hz（无抗混叠，均匀网格）
    新 = 两侧先低通再抽（同型抗混叠）
差值 = **抗混叠这一项**单独贡献了多少。这不是换模型，是修口径。

会话标定
--------
新采会话没有配套 IMU 朝向标定，绑带朝向逐会话漂移。故**逐集用正交 Procrustes
拟合最优 3×3 旋转**，两个口径给同等自由度 —— 公平，且不依赖旧会话的 R。

用法:
    python -u sim/step3_canobs/ceiling_200hz.py
"""
from __future__ import annotations
import gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

EPS = ("ep0", "ep1")
FS_I, FS_Q = 100.0, 200.0
D2R = np.pi / 180 / 1000.0
BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))


def kin_gyro_fs(q_rad: np.ndarray, fs: float) -> np.ndarray:
    """`ab_decompose.kin_gyro` 的可变速率版（原版把 FPS=20 写死在中心差分里）。"""
    import mujoco
    from ab_decompose import MODEL
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    d = mujoco.MjData(m)
    q = np.asarray(q_rad, float)
    v = np.zeros_like(q)
    v[1:-1] = (q[2:] - q[:-2]) * fs / 2.0
    v[0] = (q[1] - q[0]) * fs
    v[-1] = (q[-1] - q[-2]) * fs
    out = np.zeros((len(q), 3))
    for t in range(len(q)):
        d.qpos[:6] = q[t]
        d.qvel[:6] = v[t]
        mujoco.mj_forward(m, d)
        out[t] = d.sensordata[3:6]
    return out


def load_streams(nm: str, arm: str = "A1"):
    """→ (gyro100 均匀, q200 均匀[rad], 均匀 100 Hz 网格, 均匀 200 Hz 网格)。

    `arm` 选会话：A1 = 会话1(2026-09-01 22:25)，A3 = 会话2。
    """
    d = json.loads((ROOT / f"outputs/probe/replay_{nm}_{arm}.json").read_text())
    off = d["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    I = np.asarray(d["imu100"], float)
    ms, ht, gy = I[:, 1], I[:, 0], I[:, 5:8]
    # 设备 ms → MONOTONIC。到达延迟恒为正 ⇒ 斜率用最小二乘, 常数项由后面的
    # 位移扫描吸收(逐口径各自扫), 故这里不必求下包络。
    A = np.stack([ms, np.ones_like(ms)], 1)
    a, b = np.linalg.lstsq(A, ht, rcond=None)[0]
    t_i = a * ms + b                                   # IMU 各样本的 MONOTONIC 时刻
    Q = np.asarray(d["can_q200"], float)
    tq, qv = [], []
    for gid in (1, 3, 5):
        m = Q[:, 1] == gid
        tq.append(Q[m, 0] - off)                       # REALTIME → MONOTONIC
        qv.append(Q[m, 2:4])
    t0 = max(t_i[0], *[t[0] for t in tq])
    t1 = min(t_i[-1], *[t[-1] for t in tq])
    tg_q = np.arange(t0, t1, 1.0 / FS_Q)               # 均匀 200 Hz
    tg_i = np.arange(t0, t1, 1.0 / FS_I)               # 均匀 100 Hz
    q200 = np.zeros((len(tg_q), 6))
    for gid, t_, v_ in zip((1, 3, 5), tq, qv):
        for c in (0, 1):
            q200[:, gid - 1 + c] = np.interp(tg_q, t_, v_[:, c]) * D2R
    g100 = np.stack([np.interp(tg_i, t_i, gy[:, k]) for k in range(3)], 1)
    return g100, q200, tg_i, tg_q


def main() -> int:
    gc.disable()
    from scipy.signal import butter, filtfilt, firwin
    from scipy.linalg import orthogonal_procrustes

    def lp(x, fc, fs):
        return filtfilt(*butter(4, fc / (fs / 2)), x, axis=0)

    def bp(x, lo, hi, fs):
        k = firwin(int(2 * fs) | 1, [lo, hi], fs=fs, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same")
                         for j in range(x.shape[1])], 1)

    def score(A, S):
        aa = float((A * A).sum()); ss = float((S * S).sum()); as_ = float((A * S).sum())
        return ((aa + ss - 2 * as_) / max(aa, 1e-30),
                as_ ** 2 / max(aa * ss, 1e-30), as_ / max(ss, 1e-30))

    def evaluate(A_, S_, fs, maxshift):
        """逐集拟合安装旋转 + 扫最优位移 → {band: (残差, ρ², 增益)}。"""
        best = None
        for L in range(0, maxshift + 1):
            k = min(len(A_) - L, len(S_))
            A = A_[L:L + k]; S = S_[:k]
            R, _ = orthogonal_procrustes(S, A)
            Sr = S @ R
            v = sum(score(bp(A, *b, fs), bp(Sr, *b, fs))[0] for b in BANDS)
            if best is None or v < best[0]:
                best = (v, L, A, Sr)
        _, L, A, Sr = best
        return L, {b: score(bp(A, *b, fs), bp(Sr, *b, fs)) for b in BANDS}

    print("200 Hz 关节角 → 腕部陀螺：抗混叠口径统一前后")
    print("（时基全部用设备/内核时间戳；不使用那两条集里坏掉的 20 Hz 落盘字段）\n")
    acc = {k: {b: np.zeros(3) for b in BANDS} for k in ("naive20", "aa20", "aa100")}
    for nm in EPS:
        g100, q200, tg_i, tg_q = load_streams(nm)
        kin200 = kin_gyro_fs(q200, FS_Q)
        # ---- 100 Hz 上限（两侧都不抽取，只把 200→100 抗混叠）----
        kin100 = lp(kin200, 45.0, FS_Q)[::2][:len(g100)]
        # ---- 20 Hz：旧=朴素抽取 / 新=先低通 9 Hz 再抽取 ----
        g20_naive, k20_naive = g100[::5], kin100[::5]
        g20_aa = lp(g100, 9.0, FS_I)[::5]
        k20_aa = lp(kin100, 9.0, FS_I)[::5]
        for key, A_, S_, fs, msh in (
                ("naive20", g20_naive, k20_naive, 20.0, 3),
                ("aa20", g20_aa, k20_aa, 20.0, 3),
                ("aa100", g100, kin100, 100.0, 15)):
            L, r = evaluate(A_, S_, fs, msh)
            for b in BANDS:
                acc[key][b] += np.array(r[b])
            print(f"  {nm} {key:>8}  L={L:>2}  "
                  f"ρ² = {' / '.join(f'{r[b][1]:.3f}' for b in BANDS)}")

    print(f"\n  {'口径':<34}{'0.5–3 Hz':>22}{'3–9 Hz':>22}{'0.5–9 Hz':>22}")
    print(f"  {'':<34}" + "".join(f"{'ρ²':>10}{'残差':>7}{'增益':>5}" for _ in BANDS))
    for key, lbl in (("naive20", "20 Hz 旧口径: 两侧朴素抽取"),
                     ("aa20", "20 Hz 新口径: 两侧抗混叠"),
                     ("aa100", "100 Hz 原生(不抽到 20 Hz)")):
        s = ""
        for b in BANDS:
            r, r2, g = acc[key][b] / len(EPS)
            s += f"{r2:>10.3f}{r:>7.3f}{g:>5.2f}"
        print(f"  {lbl:<34}{s}")
    d = (acc["aa20"][(3.0, 9.0)][1] - acc["naive20"][(3.0, 9.0)][1]) / len(EPS)
    print(f"\n  3–9 Hz 抗混叠单独带来的 ρ² 变化: {d:+.3f}")
    print("  口径: pp2 诊断集 2 条(同一会话 2026-09-01)。**不可与 17 集开发基准集**"
          "的 §28 数字并列 —— 集合不同、会话不同、R 的来源也不同。")
    return 0


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    import os
    os._exit(0)
