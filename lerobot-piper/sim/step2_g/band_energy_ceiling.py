"""**频带能量天花板** —— 3-9 Hz 的 IMU 内容, 编码器侧最多能解释多少？

四个来源在 pp2 17 集上的陀螺频带能量（占实机的比例）:

| 来源 | 0.5-3 Hz | **3-9 Hz** |
|---|---|---|
| 实机 IMU | 1.00 | **1.00** |
| **kin(实测编码器)** | 0.83 | **0.16** |
| kin(G-q-v1 关节) | 0.61 | 0.04 |
| G-q-v1 动力学 | 0.64 | 0.08 |

⇒ **即使喂真实编码器, 运动学渲染也只产生实机 3-9 Hz 陀螺能量的 16%。**
⇒ **真机 84% 的 3-9 Hz 腕部陀螺能量不来自 20 Hz 编码器能看到的关节运动。**
⇒ 任何「改进关节预测」的路线在 3-9 Hz 上的天花板就是 16%。

用法: python sim/step2_g/band_energy_ceiling.py
"""
import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

def main():
    from scipy.signal import firwin
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    from ab_decompose import kin_gyro
    import g_q_v1 as GQ
    FPS = 20.0; R = load_calib(CALIB)[0]
    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)
    acc = {k: {b: 0.0 for b in ((0.5,3.0),(3.0,9.0))} for k in
           ("real", "kin_enc", "kin_gq", "dyn_gq")}
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        q = GQ.predict_qpos(act); dyn = GQ.predict_gyro(act)
        k = min(len(q), len(dyn), len(enc), len(real) - 1)
        S = {"real": real[1:1+k],
             "kin_enc": kin_gyro(enc[:k]) @ R.T,
             "kin_gq":  kin_gyro(q[:k]) @ R.T,
             "dyn_gq":  dyn[:k] @ R.T}
        for nm, v in S.items():
            for b in ((0.5,3.0),(3.0,9.0)):
                acc[nm][b] += float((bp(v, *b)**2).sum())
    print("**陀螺侧的频带能量**（pp2 17 集合计，rad/s 的平方和）\n")
    print(f"  {'来源':22s}{'0.5-3 Hz':>14s}{'占实测':>9s}{'3-9 Hz':>14s}{'占实测':>9s}")
    for nm, lab in (("real","实机 IMU"), ("kin_enc","kin(实测编码器)"),
                    ("kin_gq","kin(G-q-v1 关节)"), ("dyn_gq","G-q-v1 动力学")):
        a, b = acc[nm][(0.5,3.0)], acc[nm][(3.0,9.0)]
        print(f"  {lab:22s}{a:>14.4g}{a/acc['real'][(0.5,3.0)]:>9.2f}"
              f"{b:>14.4g}{b/acc['real'][(3.0,9.0)]:>9.2f}")
    print(f"\n  ⇒ 若 kin(实测编码器) 在 3-9 Hz 也远小于实机 ⇒ **真机的 3-9 Hz 陀螺内容")
    print(f"    根本不来自 20 Hz 编码器能看到的关节运动** —— 那 §19 与关节层不矛盾,")
    print(f"    而且「改关节层」也救不了它。")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
