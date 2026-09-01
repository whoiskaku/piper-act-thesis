import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

BANDS = ((0.5, 3.0), (3.0, 9.0), (0.5, 9.0))
LS = (0, 1, 2)

def main():
    from scipy.signal import firwin
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    from ab_decompose import kin_gyro
    import g_q_v1 as GQ
    FPS = 20.0
    R = load_calib(CALIB)[0]
    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)
    SRC = ("dyn", "kin_gq", "kin_enc")
    acc = {s: {(b, L): [0.0, 0.0] for b in BANDS for L in LS} for s in SRC}
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        q = GQ.predict_qpos(act); dyn = GQ.predict_gyro(act)
        k = min(len(q), len(dyn), len(enc), len(real))
        S = {"dyn": dyn[:k] @ R.T, "kin_gq": kin_gyro(q[:k]) @ R.T,
             "kin_enc": kin_gyro(enc[:k]) @ R.T}
        for nm, v in S.items():
            for L in LS:
                kk = min(k, len(real) - L)
                A = real[L:L + kk]
                for b in BANDS:
                    acc[nm][(b, L)][0] += float((bp(A - v[:kk], *b) ** 2).sum())
                    acc[nm][(b, L)][1] += float((bp(A, *b) ** 2).sum())
    print("**pp2 开发集 17 集：残差/实测 的频带天花板（逐来源取自己最优位移）**\n")
    print(f"  {'来源':26s}{'0.5-3 Hz':>15s}{'3-9 Hz':>14s}{'0.5-9 Hz':>15s}")
    best = {}
    for nm, lab in (("dyn", "G-q-v1（动力学，现状）"), ("kin_gq", "kin(G-q-v1 关节)"),
                    ("kin_enc", "kin(实测编码器)=天花板")):
        row = ""
        for b in BANDS:
            vs = [(acc[nm][(b, L)][0] / max(acc[nm][(b, L)][1], 1e-30), L) for L in LS]
            v, L = min(vs); best[(nm, b)] = v
            row += f"{v:>11.4f}(L{L})"
        print(f"  {lab:26s}{row}")
    print()
    for b, nm in ((BANDS[0], "0.5-3 Hz"), (BANDS[1], "3-9 Hz"), (BANDS[2], "0.5-9 Hz")):
        a0, a1 = best[("dyn", b)], best[("kin_enc", b)]
        print(f"  {nm:9s} 现状 {a0:.4f} → 天花板 {a1:.4f}   "
              f"**可达改善上限 {(1 - a1 / a0) * 100:+.1f}%**")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
