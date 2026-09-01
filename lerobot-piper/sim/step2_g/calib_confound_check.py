import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

def main():
    from scipy.signal import firwin
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    import g_q_v1 as GQ
    FPS = 20.0
    R0 = load_calib(CALIB)[0]
    def bp(x, lo=0.5, hi=9.0):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)
    E = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        g = GQ.predict_gyro(act)                      # 机体系, 未旋
        k = min(len(g), len(real) - 1)
        E.append((p.stem, bp(g[:k]), bp(real[1:1+k])))
    def proc(idx):
        """正交 Procrustes: 求 R' 使 ‖A − S R'ᵀ‖² 最小 (S=sim机体系, A=实测)。"""
        M = np.zeros((3, 3))
        for i in idx:
            _, S, A = E[i]; M += A.T @ S
        U, _, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt))
        return U @ np.diag([1, 1, d]) @ Vt
    def score(Rm, idx):
        n = d = 0.0
        for i in idx:
            _, S, A = E[i]
            n += float(((A - S @ Rm.T)**2).sum()); d += float((A**2).sum())
        return n / d
    ang = lambda A, B: np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2, -1, 1)))
    N = len(E); allidx = list(range(N))
    print(f"**标定混淆的量化**（pp2 开发集 {N} 集，0.5–9 Hz）\n")
    print(f"  沿用 pp2a 的 R:            残差/实测 = **{score(R0, allidx):.4f}**")
    Rf = proc(allidx)
    print(f"  全 {N} 集拟合 R'（**in-sample**）: **{score(Rf, allidx):.4f}**"
          f"   与 pp2a 的 R 相差 **{ang(R0, Rf):.2f}°**")
    # 两折交叉: 半数拟合, 另半数评估
    A_, B_ = allidx[0::2], allidx[1::2]
    ra, rb = proc(A_), proc(B_)
    print(f"\n  **两折交叉（拟合侧不参与评估）**")
    print(f"    折1 拟合 → 折2 评估: {score(R0,B_):.4f} → **{score(ra,B_):.4f}**"
          f"   ({(score(ra,B_)/score(R0,B_)-1)*100:+.1f}%)   旋转 {ang(R0,ra):.2f}°")
    print(f"    折2 拟合 → 折1 评估: {score(R0,A_):.4f} → **{score(rb,A_):.4f}**"
          f"   ({(score(rb,A_)/score(R0,A_)-1)*100:+.1f}%)   旋转 {ang(R0,rb):.2f}°")
    print(f"    两折拟合出的 R' 彼此相差 {ang(ra,rb):.2f}°  (小 ⇒ 是稳定的标定量, 不是过拟合)")
    # 逐带
    print(f"\n  **分频带**（用折1拟合的 R'，在折2 上评估）")
    for lo, hi in ((0.5,3.0),(3.0,9.0)):
        def sc(Rm):
            n=d=0.0
            for i in B_:
                _, S, A = E[i]
                # 重新按子带滤波
                n += float((bp(A - S@Rm.T, lo, hi)**2).sum()); d += float((bp(A, lo, hi)**2).sum())
            return n/max(d,1e-30)
        print(f"    {lo}-{hi} Hz: {sc(R0):.4f} → **{sc(ra):.4f}**  ({(sc(ra)/sc(R0)-1)*100:+.1f}%)")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
