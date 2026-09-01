"""**关节层的频带分解** —— G-q-v1 的关节轨迹在 3-9 Hz 缺不缺？

结论: **不缺能量**。3-9 Hz 的「模型/实测」约 1.00, 残差/实测 0.38, 相干 γ² 约 0.45
⇒ 幅度对、相位/细节不对。且 **3-9 Hz 只占关节总能量的 0.1%**
（0.5-3 Hz 占 99.9%, 残差/实测约 0.00）⇒ 全带 RMS 对目标频带几乎零权重。

但这**不构成可利用的杠杆** —— 见 `band_energy_ceiling.py`:
即使喂真实编码器, 3-9 Hz 的陀螺天花板也只有 16%。

用法: python sim/step2_g/joint_band_gap.py
"""
import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

def main():
    from scipy.signal import welch, coherence, firwin
    from sim.step2_gap_baseline import bare_logs
    import g_q_v1 as GQ
    FPS = 20.0
    def bp(x, lo, hi):
        k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
        return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)

    E = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        q = GQ.predict_qpos(act)
        k = min(len(q), len(enc))
        E.append((q[:k], enc[:k]))
    print("**关节层的频带分解**（pp2 17 集，G-q-v1 预测关节 vs 真实编码器）\n")
    print(f"  {'关节':5s}{'带':>9s}{'实测能量':>12s}{'模型/实测':>10s}{'残差/实测':>10s}{'相干γ²':>9s}")
    tot = {}
    for j in range(6):
        for lo, hi in ((0.5, 3.0), (3.0, 9.0)):
            num = den = 0.0; cs = []
            for q, e in E:
                a = bp(e, lo, hi)[:, j]; b = bp(q, lo, hi)[:, j]
                den += float((a**2).sum()); num += float(((a-b)**2).sum())
                if len(a) > 256:
                    f_, c_ = coherence(a, b, fs=FPS, nperseg=128)
                    m = (f_ >= lo) & (f_ <= hi); cs.append(float(c_[m].mean()))
            mm = 0.0
            for q, e in E:
                mm += float((bp(q, lo, hi)[:, j]**2).sum())
            tot[(j, lo)] = (den, mm/max(den,1e-30), num/max(den,1e-30),
                            float(np.mean(cs)) if cs else np.nan)
            d = tot[(j, lo)]
            print(f"  j{j+1:<4d}{f'{lo}-{hi}':>9s}{d[0]:>12.4g}{d[1]:>10.2f}"
                  f"{d[2]:>10.2f}{d[3]:>9.2f}")
    print()
    for lo, hi in ((0.5, 3.0), (3.0, 9.0)):
        dsum = sum(tot[(j, lo)][0] for j in range(6))
        nsum = sum(tot[(j, lo)][0]*tot[(j, lo)][2] for j in range(6))
        msum = sum(tot[(j, lo)][0]*tot[(j, lo)][1] for j in range(6))
        print(f"  **{lo}-{hi} Hz 合计**: 模型/实测 **{msum/dsum:.2f}** · "
              f"残差/实测 **{nsum/dsum:.2f}** · 占关节总能量 "
              f"{dsum/sum(tot[(j,l)][0] for j in range(6) for l in (0.5,3.0))*100:.1f}%")
    print(f"\n  ⇒ 若 3-9 Hz 的「模型/实测」远小于 1 ⇒ **关节层在目标频带欠拟合**,")
    print(f"    那是编码器可直接观测的, 且可用带权关节目标重拟 —— 不需换模型类。")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
