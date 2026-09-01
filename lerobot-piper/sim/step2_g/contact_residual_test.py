"""**接触假设的零机时检验** —— pp2 的残差是不是集中在抓取/放置时刻？

`piper_model.xml` 里 `contype/conaffinity=0`，且**没有被操作物体** ⇒ 接触/冲击
**完全未建模**。而 pp2 的残差 74% 在 3 Hz 以上（激励套件是 71% 在 3 Hz 以下）——
这个不对称指向一个 pp2 独有的宽带高频源。

判据：以夹爪开合时刻 ±0.5 s 为接触窗。
**富集倍数 = 窗占残差能量 / 窗占帧数**。>> 1 且窗内 r/m 明显高于窗外 ⇒ 接触被指向。

结果（17/17 集一致）：富集 **2.44×**、窗内 r/m **0.778** vs 窗外 **0.494**、
真机在窗内的实测能量密度也高 **2.02×**（说明冲击是真实存在的）。
**但天花板只有 12.5%** —— 见 `RESULT_OPTUNA_IDENT_ENC.md` §20。

用法:
    python sim/step2_g/contact_residual_test.py
"""
import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

def main():
    from sim.step1_v2_loss import bandpass41, load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import bare_logs
    import g_q_v1 as GQ
    R = load_calib(CALIB)[0]; L = 1; W = 10   # ±10 帧 = ±0.5 s
    tot_in = tot_out = mea_in = mea_out = 0.0
    nf_in = nf_tot = 0
    per = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        g = GQ.predict_gyro(act)
        k = min(len(g), len(real) - L)
        a_ = bandpass41(real[L:L+k]); b_ = bandpass41(g[:k] @ R.T)
        res = (a_ - b_) ** 2; mea = a_ ** 2
        grip = act[:k, 6]
        d = np.abs(np.diff(grip, prepend=grip[:1]))
        ev = np.where(d > 0.5 * max(d.max(), 1e-9))[0]      # 夹爪开合时刻
        m = np.zeros(k, bool)
        for e in ev:
            m[max(0, e-W):min(k, e+W+1)] = True
        tot_in += res[m].sum(); tot_out += res[~m].sum()
        mea_in += mea[m].sum(); mea_out += mea[~m].sum()
        nf_in += int(m.sum()); nf_tot += k
        per.append((p.stem, len(ev), m.mean(),
                    res[m].sum()/max(res.sum(),1e-30),
                    (res[m].sum()/max(mea[m].sum(),1e-30)) if m.any() else np.nan,
                    res[~m].sum()/max(mea[~m].sum(),1e-30)))
    print("**接触假设的零机时检验**（夹爪开合 ±0.5 s 为接触窗）\n")
    print(f"  {'集':22s}{'事件':>5s}{'窗占帧':>8s}{'窗占残差':>9s}{'窗内 r/m':>10s}{'窗外 r/m':>10s}")
    for r in per:
        print(f"  {r[0]:22s}{r[1]:>5d}{r[2]*100:>7.1f}%{r[3]*100:>8.1f}%"
              f"{r[4]:>10.3f}{r[5]:>10.3f}")
    fr_frames = nf_in / nf_tot
    fr_res = tot_in / (tot_in + tot_out)
    print(f"\n  **汇总（17 集合并）**")
    print(f"    接触窗占帧数        **{fr_frames*100:.1f}%**")
    print(f"    接触窗占残差能量    **{fr_res*100:.1f}%**   富集倍数 **{fr_res/max(fr_frames,1e-9):.2f}×**")
    print(f"    窗内 残差/实测      **{tot_in/max(mea_in,1e-30):.3f}**")
    print(f"    窗外 残差/实测      **{tot_out/max(mea_out,1e-30):.3f}**")
    print(f"    窗内/窗外 实测能量密度比 "
          f"**{(mea_in/max(nf_in,1))/max(mea_out/max(nf_tot-nf_in,1),1e-30):.2f}×**")
    print(f"\n  判读: 富集倍数 >> 1 且窗内 r/m 明显高于窗外 ⇒ 接触被实证指向;")
    print(f"        富集倍数 ≈ 1 ⇒ 残差是全程弥散的, 接触不是主因。")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
