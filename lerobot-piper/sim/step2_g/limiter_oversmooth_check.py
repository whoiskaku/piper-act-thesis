"""**限幅器是否过度平滑** —— 只用编码器证据（研究者定的前置条件）。

研究者的规则：「限幅器只在**编码器证明确实过度平滑**时再改。」本模块就是查这件事。

两个检验（零机时）：
1. pp2 上**真机实测**的关节加速度（编码器二阶差分）vs 冻结的 `A_MAX`
   —— `A_MAX` 是在**激励套件的大幅值阶跃**上拟合的，pp2 是不同的工作区间。
2. 松开 `A_MAX` 后 pp2 的**关节 RMS**（只看编码器，不看 IMU）。

结论（2026-08-30）：**两项都判否**。
实测 p95 126–270、p99 124–462 °/s²，超 `A_MAX` 的帧只占 0.01–1.78%；
松开 `A_MAX` 单调变差（×1.5 +10.6% / ×2 +18.8% / ×3 +28.3% / 无限幅 +45.1%），
`×0.7` 与 `×1` 基本相同 ⇒ **`×1` 处于平坦最优，若有偏也是偏松而非偏紧。**
⇒ **前置条件不成立，不改限幅器。**

用法:
    python sim/step2_g/limiter_oversmooth_check.py
"""
import sys, gc, json, faulthandler
faulthandler.enable(); gc.disable()
for p in ('.','sim','sim/step2_g'): sys.path.insert(0,p)
import numpy as np

def main():
    from sim.step2_fidelity import A_MAX
    from sim.step2_gap_baseline import bare_logs
    FPS = 20.0
    eps = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        eps.append((p.stem, np.asarray([f["a"] for f in fr], float),
                    np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0))
    # ── 检验 1: 真机实际做到的关节加速度 vs A_MAX ──
    print("**检验 1: pp2 上真机实测的关节加速度 vs 冻结的 A_MAX**")
    print("  (编码器二阶差分, 度/s²; A_MAX 是在**激励套件的大幅值阶跃**上拟合的)\n")
    print(f"  {'关节':5s}{'A_MAX':>8s}{'实测 p50':>10s}{'p95':>9s}{'p99':>9s}"
          f"{'max':>9s}{'超 A_MAX 帧占比':>16s}")
    over = []
    for j in range(6):
        acc = []
        for nm, a, e in eps:
            q = e[:, j]
            acc.append(np.diff(q, 2) * FPS * FPS)
        acc = np.abs(np.concatenate(acc))
        f = float((acc > A_MAX[j]).mean())
        over.append(f)
        print(f"  j{j+1:<4d}{A_MAX[j]:>8.0f}{np.percentile(acc,50):>10.1f}"
              f"{np.percentile(acc,95):>9.1f}{np.percentile(acc,99):>9.1f}"
              f"{acc.max():>9.1f}{f*100:>15.2f}%")
    print(f"\n  ⇒ 若 p95/p99 明显低于 A_MAX 且超限帧极少 ⇒ 限幅器**没有**过度平滑;")
    print(f"    若实测经常超过 A_MAX ⇒ 限幅器切多了。")

    # ── 检验 2: 松开 A_MAX 后编码器侧拟合会不会变好 ──
    print(f"\n**检验 2: 松开 A_MAX 后 pp2 的关节 RMS（只看编码器，不看 IMU）**\n")
    import mujoco, params as P
    from optuna_ident_enc import accel_limited_dt
    from g_q_v1 import THETA_STAR as TH
    from sim.step2_gap_baseline import MODEL
    from scipy.signal import butter, filtfilt
    m = mujoco.MjModel.from_xml_path(str(MODEL)); P.apply(m, TH)
    dt = m.opt.timestep
    def run(a, mul):
        n = len(a); cmd = np.degrees(a[:, :6])
        ref = np.empty((n, 6))
        for j in range(6):
            ref[:, j] = (cmd[:, j] if mul is None else
                         accel_limited_dt(cmd[:, j], float(A_MAX[j])*mul, np.full(n, 1/FPS)))
        ref = np.radians(ref)
        d = mujoco.MjData(m); d.qpos[:6] = a[0, :6]
        g0 = float(np.clip(a[0, 6], 0, 0.035)) if a.shape[1] > 6 else 0.0
        if m.nq > 7: d.qpos[6], d.qpos[7] = g0, -g0
        mujoco.mj_forward(m, d); d.ctrl[:6] = ref[0]
        if m.nu > 7: d.ctrl[6], d.ctrl[7] = g0, -g0
        for _ in range(int(round(0.3/dt))): mujoco.mj_step(m, d)
        send = np.rint(np.arange(n)/FPS/dt).astype(int); read = send+1
        ntot = int(read[-1])+1; q = np.empty((n,6)); si=ri=0
        for s in range(ntot):
            while si < n and send[si] <= s: d.ctrl[:6] = ref[si]; si += 1
            while ri < n and read[ri] <= s: q[ri] = np.degrees(d.qpos[:6]); ri += 1
            mujoco.mj_step(m, d)
        while ri < n: q[ri] = np.degrees(d.qpos[:6]); ri += 1
        return q
    print(f"  {'A_MAX 倍数':>12s}{'pp2 关节 RMS (度)':>19s}{'vs ×1':>10s}")
    base = None
    for mul in (0.7, 1.0, 1.5, 2.0, 3.0, None):
        rs = []
        for nm, a, e in eps:
            q = run(a, mul); k = min(len(q), len(e))
            rs.append(float(np.sqrt(np.mean((q[:k]-e[:k])**2))))
        v = float(np.mean(rs))
        if mul == 1.0: base = v
        lab = "无限幅" if mul is None else f"×{mul:g}"
        print(f"  {lab:>12s}{v:>19.4f}"
              f"{('' if base is None else f'{(v/base-1)*100:+9.1f}%')}", flush=True)
    print(f"\n  ⇒ 若 ×1 就是最优 ⇒ 限幅器没有过度平滑; 若更松更好 ⇒ 切多了。")

if __name__ == "__main__":
    main(); sys.stdout.flush(); import os; os._exit(0)
