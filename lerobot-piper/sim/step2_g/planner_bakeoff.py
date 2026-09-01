#!/usr/bin/env python3
"""**规划器结构比较**（固定预算）—— 3–9 Hz 的杠杆在规划器，不在伺服。

## 诊断依据（`RESULT_OPTUNA_IDENT_ENC.md` §28）

pp2 17 集，只算该集行程 >1° 的关节，关节角残差/实测（括号=相干 γ²）：

| 关节轨迹来源 | 0.5–3 Hz | **3–9 Hz** |
|---|---|---|
| ① ACT 原始指令 | 0.0170 (0.74) | **1.5174 (0.30)** |
| ② 限幅器输出 `ref` | 0.0089 (0.84) | **0.6849 (0.41)** |
| ③ G-q-v1 仿真 `q` | 0.0038 (0.85) | **0.3796 (0.42)** |

- **限幅器（规划器）承担的降幅（1.52→0.68）比伺服（0.68→0.38）还大。**
- `sim` 的 0.3796 **已低于从原始指令出发的线性预测界 `1−γ² = 0.70`**
  ⇒ 现有的非线性处理已在起作用，**这条路没被线性界封死**。
- 而现行限幅器是**在大幅值阶跃上拟合的、每关节一个参数的 bang-bang**，
  pp2 却是 **20 Hz 连续小增量** —— 工况完全不同。

⇒ **换规划器结构，伺服 θ\* 冻结不动。**

## 六个臂（伺服 = G-q-v1 的 θ\*，逐字冻结）

| 臂 | 结构 | 新增自由量 | 前科 |
|---|---|---|---|
| P0 | 现行纯加速度限幅（bang-bang） | 0 | 基线 |
| P1 | + **速度限** `v_max = m_v · p99(|q̇_enc|)` | 1（全局 `m_v`） | 未试过 |
| P2 | + **jerk 限**（S 曲线）`j_max = m_j · a_max · FPS` | 1（全局 `m_j`） | 阶跃上输给纯加速度限（§7） |
| P3 | + **纯传输延迟** `d` | 1（全局） | **曾判否**（"ad" 模式，0.5198→0.8140）——但那是**旧伺服**下 |
| P4 | **流式同步**（跟踪移动目标） | 0 | **曾判否**（G-v3）——但那是**旧伺服 + IMU 四区域目标**下 |
| P5 | + **一阶滞后** `tau`（两段式） | 1（全局） | 未在此口径下试过 |

**P3/P4 的前科如实标注**：两者都是在 G-q-v1 之前、用不同伺服参数与不同目标判的否。
本次是**编码器域 + 已辨识伺服**，是不同的测试，**不是重跑**。

## 预算（固定，不追加）

5 个带参臂 × **8 点网格** + 2 个零参臂 = **42 次评估**（训练 62 条）。

## 选参（只看训练集）

**约束式**：在 `L_q ≤ 1.05 × P0 基线` 的网格点中，取 **3–9 Hz 关节角残差**最小者。
（这次**不拿角度换高频** —— 上一轮速度目标的失败正是因为角度代价过大。）

## 门禁（预注册，跑前定死）

| # | 条件 | 阈值 |
|---|---|---|
| 1 | pp2 全带 IMU 改善 | **≥ 10%** |
| 2 | pp2 逐集改善 / 中位 | **≥ 9/17** 且 > 0 |
| 3 | pp2 关节角 RMS 不恶化 | **≤ +5%** |
| 4 | 套件留出 A、留出 B 的 IMU 不恶化 | ≤ +5% |
| 5 | **套件留出 A 的 3–9 Hz 关节残差改善** | **≥ 10%**（机制必须真的起作用，且跨位形） |
| 6 | 胜出臂相对次优臂（pp2 全带 IMU） | ≥ 5%，否则记「两者都不足以判定」 |

**若无臂通过 ⇒ 记录「规划器结构层亦已到顶」**：
§27 的 43% 可达空间不能由 {常系数伺服参数, 规划器结构} 取得。

改完 G 必出 pp2 波形图（常规；仅在有臂通过时触发）。

用法:
    python sim/step2_g/planner_bakeoff.py --run
"""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(_p))

FPS = 20.0
OUT = ROOT / "outputs/step2_g/planner"
GRID = 8


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def plan_av(cmd, a_max, v_max, dts, sub=20):
    """加速度限 + **速度限**。"""
    q, v, out = float(cmd[0]), 0.0, []
    for tgt, T in zip(cmd, dts):
        dt = T / sub
        for _ in range(sub):
            e = float(tgt) - q
            s = e - v * abs(v) / (2.0 * a_max)
            if abs(s) < a_max * dt * dt and abs(v) < a_max * dt:
                q, v = float(tgt), 0.0
            else:
                v += (a_max if s > 0 else -a_max) * dt
                v = _clip(v, -v_max, v_max)
                q += v * dt
        out.append(q)
    return np.asarray(out)


def plan_aj(cmd, a_max, j_max, dts, sub=20):
    """加速度限 + **jerk 限**（S 曲线）。"""
    q, v, a, out = float(cmd[0]), 0.0, 0.0, []
    for tgt, T in zip(cmd, dts):
        dt = T / sub
        for _ in range(sub):
            e = float(tgt) - q
            s = e - v * abs(v) / (2.0 * a_max)
            if abs(s) < a_max * dt * dt and abs(v) < a_max * dt:
                q, v, a = float(tgt), 0.0, 0.0
            else:
                ad = a_max if s > 0 else -a_max
                a += _clip(ad - a, -j_max * dt, j_max * dt)
                v += a * dt
                q += v * dt
        out.append(q)
    return np.asarray(out)


def make_ref(cmd_deg, kind, par, a_max, vp99, dts):
    """→ 参考轨迹（度，(T,6)）。`cmd_deg` 已是度。"""
    from optuna_ident_enc import accel_limited_dt
    from sim.step2_g.accel_limit import accel_limited_stream, pure_delay, lag1
    n, ref = len(cmd_deg), np.empty_like(cmd_deg)
    for j in range(6):
        c, am = cmd_deg[:, j], float(a_max[j])
        if kind == "P0":
            ref[:, j] = accel_limited_dt(c, am, dts)
        elif kind == "P1":
            ref[:, j] = plan_av(c, am, max(par * vp99[j], 1e-6), dts)
        elif kind == "P2":
            ref[:, j] = plan_aj(c, am, max(par * am * FPS, 1e-6), dts)
        elif kind == "P3":
            ref[:, j] = pure_delay(accel_limited_dt(c, am, dts), par)
        elif kind == "P4":
            ref[:, j] = accel_limited_stream(c, am)
        elif kind == "P5":
            ref[:, j] = lag1(accel_limited_dt(c, am, dts), par)
        else:
            raise ValueError(kind)
    return ref


def bp(x, lo, hi):
    from scipy.signal import firwin
    k = firwin(41, [lo, hi], fs=FPS, pass_zero=False)
    x = np.asarray(x, float)
    return np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], 1)


def roll_ref(m, ref_rad, enc0, g0, send_step, read_step):
    """事件精确回放（与 g_q_v1 同惯例），**参考轨迹由外部给定**。"""
    import mujoco
    from scipy.signal import butter, filtfilt
    dt = m.opt.timestep
    d = mujoco.MjData(m)
    d.qpos[:6] = np.radians(enc0)
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = g0, -g0
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = ref_rad[0]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = g0, -g0
    for _ in range(int(round(0.3 / dt))):
        mujoco.mj_step(m, d)
    n = len(ref_rad); ntot = int(read_step[-1]) + 1
    hi = np.empty((ntot, 3)); q = np.empty((n, 6)); si = ri = 0
    for s in range(ntot):
        while si < n and send_step[si] <= s:
            d.ctrl[:6] = ref_rad[si]; si += 1
        while ri < n and read_step[ri] <= s:
            q[ri] = np.degrees(d.qpos[:6]); ri += 1
        hi[s] = d.sensordata[3:6]
        mujoco.mj_step(m, d)
    while ri < n:
        q[ri] = np.degrees(d.qpos[:6]); ri += 1
    ba = butter(4, 0.45 * FPS / ((1.0 / dt) / 2.0))
    return q, filtfilt(*ba, hi, axis=0)[np.clip(read_step, 0, ntot - 1)]


_G = {}


def _init():
    import gc as _gc, mujoco
    _gc.disable()
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB, A_MAX
    from sim.step2_gap_baseline import MODEL, bare_logs
    from optuna_ident_enc import split, prep
    from g_q_v1 import THETA_STAR
    import params as P
    m = mujoco.MjModel.from_xml_path(str(MODEL)); P.apply(m, THETA_STAR)
    _G["m"] = m; _G["R"] = load_calib(CALIB)[0]; _G["A"] = np.asarray(A_MAX, float)
    tr, ha, hb = split()
    _G["suite"] = {"train": prep(tr, m.opt.timestep), "hA": prep(ha, m.opt.timestep),
                   "hB": prep(hb, m.opt.timestep)}
    pp = []
    for p in bare_logs():
        fr = json.loads(p.read_text())["frames"]
        act = np.asarray([f["a"] for f in fr], float)
        enc = np.asarray([f["state"] for f in fr], float)[:, :6] / 1000.0
        real = np.asarray([f["imu"] for f in fr], float)[:, 3:6]
        n = len(act)
        ss = np.rint(np.arange(n) / FPS / m.opt.timestep).astype(int)
        cm = np.degrees(act[:, :6])
        pp.append({"cmd": cm, "enc": enc, "real": real, "send_step": ss,
                   "read_step": ss + 1, "n": n,
                   "g0": float(np.clip(act[0, 6], 0, 0.035)) if act.shape[1] > 6 else 0.0,
                   "travel": cm.max(0) - cm.min(0)})
    _G["pp2"] = pp
    # 速度 p99（供 P1 用），由**编码器**给出
    v = []
    for it in _G["suite"]["train"]:
        v.append(np.abs(np.diff(it["enc"], axis=0)) * FPS)
    _G["vp99"] = np.percentile(np.vstack(v), 99, axis=0)


def _dts_of(it, dt):
    ss = np.asarray(it["send_step"])
    d = np.diff(ss) * dt
    return np.append(d, np.median(d) if len(d) else 1.0 / FPS)


def eval_suite(kind, par, key):
    """→ (L_q 全带, 3–9 Hz 关节残差/实测, IMU 残差/实测 等权)。"""
    from sim.step1_v2_loss import bandpass41
    m, R, A, vp = _G["m"], _G["R"], _G["A"], _G["vp99"]
    dt = m.opt.timestep
    pj, hn, hd, im = [], 0.0, 0.0, []
    for it in _G["suite"][key]:
        ref = make_ref(it["cmd"], kind, par, A, vp, _dts_of(it, dt))
        q, g = roll_ref(m, np.radians(ref), it["enc0"], it["g0"],
                        it["send_step"], it["read_step"])
        if not (np.isfinite(q).all() and np.isfinite(g).all()):
            return 9e9, 9e9, 9e9
        k = min(len(q), len(it["enc"]))
        pj.append(float(np.mean([np.sqrt(np.mean((q[:k, j] - it["enc"][:k, j]) ** 2))
                                 / max(it["travel"][j], 1e-9) for j in it["target"]])))
        e = bp(q[:k] - it["enc"][:k], 3.0, 9.0); a_ = bp(it["enc"][:k], 3.0, 9.0)
        for j in it["target"]:
            hn += float((e[:, j] ** 2).sum()); hd += float((a_[:, j] ** 2).sum())
        kk = min(len(g), len(it["abp"]))
        b_ = bandpass41(g[:kk] @ R.T)
        im.append(float(np.square(it["abp"][:kk] - b_).sum()) / max(it["ten"], 1e-30))
    return float(np.mean(pj)), hn / max(hd, 1e-30), float(np.mean(im))


def eval_pp2(kind, par):
    """→ (IMU 全带 能量加权, 逐集数组, 关节角 RMS 度)。"""
    from sim.step1_v2_loss import bandpass41
    m, R, A, vp = _G["m"], _G["R"], _G["A"], _G["vp99"]
    num = den = 0.0; per, jr = [], []
    for e in _G["pp2"]:
        n = e["n"]
        ref = make_ref(e["cmd"], kind, par, A, vp, np.full(n, 1.0 / FPS))
        q, g = roll_ref(m, np.radians(ref), e["enc"][0], e["g0"],
                        e["send_step"], e["read_step"])
        if not (np.isfinite(q).all() and np.isfinite(g).all()):
            return 9e9, np.full(len(_G["pp2"]), 9e9), 9e9
        k = min(len(g), len(e["real"]) - 1, len(e["enc"]))
        a_ = bandpass41(e["real"][1:1 + k]); b_ = bandpass41(g[:k] @ R.T)
        nn = float(np.square(a_ - b_).sum()); dd = float(np.square(a_).sum())
        num += nn; den += dd; per.append(nn / max(dd, 1e-30))
        jr.append(float(np.sqrt(np.mean((q[:k] - e["enc"][:k]) ** 2))))
    return num / max(den, 1e-30), np.array(per), float(np.mean(jr))


def _job(a):
    kind, par = a
    lq, h39, imu = eval_suite(kind, par, "train")
    return kind, par, lq, h39, imu


ARMS = {
    "P0": ("现行纯加速度限幅", [0.0]),
    "P1": ("+ 速度限 m_v", list(np.exp(np.linspace(np.log(0.5), np.log(4.0), GRID)))),
    "P2": ("+ jerk 限 m_j", list(np.exp(np.linspace(np.log(0.5), np.log(20.0), GRID)))),
    "P3": ("+ 纯延迟 d(s)", list(np.linspace(0.005, 0.10, GRID))),
    "P4": ("流式同步", [0.0]),
    "P5": ("+ 一阶滞后 tau(s)", list(np.exp(np.linspace(np.log(0.005), np.log(0.15), GRID)))),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    a_ = ap.parse_args()
    gc.disable()
    if not a_.run:
        print(__doc__.split("用法:")[0]); return 0
    import multiprocessing as mp
    cells = [(k, float(p)) for k, (_, ps) in ARMS.items() for p in ps]
    print(f"**规划器结构比较** —— 伺服 θ* 冻结；{len(cells)} 次评估（固定预算）\n", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a_.workers, initializer=_init) as pool:
        res = pool.map(_job, cells)
    by = {}
    for kind, par, lq, h39, imu in res:
        by.setdefault(kind, []).append((par, lq, h39, imu))
    base = min(by["P0"], key=lambda r: r[1])
    print(f"  {'臂':6s}{'结构':22s}{'参数':>10s}{'训练 L_q':>10s}{'训练 3–9Hz':>12s}"
          f"{'训练 IMU':>10s}")
    sel = {}
    for k in ARMS:
        okc = [r for r in by[k] if r[1] <= base[1] * 1.05]
        pick = min(okc, key=lambda r: r[2]) if okc else min(by[k], key=lambda r: r[2])
        sel[k] = (pick, len(okc), len(by[k]))
        print(f"  {k:6s}{ARMS[k][0]:22s}{pick[0]:>10.4g}{pick[1]:>10.4f}"
              f"{pick[2]:>12.4f}{pick[3]:>10.4f}   合格 {len(okc)}/{len(by[k])}", flush=True)

    _init()
    print(f"\n  ── 选中格在留出集 / pp2 上评估 ──", flush=True)
    print(f"  {'臂':6s}{'留出A 3–9':>11s}{'留出A IMU':>11s}{'留出B IMU':>11s}"
          f"{'pp2 IMU':>10s}{'pp2 关节°':>11s}{'pp2 逐集':>10s}")
    full = {}
    for k in ARMS:
        par = sel[k][0][0]
        _, hA39, hAim = eval_suite(k, par, "hA")
        _, _, hBim = eval_suite(k, par, "hB")
        pim, pper, pjr = eval_pp2(k, par)
        full[k] = (hA39, hAim, hBim, pim, pjr, pper)
        print(f"  {k:6s}{hA39:>11.4f}{hAim:>11.4f}{hBim:>11.4f}{pim:>10.4f}"
              f"{pjr:>11.4f}", flush=True)

    b = full["P0"]
    print(f"\n{'='*88}\n**门禁**（预注册；基线 = P0）")
    passed = []
    for k in ARMS:
        if k == "P0":
            continue
        f = full[k]
        d = (b[5] - f[5]) / b[5]
        c = {1: f[3] <= b[3] * 0.90, 2: int((d > 0).sum()) >= 9 and float(np.median(d)) > 0,
             3: f[4] <= b[4] * 1.05, 4: f[1] <= b[1] * 1.05 and f[2] <= b[2] * 1.05,
             5: f[0] <= b[0] * 0.90}
        if all(c.values()):
            passed.append(k)
        print(f"  {k}  pp2IMU {(f[3]/b[3]-1)*100:+6.1f}%{'✓' if c[1] else '✗'}"
              f" · 逐集 {int((d>0).sum())}/{len(d)}{'✓' if c[2] else '✗'}"
              f" · 关节 {(f[4]/b[4]-1)*100:+6.1f}%{'✓' if c[3] else '✗'}"
              f" · 留出IMU {(f[1]/b[1]-1)*100:+6.1f}/{(f[2]/b[2]-1)*100:+6.1f}%{'✓' if c[4] else '✗'}"
              f" · 留出A 3–9 {(f[0]/b[0]-1)*100:+6.1f}%{'✓' if c[5] else '✗'}")
    print()
    if not passed:
        print("⇒ **无臂通过 ⇒ 记录「规划器结构层亦已到顶」**：§27 的 43% 可达空间"
              "\n   不能由 {常系数伺服参数, 规划器结构} 取得。")
    else:
        print(f"⇒ **通过: {passed}** —— 需再看条件 6（相对次优臂 ≥5%）")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bakeoff.json").write_text(json.dumps(
        {"grid": {k: [[r[0], r[1], r[2], r[3]] for r in v] for k, v in by.items()},
         "selected": {k: list(sel[k][0]) for k in sel},
         "full": {k: [full[k][0], full[k][1], full[k][2], full[k][3], full[k][4]]
                  for k in full},
         "passed": passed}, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'bakeoff.json'}")
    return 0


if __name__ == "__main__":
    _c = main(); sys.stdout.flush(); import os; os._exit(_c)
