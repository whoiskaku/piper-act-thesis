#!/usr/bin/env python3
"""把「指令→关节」之间那个限幅器的**机制**钉死：加速度限 vs jerk 限 vs 二者兼有。

背景: `accel_limit.py` 已证明被控对象非线性(模型无关, 见报告 §2), 并给出
「加速度上限 a=307–391 °/s² + 纯延迟 d=65–90 ms」的拟合。但**峰速标度指数**
无法单独裁决:

    纯加速度限   v_pk = √(aA)                    ⇒ v_pk ∝ A^0.500
    纯 jerk 限   v_pk ∝ j^(1/3) A^(2/3)          ⇒ v_pk ∝ A^0.667
    实测(编码器)                                    0.52–0.67   ← 两者都覆盖
    实测(100Hz 陀螺)                                0.343–0.539 ← 偏向加速度限

⇒ 必须做正面的模型对比拟合。本模块做四件事:

  1. 只用 **r2 档**(指令是**单帧真阶跃**, 已核: 非零 Δ 帧数 = 1)。
     跳变间隔 32–40 帧 = 1.6–2.0 s, 移动约 0.6 s ⇒ 每次跳变都是**静止到静止**,
     正好是经典点到点问题, 可用闭式廓线, 不需要手搓时间最优跟踪器。
  2. 四个假设各自拟合 (参数 + 纯延迟 d), 逐关节。
  3. 判据三条并列: 样本内 RMS、**跨幅值留出**、以及参数个数不同故用 **AIC**。
  4. **决定性判据在 100 Hz 陀螺**: 速度廓线的**形状**。
     加速度限 ⇒ 三角形(上升段斜率恒定); jerk 限 ⇒ S 形(斜率先增后减)。
     位置数据被两次积分平滑, 形状差异被抹掉; 陀螺直接测速度。

用法:
    python sim/step2_g/mechanism.py            # 全部
    python sim/step2_g/mechanism.py --shape    # 只做 100Hz 形状判别
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"
FPS = 20.0
SUB = 400.0            # 廓线积分子步率 (Hz)


# ────────────────────── 闭式/数值 点到点廓线 ──────────────────────
def _integrate(jerk_seq, dt):
    """给定 jerk 时间序列, 返回 (t, q, v, a)。"""
    n = len(jerk_seq)
    a = np.concatenate([[0.0], np.cumsum(jerk_seq) * dt])
    v = np.concatenate([[0.0], np.cumsum(a[:-1] + a[1:]) * dt / 2])
    q = np.concatenate([[0.0], np.cumsum(v[:-1] + v[1:]) * dt / 2])
    return np.arange(n + 1) * dt, q, v, a


def prof_accel(A, a_max):
    """纯加速度限 —— 三角形速度廓线。T = 2√(A/a), v_pk = √(aA)。"""
    A = abs(A)
    T = 2.0 * np.sqrt(A / a_max)
    t = np.arange(0.0, T + 1.0 / SUB, 1.0 / SUB)
    q = np.where(t <= T / 2, 0.5 * a_max * t ** 2,
                 A - 0.5 * a_max * np.clip(T - t, 0, None) ** 2)
    return t, np.clip(q, 0, A)


def prof_jerk(A, j_max):
    """纯 jerk 限(加速度不受限) —— 四段 double-S: jerk = +j, −j, −j, +j。

    a 在 T/4 达峰、T/2 归零、3T/4 反向达峰、T 归零; v 为 S 形。
    标度: A ∝ j·T³ ⇒ T ∝ (A/j)^(1/3), v_pk ∝ j^(1/3) A^(2/3)。
    """
    A = abs(A)
    # 先用单位 T 积分求形状常数, 再解 T
    dt = 1.0 / SUB
    def disp(T):
        n = max(4, int(round(T / dt)))
        q4 = n // 4
        js = np.concatenate([np.full(q4, j_max), np.full(q4, -j_max),
                             np.full(q4, -j_max), np.full(n - 3 * q4, j_max)])
        return _integrate(js, dt)
    lo, hi = 1e-3, 20.0
    for _ in range(60):                       # 对 T 二分
        mid = 0.5 * (lo + hi)
        if disp(mid)[1][-1] < A: lo = mid
        else: hi = mid
    T = 0.5 * (lo + hi)
    t, q, _, _ = disp(T)
    return t, np.clip(q * (A / max(q[-1], 1e-12)), 0, A)


def prof_sj(A, a_max, j_max):
    """加速度**与** jerk 都受限 —— 标准 S 曲线(梯形加速度廓线)。

    小行程时 jerk 主导(∝A^2/3), 大行程时加速度主导(∝A^1/2), 二者的插值。
    """
    A = abs(A)
    dt = 1.0 / SUB
    ta = a_max / j_max                        # 加速度爬升时间
    def disp(tc):                             # tc = 加速度平台时长
        seg = [np.full(max(1, int(round(ta / dt))), j_max),
               np.zeros(max(0, int(round(tc / dt)))),
               np.full(max(1, int(round(ta / dt))), -j_max)]
        half = np.concatenate(seg)
        js = np.concatenate([half, -half])
        return _integrate(js, dt)
    if disp(0.0)[1][-1] >= A:                 # jerk 主导, 加速度未达上限
        return prof_jerk(A, j_max)
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if disp(mid)[1][-1] < A: lo = mid
        else: hi = mid
    t, q, _, _ = disp(0.5 * (lo + hi))
    return t, np.clip(q * (A / max(q[-1], 1e-12)), 0, A)


def prof_va(A, v_max, a_max):
    """速度**与**加速度都受限 —— 梯形速度廓线。(100Hz 已测平台占比 0%, 备查)"""
    A = abs(A)
    dt = 1.0 / SUB
    tr = v_max / a_max
    if a_max * tr * tr >= A:                  # 三角形, 未达速度上限
        return prof_accel(A, a_max)
    tc = (A - a_max * tr * tr) / v_max
    n1, n2 = max(1, int(round(tr / dt))), max(0, int(round(tc / dt)))
    js = None
    a = np.concatenate([np.full(n1, a_max), np.zeros(n2), np.full(n1, -a_max)])
    v = np.concatenate([[0.0], np.cumsum(a) * dt])
    q = np.concatenate([[0.0], np.cumsum(v[:-1] + v[1:]) * dt / 2])
    t = np.arange(len(q)) * dt
    return t, np.clip(q * (A / max(q[-1], 1e-12)), 0, A)


MODELS = {
    "加速度限":      dict(fn=lambda A, p: prof_accel(A, p[0]),          np=1),
    "jerk限":       dict(fn=lambda A, p: prof_jerk(A, p[0]),           np=1),
    "加速度+jerk":  dict(fn=lambda A, p: prof_sj(A, p[0], p[1]),       np=2),
    "速度+加速度":   dict(fn=lambda A, p: prof_va(A, p[0], p[1]),       np=2),
}


# ────────────────────── 数据 ──────────────────────
def transitions(nm, man, only_true_step=True):
    """抽出所有**静止到静止**的跳变。only_true_step ⇒ 只要单帧真阶跃(r2)。"""
    d = json.loads((COL / f"replay_{nm}.json").read_text())
    t = man[nm]["target"][0]
    c = np.degrees(np.array(d["cmd"], float)[:, t])
    s = np.array(d["state"], float)[:, t] / 1000.0
    dc = np.diff(c)
    ed = [i + 1 for i in range(1, len(dc))
          if abs(dc[i]) > 1e-9 and abs(dc[i - 1]) <= 1e-9]
    out = []
    for k, nxt in zip(ed, ed[1:] + [len(c)]):
        e = k
        while e < len(dc) and abs(dc[e]) > 1e-9:
            e += 1
        if only_true_step and (e - k) > 1:
            continue
        A = c[min(e + 1, len(c) - 1)] - c[k - 1]
        if abs(A) < 0.5 or nxt - k < 20:
            continue
        out.append((t, k, A, c[k - 1], s[k - 1:nxt]))
    return out


_PCACHE = {}


def _profile(model, A, p):
    """廓线只依赖 (模型, |A|, 参数) —— 缓存。同一关节的幅值只有 2 个不同值,
    而网格搜索会把同一组合算上千次(prof_sj 每次还要 60 步二分积分)。"""
    key = (model, round(abs(A), 4)) + tuple(round(float(v), 6) for v in p)
    r = _PCACHE.get(key)
    if r is None:
        r = MODELS[model]["fn"](abs(A), p)
        _PCACHE[key] = r
    return r


def predict(model, A, q0, npts, d_s, p):
    """把廓线采到 20 Hz 网格上, 含纯延迟 d_s(亚帧线性插值)。"""
    tp, qp = _profile(model, A, p)
    sg = np.sign(A) if A != 0 else 1.0
    tg = np.arange(npts) / FPS - d_s
    y = np.interp(tg, tp, qp, left=0.0, right=abs(A))
    return q0 + sg * y


# ────────────────────── 拟合与判据 ──────────────────────
def rms_of(model, items, d_s, p, W=26):
    e, n = 0.0, 0
    for t, k, A, q0, s in items:
        m = min(W, len(s))
        y = predict(model, A, q0, m, d_s, p)
        e += float(np.sum((y - s[:m]) ** 2)); n += m
    return float(np.sqrt(e / max(n, 1)))


def fit_model(model, items, grids, dgrid):
    best = (1e18, None, None)
    if MODELS[model]["np"] == 1:
        for g in grids[0]:
            for d in dgrid:
                r = rms_of(model, items, d, (g,))
                if r < best[0]: best = (r, (g,), d)
    else:
        for g1 in grids[0]:
            for g2 in grids[1]:
                for d in dgrid:
                    r = rms_of(model, items, d, (g1, g2))
                    if r < best[0]: best = (r, (g1, g2), d)
    return best


def aic(rms, n, k):
    """高斯残差下的 AIC = n·ln(σ²) + 2k。k 含参数 + 延迟 + σ。"""
    return n * np.log(max(rms, 1e-9) ** 2) + 2 * (k + 2)


def gyro_shape(nm, man, tol=0.85):
    """**决定性判据**: 100 Hz 陀螺的速度廓线形状。

    加速度限 ⇒ 上升段斜率恒定(三角形); jerk 限 ⇒ 斜率先增后减(S 形)。
    位置数据经两次积分已把形状抹平, 只有陀螺能分。
    返回每次跳变的 (线性拟合 R², 前半/后半斜率比, S 形指标)。
    """
    d = json.loads((COL / f"replay_{nm}.json").read_text())
    t = man[nm]["target"][0]
    c = np.degrees(np.array(d["cmd"], float)[:, t])
    i100 = np.array(d["imu100"], float)
    ms = (i100[:, 1] - i100[0, 1]) / 1000.0
    g = i100[:, 5:8]
    en = np.abs(g).sum(1)
    mv = en > np.percentile(en, 75)
    u = np.linalg.svd(g[mv] - g[mv].mean(0), full_matrices=False)[2][0]
    pr = np.degrees(g @ u)
    dc = np.diff(c)
    ed = [i + 1 for i in range(1, len(dc))
          if abs(dc[i]) > 1e-9 and abs(dc[i - 1]) <= 1e-9]
    out = []
    for k, nxt in zip(ed, ed[1:] + [len(c)]):
        e = k
        while e < len(dc) and abs(dc[e]) > 1e-9:
            e += 1
        if (e - k) > 1 or abs(c[e] - c[k - 1]) < 3.0:
            continue
        w = (ms > k / FPS - 0.05) & (ms < nxt / FPS)
        y = pr[w] - np.median(pr[w][-20:])
        if len(y) < 40:
            continue
        s_ = np.sign(y[np.abs(y).argmax()])
        y = y * s_
        pk = y.max()
        i_pk = int(y.argmax())
        # 上升段 10%→90% 峰值
        try:
            i_lo = int(np.where(y[:i_pk] >= .10 * pk)[0][0])
            i_hi = int(np.where(y[:i_pk] >= .90 * pk)[0][0])
        except IndexError:
            continue
        if i_hi - i_lo < 6:
            continue
        seg = y[i_lo:i_hi + 1]
        tt = np.arange(len(seg)) / 100.0
        cf = np.polyfit(tt, seg, 1)
        r2 = 1 - np.sum((seg - np.polyval(cf, tt)) ** 2) / max(
            np.sum((seg - seg.mean()) ** 2), 1e-12)
        h = len(seg) // 2
        s1 = np.polyfit(tt[:h], seg[:h], 1)[0]
        s2 = np.polyfit(tt[h:], seg[h:], 1)[0]
        # S 形指标: 二次项系数 / (峰值/上升时长²) —— 三角形 ≈ 0
        q2 = np.polyfit(tt, seg, 2)[0]
        out.append((r2, s2 / max(s1, 1e-9), q2 * tt[-1] ** 2 / max(pk, 1e-9)))
    return out



def vel_profile(nm, man, min_amp=3.0):
    """抽 100 Hz 陀螺的**整段**速度廓线(0→100%, 不截 10–90%)。

    10–90% 窗口对两种假设**没有分辨力** —— 理论参照: 纯加速度限 R²=1.0000,
    纯 jerk 限 R²=0.9954, 而实测噪声底给 0.92–0.98, 两者都覆盖不了。
    double-S 的 S 形在**首尾**, 10–90% 恰好把它切掉。故必须用整段。
    返回 [(A, t_rel, v_deg_s), ...]。
    """
    d = json.loads((COL / f"replay_{nm}.json").read_text())
    t = man[nm]["target"][0]
    c = np.degrees(np.array(d["cmd"], float)[:, t])
    i100 = np.array(d["imu100"], float)
    ms = (i100[:, 1] - i100[0, 1]) / 1000.0
    g = i100[:, 5:8]
    en = np.abs(g).sum(1)
    u = np.linalg.svd(g[en > np.percentile(en, 75)]
                      - g[en > np.percentile(en, 75)].mean(0),
                      full_matrices=False)[2][0]
    pr = np.degrees(g @ u)
    # **SVD 主方向的符号是任意的**, 逐序列可能翻转。必须用编码器速度定符号,
    # 否则 `y *= sign(A)` 之后取 max 会取到错误的半周(实测把 j2 的峰速比压到
    # 0.878 —— 小幅值峰速反而更高, 物理上不可能)。
    s20 = np.array(d["state"], float)[:, t] / 1000.0
    ve = np.diff(s20) * FPS                                  # 编码器速度 (20 Hz)
    pr20 = np.interp(np.arange(len(ve)) / FPS + 0.5 / FPS, ms, pr)
    if np.dot(pr20 - pr20.mean(), ve - ve.mean()) < 0:
        pr = -pr
    dc = np.diff(c)
    ed = [i + 1 for i in range(1, len(dc))
          if abs(dc[i]) > 1e-9 and abs(dc[i - 1]) <= 1e-9]
    out = []
    for k, nxt in zip(ed, ed[1:] + [len(c)]):
        e = k
        while e < len(dc) and abs(dc[e]) > 1e-9:
            e += 1
        A = c[min(e + 1, len(c) - 1)] - c[k - 1]
        if (e - k) > 1 or abs(A) < min_amp:
            continue
        w = (ms > k / FPS - 0.10) & (ms < nxt / FPS - 0.05)
        if w.sum() < 50:
            continue
        y = pr[w].copy()
        y -= np.median(y[-15:])                     # 去静止偏置
        y *= np.sign(A)
        out.append((abs(A), ms[w] - k / FPS, y))
    return out


def move_duration(tt, v, frac=0.02):
    """移动时长: 速度首次/末次超过峰值 frac 的时刻之差。对形状不敏感。"""
    pk = v.max()
    idx = np.where(v > frac * pk)[0]
    if len(idx) < 5:
        return np.nan, np.nan
    return float(tt[idx[-1]] - tt[idx[0]]), float(pk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="store_true")
    a_ = ap.parse_args()
    man = json.loads((SUITE / "manifest.json").read_text())
    done = {r["name"] for r in json.loads((COL / "state.json").read_text())["done"]}
    steps = [n for n in sorted(done) if man[n]["kind"] == "step"]

    # ── 判据 A: 100 Hz 陀螺速度廓线形状 ──
    print("═" * 74)
    print("判据 A —— 100 Hz 陀螺的速度廓线形状（位置数据分不出，陀螺能分）")
    print("  纯三角形(加速度限): 线性 R²→1, 后/前半斜率比→1, S 形指标→0")
    print("  纯 S 形(jerk 限)  : 线性 R² 明显 <1, 斜率比 ≫1 或 ≪1, |S 形指标| 大")
    print("═" * 74)
    print(f"{'关节':>5s}{'n跳变':>7s}{'线性R²':>9s}{'后/前斜率比':>13s}{'S形指标':>10s}")
    ref = {}
    for j in range(6):
        rows = []
        for nm in steps:
            if man[nm]["target"][0] != j or "_r2_" not in nm:
                continue
            rows += gyro_shape(nm, man)
        if not rows:
            print(f"j{j+1:<4d}{'(无 r2 真阶跃或幅值不足)':>40s}")
            continue
        R = np.array(rows)
        ref[j] = R
        print(f"j{j+1:<4d}{len(R):>7d}{np.median(R[:,0]):>9.4f}"
              f"{np.median(R[:,1]):>13.2f}{np.median(R[:,2]):>10.3f}")

    # 理论参照: 把两种廓线本身做同样的度量
    print("\n  ── 理论参照（对廓线本身做完全相同的度量）──")
    for lab, tp, qp in [("纯加速度限", *prof_accel(16.0, 320.0)),
                        ("纯 jerk 限", *prof_jerk(16.0, 3000.0))]:
        v = np.gradient(qp, tp)
        pk = v.max(); ip = int(v.argmax())
        il = int(np.where(v[:ip] >= .10 * pk)[0][0])
        ih = int(np.where(v[:ip] >= .90 * pk)[0][0])
        seg = v[il:ih + 1]; tt = np.arange(len(seg)) * (tp[1] - tp[0])
        cf = np.polyfit(tt, seg, 1)
        r2 = 1 - np.sum((seg - np.polyval(cf, tt)) ** 2) / np.sum((seg - seg.mean()) ** 2)
        h = len(seg) // 2
        s1 = np.polyfit(tt[:h], seg[:h], 1)[0]; s2 = np.polyfit(tt[h:], seg[h:], 1)[0]
        q2 = np.polyfit(tt, seg, 2)[0]
        print(f"  {lab:12s} 线性R²={r2:.4f}  后/前斜率比={s2/s1:6.2f}  "
              f"S形指标={q2*tt[-1]**2/pk:+.3f}")
    if a_.shape:
        return 0

    # ── 判据 B: 四模型对同一份位置数据的拟合 + 跨幅值留出 + AIC ──
    print("\n" + "═" * 74)
    print("判据 B —— 四个假设拟合同一份 r2 真阶跃位置数据（20 Hz 编码器）")
    print("═" * 74)
    G = {
        "加速度限":     ([np.exp(np.linspace(np.log(120), np.log(1500), 28))], ),
        "jerk限":      ([np.exp(np.linspace(np.log(300), np.log(60000), 28))], ),
        "加速度+jerk": (np.exp(np.linspace(np.log(150), np.log(1500), 14)),
                        np.exp(np.linspace(np.log(600), np.log(60000), 14))),
        "速度+加速度":  (np.exp(np.linspace(np.log(20), np.log(400), 14)),
                        np.exp(np.linspace(np.log(150), np.log(1500), 14))),
    }
    dg = np.linspace(0.0, 0.14, 15)
    for j in range(6):
        items = []
        for nm in steps:
            if man[nm]["target"][0] == j and "_r2_" in nm:
                items += transitions(nm, man)
        if len(items) < 6:
            print(f"\nj{j+1}: r2 跳变仅 {len(items)} 次, 跳过")
            continue
        big = [x for x in items if abs(x[2]) > np.median([abs(y[2]) for y in items])]
        sml = [x for x in items if abs(x[2]) <= np.median([abs(y[2]) for y in items])]
        n = sum(min(26, len(x[4])) for x in big)
        print(f"\nj{j+1}  ({len(items)} 次真阶跃, 幅值 "
              f"{min(abs(x[2]) for x in items):.1f}–{max(abs(x[2]) for x in items):.1f}°)")
        print(f"  {'模型':14s}{'参数':>26s}{'d ms':>7s}{'样本内RMS':>11s}"
              f"{'留出RMS':>10s}{'ΔAIC':>9s}")
        res = []
        for mk in MODELS:
            grids = G[mk] if MODELS[mk]["np"] == 2 else (G[mk][0][0],)
            r, p, d = fit_model(mk, big, grids, dg)
            ho = rms_of(mk, sml, d, p)
            res.append((mk, p, d, r, ho, aic(r, n, MODELS[mk]["np"])))
        a0 = min(x[5] for x in res)
        for mk, p, d, r, ho, ai in res:
            ps = " · ".join(f"{v:.0f}" for v in p)
            print(f"  {mk:14s}{ps:>26s}{d*1000:>7.0f}{r:>11.4f}{ho:>10.4f}"
                  f"{ai-a0:>9.1f}{'  ← 最优' if ai == a0 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
