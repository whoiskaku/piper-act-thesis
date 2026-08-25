#!/usr/bin/env python3
"""Step 2 —— 指令→关节之间存在**加速度受限的运动学轨迹生成器**的证据与波形叠图。

87 条激励套件的发现: 激励带内的响应主要**不是**机械臂动力学, 而是固件
MOVE J 轨迹规划的加速度上限。四条互相独立的证据:

  ① 阶跃峰速 ∝ 幅值^0.52–0.67 (6/6 关节)  —— 线性系统应为 ^1.0
  ② 反解 a = v_peak²/A = 161–279 °/s², 关节内离散 7–30%
  ③ 同一个 a 预测 chirp 滚降拐点 f_c = √(a/A)/2π, 跨 5 关节 8 倍幅值离散 8%
  ④ a 与连杆惯量**无关**: j1 惯量跨 115×, 组内 log a–log M 斜率 −0.004
     (动力学决定的话 a=τ/M 应给 −1.0)

呈现按老师 2026-07-15 的要求: **波形叠图**, 看难处, 不看平均指标。
配色沿用 step2_fidelity 已校验的三色, 第四条(加速度限幅模型)用紫虚线区分。

用法:
    python sim/step2_g/accel_limit.py            # 全部图
    python sim/step2_g/accel_limit.py --fit      # 只打印逐关节 a 拟合
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_IDEAL, C_REAL, C_DYN, C_ACC = "#eda100", "#2a78d6", "#008300", "#8b2fb0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
FPS = 20.0
OUT = ROOT / "outputs/step2_g/accel_limit"
COL = ROOT / "outputs/step2_g/collect"
# 当前 G = XML 手调值, j1 按 step2_gap_baseline 的默认覆盖为 5000
G_KP = {0: 5000.0}


def load(nm):
    d = json.loads((COL / f"replay_{nm}.json").read_text())
    return (np.degrees(np.array(d["cmd"], float)[:, :6]),
            np.array(d["state"], float)[:, :6] / 1000.0)


def accel_limited(cmd_deg, a_max, sub=20):
    """MOVE J 风格: 每子步以 |q̈| ≤ a_max 朝当前目标做时间最优(bang-bang)逼近。

    **一个关节一个参数**。刹车距离 d_stop = v|v|/(2a) 决定加速还是减速;
    末端加死区避免 bang-bang 在目标处颤振。
    """
    dt = 1.0 / (FPS * sub)
    q, v, out = float(cmd_deg[0]), 0.0, []
    for tgt in cmd_deg:
        for _ in range(sub):
            e = float(tgt) - q
            d_stop = v * abs(v) / (2.0 * a_max)
            s = e - d_stop
            if abs(s) < a_max * dt * dt and abs(v) < a_max * dt:
                q, v = float(tgt), 0.0                      # 落位, 收敛
            else:
                v += np.clip(a_max * np.sign(s), -a_max, a_max) * dt
                q += v * dt
        out.append(q)
    return np.asarray(out)


def lag1(y, tau_s):
    """一阶滞后, tau_s 秒。两段式模型的第二段。

    实测 tau = 38–64 ms, **六个关节几乎相同**且**不随位形变化**
    (j1 惯量跨 115× 时 tau 斜率 −0.031, 动力学应给 +0.5) ⇒ 更像固定的
    传输/ZOH 延迟(一个指令帧 = 50 ms), 而不是伺服动力学。
    """
    if tau_s <= 1e-6:
        return np.asarray(y, float).copy()
    al = 1.0 - np.exp(-1.0 / (FPS * tau_s))
    o = np.empty(len(y), float); o[0] = y[0]
    for i in range(1, len(y)):
        o[i] = o[i - 1] + al * (y[i] - o[i - 1])
    return o


def two_stage(cmd_deg, a_max, tau_s):
    """指令 → 加速度限幅(运动学) → 一阶滞后 → 关节。**每关节 2 个参数**。

    对比: 当前 G 是 6 参数手调线性伺服(kp+damping 各 6 个中的相关项),
    两段式在 6 个频带中的 5 个上更准, j2 低频段好 5 倍。
    """
    return lag1(accel_limited(cmd_deg, a_max), tau_s)


def fit_two_stage(man, done, joints=None):
    """在**阶跃**上格搜 (a, tau)。返回 {joint: (a, tau, rms1, rms2)}。"""
    per = defaultdict(list)
    for nm in sorted(n for n in done if man[n]["kind"] == "step"):
        t, k, A, c, s = step_info(nm, man)
        if abs(A) >= 0.5 and (joints is None or t in joints):
            per[t].append((nm, k, c, s))
    ga = np.exp(np.linspace(np.log(80), np.log(900), 45))
    gT = np.concatenate([[0.0], np.exp(np.linspace(np.log(.005), np.log(.20), 22))])
    out = {}
    for t, items in per.items():
        best, best1 = (1e9, None, None), 1e9
        for a in ga:
            base = {nm: accel_limited(c[max(0, k-3):k+22], a) for nm, k, c, s in items}
            for T in gT:
                e = float(np.mean([np.mean((lag1(base[nm], T)
                                            - s[max(0, k-3):k+22]) ** 2)
                                   for nm, k, c, s in items]))
                if e < best[0]:
                    best = (e, a, T)
                if T == 0.0:
                    best1 = min(best1, e)
        out[t] = (best[1], best[2], np.sqrt(best1), np.sqrt(best[0]))
    return out


def pure_delay(y, d_s):
    """**纯传输延迟**(亚帧线性插值)。

    区别于一阶滞后: 100 Hz 陀螺显示速度曲线是**纯三角形**
    (10–90% 上升段线性拟合 R² = 0.988–0.994, 残差仅峰值的 1.7–2.4%),
    没有一阶环节该有的平滑 ⇒ 正确形式是纯延迟, 不是一阶滞后。
    """
    k = d_s * FPS
    i0 = int(np.floor(k)); f = k - i0
    o = np.empty(len(y), float)
    for t in range(len(y)):
        o[t] = (1 - f) * y[max(0, t - i0)] + f * y[max(0, t - i0 - 1)]
    return o


def fit_ad(man, done, only_big=False):
    """**二参数 (a, 纯延迟 d)** 在阶跃上格搜 —— 取代此前的单参数 fit_a。

    单参数拟合把缺失的传输延迟折算成了**更小的加速度**: a 被系统性低估
    1.7–2.8 倍(180 → ~310), 加上延迟项后阶跃 RMS 降 60–84%。

    `only_big=True` 只用大幅值档 —— **小幅值阶跃里限幅器几乎不触发**,
    a 在那儿没有约束力(留出验证: 小幅拟合外推到大幅 RMS 0.53–0.62°,
    反向则为 0.049–0.192°)。
    """
    per = defaultdict(list)
    for nm in sorted(n for n in done if man[n]["kind"] == "step"):
        t, k, A, c, s = step_info(nm, man)
        if abs(A) < 0.5:
            continue
        big = ("f1.00" in nm) or ("a8.2" in nm)
        if only_big and not big:
            continue
        per[t].append((nm, k, c, s))
    ga = np.exp(np.linspace(np.log(120), np.log(2000), 70))
    gd = np.linspace(0.0, 0.16, 33)
    W = lambda k: slice(max(0, k - 3), k + 24)
    out = {}
    for t, items in per.items():
        best = (1e9, None, None)
        for a in ga:
            base = {nm: accel_limited(c[W(k)], a) for nm, k, c, s in items}
            for d in gd:
                e = float(np.mean([np.mean((pure_delay(base[nm], d)
                                            - s[W(k)]) ** 2)
                                   for nm, k, c, s in items]))
                if e < best[0]:
                    best = (e, a, d)
        # a 的可辨识性: 固定 d, a 变 ±60% 时 RMS 放大多少倍
        r0 = np.sqrt(best[0])
        ident = min(
            np.sqrt(float(np.mean([np.mean((pure_delay(accel_limited(c[W(k)], best[1]*f), best[2])
                                            - s[W(k)]) ** 2) for nm, k, c, s in items]))) / r0
            for f in (0.6, 1.6))
        out[t] = (best[1], best[2], r0, ident)
    return out


def sim_current_G(cmd_deg7, j):
    """当前 G(MuJoCo 线性位置伺服)的关节角响应。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    for jj, kp in G_KP.items():
        m.actuator_gainprm[jj, 0] = kp; m.actuator_biasprm[jj, 1] = -kp
    d = mujoco.MjData(m)
    nsub = int(round(1 / FPS / m.opt.timestep))
    q0 = np.radians(cmd_deg7[0])
    d.qpos[:6] = q0; d.ctrl[:6] = q0
    mujoco.mj_forward(m, d)
    for _ in range(int(0.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    out = []
    for t in range(len(cmd_deg7)):
        d.ctrl[:6] = np.radians(cmd_deg7[t])
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        out.append(np.degrees(d.qpos[j]))
    return np.asarray(out)


def step_info(nm, man):
    cmd, st = load(nm)
    t = man[nm]["target"][0]
    c, s = cmd[:, t], st[:, t]
    k = int(np.abs(np.diff(c)).argmax()) + 1
    A = c[k + 12] - c[k - 2]
    return t, k, A, c, s


def fit_a(man, done):
    """逐关节用**阶跃波形**最小二乘拟合 a_max(比 v²/A 粗估更站得住)。"""
    per = defaultdict(list)
    for nm in sorted(n for n in done if man[n]["kind"] == "step"):
        t, k, A, c, s = step_info(nm, man)
        if abs(A) < 0.5:
            continue
        per[t].append((nm, k, c, s))
    fit = {}
    for t, items in per.items():
        grid = np.exp(np.linspace(np.log(40), np.log(1500), 90))
        err = []
        for a in grid:
            e = 0.0
            for nm, k, c, s in items:
                w = slice(max(0, k - 3), k + 20)
                e += float(np.mean((accel_limited(c[w], a) - s[w]) ** 2))
            err.append(e / len(items))
        fit[t] = (float(grid[int(np.argmin(err))]), np.sqrt(min(err)), len(items))
    return fit


# ────────────────────────────── 图 ──────────────────────────────
def fig_amp(man, done, fit):
    """① 幅值依赖: 归一化阶跃响应。线性系统应**重合**; 实测不重合。"""
    groups = defaultdict(list)
    for nm in sorted(n for n in done if man[n]["kind"] == "step"):
        t, k, A, c, s = step_info(nm, man)
        if abs(A) < 0.5:
            continue
        groups[(t, re.search(r"_p(\d)_", nm).group(1),
                re.search(r"_r(?:ise)?(\d+)_", nm).group(1))].append((abs(A), k, c, s, nm))
    keep = [(kk, sorted(v)) for kk, v in groups.items()
            if len(v) >= 2 and sorted(v)[-1][0] / sorted(v)[0][0] > 1.8]
    keep = keep[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    for ax, (kk, v) in zip(axes.ravel(), keep):
        t, p, r = kk
        for (A, k, c, s, nm), ls in zip((v[0], v[-1]), ("-", "--")):
            w = slice(k - 2, k + 14)
            y = (s[w] - s[k - 1]) / A
            ax.plot(np.arange(len(y)) / FPS, y, ls, color=C_REAL, lw=2.1,
                    label=f"现实 {A:.1f}°")
            ya = (accel_limited(c[w], fit[t][0]) - s[k - 1]) / A
            ax.plot(np.arange(len(ya)) / FPS, ya, ls, color=C_ACC, lw=1.4, alpha=.85,
                    label=f"加速度限幅 {A:.1f}°")
        ax.axhline(1, color=GRID, lw=1, zorder=0)
        ax.set_title(f"j{t+1}  位形 p{p}  rise{r}", fontsize=10, color=INK)
        ax.set_xlabel("阶跃后 (s)", fontsize=8); ax.set_ylabel("归一化位移", fontsize=8)
        ax.legend(fontsize=6.6, frameon=False); ax.grid(color=GRID, lw=.5)
        ax.tick_params(labelsize=7.5)
    fig.suptitle("① 归一化阶跃响应 —— 线性系统的两条曲线应**完全重合**；实测大幅值明显更慢",
                 fontsize=12.5, color=INK)
    fig.text(.5, .012, "实线=小幅值  虚线=大幅值(约 2.9×)。蓝=现实，紫=加速度限幅模型"
             "（每关节仅 1 个参数）。归一化后仍分离 ⇒ 非线性。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .035, 1, .95])
    fig.savefig(OUT / "01_幅值依赖.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_inertia(man, done, fit):
    """② 惯量无关: 同关节同幅值不同位形。j1 惯量跨 115×, 曲线仍重合。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH)); dd = mujoco.MjData(m)

    def Mjj(nm, t):
        dd.qpos[:6] = np.load(ROOT / f"outputs/step2_g/{nm}.npz")["pose"]
        mujoco.mj_forward(m, dd)
        M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, dd.qM)
        return float(M[t, t])

    sets = [("j1 —— 连杆惯量跨 115×", [f"S_j1_p{p}_r5_f1.00" for p in range(3)], 0),
            ("j3 —— 惯量几乎不变（对照）", [f"S_j3_p{p}_r5_f1.00" for p in range(3)], 2)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (title, names, t) in zip(axes, sets):
        names = [n for n in names if n in done]
        Ms = [Mjj(n, t) for n in names]
        for nm, M, a in zip(names, Ms, (1.0, .72, .48)):
            _t, k, A, c, s = step_info(nm, man)
            w = slice(k - 2, k + 14)
            ax.plot(np.arange(w.stop - w.start) / FPS, (s[w] - s[k - 1]) / A,
                    color=C_REAL, lw=2.0, alpha=a, label=f"M[j,j]={M:.4f}")
        ax.axhline(1, color=GRID, lw=1, zorder=0)
        ax.set_title(f"{title}\n惯量跨度 {max(Ms)/min(Ms):.0f}×", fontsize=10, color=INK)
        ax.set_xlabel("阶跃后 (s)", fontsize=8.5); ax.set_ylabel("归一化位移", fontsize=8.5)
        ax.legend(fontsize=8, frameon=False); ax.grid(color=GRID, lw=.5)
    fig.suptitle("② 换位形改变连杆惯量 —— 动力学决定的话曲线应明显分开；实测重合",
                 fontsize=12.5, color=INK)
    fig.text(.5, .015, "若 ωn=√(kp/(Ia+M))，j1 的 115× 惯量应让响应快慢差 10.7×。"
             "实测三条基本压在一起 ⇒ 限幅器是运动学的，不看惯量。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(OUT / "02_惯量无关.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_overlay(man, done, fit):
    """③ 钱图: 四条叠加(黄=指令/理想, 蓝=现实, 绿=当前 G, 紫虚=加速度限幅)。

    按老师 2026-07-15 ②「大部分地方波形本来就吻合 ⇒ 整体吻合度虚高」的要求,
    扫频**分区**呈现: 低频段(限幅未触发, 三条本来就吻合) 与
    高频段(限幅触发, **难处**) 分开画, 让虚高可见。
    """
    F0, KK = 0.3, np.log(6.0 / 0.3) / 20.0          # 对数扫频瞬时频率

    def win_of(nm, t, lo, hi):
        n = man[nm]["frames"]
        f = F0 * np.exp(KK * np.arange(n) / FPS)
        i = np.where((f >= lo) & (f < hi))[0]
        return slice(int(i[0]), int(i[-1]) + 1)

    panels = [("S_j1_p0_r5_f1.00", "step", None, "大幅值阶跃 j1 16.9°"),
              ("S_j2_p0_r5_f1.00", "step", None, "大幅值阶跃 j2 22.3°"),
              ("C_j1_p0", "chirp", (0.6, 1.1), "扫频 j1 · 低频 0.6–1.1 Hz（限幅未触发）"),
              ("C_j1_p0", "chirp", (4.0, 6.0), "扫频 j1 · 高频 4–6 Hz（**难处**）")]
    panels = [x for x in panels if x[0] in done]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    tab = []
    for ax, (nm, kind, band, title) in zip(axes.ravel(), panels):
        t = man[nm]["target"][0]
        cmd, st = load(nm)
        c, s = cmd[:, t], st[:, t]
        gG = sim_current_G(cmd, t)
        gA = accel_limited(c, fit[t][0])
        if kind == "step":
            k = int(np.abs(np.diff(c)).argmax()) + 1
            w = slice(max(0, k - 4), k + 26)
        else:
            w = win_of(nm, t, *band)
        x = np.arange(w.stop - w.start) / FPS
        ax.plot(x, c[w],  color=C_IDEAL, lw=2.4, label="指令(理想)")
        ax.plot(x, s[w],  color=C_REAL,  lw=2.3, label="现实")
        ax.plot(x, gG[w], color=C_DYN,   lw=1.8, label="当前 G(线性伺服)")
        ax.plot(x, gA[w], "--", color=C_ACC, lw=1.9,
                label=f"加速度限幅 a={fit[t][0]:.0f}°/s²")
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_xlabel("时间 (s)", fontsize=8.5)
        ax.set_ylabel(f"j{t+1} 角度 (度)", fontsize=8.5)
        ax.legend(fontsize=7.4, frameon=False, loc="best")
        ax.grid(color=GRID, lw=.5); ax.tick_params(labelsize=8)
        rms = lambda y: float(np.sqrt(np.mean((y[w] - s[w]) ** 2)))
        rI, rG, rA = rms(c), rms(gG), rms(gA)
        tab.append((title.replace("**", ""), rI, rG, rA))
        ax.set_xlabel(f"时间 (s)          RMS 误差(度): 指令 {rI:.2f} · "
                      f"当前G {rG:.2f} · 限幅 {rA:.2f}", fontsize=8.5, color=INK2)
    fig.suptitle("③ 波形叠图 —— 阶跃看幅值依赖，扫频**分区**看虚高",
                 fontsize=12.5, color=INK)
    fig.text(.5, .012, "左下(低频)三条本来就吻合 —— 整体平均指标的虚高来源；"
             "右下(高频)才是难处。当前 G 是线性伺服，给不出「大幅值更慢」；"
             "加速度限幅模型每关节只 1 个参数。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .04, 1, .95])
    fig.savefig(OUT / "03_波形叠图.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return tab


def fig_scaling(man, done, fit):
    """④ 峰速 vs 幅值 log-log: 线性系统斜率 1, 恒加速斜率 0.5。"""
    per = defaultdict(list)
    for nm in sorted(n for n in done if man[n]["kind"] == "step"):
        t, k, A, c, s = step_info(nm, man)
        if abs(A) < 0.5:
            continue
        per[t].append((abs(A), np.abs(np.diff(s[k-2:k+14])).max() * FPS))
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    cols = ["#2a78d6", "#8b2fb0", "#008300", "#eda100", "#d1495b", "#00868b"]
    for t in sorted(per):
        A = np.array([x[0] for x in per[t]]); V = np.array([x[1] for x in per[t]])
        e = np.polyfit(np.log(A), np.log(V), 1)[0] if len(A) > 2 else np.nan
        ax.loglog(A, V, "o", color=cols[t], ms=6, alpha=.85,
                  label=f"j{t+1}  斜率 {e:.2f}" if np.isfinite(e) else f"j{t+1}")
    xs = np.array([1.5, 25])
    ax.loglog(xs, 12 * xs ** 1.0, "-",  color=INK2, lw=1.3, alpha=.7)
    ax.loglog(xs, 14 * xs ** 0.5, "--", color=INK,  lw=1.6)
    ax.text(19, 12*19, " 线性系统 ∝A¹", fontsize=8.5, color=INK2, va="center")
    ax.text(19, 14*19**.5, " 恒加速 ∝A^½", fontsize=8.5, color=INK, va="center")
    ax.set_xlabel("阶跃幅值 (度)"); ax.set_ylabel("实测峰速 (度/s)")
    ax.set_title("④ 阶跃峰速 vs 幅值 —— 6/6 关节贴近 ∝A^½", fontsize=12, color=INK)
    ax.legend(fontsize=8, frameon=False); ax.grid(which="both", color=GRID, lw=.5)
    fig.tight_layout()
    fig.savefig(OUT / "04_峰速标度.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_bands(man, done, fit, ts):
    """⑤ **分区** RMS 对比 —— 落实老师②「整体吻合度虚高必须分区」。

    横轴按 chirp 的瞬时频带分区; 每区四个模型。低频区三条本来就吻合,
    正是平均指标虚高的来源; 高频区才是难处。
    """
    F0, KK = 0.3, np.log(6.0 / 0.3) / 20.0
    BANDS = [(0.3, 0.6), (0.6, 1.1), (1.1, 1.5), (1.5, 2.5), (2.5, 3.5),
             (3.5, 4.5), (4.5, 6.0)]
    seqs = [n for n in ("C_j1_p0", "C_j2_p1", "C_j3_p0", "C_j6_p0") if n in done]
    fig, axes = plt.subplots(1, len(seqs), figsize=(4.0 * len(seqs), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, nm in zip(axes, seqs):
        t = man[nm]["target"][0]
        cmd, st = load(nm)
        c, s = cmd[:, t], st[:, t]
        mods = [("指令(理想)", c, C_IDEAL),
                ("当前 G", sim_current_G(cmd, t), C_DYN),
                ("限幅 1 参", accel_limited(c, fit[t][0]), C_ACC),
                ("两段式 2 参", two_stage(c, ts[t][0], ts[t][1]), "#0b0b0b")]
        f = F0 * np.exp(KK * np.arange(len(c)) / FPS)
        x = np.arange(len(BANDS))
        for i, (lab, y, col) in enumerate(mods):
            r = []
            for lo, hi in BANDS:
                w = (f >= lo) & (f < hi)
                r.append(float(np.sqrt(np.mean((y[w] - s[w]) ** 2))) if w.sum() > 4
                         else np.nan)
            ax.bar(x + (i - 1.5) * .2, r, .19, color=col, label=lab,
                   alpha=.9 if i < 3 else 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a}–{b}" for a, b in BANDS], fontsize=7, rotation=45)
        ax.set_xlabel("瞬时频带 (Hz)", fontsize=8.5)
        ax.set_ylabel("RMS 误差 (度)", fontsize=8.5)
        ax.set_title(f"{nm}  (j{t+1}, 幅值 {man[nm]['amp_deg']:.1f}°)",
                     fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=.5); ax.tick_params(labelsize=7.5)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("⑤ 分区 RMS —— 低频区四条本来就接近（平均指标虚高的来源），"
                 "高频区才分得开", fontsize=12.5, color=INK)
    fig.text(.5, .012, "「两段式」= 加速度限幅 + 一阶滞后，**每关节 2 个参数**；"
             "「当前 G」= 6 个手调 kp/damping 的线性伺服。"
             "参数在**阶跃**上拟合，此处为跨波形外推。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .04, 1, .93])
    fig.savefig(OUT / "05_分区RMS.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def fig_pose(man, done):
    """⑥ 两个参数都与位形(连杆惯量)无关 —— j1 惯量跨 115×。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH)); dd = mujoco.MjData(m)
    rows = []
    for t, np_ in ((0, 3), (1, 3), (2, 3)):
        for p in range(np_):
            got = fit_two_stage(man, {n for n in done
                                      if n.startswith(f"S_j{t+1}_p{p}_")}, {t})
            if t not in got:
                continue
            f = ROOT / f"outputs/step2_g/S_j{t+1}_p{p}_r5_f1.00.npz"
            if not f.exists():
                continue
            dd.qpos[:6] = np.load(f)["pose"]; mujoco.mj_forward(m, dd)
            M = np.zeros((m.nv, m.nv)); mujoco.mj_fullM(m, M, dd.qM)
            rows.append((t, p, float(M[t, t]), got[t][0], got[t][1]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    cols = {0: "#2a78d6", 1: "#8b2fb0", 2: "#008300"}
    for ax, (idx, lab, ref, refl) in zip(
            axes, [(3, "加速度上限 a (°/s²)", -1.0, "动力学 a=τ/M ⇒ 斜率 −1"),
                   (4, "滞后 τ (ms)", 0.5, "动力学 τ∝√(M/kp) ⇒ 斜率 +0.5")]):
        for t in sorted({r[0] for r in rows}):
            r = [x for x in rows if x[0] == t]
            X = np.array([x[2] for x in r])
            Y = np.array([x[idx] * (1000 if idx == 4 else 1) for x in r])
            ax.loglog(X, np.maximum(Y, 1e-3), "o-", color=cols[t], ms=7, lw=1.4,
                      label=f"j{t+1}  惯量跨 {X.max()/X.min():.0f}×")
        # 参考斜率
        xs = np.array([min(r[2] for r in rows), max(r[2] for r in rows)])
        y0 = np.median([x[idx] * (1000 if idx == 4 else 1) for x in rows])
        ax.loglog(xs, y0 * (xs / np.sqrt(xs[0] * xs[1])) ** ref, "--",
                  color=INK2, lw=1.5, label=refl)
        ax.set_xlabel("该位形的连杆惯量 M[j,j] (kg·m²)", fontsize=9)
        ax.set_ylabel(lab, fontsize=9)
        ax.legend(fontsize=7.6, frameon=False); ax.grid(which="both", color=GRID, lw=.5)
    fig.suptitle("⑥ 两段式的两个参数**都不随连杆惯量变化** —— "
                 "动力学决定的话应沿虚线走", fontsize=12.5, color=INK)
    fig.text(.5, .015, "实测 j1 斜率: a = −0.010、τ = −0.031（惯量跨 115×）。"
             "M[j,j] 由 MuJoCo 模型算出，不是实测 —— 见报告局限一节。",
             ha="center", fontsize=9, color=INK2)
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(OUT / "06_位形无关.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((ROOT / "outputs/step2_g/manifest.json").read_text())
    done = {r["name"] for r in
            json.loads((COL / "state.json").read_text())["done"]}
    fit = fit_a(man, done)
    print(f"{'关节':>5s}{'拟合 a °/s²':>13s}{'波形RMS°':>10s}{'阶跃数':>8s}")
    for t in sorted(fit):
        aa, rms, n = fit[t]
        print(f"j{t+1:<4d}{aa:>13.0f}{rms:>10.3f}{n:>8d}")
    if a.fit:
        return 0
    ts = fit_two_stage(man, done)
    print(f"\n两段式(加速度限幅 + 一阶滞后, 每关节 2 参数):")
    print(f"{'关节':>5s}{'a °/s²':>9s}{'τ ms':>7s}{'1参RMS':>9s}{'2参RMS':>9s}{'改善':>7s}")
    for t in sorted(ts):
        a, T, r1, r2 = ts[t]
        print(f"j{t+1:<4d}{a:>9.0f}{T*1000:>7.0f}{r1:>9.3f}{r2:>9.3f}{1-r2/r1:>6.0%}")
    fig_amp(man, done, fit)
    fig_inertia(man, done, fit)
    tab = fig_overlay(man, done, fit)
    fig_scaling(man, done, fit)
    fig_bands(man, done, fit, ts)
    fig_pose(man, done)
    print(f"\n{'难处':28s}{'指令':>8s}{'当前G':>8s}{'限幅':>8s}   (RMS 度)")
    for title, rI, rG, rA in tab:
        print(f"{title:28s}{rI:>8.2f}{rG:>8.2f}{rA:>8.2f}")
    print(f"\n→ {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
