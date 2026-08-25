#!/usr/bin/env python3
"""Step 2 激励序列生成器 —— 20 Hz 指令，供 `replay_probe.py` 开环复放。

设计依据全部来自零机时的结构性分析，不是猜的：

  · 绕 j 轴的等效惯量**只**取决于比 j 更远端的关节姿态
    ⇒ 激励 j 时要变的是 `j+1..6`（`step2_identifiability.py` 实测：
      改近端完全无效，probe_c 三位形的 `I_link@j3` 极差 0.0%）。
  · `b` 与 `kv` 精确简并 ⇒ 参数集已合并为 `c`，序列设计**不必**为分离二者服务
    （`step2_g/test_degeneracy.py`）。慢速双向扫描仍然要做 —— 它是为 `fc` 的。
  · 确定性内容在 **9–12 Hz** 截止（`step2_ceiling.py`）⇒ chirp 不必往上扩。
    且指令率只有 20 Hz，故 chirp 封顶 **6 Hz**（≥3.3 样本/周期）；
    更高频的激励靠**阶跃**的宽带内容提供。
  · **j4 在 pp2 里的指令范围是 `0.0…0.0`** —— 从未动过。j5 也几乎不动。
    这些关节的全部参数目前无数据可辨识，必须直接激励。

安全口径（重要）：
  本模块只做**关节限位**与**速度/加速度**检查。`piper_model.xml` 的碰撞是
  **全关**的（`contype/conaffinity` 非零 0/17），所以**不得**声称「碰撞安全通过」，
  只能说「关节限位通过」。序列会走出 pp2 已验证的包络（尤其 j4/j5），
  **上机前必须由人在 MuJoCo 里目视确认无碰撞**。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import mujoco

FPS = 20.0
DT = 1.0 / FPS
LIMIT_MARGIN_DEG = 5.0
CHIRP_F_MAX = 6.0                    # 指令率 20 Hz ⇒ ≥3.3 样本/周期
GRIP_HOLD = 0.0                      # 夹爪全程不动

# 速度/加速度上限：pp2 实测 p95；j4 在 pp2 里从未动过，借 j5 的（更保守的腕关节值）
VMAX_DEG = np.array([119.2, 293.7, 123.9, 58.3, 58.3, 143.7])
# 近端关节的停放姿态 = pp2 实测指令范围的中点 —— 真机已反复安全经过, 且远离限位。
# **不能**简单置 0: j2 的下限就是 0、j3 的上限就是 0, 置 0 即触 margin。
HOME_DEG = np.array([-4.9, 51.5, -25.2, 0.0, 9.4, 4.0])
AMAX_DEG = np.array([2429.0, 6475.0, 2448.0, 850.0, 850.0, 2976.0])


@dataclass
class Seq:
    name: str
    kind: str
    target: tuple[int, ...]
    cmd: np.ndarray                        # (T,7) rad
    pose_distal: np.ndarray                # 远端关节配置(rad)，决定 I_link[target]
    meta: dict = field(default_factory=dict)

    @property
    def sha(self) -> str:
        return hashlib.sha256(np.round(self.cmd, 9).tobytes()).hexdigest()[:12]


TABLE_Z = -0.001                     # deco_table 台面高度(m)
Z_CLEAR = 0.02                       # 台面之上要留的余量(m)
# 过冲不能靠猜系数: 实测本套件 j2 快阶跃的仿真过冲达 **1.57–2.01×**, 远超
# 曾假设的 1.2×(那是慢扫的值)。故 `fit_workspace` **直接跑动力学**并判仿真轨迹,
# 与审查端(visual_check 渲染仿真位形)量的是同一个东西, 不留口径差。
# 局限: 用的是**未标定**模型的动力学, 过冲估计本身有不确定性 —— 又一条
# 「人工目视签字仍然必需」的理由。
WS_BISECT = 8                        # 动力学校验的二分轮数
# 二分**不能停在阈值上**: 收敛到 dyn_z == need 时, 审查端 0.1–0.2 mm 的数值差
# (采样相位/浮点次序)就会把结论翻到线下 —— 实测 3 条 j2 序列生成端 +0.0190、
# 审查端 +0.0188~9, need 恰为 0.0190。留 10 倍于该差异的收敛余量。
BISECT_MARGIN = 0.002
# 位形能不能用, 不能靠「多留 X 米余量」这种硬阈值判 —— 定小了会选出「静止余量
# 1 mm、一摆就触底」的位形, 定大了(实测 0.03 m)会把 j1 的候选**全部**筛掉。
# 改为直接问「这个位形还摆得动多少」: 要求至少能摆 MIN_USABLE_AMP_DEG。
MIN_USABLE_AMP_DEG = 1.0
DYN_GUARD = 2.5                      # 位形筛选时给动力学过冲留的倍数(实测最高 2×)
PREMOVE_S, SETTLE_S, RETURN_S = 2.0, 1.0, 2.0     # 上机时间线的三段


# 机械臂的**实际停放位姿** = SDK 的 safe_disable_position, 不是 HOME_DEG。
# 上机时 replay_probe 的预移动从这里出发, 归位也回到这里。
SAFE_DISABLE_DEG = np.array([0.0, 0.0, 0.0, 0.0, 29.794, 0.0])
# 分段归位: 与 replay_probe.RETURN_STAGES 必须一致
RETURN_STAGES = ((5,), (3, 4), (0, 1, 2))
RETURN_STAGE_S = 3.0


def full_timeline(cmd: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, str]]]:
    """完整上机时间线：停放 →(预移动)→ 起点 →(稳定)→ 激励 →(分段归位)→ 停放。

    **工作空间检查必须查全程, 不能只查激励段** —— 预移动同样会扫过空间。
    实测只查激励段时, 生成端判过、审查端(查全程)仍报「余量不足」, 两端打架。
    生成与审查共用本函数, 消除口径差。

    ── 2026-08-26 修正两端: 此前本函数建模的是「HOME →预移动→ … →2s 插值回 HOME」,
    而真机执行的是「**safe_disable** →预移动→ … →固件一次性无插值写 safe_disable」。
    **起点、终点、路径、有无插值四者全不同** ⇒ 检查过的时间线与执行的时间线
    从来不是同一条。第 54 条 S_j4_p2_rise10_a8.2 就是在这个未覆盖的归位段撞的。
    现在两端都按真机改: 起点/终点均为 SAFE_DISABLE_DEG, 归位改为**分段插值**
    (j6 单独先走 → j4/j5 → j1–j3), 与 replay_probe.staged_return 一一对应。
    """
    park = np.zeros(7)
    park[:6] = np.deg2rad(SAFE_DISABLE_DEG)
    npre, nset = (int(x * FPS) for x in (PREMOVE_S, SETTLE_S))
    r1 = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, npre)))[:, None]
    pre = park[None, :] + (cmd[0] - park)[None, :] * r1
    setl = np.repeat(cmd[0][None, :], nset, axis=0)
    # 分段归位: 一次只动一组, 每段 RETURN_STAGE_S 秒余弦插值
    nst = int(RETURN_STAGE_S * FPS)
    r2 = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, nst)))[:, None]
    q = cmd[-1].copy()
    segs, names = [], []
    for idx in RETURN_STAGES:
        tgt = q.copy()
        for j in idx:
            tgt[j] = park[j]
        segs.append(q[None, :] + (tgt - q)[None, :] * r2)
        names.append("归位 j" + "/j".join(str(j + 1) for j in idx))
        q = tgt
    tail = np.repeat(park[None, :], int(0.5 * FPS), axis=0)
    full = np.concatenate([pre, setl, cmd] + segs + [tail])
    k, spans = 0, []
    parts = [("预移动", npre), ("稳定", nset), ("激励", len(cmd))]
    parts += [(n, nst) for n in names]
    parts += [("停放", len(tail))]
    for nm, n in parts:
        spans.append((k, k + n, nm))
        k += n
    return full, spans


def lowest_geom_z(m, d, q6: np.ndarray) -> float:
    """给定位形下**所有 geom 的保守最低点** z。

    用 `geom_xpos - geom_rbound`（包围球）而非 body 原点：连杆的几何体比原点低，
    只看原点会漏判。

    **只看手臂自己的 geom**（`geom_bodyid != 0`）。模型里的 `deco_floor/table/
    tray/cube/battery/slot_*` 都挂在 `world` 上、本来就位于台面高度或以下 ——
    把它们算进来会让任何位形都判为「低于台面」（实测最小 z−rbound = −0.575，
    全部来自装饰件）。

    这仍是**粗粒度几何护栏**，不是碰撞判定（模型碰撞是关闭的），
    人工目视签字仍然必需。
    """
    d.qpos[:6] = q6
    d.qpos[6] = d.qpos[7] = 0.0
    mujoco.mj_forward(m, d)
    arm = m.geom_bodyid != 0                      # 排除 world 上的静态装饰件
    return float((d.geom_xpos[arm, 2] - m.geom_rbound[arm]).min())



# ── 位形选择 ────────────────────────────────────────────────────────────
def link_inertia(m, d, M, q6: np.ndarray) -> np.ndarray:
    d.qpos[:6] = q6
    d.qpos[6] = d.qpos[7] = 0.0
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, M, d.qM)
    return np.diag(M)[:6].copy()


def pick_distal_poses(m, j: int, n: int = 3, n_sample: int = 20000,
                      seed: int = 20260824) -> list[np.ndarray]:
    """为激励关节 `j` 选 n 个**远端**配置，使 `I_link[j]` 的跨度尽量大 ——
    但**只在工作空间可行的位形里选**。

    近端关节固定在 pp2 的典型姿态（不影响 `I_link[j]`，但决定重力负载）。

    **必须先筛可行再取极值**：惯量最极端的位形往往正是把手臂折进桌面的那些。
    实测未筛时 j1 的最大惯量位形(j2=170°, j3=−164°)在 HOME 处 z=−0.059、
    j2 的最小惯量位形(j3=−6°, j4=−13°) z=−0.016 —— 都是**静止时就在桌下**，
    先取极值再筛只会得到空集或幅值被压到 0。
    **采样数**: 加上「路径可行 + 摆得动」两道过滤后，高惯量那一端的幸存者很稀疏。
    3000 次采样时 j1 的最高惯量位形(0.544)被全部滤掉，跨度从 135.6× 塌到 26.3×，
    而 `Ia1` 恰恰是靠位形跨度才可辨识的。采样是构建里**最便宜的部分**(纯运动学)，
    故提到 2 万。

    可行性判两件事，缺一不可：
      · **位形本身**（目标关节置 HOME，即摆动中心）在台面之上；
      · **从 HOME 到该位形的接近路径**全程在台面之上 —— `full_timeline` 的预移动
        是两点间的关节直线插值，**两端都可行不代表中间可行**。实测 j1 的最大惯量
        位形两端都过、路径穿桌，会在激励**开始之前**就撞上去。
    整条激励轨迹的余量再由 `fit_workspace()` 缩幅保证。
    """
    d, M = mujoco.MjData(m), np.zeros((m.nv, m.nv))
    lo = m.jnt_range[:6, 0] + np.deg2rad(LIMIT_MARGIN_DEG)
    hi = m.jnt_range[:6, 1] - np.deg2rad(LIMIT_MARGIN_DEG)
    home = np.deg2rad(HOME_DEG)
    if j >= 5:                                    # j6 无远端 ⇒ 只有一个位形
        return [home.copy()]
    rng = np.random.default_rng(seed + j)
    Q = rng.uniform(lo, hi, size=(n_sample, 6))
    Q[:, : j + 1] = home[: j + 1]                 # 近端停 HOME, 只变远端
    need = TABLE_Z + Z_CLEAR
    home = np.deg2rad(HOME_DEG)
    keep = []
    for q in Q:
        qc = q.copy()
        qc[j] = np.deg2rad(HOME_DEG[j])            # 摆动中心
        # 位形本身 + 从 HOME 过去的**整条接近路径**
        path = [home + (qc - home) * a_ for a_ in np.linspace(0.0, 1.0, 13)]
        if all(lowest_geom_z(m, d, x) >= need for x in path):
            keep.append(q)
    if not keep:
        raise RuntimeError(f"j{j+1}: 没有工作空间可行的远端位形")
    Q = np.array(keep)
    I = np.array([link_inertia(m, d, M, q)[j] for q in Q])
    order = list(np.argsort(I))

    def usable(q: np.ndarray) -> bool:
        """该位形下还能摆 ≥ MIN_USABLE_AMP_DEG 吗。

        运动学判(便宜), 但按 `DYN_GUARD` 倍幅值检查 —— 动力学过冲实测最高 2×,
        不留这个倍数会选出「运动学摆得动、动力学一摆就触底」的位形。
        """
        c = place(m, smooth_step(j, q, MIN_USABLE_AMP_DEG * DYN_GUARD, 5), j)
        f, _ = full_timeline(c)
        return all(lowest_geom_z(m, d, f[k, :6]) >= need
                   for k in range(0, len(f), 5))

    def best_in(seq):
        """在给定的一档里，取**离台面最高**的那个位形。

        不能直接取惯量极值: 惯量最大的位形往往是手臂伸最远的那个, 静止时勉强
        过关、动力学过冲一来就触底 —— 实测 j1 惯量 0.550 的位形被选中后, 它的
        7 条序列**全部**因幅值不足被丢弃, 套件里只剩 0.101, 跨度从 143× 塌到 26×。
        在同一惯量档内换一个余量大的, 惯量只损失一点, 摆动空间大得多。
        """
        cand = [i for i in seq if usable(Q[i])]
        if not cand:
            return None
        return max(cand, key=lambda i: lowest_geom_z(m, d, _center(Q[i])))

    def _center(q):
        qc = q.copy()
        qc[j] = np.deg2rad(HOME_DEG[j])
        return qc

    k = max(1, len(order) // 20)                      # 每档取 5%
    picks = [best_in(order[:k]), best_in(order[-k:])]           # 低惯量档 / 高惯量档
    if n >= 3:
        mid = len(order) // 2
        picks.insert(1, best_in(order[mid - k // 2: mid + k // 2 + 1]))
    picks = [i for i in picks if i is not None]
    if not picks:
        raise RuntimeError(f"j{j+1}: 没有既可行、又摆得动 {MIN_USABLE_AMP_DEG}° 的远端位形")
    seen, out = set(), []
    for i in picks[:n]:
        if i not in seen:
            seen.add(i)
            out.append(Q[i])
    return out


# ── 基元 ────────────────────────────────────────────────────────────────
def safe_center(m, j: int, amp_deg: float) -> float:
    """关节 j 的起始角(度)，保证 ±amp 全程留在限位内。

    **仅供多正弦的基准姿态使用。** 单关节序列改用 `place()` —— 它按波形的实际
    偏移范围居中，而本函数假设的是关于起点对称的 ±amp（阶跃并不对称）。

    必需，因为若干关节的限位**一端就是 0**（j2 下限 0、j3 上限 0），
    从 0 出发做双向摆动必然越界。
    """
    lo = np.rad2deg(m.jnt_range[j, 0]) + LIMIT_MARGIN_DEG + abs(amp_deg)
    hi = np.rad2deg(m.jnt_range[j, 1]) - LIMIT_MARGIN_DEG - abs(amp_deg)
    if lo > hi:                                   # 幅值过大，取中点并由调用方缩幅
        return float((lo + hi) / 2)
    return float(np.clip(0.0, lo, hi))


def _base(pose: np.ndarray, T: int) -> np.ndarray:
    c = np.zeros((T, 7))
    c[:, :6] = pose[None, :6]
    c[:, 6] = GRIP_HOLD
    return c


def _raised_cos(n: int) -> np.ndarray:
    """0→1 的平滑过渡（升余弦），端点速度与加速度均为 0。"""
    t = np.linspace(0.0, 1.0, n, endpoint=True)
    return 0.5 * (1.0 - np.cos(np.pi * t))


def smooth_step(j: int, pose: np.ndarray, amp_deg: float, rise_frames: int,
                hold_s: float = 1.5, n_rep: int = 2, q0_deg: float = 0.0) -> np.ndarray:
    """平滑阶跃（升余弦上升）+ 保持，双向各 n_rep 次。"""
    hold = int(hold_s * FPS)
    unit = rise_frames + hold
    T = 2 * n_rep * unit + hold
    c = _base(pose, T)
    q = np.full(T, np.deg2rad(q0_deg))
    k = hold
    for r in range(n_rep):
        for sgn in (+1, -1):
            q[k:k + rise_frames] = q[k - 1] + sgn * np.deg2rad(amp_deg) * _raised_cos(rise_frames)
            q[k + rise_frames:k + unit] = q[k + rise_frames - 1]
            k += unit
    c[:, j] = q
    return c


def slow_sweep(j: int, pose: np.ndarray, amp_deg: float, speed_dps: float,
               n_rev: int = 4, q0_deg: float = 0.0) -> np.ndarray:
    """慢速**双向**扫描（三角波）—— 换向处的死区是库仑摩擦 `fc` 的直接签名。"""
    leg = max(int(abs(amp_deg) / max(speed_dps, 1e-6) * FPS), 4)
    T = n_rev * leg + int(1.0 * FPS)
    c = _base(pose, T)
    q = np.full(T, np.deg2rad(q0_deg))
    k = int(0.5 * FPS)
    sgn = 1
    while k + leg <= T:
        q[k:k + leg] = q[k - 1] + sgn * np.deg2rad(amp_deg) * np.linspace(0, 1, leg)
        k += leg
        sgn = -sgn
    q[k:] = q[k - 1]
    c[:, j] = q
    return c


def chirp(j: int, pose: np.ndarray, amp_deg: float, f0: float = 0.3,
          f1: float = CHIRP_F_MAX, dur_s: float = 20.0, q0_deg: float = 0.0) -> np.ndarray:
    """对数扫频。封顶 `CHIRP_F_MAX` —— 指令率 20 Hz 限制，且 >12 Hz 无可复现信号。"""
    T = int(dur_s * FPS)
    c = _base(pose, T)
    t = np.arange(T) * DT
    k = np.log(f1 / f0) / dur_s
    ph = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    env = np.clip(np.minimum(t / 1.0, (dur_s - t) / 1.0), 0, 1)      # 首尾渐入渐出
    c[:, j] = np.deg2rad(q0_deg) + np.deg2rad(amp_deg) * env * np.sin(ph)
    return c


def multisine(joints: tuple[int, ...], pose: np.ndarray, amps_deg: dict[int, float],
              dur_s: float = 25.0, n_tone: int = 12, seed: int = 0) -> np.ndarray:
    """多关节随机相位多正弦 —— 唯一能激励**惯量耦合**项的形式。"""
    T = int(dur_s * FPS)
    c = _base(pose, T)
    t = np.arange(T) * DT
    rng = np.random.default_rng(seed)
    f = np.logspace(np.log10(0.3), np.log10(CHIRP_F_MAX), n_tone)
    env = np.clip(np.minimum(t / 1.5, (dur_s - t) / 1.5), 0, 1)
    for j in joints:
        ph = rng.uniform(0, 2 * np.pi, n_tone)
        s = np.sum(np.sin(2 * np.pi * f[:, None] * t[None, :] + ph[:, None]), axis=0)
        s /= np.abs(s).max()
        c[:, j] = pose[j] + np.deg2rad(amps_deg[j]) * env * s
    return c


# ── 安全与度量 ──────────────────────────────────────────────────────────
def _dev(cmd: np.ndarray, j: int) -> np.ndarray:
    """关节 j 相对**本序列起点**的偏移。"""
    return cmd[:, j] - cmd[0, j]


def max_scale(m, cmd: np.ndarray, j: int) -> float:
    """满足 速度 / 加速度 / **行程** 三项限制的最大等比缩放。**不截到 1。**

    2026-08-24 修: 旧实现 `fit_limits` 把比例截在 `min(1.0, …)`, 而 `max_amp`
    用 1° 探针去问「最大幅值」—— 探针本就在限内 ⇒ 比例恒为 1 ⇒ **每个关节的
    「可达最大幅值」都返回 1.0°**。实测真正可达 2.9°–14.7°, 即整个套件被压到
    1° 附近, **欠激励 3–15 倍**。此函数返回未截断的比例, 供 `max_amp` 使用;
    `fit_limits`（只缩不放）保留给多正弦等已显式定幅的波形。
    """
    d = _dev(cmd, j)
    v = np.abs(np.diff(d)) * FPS
    a = np.abs(np.diff(d, n=2)) * FPS ** 2
    sv = np.deg2rad(VMAX_DEG[j]) / max(v.max(), 1e-12)
    sa = np.deg2rad(AMAX_DEG[j]) / max(a.max(), 1e-12)
    room = (m.jnt_range[j, 1] - m.jnt_range[j, 0]) - 2 * np.deg2rad(LIMIT_MARGIN_DEG)
    sq = room / max(d.max() - d.min(), 1e-12)
    return float(min(sv, sa, sq))


def fit_limits(m, cmd: np.ndarray, j: int) -> tuple[np.ndarray, float]:
    """把关节 j 的幅值**只缩不放**到限内。→ (指令, 缩放比 ≤ 1)。"""
    s = min(1.0, max_scale(m, cmd, j))
    if s < 1.0:
        cmd = place(m, cmd, j, s)
    return cmd, s


def place(m, cmd: np.ndarray, j: int, s: float = 1.0) -> np.ndarray:
    """把关节 j 的轨迹按 s 缩放, 并**锚定在 HOME 附近**, 仅在触限时最小平移。

    **不能**居中到关节的机械量程中点: j2 的量程是 0–180°, 中点 90° 那个姿态
    配上折叠的远端位形本身就在**桌面以下** —— 实测按量程居中会让工作空间二分
    一路缩到幅值 0。锚在 HOME(pp2 已反复安全经过的姿态)才是对的参照。

    比原来的 `safe_center` 也更正确: 后者假设摆动关于起点对称 ±amp, 而阶跃的
    偏移范围是 [0, amp]、扫频是 [−amp, +amp] —— 形状不同, 锚点也不同。
    """
    out = cmd.copy()
    d = _dev(cmd, j) * s
    q = np.deg2rad(HOME_DEG[j]) - (d.max() + d.min()) / 2 + d      # 摆动中心 = HOME
    lo = m.jnt_range[j, 0] + np.deg2rad(LIMIT_MARGIN_DEG)
    hi = m.jnt_range[j, 1] - np.deg2rad(LIMIT_MARGIN_DEG)
    q = q + max(0.0, lo - q.min()) - max(0.0, q.max() - hi)        # 触限才平移
    out[:, j] = q
    return out


def fit_workspace(m, cmd: np.ndarray, j: int, stride: int = 5) -> tuple[np.ndarray, float]:
    """把关节 j 的幅值二分缩小，直到全程所有 geom 都在台面之上。

    起因：`W_j2_p1_v2` 的慢扫只受**行程**约束（j2 量程 0–180°），幅值涨到 132°，
    配上折叠的远端位形把夹爪扫到 **z = −0.208 m**，即台面下 21 cm。
    速度/加速度/关节限位三项检查都发现不了这个 —— 它是**工作空间**问题。
    """
    d = mujoco.MjData(m)
    need = TABLE_Z + Z_CLEAR + BISECT_MARGIN      # 目标留余量, 别收敛到边界

    def kin_ok(s: float) -> bool:
        c, _ = full_timeline(place(m, cmd, j, s))      # 查**全程**, 含预移动/回停
        return all(lowest_geom_z(m, d, c[k, :6]) >= need
                   for k in range(0, len(c), stride))

    def dyn_z(s: float) -> float:
        """跑动力学, 返回**完整上机时间线**仿真位形的最低 geom z。"""
        c, _ = full_timeline(place(m, cmd, j, s))
        dd = mujoco.MjData(m)
        dd.qpos[:6] = c[0, :6]
        mujoco.mj_forward(m, dd)
        dd.ctrl[:6] = c[0, :6]
        sub = int(round(1 / FPS / m.opt.timestep))
        z = np.inf
        for k in range(len(c)):
            dd.ctrl[:6] = c[k, :6]
            for _ in range(sub):
                mujoco.mj_step(m, dd)
            if k % stride == 0:
                z = min(z, float((dd.geom_xpos[m.geom_bodyid != 0, 2]
                                  - m.geom_rbound[m.geom_bodyid != 0]).min()))
        return z

    # 先用**运动学**二分快速定位(便宜), 再对**动力学**二分(准确)。
    # 早先用「固定比例逐轮收缩、上限 5 轮」, 若 5 轮后仍不达标就**静默返回不合规的
    # 序列** —— 实测 j1 的 p2 位形有 7 条这样漏出去, 在审查端才被报出来。
    # 二分会收敛; 若连极小幅值都不达标, 明确抛错而不是放行。
    s = 1.0
    if not kin_ok(1.0):
        lo_, hi_ = 0.0, 1.0
        for _ in range(10):
            mid = (lo_ + hi_) / 2
            if kin_ok(mid):
                lo_ = mid
            else:
                hi_ = mid
        s = lo_
    if dyn_z(s) < need:
        lo_, hi_ = 0.0, s
        for _ in range(WS_BISECT):
            mid = (lo_ + hi_) / 2
            if dyn_z(mid) >= need:
                lo_ = mid
            else:
                hi_ = mid
        s = lo_
    # **不抛错**: 位形筛选用的是运动学判据, 而这里是动力学(过冲最高 2×), 两者必然
    # 有落差。硬崩会让整套构建失败; 改为返回能找到的最大幅值, 由 `build()` 按
    # 「幅值是否够用」决定丢弃并**报告**, 而不是静默放行或整体崩溃。
    return place(m, cmd, j, s), s


def max_amp(m, j: int, gen, probe_deg: float = 1.0) -> float:
    """关节 j 在 速度/加速度/行程 限内的**真正可达最大幅值**(度)。"""
    return probe_deg * max_scale(m, gen(probe_deg), j)


def check_limits(m, cmd: np.ndarray) -> dict:
    """**只**检查关节限位与速度/加速度。碰撞在模型里是关闭的 ⇒ 不做碰撞判定。"""
    q = cmd[:, :6]
    lo = m.jnt_range[:6, 0] + np.deg2rad(LIMIT_MARGIN_DEG)
    hi = m.jnt_range[:6, 1] - np.deg2rad(LIMIT_MARGIN_DEG)
    v = np.abs(np.diff(q, axis=0)) * FPS
    a = np.abs(np.diff(q, n=2, axis=0)) * FPS ** 2
    return {
        "limit_ok": bool((q >= lo).all() and (q <= hi).all()),
        "vel_ok": bool((v.max(0) <= np.deg2rad(VMAX_DEG) + 1e-9).all()),
        "acc_ok": bool((a.max(0) <= np.deg2rad(AMAX_DEG) + 1e-9).all()),
        "v_max_deg": np.rad2deg(v.max(0)),
        "a_max_deg": np.rad2deg(a.max(0)),
        "q_lo_deg": np.rad2deg(q.min(0)), "q_hi_deg": np.rad2deg(q.max(0)),
    }


def inertia_span(m, cmd: np.ndarray, j: int) -> tuple[float, float]:
    d, M = mujoco.MjData(m), np.zeros((m.nv, m.nv))
    I = np.array([link_inertia(m, d, M, q)[j] for q in cmd[::10, :6]])
    return float(I.min()), float(I.max())
