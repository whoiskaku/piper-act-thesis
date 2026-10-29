#!/usr/bin/env python3
"""Ĝ 参考实现 —— 单文件通读版（学习 / 文档 / 论文附录用；**非生产路径**）。

生产路径散在 6 个文件里（历史演进所致），不便通读：
    层① 伺服FIR   bakeoff_g.py:fit_c_enc
    层② 刚体雅可比 g_verdict.py:jr_at + fit_g_production.py:rigid_parts
    层③ 柔性模态   fit_g_level2.py:FlexL2 + fit_flex
    集成          fit_g_ensemble.py
    快速推理      train_pi_e2.py:GEnsemble/apply_G
本文件把它们按"读者顺序"重排为一份自洽实现，并用 §7 自检**逐点锁死**与生产路径
一致（判据 <1%）。所以它可以放心当教材读：若哪天两边漂移，自检会立刻失败。

────────────────────────────────────────────────────────────────────────
Ĝ 是什么：一台"纸上机器人"。
    输入  = 发给机器人的指令序列 a(t)（6关节，rad，20Hz），可叠加修正 δa
    输出  = 这串指令打在真机上，IMU 陀螺会读到的角速度（rad/s，3轴）
    用途  = 离线回答"如果换个修法，后果会怎样"（真机实时回答不了的反事实）

三层结构（并联，非串联）：
    ① 伺服 FIR      指令 → 真实关节运动（延迟 ~100ms + 低通钝化）  60 参数，闭式最小二乘
    ② 刚体雅可比    关节运动 → IMU 刚体角速度（纯几何）            0 参数，解析
    ③ 柔性模态      指令加速度敲击结构 → 共振余振（位形调度响度）   75 参数，600步 Adam
    输出 = ②的刚体项 + ③的柔性项           （①的输出同时喂给②和③）

两个使用场合：
    场合一 forward()        δa≡0，模拟裸 ACT 的执行           → 与实测陀螺对账（验收 Ĝ）
    场合二 increment()      给 δa，只返回"多加这点修正引起的变化" → π 的训练场
    场合二有两种算法，必须逐点相等（今日 bug 的教训所在）：
        increment_ref()   定义式：Ĝ(a+δa) − Ĝ(a)，慢但权威
        increment_fast()  核式：数值探针取脉冲响应 + 三通路卷积，快，供 π 百万次调用

坐标与索引约定：
    · 内部一律在 **sim 坐标系** 计算，出口用标定旋转 R 转到真机 IMU 系（real = R·sim）
    · 本文件一律用 **墙钟索引**。生产路径 train_pi_e2.apply_G 会在最后左移
      LSHIFT=3 帧转成 e-索引（因 e(i)=real[i+3]−ideal[i] 自带滞后补偿），
      那是下游的记账约定，不属于 Ĝ 本身。

用法：
    python sim/g_model.py            # §7 自检（与生产路径对账）
    python sim/g_model.py --demo     # 画一记 δa 脉冲的三通路响应
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
FPS, DT = 20.0, 0.05
NTAP = 10                      # 层①抽头数 = 0.5s 记忆
NMODE, KKER = 3, 60            # 层③模态数 / 谐振核长度(3s)
LNEG, LPOS = 4, 16             # 脉冲响应核的负/正滞后格数（见 §5 注）
BAND = (0.5, 9.0)              # 柔性层监督频带（Hz）


# ══════════════════════════════════════════════════════════════════════
# §0  小工具：因果 FIR 与带负滞后的 FIR
# ══════════════════════════════════════════════════════════════════════
def fir(x: np.ndarray, kern: np.ndarray, nfut: int = 0) -> np.ndarray:
    """x(T,C) 逐通道卷积 kern(C,L)；kern 第 l 格代表滞后 (l − nfut)。

    nfut=0 即普通因果 FIR：y[t] = Σ_l k[l]·x[t−l]
    nfut>0 允许核有"提前"分量（层③的激励核确实有，见 §3 注）。
    """
    T = len(x)
    y = np.zeros_like(x)
    for l in range(kern.shape[1]):
        lag = l - nfut
        if lag >= 0:
            y[lag:] += kern[:, l][None, :] * x[:T - lag]
        else:                                   # 负滞后 = 用未来的输入
            y[:T + lag] += kern[:, l][None, :] * x[-lag:]
    return y


def d_dt(x: np.ndarray) -> np.ndarray:
    """后向差分求导（与生产路径一致：ẋ[t] = (x[t]−x[t−1])/DT，首帧补 0）。"""
    return np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)]) / DT


def softplus(z):
    return np.logaddexp(0.0, z)                 # 数值稳定版 log(1+e^z)


# ══════════════════════════════════════════════════════════════════════
# §1  层① 伺服 FIR：指令 → 真实关节运动
# ══════════════════════════════════════════════════════════════════════
# 物理：伺服不会瞬间到位——有延迟(~100ms)、有低通钝化、各关节脾气不同。
# 建模：每关节一条 10 抽头 FIR，把"指令速度"映射到"实测关节速度"。
# 监督：编码器（唯一有逐帧真值的一层，拟合后相关 0.99）。
# 求解：闭式最小二乘，无迭代——把 10 个滞后拷贝排成设计矩阵解一次即可。
def fit_servo_fir(sessions: list[dict], ntap: int = NTAP) -> np.ndarray:
    """sessions: [{"cmd": (T,6) 指令角, "enc": (T,6) 编码器角}]  →  C(6, ntap)"""
    C = np.zeros((6, ntap), np.float32)
    for j in range(6):
        Xs, ys = [], []
        for s in sessions:
            ad = d_dt(s["cmd"])[:, j]           # 指令速度（输入，无噪声）
            qd = d_dt(s["enc"])[:, j]           # 编码器速度（目标，有噪声）
            T = len(ad)
            X = np.zeros((T, ntap), np.float32)
            for k in range(ntap):
                X[k:, k] = ad[:T - k]           # 第 k 列 = 指令速度延迟 k 帧
            Xs.append(X)
            ys.append(qd)
        C[j] = np.linalg.lstsq(np.vstack(Xs), np.concatenate(ys), rcond=None)[0]
    return C
    # 注：目标(编码器)有噪声不影响斜率无偏；输入(指令)无噪声，故不存在回归衰减偏差。


# ══════════════════════════════════════════════════════════════════════
# §2  层② 刚体雅可比：关节运动 → IMU 刚体角速度
# ══════════════════════════════════════════════════════════════════════
# 纯几何，零拟合、零参数、不可能错。J 随姿态变（手臂伸直 vs 收拢，同样转一度
# IMU 划的弧完全不同）——这正是"只喂 δa 差分不足以定后果"的数学原因。
class Rigid:
    def __init__(self):
        import mujoco
        from ideal_imu_kinematic import MODEL_PATH
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")

    def jacobian(self, q6: np.ndarray) -> np.ndarray:
        """姿态 q6 → site 系旋转雅可比 (3,6)：各关节转 1rad/s，IMU 感受多少角速度。"""
        from ideal_imu_kinematic import full_qpos
        self.data.qpos[:] = full_qpos(np.concatenate([q6, [0.0]]))
        self.mj.mj_fwdPosition(self.model, self.data)   # 只做正运动学，不积分动力学
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        self.mj.mj_jacSite(self.model, self.data, jacp, jacr, self.sid)
        R_site = self.data.site_xmat[self.sid].reshape(3, 3)
        return (R_site.T @ jacr[:, :6]).astype(np.float32)

    def series(self, cmd: np.ndarray) -> np.ndarray:
        """逐帧雅可比 (T,3,6)。姿态取 **2 帧前的指令** ≈ 当下真实姿态（伺服滞后）。"""
        return np.stack([self.jacobian(cmd[max(t - 2, 0)]) for t in range(len(cmd))])


# ══════════════════════════════════════════════════════════════════════
# §3  层③ 柔性模态：加速度敲击结构 → 共振余振
# ══════════════════════════════════════════════════════════════════════
def modal_impulse(omega, zeta, ktap: int = KKER, dt: float = DT):
    """二阶欠阻尼振子的**解析**冲激响应 (M, ktap)：h[n]=e^{−ζωn·dt}·sin(ω_d n·dt)/ω_d·dt

    对 ω、ζ 可微 ⇒ "共振频率"本身是可训练参数（拟合出 3.68Hz ≈ 实测 3.7Hz）。
    """
    n = torch.arange(ktap, dtype=torch.float32)[None, :]
    om, ze = omega[:, None], zeta[:, None]
    om_d = om * torch.sqrt(torch.clamp(1 - ze ** 2, min=1e-4))
    return torch.exp(-ze * om * n * dt) * torch.sin(om_d * n * dt) / om_d * dt


def cmd_accel(cmd: np.ndarray, hi: float = 200.0) -> np.ndarray:
    """指令的 20Hz 精确二阶导：样条升 200Hz → 解析求导 → 抗混叠 → 降采。

    **不用二次差分**：差分把噪声按 ω² 放大且严重混叠。（这里的 filtfilt 是零相位
    滤波，会把约 5% 能量摊到脉冲之前 —— 这就是 §5 的核需要 LNEG 个负滞后格的原因。）
    """
    T = len(cmd)
    t20 = np.arange(T) / FPS
    sp = CubicSpline(t20, cmd, axis=0)
    th = np.arange(int(round(t20[-1] * hi)) + 1) / hi
    acc = sp(th, 2)
    ba = butter(4, 0.45 * FPS / (hi / 2))
    return filtfilt(*ba, acc, axis=0)[::int(hi // FPS)][:T].astype(np.float32)


def excitation(cmd: np.ndarray, C: np.ndarray) -> np.ndarray:
    """结构感受到的"锤击" u = 伺服FIR ⊛ 指令加速度。(T,6)

    物理：结构被晃起来靠的是**加速度**（匀速运动不激起振动，急停会）。
    """
    return fir(cmd_accel(cmd), C)


def pose_features(cmd7: np.ndarray) -> np.ndarray:
    """φ(t) 的 8 项 (T,8)：全部从指令算出，不用任何传感器。

    [1, q2, q3, q2², q3², q2·q3, 速度, 夹爪]
    肩肘的一次+二次+交叉项 = 表达"悬臂伸展度"的最小二次曲面（残差画像确诊：
    位形是响度差 2.3× 的主因，载体是 q2/q3）。速度：快动激励更强。夹爪：末端质量。
    """
    q2, q3 = cmd7[:, 1], cmd7[:, 2]
    spd = np.convolve(np.linalg.norm(d_dt(cmd7[:, :6]), axis=1), np.ones(5) / 5, "same")
    grip = np.clip(cmd7[:, 6] / 0.035, -2, 2) if cmd7.shape[1] > 6 else np.zeros(len(cmd7))
    return np.stack([np.ones(len(cmd7)), q2, q3, q2 ** 2, q3 ** 2, q2 * q3,
                     spd, grip], 1).astype(np.float32)


class FlexLayer(nn.Module):
    """75 个参数，每个都有物理名字。参数化技巧见各行注释。"""

    def __init__(self, f_init=(3.1, 5.1, 5.7), nfeat: int = 8, M: int = NMODE):
        super().__init__()
        # log 存频率：exp() 保证 ω>0 恒成立（无需约束），且 log 空间梯度=乘性更新
        self.log_om = nn.Parameter(torch.log(torch.tensor(f_init, dtype=torch.float32) * 2 * np.pi))
        # sigmoid×0.5 把阻尼锁进 (0, 0.5)：**永远欠阻尼、永远稳定**，梯度怎么走都不发散
        self.raw_ze = nn.Parameter(torch.full((M,), -2.0))     # → ζ≈0.06 轻阻尼，会响铃
        self.W = nn.Parameter(torch.randn(M, 6) * 0.1)         # 关节激励 → 模态（谁敲响谁）
        self.B = nn.Parameter(torch.randn(M, 3) * 0.1)         # 模态 → 陀螺轴（哪个铃摇哪轴）
        self.D = nn.Parameter(torch.zeros(3, 6))               # 直通：瞬时刚性响应，不经谐振
        th = torch.zeros(M, nfeat)
        th[:, 0] = 0.5413                                      # softplus(0.5413)=1.0 精确
        self.theta = nn.Parameter(th)                          # → 开局"无调度"，让数据决定

    def gains(self, phi):
        """位形调度的响度 α(t) (T,M)。"""
        return torch.nn.functional.softplus(phi @ self.theta.T)

    def forward(self, u, phi):                                 # u(T,6), phi(T,8) → (T,3)
        ker = modal_impulse(torch.exp(self.log_om), torch.sigmoid(self.raw_ze) * 0.5)
        up = torch.einsum("mj,tj->tm", self.W, u) * self.gains(phi)
        #  ↑ α 乘在**输入侧**（进振子之前）：姿态调制的是"这一锤敲得多重"。
        #    α 的这个位置是模型定义的一部分 —— 把它挪走就是另一个模型（生产路径
        #    曾把激励核与谐振核预乘成单核、迫使 α 退到最前，实测偏差 80–108%）。
        x = up.T[None]                                          # (1,M,T)
        s = torch.nn.functional.conv1d(
            torch.nn.functional.pad(x, (ker.shape[1] - 1, 0)),  # 只左补零 ⇒ **因果**
            ker.flip(-1)[:, None, :],                           # flip ⇒ 真卷积（非互相关）
            groups=up.shape[1])[0].T                            # groups ⇒ 模态间不串味
        return s @ self.B + u @ self.D.T                        # 铃声项 + 直通项


# ══════════════════════════════════════════════════════════════════════
# §4  层③的训练
# ══════════════════════════════════════════════════════════════════════
def bandpass(x: np.ndarray) -> np.ndarray:
    ba = butter(3, [BAND[0] / (FPS / 2), BAND[1] / (FPS / 2)], "bandpass")
    return filtfilt(*ba, x, axis=0).astype(np.float32)


def fit_flex(flex, us, targets, phis, masks, iters: int = 600, lr: float = 1e-2):
    """全批量 Adam。75 参数 / 1.7 万帧全在内存，每次迭代看完所有数据走一步。

    监督对象 = **残差**（层①②解释不掉的部分），带通 0.5–9Hz，接触帧掩码剔除：
      · 减刚体：柔性层只对"剩下的震铃"负责，防它去抢刚体层的活（可分离对账）
      · 带通：柔性只拥有共振频段；低频归伺服、高频是传感噪声
      · 剔接触：碰撞是**外力**敲的铃，不是指令敲的；留着会教它"看到指令就幻想碰撞"
    """
    opt = torch.optim.Adam(flex.parameters(), lr=lr)
    for _ in range(iters):
        loss = 0.0
        for u, r, ph, m in zip(us, targets, phis, masks):
            d = flex(u, ph) - r
            loss = loss + (d[m] ** 2).mean()   # 逐 episode 取均值再求和 ⇒ 每集等权
        opt.zero_grad()
        loss.backward()
        opt.step()
    return flex


# ══════════════════════════════════════════════════════════════════════
# §5  整机：三层合体 + 两个使用场合
# ══════════════════════════════════════════════════════════════════════
class GModel:
    """一个成员的 Ĝ。参数 = C(60) + FlexLayer(75) + 标定 R + φ标准化统计。"""

    def __init__(self, C, flex, mu, sd, R, rigid: Rigid | None = None):
        self.C, self.flex = C.astype(np.float32), flex
        self.mu, self.sd, self.R = mu, sd, R.astype(np.float32)
        self.rigid = rigid or Rigid()
        self._vker = self._exck = None

    # ---------- 场合一：cmd → 模拟 IMU ----------
    def forward(self, cmd7: np.ndarray, J=None, phi=None) -> np.ndarray:
        """(T,7) 指令 → (T,3) 模拟陀螺（真机系）。δa≡0 时即"模拟裸 ACT 的执行"。

        J / phi 可外部传入以**冻结调度**（场合二需要；见 increment_ref）。
        """
        cmd = cmd7[:, :6].astype(np.float32)
        qd = fir(d_dt(cmd), self.C)                            # ① 指令 → 关节速度
        if J is None:
            J = self.rigid.series(cmd)                         # ② 逐帧几何
        rig = np.einsum("tcj,tj->tc", J, qd)                   #    刚体角速度
        u = excitation(cmd, self.C)                            # ③ 结构受到的锤击
        if phi is None:
            phi = (pose_features(cmd7) - self.mu) / self.sd
        with torch.no_grad():
            flx = self.flex(torch.from_numpy(u),
                            torch.from_numpy(phi.astype(np.float32))).numpy()
        return ((rig + flx) @ self.R.T).astype(np.float32)     # sim → 真机系

    def context(self, cmd7: np.ndarray):
        """名义轨迹烘焙出的**冻结调度**：几何地图 J(t) 与响度地图 φ(t)。

        这就是"名义 command 在哪里进入模型"的答案（老师面談①的关切）：
        同一个 δa，手臂伸直 vs 收拢，J 与 α 不同 ⇒ 后果不同，绝不会混淆。
        """
        J = self.rigid.series(cmd7[:, :6].astype(np.float32))
        phi = ((pose_features(cmd7) - self.mu) / self.sd).astype(np.float32)
        return J, phi

    # ---------- 场合二 · 定义式（慢，权威） ----------
    def increment_ref(self, cmd7: np.ndarray, da: np.ndarray) -> np.ndarray:
        """Ĝ(a+δa) − Ĝ(a)，调度冻结在名义轨迹。(T,6)→(T,3)

        冻结的理由：δa 是 mrad 级微扰，对 rad 级姿态的影响 ~0.1%；冻结换来的是
        **扰动通道严格线性**（δa 翻倍 ⇒ Δe 精确翻倍），这是辨识桥梁成立的前提。
        """
        J, phi = self.context(cmd7)
        pert = cmd7.copy()
        pert[:, :6] = pert[:, :6] + da
        return self.forward(pert, J, phi) - self.forward(cmd7, J, phi)

    # ---------- 场合二 · 核式（快，供 π 百万次调用） ----------
    def kernels(self):
        """数值探针取脉冲响应核。**不解析推导** —— 今日 bug 的教训：

        手推的二次差分核比样条精确二阶导**晚 1 帧**、且丢掉 filtfilt 的非因果部分。
        而 Ĝ 对指令是线性算子 ⇒ 喂一个单位脉冲，出来的就是精确核，没有推导误差余地。
        """
        if self._vker is None:
            N, T0 = 96, 48
            vk = np.zeros((6, LNEG + LPOS), np.float32)
            ek = np.zeros((6, LNEG + LPOS), np.float32)
            for j in range(6):
                imp = np.zeros((N, 6), np.float32)
                imp[T0, j] = 1.0                               # 单位角度脉冲
                vk[j] = fir(d_dt(imp), self.C)[T0 - LNEG:T0 + LPOS, j]   # → 关节速度
                ek[j] = excitation(imp, self.C)[T0 - LNEG:T0 + LPOS, j]  # → 结构激励
            self._vker, self._exck = vk, ek
        return self._vker, self._exck

    def increment_fast(self, cmd7: np.ndarray, da: np.ndarray,
                       ctx=None) -> np.ndarray:
        """三通路卷积。必须与 increment_ref 逐点相等（§7 自检）。

            刚体  Δω = J(t)·(vker ⊛ δa)
            模态  Δω = B·(km ⊛ [α(t)·(W·(exck ⊛ δa))])   ← α 夹在两次卷积之间
            直通  Δω = D·(exck ⊛ δa)                      ← 不经谐振、不受调度
        """
        J, phi = ctx or self.context(cmd7)
        vker, exck = self.kernels()
        da = da.astype(np.float32)
        qd = fir(da, vker, LNEG)                               # 修正引起的关节速度
        ud = fir(da, exck, LNEG)                               # 修正引起的激励
        rig = np.einsum("tcj,tj->tc", J, qd)
        with torch.no_grad():
            W = self.flex.W.numpy()
            D = self.flex.D.numpy()
            al = self.flex.gains(torch.from_numpy(phi)).numpy()
            km = modal_impulse(torch.exp(self.flex.log_om),
                               torch.sigmoid(self.flex.raw_ze) * 0.5).numpy()
            B = self.flex.B.numpy()
        mi = (ud @ W.T) * al                                   # 模态激励 × 响度
        s = fir(mi, km, 0)                                     # 谐振（严格因果：铃不提前响）
        return ((rig + s @ B + ud @ D.T) @ self.R.T).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# §6  集成：K=5 份 Ĝ（防 π 钻单一模型的空子）
# ══════════════════════════════════════════════════════════════════════
def load_ensemble(npz=None, rigid: Rigid | None = None) -> list[GModel]:
    """从生产路径产出的 g_ensemble.npz 读回 5 个成员。

    成员 0 = 全量 16 集；成员 1-4 = **按集**有放回重采样（集内帧强相关，按帧采
    会假装样本量很大；按集才是诚实的自助法）。成员间分歧 = 模型自报的不确定度，
    π 训练时"均值训练 / 最差成员评估"的原料。
    """
    z = np.load(npz or (ROOT / "outputs/p2_bakeoff/g_ensemble.npz"))
    rigid = rigid or Rigid()
    out = []
    for k in range(len(z["c_enc"])):
        flex = FlexLayer()
        with torch.no_grad():
            for n in ["log_om", "raw_ze", "W", "B", "D", "theta"]:
                getattr(flex, n).copy_(torch.from_numpy(z[n][k]))
        out.append(GModel(z["c_enc"][k], flex, z["mu"][k], z["sd"][k],
                          np.eye(3, dtype=np.float32), rigid))   # R 由调用方按会话设置
    return out


# ══════════════════════════════════════════════════════════════════════
# §7  自检：与生产路径逐点对账（本文件可信度的唯一凭据）
# ══════════════════════════════════════════════════════════════════════
def _session():
    """取一条留出集 episode 的指令与标定（不需联网/HF）。"""
    import json
    tag, ep = "pp2-deploy", 8
    d = ROOT / "outputs/p2_labels" / tag
    z = np.load(d / f"ep{ep:03d}.npz")
    R = np.array(json.loads((d / "manifest.json").read_text())["R"], np.float32)
    cmd7 = np.concatenate([z["a_arm"], z["grip"][:, None]], 1).astype(np.float32)
    return cmd7, R


def self_check() -> int:
    import train_pi_e2 as TP
    from fit_g_level2 import FlexL2
    from fit_g_production import rigid_parts

    cmd7, R = _session()
    models = load_ensemble()
    z = np.load(ROOT / "outputs/p2_bakeoff/g_ensemble.npz")
    worst = 0.0
    rel = lambda a, b: float(np.sqrt(((a - b) ** 2).mean()) / max(np.sqrt((b ** 2).mean()), 1e-12))

    print("① forward()  vs  生产路径(rigid_parts + FlexL2)")
    for k in (0, 2, 4):
        g = models[k]
        g.R = R
        mine = g.forward(cmd7)
        prod_flex = FlexL2([3.1, 5.1, 5.7])
        with torch.no_grad():
            for n in ["log_om", "raw_ze", "W", "B", "D", "theta"]:
                getattr(prod_flex, n).copy_(torch.from_numpy(z[n][k]))
        _, _, w_rig, _ = rigid_parts({"a": cmd7[:, :6]}, z["c_enc"][k])
        from fit_g_level2 import excitation as prod_exc, raw_feats as prod_feats
        with torch.no_grad():
            fl = prod_flex(torch.from_numpy(prod_exc(cmd7[:, :6], z["c_enc"][k])),
                           torch.from_numpy(((prod_feats(cmd7) - z["mu"][k]) / z["sd"][k]
                                             ).astype(np.float32))).numpy()
        e = rel(mine, (w_rig + fl) @ R.T)
        worst = max(worst, e)
        print(f"   成员{k}: 相对误差 {e*100:.4f}%")

    print("② increment_fast()  vs  increment_ref()（核式 vs 定义式）")
    rng = np.random.default_rng(0)
    da = np.zeros((len(cmd7), 6), np.float32)
    da[200:400] = rng.normal(0, 0.003, (200, 6))               # 连续随机修正，非单脉冲
    for k in (0, 2, 4):
        g = models[k]
        g.R = R
        ctx = g.context(cmd7)
        e = rel(g.increment_fast(cmd7, da, ctx), g.increment_ref(cmd7, da))
        worst = max(worst, e)
        print(f"   成员{k}: 相对误差 {e*100:.4f}%")

    print("③ increment_fast()  vs  生产路径 train_pi_e2.apply_G（去掉 e-索引移位）")
    G = TP.GEnsemble()
    zl = np.load(ROOT / "outputs/p2_labels/pp2-deploy/ep008.npz")
    epJ, alpha, RB, RD = G.ep_context(zl["a_arm"], zl["grip"], R)
    dt_ = torch.from_numpy(da).unsqueeze(0).to(TP.DEV)
    with torch.no_grad():
        prod = TP.apply_G(dt_, epJ.unsqueeze(0), alpha.unsqueeze(0),
                          RB.unsqueeze(0), RD.unsqueeze(0), G)[0].cpu().numpy()
    for k in (0, 2, 4):
        g = models[k]
        g.R = R
        mine = g.increment_fast(cmd7, da)
        shifted = np.roll(mine, -TP.LSHIFT, 0)                 # 墙钟 → e-索引
        sl = slice(180, 420)
        e = rel(shifted[sl], prod[k][sl])
        worst = max(worst, e)
        print(f"   成员{k}: 相对误差 {e*100:.4f}%")

    ok = worst < 0.01
    print(f"\n最坏相对误差 = {worst*100:.4f}%   判决: {'✅ PASS' if ok else '❌ FAIL'}")
    return 0 if ok else 1


def demo():
    """画一记 δa 脉冲的三通路分解（教学用）。"""
    import matplotlib.pyplot as plt
    cmd7, R = _session()
    g = load_ensemble()[0]
    g.R = R
    tau, jj = 200, 1
    da = np.zeros((len(cmd7), 6), np.float32)
    da[tau, jj] = 0.005
    J, phi = g.context(cmd7)
    vker, exck = g.kernels()
    qd, ud = fir(da, vker, LNEG), fir(da, exck, LNEG)
    with torch.no_grad():
        al = g.flex.gains(torch.from_numpy(phi)).numpy()
        km = modal_impulse(torch.exp(g.flex.log_om),
                           torch.sigmoid(g.flex.raw_ze) * 0.5).numpy()
        W, B, D = g.flex.W.numpy(), g.flex.B.numpy(), g.flex.D.numpy()
    rig = np.einsum("tcj,tj->tc", J, qd) @ R.T
    flx = (fir((ud @ W.T) * al, km, 0) @ B) @ R.T
    dirc = (ud @ D.T) @ R.T
    sl = slice(tau - 5, tau + 20)
    t = (np.arange(sl.start, sl.stop) - tau) * 50.0
    ch = int(np.argmax(np.abs(rig[sl] + flx[sl] + dirc[sl]).max(0)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t, (rig + flx + dirc)[sl, ch] * 1000, "k", lw=2.2, label="total $\\Delta\\omega$")
    ax.plot(t, rig[sl, ch] * 1000, lw=1.4, label="rigid  $J\\cdot(vker*\\delta a)$")
    ax.plot(t, flx[sl, ch] * 1000, lw=1.4, label="modal ringing")
    ax.plot(t, dirc[sl, ch] * 1000, lw=1.4, label="direct feedthrough $D$")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("wall-clock time since pulse (ms)")
    ax.set_ylabel("$\\Delta\\omega$ (mrad/s)")
    ax.set_title(f"$\\hat{{G}}$ response to a +5 mrad pulse on j{jj+1} "
                 f"— three paths (gyro-{'xyz'[ch]})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "reports/figures/G_three_paths.png"
    fig.savefig(out, dpi=160)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    raise SystemExit(demo() if a.demo else self_check())
