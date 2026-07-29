"""B 校正模块运行时 — 训练/部署共用的单一真源(2026-07-14)。

包含: TCN 骨干 · 理想chunk渲染器(运动学, 源自 sim/ideal_imu_kinematic.py) ·
FF1/FF2(源自 sim/correction_ff.py, FF3 已弃) · 因果特征构造(离线 build_offline 与
流式 FeatureAssembler 共用 row_features — 一致性靠构造保证)。

因果口径(部署强制, 训练同步 — 修正 2026-07-14 发现的 L 帧泄漏):
  命令帧 i 时刻可知: ideal[0..chunk内], real 墙钟 ≤ i → e_full[j]=real[j+L]−ideal_cal[j]
  只有 j ≤ i−L 可用。特征 e 窗以 e_full[i−L] 结尾; gate 用墙钟≤i 的 real gyro 因果均值;
  夹爪归一化用固定尺度 ÷0.035(每集最大值部署不可知)。
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FPS, DT, CHUNK = 20.0, 0.05, 100
EPS_B = 0.010
FEAT_DIM, WIN = 51, 16          # 51 = 45 + q̇(6)(2026-07-15 用户批准); TCN 感受野15 → 流式窗16
GRIP_SCALE = 0.035
GATE_TH = 0.04                  # rad/s, 与 wall_portrait still 判据一致
_MJCF = Path(__file__).resolve().parents[4] / "sim/piper_model.xml"


# ---------------- 骨干 (bake-off 赢家, 配置勿改) ----------------
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 6)

    def forward(self, h):
        return torch.tanh(self.lin(h)) * EPS_B


class TCN(nn.Module):
    def __init__(self, d=FEAT_DIM, h=32):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(d, h, 3, dilation=1), nn.Conv1d(h, h, 3, dilation=2),
            nn.Conv1d(h, h, 3, dilation=4)])
        self.pads = [2, 4, 8]
        self.head = Head(h)

    def forward(self, u):                                  # (B,T,45)
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = torch.nn.functional.gelu(c(torch.nn.functional.pad(x, (p, 0))))
        return self.head(x.transpose(1, 2))


def load_tcn(path: str | Path) -> TCN:
    m = TCN()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


# ---------------- 理想 chunk 渲染器 (运动学, 零标定) ----------------
class IdealChunkRenderer:
    """指令(T,7) → 20Hz 抗混叠理想 IMU(T,6)。带左侧指令上下文以贴近全程渲染口径。"""

    def __init__(self, mjcf: str | Path = _MJCF, hi_fps: float = 200.0, ctx: int = 8):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
        self.hi = hi_fps
        self.ctx = ctx
        # render 用 imu[::round(hi/FPS)] 降采样 —— hi/FPS 必须**整除**, 否则输出的实际
        # 采样率是 hi/round(hi/FPS) 而非 FPS, 时基逐帧漂移(2026-07-29 在 30 Hz 复用时
        # 踩到: 200/30→stride 7 ⇒ 28.57 Hz, 800 帧漂 ~38 帧, 滞后估计完全失效)。
        # 生产口径 200/20=10 一直正确; 这里只是把静默失败变成显式失败。
        if abs(hi_fps / FPS - round(hi_fps / FPS)) > 1e-9:
            raise ValueError(f"hi_fps({hi_fps}) 必须是 FPS({FPS}) 的整数倍, "
                             f"否则降采样时基漂移; 30 Hz 请用 240")
        from scipy.signal import butter
        self._ba = butter(4, 0.45 * FPS / (hi_fps / 2))

    @staticmethod
    def _full_qpos(q7):
        out = np.empty(8)
        out[:6] = q7[:6]
        g = float(np.clip(q7[6], 0.0, 0.035))
        out[6], out[7] = g, -g
        return out

    @staticmethod
    def _full_qvel(qd7):
        out = np.empty(8)
        out[:6] = qd7[:6]
        out[6], out[7] = qd7[6], -qd7[6]
        return out

    def render(self, cmd: np.ndarray, prev_cmd_tail: np.ndarray | None = None) -> np.ndarray:
        from scipy.interpolate import CubicSpline
        from scipy.signal import filtfilt
        mj = self._mj
        nctx = 0
        seq = cmd
        if prev_cmd_tail is not None and len(prev_cmd_tail) >= 2:
            nctx = min(self.ctx, len(prev_cmd_tail))
            seq = np.concatenate([prev_cmd_tail[-nctx:], cmd])
        T = len(seq)
        t20 = np.arange(T) / FPS
        sp = CubicSpline(t20, seq, axis=0)
        stride = int(round(self.hi / FPS))
        t_hi = np.arange(int(round(t20[-1] * self.hi)) + 1) / self.hi
        imu = np.zeros((len(t_hi), 6))
        q, qd, qdd = sp(t_hi), sp(t_hi, 1), sp(t_hi, 2)
        for k in range(len(t_hi)):
            self.data.qpos[:] = self._full_qpos(q[k])
            self.data.qvel[:] = self._full_qvel(qd[k])
            mj.mj_fwdPosition(self.model, self.data)
            mj.mj_fwdVelocity(self.model, self.data)
            self.data.qacc[:] = self._full_qvel(qdd[k])
            mj.mj_rnePostConstraint(self.model, self.data)
            mj.mj_sensorPos(self.model, self.data)
            mj.mj_sensorVel(self.model, self.data)
            mj.mj_sensorAcc(self.model, self.data)
            imu[k] = self.data.sensordata[:6]
        imu = filtfilt(*self._ba, imu, axis=0)[::stride][:T]
        return imu[nctx:]                                  # 去掉上下文段


# ---------------- FF1/FF2 (确定性前馈, 可选层) ----------------
def apply_ff12(chunk: np.ndarray, prev_tail_sent: np.ndarray | None,
               lag: int = 2, blend: int = 4) -> np.ndarray:
    out = chunk.copy()
    ARM = slice(0, 6)
    if blend and prev_tail_sent is not None and len(prev_tail_sent) >= 2:
        base = prev_tail_sent[-1, ARM]
        vel = prev_tail_sent[-1, ARM] - prev_tail_sent[-2, ARM]
        for k in range(min(blend, len(out))):
            alpha = 0.5 * (1 - np.cos(np.pi * (k + 1) / (blend + 1)))
            out[k, ARM] = alpha * out[k, ARM] + (1 - alpha) * (base + vel * (k + 1))
    if lag > 0:
        out[:, ARM] = np.vstack([out[lag:, ARM], np.repeat(out[-1:, ARM], lag, axis=0)])
    return out


# ---------------- 因果特征: 共用行构造 ----------------
def row_features(i: int, e_full, ideal, a_arm, qd, grip, gyr_norm, L: int, cap: int) -> np.ndarray:
    """命令帧 i 的 51 维特征。e_full/gyr_norm/qd 仅可用到 i−L / i(墙钟因果)。
    布局: [e(6) | ideal(t),(t+4),(t+8) (18) | a(t),(t+4),(t+8) (18) | q̇(6) | phase(3)]"""
    def ev(j):
        return e_full[j] if 0 <= j <= i - L and j < len(e_full) else np.zeros(6, np.float32)
    j4, j8 = min(i + 4, cap), min(i + 8, cap)
    gate_src = [gyr_norm[k] for k in range(max(0, i - L - 4), max(0, i - L + 1))]
    gate = float(np.mean(gate_src)) if gate_src else 0.0
    phase = np.array([(i % CHUNK) / CHUNK,
                      float(np.clip(grip[i] / GRIP_SCALE, -2, 2)),
                      gate], np.float32)
    return np.concatenate([ev(i - L), ideal[i], ideal[j4], ideal[j8],
                           a_arm[i], a_arm[j4], a_arm[j8], qd[i], phase]).astype(np.float32)


def build_offline(z, L: int = 3):
    """从标签 npz 构造因果特征(训练用)。返回 X(n,51), y, active_mask。需 npz 含 qw。"""
    e, ideal, a, grip = z["e"], z["ideal"], z["a_arm"], z["grip"]
    ra, qw = z["real"], z["qw"]
    n = len(e)
    gyr_norm = np.linalg.norm(ra[:, 3:], axis=1)           # gyr_norm[j] = real 墙钟 j+L
    qd = np.vstack([np.zeros((1, 6), np.float32),
                    np.diff(qw, axis=0).astype(np.float32)]) / DT   # 编码器速度(墙钟 i 可得)
    idx = np.arange(n)
    caps = np.minimum((idx // CHUNK) * CHUNK + CHUNK - 1, n - 1)
    X = np.stack([row_features(i, e, ideal, a, qd, grip, gyr_norm, L, int(caps[i]))
                  for i in range(n)])
    return X, z["label"].astype(np.float32), ~z["mask_still"]


class FeatureAssembler:
    """部署流式装配器 — 与 build_offline 共用 row_features, 一致性靠构造。"""

    def __init__(self, R: np.ndarray, b: np.ndarray, L: int = 3):
        self.R, self.b, self.L = R.astype(np.float32), b.astype(np.float32), L
        self.reset()

    def reset(self):
        self.ideal, self.a_arm, self.grip = [], [], []
        self.e_full, self.gyr_norm = [], []
        self.real_wall, self.qw = [], []
        self.i = 0
        self.cap = -1
        self.rows = deque(maxlen=WIN)

    def on_new_chunk(self, cmd: np.ndarray, ideal_raw: np.ndarray):
        """cmd(T,7)=本chunk将发送的指令; ideal_raw=渲染器输出(sim系)。"""
        ic = np.empty_like(ideal_raw)
        ic[:, :3] = ideal_raw[:, :3] @ self.R.T + self.b
        ic[:, 3:] = ideal_raw[:, 3:] @ self.R.T
        self.ideal.extend(np.asarray(ic, np.float32))
        self.a_arm.extend(np.asarray(cmd[:, :6], np.float32))
        self.grip.extend(np.asarray(cmd[:, 6], np.float32))
        self.cap = len(self.a_arm) - 1

    def step(self, imu6: np.ndarray, q6: np.ndarray) -> tuple[np.ndarray, float]:
        """墙钟 i 收到 real IMU + 编码器角 → 返回(特征窗 (WIN,51), gate电平)。之后调用方自增。"""
        i = self.i
        self.real_wall.append(imu6.astype(np.float32))
        self.gyr_norm.append(float(np.linalg.norm(imu6[3:])))
        self.qw.append(q6.astype(np.float32))
        j = i - self.L
        if j >= 0:
            self.e_full.append(self.real_wall[i] - self.ideal[j])
        # gate 索引口径: build_offline 的 gyr_norm[k]=real墙钟k+L → 流式等价序列 = 墙钟移L
        gshift = self.gyr_norm[self.L:] if len(self.gyr_norm) > self.L else []
        qd_i = (self.qw[i] - self.qw[i - 1]) / DT if i >= 1 else np.zeros(6, np.float32)
        qd = {i: qd_i}
        row = row_features(i, self.e_full, self.ideal, self.a_arm, qd, self.grip,
                           gshift, self.L, self.cap)
        self.rows.append(row)
        win = np.zeros((WIN, FEAT_DIM), np.float32)
        got = list(self.rows)
        win[-len(got):] = np.stack(got)
        self.i += 1
        return win, row[-1]
