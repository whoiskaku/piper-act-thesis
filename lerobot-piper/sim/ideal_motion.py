"""老师定义的理想动作，以及实际运动相对它的偏差（用户 11-12 定为主指标；旧的零阶保持指标另报）。

老师 10 月文件：「理想には、Piper の動作の制約のうち、指令の出し方で越えられない加速度の上限を入れましょう。
遅れは…理想には入れず、直すべき誤差とします」。
做法（每个关节）：5 ms 网格上的轨迹 x，|加速度| ≤ A_MAX、|速度| ≤ V_MAX，在每帧的发令时刻尽量正好经过 ACT 指令：
最小化 Σ_k |x(t_k) − a_k| + EPS·Σ|Δ²x|（线性规划）。物理上做得到就精确经过，做不到才偏开；两帧之间取加速度最省的走法。
"""
from __future__ import annotations

import numpy as np

import cmd_tracking as T

A_MAX = 5.0          # rad/s²：固件 S-V1.5-8 各关节最大加速度 500（× 0.01），已是可设的最大值
V_MAX = 3.0          # rad/s：固件各关节最大速度 300（× 0.01）
DT, SUB = 0.05, 10   # 帧长；每帧 10 个网格点
DTG = DT / SUB       # 5 ms 网格
EPS = 1e-3           # 两帧之间少用加速度的权重：远小于经过指令点的权重
TAUS = np.arange(-0.06, 0.1505, 0.002)   # 找「整体晚多少最贴近理想」的时移范围（s）
EARLY_S = 0.024      # Shift 60 ms 和 ILC 教师都比理想晚约 24 ms（39 条教师轨迹离线；真机会话 20261112v 确认）
LEAD_S = 0.084       # 补上这 24 ms 的 Shift 提前量（60 + 24）
ACT = list(T.ACTIVE)


def _joint(c: np.ndarray) -> np.ndarray:
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, diags, eye, hstack, vstack
    K = len(c)
    N = (K - 1) * SUB + 1
    nu = N - 2
    S = csr_matrix((np.ones(K), (np.arange(K), np.arange(K) * SUB)), shape=(K, N))       # 取第 k 帧的网格点
    D1 = diags([-1.0, 1.0], [0, 1], shape=(N - 1, N))
    D2 = diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nu, N))
    Z = lambda r, k: csr_matrix((r, k))                                                     # noqa: E731
    A = vstack([hstack([S, -eye(K), Z(K, nu)]), hstack([-S, -eye(K), Z(K, nu)]),            # |x_k − c_k| ≤ s_k
                hstack([D2, Z(nu, K), -eye(nu)]), hstack([-D2, Z(nu, K), -eye(nu)]),        # |Δ²x| ≤ u ≤ A_MAX·dt²
                hstack([D1, Z(N - 1, K + nu)]), hstack([-D1, Z(N - 1, K + nu)])]).tocsr()  # |Δx| ≤ V_MAX·dt
    b = np.r_[c, -c, np.zeros(2 * nu), np.full(2 * (N - 1), V_MAX * DTG)]
    bounds = [(None, None)] * N + [(0, None)] * K + [(0, A_MAX * DTG * DTG)] * nu
    r = linprog(np.r_[np.zeros(N), np.ones(K), np.full(nu, EPS)], A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(f"理想动作求解失败：{r.message}")
    return r.x[:N]


def earlier(cmd7: np.ndarray, s: float) -> np.ndarray:
    """整条指令提前 s 秒（0 ≤ s ≤ DT）：第 k 帧取 t_k + s 处的线性插值；末帧保持，夹爪不动。teacher24 = earlier(教师, EARLY_S)。"""
    c = np.array(cmd7, float)
    r = s / DT
    c[:-1, :6] = (1 - r) * c[:-1, :6] + r * c[1:, :6]
    return c


def ideal(a7: np.ndarray) -> np.ndarray:
    """(n, ≥6) ACT 指令 → 理想 (N, 6)，N = (n − 1)·SUB + 1；第 i 个网格点在第 0 帧发令时刻之后 i·DTG 秒。"""
    a = np.asarray(a7, float)
    return np.stack([_joint(a[:, j]) for j in range(6)], 1)


class IdealCmd:
    """理想按 `sp(t, nu)` 接口取值（同 shift_replay.LinCmd，给 ideal_imu_kinematic.ideal_imu_traj 用），t 从第 0 帧发令时刻算起：
    位置在网格间线性插值；速度、加速度取所在格子的一阶、二阶差分（与线性插值的位置一致）。返回 7 列，夹爪列为 0。"""

    def __init__(self, x: np.ndarray):
        self.t, self.x = np.arange(len(x)) * DTG, x
        self.v, self.a = np.diff(x, axis=0) / DTG, np.diff(x, 2, axis=0) / DTG ** 2

    def __call__(self, t, nu: int = 0) -> np.ndarray:
        t = np.atleast_1d(t)
        if nu == 0:
            y = np.stack([np.interp(t, self.t, self.x[:, j]) for j in range(6)], 1)
        else:
            d = self.v if nu == 1 else self.a
            y = d[np.clip((t / DTG).astype(int), 0, len(d) - 1)]
        return np.column_stack([y, np.zeros(len(t))])


def nominal(t: np.ndarray, t_send: np.ndarray) -> np.ndarray:
    """回放时刻 → 名义时刻（第 k 帧在 k·DT）：按这一遍各帧的发令时刻分段线性对应。"""
    return np.interp(t, t_send, np.arange(len(t_send)) * DT)


def joint_ideal(ep: dict, x: np.ndarray, free: np.ndarray) -> dict:
    """五个会动关节相对理想的偏差（mrad，free 里的编码器采样，RMS）：dev = 原样比；lag_ms = 实际运动整体晚多少时最贴近理想；
    shape = 去掉这段整体滞后之后的偏差。ep 要有 t_send、tq、q（同 cmd_tracking.from_record）。"""
    cmd, tn, q = IdealCmd(x), nominal(ep["tq"], ep["t_send"]), np.asarray(ep["q"])[:, ACT]
    rms = np.array([np.sqrt(np.mean((q - cmd(tn - tau)[:, ACT])[free] ** 2)) * 1e3 for tau in TAUS])
    i = int(rms.argmin())
    return {"dev": float(rms[np.argmin(np.abs(TAUS))]), "lag_ms": float(TAUS[i] * 1e3), "shape": float(rms[i])}


def gyro_ideal(ep: dict, x: np.ndarray, calib: dict, model, sid: int, tau: float = 0.0) -> np.ndarray:
    """理想经运动学渲染、按标定换到 IMU 坐标的陀螺读数（IMU 采样时刻上，同 shift_replay.raw_gyro）；tau = 理想整体推迟多少秒。"""
    import mujoco
    from ideal_imu_kinematic import ideal_imu_traj
    tn = nominal(np.asarray(ep["ti"]) - calib["delay_s"], ep["t_send"]) - tau
    return (ideal_imu_traj(model, mujoco.MjData(model), IdealCmd(x), tn, sid)[1][:, 3:] @ np.asarray(calib["R"]).T
            + np.asarray(calib["gyro_bias"]))
