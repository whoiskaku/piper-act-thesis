"""B4 的在线量：控制回路里第 k 帧当时就算得出的东西，离线复算也用同一批函数（方案 `reports/PLAN_b4.md` 约束 2）。

- `te_states`：由每帧记下的 ACT 整段预测，逐帧复算 TE 的输出动作与之后各帧的部分融合值。
- `online_shift`：提前 60 ms 的指令 = 第 k+1、k+2 帧部分融合值的线性插值（0.8 / 0.2）。
- `online_window`：第 k 帧的窗（疑似接触、限位饱和处为 0，余弦过渡），只用第 k 帧当时已知的指令。
- `online_input`：网络输入 = 过去 P 帧最终指令 + 未来 F 帧部分融合值。
- 重新生成网络 ②（方案 `reports/PLAN_b4_regen.md`）：`enc_window`、`imu_window`（截止时刻以前的实测窗口）、
  `regen_act`（过去与之后的 ACT 指令）、`regen_tokens`（输入单元）。训练数据和在线回放都只经过这几个函数。
  `regen_command`（按关节在网络与 Shift 之间切换）、`step_check`（逐帧检查，停机条件）、`regen_bounds`（位置界）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTIVE = (0, 1, 2, 4, 5)          # 补正只加在这五个关节（j4、夹爪不动）
DT = 0.05                         # 名义帧长（s），同回放管线
LEAD_S = 0.060                    # Shift 提前量（s）
CLIP = 0.015                      # |Δ| 上限（rad）
HARD_SLACK = 0.035                # 管线硬界 = 模型限位 ± 0.035 rad（同 replay_probe）
BOUND_MARGIN = np.radians(0.2)    # Shift + Δ 离硬界至少 0.2°


def te_states(chunks: np.ndarray, coeff: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """chunks (N, C, 7)：第 k 帧 ACT 预测的整段（第 k … k+C−1 帧）。
    返回 a_pi (N, 7)：TE 在第 k 帧输出的动作；ens (N, C−1, 7)：第 k 帧更新后，第 k+1+j 帧的部分融合值。
    算法与 `ACTTemporalEnsembler.update`（modeling_act.py）相同：权重 w_i = exp(−coeff·i)，最旧的预测是 w_0；float32、运算顺序相同。"""
    chunks = np.asarray(chunks, np.float32)
    n, c, d = chunks.shape
    w = np.exp(np.float32(-coeff) * np.arange(c, dtype=np.float32))
    cw = np.cumsum(w, dtype=np.float32)
    a_pi, ens = np.empty((n, d), np.float32), np.empty((n, c - 1, d), np.float32)
    acc = cnt = None                                  # 还没消耗的各帧的融合值，以及各自已融合了几段
    for k, x in enumerate(chunks):
        if acc is None:
            acc, cnt = x.copy(), np.ones(c, int)
        else:
            acc = acc * cw[cnt - 1][:, None]
            acc = acc + x[:-1] * w[cnt][:, None]
            acc = acc / cw[cnt][:, None]
            cnt = np.minimum(cnt + 1, c)
            acc, cnt = np.concatenate([acc, x[-1:]]), np.concatenate([cnt, [1]])
        a_pi[k], acc, cnt = acc[0], acc[1:], cnt[1:]   # 当前帧被「消耗」，其余留给之后
        ens[k] = acc
    return a_pi, ens


def _lead(lead_s: float = LEAD_S, dt: float = DT) -> tuple[int, float]:
    """提前 lead_s 落在第 k+1+i 帧与第 k+2+i 帧之间、离前者 r 帧。"""
    f = lead_s / dt - 1.0
    i = int(np.floor(f))
    return i, f - i


def online_shift(ens_k: np.ndarray, lead_s: float = LEAD_S, dt: float = DT) -> np.ndarray:
    """第 k 帧的在线 Shift (7,)：在 k + lead_s/dt 处对部分融合值线性插值；ens_k[0] 是第 k+1 帧。"""
    i, r = _lead(lead_s, dt)
    return (1.0 - r) * ens_k[i] + r * ens_k[i + 1]


@dataclass(frozen=True)
class OnlineCfg:
    """在线窗与网络输入的配置；`sim/b4_common.py cfg` 生成 `outputs/b4/online_cfg.json`。"""
    center: tuple[float, float, float]   # 目标区中心（m）：示教里放下点的中位数
    jnt_lo: tuple[float, ...]            # 6 个模型限位（rad）
    jnt_hi: tuple[float, ...]
    model_path: str
    contact_s: float = 0.5               # 夹爪切换前后算疑似接触（s）
    contact_pad_s: float = 0.1           # 在线再多留（s），偏保守
    zone_m: float = 0.03                 # 指令末端离目标中心多近算疑似接触（m）
    grip_open: float = 0.04              # 夹爪指令的张开阈值
    taper: int = 10                      # 窗的余弦过渡帧数（0.5 s）
    tau_frames: int = 1                  # 限位饱和的前瞻帧数（同 ilc_route1.saturated）
    P: int = 8                           # 网络输入：过去几帧
    F: int = 8                           # 网络输入：未来几帧


def load_cfg(path) -> OnlineCfg:
    d = json.loads(Path(path).read_text())
    return OnlineCfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


def cfg_sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class Tcp:
    """关节角 → 两指中点（link7、link8）的位置（m），同 sim/cmd_tracking.tcp。"""

    def __init__(self, model_path: str):
        import mujoco
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("link7", "link8")]

    def __call__(self, q6: np.ndarray) -> np.ndarray:
        q6 = np.atleast_2d(q6)
        out = np.empty((len(q6), 3))
        for k, qk in enumerate(q6):
            self.data.qpos[:] = 0
            self.data.qpos[:6] = qk
            self._mj.mj_kinematics(self.model, self.data)
            out[k] = 0.5 * (self.data.xpos[self.bodies[0]] + self.data.xpos[self.bodies[1]])
        return out


def predicted(a_hist: np.ndarray, ens_k: np.ndarray, k: int, lo: int, hi: int) -> np.ndarray:
    """第 k 帧时已知的第 lo … hi 帧指令 (hi − lo + 1, 7)：第 k 帧及以前用最终值（第 0 帧之前用第 0 帧），
    之后用部分融合值（不够长用最后一行）。"""
    fut = ens_k if len(ens_k) else a_hist[k:k + 1]
    j = np.arange(lo, hi + 1)
    return np.where((j <= k)[:, None], a_hist[np.clip(j, 0, k)], fut[np.clip(j - k - 1, 0, len(fut) - 1)])


def online_window(a_hist: np.ndarray, ens_k: np.ndarray, k: int, cfg: OnlineCfg, tcp: Tcp) -> np.ndarray:
    """第 k 帧的窗 (5,)，只用第 k 帧当时已知的指令（约束 2）。零帧 =
    第 0 帧之前；疑似接触（夹爪切换前后 contact_s + contact_pad_s，或指令末端离目标中心 ≤ zone_m；指令与预测 Shift 两份取并集）；
    限位饱和（逐关节：指令、Shift、下一帧指令越限）。离最近零帧 d 帧：d ≥ taper 为 1，否则 0.5 − 0.5 cos(π d / taper)。
    同 v0.10 的窗（ilc_route1），差别：夹爪切换按整数帧距；没有片尾那一帧（在线不可知）。"""
    tp, n_c = cfg.taper, int(round((cfg.contact_s + cfg.contact_pad_s) / DT))
    lo = k - tp - n_c - 1                                    # 多取一帧：切换要看前后两帧
    c = predicted(a_hist, ens_k, k, lo, k + tp + n_c + 2)
    i, r = _lead()
    s = (1.0 - r) * c[1 + i:len(c) - 1 + i] + r * c[2 + i:len(c) + i]   # 第 lo … 帧的预测 Shift
    ev = [lo + 1 + np.flatnonzero(np.diff(x[:, 6] > cfg.grip_open)) for x in (c, s)]
    ev = np.concatenate(ev)
    j = np.arange(k - tp, k + tp + 1)
    cj, sj, cn = c[j - lo], s[j - lo], c[j + cfg.tau_frames - lo]
    grip = (np.abs(j[:, None] - ev[None, :]) <= n_c).any(1) if len(ev) else np.zeros(len(j), bool)
    center = np.asarray(cfg.center)
    zone = ((np.linalg.norm(tcp(cj[:, :6]) - center, axis=1) <= cfg.zone_m)
            | (np.linalg.norm(tcp(sj[:, :6]) - center, axis=1) <= cfg.zone_m))
    lim_lo, lim_hi = np.asarray(cfg.jnt_lo), np.asarray(cfg.jnt_hi)
    sat = np.zeros((len(j), 6), bool)
    for x in (cj, sj, cn):
        sat |= (x[:, :6] < lim_lo) | (x[:, :6] > lim_hi)
    zero = (j < 0)[:, None] | (grip | zone)[:, None] | sat[:, list(ACTIVE)]
    d = np.where(zero, np.abs(j - k)[:, None], tp).min(0)
    return np.where(d >= tp, 1.0, 0.5 - 0.5 * np.cos(np.pi * d / tp))


def online_input(a_hist: np.ndarray, ens_k: np.ndarray, k: int, P: int, F: int) -> np.ndarray:
    """网络输入 ((P + F)·7,) float32：第 k − P + 1 … k 帧的最终指令 + 第 k+1 … k+F 帧的部分融合值（展平）。"""
    return predicted(a_hist, ens_k, k, k - P + 1, k + F).astype(np.float32).ravel()


def apply_correction(shift7: np.ndarray, w5: np.ndarray, d5: np.ndarray, cfg: OnlineCfg) -> np.ndarray:
    """实发指令 u7 (float32) = 在线 Shift + 补正。补正只在窗 > 0 的帧发（二值门），限幅到 ±CLIP 且 Shift + Δ 离管线硬界
    至少 BOUND_MARGIN（同 ilc_route1.bounds）；**不乘窗本身**：教师的补正（训练标签）已经含窗，再乘一次就和教师发的不一致
    （约束 1）。加法与 ilc_route1.write_seq 相同（float32）。部署和离线复算的实发补正都只经过这里。"""
    act = list(ACTIVE)
    u = np.array(shift7, np.float32)
    s = u[act].astype(float)
    lo = np.maximum(-CLIP, np.asarray(cfg.jnt_lo)[act] - HARD_SLACK + BOUND_MARGIN - s)
    hi = np.minimum(CLIP, np.asarray(cfg.jnt_hi)[act] + HARD_SLACK - BOUND_MARGIN - s)
    u[act] += ((np.asarray(w5) > 0) * np.clip(d5, lo, hi)).astype(np.float32)
    return u


# ── 补正网络（实现计划任务 9）──────────────────────────────────────────────
import torch  # noqa: E402
from torch import nn  # noqa: E402


class B4Net(nn.Module):
    """网络输入 ((P+F)·7) → n_out 维（rad）：一帧版是第 k 帧五个活动关节的补正（n_out = 5）；
    整段版 ① 是第 k … k+N−1 帧相对参考的指令，按帧优先展平（n_out = 5·N）。
    输入先经 `prep`（整段版 v0.3 的相对化；一帧版原样），再按训练集的均值、标准差标准化（buffer mu、sd）。"""

    def __init__(self, n_in: int, hidden: int = 128, n_out: int = len(ACTIVE), rel_P: int | None = None):
        super().__init__()
        self.rel_P = rel_P
        n_feat = n_in + 7 if rel_P else n_in
        self.register_buffer("mu", torch.zeros(n_feat))
        self.register_buffer("sd", torch.ones(n_feat))
        self.mlp = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def prep(self, x):
        """整段版 v0.3（方案 `reports/PLAN_b4_seq.md` §4.2）：x 按 (P+F, 7) 帧排，第 rel_P−1 帧是 a_π(k)；
        各帧减去它（展平）再附上它本身。提前量于是成了几个小差值的线性组合。rel_P = None（一帧版）原样返回。"""
        if self.rel_P is None:
            return x
        f = x.reshape(*x.shape[:-1], -1, 7)
        cur = f[..., self.rel_P - 1, :]
        return torch.cat([(f - cur.unsqueeze(-2)).flatten(-2), cur], -1)

    def forward(self, x):
        return self.mlp((self.prep(x) - self.mu) / self.sd)


def load_frozen(path: str):
    """冻结文件（JSON）→ (CPU 上的推理函数 np (n_in,) → np (n_out,), 权重哈希, P, F)。权重文件的哈希和冻结时不同就停。
    n_out、rel_P 取权重文件里记的；一帧版的旧文件没有这两个键，就是 5 和 None（不相对化）。"""
    d = json.loads(Path(path).read_text())
    w = Path(d["weights"])
    sha = hashlib.sha256(w.read_bytes()).hexdigest()[:16]
    if sha != d["weights_sha"]:
        raise SystemExit(f"⛔ 冻结网络的权重文件被改过（{d['weights_sha']} → {sha}）")
    m = torch.load(w, map_location="cpu")
    net = B4Net(m["n_in"], m["hidden"], m.get("n_out", len(ACTIVE)), m.get("rel_P"))
    net.load_state_dict(m["state_dict"])
    net.eval()

    @torch.no_grad()
    def fn(x: np.ndarray) -> np.ndarray:
        return net(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()

    return fn, sha, int(d["P"]), int(d["F"])


# ── 重新生成网络 ②（方案 `reports/PLAN_b4_regen.md` v0.2；实现计划任务 1）────────────────────────────
REGEN_P = 8                       # 过去几帧当输入单元
REGEN_N = 10                      # 之后几帧：ACT 指令段与输出段同形
CUT_S = 0.010                     # 实测截止时刻 = 发令时刻（在线：本帧开始）− 10 ms
WIN_S = 0.400                     # 实测窗口长度（= REGEN_P 帧）
ENC_DT, IMU_DT = 0.005, 0.010     # 编码器 200 Hz、IMU 100 Hz 的网格间隔
CAN_GROUPS = ((1, (0, 1)), (3, (2, 3)), (5, (4, 5)))   # can_q200 的组号 → 两个关节
_PAD_S = 0.05                     # 窗口起点之前还用多少秒的采样（只为插值第一个网格点；离线、在线都按它截，结果才一致）


def _window(t: np.ndarray, v: np.ndarray, cutoff: float, dt: float) -> tuple[np.ndarray, float]:
    """到达时刻 t（已按时间排好）落在 [cutoff − WIN_S − _PAD_S, cutoff) 的采样，线性插值到 cutoff − WIN_S + dt·i（i = 1…）。
    比最后一个采样晚的网格点保持最后的值（np.interp 的右端）。返回 (窗口, 最新采样距 cutoff 的秒数)；没有采样时为 NaN 与 inf。"""
    grid = cutoff - WIN_S + dt * np.arange(1, round(WIN_S / dt) + 1)
    lo, hi = np.searchsorted(t, [cutoff - WIN_S - _PAD_S, cutoff])
    if hi == lo:
        return np.full((len(grid), v.shape[1]), np.nan), np.inf
    t, v = t[lo:hi], v[lo:hi]
    return np.stack([np.interp(grid, t, v[:, c]) for c in range(v.shape[1])], 1), float(cutoff - t[-1])


def enc_window(q200: np.ndarray, offset: float, cutoff: float) -> tuple[np.ndarray, float]:
    """编码器窗口 (80, 6) rad。q200 是 can_q200 行 [t_realtime, 组号, q_a, q_b]（0.001°），按到达时刻排好；
    到达时刻 = t_realtime − offset（offset = can_hs_meta 里的 offset_realtime_minus_monotonic_s）。
    返回的秒数取三组里最旧的那个「最新采样」；某组没有可用采样时，它的两个关节为 NaN，秒数为 inf。"""
    q200 = np.asarray(q200, float)
    out, age = np.empty((round(WIN_S / ENC_DT), 6)), 0.0
    for g, cols in CAN_GROUPS:
        r = q200[q200[:, 1] == g]
        w, a = _window(r[:, 0] - offset, np.radians(r[:, 2:4] * 1e-3), cutoff, ENC_DT)
        out[:, list(cols)], age = w, max(age, a)
    return out, age


def imu_window(imu100: np.ndarray, calib: dict, cutoff: float) -> tuple[np.ndarray, float]:
    """IMU 窗口 (40, 6) = [角速度 ω, 比力 f]，按会话标定换到统一坐标：ω = Rᵀ(陀螺 − gyro_bias)，f = Rᵀ(加速度 − b)。
    imu100 行 [到达时刻, 设备 ms, 加速度 3 维, 陀螺 3 维]，按到达时刻排好。"""
    rows = np.asarray(imu100, float)
    R = np.asarray(calib["R"], float)
    v = np.c_[(rows[:, 5:8] - calib["gyro_bias"]) @ R, (rows[:, 2:5] - calib["b"]) @ R]    # 行向量：Rᵀx = x @ R
    return _window(rows[:, 0], v, cutoff, IMU_DT)


IDEAL_DT = 0.005                  # 老师定义的理想（sim/ideal_motion）的网格间隔（s）


def ideal_err(enc: np.ndarray, cutoff: float, t_send: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """编码器窗口 enc (80, 6) 各采样处的「理想 − 实测」(rad)，乘采样所在帧的窗 w (n, 5)（会动的关节用自己的，j4 用五个关节里最小的）。
    采样时刻 → 理想的时间（第 0 帧发令时刻为 0）：按已经发出的各帧的发令时刻 t_send (k,) 分段线性对应，第 0 帧之前、最后一帧之后按 1 秒 = 1 秒外推。
    理想 x (N, 6) 是 IDEAL_DT 网格上的关节角，网格之间线性插值。还没发过帧（k = 0）时全为 0。"""
    k = len(t_send)
    if k == 0:
        return np.zeros_like(enc)
    t = cutoff - WIN_S + ENC_DT * np.arange(1, len(enc) + 1)
    nom = np.where(t < t_send[0], t - t_send[0],
                   np.where(t > t_send[-1], DT * (k - 1) + t - t_send[-1], np.interp(t, t_send, DT * np.arange(k))))
    ideal = np.stack([np.interp(nom, IDEAL_DT * np.arange(len(x)), x[:, j]) for j in range(6)], 1)
    wf = w[np.clip(np.floor(nom / DT).astype(int), 0, len(w) - 1)]
    win = np.empty((len(t), 6))
    win[:, list(ACTIVE)], win[:, 3] = wf, wf.min(1)
    return (ideal - enc) * win


def regen_act(a7: np.ndarray, a_pi: np.ndarray, ens: np.ndarray, k: int, hold: int) -> tuple[np.ndarray, np.ndarray]:
    """回放第 k 帧（开头 hold 帧保持）时已知的 ACT 指令：过去 REGEN_P 帧的最终值 (8, 6)（帧号小于 0 取第 0 帧），
    之后 REGEN_N 帧 (10, 6)（录制第 k − hold 帧当时已知的值；保持段里用录制第 0 帧的知识）。"""
    kn = max(k - hold, 0)
    return (a7[np.clip(np.arange(k - REGEN_P, k), 0, None), :6],
            predicted(a_pi, ens[kn], kn, k - hold, k - hold + REGEN_N - 1)[:, :6])


def regen_tokens(act_past, sent_past, enc, imu, act_seg) -> tuple[np.ndarray, np.ndarray]:
    """输入单元：过去 (8, 102) = 每帧 [ACT 指令 6, 已发出的指令 6, 编码器 10 个采样 × 6, IMU 5 个采样 × 6]；之后 (10, 6) = ACT 指令段。"""
    past = np.concatenate([act_past, sent_past, np.reshape(enc, (REGEN_P, -1)), np.reshape(imu, (REGEN_P, -1))], 1)
    return past.astype(np.float32), np.asarray(act_seg, np.float32)


GROUP_COLS = {"sent": slice(6, 12), "enc": slice(12, 72), "imu": slice(72, 102)}   # 过去单元里可去掉的输入组（对照用；prep 前后列位置相同）


class RegenNet(nn.Module):
    """重新生成网络 ②（方案 `reports/PLAN_b4_regen.md` §4.4；实现计划任务 3）：过去 8 个单元（ACT 指令、已发出的指令、编码器、IMU）
    与之后 10 个单元（ACT 指令）→ 之后 10 帧 6 个关节的指令，绝对角度（rad）。
    输入先经 `prep` 改成相对量（用户 11-11 选 B），输入、输出再按训练集的均值和标准差标准化（buffer）；drop 里的输入组在标准化之后置 0。
    输出 = mu_y + sd_y × 线性头：网络里没有把 ACT 或 Shift 指令加回输出的通路。"""

    def __init__(self, n_past: int = 102, n_fut: int = 6, d: int = 128, layers: int = 4, heads: int = 4, ff: int = 512,
                 dropout: float = 0.1, drop: tuple[str, ...] = ()):
        super().__init__()
        self.drop = tuple(drop)
        for name, n in (("p", n_past), ("f", 2 * n_fut), ("y", 6)):
            self.register_buffer(f"mu_{name}", torch.zeros(n))
            self.register_buffer(f"sd_{name}", torch.ones(n))
        keep = torch.ones(n_past)
        for g in self.drop:
            keep[GROUP_COLS[g]] = 0.0
        self.register_buffer("keep", keep)
        self.emb_p, self.emb_f = nn.Linear(n_past, d), nn.Linear(2 * n_fut, d)
        self.pos = nn.Parameter(0.02 * torch.randn(REGEN_P + REGEN_N, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 6)

    @staticmethod
    def prep(past: torch.Tensor, fut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """r = 第 k 帧的 ACT 指令（之后单元的第一帧）。过去单元：ACT 指令、已发出的指令减 r，编码器各点减同一帧已发出的指令
        （跟踪误差），IMU 不变；之后单元：ACT 指令减 r，再附上 r 本身（12 维）。绝对值都能由这些量还原，信息没有丢。"""
        r = fut[:, :1]
        sent = past[..., 6:12]
        enc = past[..., 12:72].unflatten(-1, (-1, 6)) - sent.unsqueeze(-2)
        return (torch.cat([past[..., :6] - r, sent - r, enc.flatten(-2), past[..., 72:]], -1),
                torch.cat([fut - r, r.expand_as(fut)], -1))

    def forward(self, past: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
        p, f = self.prep(past, fut)
        p = (p - self.mu_p) / self.sd_p * self.keep
        f = (f - self.mu_f) / self.sd_f
        h = self.enc(torch.cat([self.emb_p(p), self.emb_f(f)], 1) + self.pos)
        return self.mu_y + self.sd_y * self.head(h[:, REGEN_P:])


def regen_fn(m: dict, device: str = "cpu"):
    """权重文件的内容（torch.load 的结果）→ 推理函数 (8, 102), (10, 6) → (10, 6) float64（原始输入单元，网络里自己相对化）。
    device="cuda"：加载时把整个前向录成一张 CUDA graph，每帧只复制输入、回放、取回输出（用户 11-12）——
    真机上 CPU 会降频、还有机械臂的后台线程，CPU 上的调用越少越不受影响（影子运行里 CPU 推理每帧 p50 约 11 ms）。"""
    net = RegenNet(**m["arch"], drop=tuple(m["drop"]))
    net.load_state_dict(m["state_dict"])
    net.eval().to(device)
    t = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None]   # noqa: E731
    if device == "cpu":
        @torch.no_grad()
        def fn(past: np.ndarray, fut: np.ndarray) -> np.ndarray:
            return net(t(past), t(fut))[0].double().numpy()
        return fn
    p_in = torch.zeros(1, REGEN_P, m["arch"]["n_past"], device=device)
    f_in = torch.zeros(1, REGEN_N, m["arch"]["n_fut"], device=device)
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    with torch.no_grad():
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):                     # 录图前先在旁路流上跑几遍（分配显存、选算法）
            for _ in range(3):
                net(p_in, f_in)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            out = net(p_in, f_in)

    def fn(past: np.ndarray, fut: np.ndarray) -> np.ndarray:
        p_in.copy_(t(past))
        f_in.copy_(t(fut))
        graph.replay()
        return out[0].double().cpu().numpy()              # 取回 CPU 时会等 GPU 算完

    fn.net = net                                          # 图回放直接读参数所在的显存：网络必须一直活着，不能被回收
    return fn


def load_regen(path: str, device: str = "cpu"):
    """冻结文件（JSON）→ (推理函数（见 regen_fn）, 权重哈希)。权重文件的哈希和冻结时不同就停。"""
    d = json.loads(Path(path).read_text())
    w = Path(d["weights"])
    sha = hashlib.sha256(w.read_bytes()).hexdigest()[:16]
    if sha != d["weights_sha"]:
        raise SystemExit(f"⛔ 冻结网络的权重文件被改过（{d['weights_sha']} → {sha}）")
    return regen_fn(torch.load(w, map_location="cpu"), device), sha


# ── ② 的逐帧检查、指令切换与界（设计 §4.5–4.6；实现计划任务 4）────────────────────────────
STEP_MAX = np.array([0.060, 0.045, 0.045, 0.005, 0.025, 0.045])   # 每帧变化的上限（rad）
DIFF_MAX = np.array([0.100, 0.100, 0.100, 0.010, 0.050, 0.100])   # 与 ACT 原始指令之差的上限（rad）
POS_PAD = 0.05                    # 位置界 = 已发过的范围 ± POS_PAD（rad），再与管线硬界内收 BOUND_MARGIN 取交集
STALE_ENC_S = 0.030               # 实测中断：编码器任一组的最新采样早于截止时刻这么久（s）
STALE_IMU_S = 0.050               # 实测中断：IMU 的最新采样早于截止时刻这么久（s）


def regen_bounds(cfg: OnlineCfg, sent_lo: np.ndarray, sent_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐帧检查的位置界（6 个关节，rad）：[已发过的最小值 − POS_PAD, 最大值 + POS_PAD] ∩ [模型限位 − HARD_SLACK + BOUND_MARGIN, … ]。"""
    lo = np.maximum(np.asarray(sent_lo) - POS_PAD, np.asarray(cfg.jnt_lo) - HARD_SLACK + BOUND_MARGIN)
    hi = np.minimum(np.asarray(sent_hi) + POS_PAD, np.asarray(cfg.jnt_hi) + HARD_SLACK - BOUND_MARGIN)
    return lo, hi


def step_check(u6: np.ndarray, prev6: np.ndarray, a6: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> str | None:
    """要发出的 6 个关节指令的逐帧检查（停机条件，不是限幅）：返回第一个违反的规则（规则名开头，再写关节和数值），都不违反返回 None。
    顺序：非有限 → 位置 → 每帧变化（相对上一帧实发的） → 与 ACT 指令 a6 的差。"""
    u6 = np.asarray(u6, float)
    if not np.isfinite(u6).all():
        return f"非有限：{u6.tolist()}"
    step, diff = np.abs(u6 - prev6), np.abs(u6 - a6)
    for rule, bad, txt in (("位置", (u6 < lo) | (u6 > hi), lambda j: f"{u6[j]:.4f} 不在 [{lo[j]:.4f}, {hi[j]:.4f}]"),
                           ("每帧变化", step > STEP_MAX, lambda j: f"{step[j]:.4f} > {STEP_MAX[j]}"),
                           ("与 ACT 指令的差", diff > DIFF_MAX, lambda j: f"{diff[j]:.4f} > {DIFF_MAX[j]}")):
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            return f"{rule}：j{j + 1} {txt(j)} rad"
    return None


def regen_command(first6: np.ndarray, s7k: np.ndarray, w5k: np.ndarray) -> np.ndarray:
    """实发指令 (7,) float32：会动的关节窗 > 0 用网络第一帧、否则用 Shift；j4 在五个关节的窗都 > 0 时用网络；夹爪用 Shift。"""
    use = np.zeros(6, bool)
    use[list(ACTIVE)] = np.asarray(w5k) > 0
    use[3] = np.min(w5k) > 0
    u = np.array(s7k, np.float32)
    u[:6] = np.where(use, first6, u[:6])
    return u
