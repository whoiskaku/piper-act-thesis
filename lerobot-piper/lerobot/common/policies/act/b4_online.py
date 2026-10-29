"""B4 的在线量：控制回路里第 k 帧当时就算得出的东西，离线复算也用同一批函数（方案 `reports/PLAN_b4.md` 约束 2）。

- `te_states`：由每帧记下的 ACT 整段预测，逐帧复算 TE 的输出动作与之后各帧的部分融合值。
- `online_shift`：提前 60 ms 的指令 = 第 k+1、k+2 帧部分融合值的线性插值（0.8 / 0.2）。
- `online_window`：第 k 帧的窗（疑似接触、限位饱和处为 0，余弦过渡），只用第 k 帧当时已知的指令。
- `online_input`：网络输入 = 过去 P 帧最终指令 + 未来 F 帧部分融合值。
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
    """网络输入 ((P+F)·7) → 第 k 帧五个活动关节的补正（rad）。输入先按训练集的均值、标准差标准化（buffer mu、sd）。"""

    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.register_buffer("mu", torch.zeros(n_in))
        self.register_buffer("sd", torch.ones(n_in))
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, len(ACTIVE)))

    def forward(self, x):
        return self.mlp((x - self.mu) / self.sd)


def load_frozen(path: str):
    """冻结文件（JSON）→ (CPU 上的推理函数 np (n_in,) → np (5,), 权重哈希, P, F)。权重文件的哈希和冻结时不同就停。"""
    d = json.loads(Path(path).read_text())
    w = Path(d["weights"])
    sha = hashlib.sha256(w.read_bytes()).hexdigest()[:16]
    if sha != d["weights_sha"]:
        raise SystemExit(f"⛔ 冻结网络的权重文件被改过（{d['weights_sha']} → {sha}）")
    m = torch.load(w, map_location="cpu")
    net = B4Net(m["n_in"], m["hidden"])
    net.load_state_dict(m["state_dict"])
    net.eval()

    @torch.no_grad()
    def fn(x: np.ndarray) -> np.ndarray:
        return net(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()

    return fn, sha, int(d["P"]), int(d["F"])
