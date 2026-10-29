"""B4 部署包装（方案 `reports/PLAN_b4.md` §5.1、§5.5）：挂在开 TE 的 ACT 外面，每帧

1. 照 ACT 的 `select_action`（TE 分支）推理并更新 TE，得到原始策略指令 a_π；
2. 用 TE 里还没执行的各帧部分融合值，算在线 Shift、在线窗、网络输入（`b4_online`，只用当时已知的信息）；
3. 按模式发：raw = a_π，shift = 在线 Shift，net = 在线 Shift + 网络补正（`apply_correction`，不乘窗）。

每帧记下 a_π、整段预测、Shift、窗、网络输入、补正、实发指令，一集一个 npz（`record_episode` 存在高频记录旁）。
`control_robot.py` 用环境变量 `B4_MODE` 开启。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import lerobot.common.policies.act.b4_online as O

MODES = ("raw", "shift", "net")
KEYS = ("a_pi", "chunk", "shift", "w", "x", "delta", "u")


class B4Deploy:
    def __init__(self, act_policy, mode: str, cfg: O.OnlineCfg, cfg_sha: str,
                 net: Callable[[np.ndarray], np.ndarray] | None = None, net_sha: str = "",
                 P: int | None = None, F: int | None = None):
        if mode not in MODES:
            raise ValueError(f"B4 模式只能是 {MODES}，收到 {mode!r}")
        if mode == "net" and net is None:
            raise ValueError("net 模式需要网络")
        self.act, self.mode, self.cfg, self.cfg_sha, self.net, self.net_sha = act_policy, mode, cfg, cfg_sha, net, net_sha
        self.P, self.F = cfg.P if P is None else P, cfg.F if F is None else F
        self.tcp = O.Tcp(cfg.model_path)
        self.log = {k: [] for k in KEYS}

    def __getattr__(self, name):          # 其余属性（config、to、parameters …）透传给 ACT
        if name == "act":
            raise AttributeError(name)
        return getattr(self.act, name)

    def reset(self):
        self.act.reset()                  # 清 TE
        self.log = {k: [] for k in KEYS}

    @torch.no_grad()
    def select_action(self, batch) -> torch.Tensor:
        act = self.act
        act.eval()
        chunk = act.predict_chunk(batch)
        a = act.temporal_ensembler.update(chunk)                  # 同 ACTPolicy.select_action 的 TE 分支
        ens_k = act.temporal_ensembler.ensembled_actions[0].cpu().numpy()
        L = self.log
        L["a_pi"].append(a[0].cpu().numpy())
        L["chunk"].append(chunk[0].cpu().numpy())
        k, a_hist = len(L["a_pi"]) - 1, np.stack(L["a_pi"])
        shift = np.asarray(O.online_shift(ens_k), np.float32)
        w = O.online_window(a_hist, ens_k, k, self.cfg, self.tcp)
        x = O.online_input(a_hist, ens_k, k, self.P, self.F)
        if self.mode == "raw":
            u = L["a_pi"][-1]
        elif self.mode == "shift":
            u = shift
        else:
            d = np.asarray(self.net(x), np.float32)
            if not np.all(np.isfinite(d)):            # 网络输出 NaN / inf：这一帧不补正，不把 NaN 发给机械臂
                d = np.zeros_like(d)
            u = O.apply_correction(shift, w, d, self.cfg)
        delta = u[list(O.ACTIVE)] - shift[list(O.ACTIVE)] if self.mode == "net" else np.zeros(5, np.float32)
        for key, v in zip(KEYS[2:], (shift, w, x, delta, u)):
            L[key].append(v)
        return a if self.mode == "raw" else torch.from_numpy(np.array(u))[None].to(a.device, a.dtype)

    def save_episode_log(self, path: Path, error: str | None = None):
        """一集的记录存成 npz；error 非空 = 这一集中途出错（含 Ctrl+C），已有的帧照存、标为不完整。
        同一集号已有记录（中断后补录、按 ← 重录）：旧的改名 <名>.try1.npz、try2 … 留档，不覆盖。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            n = 1
            while (old := path.with_name(f"{path.stem}.try{n}.npz")).exists():
                n += 1
            path.rename(old)
        np.savez(path, **{k: np.asarray(v) for k, v in self.log.items()}, mode=self.mode, cfg_sha=self.cfg_sha,
                 net_sha=self.net_sha, P=self.P, F=self.F, complete=error is None, error=error or "")


def build_b4_policy(act_policy, mode: str, cfg_path: str, net_path: str | None = None) -> B4Deploy:
    """raw / shift / net 模式的包装；net 模式读冻结网络（net_path），输入窗口 P、F 用冻结文件里的值。"""
    if mode not in MODES:
        raise ValueError(f"B4 模式只能是 {MODES}，收到 {mode!r}")
    cfg, sha = O.load_cfg(cfg_path), O.cfg_sha(cfg_path)
    if mode != "net":
        return B4Deploy(act_policy, mode, cfg, sha)
    if not net_path:
        raise ValueError("net 模式需要冻结网络：B4_NET=<outputs/b4/net/frozen.json>")
    fn, net_sha, P, F = O.load_frozen(net_path)
    return B4Deploy(act_policy, "net", cfg, sha, net=fn, net_sha=net_sha, P=P, F=F)
