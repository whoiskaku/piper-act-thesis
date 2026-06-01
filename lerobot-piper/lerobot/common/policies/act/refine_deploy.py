"""Stage 5 — RefineNet 真机部署 wrapper（ACT → RefineNet → 边界 blend）。

把已加载的 frozen ACTPolicy 包一层：在 ACT 生成动作 chunk 的时刻插入 RefineNet 残差
精修 + chunk 边界续接 blend，再按 receding-horizon 逐步下发。接口与 ACTPolicy 一致
（select_action / reset / .config），可直接 drop-in 进 control_robot 的 record loop。

集成（control_robot 里 policy 加载之后，一行包起来）：
    from lerobot.common.policies.act.refine_deploy import build_refine_deploy_policy
    policy = build_refine_deploy_policy(
        policy,                                   # 已加载的 frozen ACTPolicy
        refine_ckpt="outputs/train/refine_net/phase1_runB_hf0.10_dl0.1_lr0.0003_bs64_ep200/ckpt_best.pt",
        device=device, refine_enabled=True)       # A/B: True=ACT+RefineNet, False=纯ACT

A/B 协议（Stage 5 第一实验）：
    A(baseline): refine_enabled=False（= 纯 ACT，等价不包 wrapper）
    B(method):   refine_enabled=True
    两者各录若干 episode → eval_smoothness_realrobot.py 比真机 IMU hf/RMS。

复用部署侧 history：control_utils 已把 observation.history_action/history_imu/
history_pad_mask 注入 observation（history_buffer），RefineNet 直接取用，与训练分布一致。
唯一需转换的是 current_state（obs.state 单位假设为 dataset 的 mdeg → 转 rad 再 z-score）。

⚠️ 真机首跑务必核对：(1) obs.state 单位确为 mdeg；(2) 打印的 z-scored state 量级正常
（|z|≲5）；(3) RefineNet ckpt 的 state_mean/std 与本任务一致。否则 current_state 会喂错。
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import torch

from lerobot.common.policies.act.refine_net import RefineNet

# 与 sim/replay_demo_validation 一致的状态单位换算（训练 current_state 用 rad）。
_MDEG_TO_RAD = math.pi / 180.0 / 1000.0
_GRIPPER_SCALE = 1.0 / 800000.0
_ACTIVE_JOINTS = (0, 1, 2, 6)


class RefineNetDeployWrapper:
    """Drop-in policy: frozen ACT + RefineNet residual + boundary blend。"""

    def __init__(self, act_policy, refine_model, state_mean, state_std,
                 device, blend_m: int = 10, refine_enabled: bool = True,
                 state_is_mdeg: bool = True, verbose: bool = True):
        self.act = act_policy
        self.refine = refine_model.eval()
        self.config = act_policy.config            # 暴露给 control_robot 兼容
        self.device = torch.device(device)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        self.blend_m = blend_m
        self.refine_enabled = refine_enabled
        self.state_is_mdeg = state_is_mdeg
        self.verbose = verbose
        self.n_action_steps = act_policy.config.n_action_steps
        self._queue = deque(maxlen=self.n_action_steps)
        self._last_action = None                   # 上一步下发的动作(用于边界 blend)
        self._warned_state = False

    # ---- 与 ACTPolicy 接口一致 ----
    def reset(self):
        self.act.reset()
        self._queue.clear()
        self._last_action = None

    def to(self, device):
        self.device = torch.device(device)
        self.refine.to(device)
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        return self

    def eval(self):
        self.act.eval(); self.refine.eval(); return self

    def parameters(self):
        return self.refine.parameters()

    # ---- current_state: obs.state(mdeg) → rad → z-score ----
    def _current_state(self, batch):
        s = batch["observation.state"].to(self.device).float()    # (1,7)
        if self.state_is_mdeg:
            rad = s.clone()
            rad[:, :6] = s[:, :6] * _MDEG_TO_RAD
            rad[:, 6] = torch.clamp(s[:, 6] * _GRIPPER_SCALE, 0.0, 0.035)
        else:
            rad = s
        z = (rad - self.state_mean) / (self.state_std + 1e-8)
        if self.verbose and not self._warned_state:
            self._warned_state = True
            mx = float(z.abs().max())
            print(f"[refine_deploy] current_state z-score |max|={mx:.2f} "
                  f"({'OK' if mx < 8 else '⚠ 量级异常: 核对 obs.state 单位/stats'})")
        return z

    # ---- 在线边界 blend: 新 chunk 前 m 步从上一动作续接 ----
    def _blend(self, refined):
        if self._last_action is None:
            return refined
        r = refined.clone()
        offset = self._last_action.to(refined.device) - refined[:, 0]    # (1,7)
        for i in range(min(self.blend_m, refined.shape[1])):
            r[:, i] = refined[:, i] + offset * (1.0 - i / self.blend_m)
        return r

    @torch.no_grad()
    def select_action(self, batch):
        if len(self._queue) == 0:
            # 1) ACT 生成整 chunk(镜像 ACTPolicy.select_action 的 chunk 生成)
            b = self.act.normalize_inputs(batch)
            if self.act.config.image_features:
                b = dict(b)
                b["observation.images"] = torch.stack(
                    [b[k] for k in self.act.config.image_features], dim=-4)
            act_chunk = self.act.model(b)[0][:, : self.n_action_steps]    # (1,T,7) normalized
            act_chunk = self.act.unnormalize_outputs({"action": act_chunk})["action"]  # raw

            if self.refine_enabled and "observation.history_action" in batch:
                ha = batch["observation.history_action"].to(self.device).float()
                hi = batch["observation.history_imu"].to(self.device).float()
                hp = batch.get("history_pad_mask")
                hp = hp.to(self.device) if hp is not None else None
                state = self._current_state(batch)
                refined, _ = self.refine(act_chunk.to(self.device), ha, hi, state,
                                         history_pad_mask=hp)
                refined = self._blend(refined)
            else:
                refined = act_chunk

            self._queue.extend(refined.transpose(0, 1))
        a = self._queue.popleft()
        self._last_action = a.detach()
        return a


def build_refine_deploy_policy(act_policy, refine_ckpt, device,
                               blend_m: int = 10, refine_enabled: bool = True,
                               state_is_mdeg: bool = True):
    """加载 RefineNet ckpt 并包住已加载的 frozen ACTPolicy。"""
    ckpt = torch.load(Path(refine_ckpt), map_location=device, weights_only=False)
    refine = RefineNet().to(device)
    refine.load_state_dict(ckpt["model_state"])
    refine.eval()
    print(f"[refine_deploy] RefineNet ckpt ep{ckpt.get('epoch')} loaded; "
          f"refine_enabled={refine_enabled} blend_m={blend_m}")
    return RefineNetDeployWrapper(
        act_policy, refine, ckpt["state_mean"], ckpt["state_std"],
        device=device, blend_m=blend_m, refine_enabled=refine_enabled,
        state_is_mdeg=state_is_mdeg)
