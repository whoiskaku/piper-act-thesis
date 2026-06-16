"""Stage 5 — RefineNet 真机部署 wrapper(ACT → [残差精修] → [边界 blend])。

把已加载的 frozen ACTPolicy 包一层,在 chunk 生成时刻插入两个 **独立开关** 的步骤:
  - residual_enabled: RefineNet 残差精修(学到的 Δ)
  - blend_enabled:    chunk 边界续接 blend(确定性,无需学习)
接口与 ACTPolicy 一致(select_action / reset / .config),drop-in 进 control_robot。

⚠️ 设计修正(2026-06-11):旧版用单一 refine_enabled 同时开关 blend+残差,导致
Stage 5 真机 A/B 把两者的贡献混在一起(部署分解证明旧 fraction 残差实为有害、
收益全来自 blend)。现解耦,支持真机三方 A/B:
    A 裸 ACT:        REFINE_MODE=off   (或不设 REFINE_CKPT)
    B 纯 blend:      REFINE_MODE=blend
    C blend+残差:    REFINE_MODE=full  (用 #64/#65 的 absboth ckpt!)
    (诊断用: REFINE_MODE=residual 仅残差无 blend)

架构按 ckpt['args'] 自动复原(variant=S0 等; use_act_img_attn 时在 frozen ACT 的
transformer encoder 上挂 hook,部署时取 image tokens 喂 RefineNet 注意力读出,
与 sim/precompute_act_img.py 的 offset 逻辑一致)。

集成(control_robot 里 policy 加载之后):
    from lerobot.common.policies.act.refine_deploy import build_refine_deploy_policy
    policy = build_refine_deploy_policy(policy, refine_ckpt=..., device=...,
                                        mode="full", blend_m=10)

复用部署侧 history:control_utils 已把 observation.history_action/history_imu/
history_pad_mask 注入 observation,RefineNet 直接取用。唯一需转换的是
current_state(obs.state mdeg → rad → z-score)。

⚠️ 真机首跑务必核对:(1) obs.state 单位确为 mdeg;(2) 打印的 z-scored state 量级
正常(|z|≲5);(3) ckpt 的 state_mean/std 与本任务一致;(4) Hattn ckpt 时启动日志
有 "act_img hook" 行且 n_img=300。
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import torch

from lerobot.common.policies.act.refine_net import RefineNet

# 与 sim/replay_demo_validation 一致的状态单位换算(训练 current_state 用 rad)。
_MDEG_TO_RAD = math.pi / 180.0 / 1000.0
_GRIPPER_SCALE = 1.0 / 800000.0
_ACTIVE_JOINTS = (0, 1, 2, 6)

_VARIANTS = {"S0": (False, False, False), "S1": (False, True, True),
             "S2": (True, True, False), "S3": (True, False, False),
             "full": (True, True, True)}


class RefineNetDeployWrapper:
    """Drop-in policy: frozen ACT + [RefineNet residual] + [boundary blend]。"""

    def __init__(self, act_policy, refine_model, state_mean, state_std,
                 device, blend_m: int = 10,
                 residual_enabled: bool = True, blend_enabled: bool = True,
                 use_act_img: bool = False,
                 rlt_compressor=None, rlt_mean=None, rlt_std=None,
                 state_is_mdeg: bool = True, verbose: bool = True):
        self.act = act_policy
        self.refine = refine_model.eval()
        self.config = act_policy.config            # 暴露给 control_robot 兼容
        self.device = torch.device(device)
        self.state_mean = torch.as_tensor(state_mean, dtype=torch.float32, device=self.device)
        self.state_std = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        self.blend_m = blend_m
        self.residual_enabled = residual_enabled
        self.blend_enabled = blend_enabled
        self.state_is_mdeg = state_is_mdeg
        self.verbose = verbose
        self.n_action_steps = act_policy.config.n_action_steps
        self._queue = deque(maxlen=self.n_action_steps)
        self._last_action = None                   # 上一步下发的动作(用于边界 blend)
        self._warned_state = False

        # ---- #69 RLT-AE: 部署时实时跑压缩器(image tokens → K 个 RLT token)----
        self.rlt_compressor = rlt_compressor.eval() if rlt_compressor is not None else None
        self.use_act_rlt = rlt_compressor is not None
        if self.use_act_rlt:
            self.rlt_mean = torch.as_tensor(rlt_mean, dtype=torch.float32, device=self.device)
            self.rlt_std = torch.as_tensor(rlt_std, dtype=torch.float32, device=self.device)

        # ---- 挂 ACT encoder hook 取 image tokens(Hattn 直接用 / RLT 经压缩器)----
        # 同 precompute_act_img: offset=n_prefix+n_history
        self.use_act_img = use_act_img
        self._enc_cap = {}
        self._img_offset = None
        if use_act_img or self.use_act_rlt:
            p = act_policy.config
            n_prefix = 1
            n_prefix += int(getattr(p, "robot_state_feature", None) is not None)
            n_prefix += int(getattr(p, "env_state_feature", None) is not None)
            n_prefix += int(getattr(p, "enable_imu_input", False)
                            and getattr(p, "imu_feature", None) is not None)
            hL = getattr(p, "history_max_length", 0)
            n_hist = ((hL if getattr(p, "history_use_action", False) else 0)
                      + (hL if getattr(p, "history_use_imu", False) else 0))
            self._img_offset = n_prefix + n_hist
            act_policy.model.encoder.register_forward_hook(
                lambda m, i, o: self._enc_cap.__setitem__("e", o))
            if verbose:
                print(f"[refine_deploy] act_img hook installed "
                      f"(n_prefix={n_prefix} n_hist={n_hist} → image tokens 从 {self._img_offset} 起)")

    def __getattr__(self, name):
        """未定义的属性/方法委托给内部 frozen ACTPolicy
        (如 normalize_targets / normalize_inputs / unnormalize_outputs,
        供 HistoryBuffer / control_robot 等沿用 ACT 接口)。"""
        act = self.__dict__.get("act")
        if act is None:
            raise AttributeError(name)
        return getattr(act, name)

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
    def _blend(self, chunk):
        if self._last_action is None:
            return chunk
        r = chunk.clone()
        offset = self._last_action.to(chunk.device) - chunk[:, 0]    # (1,7)
        for i in range(min(self.blend_m, chunk.shape[1])):
            r[:, i] = chunk[:, i] + offset * (1.0 - i / self.blend_m)
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
            chunk = act_chunk

            # 2) 残差精修(独立开关)
            if self.residual_enabled and "observation.history_action" in batch:
                ha = batch["observation.history_action"].to(self.device).float()
                hi = batch["observation.history_imu"].to(self.device).float()
                hp = batch.get("history_pad_mask")
                hp = hp.to(self.device) if hp is not None else None
                state = self._current_state(batch)
                act_img = act_rlt = None
                if self.use_act_img or self.use_act_rlt:
                    enc = self._enc_cap.get("e")
                    assert enc is not None, "encoder hook 未捕获输出"
                    img = enc[self._img_offset:].permute(1, 0, 2).float()  # (1, n_img, 512)
                    if self.use_act_rlt:                                   # #69 实时压缩 → K 个 RLT token
                        img_n = (img - self.rlt_mean) / self.rlt_std
                        act_rlt = self.rlt_compressor.encode(img_n)        # (1, K, d)
                    else:
                        act_img = img
                chunk, _ = self.refine(chunk.to(self.device), ha, hi, state,
                                       history_pad_mask=hp, act_img=act_img, act_rlt=act_rlt)

            # 3) 边界 blend(独立开关)
            if self.blend_enabled:
                chunk = self._blend(chunk)

            self._queue.extend(chunk.transpose(0, 1))
        a = self._queue.popleft()
        self._last_action = a.detach()
        return a


def build_refine_deploy_policy(act_policy, refine_ckpt, device,
                               blend_m: int = 10, mode: str = "full",
                               state_is_mdeg: bool = True, rlt_compressor_path=None):
    """加载 RefineNet ckpt(架构按 ckpt['args'] 复原)并包住 frozen ACTPolicy。

    mode: off=裸ACT直通 | blend=纯边界blend | residual=仅残差 | full=残差+blend
    支持 #63a Hattn(n_img_readout)/#69 M-q/#69 RLT-AE(需 rlt_compressor_path, 默认按 ckpt 旁找)。
    """
    assert mode in ("off", "blend", "residual", "full"), f"未知 mode: {mode}"
    ckpt = torch.load(Path(refine_ckpt), map_location=device, weights_only=False)
    targs = ckpt.get("args", {}) or {}
    use_state, use_ha, use_hi = _VARIANTS.get(targs.get("variant", "full"), (True, True, True))
    use_img = bool(targs.get("use_act_img_attn", False))
    n_readout = int(targs.get("n_img_readout", 1))            # #69 M-q
    use_rlt = bool(targs.get("use_act_rlt", False))           # #69 RLT-AE

    # RLT-AE: 加载重构预训练压缩器(部署时实时把 image tokens 压成 K 个 RLT token)
    comp = comp_mean = comp_std = None
    n_rlt = 8; rlt_d = 256
    if use_rlt:
        import sys
        repo = Path(__file__).resolve().parents[4]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from sim.train_rlt_compressor import RLTCompressor
        cpath = Path(rlt_compressor_path) if rlt_compressor_path else \
            repo / "outputs/sim/rlt_compressor_k8_d256.pt"
        cc = torch.load(cpath, map_location=device, weights_only=False)
        n_rlt, rlt_d = cc["k"], cc["d"]
        comp = RLTCompressor(k=n_rlt, d=rlt_d).to(device)
        comp.load_state_dict(cc["model_state"]); comp.eval()
        comp_mean, comp_std = cc["mean"], cc["std"]
        print(f"[refine_deploy] RLT 压缩器 {cpath.name} 加载(K={n_rlt} d={rlt_d} "
              f"explained_var={cc.get('val_explained_var', float('nan')):.3f})")

    refine = RefineNet(use_state=use_state, use_history_action=use_ha,
                       use_history_imu=use_hi,
                       use_act_hidden=bool(targs.get("use_act_hidden", False)),
                       use_act_img_attn=use_img, n_img_readout=n_readout,
                       use_act_rlt=use_rlt, n_rlt_tokens=n_rlt, act_rlt_dim=rlt_d).to(device)
    sd = ckpt["model_state"]
    te = sd.get("type_embed.weight")
    if te is not None and te.shape[0] < refine.type_embed.weight.shape[0]:
        sd["type_embed.weight"] = torch.cat(
            [te, refine.type_embed.weight.data[te.shape[0]:].clone()], 0)
    refine.load_state_dict(sd)
    refine.eval()
    print(f"[refine_deploy] ckpt ep{ckpt.get('epoch')} "
          f"variant={targs.get('variant', 'full(legacy)')} hf_mode={targs.get('hf_mode', 'fraction')} "
          f"img_attn={use_img}x{n_readout} rlt={use_rlt} | mode={mode} blend_m={blend_m}")
    return RefineNetDeployWrapper(
        act_policy, refine, ckpt["state_mean"], ckpt["state_std"],
        device=device, blend_m=blend_m,
        residual_enabled=mode in ("residual", "full"),
        blend_enabled=mode in ("blend", "full"),
        use_act_img=use_img, rlt_compressor=comp, rlt_mean=comp_mean, rlt_std=comp_std,
        state_is_mdeg=state_is_mdeg)
