#!/usr/bin/env python3
"""Stage 3b — Train RefineNet on Phase 1 sim paired data.

Per RESEARCH_PLAN_v4.3 §5 Stage 3b + Run #1 diagnosis (see STAGE3B_RESULTS.md):
  - Output: raw rad (residual ΔA on top of frozen ACT_output)
  - current_state input: z-scored externally (stats from train split)
  - HEADLINE metric: active-joint (j1/j2/j3/grip) mean hf_fraction.
    jerk_RMS is kept ONLY as a secondary/continuity metric (proven unreliable).
  - Loss:  L_BC + λ_hf·L_hf_fraction + λ_delta·|ΔA| (+ optional λ_acc·L_acc)
       L_hf_fraction uses a stopgrad denominator (no low-freq-inflation cheat).
  - AdamW, lr=3e-4, batch=64, epochs=200

Run plan:
  Run A  L_BC only:                 --lambda-hf 0 --lambda-delta 0 --lambda-acc 0
  Run B  main path:                 --lambda-hf 0.05 --lambda-delta 0.2
  (λ ranges from sim/analyze_loss_scales.py: λ_hf∈[0.024,0.072], λ_delta∈[0.10,0.30])

Usage:
    python sim/train_refine_net.py --lambda-hf 0 --lambda-delta 0 --epochs 200   # Run A
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.common.policies.act.refine_net import RefineNet
from sim.refine_net_dataset import RefinePhase1Dataset


H5_PATH = REPO_ROOT / "outputs/sim/phase1_paired_data.h5"
FPS = 30.0
DT = 1.0 / FPS
HF_CUTOFF = 5.0          # Hz; energy above this = "jitter" we want to remove
EPS = 1e-8

# Active joints in the v3 pick-place task (j4/j5/j6 are stationary → excluded
# from the headline hf metric so their 0-energy doesn't dilute the mean).
ACTIVE_JOINTS = [0, 1, 2, 6]   # j1, j2, j3, grip(j7)


# ─── Spectral helpers ──────────────────────────────────────────────
def hf_power(actions: torch.Tensor, fs: float = FPS, cutoff: float = None):
    """Per-(B, joint) high-freq power and total AC power. actions: (B, T, A).

    Returns (E_high, E_total), each shaped (B, A). DC removed, Hann-windowed.
    cutoff=None → 读当前模块全局 HF_CUTOFF(支持 --hf-cutoff 运行时覆盖;
    默认参数在 import 时绑定, 不能直接写 =HF_CUTOFF)。
    """
    if cutoff is None:
        cutoff = HF_CUTOFF
    B, T, A = actions.shape
    x_ac = actions - actions.mean(dim=1, keepdim=True)
    w = torch.hann_window(T, device=actions.device).view(1, T, 1)
    X = torch.fft.rfft(x_ac * w, dim=1)
    P = X.abs() ** 2                                       # (B, F, A)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(actions.device)
    hf_mask = (freqs > cutoff).view(1, -1, 1)
    E_high = (P * hf_mask).sum(dim=1)                      # (B, A)
    E_total = P.sum(dim=1)                                 # (B, A)
    return E_high, E_total


def hf_fraction_per_joint(actions: torch.Tensor) -> torch.Tensor:
    """Per-(B, joint) hf-energy fraction in [0,1] (no stopgrad — eval metric)."""
    E_high, E_total = hf_power(actions)
    return E_high / (E_total + EPS)                        # (B, A)


def hf_fraction_loss(actions: torch.Tensor, active=ACTIVE_JOINTS) -> torch.Tensor:
    """Smoothness loss = mean active-joint hf fraction with stopgrad denominator.

    Detaching E_total stops the model lowering the *fraction* by inflating
    low-freq energy; gradient flows only through the high-freq numerator.
    """
    E_high, E_total = hf_power(actions)
    frac = E_high / (E_total.detach() + EPS)               # (B, A)
    return frac[:, active].mean()


def hf_abs_loss(refined: torch.Tensor, ref_signal: torch.Tensor,
                active=ACTIVE_JOINTS) -> torch.Tensor:
    """Absolute high-freq loss (2026-06-10) — push refined's >5Hz energy below
    the *frozen input* (ACT). Denominator = E_high(ref_signal), detached:
    a FIXED per-(B,joint) reference (the input being refined), NOT refined's own
    total energy → not the fraction/denominator trap. <1 = genuine absolute
    smoothing; only this objective can reward Δ that anti-phase-cancels ACT jitter.
    """
    Eh_r, _ = hf_power(refined)
    Eh_a, _ = hf_power(ref_signal)
    # ratio-of-SUMS (energy-weighted), NOT mean-of-ratios: the latter is dominated
    # by near-static joints (E_high(ACT)≈0 → ratio blows up) and never touches the
    # high-energy joints that carry the actual absolute jitter (mean-of-ratios trap).
    return Eh_r[:, active].sum() / (Eh_a[:, active].sum().detach() + EPS)


# FIR high-pass for the uniform-weight absolute loss (hf_mode=absfir).
# The Hann window in hf_power has ~0 weight at chunk edges — the abs-trained
# model learned to DUMP high-freq garbage there (t[0:5] 9×, t[95:100] 1.8×)
# while genuinely smoothing the core. A linear-phase FIR applied along time
# weights every step equally → no blind spot to game.
from scipy.signal import firwin as _firwin
_FIR_TAPS = torch.from_numpy(
    _firwin(31, HF_CUTOFF / (FPS / 2.0), pass_zero=False).astype(np.float32))


def _fir_highpass(x: torch.Tensor) -> torch.Tensor:
    """x (B,T,A) → 5Hz FIR high-passed along T, same length.

    AC-remove + reflect padding: zero-padding on the raw (~0.5 rad) signal
    creates a boundary transient ~5 orders of magnitude above real jitter —
    the loss then only "sees" that artifact (and drags chunk edges toward 0,
    destroying L1, as the first absfir/absboth runs did).
    """
    B, T, A = x.shape
    k = _FIR_TAPS.to(x.device, x.dtype)
    x = x - x.mean(dim=1, keepdim=True)
    xt = x.permute(0, 2, 1).reshape(B * A, 1, T)
    pad = k.numel() // 2
    xt = torch.nn.functional.pad(xt, (pad, pad), mode="reflect")
    y = torch.nn.functional.conv1d(xt, k.view(1, 1, -1))
    return y.reshape(B, A, T).permute(0, 2, 1)


def hf_absfir_loss(refined: torch.Tensor, ref_signal: torch.Tensor,
                   active=ACTIVE_JOINTS) -> torch.Tensor:
    """Uniform-weight absolute high-freq loss: FIR-highpass energy of refined
    over that of the frozen input (detached), ratio-of-sums."""
    Er = _fir_highpass(refined)[:, :, active].pow(2).sum()
    Ea = _fir_highpass(ref_signal)[:, :, active].pow(2).sum().detach()
    return Er / (Ea + EPS)


# ─── Controller surrogate C(·): commanded action → measured state (#71) ──
# fit_controller_surrogate.py 拟合的实测 action→state 低通(per-joint FIR)。
# controller-aware 训练把 HF 平滑损失搬到 C(refined) 上算 → 残差只在 2-4.5Hz 透传带
# (实测增益~0.2)拿梯度, 不在被控制器砍光的 >5Hz(增益~0.04)白学; 且 #66 残差在
# 2-4.5Hz 的放大(真机 IMU 变差元凶)会被直接惩罚, 因为该带 C 透传。
_CTRL_KERNELS = None   # (7, L) torch tensor, lazy-loaded


def load_controller(npz_path=None):
    global _CTRL_KERNELS
    p = Path(npz_path) if npz_path else (REPO_ROOT / "outputs/sim/controller_surrogate.npz")
    d = np.load(p)
    _CTRL_KERNELS = torch.from_numpy(d["kernels"].astype(np.float32))   # (7,L) 对称线性相位
    print(f"[#71] controller surrogate loaded: {p} kernels{tuple(_CTRL_KERNELS.shape)}")


def apply_controller(x: torch.Tensor) -> torch.Tensor:
    """x (B,T,A) → C(x): per-channel 零相位 FIR(depthwise, reflect pad, 居中 'same')。
    对称核 → cross-correlation=convolution, 无需翻转; 居中输出 = 零相位幅频。"""
    B, T, A = x.shape
    k = _CTRL_KERNELS.to(x.device, x.dtype)            # (7,L)
    L = k.shape[1]; pad = L // 2
    w = k[:A].unsqueeze(1)                             # (A,1,L)
    xt = x.permute(0, 2, 1)                            # (B,A,T)
    xt = torch.nn.functional.pad(xt, (pad, pad), mode="reflect")
    y = torch.nn.functional.conv1d(xt, w, groups=A)    # (B,A,T)
    return y.permute(0, 2, 1)


def _accel(x: torch.Tensor) -> torch.Tensor:
    """proxy IMU = 沿 T 的二阶差分(加速度)。(B,T,A)→(B,T-2,A)。"""
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def deploy_transform(x: torch.Tensor, controller=False, proxy_imu=False) -> torch.Tensor:
    """把命令空间信号搬到部署可观测空间: C(·)=控制器透传后 state, 再可选二阶差分=代理IMU。"""
    if controller:
        x = apply_controller(x)
    if proxy_imu:
        x = _accel(x)
    return x


# ─── Time-domain regularizers ──────────────────────────────────────
def l_delta_term(delta: torch.Tensor) -> torch.Tensor:
    """Residual-shrinkage / trust-region: mean |ΔA|."""
    return delta.abs().mean()


def l_acc_term(actions: torch.Tensor) -> torch.Tensor:
    """Mean |second difference| (no /dt²). Optional, weak discriminator."""
    d2 = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return d2.abs().mean()


# ─── Secondary / continuity metric ─────────────────────────────────
def jerk_rms(actions: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Per-batch jerk RMS — SECONDARY metric only, NOT in the loss.

    Shown in Run #1 diagnosis to be dominated by bulk-motion amplitude rather
    than the high-freq jitter we care about. See STAGE3B_RESULTS.md.
    """
    v = (actions[:, 1:] - actions[:, :-1]) / dt
    a = (v[:, 1:] - v[:, :-1]) / dt
    j = (a[:, 1:] - a[:, :-1]) / dt
    return (j ** 2).mean().sqrt()


# ─── Train / Eval loops ────────────────────────────────────────────
def run_epoch(model, loader, optimizer, lambdas, device, train=True, input_key="ACT_output",
              zero_act_img=False, hf_mode="fraction", controller_aware=False, proxy_imu=False,
              combine_cmd=False):
    """lambdas = dict(hf=..., delta=..., acc=...). hf_mode: fraction|abs.
    controller_aware: HF 平滑损失在 C(refined) 上算(#71); proxy_imu: 再二阶差分=代理IMU。
    combine_cmd: controller_aware 时再加命令空间损失(0.5/0.5), 堵住 >5Hz 命令倾倒盲区。"""
    model.train(train)
    n = 0
    acc = dict(  # running sums (× batch size)
        l_bc=0.0, l_hf=0.0, l_delta=0.0, l_acc=0.0, loss=0.0,
        jerk_refined=0.0, jerk_act=0.0,
        l_bc_act=0.0,            # guardrail ref: |ACT_output - A_clean|
    )
    # per-active-joint hf fraction sums for refined / act / clean
    hf_ref = np.zeros(len(ACTIVE_JOINTS))
    hf_act = np.zeros(len(ACTIVE_JOINTS))
    hf_cln = np.zeros(len(ACTIVE_JOINTS))
    # absolute high-freq energy sums (active joints) → energy-weighted abs ratio
    eh_ref_sum = 0.0; eh_act_sum = 0.0
    # FIR (uniform-weight) absolute energy sums — no Hann edge blind spot
    fir_ref_sum = 0.0; fir_act_sum = 0.0
    # controller-space (#71): 同样的绝对高频比, 但在 C(·)(可选+二阶差分) 透传后算
    ctrl_eh_ref_sum = 0.0; ctrl_eh_act_sum = 0.0
    ctrl_fir_ref_sum = 0.0; ctrl_fir_act_sum = 0.0

    grad_context = torch.enable_grad() if train else torch.no_grad()
    with grad_context:
        for batch in loader:
            B = batch["ACT_output"].shape[0]
            act_output       = batch[input_key].to(device, non_blocking=True)  # 待精修输入(ACT_output 或 A_noisy, #62)
            history_action   = batch["history_action"].to(device, non_blocking=True)
            history_imu      = batch["history_imu"].to(device, non_blocking=True)
            history_pad_mask = batch["history_pad_mask"].to(device, non_blocking=True)
            current_state    = batch["current_state"].to(device, non_blocking=True)
            A_clean          = batch["A_clean"].to(device, non_blocking=True)
            act_hidden = (batch["act_hidden"].to(device, non_blocking=True)
                          if "act_hidden" in batch else None)
            act_img = (batch["act_img"].to(device, non_blocking=True)
                       if "act_img" in batch else None)
            if zero_act_img and act_img is not None:
                act_img = torch.zeros_like(act_img)        # 零-image 对照: 同架构无真实感知信息
            act_rlt = (batch["act_rlt"].to(device, non_blocking=True)
                       if "act_rlt" in batch else None)    # #69 RLT-AE 压缩 token

            refined, delta = model(
                act_output, history_action, history_imu, current_state,
                history_pad_mask=history_pad_mask, act_hidden=act_hidden,
                act_img=act_img, act_rlt=act_rlt,
            )

            l_bc = nn.functional.l1_loss(refined, A_clean)   # 保真损失始终在命令空间
            # controller-aware: HF 平滑损失搬到 C(refined) vs C(ACT)(可选再二阶差分=代理IMU)。
            # ratio-of-sums 损失尺度不变 → C 后幅度虽小(透传~0.2)梯度不塌。fraction 模式不变。
            def _hf_pair(r, a):
                if hf_mode == "abs":
                    return hf_abs_loss(r, a)
                if hf_mode == "absfir":
                    return hf_absfir_loss(r, a)
                # absboth: Hann-FFT(频率尖锐/边缘盲) + FIR(时间均匀/过渡带软) 盲区互补
                return 0.5 * (hf_abs_loss(r, a) + hf_absfir_loss(r, a))

            if hf_mode == "fraction":
                l_hf = hf_fraction_loss(refined)
            else:
                r_hf = deploy_transform(refined, controller_aware, proxy_imu)
                a_hf = deploy_transform(act_output, controller_aware, proxy_imu)
                l_hf = _hf_pair(r_hf, a_hf)
                # combine_cmd: 再加命令空间损失 → 透传带强调 + >5Hz 命令仍被罚(防倾倒盲区)
                if combine_cmd and (controller_aware or proxy_imu):
                    l_hf = 0.5 * l_hf + 0.5 * _hf_pair(refined, act_output)
            l_dl = l_delta_term(delta)
            l_ac = l_acc_term(refined)
            loss = (l_bc
                    + lambdas["hf"] * l_hf
                    + lambdas["delta"] * l_dl
                    + lambdas["acc"] * l_ac)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            acc["l_bc"]    += l_bc.item() * B
            acc["l_hf"]    += l_hf.item() * B
            acc["l_delta"] += l_dl.item() * B
            acc["l_acc"]   += l_ac.item() * B
            acc["loss"]    += loss.item() * B
            acc["jerk_refined"] += jerk_rms(refined).item() * B
            acc["jerk_act"]     += jerk_rms(act_output).item() * B
            acc["l_bc_act"]     += nn.functional.l1_loss(act_output, A_clean).item() * B

            with torch.no_grad():
                fr = hf_fraction_per_joint(refined).mean(dim=0).cpu().numpy()
                fa = hf_fraction_per_joint(act_output).mean(dim=0).cpu().numpy()
                fc = hf_fraction_per_joint(A_clean).mean(dim=0).cpu().numpy()
                eh_r, _ = hf_power(refined); eh_a, _ = hf_power(act_output)
                eh_ref_sum += eh_r[:, ACTIVE_JOINTS].sum().item()
                eh_act_sum += eh_a[:, ACTIVE_JOINTS].sum().item()
                fir_ref_sum += _fir_highpass(refined)[:, :, ACTIVE_JOINTS].pow(2).sum().item()
                fir_act_sum += _fir_highpass(act_output)[:, :, ACTIVE_JOINTS].pow(2).sum().item()
                if controller_aware or proxy_imu:
                    rC = deploy_transform(refined, controller_aware, proxy_imu)
                    aC = deploy_transform(act_output, controller_aware, proxy_imu)
                    ce_r, _ = hf_power(rC); ce_a, _ = hf_power(aC)
                    ctrl_eh_ref_sum += ce_r[:, ACTIVE_JOINTS].sum().item()
                    ctrl_eh_act_sum += ce_a[:, ACTIVE_JOINTS].sum().item()
                    ctrl_fir_ref_sum += _fir_highpass(rC)[:, :, ACTIVE_JOINTS].pow(2).sum().item()
                    ctrl_fir_act_sum += _fir_highpass(aC)[:, :, ACTIVE_JOINTS].pow(2).sum().item()
            hf_ref += fr[ACTIVE_JOINTS] * B
            hf_act += fa[ACTIVE_JOINTS] * B
            hf_cln += fc[ACTIVE_JOINTS] * B
            n += B

    out = {k: v / n for k, v in acc.items()}
    out["hf_refined_joints"] = (hf_ref / n).tolist()
    out["hf_act_joints"]     = (hf_act / n).tolist()
    out["hf_clean_joints"]   = (hf_cln / n).tolist()
    out["hf_refined"] = float((hf_ref / n).mean())   # headline metric
    out["hf_act"]     = float((hf_act / n).mean())
    out["hf_clean"]   = float((hf_cln / n).mean())
    out["hf_ratio"]   = out["hf_refined"] / (out["hf_act"] + EPS)  # primary criterion (fraction)
    out["abs_hf_ratio"] = eh_ref_sum / (eh_act_sum + EPS)          # absolute high-freq (energy-weighted); <1 = real smoothing
    out["absfir_hf_ratio"] = fir_ref_sum / (fir_act_sum + EPS)     # FIR uniform-weight absolute; immune to Hann edge gaming
    out["absmax_ratio"] = max(out["abs_hf_ratio"], out["absfir_hf_ratio"])  # worst-case of the two (anti-gaming selection)
    if ctrl_eh_act_sum > 0:    # controller-space (#71): 部署可观测空间的绝对高频比, <1=真降透传后高频
        out["ctrl_abs_ratio"] = ctrl_eh_ref_sum / (ctrl_eh_act_sum + EPS)
        out["ctrl_absfir_ratio"] = ctrl_fir_ref_sum / (ctrl_fir_act_sum + EPS)
        out["ctrl_absmax_ratio"] = max(out["ctrl_abs_ratio"], out["ctrl_absfir_ratio"])
    return out


def grade(ratio: float) -> str:
    if ratio < 0.1:  return "EXCELLENT"
    if ratio < 0.25: return "good"
    if ratio < 0.5:  return "pass"
    return "FAIL"


# ─── Main ──────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-mode", choices=["fraction", "abs", "absfir", "absboth"], default="fraction",
                    help="fraction=招牌(E_hi/E_tot stopgrad); abs=绝对(Hann-FFT, 有边缘盲区); "
                         "absfir=绝对(FIR 均匀权重, 无边缘盲区可倒垃圾)")
    ap.add_argument("--lambda-hf", type=float, default=0.0,
                    help="weight on active-joint hf_fraction loss (main smoothness term)")
    ap.add_argument("--lambda-delta", type=float, default=0.0,
                    help="weight on |ΔA| residual-shrinkage / trust-region")
    ap.add_argument("--lambda-acc", type=float, default=0.0,
                    help="weight on |2nd-diff| (optional, weak; default off)")
    ap.add_argument("--hf-cutoff", type=float, default=5.0,
                    help="平滑损失/度量的高频截止(Hz); #66 发现 5Hz 截止留 2-4.5Hz 放大盲区, "
                         "降到 ~2.5 贴任务带宽(0.6Hz)可纳入该带惩罚")
    ap.add_argument("--tag", type=str, default=None,
                    help="run name suffix; auto-built from λ if omitted")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--variant", choices=["S0", "S1", "S2", "S3", "full"],
                    default="full", help="输入消融: S0=仅ACT, S1=+state, S2=+state+hist_action, S3/full=全部")
    ap.add_argument("--input-signal", choices=["act_output", "a_noisy"], default="act_output",
                    help="#62 待精修输入: act_output(现状) 或 a_noisy(去噪对照)")
    ap.add_argument("--use-act-hidden", action="store_true",
                    help="#63 加 ACT encoder 摘要作 context token(变体 S0+H)")
    ap.add_argument("--use-act-img-attn", action="store_true",
                    help="#63(a) 对 ACT image tokens 做注意力读出作 context token(变体 S0+Hattn)")
    ap.add_argument("--zero-act-img", action="store_true",
                    help="#63(a) 对照: act_img 喂零(同架构无真实感知信息) → 隔离 信息 vs 参数容量")
    ap.add_argument("--n-img-readout", type=int, default=1,
                    help="#69 M-q: image-token 注意力读出的 query 数(1=#63a现状, >1 加容量)")
    ap.add_argument("--use-act-rlt", action="store_true",
                    help="#69 RLT-AE: 用重构预训练压缩器的 K 个冻结 token 作 context(需 H5 有 act_rlt_tokens)")
    ap.add_argument("--controller-aware", action="store_true",
                    help="#71 把 HF 平滑损失搬到实测控制器透传后 C(refined)(action→state 低通) → "
                         "残差只在 2-4.5Hz 透传带学、惩罚 #66 中频放大; 需 controller_surrogate.npz")
    ap.add_argument("--proxy-imu", action="store_true",
                    help="#71 在 C(·) 后再二阶差分=代理 IMU 加速度, HF 损失更贴真机 IMU 口径")
    ap.add_argument("--combine-cmd", action="store_true",
                    help="#71 controller-aware 时再加命令空间损失(0.5/0.5), 堵住 >5Hz 命令倾倒盲区")
    ap.add_argument("--controller-npz", type=Path, default=None,
                    help="控制器代理核路径(默认 outputs/sim/controller_surrogate.npz)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--h5", type=Path, default=H5_PATH)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "outputs/train/refine_net")
    args = ap.parse_args()

    lambdas = {"hf": args.lambda_hf, "delta": args.lambda_delta, "acc": args.lambda_acc}

    # #66: 运行时覆盖高频截止(影响 hf_power 的 freq mask + FIR taps + 全部 hf 度量)
    global HF_CUTOFF, _FIR_TAPS
    if args.hf_cutoff != HF_CUTOFF:
        HF_CUTOFF = args.hf_cutoff
        _FIR_TAPS = torch.from_numpy(
            _firwin(31, HF_CUTOFF / (FPS / 2.0), pass_zero=False).astype(np.float32))
        print(f"[#66] HF_CUTOFF 覆盖为 {HF_CUTOFF}Hz(FIR taps 已重算)")

    # #71: controller-aware 需载入实测控制器代理核
    if args.controller_aware or args.proxy_imu:
        load_controller(args.controller_npz)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 输入消融变体 → RefineNet 输入开关
    VARIANTS = {  # (use_state, use_history_action, use_history_imu)
        "S0": (False, False, False),
        "S1": (True,  False, False),
        "S2": (True,  True,  False),
        "S3": (True,  True,  True),
        "full": (True, True, True),
    }
    use_state, use_ha, use_hi = VARIANTS[args.variant]

    input_key = "A_noisy" if args.input_signal == "a_noisy" else "ACT_output"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _hsfx = ('+H' if args.use_act_hidden else '') + ('+Hattn' if args.use_act_img_attn else '')
    _hsfx += (f'x{args.n_img_readout}' if (args.use_act_img_attn and args.n_img_readout != 1) else '')
    _hsfx += ('+RLT' if args.use_act_rlt else '')
    _hsfx += ('+zero' if args.zero_act_img else '')
    _hsfx += {'abs': '+abs', 'absfir': '+absfir', 'absboth': '+absboth'}.get(args.hf_mode, '')
    _hsfx += ('' if args.hf_cutoff == 5.0 else f'+cut{args.hf_cutoff:g}')
    _hsfx += ('+ctrlimu' if (args.controller_aware and args.proxy_imu)
              else '+ctrl' if args.controller_aware else '')
    _hsfx += ('+cmb' if (args.combine_cmd and (args.controller_aware or args.proxy_imu)) else '')
    _ssfx = '' if args.seed == 42 else f'_s{args.seed}'   # seed42 保持原 tag(兼容已有目录)
    tag = args.tag or (
        f"{args.variant}{_hsfx}_in-{args.input_signal}_hf{args.lambda_hf}_dl{args.lambda_delta}{_ssfx}"
    )
    out_dir = args.out_dir / f"phase1_{tag}_lr{args.lr}_bs{args.batch_size}_ep{args.epochs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[device] {device}")
    print(f"[out_dir] {out_dir}")
    print(f"[loss] L_BC + {args.lambda_hf}·hf + {args.lambda_delta}·|ΔA| + {args.lambda_acc}·acc")

    # ---- Data ----
    print(f"[load] {args.h5}")
    train_ds = RefinePhase1Dataset(args.h5, "train")
    state_mean, state_std = train_ds.compute_state_stats()
    print(f"  state_mean: {state_mean}")
    print(f"  state_std:  {state_std}")
    train_ds.state_mean = state_mean
    train_ds.state_std = state_std
    val_ds = RefinePhase1Dataset(args.h5, "val", state_mean, state_std)
    print(f"  train n = {len(train_ds)}, val n = {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True,
    )

    # ---- Model ----
    model = RefineNet(use_state=use_state, use_history_action=use_ha,
                      use_history_imu=use_hi, use_act_hidden=args.use_act_hidden,
                      use_act_img_attn=args.use_act_img_attn,
                      n_img_readout=args.n_img_readout,
                      use_act_rlt=args.use_act_rlt).to(device)
    print(f"[variant] {args.variant}{'+H' if args.use_act_hidden else ''}: "
          f"use_state={use_state} use_hist_action={use_ha} use_hist_imu={use_hi} use_act_hidden={args.use_act_hidden}")
    n_params = model.num_parameters()
    print(f"[model] RefineNet n_params = {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine LR schedule with linear warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Save run config ----
    cfg_dump = {
        "lambda_hf": args.lambda_hf, "lambda_delta": args.lambda_delta,
        "hf_mode": args.hf_mode,
        "lambda_acc": args.lambda_acc, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "active_joints": ACTIVE_JOINTS, "hf_cutoff": HF_CUTOFF,
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_params": n_params,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_dump, indent=2))

    # ---- Training loop ----
    history = []
    # Model-selection criterion: lowest hf_ratio subject to BC guardrail.
    # hf_mode=abs selects on abs_hf_ratio (the real absolute-smoothing objective);
    # fraction keeps the legacy headline metric. (val loss dominated by λ choice.)
    sel_key = {"abs": "abs_hf_ratio", "absfir": "absfir_hf_ratio",
               "absboth": "absmax_ratio"}.get(args.hf_mode, "hf_ratio")
    if args.controller_aware or args.proxy_imu:   # #71 按部署可观测空间(透传后)选模型
        sel_key = "ctrl_absmax_ratio"
    best_hf_ratio = float("inf")
    t_start = time.time()

    for epoch in range(args.epochs):
        t_ep = time.time()
        train_stats = run_epoch(model, train_loader, optimizer, lambdas, device, train=True, input_key=input_key, zero_act_img=args.zero_act_img, hf_mode=args.hf_mode, controller_aware=args.controller_aware, proxy_imu=args.proxy_imu, combine_cmd=args.combine_cmd)
        val_stats = run_epoch(model, val_loader, None, lambdas, device, train=False, input_key=input_key, zero_act_img=args.zero_act_img, hf_mode=args.hf_mode, controller_aware=args.controller_aware, proxy_imu=args.proxy_imu, combine_cmd=args.combine_cmd)
        scheduler.step()

        elapsed_ep = time.time() - t_ep
        elapsed_total = time.time() - t_start
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch, "lr": cur_lr,
            "train": train_stats, "val": val_stats, "time_s": elapsed_ep,
        })

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"[ep {epoch:3d}/{args.epochs}] lr={cur_lr:.2e}  "
                  f"L_BC={val_stats['l_bc']:.4f}(act {val_stats['l_bc_act']:.4f})  "
                  f"|ΔA|={val_stats['l_delta']:.4f}  "
                  f"hf_ref={val_stats['hf_refined']:.4f} hf_act={val_stats['hf_act']:.4f} "
                  f"hf_cln={val_stats['hf_clean']:.4f}  "
                  f"ratio={val_stats['hf_ratio']:.3f} [{grade(val_stats['hf_ratio'])}]  "
                  f"abs={val_stats['abs_hf_ratio']:.3f} fir={val_stats['absfir_hf_ratio']:.3f}  "
                  + (f"ctrl_abs={val_stats['ctrl_abs_ratio']:.3f} ctrl_fir={val_stats['ctrl_absfir_ratio']:.3f}  "
                     if 'ctrl_absmax_ratio' in val_stats else "")
                  + f"[{elapsed_ep:.1f}s, tot {elapsed_total/60:.1f}m]")

        # Save best ckpt by headline metric, with BC guardrail
        # (refined BC must not exceed the ACT-baseline BC by more than 10%).
        bc_ok = val_stats["l_bc"] <= 1.1 * val_stats["l_bc_act"]
        if bc_ok and val_stats[sel_key] < best_hf_ratio:
            best_hf_ratio = val_stats[sel_key]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / "ckpt_best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "state_mean": state_mean, "state_std": state_std,
                "args": vars(args), "val_stats": val_stats,
            }, out_dir / f"ckpt_ep{epoch:04d}.pt")

    torch.save({
        "epoch": args.epochs - 1, "model_state": model.state_dict(),
        "state_mean": state_mean, "state_std": state_std, "args": vars(args),
    }, out_dir / "ckpt_final.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final = history[-1]["val"]
    print(f"\n[done] total time {(time.time() - t_start) / 60:.1f}m")
    print(f"[done] best hf_ratio (BC-guarded) = {best_hf_ratio:.3f} [{grade(best_hf_ratio)}]")
    print(f"[done] final val: hf_ref={final['hf_refined']:.4f} hf_act={final['hf_act']:.4f} "
          f"ratio={final['hf_ratio']:.3f}  L_BC={final['l_bc']:.4f}(act {final['l_bc_act']:.4f})")
    print(f"[done] final per-joint hf refined {['%.4f'%x for x in final['hf_refined_joints']]} "
          f"vs act {['%.4f'%x for x in final['hf_act_joints']]} (j1,j2,j3,grip)")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
