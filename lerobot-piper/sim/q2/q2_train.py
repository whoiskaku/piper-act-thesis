"""阶段二的训练、预测与计分 —— 预注册 §5、§6。

- 名义 2a：直接复用阶段一的 `train_one("2a", …)`（同结构、同损失、同确定性），只换训练数据（近期两个会话的裸跑）。
- 增量（两种形式）：名义冻结；只用给定的补正运行训练；损失作用在「名义 + 增量」的总预测上（ω 项 + 关节项，同阶段一）；
  δa 尺度、指令侧归一化统计与损失尺度只用给定的训练数据（按内层划分时就是那一个内层训练会话）。
- 选步规则同阶段一：检查点处快照，验证 ρ 的平均最小者，完整精度平局取最早；第 0 步 = 零响应，是有效选择。
CPU 单线程 + 确定性算法 ⇒ 同种子逐位复现；同种子下训到第 s 步的权重 = 长跑第 s 步的快照。
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import q1_data as D1
import q1_e2e as E
import q1_losses as LS
import q1_train as TR1
import q1_windows as W
import q2_models as Mo
import q2_windows as W2

CONFIG = {"max_steps": 10000,
          "ckpts": (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000),
          "select": "inner two-way leave-one-session-out; min mean rho; earliest tie",
          "tie_rtol": 1e-12, "tie_atol": 1e-15,
          "lr": 1e-3, "clip": 1.0, "threads": 1, "device": "cpu",
          "seeds": (0, 1, 2), "forms": Mo.FORMS,
          "n_basis": Mo.N_BASIS, "n_lag": Mo.N_LAG, "basis_sd": Mo.BASIS_SD, "scale_floor": Mo.SCALE_FLOOR,
          "gate_d": 0.05, "nonlin_margin": 0.02,
          "budget_hours": 24}
CODE_FILES_Q2 = ("q2_data.py", "q2_roster.json", "q2_windows.py", "q2_models.py", "q2_train.py",
                 "q2_run.py", "q2_report.py")
CODE_FILES_Q1 = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_e2e.py", "q1_train.py", "q1_g2ref.py", "q1_run.py")
Q2_DIR = Path(__file__).resolve().parent


def code_sha() -> str:
    """阶段二代码 + 只读复用的阶段一代码的指纹。"""
    h = hashlib.sha256()
    for d, files in ((Q2_DIR, CODE_FILES_Q2), (D1.HERE, CODE_FILES_Q1)):
        for f in files:
            h.update(f.encode())
            h.update((d / f).read_bytes())
    return h.hexdigest()


def config_hash() -> str:
    """选模规则、门槛与训练配置 + 代码指纹。锁定的是规则，不是各作业选中的步数。"""
    blob = json.dumps({"config": CONFIG, "code": code_sha()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick_step(V: dict) -> int:
    steps = sorted(V)
    vmin = min(V[s] for s in steps)
    tol = CONFIG["tie_atol"] + CONFIG["tie_rtol"] * abs(vmin)
    return int(min(s for s in steps if abs(V[s] - vmin) <= tol))


# ── 名义 ──────────────────────────────────────────────────────────────────
def train_nominal(bare_eps, seed: int, ckpts=None, log=None) -> dict:
    return TR1.train_one("2a", bare_eps, seed, ckpts=tuple(CONFIG["ckpts"] if ckpts is None else ckpts), log=log)


def nominal_windows(model, st, ep, windows) -> tuple:
    """冻结名义 2a 的逐窗 (ê, q̂)（numpy；无梯度）。"""
    import torch
    with torch.no_grad():
        Ee, Qq, _ = TR1.forward_windows("2a", model, st, ep, windows)
    return [x.double().numpy() for x in Ee], [x.double().numpy() for x in Qq]


def nominal_episode(model, st, ep) -> np.ndarray:
    ws = W.chunk_windows(ep)
    return W.stitch(ep.n, ws, nominal_windows(model, st, ep, ws)[0])


def h_episode(H, ep) -> np.ndarray:
    ws = W.chunk_windows(ep)
    return W.stitch(ep.n, ws, [W2.h_window(H, ep, w) for w in ws])


# ── 增量 ──────────────────────────────────────────────────────────────────
def train_increment(form: str, nom, corr_eps, seed: int, ckpts=None, log=None) -> dict:
    """nom = (冻结名义模型, 其统计量)。只用 corr_eps 训练增量；在每个检查点（含第 0 步）快照。"""
    import torch
    ckpts = tuple(CONFIG["ckpts"] if ckpts is None else ckpts)
    if list(ckpts) != sorted(set(ckpts)) or ckpts[0] < 0:
        raise ValueError(f"检查点须严格递增且非负：{ckpts}")
    nom_model, nom_st = nom
    TR1._determinism(seed)                                            # 先定种子再建模
    model = Mo.make(form)
    st_inc = E.norm_stats(corr_eps)
    scale = Mo.da_scale(corr_eps)
    norms = LS.loss_norms(corr_eps)
    data = []
    for e in corr_eps:                                                # 名义冻结 ⇒ 预测与输入只算一次
        ws = W.chunk_windows(e)
        En, Qn = nominal_windows(nom_model, nom_st, e, ws)
        X, Dd = Mo.window_inputs(e, ws, st_inc, scale)
        data.append((e, ws, torch.tensor(np.concatenate(En)), torch.tensor(np.concatenate(Qn)),
                     torch.tensor(X), torch.tensor(Dd)))
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = np.random.default_rng(seed)
    snaps, hist, t_start = {}, [], time.time()

    def snap():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    if 0 in ckpts:
        snaps[0] = snap()
    for it in range(ckpts[-1]):
        e, ws, En, Qn, X, Dd = data[int(rng.integers(len(data)))]
        dE, dQ = Mo.decode_increment(Mo.increment(model, form, X, Dd), ws, nom_st)
        loss = (LS.joint_loss(e, Qn + torch.cat(dQ).double(), norms)
                + LS.omega_loss(e, En + torch.cat(dE).double(), norms))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
        opt.step()
        hist.append(float(loss.item()))
        if it + 1 in ckpts:
            snaps[it + 1] = snap()
        if log and (it + 1) % 500 == 0:
            log(f"    {form} seed{seed} step {it + 1}/{ckpts[-1]} loss {np.mean(hist[-500:]):.5f} "
                f"{time.time() - t_start:.0f}s")
    return {"model": model, "st_inc": st_inc, "scale": scale, "norms": norms, "loss": hist,
            "sec": time.time() - t_start, "ckpt": snaps}


def with_increment(form: str, state_dict):
    model = Mo.make(form)
    model.load_state_dict(state_dict)
    return model


def increment_episode(form: str, model, st_inc, scale, nom_st, ep) -> np.ndarray:
    """拼接后的 Δê (n, 3)。"""
    import torch
    ws = W.chunk_windows(ep)
    X, Dd = Mo.window_inputs(ep, ws, st_inc, scale)
    with torch.no_grad():
        dE, _ = Mo.decode_increment(Mo.increment(model, form, X, Dd), ws, nom_st)
    return W.stitch(ep.n, ws, [x.double().numpy() for x in dE])


# ── 计分 ──────────────────────────────────────────────────────────────────
def ep_metrics(ep, ehat, K) -> dict:
    num, den, f3 = W.score(ep.gap, ehat, K)
    return {"name": ep.name, "fold": ep.fold, "arm": ep.arm, "num": num, "den": den, "f3": f3,
            "lead": W.lead_score(ep.gap, ehat, W.chunk_windows(ep), K)}


def pooled_rho(rows) -> float:
    return sum(r["num"] for r in rows) / sum(r["den"] for r in rows)


def beta_parts(ep, e_nom, d_e, K) -> tuple:
    """投影尺度 β 的一集分子与分母：⟨P(gap − ê_nom), P(Δê)⟩ 与 ‖P(Δê)‖²（汇总后再相除）。"""
    r, h = W.P(ep.gap - e_nom, K), W.P(d_e, K)
    return float((r * h).sum()), float((h * h).sum())
