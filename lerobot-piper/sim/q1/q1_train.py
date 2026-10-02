"""阶段一的训练与预测 —— 6 个模型 × 留一会话 × 训练种子。

每步均匀抽一个训练集，取它的全部主窗口（chunk 开头起点），拼接后算损失。
**停止规则**（用户 2026-10-02 定）：训练上限 10000 步，在 11 个检查点处快照；step* 由**内层留一会话**
选出（外层四个训练会话各轮流当验证会话，验证 ρ 的等权平均最小者；完整精度平局取最早）；
最终权重 = 在外层四个训练会话上同种子训到 step* 的权重。外层留出会话只在最后预测一次，不参与选步。
同种子 ⇒ 训到第 s 步的权重与长跑中第 s 步的快照逐位相同。CPU 单线程 + 确定性算法。
"""
from __future__ import annotations
import hashlib
import json
import time

import numpy as np

import q1_data as D
import q1_e2e as E
import q1_layered as LM
import q1_losses as LS
import q1_sensor as S
import q1_windows as W

MODELS = {"1a": ("layered", "a"), "1b": ("layered", "b"),
          "2a": ("tcn", "a"), "2b": ("tcn", "b"),
          "3a": ("tr", "a"), "3b": ("tr", "b")}
CONFIG = {"max_steps": 10000,
          "ckpts": (0, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000),
          "select": "inner leave-one-session-out; min equal-weight mean of per-session rho; earliest tie",
          "tie_rtol": 1e-12, "tie_atol": 1e-15,
          "lr": 1e-3, "clip": 1.0, "threads": 1, "device": "cpu"}
CODE_FILES = ("q1_data.py", "q1_windows.py", "q1_losses.py", "q1_layered.py", "q1_e2e.py",
              "q1_sensor.py", "q1_train.py", "q1_ar.py", "q1_g2ref.py", "q1_run.py", "q1_report.py")


def code_sha() -> str:
    h = hashlib.sha256()
    for f in CODE_FILES:
        h.update(f.encode())
        h.update((D.HERE / f).read_bytes())
    return h.hexdigest()


def config_hash() -> str:
    """选模规则与训练配置（上限、检查点、平局容差、优化器……）+ 代码指纹。
    锁定的是**规则**，不是各作业选中的 step*。"""
    blob = json.dumps({"config": CONFIG, "code": code_sha()}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def pick_step(V: dict) -> int:
    """`V[step]`（内层验证会话 ρ 的等权平均）→ 最小值的检查点；完整精度平局取最早。"""
    steps = sorted(V)
    vmin = min(V[s] for s in steps)
    tol = CONFIG["tie_atol"] + CONFIG["tie_rtol"] * abs(vmin)
    return int(min(s for s in steps if abs(V[s] - vmin) <= tol))


def split_fold(eps, held: str):
    """→ (训练折, 留出会话)。留出会话不存在即报错。"""
    te = [e for e in eps if e.sess == held]
    tr = [e for e in eps if e.sess != held]
    if not te:
        raise ValueError(f"留出会话 {held} 不存在")
    return tr, te


def _determinism(seed: int):
    import torch
    torch.set_num_threads(CONFIG["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)


def rebuild(model_id: str, st: dict):
    """按统计量重建网络结构（载入权重前用）。"""
    family, mode = MODELS[model_id]
    if family == "layered":
        return LM.make_servo(st)
    ch = E.CH_B if mode == "b" else E.CH_A
    return E.make_tcn(ch) if family == "tcn" else E.make_transformer(ch)


def build(model_id: str, tr_eps):
    """→ (模型, 统计量)。统计量只用训练折。调用前须已 `_determinism(seed)`。"""
    st = LM.joint_stats(tr_eps) if MODELS[model_id][0] == "layered" else E.norm_stats(tr_eps)
    return rebuild(model_id, st), st


def forward_windows(model_id, model, st, ep, windows):
    """→ (每窗 ê 或 None, 每窗 q̂（帧 t0+3 … s1+3）, 分层模型的 (历史, 滚动))。"""
    import torch
    family, mode = MODELS[model_id]
    if family == "layered":
        traj = LM.open_loop_traj(model, ep, max(w.t0 for w in windows)) if mode == "a" else None
        Q, U, M, F, _ = LM.window_inputs(model, ep, windows, mode, traj)
        qh = model.rollout(Q, U, M, F)
        return None, [qh[i, np.arange(w.n_scored) + 2] for i, w in enumerate(windows)], (Q, qh)
    X = torch.tensor(np.stack([E.window_input(ep, w, st, mode == "b") for w in windows]))
    Ee, Qq = E.decode(model(X), ep, windows, st)
    return Ee, Qq, None


def train_one(model_id: str, tr_eps, seed: int, ckpts=None, log=None) -> dict:
    """训到 `max(ckpts)` 步，在每个检查点（含第 0 步 = 初始权重）保存 `state_dict` 深拷贝。"""
    import torch
    ckpts = tuple(CONFIG["ckpts"] if ckpts is None else ckpts)
    if list(ckpts) != sorted(set(ckpts)) or ckpts[0] < 0:
        raise ValueError(f"检查点须严格递增且非负：{ckpts}")
    steps = ckpts[-1]
    _determinism(seed)
    model, st = build(model_id, tr_eps)
    norms = LS.loss_norms(tr_eps)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    rng = np.random.default_rng(seed)
    wins = [W.chunk_windows(e) for e in tr_eps]
    hist, t_start = [], time.time()
    snaps = {}

    def snap():
        return {k: v.detach().clone() for k, v in model.state_dict().items()}

    if 0 in ckpts:
        snaps[0] = snap()
    for it in range(steps):
        i = int(rng.integers(len(tr_eps)))
        Ee, Qq, _ = forward_windows(model_id, model, st, tr_eps[i], wins[i])
        loss = LS.joint_loss(tr_eps[i], torch.cat(Qq).double(), norms)
        if Ee is not None:
            loss = loss + LS.omega_loss(tr_eps[i], torch.cat(Ee).double(), norms)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["clip"])
        opt.step()
        hist.append(float(loss.item()))
        if it + 1 in ckpts:
            snaps[it + 1] = snap()
        if log and (it + 1) % 500 == 0:
            log(f"    {model_id} seed{seed} step {it + 1}/{steps} "
                f"loss {np.mean(hist[-500:]):.5f} {time.time() - t_start:.0f}s")
    return {"model": model, "stats": st, "norms": norms, "loss": hist, "sec": time.time() - t_start,
            "ckpt": snaps}


def with_weights(model_id: str, st: dict, state_dict):
    """按统计量重建网络并载入某个快照。"""
    model = rebuild(model_id, st)
    model.load_state_dict(state_dict)
    return model


def predict_windows(model_id, model, st, ep, windows, R=None, Wrs=None, C=None) -> list:
    """→ 每窗 ê `(n_scored,3)`。分层模型需要该折的 `Wrs`（1b 另需 `C`）。"""
    import torch
    family, mode = MODELS[model_id]
    with torch.no_grad():
        Ee, _, extra = forward_windows(model_id, model, st, ep, windows)
    if family != "layered":
        return [e.double().numpy() for e in Ee]
    Q, qh = extra
    Kr = qh.shape[1]
    preds = []
    for i, w in enumerate(windows):
        lo = w.t0 - D.HIST
        q_span = np.concatenate([Q[i].double().numpy().T, qh[i].double().numpy()], 0)
        u_span = W.span(W.visible(ep.cmd, w.t1, w.t0 + Kr), lo, w.t0 + Kr, ep.cmd[0])
        i_span = W.span(W.visible(ep.ideal, w.t1, w.t0 + Kr), lo, w.t0 + Kr, 0.0)
        om = S.omega_span(q_span, u_span, i_span, R, Wrs)
        ts = np.arange(w.t0, w.s1 + 1)
        eh = om[ts + D.ALIGN_L - lo] - ep.ideal[ts]
        if mode == "b":
            S.check_window_support(ep, w.t0)
            x = S.residual_history(ep, w.t0, R, Wrs)
            eh = eh + np.stack([x @ C[h] for h in range(w.n_scored)])
        preds.append(eh)
    return preds


def predict_episode(model_id, model, st, ep, windows, R=None, Wrs=None, C=None) -> np.ndarray:
    return W.stitch(ep.n, windows, predict_windows(model_id, model, st, ep, windows, R, Wrs, C))


def session_scores(model_id, model, st, eps, R=None, Wrs=None, C=None) -> dict:
    """主窗口预测 → {会话: [Σnum, Σden]}（会话内按集池化；ρ = num/den）。"""
    K = W.kernel()
    out: dict = {}
    for e in eps:
        n, d, _ = W.score(e.gap, predict_episode(model_id, model, st, e, W.chunk_windows(e), R, Wrs, C), K)
        a = out.setdefault(e.sess, [0.0, 0.0])
        a[0] += n
        a[1] += d
    return out


def state_dict_sha(state_dict) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def state_sha256(model) -> str:
    return state_dict_sha(model.state_dict())
