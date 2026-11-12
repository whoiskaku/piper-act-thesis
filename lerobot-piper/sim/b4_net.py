"""B4 的补正网络：数据（方案 `reports/PLAN_b4.md` §5.3；实现计划任务 8，训练在任务 9）。

  python -u sim/b4_net.py build --P 8 --F 8     # 训练、验证里验收通过的教师 → outputs/b4/net/data_P8_F8.npz
  python -u sim/b4_net.py build --P 8 --F 16 --N 10   # 整段版 ①（reports/PLAN_b4_seq.md）的数据 → outputs/b4/net_seq/
  python -u sim/b4_net.py select                # 候选 (P, F) × hidden × 3 种子，按验证集损失选 → selection.json
  python -u sim/b4_net.py curve                 # 选中的组合下，训练 10 / 20 / 30 条时验证集的 R² → curve.json、curve.png
  python -u sim/b4_net.py freeze                # 冻结选中的那一个（只能冻结一次）→ frozen.json、frozen.pt
  python -u sim/b4_net.py seq [--N 10]          # 整段版 ①：3 个种子，按验证集第一帧的损失选 → net_seq/selection.json
  python -u sim/b4_net.py check                 # 整段版 ①：离线检查与上真机门槛 → net_seq/check.json
  python -u sim/b4_net.py freeze --seq          # 冻结整段版 ①（真机验证集检查之后）→ net_seq/frozen.json

按录制帧下标（去掉回放开头的保持段）：
- X = 第 k 帧当时可得的网络输入：由记下的整段预测复算（te_states + online_input），P、F 与记录时相同时必须等于记下的输入（约束 2）；
- Y = 教师实发的补正 = 冻结的教师指令 − 冻结的在线 Shift（已含窗，约束 1）；
- Wt = 录制时的在线窗，作损失权重（窗 = 0 的帧 Y 一定是 0）。
测试集一律不进；验收没通过或没跑完的教师不进。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b4_common as C  # noqa: E402
import b4_positions as P  # noqa: E402
import b4_teacher as B  # noqa: E402
import repeat_reversal as RR  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

NET = C.OUT / "net"
NET_SEQ = C.OUT / "net_seq"                       # 整段版 ① 的一切输出，和一帧版分开
CANDIDATES = [(4, 4), (8, 8), (8, 16)]
HIDDEN, SEEDS = (64, 128), (0, 1, 2)
JOINTS = [f"j{i + 1}" for i in O.ACTIVE]          # j1 j2 j3 j5 j6


def assemble(log: dict, seq: dict, teacher: dict, P: int, F: int) -> dict:
    """一条轨迹 → {"X" (N, (P+F)·7) float32, "Y" (N, 5), "Wt" (N, 5)}，N = 录制帧数。"""
    a_pi = np.asarray(log["a_pi"])
    _, ens = O.te_states(log["chunk"])
    X = np.stack([O.online_input(a_pi, ens[k], k, P, F) for k in range(len(a_pi))])
    if P == int(log["P"]) and F == int(log["F"]):
        gap = float(np.abs(X - log["x"]).max())
        if gap > 1e-5:
            raise SystemExit(f"⛔ 复算的网络输入与控制回路记下的不同（最大差 {gap:.2e}）")
    act, H = list(O.ACTIVE), RR.HOLD
    Y = teacher["cmd7"][H:, act].astype(float) - seq["s7"][H:, act].astype(float)
    Wt = np.asarray(log["w"], float)
    if Y[Wt == 0].any():
        raise SystemExit("⛔ 窗 = 0 的帧上教师补正不为 0：标签和部署时的窗对不上")
    return {"X": X, "Y": Y, "Wt": Wt}


def assemble_seq(log: dict, seq: dict, teacher: dict, P: int, F: int, N: int) -> dict:
    """整段版 ①（方案 `reports/PLAN_b4_seq.md` §4.2）的一条轨迹，n = 录制帧数：
    X 同 `assemble`（含与记录的核对）；R (n, N, 5) = 第 k 帧当时 ACT 对第 k … k+N−1 帧的值（第 k 帧用最终值，之后用部分融合值）；
    Y (n, N·5) = 教师实发指令 − R，Wt = 目标帧录制时的在线窗，都按帧优先展平，超出轨迹末尾的目标为 0；
    D (n, 5) = 冻结的在线 Shift − a_π，于是教师补正 Δ* = Y[:, :5] − D。"""
    X = assemble(log, seq, teacher, P, F)["X"]
    a_pi, act, H = np.asarray(log["a_pi"]), list(O.ACTIVE), RR.HOLD
    _, ens = O.te_states(log["chunk"])
    n = len(a_pi)
    R = np.stack([O.predicted(a_pi, ens[k], k, k, k + N - 1)[:, act] for k in range(n)]).astype(float)
    tgt = np.arange(n)[:, None] + np.arange(N)                    # 第 k 段第 j 列的目标帧 k + j
    keep = (tgt < n)[..., None]
    tgt = np.minimum(tgt, n - 1)
    Y = np.where(keep, teacher["cmd7"][H:, act].astype(float)[tgt] - R, 0.0)
    Wt = np.where(keep, np.asarray(log["w"], float)[tgt], 0.0)
    D = seq["s7"][H:, act].astype(float) - a_pi[:, act].astype(float)
    return {"X": X, "R": R, "Y": Y.reshape(n, -1), "Wt": Wt.reshape(n, -1), "D": D}


def _load_case(tid: str, rows: dict):
    """一条教师 → (录制记录, 回放序列, 冻结的教师)。"""
    r = json.loads(P.REC.read_text())[tid]
    log = dict(np.load(C.T.LR / r["repo"] / "high_rate" / f"episode_{r['episode']:06d}_b4.npz"))
    teacher = dict(np.load(B.TEACH / "sessions" / rows[tid]["session"] / f"teacher_{tid}.npz"))
    return log, B.load_seq(tid), teacher


def build(P: int, F: int, rows: dict | None = None, loader=None, out: Path = NET, N: int | None = None) -> Path:
    """训练、验证、备用位置里验收通过的教师拼成一份数据；备用位置是补训练集的，标成 train。
    rows = {id: {"split", "passed", …}}（默认读教师的 summary.json）。给 N 时是整段版（`assemble_seq`）：文件名带 _N{N}，多存 D、N。"""
    if rows is None:
        rows = json.loads((B.TEACH / "summary.json").read_text())
    loader = loader or (lambda tid: _load_case(tid, rows))
    ids = sorted(t for t, r in rows.items() if r["split"] in ("train", "val", "spare") and r["passed"] is True)
    parts = {t: assemble(*loader(t), P, F) if N is None else assemble_seq(*loader(t), P, F, N) for t in ids}
    X, Y, Wt = (np.concatenate([parts[t][k] for t in ids]) for k in ("X", "Y", "Wt"))
    traj = np.concatenate([[t] * len(parts[t]["Y"]) for t in ids])
    split = np.array([{"spare": "train"}.get(rows[t]["split"], rows[t]["split"]) for t in traj])
    sha = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (X, Y, Wt))).hexdigest()[:16]
    seq_part = {} if N is None else {"D": np.concatenate([parts[t]["D"] for t in ids]), "N": N}
    p = Path(out) / (f"data_P{P}_F{F}.npz" if N is None else f"data_P{P}_F{F}_N{N}.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, Y=Y, Wt=Wt, traj_id=traj, split=split, data_sha=sha, P=P, F=F, **seq_part)
    return p


# ── 任务 9：训练、选择、学习曲线、冻结 ─────────────────────────────────────
def weighted_mse(pred, y, w):
    """Σ w (pred − y)² / Σ w：窗 = 0 的帧不计；过渡段按窗加权。"""
    return (w * (pred - y) ** 2).sum() / w.sum()


def _r2(pred: np.ndarray, y: np.ndarray, w: np.ndarray) -> list[float]:
    """逐关节、按窗加权的 R²。"""
    out = []
    for j in range(y.shape[1]):
        wj = w[:, j]
        m = (wj * y[:, j]).sum() / wj.sum()
        out.append(float(1 - (wj * (y[:, j] - pred[:, j]) ** 2).sum() / (wj * (y[:, j] - m) ** 2).sum()))
    return out


def train(data: Path, seed: int, hidden: int, epochs: int = 300, lr: float = 1e-3, patience: int = 30,
          out: Path = NET / "runs", val_split: str = "val", ids: list[str] | None = None,
          rel_P: int | None = None, first: bool = False) -> dict:
    """Adam、批大小 4096，按验证集损失早停（留最好的那一轮）。ids = 只用训练集里的这些轨迹（学习曲线用）。
    网络输出维数 = 标签列数（一帧版 5，整段版 5·N）。rel_P：整段版 v0.3 的相对输入（见 `B4Net.prep`）。
    first（整段版 v0.3，N ≥ 2）：训练损失里第一帧（前 5 列，实发的那一帧）的权重 × (N − 1)，早停按验证集第一帧的损失。
    返回 {"path", "train_loss", "val_loss", "val_loss_first", "val_r2"}：val_loss、val_loss_first 是存下那一轮全部列、第一帧的验证损失；
    权重存 out/P{P}_F{F}_h{hidden}_s{seed}.pt。"""
    import torch
    z = np.load(data)
    P, F = int(z["P"]), int(z["F"])
    tr = z["split"] == "train"
    if ids is not None:
        tr &= np.isin(z["traj_id"], ids)
    va = z["split"] == val_split
    T = lambda m: [torch.from_numpy(np.asarray(z[k][m], np.float32)) for k in ("X", "Y", "Wt")]   # noqa: E731
    (Xt, Yt, Wt), (Xv, Yv, Wv) = T(tr), T(va)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = O.B4Net(Xt.shape[1], hidden, Yt.shape[1], rel_P)
    net.mu.copy_(net.prep(Xt).mean(0))
    net.sd.copy_(net.prep(Xt).std(0).clamp_min(1e-6))
    k = len(O.ACTIVE)
    if first:
        Wt[:, :k] *= Yt.shape[1] // k - 1                         # 第一帧的权重 = 其余 N − 1 帧之和
    opt = torch.optim.Adam(net.parameters(), lr)
    best, state, bad = float("inf"), None, 0
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), 4096):
            b = perm[i:i + 4096]
            opt.zero_grad()
            weighted_mse(net(Xt[b]), Yt[b], Wt[b]).backward()
            opt.step()
        with torch.no_grad():
            pv = net(Xv)
            vl = (weighted_mse(pv[:, :k], Yv[:, :k], Wv[:, :k]) if first else weighted_mse(pv, Yv, Wv)).item()
        if vl < best:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        tl = weighted_mse(net(Xt), Yt, Wt).item()
        pv = net(Xv)
        va_all = weighted_mse(pv, Yv, Wv).item()
        vf = weighted_mse(pv[:, :k], Yv[:, :k], Wv[:, :k]).item()
        r2 = _r2(pv.numpy(), Yv.numpy(), Wv.numpy())
    p = Path(out) / f"P{P}_F{F}_h{hidden}_s{seed}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "n_in": Xt.shape[1], "hidden": hidden, "n_out": Yt.shape[1], "P": P, "F": F,
                "rel_P": rel_P, "first": first, "seed": seed, "data_sha": str(z["data_sha"])}, p)
    return {"path": str(p), "train_loss": tl, "val_loss": va_all, "val_loss_first": vf, "val_r2": r2}


def select() -> dict:
    """候选 (P, F) × hidden × 3 个种子（数据按当前验收通过的教师重建）：先按 3 个种子的验证损失平均选组合，再取该组合里验证损失最低的种子。"""
    runs = []
    for P_, F_ in CANDIDATES:
        data = build(P_, F_)                          # 每次都重建：三份候选数据用同一批教师
        for h in HIDDEN:
            for s in SEEDS:
                r = train(data, s, h)
                runs.append({"P": P_, "F": F_, "hidden": h, "seed": s, **r})
                print(f"P{P_} F{F_} h{h} s{s}：验证损失 {r['val_loss']:.3e}，逐关节 R² {np.round(r['val_r2'], 2)}")
    key = lambda r: (r["P"], r["F"], r["hidden"])   # noqa: E731
    combos = {key(r): np.mean([x["val_loss"] for x in runs if key(x) == key(r)]) for r in runs}
    best = min(combos, key=combos.get)
    pick = min((r for r in runs if key(r) == best), key=lambda r: r["val_loss"])
    sel = {"chosen": pick, "combos": {f"P{k[0]}_F{k[1]}_h{k[2]}": v for k, v in combos.items()}, "runs": runs}
    (NET / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    return sel


def select_seq(N: int = 10, P: int = 8, F: int = 16, hidden: int = 128) -> dict:
    """整段版 ①（方案 `reports/PLAN_b4_seq.md` §4.2，v0.3）：P、F、hidden 沿用一帧版，数据每次重建；相对输入、第一帧加权；
    种子 0/1/2 各训一次，按验证集第一帧（实发那一帧）的损失选 → net_seq/selection.json。"""
    data = build(P, F, out=NET_SEQ, N=N)
    runs = []
    for s in SEEDS:
        r = train(data, s, hidden, out=NET_SEQ / "runs", rel_P=P, first=True)
        runs.append({"P": P, "F": F, "N": N, "hidden": hidden, "rel_P": P, "first": True, "seed": s, **r})
        print(f"种子 {s}：验证损失（{N} 帧）{r['val_loss']:.3e}，第一帧 {r['val_loss_first']:.3e}")
    sel = {"chosen": min(runs, key=lambda r: r["val_loss_first"]), "runs": runs}
    (NET_SEQ / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    return sel


def act_average(C: np.ndarray, coeff: float = 0.01) -> np.ndarray:
    """ACT 式多段平均（方案 `reports/PLAN_b4_seq.md` §4.4，只离线）：C (n, N, 5) 的第 k 段 = 第 k 帧生成的第 k … k+N−1 帧指令。
    第 k 帧 = 能用的 C[k−i, i]（i = 0 … min(N−1, k)）的加权平均，权重 exp(−coeff·idx)，idx = min(N−1, k) − i（最旧的一段 idx = 0，同 te_states）。"""
    out = np.empty((len(C), C.shape[2]))
    for k in range(len(C)):
        i = np.arange(min(C.shape[1] - 1, k) + 1)
        w = np.exp(-coeff * (i[-1] - i))
        out[k] = w @ C[k - i, i] / w.sum()
    return out


def gate(r2: np.ndarray, clip: np.ndarray, base: np.ndarray) -> tuple[bool, list[str]]:
    """上真机的门槛（用户 11-06，方案 §4.4）：每个关节 R² ≥ 一帧版 − 0.10，且限幅触发率 ≤ 1%。返回 (是否通过, 不过的项)。"""
    why = [f"{j} R² {a:.3f} < 门槛 {b - 0.10:.3f}" for j, a, b in zip(JOINTS, r2, base) if a < b - 0.10]
    why += [f"{j} 限幅触发率 {c:.2%} > 1%" for j, c in zip(JOINTS, clip) if c > 0.01]
    return not why, why


def check_seq() -> dict:
    """整段版 ① 上真机前的离线检查（方案 `reports/PLAN_b4_seq.md` §4.4）→ net_seq/check.json。只用验证集里验收通过的教师。
    R² 以教师补正 Δ* 为目标、按第一帧的窗加权（同一帧版）；限幅触发率在窗 > 0 的帧上；≥ 2 Hz 成分按实发（二值门、限幅后）算。
    另用一帧版 ① 在同一批数据上复算 R²：应等于它选模型时的值，用来核对口径。"""
    import torch
    from step4_gvnext import data as D
    sel = json.loads((NET_SEQ / "selection.json").read_text())["chosen"]
    m = torch.load(sel["path"], map_location="cpu")
    net = O.B4Net(m["n_in"], m["hidden"], m["n_out"], m.get("rel_P"))
    net.load_state_dict(m["state_dict"])
    net.eval()
    n_seg = m["n_out"] // len(O.ACTIVE)
    one, _, P1, F1 = O.load_frozen(str(NET / "frozen.json"))
    base = np.array(json.loads((NET / "selection.json").read_text())["chosen"]["val_r2"])
    rows = json.loads((B.TEACH / "summary.json").read_text())
    ids = sorted(t for t, r in rows.items() if r["split"] == "val" and r["passed"] is True)
    kinds = ("first", "avg", "one", "star")             # 整段版第一帧、离线平均、一帧版 ①、教师
    pool, hf = {k: [] for k in (*kinds, "w")}, {k: [] for k in kinds}
    frames = [[] for _ in range(n_seg)]                 # 第 j 帧：(预测的 Δ, 教师 Δ*, 权重)
    for tid in ids:
        log, seq, teacher = _load_case(tid, rows)
        d = assemble_seq(log, seq, teacher, m["P"], m["F"], n_seg)
        n = len(d["X"])
        with torch.no_grad():
            y = net(torch.from_numpy(d["X"])).numpy().astype(float).reshape(n, n_seg, -1)
        star = d["Y"][:, :5] - d["D"]
        Yr, Wr = d["Y"].reshape(n, n_seg, -1), d["Wt"].reshape(n, n_seg, -1)
        for j in range(n_seg):
            tj = star[np.minimum(np.arange(n) + j, n - 1)]
            frames[j].append((tj + y[:, j] - Yr[:, j], tj, Wr[:, j]))
        w = np.asarray(log["w"], float)
        cur = {"first": y[:, 0] - d["D"], "avg": act_average(d["R"] + y) - d["R"][:, 0] - d["D"],
               "one": np.stack([one(x) for x in assemble(log, seq, teacher, P1, F1)["X"]]), "star": star}
        for k, v in cur.items():
            pool[k].append(v)
            sent = v if k == "star" else np.where(w > 0, np.clip(v, -O.CLIP, O.CLIP), 0.0)
            hf[k].append(np.where(w > 0, sent - D.lp(sent, 2.0, D.FPS), np.nan))
        pool["w"].append(w)
    W = np.concatenate(pool["w"])
    on = W > 0
    res = {k: np.concatenate(pool[k]) for k in kinds}
    r2 = {k: np.array(_r2(res[k], res["star"], W)) for k in ("first", "avg", "one")}
    clip = {k: ((np.abs(res[k]) > O.CLIP) & on).sum(0) / on.sum(0) for k in ("first", "avg")}
    r2_frames = np.array([_r2(*(np.concatenate(x) for x in zip(*frames[j]))) for j in range(n_seg)])
    hf_mrad = {k: 1e3 * np.sqrt(np.nanmean(np.concatenate(hf[k]) ** 2, 0)) for k in kinds}
    ok, why = gate(r2["first"], clip["first"], base)
    out = {"chosen": sel["path"], "val_ids": ids, "baseline_r2": base.tolist(), "threshold_r2": (base - 0.10).tolist(),
           "r2_first": r2["first"].tolist(), "r2_avg": r2["avg"].tolist(), "r2_one_recomputed": r2["one"].tolist(),
           "r2_frames": r2_frames.tolist(), "clip_first": clip["first"].tolist(), "clip_avg": clip["avg"].tolist(),
           "hf_mrad": {k: v.tolist() for k, v in hf_mrad.items()}, "gate": {"passed": ok, "why": why}}
    (NET_SEQ / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    row = lambda name, v, f="{:8.3f}": print(f"{name:<14}" + "".join(f.format(x) for x in v))   # noqa: E731
    row("", JOINTS, "{:>8}")
    for name, v in (("R² 第一帧", r2["first"]), ("R² 门槛", base - 0.10), ("R² 离线平均", r2["avg"]),
                    ("R² 一帧版复算", r2["one"])):
        row(name, v)
    row("触发率 第一帧", 100 * clip["first"], "{:7.2f}%")
    row("触发率 离线平均", 100 * clip["avg"], "{:7.2f}%")
    for k, name in (("first", "≥2Hz 第一帧"), ("avg", "≥2Hz 离线平均"), ("one", "≥2Hz 一帧版"), ("star", "≥2Hz 教师")):
        row(name, hf_mrad[k])
    print(f"（≥2Hz 单位 mrad）第 1…{n_seg} 帧的 R²（五关节平均）：" + " ".join(f"{x:.2f}" for x in r2_frames.mean(1)))
    print(f"五关节平均 R²（只报告）：第一帧 {r2['first'].mean():.3f}，一帧版 {base.mean():.3f}")
    print("门槛：通过" if ok else f"门槛：不通过（{'；'.join(why)}）")
    return out


def curve() -> dict:
    """选中的组合下，训练集取 10 / 20 / 30 条（固定种子抽子集）时，验证集逐关节 R² 的平均。"""
    pick = json.loads((NET / "selection.json").read_text())["chosen"]
    data = NET / f"data_P{pick['P']}_F{pick['F']}.npz"
    z = np.load(data)
    ids = sorted(set(z["traj_id"][z["split"] == "train"]))
    order = list(np.random.default_rng(0).permutation(ids))
    pts = []
    for n in (10, 20, 30):
        if n > len(ids):
            break
        r = train(data, pick["seed"], pick["hidden"], out=NET / "curve", ids=order[:n])
        pts.append({"n": n, "val_r2_mean": float(np.mean(r["val_r2"])), "val_r2": r["val_r2"]})
        print(f"训练 {n} 条：验证集 R² 平均 {pts[-1]['val_r2_mean']:.3f}")
    (NET / "curve.json").write_text(json.dumps(pts, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Noto Sans CJK JP"           # 没有这行中文显示成方框
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([p["n"] for p in pts], [p["val_r2_mean"] for p in pts], "o-", color="#2a6fdb")
    ax.set_xlabel("训练轨迹条数")
    ax.set_ylabel("验证集 R²（五关节平均）")
    ax.set_title("补正网络的学习曲线", fontsize=10)
    fig.tight_layout()
    fig.savefig(NET / "curve.png", dpi=150)
    return {"points": pts}


def freeze(run: Path, out: Path = NET) -> Path:
    """冻结一次：权重复制到 out/frozen.pt，frozen.json 记权重哈希、P、F、hidden、在线配置哈希、数据哈希、代码提交号、时间。"""
    import torch
    f = Path(out) / "frozen.json"
    if f.exists():
        raise SystemExit(f"⛔ {f} 已存在：网络只冻结一次")
    m = torch.load(run, map_location="cpu")
    w = Path(out) / "frozen.pt"
    shutil.copyfile(run, w)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    f.write_text(json.dumps({"weights": str(w.resolve()), "weights_sha": hashlib.sha256(w.read_bytes()).hexdigest()[:16],
                             "P": m["P"], "F": m["F"], "hidden": m["hidden"], "n_in": m["n_in"], "seed": m["seed"],
                             "data_sha": m["data_sha"], "cfg_sha": O.cfg_sha(C.CFG) if C.CFG.exists() else "",
                             "git_head": head, "source_run": str(run), "time": dt.datetime.now().isoformat(timespec="seconds")},
                            ensure_ascii=False, indent=1))
    return f


def main(argv=None) -> int:
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser(description="B4 的补正网络")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="拼训练数据")
    b.add_argument("--P", type=int, default=8)
    b.add_argument("--F", type=int, default=8)
    b.add_argument("--N", type=int, default=None, help="整段版 ①：每帧输出几帧（写到 outputs/b4/net_seq/）")
    sub.add_parser("select", help="选网络（18 次训练）")
    sq = sub.add_parser("seq", help="整段版 ①：3 个种子训练、按第一帧选")
    sq.add_argument("--N", type=int, default=10, help="段长（帧）")
    sub.add_parser("check", help="整段版 ①：离线检查与上真机门槛")
    sub.add_parser("curve", help="学习曲线")
    fz = sub.add_parser("freeze", help="冻结选中的网络（只一次）")
    fz.add_argument("--seq", action="store_true", help="冻结整段版 ①（outputs/b4/net_seq/）")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        print(f"选中：{select()['chosen']['path']}")
        return 0
    if a.cmd == "seq":
        print(f"选中：{select_seq(N=a.N)['chosen']['path']}")
        return 0
    if a.cmd == "check":
        check_seq()
        return 0
    if a.cmd == "curve":
        curve()
        return 0
    if a.cmd == "freeze":
        d = NET_SEQ if a.seq else NET
        print(f"→ {freeze(Path(json.loads((d / 'selection.json').read_text())['chosen']['path']), out=d)}")
        return 0
    p = build(a.P, a.F) if a.N is None else build(a.P, a.F, out=NET_SEQ, N=a.N)
    z = np.load(p)
    for s in ("train", "val"):
        m = z["split"] == s
        print(f"{s}：{len(set(z['traj_id'][m]))} 条、{int(m.sum())} 帧")
    print(f"→ {p}，数据哈希 {z['data_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
