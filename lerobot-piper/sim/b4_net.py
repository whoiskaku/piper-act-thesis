"""B4 的补正网络：数据（方案 `reports/PLAN_b4.md` §5.3；实现计划任务 8，训练在任务 9）。

  python -u sim/b4_net.py build --P 8 --F 8     # 训练、验证里验收通过的教师 → outputs/b4/net/data_P8_F8.npz
  python -u sim/b4_net.py select                # 候选 (P, F) × hidden × 3 种子，按验证集损失选 → selection.json
  python -u sim/b4_net.py curve                 # 选中的组合下，训练 10 / 20 / 30 条时验证集的 R² → curve.json、curve.png
  python -u sim/b4_net.py freeze                # 冻结选中的那一个（只能冻结一次）→ frozen.json、frozen.pt

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
CANDIDATES = [(4, 4), (8, 8), (8, 16)]
HIDDEN, SEEDS = (64, 128), (0, 1, 2)


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


def _load_case(tid: str, rows: dict):
    """一条教师 → (录制记录, 回放序列, 冻结的教师)。"""
    r = json.loads(P.REC.read_text())[tid]
    log = dict(np.load(C.T.LR / r["repo"] / "high_rate" / f"episode_{r['episode']:06d}_b4.npz"))
    teacher = dict(np.load(B.TEACH / "sessions" / rows[tid]["session"] / f"teacher_{tid}.npz"))
    return log, B.load_seq(tid), teacher


def build(P: int, F: int, rows: dict | None = None, loader=None, out: Path = NET) -> Path:
    """训练、验证、备用位置里验收通过的教师拼成一份数据；备用位置是补训练集的，标成 train。
    rows = {id: {"split", "passed", …}}（默认读教师的 summary.json）。"""
    if rows is None:
        rows = json.loads((B.TEACH / "summary.json").read_text())
    loader = loader or (lambda tid: _load_case(tid, rows))
    ids = sorted(t for t, r in rows.items() if r["split"] in ("train", "val", "spare") and r["passed"] is True)
    parts = {t: assemble(*loader(t), P, F) for t in ids}
    X, Y, Wt = (np.concatenate([parts[t][k] for t in ids]) for k in ("X", "Y", "Wt"))
    traj = np.concatenate([[t] * len(parts[t]["Y"]) for t in ids])
    split = np.array([{"spare": "train"}.get(rows[t]["split"], rows[t]["split"]) for t in traj])
    sha = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (X, Y, Wt))).hexdigest()[:16]
    p = Path(out) / f"data_P{P}_F{F}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=X, Y=Y, Wt=Wt, traj_id=traj, split=split, data_sha=sha, P=P, F=F)
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
          out: Path = NET / "runs", val_split: str = "val", ids: list[str] | None = None) -> dict:
    """Adam、批大小 4096，按验证集损失早停（留最好的那一轮）。ids = 只用训练集里的这些轨迹（学习曲线用）。
    返回 {"path", "train_loss", "val_loss", "val_r2"}；权重存 out/P{P}_F{F}_h{hidden}_s{seed}.pt。"""
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
    net = O.B4Net(Xt.shape[1], hidden)
    net.mu.copy_(Xt.mean(0))
    net.sd.copy_(Xt.std(0).clamp_min(1e-6))
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
            vl = weighted_mse(net(Xv), Yv, Wv).item()
        if vl < best:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        tl = weighted_mse(net(Xt), Yt, Wt).item()
        r2 = _r2(net(Xv).numpy(), Yv.numpy(), Wv.numpy())
    p = Path(out) / f"P{P}_F{F}_h{hidden}_s{seed}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "n_in": Xt.shape[1], "hidden": hidden, "P": P, "F": F,
                "seed": seed, "data_sha": str(z["data_sha"])}, p)
    return {"path": str(p), "train_loss": tl, "val_loss": best, "val_r2": r2}


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
    sub.add_parser("select", help="选网络（18 次训练）")
    sub.add_parser("curve", help="学习曲线")
    sub.add_parser("freeze", help="冻结选中的网络（只一次）")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        print(f"选中：{select()['chosen']['path']}")
        return 0
    if a.cmd == "curve":
        curve()
        return 0
    if a.cmd == "freeze":
        print(f"→ {freeze(Path(json.loads((NET / 'selection.json').read_text())['chosen']['path']))}")
        return 0
    p = build(a.P, a.F)
    z = np.load(p)
    for s in ("train", "val"):
        m = z["split"] == s
        print(f"{s}：{len(set(z['traj_id'][m]))} 条、{int(m.sum())} 帧")
    print(f"→ {p}，数据哈希 {z['data_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
