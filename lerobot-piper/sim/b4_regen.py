"""重新生成网络 ② 的离线部分（方案 `reports/PLAN_b4_regen.md` v0.2；实现计划 `reports/PLAN_IMPL_b4_regen.md`）。
用户 11-12 改：标签 = teacher24（ILC 教师整体提前 24 ms）、基线 = shift84、网络只看 ACT 指令；输出在 outputs/b4/regen_t24
（旧方案的输出留在 outputs/b4/regen）。下面命令里的路径都相应换成 regen_t24。

  python -u sim/b4_regen.py build      # 验收通过的教师轨迹、在教师会话里的全部有效回放 → outputs/b4/regen/data.npz
  python -u sim/b4_regen.py train      # 种子 0、1、2 各训一次，按验证集第一帧损失选种子 → outputs/b4/regen/selection.json
  python -u sim/b4_regen.py ablate     # 对照（种子 0）：去掉实测、只去掉 IMU、去掉已发出的指令 → outputs/b4/regen/ablation.json
  python -u sim/b4_regen.py bounds     # 教师会话里发过的指令范围 → 逐帧检查的位置界 → outputs/b4/regen/bounds.json
  python -u sim/b4_regen.py check      # 上机前的离线检查（门槛 1、2 与只报告的各项）→ outputs/b4/regen/check.json
  python -u sim/b4_regen.py freeze     # 冻结选中的那一个（只能冻结一次）→ outputs/b4/regen/frozen.json、frozen.pt

样本 = 一遍回放的一帧 k：输入单元是当时已有的 ACT 指令、已发出的指令、编码器与 IMU 窗口（`b4_online` 的 regen_* 函数，
在线回放用的是同一批函数），训练时后三组在网络里置 0（DROP）；标签是这条轨迹的 teacher24（第 k … k+9 帧，6 个关节）；权重按在线窗。
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
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
import ideal_motion as IM  # noqa: E402
import repeat_reversal as RR  # noqa: E402

import lerobot.common.policies.act.b4_online as O  # noqa: E402

REGEN = C.OUT / "regen_t24"                               # ② 的一切输出（11-12 起标签 teacher24；旧方案在 C.OUT / "regen"）
BAD_SESSIONS = ("20261104b", "20261104c", "20261104d")    # b、d：IMU 固定松动；c：朝向偏差 7.08°
SHIFT_RIDS = ("b01", "b02", "as01", "as02", "as03")       # 教师会话里的 Shift 回放（离线门槛 1 用）
ARCH = {"n_past": 102, "n_fut": 6, "d": 128, "layers": 4, "heads": 4, "ff": 512, "dropout": 0.0}
MRAD = 1e-3                                               # 损失的单位：误差按 mrad 计、各关节等权（同主指标；用户 11-11）
DROP = ("sent", "enc", "imu")                             # 只看 ACT 指令（用户 11-12）：带实测输入的版本闭环会漂移
ABLATIONS = ()                                            # 主模型只看 ACT 指令后，对照另定


def cases(rows: dict | None = None) -> list[tuple[str, str, str]]:
    """(轨迹, 划分, 会话名)：验收通过的教师；train、spare 算训练，val 算验证，test 不用；BAD_SESSIONS 里的不用。被排除的打印原因。"""
    rows = rows if rows is not None else json.loads((B.TEACH / "summary.json").read_text())
    out = []
    for tid in sorted(rows):
        r = rows[tid]
        why = ("验收没通过" if r["passed"] is not True else "测试集不用" if r["split"] == "test"
               else f"会话 {r['session']} 不用" if r["session"] in BAD_SESSIONS else None)
        if why:
            print(f"  {tid}：跳过（{why}）")
            continue
        out.append((tid, "val" if r["split"] == "val" else "train", r["session"]))
    return out


def valid_replays(sess: Path, tid: str, n: int) -> list[tuple[str, dict]]:
    """会话里这条轨迹的有效回放 (rid, 记录)，按 rid 排序。有效 = 没有中途停、帧数为 n、有编码器和 IMU、
    回放序列的哈希和记录一致、到达时刻已按顺序（窗口函数假定如此，这里是唯一的检查）。无效的打印原因并跳过。"""
    out = []
    for f in sorted(Path(sess).glob(f"replay_{tid}_*_r1.json")):
        rid = f.name[len("replay_"):-len("_r1.json")]
        h = json.loads(f.read_text())
        seq = Path(sess) / f"seq_{rid}.npz"
        why = ("中途停了" if h.get("stopped") is not None
               else "帧数不符" if len(h.get("t_send", [])) != n
               else "没有编码器" if not h.get("can_q200")
               else "没有 IMU" if not h.get("imu100")
               else "没有回放序列" if not seq.exists()
               else "回放序列的哈希不符" if json.loads(str(np.load(seq)["source"]))["cmd_hash"] != h.get("cmd_hash")
               else "到达时刻没有按顺序" if (np.diff(np.asarray(h["can_q200"], float)[:, 0]).min(initial=0) < 0
                                       or np.diff(np.asarray(h["imu100"], float)[:, 0]).min(initial=0) < 0)
               else None)
        if why:
            print(f"  {rid}：跳过（{why}）")
            continue
        out.append((rid, h))
    return out


def _token_fn(h: dict, base: dict, log: dict, calib: dict):
    """一遍回放 → tokens(k, sent6) = 第 k 帧的输入单元 (past, fut)。实测只用截止时刻 t_send[k] − CUT_S 以前到达的；
    已发出的指令取 sent6 的第 k−8 … k−1 行（帧号小于 0 取第 0 行）。训练数据、first_frames、preview 都经过这里。"""
    a7 = np.asarray(base["a7"], float)
    a_pi, (_, ens) = np.asarray(log["a_pi"], float), O.te_states(log["chunk"])
    ts = np.asarray(h["t_send"], float)
    off = h["can_hs_meta"]["clock"]["offset_realtime_minus_monotonic_s"]
    q200, imu = np.asarray(h["can_q200"], float), np.asarray(h["imu100"], float)
    tq = q200[:, 0] - off

    def tokens(k: int, sent6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cut = ts[k] - O.CUT_S
        qs = slice(*np.searchsorted(tq, [cut - 2 * O.WIN_S, cut]))        # 只为提速；窗口函数自己再按截止时刻截
        js = slice(*np.searchsorted(imu[:, 0], [cut - 2 * O.WIN_S, cut]))
        enc, _ = O.enc_window(q200[qs], off, cut)
        imw, _ = O.imu_window(imu[js], calib, cut)
        act_past, act_seg = O.regen_act(a7, a_pi, ens, k, RR.HOLD)
        return O.regen_tokens(act_past, sent6[np.clip(np.arange(k - O.REGEN_P, k), 0, None)], enc, imw, act_seg)

    return tokens


def assemble_regen(h: dict, base: dict, log: dict, teacher7: np.ndarray, calib: dict) -> dict:
    """一遍回放 → 每一帧的样本，只留输出窗口（第 k … k+9 帧）里至少有一个权重 > 0 的帧。截止时刻 = t_send[k] − CUT_S。
    权重：会动的五个关节各自取 w > 0，j4 取五个关节的窗都 > 0，超出末尾为 0。标签超出末尾时取最后一帧（权重为 0，不学）。"""
    w = np.asarray(base["w"], float)
    n, N = len(base["a7"]), O.REGEN_N
    tokens, sent = _token_fn(h, base, log, calib), np.asarray(h["cmd"], float)[:, :6]
    wf = np.zeros((n + N, 6), np.float32)
    wf[:n, list(O.ACTIVE)] = w > 0
    wf[:n, 3] = w.min(1) > 0
    t7 = np.asarray(teacher7, np.float32)[:, :6]
    lab = np.concatenate([t7, np.repeat(t7[-1:], N, 0)])
    out = {key: [] for key in ("past", "fut", "Y", "W", "k")}
    for k in range(n):
        if not wf[k:k + N].any():
            continue
        past, fut = tokens(k, sent)
        for key, v in (("past", past), ("fut", fut), ("Y", lab[k:k + N]), ("W", wf[k:k + N]), ("k", k)):
            out[key].append(v)
    return {key: np.asarray(v) for key, v in out.items()}


def first_frames(fn, h: dict, base: dict, log: dict, calib: dict) -> np.ndarray:
    """这一遍的状态上（已发出的指令取记录里的）网络每帧的第一帧输出 (n, 6)；按帧号顺序每帧调用一次 fn。"""
    tokens, sent = _token_fn(h, base, log, calib), np.asarray(h["cmd"], float)[:, :6]
    return np.stack([fn(*tokens(k, sent))[0] for k in range(len(sent))])


def preview(fn, h: dict, base: dict, log: dict, calib: dict, lo: np.ndarray, hi: np.ndarray) -> dict:
    """按使用方式逐帧跑（设计 §4.5）：已发出的指令用自己的 regen_command 结果，实测取录好的这一遍（不随自己的指令变）。
    违反逐帧检查时记下 (k, 原因)，不停，照样当作已发出。第 0 帧之前机械臂停在 Shift 第一帧的位置。
    返回 {"u7": (n, 7) float32, "first": (n, 6), "stops": [(k, 原因)]}；按帧号顺序每帧调用一次 fn。"""
    a7, s7, w = (np.asarray(base[x], float) for x in ("a7", "s7", "w"))
    tokens, sent = _token_fn(h, base, log, calib), s7[:, :6].copy()
    n = len(a7)
    u7, first, stops = np.zeros((n, 7), np.float32), np.zeros((n, 6)), []
    for k in range(n):
        first[k] = fn(*tokens(k, sent))[0]
        u7[k] = O.regen_command(first[k], s7[k], w[k])
        why = O.step_check(u7[k, :6], sent[k - 1] if k else s7[0, :6], a7[k, :6], lo, hi)
        if why:
            stops.append((k, why))
        sent[k] = u7[k, :6]
    return {"u7": u7, "first": first, "stops": stops}


def _calib(sess: Path) -> dict | None:
    """会话 session.json 记的标定哈希 → imu_calib*.json 里对应的那一份；找不到返回 None。"""
    want = json.loads((Path(sess) / "session.json").read_text())["calib_sha"]
    return next((json.loads(p.read_text()) for p in sorted(C.CALIB.parent.glob("imu_calib*.json")) if B._sha(p) == want), None)


def _log(tid: str) -> dict:
    """一条轨迹的录制记录（每帧 ACT 的整段预测等）。"""
    r = json.loads(P.REC.read_text())[tid]
    return dict(np.load(C.T.LR / r["repo"] / "high_rate" / f"episode_{r['episode']:06d}_b4.npz"))


def _load(tid: str, session: str):
    """从磁盘读一条教师轨迹 → (base, log, teacher7, calib, 有效回放)。会话的标定找不到时打印原因、返回 None。"""
    sess = B.TEACH / "sessions" / session
    calib = _calib(sess)
    if calib is None:
        print(f"  {tid}：跳过（会话 {session} 用的标定找不到）")
        return None
    base = B.load_seq(tid)
    t24 = IM.earlier(np.load(sess / f"teacher_{tid}.npz")["cmd7"], IM.EARLY_S)      # 标签 teacher24（用户 11-12）
    return base, _log(tid), t24, calib, valid_replays(sess, tid, len(base["a7"]))


def shift_cmd(tid: str, lead: float) -> np.ndarray:
    """这条轨迹按 lead 秒提前量的 Shift 指令 (n, 7)：同录制时的算法（O.online_shift，开头接保持段），夹爪取录制时 Shift 的。
    先用 60 ms 重建，和记录的 Shift 对不上就停（保证换提前量时算法没变）。"""
    s7, (_, ens) = np.asarray(B.load_seq(tid)["s7"], float), O.te_states(_log(tid)["chunk"])

    def build(ld: float) -> np.ndarray:
        u = np.stack([O.online_shift(e, ld) for e in ens])
        c = np.concatenate([np.repeat(u[:1], RR.HOLD, 0), u])
        c[:, 6] = s7[:, 6]
        return c

    gap = np.abs(build(O.LEAD_S)[:, :6] - s7[:, :6]).max()
    if gap > 1e-6:
        raise SystemExit(f"⛔ {tid}：用 online_shift 重建的 60 ms Shift 与记录的差 {gap:.2e} rad，算法对不上")
    return build(lead)


def build_regen(rows: dict | None = None, out: Path = REGEN, loader=None) -> Path:
    """全部样本 → out/data.npz：past、fut、Y、W、k、traj_id、rid、split、data_sha（past 与 Y 字节的 sha256 前 16 位）。
    loader(tid) → (base, log, teacher7, calib, [(rid, 记录), …]) 或 None；默认从磁盘读。"""
    rows = rows if rows is not None else json.loads((B.TEACH / "summary.json").read_text())
    loader = loader or (lambda tid: _load(tid, rows[tid]["session"]))
    parts, stat = [], {"train": [0, 0, 0], "val": [0, 0, 0]}          # 轨迹、遍、样本
    for tid, split, _ in cases(rows):
        got = loader(tid)
        if got is None:
            continue
        base, log, teacher7, calib, reps = got
        for rid, h in reps:
            d = assemble_regen(h, base, log, teacher7, calib)
            m = len(d["k"])
            parts.append({**d, "traj_id": np.full(m, tid), "rid": np.full(m, rid), "split": np.full(m, split)})
            stat[split][1:] = stat[split][1] + 1, stat[split][2] + m
        stat[split][0] += 1
        print(f"  {tid}（{split}）：{len(reps)} 遍", flush=True)
    z = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    z["data_sha"] = hashlib.sha256(z["past"].tobytes() + z["Y"].tobytes()).hexdigest()[:16]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "data.npz", **z)
    for split, (nt, nr, ns) in stat.items():
        print(f"{split}：{nt} 条轨迹、{nr} 遍、{ns} 个样本")
    print(f"→ {out / 'data.npz'}（数据哈希 {z['data_sha']}）")
    return out / "data.npz"


def frame_weights(W):
    """训练损失的权重：第一帧（实发的那一帧）乘 REGEN_N − 1，等于其余各帧之和（同整段版 ① v0.3）。不改输入。"""
    W = W.clone()
    W[:, 0] *= O.REGEN_N - 1
    return W


def weighted_loss(net, P_, F_, Y_, W_) -> float:
    """加权均方误差，单位 mrad²，各关节等权（不求梯度、分批）。只拿标签覆盖到的那几帧预测去比：Y_ 只给第一帧时，就只算网络输出的第一帧。"""
    import torch
    net.eval()
    num = den = 0.0
    with torch.no_grad():
        for i in range(0, len(P_), 8192):
            s = slice(i, i + 8192)
            pred = net(P_[s], F_[s])[:, :Y_.shape[1]]
            num += (W_[s] * ((pred - Y_[s]) / MRAD) ** 2).sum().item()
            den += W_[s].sum().item()
    return num / den


def train_regen(data: Path, seed: int, drop: tuple[str, ...] = (), epochs: int = 100, lr: float = 3e-4, batch: int = 1024,
                patience: int = 100, out: Path = REGEN / "runs", device: str = "cuda") -> dict:
    """AdamW（weight_decay 0.01），学习率在 epochs 轮里按余弦从 lr 降到 0；损失 = 加权均方误差（mrad²，各关节等权），第一帧权重 × (N − 1)；
    留验证集第一帧加权损失最小的那一轮；连续 patience 轮没变好就停，默认等于轮数，即跑满不早停（余弦衰减前段验证损失波动大）。
    网络内部的标准化（均值、标准差）只在训练集上、按 prep 之后的相对量算（sd 下限：输入 1e-6，输出 1e-3；j4 的标签恒为 0，sd 就是 1e-3）；
    它不影响损失的单位。
    返回 {"path", "val_first", "val_all", "train_loss", "epochs", "history_val_first"}；权重存 out/s{seed}.pt 或 s{seed}_no-{组}.pt。"""
    import torch
    z = np.load(data)
    tr, va = z["split"] == "train", z["split"] == "val"
    T = lambda m: [torch.from_numpy(np.ascontiguousarray(z[k][m])).to(device) for k in ("past", "fut", "Y", "W")]   # noqa: E731
    (Pt, Ft, Yt, Wt), (Pv, Fv, Yv, Wv) = T(tr), T(va)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = O.RegenNet(**ARCH, drop=drop).to(device)
    with torch.no_grad():
        pp, ff = net.prep(Pt, Ft)
        for name, x, floor in (("p", pp, 1e-6), ("f", ff, 1e-6), ("y", Yt, 1e-3)):
            flat = x.reshape(-1, x.shape[-1])
            getattr(net, f"mu_{name}").copy_(flat.mean(0))
            getattr(net, f"sd_{name}").copy_(flat.std(0).clamp_min(floor))
    Wf = frame_weights(Wt)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Pt), generator=g).to(device)
        for i in range(0, len(Pt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            ((Wf[b] * ((net(Pt[b], Ft[b]) - Yt[b]) / MRAD) ** 2).sum() / Wf[b].sum()).backward()
            opt.step()
        sched.step()
        hist.append(weighted_loss(net, Pv, Fv, Yv[:, :1], Wv[:, :1]) if len(Pv) else 0.0)
        if hist[-1] < best:
            best, state, bad = hist[-1], copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if ep % 10 == 9:
            print(f"    第 {ep + 1} 轮：验证第一帧损失 {hist[-1]:.3f} mrad²（最好 {best:.3f}），学习率 {sched.get_last_lr()[0]:.1e}", flush=True)
    net.load_state_dict(state)
    name = f"s{seed}" + (f"_no-{'-'.join(drop)}" if drop else "")
    p = Path(out) / f"{name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "drop": list(drop), "seed": seed,
                "data_sha": str(z["data_sha"]), "arch": ARCH}, p)
    return {"path": str(p), "val_first": best, "val_all": weighted_loss(net, Pv, Fv, Yv, Wv) if len(Pv) else 0.0,
            "train_loss": weighted_loss(net, Pt, Ft, Yt, Wf), "epochs": len(hist), "history_val_first": hist}


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_regen() -> dict:
    """种子 0、1、2 各训一次（输入去掉 DROP，只看 ACT 指令），按验证集第一帧损失选种子 → REGEN/selection.json。"""
    runs = []
    for s in (0, 1, 2):
        print(f"种子 {s}：", flush=True)
        r = train_regen(REGEN / "data.npz", s, drop=DROP, device=_device())
        runs.append({"seed": s, **r})
        print(f"  种子 {s}：{r['epochs']} 轮，验证第一帧损失 {r['val_first']:.3f} mrad²，全部帧 {r['val_all']:.3f}，训练 {r['train_loss']:.3f}", flush=True)
    sel = {"chosen": min(runs, key=lambda r: r["val_first"]), "runs": runs}
    (REGEN / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print(f"选中种子 {sel['chosen']['seed']}：{sel['chosen']['path']}")
    return sel


def ablate_regen() -> dict:
    """对照（种子 0）：分别去掉 (enc, imu)、(imu)、(sent) → REGEN/ablation.json（离线检查只报告，不进门槛）。"""
    runs = []
    for drop in ABLATIONS:
        print(f"去掉 {'、'.join(drop)}：", flush=True)
        r = train_regen(REGEN / "data.npz", 0, drop=drop, device=_device())
        runs.append({"drop": list(drop), **r})
        print(f"  去掉 {'、'.join(drop)}：{r['epochs']} 轮，验证第一帧损失 {r['val_first']:.3f} mrad²", flush=True)
    (REGEN / "ablation.json").write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
    return {"runs": runs}


def sent_range(root: Path = B.TEACH / "sessions") -> dict:
    """教师会话里全部回放序列（seq_*.npz）发过的 6 个关节指令的范围，和由它定的逐帧检查位置界（O.regen_bounds）→ REGEN/bounds.json。"""
    files = sorted(Path(root).glob("*/seq_*.npz"))
    u = np.concatenate([np.load(f)["cmd7"][:, :6] for f in files]).astype(float)
    lo, hi = O.regen_bounds(O.load_cfg(C.CFG), u.min(0), u.max(0))
    d = {"sent_lo": u.min(0).tolist(), "sent_hi": u.max(0).tolist(), "n_files": len(files), "lo": lo.tolist(), "hi": hi.tolist()}
    REGEN.mkdir(parents=True, exist_ok=True)
    (REGEN / "bounds.json").write_text(json.dumps(d, indent=1))
    for j in range(6):
        print(f"j{j + 1}：发过 {d['sent_lo'][j]:+.3f} … {d['sent_hi'][j]:+.3f}，界 {lo[j]:+.3f} … {hi[j]:+.3f} rad")
    print(f"{len(files)} 个回放序列 → {REGEN / 'bounds.json'}")
    return d


def gate1(r2: np.ndarray) -> list[str]:
    """门槛 1 不过的项（空 = 通过）：五个会动关节的 R²（网络第一帧 − shift84 对 teacher24 − shift84）都要 > 0；NaN 算不过（用户 11-12）。"""
    return [f"j{i + 1} R² {a:.3f} ≤ 0" for i, a in zip(O.ACTIVE, r2) if not a > 0]


def check_regen() -> dict:
    """上机前的离线检查 → REGEN/check.json；网络 = selection.json 选中的那一个（用户 11-12：标签 teacher24、基线 shift84、只看 ACT 指令）。
    门槛 1：验证轨迹上，网络第一帧 − shift84 对 teacher24 − shift84 的 R²（逐关节、按 w 加权），五个会动关节都 > 0，即比 shift84 更接近 teacher24。
    只看 ACT 指令时输出与回放状态无关：每条轨迹取一遍 Shift 回放的输入单元算一次。
    门槛 2：会话 20261106q 的 10 遍 qshift1 回放按使用方式预演（窗为 0 的关节和夹爪发 shift84），逐帧检查一次都不触发。
    只报告：网络、shift84、shift60 离 teacher24 的 RMS（mrad，按 w 加权）；j4 输出的标准差；网络（预演）、shift84、teacher24 在窗 > 0 帧上
    ≥ 2 Hz 成分的 RMS。"""
    import torch
    import b4_net
    from step4_gvnext import data as D
    torch.set_num_threads(1)                                  # 逐帧单样本推理，多线程只添开销
    act, J = list(O.ACTIVE), b4_net.JOINTS
    sel = json.loads((REGEN / "selection.json").read_text())["chosen"]
    fn = O.regen_fn(torch.load(sel["path"], map_location="cpu"))
    pool, dist, j4, teach = [], [], [], {}
    for tid, split, session in cases():
        got = _load(tid, session) if split == "val" else None
        if got is None:
            continue
        base, log, t24, calib, reps = got
        h = next((h for rid, h in reps if rid.split("_", 1)[1] in SHIFT_RIDS), None)
        if h is None:
            print(f"  {tid}：没有有效的 Shift 回放，跳过")
            continue
        w, t = np.asarray(base["w"], float), np.asarray(t24, float)[:, act]
        s84, s60 = shift_cmd(tid, IM.LEAD_S)[:, act], np.asarray(base["s7"], float)[:, act]
        first = first_frames(fn, h, base, log, calib)
        on = lambda x: np.where(w > 0, x, 0.0)    # noqa: E731  权重为 0 的帧不计：录制晚开始时第 0 帧没有实测，输出是 NaN
        pool.append((on(first[:, act] - s84), t - s84, w))
        dist.append([on(x - t) for x in (first[:, act], s84, s60)] + [w])
        j4.append(first[w.min(1) > 0, 3])
        teach[tid] = t24
        print(f"  {tid}", flush=True)
    r2 = np.array(b4_net._r2(*[np.concatenate(x) for x in zip(*pool)]))
    W = np.concatenate([d[3] for d in dist])
    rms = {k: 1e3 * np.sqrt((W * np.concatenate([d[i] for d in dist]) ** 2).sum(0) / W.sum(0))
           for i, k in enumerate(("net", "shift84", "shift60"))}
    sess = B.TEACH / "sessions" / "20261106q"
    calib = _calib(sess)
    if calib is None:
        raise SystemExit("⛔ 会话 20261106q 用的标定找不到")
    bd = json.loads((REGEN / "bounds.json").read_text())
    lo, hi = np.asarray(bd["lo"]), np.asarray(bd["hi"])
    stops, first_stops, hf = {}, {}, {k: [] for k in ("net", "shift84", "teacher24")}
    for f in sorted(sess.glob("replay_*_qshift1_r1.json")):
        tid = f.name.split("_")[1]
        base = {**B.load_seq(tid), "s7": shift_cmd(tid, IM.LEAD_S)}
        reps = [h for rid, h in valid_replays(sess, tid, len(base["a7"])) if rid == f"{tid}_qshift1"]
        if not reps:
            stops[tid] = None
            continue
        pv = preview(fn, reps[0], base, _log(tid), calib, lo, hi)
        stops[tid], first_stops[tid] = len(pv["stops"]), pv["stops"][:3]
        if tid in teach:
            w = np.asarray(base["w"], float)
            for k, x in (("net", pv["u7"]), ("shift84", base["s7"]), ("teacher24", teach[tid])):
                x = np.asarray(x, float)[:, act]
                hf[k].append(np.where(w > 0, x - D.lp(x, 2.0, D.FPS), np.nan))
        print(f"  预演 {tid}：触发 {stops[tid]} 次", flush=True)
    why1, ok2 = gate1(r2), all(v == 0 for v in stops.values())
    hf_mrad = {k: 1e3 * np.sqrt(np.nanmean(np.concatenate(v) ** 2, 0)) for k, v in hf.items()}
    out = {"chosen": sel["path"], "val_ids": sorted(teach),
           "gate1": {"r2": r2.tolist(), "passed": not why1, "why": why1},
           "gate2": {"stops": stops, "first_stops": first_stops, "passed": ok2},
           "reports": {"rms_to_teacher24_mrad": {k: v.tolist() for k, v in rms.items()},
                       "j4_sd_mrad": float(1e3 * np.concatenate(j4).std()), "hf_mrad": {k: v.tolist() for k, v in hf_mrad.items()}},
           "passed": not why1 and ok2}
    (REGEN / "check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    row = lambda name, v, f="{:8.3f}": print(f"{name:<22}" + "".join(f.format(x) for x in v))   # noqa: E731
    row("", J, "{:>8}")
    row("R²（相对 shift84）", r2)
    for k in ("net", "shift84", "shift60"):
        row(f"离 teacher24：{k}", rms[k], "{:8.2f}")
    for k in ("net", "shift84", "teacher24"):
        row(f"≥2Hz {k}", hf_mrad[k], "{:8.2f}")
    print(f"（离 teacher24、≥2Hz 单位 mrad）j4 输出的标准差 {out['reports']['j4_sd_mrad']:.3f} mrad；门槛 2 预演触发次数：{stops}")
    print("门槛 1：通过" if not why1 else f"门槛 1：不通过（{'；'.join(why1)}）")
    print("门槛 2：通过" if ok2 else "门槛 2：不通过（预演触发了逐帧检查，见 check.json 的 first_stops）")
    return out


def freeze_regen(run: Path, out: Path = REGEN) -> Path:
    """冻结一次：权重复制到 out/frozen.pt，frozen.json 记权重路径与哈希、数据哈希、种子、去掉的组、结构、代码提交号、来源、时间。"""
    import torch
    f = Path(out) / "frozen.json"
    if f.exists():
        raise SystemExit(f"⛔ {f} 已存在：网络只冻结一次")
    m = torch.load(run, map_location="cpu")
    w = Path(out) / "frozen.pt"
    shutil.copyfile(run, w)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    sha = hashlib.sha256(w.read_bytes()).hexdigest()[:16]
    f.write_text(json.dumps({"weights": str(w.resolve()), "weights_sha": sha, "data_sha": m["data_sha"], "seed": m["seed"],
                             "drop": m["drop"], "arch": m["arch"], "git_head": head, "source_run": str(run),
                             "time": dt.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
    print(f"→ {f}（权重哈希 {sha}，来自 {run}）")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="教师轨迹的全部有效回放 → outputs/b4/regen/data.npz")
    sub.add_parser("train", help="种子 0、1、2 各训一次并选种子 → outputs/b4/regen/selection.json")
    sub.add_parser("ablate", help="三个去掉输入组的对照 → outputs/b4/regen/ablation.json")
    sub.add_parser("bounds", help="发过的指令范围与逐帧检查的位置界 → outputs/b4/regen/bounds.json")
    sub.add_parser("check", help="上机前的离线检查 → outputs/b4/regen/check.json")
    sub.add_parser("freeze", help="冻结 selection.json 选中的那一个 → outputs/b4/regen/frozen.json、frozen.pt")
    a = ap.parse_args(argv)
    chosen = lambda: Path(json.loads((REGEN / "selection.json").read_text())["chosen"]["path"])   # noqa: E731
    {"build": build_regen, "train": select_regen, "ablate": ablate_regen, "bounds": sent_range, "check": check_regen,
     "freeze": lambda: freeze_regen(chosen())}[a.cmd]()
    return 0


if __name__ == "__main__":
    gc.disable()                      # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    code = main()
    sys.stdout.flush()
    os._exit(code)
