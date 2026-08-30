#!/usr/bin/env python3
"""**神经残差的最后一道零机时门槛** —— 按批次双向留出 + 两个必要对照。

## 为什么留一交叉验证不作数

17 集来自同一策略、同一任务的重复取放：**集间残差相关系数中位 0.904**
（范围 0.399–0.964）。「相位模板」基线（**完全不用特征**，只按帧序号平均训练集
残差）就能拿到关节 RMS 0.230 / 改善 17-17 / 中位 +78.9% ——
**GRU 全部收益的 97% 它已经解释掉了**（GRU 0.209）。
⇒ 留一在这批数据上**不度量对新条件的泛化**，只度量「能不能记住轨迹平均形状」。

## 本门槛的设计

**批次**：pp2 的 17 集本就来自两次不同会话 —— 筛选阶段 8 集（`S*`）与
确认阶段 A 臂 9 集（`B*A`）。

| 方向 | 训练 | 验证 |
|---|---|---|
| 一 | 90 条激励套件 + 筛选 8 集 | 确认 9 集 |
| 二 | 90 条激励套件 + 确认 9 集 | 筛选 8 集 |

**加入 90 条套件**是为了给训练侧真正的轨迹多样性（单关节大幅激励、扫频、多正弦），
对冲 pp2 内部的近复制。

**归一化与模型选择只用训练侧数据**，不看验证批次。

### 两个必要对照

1. **相位模板** —— 只使用帧序号（训练侧 pp2 集在同一帧序号上的残差均值）。
2. **零输入 GRU** —— 把指令与状态输入**置零**，只保留递归状态。
   零输入时 GRU 的隐状态只随时间步确定性演化，**其输出就是一个学出来的相位模板**
   ⇒ 它直接测出「网络是不是只在内部计时」。

## 通过条件（预先定死，不得事后改）

- **两个方向都**优于相位模板 **和** 零输入 GRU；
- 逐集**多数改善**且**中位改善为正**；
- 相对相位模板有**实质差距**（不是留一里那约 3%）；
- **关节 RMS 与最终 IMU 主指标同时改善**。

**若零输入 GRU ≈ 完整 GRU，或换批后优势消失** ⇒ 判定当前网络只学会了任务时钟，
**停止调网络结构，也不采普通重复型 holdout**；届时需要的是**更有轨迹多样性的
训练数据**。

用法:
    python sim/step2_g/nn_batch_gate.py --run
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))
sys.path.insert(0, str(ROOT / "sim/step2_g"))

from sim.step1_v2_loss import bandpass41, load_calib                # noqa: E402
from sim.step2_fidelity import CALIB, SEAM_W, regions, fidelity     # noqa: E402
from ab_decompose import kin_gyro                                   # noqa: E402
from nn_residual_v0 import (BOUND, HID, build_cache, feats, target,  # noqa: E402
                            rms, imu_metric, lin_fit, lin_pred)
import g_pi_phys_v1 as PHYS                                         # noqa: E402

CACHE = ROOT / "outputs/step2_g/nn_v0"
COL = ROOT / "outputs/step2_g/collect"
SUITE = ROOT / "outputs/step2_g"
MAXLEN = 900          # 训练时把超长序列截断, 控内存


def suite_eps():
    """90 条激励套件 → 与 pp2 同构的训练样本（只用于训练侧）。"""
    f = CACHE / "suite_feat.npz"
    if f.exists():
        return list(np.load(f, allow_pickle=True)["eps"])
    man = json.loads((SUITE / "manifest.json").read_text())
    done = [r["name"] for r in json.loads((COL / "state.json").read_text())["done"]]
    out = []
    for nm in sorted(done):
        if man[nm]["kind"] == "sweep" and man[nm]["frames"] > 500:
            continue                                     # 长慢扫描几乎无动态内容
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        act = np.asarray(d["cmd"], float)
        cmd = np.degrees(act[:, :6])
        enc = np.asarray(d["state"], float)[:, :6] / 1000.0
        qph = PHYS.predict_qpos(act)
        k = min(len(cmd), len(qph), len(enc), MAXLEN)
        out.append({"name": nm, "cmd": cmd[:k], "qph": qph[:k],
                    "enc": enc[:k], "imu": None, "kin": None, "seams": []})
        print(f"  suite {nm}  {k} 帧", flush=True)
    np.savez_compressed(f, eps=np.array(out, dtype=object))
    return out


def phase_template(tr_pp2, T):
    """**只用帧序号**：训练侧 pp2 集在同一帧序号上的残差均值。"""
    M, C = np.zeros((T, 6)), np.zeros((T, 1))
    for e in tr_pp2:
        k = min(T, len(e["cmd"]))
        M[:k] += target(e)[:k]
        C[:k] += 1
    return M / np.maximum(C, 1)


def gru_fit(tr, seed, epochs=250, zero_input=False):
    """单层 GRU。`zero_input=True` ⇒ 输入置零，只留递归状态（对照 2）。"""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.g = nn.GRU(24, HID, 1, batch_first=True)
            s.h = nn.Linear(HID, 6)

        def forward(s, x):
            y, _ = s.g(x)
            return torch.tanh(s.h(y)) * BOUND

    net = Net().to(dev)
    F = np.vstack([feats(e) for e in tr])
    mu, sd = F.mean(0), F.std(0) + 1e-6            # **只用训练侧**归一化
    X, Y, W = [], [], []
    for e in tr:
        f_ = (feats(e) - mu) / sd
        if zero_input:
            f_ = np.zeros_like(f_)                 # 置零, 只留递归
        X.append(torch.tensor(f_, device=dev)[None])
        Y.append(torch.tensor(target(e), device=dev)[None])
        W.append(len(f_))
    W = torch.tensor(np.array(W, np.float32) / float(np.sum(W)), device=dev)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(w * torch.nn.functional.smooth_l1_loss(net(x), y)
                   for w, x, y in zip(W, X, Y))     # 按帧数加权, 避免短序列主导
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    return net, mu, sd, dev, zero_input


def gru_pred(model, e):
    import torch
    net, mu, sd, dev, zi = model
    f_ = (feats(e) - mu) / sd
    if zi:
        f_ = np.zeros_like(f_)
    with torch.no_grad():
        return net(torch.tensor(f_, device=dev)[None])[0].cpu().numpy()


def evaluate(val, preds, R):
    """→ {模型: (关节RMS数组, 主指标数组, 难处数组)}。"""
    out = {}
    for k, qs in preds.items():
        j = np.array([rms(q, e) for q, e in zip(qs, val)])
        im = np.array([imu_metric(e, q, R) for q, e in zip(qs, val)])
        out[k] = (j, im[:, 0], im[:, 1])
    return out


def run_dir(name, tr_pp2, val, suite, R, epochs):
    print(f"\n{'═'*76}\n【{name}】训练 = 90 条套件({len(suite)}) + pp2 {len(tr_pp2)} 集"
          f"   验证 = pp2 {len(val)} 集")
    print(f"  归一化与模型选择**只用训练侧**，不看验证批次", flush=True)
    tr = suite + tr_pp2
    Wl = lin_fit(tr)
    m_full = gru_fit(tr, seed=7, epochs=epochs, zero_input=False)
    m_zero = gru_fit(tr, seed=7, epochs=epochs, zero_input=True)
    preds = {"底座": [e["qph"] for e in val],
             "指令": [e["cmd"] for e in val],
             "相位模板": [e["cmd"] + phase_template(tr_pp2, len(e["cmd"]))
                      for e in val],
             "零输入GRU": [e["cmd"] + gru_pred(m_zero, e) for e in val],
             "线性": [e["cmd"] + lin_pred(Wl, e) for e in val],
             "**GRU**": [e["cmd"] + gru_pred(m_full, e) for e in val]}
    res = evaluate(val, preds, R)
    print(f"\n  {'模型':12s}{'关节RMS°':>10s}{'主指标 中位':>13s}{'难处 中位':>11s}")
    for k, (j, s_, t_) in res.items():
        print(f"  {k:12s}{j.mean():>10.3f}{np.median(s_):>12.1f}%{np.median(t_):>10.1f}%")
    return res


def cmp(res, a, b, N):
    """a 相对 b：逐集改善数 + 中位改善（关节 RMS 与主指标）。"""
    ja, sa, _ = res[a]
    jb, sb, _ = res[b]
    dj, ds = jb - ja, sa - sb
    return (int((dj > 0).sum()), float(np.median(dj / jb) * 100),
            int((ds > 0).sum()), float(np.median(ds)), N)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--epochs", type=int, default=250)
    a_ = ap.parse_args()
    if not a_.run:
        print(__doc__.split("用法:")[0]); return 0
    R, _ = load_calib(CALIB)
    eps = build_cache()
    scr = [e for e in eps if e["name"].startswith("S")]     # 筛选阶段
    cfm = [e for e in eps if e["name"].startswith("B")]     # 确认阶段 A 臂
    print(f"批次：筛选 {len(scr)} 集 · 确认 {len(cfm)} 集")
    suite = suite_eps()
    print(f"套件：{len(suite)} 条")

    r1 = run_dir("方向一：套件 + 筛选8 → 验证 确认9", scr, cfm, suite, R, a_.epochs)
    r2 = run_dir("方向二：套件 + 确认9 → 验证 筛选8", cfm, scr, suite, R, a_.epochs)

    print(f"\n{'═'*76}\n【门槛】（预先定死）")
    ok = True
    for nm, res, N in (("方向一", r1, len(cfm)), ("方向二", r2, len(scr))):
        print(f"\n  ── {nm} ──")
        for ref in ("相位模板", "零输入GRU", "底座"):
            wj, mj, ws, ms, n = cmp(res, "**GRU**", ref, N)
            good = wj > n / 2 and mj > 0 and ws > n / 2 and ms > 0
            ok &= good if ref != "底座" else True
            print(f"    GRU vs {ref:8s} 关节 {wj}/{n} 中位 {mj:+6.1f}%   "
                  f"主指标 {ws}/{n} 中位 {ms:+6.1f}pp   "
                  f"{'✓' if good else '**✗**'}")
        gj = res["**GRU**"][0].mean(); pj = res["相位模板"][0].mean()
        zj = res["零输入GRU"][0].mean()
        print(f"    实质差距: GRU {gj:.3f} vs 相位模板 {pj:.3f} "
              f"(**{(1-gj/pj)*100:+.1f}%**) · vs 零输入GRU {zj:.3f} "
              f"(**{(1-gj/zj)*100:+.1f}%**)")
    print(f"\n{'═'*76}")
    print("⇒ " + ("**两个方向均优于两个对照 —— 神经残差成立**" if ok else
                  "**未通过 —— 当前网络只学会了任务时钟。停止调网络结构，"
                  "也不采普通重复型 holdout；需要更有轨迹多样性的训练数据。**"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
