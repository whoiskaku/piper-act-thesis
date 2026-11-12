#!/usr/bin/env python3
"""Ĝ 表达判决实验: NN(TCN) vs Gray-box — 值拟合谁都行, 导数(扰动响应)谁可信?

设定: 两模型同输入(指令历史窗 12帧×6关节)同目标(真实gyro 3ch, 会话R系)。
探针: 留出集随机站点, 关节j指令加 5mrad 单帧脉冲 → 模型预测响应序列(9帧)。
真值: 响应方向须沿 R·Jr(q(t))[:,j] (解析, 逐时刻)。
评分: 能量加权方向余弦 + 响应总幅度(防"装死"零梯度)。
"""
from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn as nn

from compare_ideal_methods import load_episode
from ideal_imu_kinematic import MODEL_PATH, full_qpos

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)
KIN, DT = 12, 0.05           # 输入窗(0.6s)
PROBE = 0.005                # 5 mrad 脉冲
LAGS = 9                     # 观察响应 0..8 帧
DATASETS = [("enolab-guo/eval_pickplace2_100k", "pp2-deploy"),
            ("enolab-guo/eval_insert_100k", "ins-deploy")]
TRAIN_EPS, VAL_EPS = list(range(8)), [8, 9]

# ---- 会话标定 R (labels manifest 里已存) ----
def load_R(tag):
    m = json.loads(open(f"../outputs/p2_labels/{tag}/manifest.json").read())
    return np.array(m["R"], np.float32)

# ---- mujoco: Jr 列 (解析真值方向) ----
import mujoco
_mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
_mj_data = mujoco.MjData(_mj_model)
_sid = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")

def jr_at(q6):
    _mj_data.qpos[:] = full_qpos(np.concatenate([q6, [0.0]]))
    mujoco.mj_fwdPosition(_mj_model, _mj_data)
    jacp = np.zeros((3, _mj_model.nv)); jacr = np.zeros((3, _mj_model.nv))
    mujoco.mj_jacSite(_mj_model, _mj_data, jacp, jacr, _sid)
    R_site = _mj_data.site_xmat[_sid].reshape(3, 3)
    return (R_site.T @ jacr[:, :6])                    # site系 3×6

# ---- 数据 ----
def load(tag_eps, with_gap=False):
    """with_gap: 追加编码器q(裁判用)与 gap=real−kin_ideal(v2评分靶)。"""
    out = []
    for repo, tag in DATASETS:
        R = load_R(tag)
        for ep in tag_eps:
            d = load_episode(repo, ep)
            a = d["action"][:, :6].astype(np.float32)
            g = d["imu_real"][:, 3:].astype(np.float32)   # 真实gyro(TSND系)
            rec = {"a": a, "g": g, "R": R, "tag": tag}
            if with_gap:
                from compare_ideal_methods import kin_ideal_20
                rec["q"] = (d["state_mdeg"][:, :6] * (np.pi / 180 / 1000)).astype(np.float32)
                kin, _, _ = kin_ideal_20(repo, ep)
                T = min(len(kin), len(g))
                ideal_g = (kin[:T, 3:] @ R.T).astype(np.float32)
                rec = {k: (v[:T] if isinstance(v, np.ndarray) and len(v) > 6 else v)
                       for k, v in rec.items()}
                rec["gap"] = rec["g"] - ideal_g
            out.append(rec)
    return out

# ---- 模型1: Gray-box  ω̂(t) = R·Jr(q_t)·(Σ_k c[k]·ȧ(t−k)) ----
class GrayBox:
    def __init__(self, K=10):
        self.K = K
        self.c = np.zeros((6, K))                       # 每关节K抽头(作用在ȧ上)

    def _feats(self, ep):
        a, R = ep["a"], ep["R"]
        T = len(a)
        ad = np.vstack([np.zeros((1, 6)), np.diff(a, axis=0)]) / DT
        J = np.stack([R @ jr_at(a[max(t - 2, 0)]) for t in range(T)])  # (T,3,6) q≈滞后指令
        # 设计矩阵: pred(t) = Σ_j Σ_k c[j,k] · J[t,:,j]·ad[t−k,j]
        X = np.zeros((T, 3, 6, self.K), np.float32)
        for k in range(self.K):
            sh = np.vstack([np.zeros((k, 6)), ad[:T - k]])
            X[:, :, :, k] = J * sh[:, None, :]
        return X

    def fit(self, eps):
        Xs, ys = [], []
        for ep in eps:
            Xs.append(self._feats(ep).reshape(len(ep["a"]), 3, -1))
            ys.append(ep["g"])
        X = np.concatenate(Xs).reshape(-1, 6 * self.K)
        y = np.concatenate(ys).reshape(-1)
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.c = w.reshape(6, self.K)
        self._cache = {}

    def predict(self, ep):
        X = self._feats(ep)
        return np.einsum("tcjk,jk->tc", X, self.c)

# ---- 模型2: NN(TCN) 指令窗→gyro ----
class TCNG(nn.Module):
    def __init__(self, h=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(6, h, 3, dilation=1),
                                    nn.Conv1d(h, h, 3, dilation=2),
                                    nn.Conv1d(h, h, 3, dilation=4)])
        self.pads = [2, 4, 8]
        self.out = nn.Linear(h, 3)

    def forward(self, u):                               # (B,T,6)
        x = u.transpose(1, 2)
        for c, p in zip(self.convs, self.pads):
            x = torch.nn.functional.gelu(c(torch.nn.functional.pad(x, (p, 0))))
        return self.out(x.transpose(1, 2))              # (B,T,3)

def fit_nn(eps, steps=4000):
    net = TCNG()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lens = np.array([len(e["a"]) - 64 for e in eps], float)
    p = lens / lens.sum()
    for i in range(steps):
        bi = np.random.choice(len(eps), 64, p=p)
        xb, yb = [], []
        for b in bi:
            o = np.random.randint(0, lens[b])
            xb.append(eps[b]["a"][o:o + 64]); yb.append(eps[b]["g"][o:o + 64])
        loss = torch.nn.functional.mse_loss(
            net(torch.from_numpy(np.stack(xb))), torch.from_numpy(np.stack(yb)))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net

# ---- 值拟合 R² ----
def r2(pred, y):
    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()

# ---- 统一探针: 模型=黑盒函数 f(a_episode)→gyro_episode ----
def probe(f, ep, sites, joints=(0, 1, 2, 4)):          # j1,j2,j3,j5(有效关节)
    a, R = ep["a"], ep["R"]
    base = f(a)
    recs = []
    for t0 in sites:
        for j in joints:
            ap = a.copy(); ap[t0, j] += PROBE
            resp = (f(ap) - base)[t0:t0 + LAGS]         # (LAGS,3)
            dirs = np.stack([R @ jr_at(a[min(t0 + k, len(a) - 1)])[:, j]
                             for k in range(LAGS)])
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            mag = np.linalg.norm(resp, axis=1)
            cosk = np.abs((resp * dirs).sum(1)) / (mag + 1e-12)
            w = mag / (mag.sum() + 1e-12)
            recs.append({"cos": float((cosk * w).sum()), "mag": float(mag.sum())})
    return recs

def main():
    train, val = load(TRAIN_EPS), load(VAL_EPS)
    gb = GrayBox(); gb.fit(train)
    nn_ = fit_nn(train)

    def f_gb(a):
        return gb.predict({"a": a, "R": val_ep["R"]})
    def f_nn(a):
        with torch.no_grad():
            return nn_(torch.from_numpy(a[None]))[0].numpy()

    # 值拟合
    r2g, r2n = [], []
    for ep in val:
        r2g.append(r2(gb.predict(ep), ep["g"]))
        with torch.no_grad():
            r2n.append(r2(nn_(torch.from_numpy(ep["a"][None]))[0].numpy(), ep["g"]))
    print(f"值拟合 R²(留出): Gray-box={np.mean(r2g):.3f}  NN={np.mean(r2n):.3f}")

    # 导数探针
    all_g, all_n = [], []
    rng = np.random.default_rng(0)
    for val_ep in val:
        sites = rng.integers(KIN, len(val_ep["a"]) - LAGS - 1, 10)
        all_g += probe(f_gb, val_ep, sites)
        all_n += probe(f_nn, val_ep, sites)
    cg = np.array([r["cos"] for r in all_g]); cn = np.array([r["cos"] for r in all_n])
    mg = np.array([r["mag"] for r in all_g]); mn = np.array([r["mag"] for r in all_n])
    print(f"\n== 导数判决 ({len(cg)} 次干预探针, 5mrad脉冲, 方向真值=R·Jr[:,j]) ==")
    print(f"方向余弦 中位[IQR]: Gray-box {np.median(cg):.3f} [{np.percentile(cg,25):.3f},{np.percentile(cg,75):.3f}]"
          f"  |  NN {np.median(cn):.3f} [{np.percentile(cn,25):.3f},{np.percentile(cn,75):.3f}]")
    print(f"响应幅度 中位:      Gray-box {np.median(mg):.4f}  |  NN {np.median(mn):.4f} "
          f" (比值 {np.median(mn)/max(np.median(mg),1e-12):.2f}×; ≪1=NN对扰动装死)")
    print(f"NN 探针中 cos<0.5 的占比: {(cn<0.5).mean()*100:.0f}%  (方向不可信的比例)")

if __name__ == "__main__":
    main()
