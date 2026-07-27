#!/usr/bin/env python3
"""诊断第5步(用户设计) — 限频校准 + 按关节拆频谱 + 动作带宽消融。

背景: 探针(5Hz小波)只充分激励 3-8Hz; π 有 38% 动作能量在 8-10Hz(未验证)。
      故第4步的"全频乘增益"= 外推压力测试, 不能当次因规模。

① 限频校准: 只在 3-8Hz 应用探针逐关节增益, 其余频段保持原 Ĝ。TF 即可(AR≈TF已证)。
② 按关节拆频谱: 8-10Hz 的 38% 是否集中在 j3? 若是, 模型失配与高频输出可能同源。
③ 动作带宽消融: δa 限到 <5/<6/<7/<8Hz, 在 原始Ĝ / 限频校准Ĝ / 悲观成员 下统一判卷。
   去掉8-10Hz后不降 → 无用抖动, 应用 γ 或因果低通压掉;
   明显下降 → π 依赖未验证频段, 必须补探针。
"""
import json, numpy as np, torch
from pathlib import Path
from scipy.signal import firwin, welch
import train_pi_e2 as TP
from compare_ideal_methods import load_episode
from lerobot.common.policies.act.b_runtime import (FeatureAssembler, IdealChunkRenderer,
                                                   load_tcn, CHUNK, FEAT_DIM)
ROOT = Path(__file__).resolve().parents[1]
TP.BP = TP.make_bp(); G = TP.GEnsemble()
man = json.loads((ROOT/"outputs/p2_labels/pp2-deploy/manifest.json").read_text())
R = np.array(man["R"], np.float32); b = np.array(man["b"], np.float32); MD = np.pi/180/1000
pi = load_tcn(ROOT/"outputs/p2_pi/pi_v1_s0.pt").to(TP.DEV); rend = IdealChunkRenderer()
FAIL = {6}; L = 3
GJ = np.array([1.40, 1.108, 2.271, 1.40, 1.326, 1.40], np.float32)   # 探针逐关节增益
BAND_OK = (3.0, 8.0)          # 探针充分激励的频段


def fir_band(x, lo, hi, taps=41):
    """零相位带通/低通(对 δa 分频, 非因果但仅用于离线分析)。"""
    from scipy.signal import filtfilt
    if lo <= 0:
        k = firwin(taps, hi, fs=20.0)
    elif hi >= 9.99:
        k = firwin(taps, lo, fs=20.0, pass_zero=False)
    else:
        k = firwin(taps, [lo, hi], fs=20.0, pass_zero=False)
    return filtfilt(k, [1.0], x, axis=0).astype(np.float32)


def energy(x):
    n = len(x)
    return float((TP.BP(torch.from_numpy(np.ascontiguousarray(x, np.float32))
                        .view(1, 1, n, 3).to(TP.DEV))[0, 0] ** 2).sum())


def prep(cmd7, imu, st):
    T = min(len(cmd7), len(imu), len(st)); cmd7, imu, st = cmd7[:T], imu[:T], st[:T]
    asm = FeatureAssembler(R, b); prev = None; ci = 0
    starts = list(range(0, T, CHUNK)) + [T]; X = np.zeros((T, FEAT_DIM), np.float32)
    for t in range(T):
        if ci < len(starts)-1 and t == starts[ci]:
            seg = cmd7[starts[ci]:starts[ci+1]]
            asm.on_new_chunk(seg, rend.render(seg, prev)); prev = seg; ci += 1
        w, _ = asm.step(imu[t], (st[t, :6]*MD).astype(np.float32)); X[t] = w[-1]
    ideal_g = np.array(asm.ideal)[:, 3:]; n = T - L
    e = (imu[L:L+n, 3:6] - ideal_g[:n]).astype(np.float32)
    epj, al, RB, RD = G.ep_context(cmd7[:n, :6], cmd7[:n, 6], R)
    return X[:n], e, (epj.unsqueeze(0), al.unsqueeze(0), RB.unsqueeze(0), RD.unsqueeze(0))


def respond(da_np, ctx, model, worst_k=0):
    """da(n,6) → Ĝ 响应(n,3)。model: raw / calib_band / calib_full / pess"""
    if model == "calib_full":
        da_np = da_np * GJ
    elif model == "calib_band":
        # 只在 3-8Hz 乘增益, 其余频段原样(增益增量只作用于带内分量)
        inb = fir_band(da_np, *BAND_OK)
        da_np = da_np + inb * (GJ - 1.0)
    d = torch.from_numpy(np.ascontiguousarray(da_np, np.float32)).unsqueeze(0).to(TP.DEV)
    de = TP.apply_G(d, *ctx, G)[0]
    return de[worst_k].cpu().numpy() if model == "pess" else de.mean(0).cpu().numpy()


def main():
    das_by_ep, eps = [], []
    for ep in range(10):
        if ep in FAIL: continue
        d = load_episode("enolab-guo/eval_ab_A", ep)
        X, e, ctx = prep(d["action"].astype(np.float32), d["imu_real"].astype(np.float32),
                         d["state_mdeg"].astype(np.float32))
        with torch.no_grad():
            da = pi(torch.from_numpy(X).unsqueeze(0).to(TP.DEV))[0].cpu().numpy()
        de_K = TP.apply_G(torch.from_numpy(da).unsqueeze(0).to(TP.DEV), *ctx, G)[0]
        wk = int(np.argmax([energy(e + de_K[k].cpu().numpy()) for k in range(de_K.shape[0])]))
        eps.append((ep, X, e, ctx, da, wk)); das_by_ep.append(da)
        print(f"  prep A-ep{ep}", flush=True)

    # ---------- ② 按关节拆频谱 ----------
    da_all = np.concatenate(das_by_ep)
    f, P = welch(da_all, fs=20.0, axis=0, nperseg=128)          # (F,6)
    print(f"\n② δa 频谱按关节拆分 (占该关节自身能量的 %)")
    print(f"{'关节':>5}{'0-3Hz':>8}{'3-8Hz':>8}{'8-10Hz':>9}{'关节能量占比':>13}")
    tot = P.sum()
    for j in range(6):
        pj = P[:, j]; s = pj.sum()
        if s / tot < 0.01:
            continue
        b1 = pj[(f >= 0) & (f < 3)].sum()/s*100
        b2 = pj[(f >= 3) & (f < 8)].sum()/s*100
        b3 = pj[(f >= 8)].sum()/s*100
        print(f"j{j+1:>4}{b1:>7.0f}%{b2:>7.0f}%{b3:>8.0f}%{s/tot*100:>12.0f}%")

    # ---------- ①③ 限频校准 × 带宽消融 (TF) ----------
    MODELS = [("raw", "原始Ĝ"), ("calib_band", "限频校准Ĝ"),
              ("calib_full", "全频校准(压力测试)"), ("pess", "悲观成员")]
    BWS = [(None, "全带(原样)"), (8.0, "<8Hz"), (7.0, "<7Hz"), (6.0, "<6Hz"), (5.0, "<5Hz")]
    print(f"\n①③ 动作带宽消融 × 响应模型  (TF, 9集均值, 模型内消减%)")
    print(f"{'δa 带宽':>12}" + "".join(f"{lbl:>20}" for _, lbl in MODELS))
    res = {}
    for bw, bl in BWS:
        row = []
        for mk, _ in MODELS:
            vals = []
            for ep, X, e, ctx, da, wk in eps:
                d_ = da if bw is None else fir_band(da, 0.0, bw)
                vals.append((1 - energy(e + respond(d_, ctx, mk, wk))/energy(e))*100)
            row.append(np.mean(vals))
        res[bl] = row
        print(f"{bl:>12}" + "".join(f"{v:>19.1f}%" for v in row))
    (ROOT/"outputs/p2_pi/step5_bandlimited.json").write_text(json.dumps(
        {"models": [m for m, _ in MODELS], "rows": res}, indent=1))
    print("\n判读:")
    print("  限频校准 vs 原始Ĝ 的差 = **可信的次因规模**(全频校准仅为外推压力测试)")
    print("  去掉 8-10Hz(<8Hz行) 后不降 → 高频是无用抖动, 用 γ/低通压掉")
    print("  明显下降 → π 依赖未验证频段, 必须补探针")


if __name__ == "__main__":
    main()
