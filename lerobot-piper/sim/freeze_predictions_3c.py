#!/usr/bin/env python3
"""③c 上机前校验 ③：**冻结 G-2 与 H 对两条序列的逐帧预测**（零机时，最后一件离线活）。

用户 2026-09-21 的三条要求，逐条实现
------------------------------------
1. **明确 G-2 的具体版本、权重哈希及 H 哈希** —— 避免混用「原始 G-2」与 `g2-incr-v1`。
   本脚本用 **`G2v0`（原始冻结 v=0，`rs_fir.npz`）**，**不是** `g2-incr-v1`（增量约束版）。
2. **正负两侧分别计算，不能通过取负生成另一侧。**
   三次独立前向：`u` · `u+δa` · `u−δa`。
3. **明确模型输入采用浮点请求还是量化后的发送序列。**
   ⇒ **主口径 = 量化后的发送序列**（那才是设备真正收到的），
   另存浮点请求版作对照。两者都冻结，主口径写死在 manifest 里。

⚠ 坐标系与对齐
--------------
当日 `(R, L)` **尚未获得** ⇒ **只冻结原生坐标（G-2 输出系）的预测**。
之后**仅按预定规则**应用由**独立 `A0`** 标定出的 `(R, L)`；
**不得用 `B` 的结果挑对齐**（③a 实测：冻结 R 给 cos 0.084，自由旋转给 0.412）。

⚠ 时间轴
--------
- `G-2` 的 `omega()` 输出在 20 Hz 名义栅格上，与指令同索引。
- `H` 的 `causal_conv` 在 **gap 时间轴**：`gap[t] = real[t+L] − ideal[t]`，`L = ALIGN_L = 3`。
- 索引约定：`np.diff` 索引 `i` ↔ 帧 `i → i+1`。
**两者不同轴，合并使用时必须显式对齐 —— 本脚本只冻结，不合并。**

用法:
    python -u sim/freeze_predictions_3c.py
"""
from __future__ import annotations
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (ROOT, HERE, ROOT / "sim/step4_gvnext"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SEQ = ROOT / "outputs/probe_3c"
OUT = SEQ / "frozen_predictions.npz"
MAN = SEQ / "frozen_predictions.json"
JOINT_FACTOR = 57324.840764
FPS = 20.0


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    import gc
    gc.disable()
    import fit_step1_candidate as S1
    from freeze_pi_g2wall import G2v0
    import g2_on_ab4 as G

    # ── 输入：浮点请求 与 量化后的发送序列 ────────────────────────────
    zp, zm = np.load(SEQ / "probe_3c_p.npz"), np.load(SEQ / "probe_3c_m.npz")
    sc = np.load(SEQ / "sent_counts.npz")
    u = zp["cmd7"].astype(np.float64)
    da_f_p, da_f_m = zp["da"].astype(np.float64), zm["da"].astype(np.float64)
    T = len(u)
    # **实际发送**的增量（量化后）—— 两侧**并非严格反号**
    c0, cp, cm = sc["c0"].astype(np.float64), sc["cp"].astype(np.float64), sc["cm"].astype(np.float64)
    da_q_p = (cp - c0) / JOINT_FACTOR
    da_q_m = (cm - c0) / JOINT_FACTOR
    asym = float(np.abs(da_q_p + da_q_m).max())
    print(f"序列 {T} 帧 · 量化后两侧非反对称度 max|δa₊+δa₋| = {asym*1e3:.5f} mrad "
          f"（浮点侧为 {np.abs(da_f_p + da_f_m).max()*1e3:.5f}）")

    # ── G-2：**明确版本** ─────────────────────────────────────────────
    g2 = G2v0()
    RQ, RS = ROOT / "outputs/step4_gvnext/rq", ROOT / "outputs/step4_gvnext/rs"
    g2_man = {
        "artifact": "G-2 (original, v=0)  ⚠ **不是 g2-incr-v1（增量约束版）**",
        "class": "freeze_pi_g2wall.G2v0",
        "W_sha256": g2.man["W_sha256"],
        "rs_fir_npz_sha256": sha(RS / "rs_fir.npz"),
        "rq_fir_npz_sha256": sha(RQ / "rq_fir.npz"),
        "rq_select_json_sha256": sha(RQ / "select.json"),
        "nlag": int(g2.nlag),
    }
    print(f"\nG-2 = {g2_man['artifact']}")
    print(f"  W sha256 {g2_man['W_sha256'][:32]}… · rs_fir {g2_man['rs_fir_npz_sha256'][:16]}…")

    h_path = ROOT / "outputs/step1_H/step1_H_v1.npz"
    H = np.ascontiguousarray(np.load(h_path)["H"], np.float64)
    h_man = {"path": str(h_path.relative_to(ROOT)), "file_sha256": sha(h_path),
             "array_sha256": hashlib.sha256(H.tobytes()).hexdigest(),
             "shape": list(H.shape), "n_lags": int(H.shape[0]),
             "time_axis": "gap 轴：gap[t] = real[t+L] − ideal[t]，L = ALIGN_L = 3"}
    print(f"H  = {h_man['path']} · array sha256 {h_man['array_sha256'][:32]}…")

    # ── 三次独立前向（**不取负**）× 两种输入口径 ───────────────────────
    def dd_of(cmd7):
        t = np.arange(len(cmd7)) / FPS            # ⚠ 名义均匀 20 Hz（当日时间戳另记）
        return {"name": "3c", "cmd": np.asarray(cmd7, float), "t_send": t, "t_read": t,
                "tg_i": np.arange(len(cmd7) * 5) / (FPS * 5),
                "gyro100": np.zeros((len(cmd7) * 5, 3))}

    store, report = {}, {}
    for kind, dap, dam in (("quantized", da_q_p, da_q_m), ("float", da_f_p, da_f_m)):
        cmds = {"base": u.copy()}
        for tag, d in (("plus", dap), ("minus", dam)):
            c = u.copy(); c[:, :6] += d          # **各自独立构造**
            cmds[tag] = c
        w = {}
        for tag, c in cmds.items():
            w[tag] = np.asarray(g2.omega(dd_of(c)), float)   # **三次独立前向**
            print(f"  G-2 前向 [{kind}/{tag}] → {w[tag].shape}")
        n = min(len(v) for v in w.values())
        for tag in w:
            w[tag] = w[tag][:n]
        # G-2 的单侧增量 / 奇 / 偶（**原生坐标系，未施加 R**）
        gp, gm = w["plus"] - w["base"], w["minus"] - w["base"]
        # H 的预测：两侧**各自**卷积（量化口径下 δa 非严格反号 ⇒ 结果不同）
        hp = S1.causal_conv(H, dap)
        hm = S1.causal_conv(H, dam)
        for k, v in (("g2_base", w["base"]), ("g2_plus", w["plus"]), ("g2_minus", w["minus"]),
                     ("g2_delta_plus", gp), ("g2_delta_minus", gm),
                     ("g2_odd", 0.5 * (gp - gm)), ("g2_even", gp + gm),
                     ("h_plus", hp), ("h_minus", hm),
                     ("h_odd", 0.5 * (hp - hm)), ("h_even", hp + hm)):
            store[f"{kind}__{k}"] = v.astype(np.float64)
        report[kind] = {
            "n_frames_g2": int(n),
            "g2_odd_rms": float(np.sqrt((store[f'{kind}__g2_odd'] ** 2).mean())),
            "g2_even_rms": float(np.sqrt((store[f'{kind}__g2_even'] ** 2).mean())),
            "h_odd_rms": float(np.sqrt((store[f'{kind}__h_odd'] ** 2).mean())),
            "h_even_rms": float(np.sqrt((store[f'{kind}__h_even'] ** 2).mean())),
        }
        print(f"  [{kind}] G-2 奇 RMS {report[kind]['g2_odd_rms']:.6f} · "
              f"偶 RMS {report[kind]['g2_even_rms']:.6f} ｜ "
              f"H 奇 RMS {report[kind]['h_odd_rms']:.6f} · "
              f"偶 RMS {report[kind]['h_even_rms']:.6f}")

    np.savez(OUT, **store)
    man = {
        "artifact": "probe-3c-frozen-predictions-v1",
        "primary_input": "quantized",
        "primary_input_note": "**主口径 = 量化后的发送序列**（设备真正收到的）；"
                              "`float`（浮点请求）另存作对照。",
        "sides_computed": "**正负两侧各自独立前向**（`u`, `u+δa`, `u−δa` 三次），"
                          "**不是**由一侧取负生成",
        "quantized_antisymmetry_break_mrad": asym * 1e3,
        "frames": {"T_sequence": T, "index_convention":
                   "np.diff 索引 i ↔ 帧 i→i+1；数组索引 = 序列帧号，从 0 起"},
        "coordinate_frame": "**原生坐标（G-2 输出系），未施加 R**。"
                            "当日 (R,L) 由**独立 A0** 标定后**按预定规则**应用；"
                            "**不得用 B 的结果挑对齐**",
        "time_axis": {"g2": "20 Hz 名义栅格，与指令同索引（时基假设均匀，当日时间戳另记）",
                      "H": h_man["time_axis"],
                      "warning": "**两者不同轴，合并使用时必须显式对齐；本文件只冻结不合并**"},
        "g2": g2_man, "H": h_man,
        "sequences": {"p_sha256": sha(SEQ / "probe_3c_p.npz"),
                      "m_sha256": sha(SEQ / "probe_3c_m.npz"),
                      "sent_counts_sha256": sha(SEQ / "sent_counts.npz")},
        "env": {"numpy": np.__version__, "python": platform.python_version()},
        "summary": report,
        "status": "**离线准备完成** —— 等待另行确认 R 阶段。"
                  "自测通过**不替代**设备安全依据；W/M/P 不动。",
    }
    man["predictions_npz_sha256"] = sha(OUT)
    MAN.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"\n→ {OUT.name}  sha {man['predictions_npz_sha256'][:32]}…")
    print(f"→ {MAN.name}")
    print(f"\n  主口径 = **{man['primary_input']}**（量化后的发送序列）")
    print(f"  坐标系 = **原生（未施加 R）**；(R,L) 待当日由独立 A0 定，"
          f"**不得用 B 挑对齐**")
    print(f"\n  ⛔ **离线准备到此完成。等待另行确认 R 阶段；W/M/P 不动。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
