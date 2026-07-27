#!/usr/bin/env python3
"""门禁自身的反向测试：合成六条记录，故意注入各类失败，probe_b2_gate 必须抓到。

上机前跑一次，改过门禁后也跑一次：
    python sim/test_probe_b2_gate.py

已覆盖：丢样 / 设备时间回退 / B−≠−B+ / 起始位形不一致 / 偶次非线性 / 响应超包络。
灵敏度（3 窗 @ 噪声底 45 mrad/s）：⑤b 宽带偶分量可检出峰值 ~20-30% 以上的二次失真；
更小的失真在该噪声与窗数下不可检出 —— 门禁"过"只代表在这个灵敏度内没发现。
"""
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_probe_b2 as A
import probe_b2_gate as G

import tempfile
TMP = Path(tempfile.gettempdir()) / "pb2_gate_test"
SRC = Path(__file__).resolve().parents[1] / "outputs/probe_b2"
FPS, MD = 20.0, np.pi / 180 / 1000


def make_rec(seq, sign, cfg, seed, g=1.4, tau=0.075, rot=8.0, nz=0.045,
             ppm=180.0, bad=None):
    rng = np.random.default_rng(seed)
    j = int(seq["joint"])
    base = np.asarray(seq["cmd7"], float)
    da = np.zeros((len(base), 6))
    if sign:
        da[:, j] = seq["da"][:, j] * sign
    cmd = base.copy(); cmd[:, :6] += da
    n20 = len(base)
    t0 = 1000.0 + rng.uniform(0, 5)
    ts = t0 + np.arange(n20) / FPS + rng.normal(0, 0.0015, n20)
    ts = np.maximum.accumulate(ts)
    ms = np.arange(0, n20 / FPS * 100) * 10.0
    tsamp = t0 + (ms / 1000.0) * (1 + ppm * 1e-6)
    host = tsamp + rng.exponential(0.0012, len(ms)) + \
        (rng.random(len(ms)) < 0.02) * rng.exponential(0.015, len(ms))
    y = A._bl_eval(seq["pred"].mean(0), (tsamp - t0) - tau) * g * (sign or 0)
    if rot and sign:
        c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        y = np.stack([c*y[:, 0]-s*y[:, 1], s*y[:, 0]+c*y[:, 1], y[:, 2]], 1)
    y = y + rng.normal(0, nz, y.shape)
    if bad == "even":                       # 注入**偶次**非线性(y² 不随 δa 变号 ⇒ 落进偶分量)
        y = y + 10.0 * y ** 2      # c=10 ⇒ 峰值处失真约 60%, 在 ⑤b 灵敏度内
    if bad == "drop":
        keep = rng.random(len(ms)) > 0.25
        ms, host, y = ms[keep], host[keep], y[keep]
    if bad == "back":
        ms = ms.copy(); ms[len(ms)//2] -= 500
    if bad == "loud":                       # 响应远超既有包络
        y = y * 12.0
    # 编码器 ≠ 指令: 真机伺服平滑指令阶跃, 且对 7Hz 探针衰减很大(Ĝ 的 servo FIR 即此)。
    # 夹具用二阶低通模拟, 否则"指令的 20Hz 差分"会比真实编码器速度高一倍。
    from scipy.signal import butter, filtfilt
    bb, aa = butter(2, 4.0, fs=FPS, btype="low")
    q = np.zeros((n20, 7))
    q[:, :6] = filtfilt(bb, aa, base[:, :6] + da * 0.30, axis=0) / MD
    q[:, 6] = base[:, 6] / MD
    imu100 = np.concatenate([host[:, None], ms[:, None],
                             np.zeros((len(ms), 3)), y], 1)
    st0 = q[0].copy()
    if bad == "state":
        st0[1] += 2.0 / MD * np.pi / 180 * 0 + 2000.0   # 起始位形差 2°
    return {"arm": cfg, "imu": np.zeros((n20, 6)).tolist(),
            "t_send": ts.tolist(), "t_send_span": (np.abs(rng.normal(3e-4, 5e-5, n20))).tolist(),
            "cmd": cmd.tolist(), "state": q.tolist(), "state_init": st0.tolist(),
            "imu100": imu100.tolist(), "clock": "time.monotonic",
            "cmd_hash": hashlib.sha256(
                np.ascontiguousarray(base, np.float32).tobytes()).hexdigest()[:16]}


def build(bad_map=None):
    bad_map = bad_map or {}
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    for f in SRC.glob("seq_ep0_*_4mrad_*.npz"):
        shutil.copy(f, TMP / f.name)
    shutil.copy(SRC / "windows_ep0.json", TMP / "windows_ep0.json")
    for i, cfg in enumerate(G.order_for(0)):
        if cfg.startswith("A"):
            seq = np.load(TMP / "seq_ep0_j2_4mrad_p.npz", allow_pickle=True); sg = 0
        else:
            jn, _, sn = cfg.split("_")
            seq = np.load(TMP / f"seq_ep0_{jn}_4mrad_p.npz", allow_pickle=True)
            sg = +1 if sn == "p" else -1
        r = make_rec(seq, sg, cfg, 10 + i, bad=bad_map.get(cfg))
        if bad_map.get(cfg) == "loud":       # 编码器也一起放大, 模拟真在剧烈运动
            c = np.asarray(r["state"]); c[:, :6] *= 1.6; r["state"] = c.tolist()
        if bad_map.get(cfg) == "asym":       # B− 不是 −B+
            c = np.asarray(r["cmd"]); c[:, 2] += 1e-4; r["cmd"] = c.tolist()
        (TMP / f"replay_ep0_{cfg}.json").write_text(json.dumps(r))


def run(label, bad_map=None, expect_pass=True):
    build(bad_map)
    G.PB = TMP
    G.ENV_CACHE = TMP / "env.json"
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = G.main()
    out = buf.getvalue()
    got = (rc == 0)
    mark = "✓" if got == expect_pass else "✗ 门禁失效!"
    fails = [l.strip() for l in out.split("\n") if l.strip().startswith("✗")]
    print(f"{mark} {label:<28} 判决={'过' if got else '不过'}"
          f"  期望={'过' if expect_pass else '不过'}"
          + (f"   抓到: {fails[:2]}" if fails else ""))
    return out, got == expect_pass


if __name__ == "__main__":
    sys.argv = ["gate", "--ep", "0"]
    print("=== 门禁反向测试 (合成六条记录) ===")
    ok = True
    out, r = run("干净数据", None, True); ok &= r
    for lbl, bm, in [("注入: 丢样 25%", {"j2_4mrad_p": "drop"}),
                     ("注入: 设备时间回退", {"j3_4mrad_m": "back"}),
                     ("注入: B− ≠ −B+", {"j2_4mrad_m": "asym"}),
                     ("注入: 起始位形差 2°", {"A2": "state"}),
                     ("注入: 偶次非线性", {"j2_4mrad_p": "even", "j2_4mrad_m": "even"}),
                     ("注入: 响应超包络", {"j3_4mrad_p": "loud", "j3_4mrad_m": "loud"})]:
        _, r = run(lbl, bm, False); ok &= r
    print(f"\n{'✅ 门禁自身验证通过' if ok else '❌ 门禁有漏检'}")
    print("\n--- 干净数据的完整输出 ---")
    print(out)
