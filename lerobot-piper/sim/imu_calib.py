#!/usr/bin/env python3
"""IMU 安装朝向 / 位置的规定动作标定（老师 2026-10 计划：静止姿态看重力方向、一次只转一个关节）。

第一步（本文件现有部分）：生成标定动作序列，交给 `sim/replay_probe.py` 在真机上回放并高频记录。
  1. 静止姿态：从 HOME 出发变换腕部 j4/j5/j6，每个姿态保持 3 s
     ⇒ 加速度计看到不同方向的重力（解朝向 R 与零偏 b）；
  2. 单关节转动：j1…j6 各自从 HOME 出发来回 2 次，其余关节不动
     ⇒ 陀螺看到绕各关节轴的转动（解朝向 R），加速度计看到离心/切向项（解安装位置）。
生成时在 MuJoCo 模型里检查：关节限位（留 5°）、速度/加速度（STEP2 的抓放实测上限）、
离台面高度（≥ 2 cm）、静止姿态下重力方向是否足够分散。不通过就不输出文件。
⚠ 模型里没有桌上的物体：执行前清空桌面；先 dry-run。

第二步 `fit`：由真机回放记录解出 IMU 的朝向 R、加计零偏 b、陀螺零偏、安装位置偏差 dp、延迟。
参照 = 编码器 200 Hz 关节角经运动学算出的安装点运动（标定的是传感器，不掺控制器的跟踪误差）；
两边用同一个 3 Hz 低通，只比刚体运动。约定见 CONVENTION（R、b 与 BCORR_CALIB 的用法一致）。

用法：
    python -u sim/imu_calib.py seq --out outputs/imu_calib/calib_seq.npz
    python -u sim/imu_calib.py view --seq outputs/imu_calib/calib_seq.npz     # 目视确认
    python -u sim/imu_calib.py replay --dry-run      # 回放管线的安全检查，不连机械臂
    python -u sim/imu_calib.py replay                # 真机执行 → outputs/imu_calib/replay_calib.json
    python -u sim/imu_calib.py fit                   # 拟合 → outputs/imu_calib/imu_calib.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "step2_g")]
import sequences as SQ  # noqa: E402  STEP2 的 HOME 姿态、限位/速度检查、离台面高度
from step4_gvnext import data as D  # noqa: E402  回放记录 → 均匀 100/200 Hz 网格、低通

FPS = SQ.FPS
HOLD_S, FIT_S = 3.0, 2.0          # 静止保持 3 s，拟合只用后 2 s（避开到位后的余振）
V_FRAC = 0.8                      # 速度最多用到抓放实测上限的 80%
# 静止姿态：相对 HOME 的变化（关节下标 0–5 → 角度°）；首尾都是 HOME
STATIC = [{}, {4: 35}, {4: -35}, {3: 60}, {3: 60, 4: 35}, {3: -60, 4: -35}, {3: -60},
          {5: 80}, {5: -80}, {}]
MOVES = [(0, 20), (1, 10), (2, 15), (3, 45), (4, 30), (5, 60)]   # (关节下标, 幅度°)
N_REP = 2
# 第二步 fit
LP_DIFF_HZ, LP_HZ = 10.0, 3.0     # 关节角求导前的平滑；两边比较用的低通（只留刚体运动，去掉柔性振动）
GYRO_MIN = 0.15                   # rad/s：只用明显在转的样本解朝向（同仓库惯例 imu_frame_calib）
TAUS = np.arange(-100, 301) / 1000.0   # IMU 相对编码器的延迟搜索范围 −100…300 ms，步长 1 ms
CONVENTION = ("real_acc ≈ R·(f + ([α]×+[ω]×²)·dp) + b；real_gyro ≈ R·ω + gyro_bias。"
              "f/ω/α = 模型安装点 ee_imu_site 的比力/角速度/角加速度（site 系），由编码器关节角经运动学算出；"
              "dp = 真实 IMU 相对 ee_imu_site 的位置偏差（site 系，m）。"
              "IMU 样本时刻（设备 ms 经最小二乘映射到 monotonic，同 step4_gvnext/data.load）− delay_s"
              " = 该样本测到的运动时刻。只在 0–3 Hz 上拟合。")


def _pose(dv: dict) -> np.ndarray:
    p = SQ.HOME_DEG.copy()
    for j, v in dv.items():
        p[j] += v
    return np.deg2rad(p)


def _ramp(a: np.ndarray, b: np.ndarray, min_s: float) -> np.ndarray:
    """升余弦从 a 到 b（不含起点 a）；时长按最大关节变化与速度上限算，不短于 min_s。"""
    T = max(min_s, float(np.max(np.abs(b - a) * np.pi
                                / (2 * V_FRAC * np.deg2rad(SQ.VMAX_DEG)))))
    s = 0.5 * (1 - np.cos(np.pi * np.linspace(0.0, 1.0, int(np.ceil(T * FPS)) + 1)))
    out = a + (b - a) * s[1:, None]
    out[-1] = b             # 终点精确落在目标上：浮点下 a+(b−a)·1 可能与 b 差 1e-16
    return out


def make_sequence():
    """→ (cmd7 (T,7) rad, segs)。segs 每项 {name, kind, start, end[, fit_start]}，帧号，end 不含。"""
    rows, segs = [_pose({})], []

    def go(target, min_s):
        if not np.allclose(target, rows[-1]):
            rows.extend(_ramp(rows[-1], target, min_s))

    def stay(seconds):
        rows.extend([rows[-1]] * int(seconds * FPS))

    for dv in STATIC:
        go(_pose(dv), 1.5)
        a = len(rows)
        stay(HOLD_S)
        name = ",".join(f"j{j + 1}{v:+d}" for j, v in dv.items()) or "home"
        segs.append({"name": f"static {name}", "kind": "static", "start": a,
                     "end": len(rows), "fit_start": len(rows) - int(FIT_S * FPS)})
    for j, amp in MOVES:
        a = len(rows)
        for _ in range(N_REP):
            go(_pose({j: amp}), 0.6)
            stay(0.5)
            go(_pose({}), 0.6)
            stay(0.5)
        segs.append({"name": f"move j{j + 1}", "kind": "move", "start": a, "end": len(rows)})
    cmd7 = np.zeros((len(rows), 7))
    cmd7[:, :6] = np.array(rows)
    cmd7[:, 6] = SQ.GRIP_HOLD
    return cmd7, segs


def check_sequence(cmd7: np.ndarray, segs: list) -> dict:
    """模型里的安全与可辨识性检查（只看关节限位/速度/加速度/离台面高度；模型无桌上物体）。"""
    import mujoco
    from ideal_imu_kinematic import MODEL_PATH, full_qpos
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    r = SQ.check_limits(m, cmd7)
    r["min_z"] = float(min(SQ.lowest_geom_z(m, d, q) for q in cmd7[:, :6]))
    r["z_ok"] = r["min_z"] >= SQ.TABLE_Z + SQ.Z_CLEAR
    # 静止姿态下 IMU 安装点坐标系里的重力方向：三个方向都要看得到（条件数越小越好）
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    g = []
    for s in segs:
        if s["kind"] == "static":
            d.qpos[:] = full_qpos(cmd7[s["start"]])
            mujoco.mj_kinematics(m, d)
            g.append(d.site_xmat[sid].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]))
    sv = np.linalg.svd(np.array(g), compute_uv=False)
    r["grav_cond"] = float(sv[0] / sv[-1])
    r["duration_s"] = len(cmd7) / FPS
    return r


def view(cmd7: np.ndarray, speed: float = 1.0) -> None:
    """在 MuJoCo 窗口里按实时速度播放序列。模型的碰撞是关闭的，自碰撞和碰桌只能靠人眼看。"""
    import time
    import mujoco
    import mujoco.viewer
    from ideal_imu_kinematic import MODEL_PATH, full_qpos
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as v:
        for q in cmd7:
            if not v.is_running():
                break
            d.qpos[:] = full_qpos(q)
            mujoco.mj_kinematics(m, d)
            v.sync()
            time.sleep(1.0 / (FPS * speed))


def _skew(v: np.ndarray) -> np.ndarray:
    """(n,3) → (n,3,3)，满足 [v]× u = v × u。"""
    S = np.zeros(v.shape + (3,))
    S[:, 0, 1], S[:, 0, 2], S[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return S - S.transpose(0, 2, 1)


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """使 Σ|Q − R·P|² 最小的旋转 R（P、Q 为 (n,3)）；写法同 imu_frame_calib.calibrate。"""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    fix = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])     # 排除镜像
    return Vt.T @ fix @ U.T


def _deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def predict_site(tg_q: np.ndarray, q200: np.ndarray):
    """编码器关节角 → 模型安装点 ee_imu_site 的比力 f、角速度 ω（site 系）与杆臂矩阵 M = [α]× + [ω]×²。
    关节角先 10 Hz 低通再用样条求导：只为压住二阶导里的量化噪声，不影响 3 Hz 以下的比较带。"""
    import mujoco
    from scipy.interpolate import CubicSpline
    from ideal_imu_kinematic import MODEL_PATH, ideal_imu_traj
    m = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee_imu_site")
    q7 = np.column_stack([D.lp(q200, LP_DIFF_HZ, D.FS_Q), np.zeros(len(q200))])  # 夹爪不影响 link6 上的安装点
    imu = ideal_imu_traj(m, mujoco.MjData(m), CubicSpline(tg_q, q7), tg_q, sid)[1]
    w = imu[:, 3:]
    alpha = np.gradient(w, tg_q, axis=0)        # site 系里 dω/dt 就是角加速度（多出的一项 ω×ω = 0）
    return imu[:, :3], w, _skew(alpha) + _skew(w) @ _skew(w)


def _imu100(path, tg_i: np.ndarray) -> np.ndarray:
    """加计+陀螺 (n,6) 插到 `data.load` 的同一个 100 Hz 网格。设备 ms → monotonic 的最小二乘映射
    照抄 load（load 只返回陀螺）。"""
    I = np.asarray(json.loads(Path(path).read_text())["imu100"], float)
    ms, ht = I[:, 1], I[:, 0]
    a, c = np.linalg.lstsq(np.stack([ms, np.ones_like(ms)], 1), ht, rcond=None)[0]
    return np.stack([np.interp(tg_i, a * ms + c, I[:, k]) for k in range(2, 8)], 1)


def fit(rec_path, seq_path) -> dict:
    """真机回放记录 + 标定序列 → {R, b, gyro_bias, dp, delay_s, convention, diag}。"""
    dd = D.load(rec_path)
    seq = np.load(seq_path)
    cmd7, segs = seq["cmd7"], json.loads(str(seq["segs"]))
    if len(dd["cmd"]) != len(cmd7) or not np.allclose(dd["cmd"][:, :6], cmd7[:, :6], atol=1e-6):
        raise ValueError("回放记录与标定序列对不上：序列文件不对，或回放没有跑完")
    tg_i, tg_q, t_send = dd["tg_i"], dd["tg_q"], dd["t_send"]
    raw = _imu100(rec_path, tg_i)
    acc, gyr = D.lp(raw[:, :3], LP_HZ, D.FS_I), D.lp(raw[:, 3:], LP_HZ, D.FS_I)
    f, w, M = (D.lp(x, LP_HZ, D.FS_Q) for x in predict_site(tg_q, dd["q200"]))    # 两边同一个低通

    def at(x, tau):            # 预测量在「IMU 时刻 − τ」的值（IMU 记下的是 τ 秒前的运动）
        y = x.reshape(len(x), -1)
        return np.stack([np.interp(tg_i - tau, tg_q, y[:, k]) for k in range(y.shape[1])],
                        1).reshape((len(tg_i),) + x.shape[1:])

    ok = (tg_i - TAUS[-1] >= tg_q[0]) & (tg_i - TAUS[0] <= tg_q[-1])     # 任何 τ 下都有编码器数据
    # 名字带上起始时刻：首尾两个 home 同名，作字典键会互相覆盖
    wins = [(f"{s['name'].removeprefix('static ')}@{s['start'] / FPS:.0f}s",
             (tg_i >= t_send[s["fit_start"]]) & (tg_i <= t_send[s["end"] - 1]))
            for s in segs if s["kind"] == "static"]
    static = np.any([m_ for _, m_ in wins], axis=0)
    bg = raw[static, 3:].mean(0)            # 静止时真实角速度为 0 ⇒ 陀螺读数的均值就是零偏
    mv = ok & (np.linalg.norm(gyr - bg, axis=1) > GYRO_MIN)   # 按实测选转动样本 ⇒ 各 τ 比的是同一批

    def gyro_fit(tau):         # 陀螺不含重力和杆臂项 ⇒ 用它定延迟和朝向
        P = at(w, tau)[mv]
        R = _kabsch(P, gyr[mv] - bg)
        return float(np.mean((gyr[mv] - bg - P @ R.T) ** 2)), R

    i = int(np.argmin([gyro_fit(t)[0] for t in TAUS]))
    tau, R = float(TAUS[i]), gyro_fit(TAUS[i])[1]

    # 加计：acc − R·f = R·M·dp + b，对 dp、b 是线性的 ⇒ 一次最小二乘
    Ff = at(f, tau)
    F, RM = Ff[ok], R @ at(M, tau)[ok]
    X = np.concatenate([RM, np.broadcast_to(np.eye(3), RM.shape)], 2).reshape(-1, 6)
    y = (acc[ok] - F @ R.T).reshape(-1)
    dp, b = np.split(np.linalg.lstsq(X, y, rcond=None)[0], 2)

    # 交叉检验：只用静止姿态的重力方向（连同零偏）另解一次朝向，应与陀螺解出的 R 一致
    Fk = np.array([Ff[m_].mean(0) for _, m_ in wins])
    Ak = np.array([acc[m_].mean(0) for _, m_ in wins])
    R_grav = _kabsch(Fk - Fk.mean(0), Ak - Ak.mean(0))

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    P, Y = at(w, tau)[mv], y.reshape(-1, 3)
    return {"R": R.tolist(), "b": b.tolist(), "gyro_bias": bg.tolist(), "dp": dp.tolist(),
            "delay_s": tau, "convention": CONVENTION,
            "diag": {"rot_deg": _deg(R), "grav_vs_gyro_deg": _deg(R_grav @ R.T),
                     "delay_at_search_edge": i in (0, len(TAUS) - 1),
                     "gyro_rms_raw_fit": [rms(gyr[mv] - P), rms(gyr[mv] - bg - P @ R.T)],
                     "acc_rms_raw_nodp_fit": [rms(acc[ok] - F), rms(Y - Y.mean(0)),
                                              rms(y - X @ np.r_[dp, b])],
                     "static_resid": {n: float(np.linalg.norm(a - R @ fk - b))
                                      for (n, _), a, fk in zip(wins, Ak, Fk)},
                     "dp_cond": float(np.linalg.cond(X[:, :3])),
                     "n_moving": int(mv.sum()), "n_static": int(static.sum()), "lowpass_hz": LP_HZ}}


def main(argv=None) -> int:
    gc.disable()        # mujoco 与 scipy 同进程时 GC 会段错误（见 infra 记录）
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seq", help="生成标定动作序列")
    s.add_argument("--out", required=True)
    v = sub.add_parser("view", help="在 MuJoCo 窗口里目视检查序列")
    v.add_argument("--seq", required=True)
    v.add_argument("--speed", type=float, default=1.0)
    r = sub.add_parser("replay", help="经回放管线在真机执行序列（加 --dry-run 不连机械臂）")
    r.add_argument("--seq", default="outputs/imu_calib/calib_seq.npz")
    r.add_argument("--outdir", default="outputs/imu_calib")
    r.add_argument("--dry-run", action="store_true")
    f = sub.add_parser("fit", help="由真机回放记录解出 IMU 的朝向、零偏、安装位置、延迟")
    f.add_argument("--rec", default="outputs/imu_calib/replay_calib.json")
    f.add_argument("--seq", default="outputs/imu_calib/calib_seq.npz")
    f.add_argument("--out", default="outputs/imu_calib/imu_calib.json")
    a = ap.parse_args(argv)

    if a.cmd == "view":
        view(np.load(a.seq)["cmd7"], a.speed)
        return 0
    if a.cmd == "replay":
        # 与以前的上机入口（run_pilot_response14 等）相同：GC 已在上面关掉，再调回放管线。
        # 直接运行 replay_probe.py 不关 GC，读 json 时可能段错误（2026-10-14 dry-run 实测崩过 1 次）。
        import replay_probe
        return replay_probe.run("calib", a.dry_run, seqfile=a.seq, outdir=Path(a.outdir))
    if a.cmd == "fit":
        r = fit(a.rec, a.seq)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1))
        g = r["diag"]
        print(f"延迟 {r['delay_s'] * 1e3:.0f} ms" + ("  ⚠ 落在搜索边界" if g["delay_at_search_edge"] else ""))
        print(f"朝向修正 {g['rot_deg']:.2f}°（陀螺解）；只用静止重力另解的朝向与之相差 {g['grav_vs_gyro_deg']:.2f}°")
        print(f"陀螺零偏 {np.round(r['gyro_bias'], 4).tolist()} rad/s · 加计零偏 {np.round(r['b'], 3).tolist()} m/s²"
              f" · 安装位置偏差 {np.round(np.array(r['dp']) * 1e3, 1).tolist()} mm（条件数 {g['dp_cond']:.1f}）")
        print(f"陀螺 RMS {g['gyro_rms_raw_fit'][0]:.4f}→{g['gyro_rms_raw_fit'][1]:.4f} rad/s（转动样本）· "
              "加计 RMS {:.3f}→{:.3f}→{:.3f} m/s²（未标定→加 R、b→再加 dp）".format(*g["acc_rms_raw_nodp_fit"]))
        print("静止姿态残差 m/s²：" + "，".join(f"{k} {v:.3f}" for k, v in g["static_resid"].items()))
        print(f"→ {a.out}")
        return 0
    cmd7, segs = make_sequence()
    r = check_sequence(cmd7, segs)
    print(f"时长 {r['duration_s']:.1f} s · {len(cmd7)} 帧 · 静止 "
          f"{sum(x['kind'] == 'static' for x in segs)} 个 · 单关节转动 "
          f"{sum(x['kind'] == 'move' for x in segs)} 段")
    print(f"限位 {r['limit_ok']} · 速度 {r['vel_ok']} · 加速度 {r['acc_ok']} · "
          f"最低点离台面 {r['min_z'] * 100:.1f} cm（{r['z_ok']}）· 重力方向条件数 {r['grav_cond']:.2f}")
    print(f"关节范围（°）: {np.round(r['q_lo_deg'], 1).tolist()} … {np.round(r['q_hi_deg'], 1).tolist()}")
    if not (r["limit_ok"] and r["vel_ok"] and r["acc_ok"] and r["z_ok"]):
        print("✗ 未通过模型检查，不输出序列")
        return 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, cmd7=cmd7.astype(np.float32), da=np.zeros((len(cmd7), 6), np.float32),
             segs=json.dumps(segs, ensure_ascii=False))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)          # 跳过解释器收尾：关 GC 的进程在退出阶段也会崩（见 infra 记录）
