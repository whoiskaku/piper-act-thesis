#!/usr/bin/env python3
"""C 受控工作点采集 —— `replay_probe.py` 的**薄封装**（c-pilot-v1，2026-07-29）。

**不复制**已上机验证过的逻辑：连接、IMU 自检、预移动、100 Hz 记录、发送、
安全退出全部调用 `replay_probe.run()`。本脚本只新增四件事：

  1. 受控位姿 + 慢速往复 + **统一进场**（把最后进场放进**被记录的序列里**，可测）
  2. C 序列生成（探针叠在冻结窗口）
  3. **fail-closed 身份校验**（tag / manifest / train-only / 位姿表 / 序列 hash）
  4. **12 个 schema 字段**后处理落盘 + 完整实际发送序列

用法:
  python sim/replay_probe_C.py --freeze                # 生成/重冻 manifest
  python sim/replay_probe_C.py --pose 14 --joint 2 --arm A --dry-run
  python sim/replay_probe_C.py --pose 14 --joint 2 --arm Bp
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import traceback
from pathlib import Path

import numpy as np

import select_poses_C as S
import make_probe_b2 as MB
import replay_probe as RP

ROOT = Path(__file__).resolve().parents[1]
PC = ROOT / "outputs/probe_c"
MANIFEST = PC / "manifest_c_pilot.json"
FPS = 20.0
F_WIDE = np.arange(2.0, 8.01, 0.5)
F_NARROW = np.array([6.0, 6.5, 7.0, 7.5])
N_APPR, N_STAB, N_TAIL = 40, 40, 40          # 进场 2s / 稳定 2s / 尾段 2s
APPROACH_OFF = np.radians(1.5)               # 统一进场的偏置量
SAFE_RESERVE = np.radians(0.3)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def h16(x):
    return hashlib.sha256(np.ascontiguousarray(x, np.float64).tobytes()).hexdigest()[:16]


def pose_table_hash(tr, ho, cands):
    """对**规范化后的完整 train+holdout 表**求 hash，不只哈希当前位姿。"""
    rows = []
    # 集合标记必须是**固定常量** —— Python 内置 hash() 对字符串逐进程随机化,
    # 用它会让身份 hash 每次运行都不同(2026-07-29: 曾使每次 verify 都失败)。
    for code, idx in ((0, sorted(tr)), (1, sorted(ho))):
        for i in idx:
            c = cands[i]
            rows.append([code, i, round(float(c["q2"]), 9),
                         round(float(c["q3"]), 9), round(float(c["grip"]), 9),
                         round(float(c["dither"]), 9), round(float(c.get("hz", 0.4)), 9)])
    return h16(np.array(rows, float))


# ======================= 序列构建 =======================

def approach_dir_of(pose, lo, hi):
    """统一进场方向：**从余量更大的一侧**进场（逐关节，+1 / −1）。

    固定符号会在贴近可达边界的位姿上把余量吃光（p15 的 q3 只剩 0.33°），
    所以方向按余量选，但对同一位姿是**确定且落盘**的。
    """
    d = {}
    for jj, q in ((1, pose["q2"]), (2, pose["q3"])):
        room_lo, room_hi = q - lo[jj], hi[jj] - q
        d[jj] = +1 if room_hi >= room_lo else -1     # +1 = 从上方降下来
    return d


def build_sequence(pose, joint, freqs, amp, sign):
    """返回 (cmd7, da, meta)。

    帧布局: [0,N_APPR) 统一进场(dither 幅值同步 0→满)
            [N_APPR, w0) 稳定振荡
            [w0, w0+NEXC) **探针窗**
            尾段衰减
    进场段被**记录在序列内**, 故 approach_dir 可由编码器实测复核。
    """
    from make_probe_b2 import NEXC
    lo, hi = MB.reach_limits()
    adir = approach_dir_of(pose, lo, hi)
    w0 = N_APPR + N_STAB
    Sn = w0 + NEXC + N_TAIL
    t = np.arange(Sn) / FPS
    hz = float(pose.get("hz", 0.4))
    ph0 = 0.0                                        # 冻结的 dither 初相
    ramp = np.ones(Sn)
    ramp[:N_APPR] = (1 - np.cos(np.pi * np.arange(N_APPR) / N_APPR)) / 2

    cmd7 = np.zeros((Sn, 7), np.float32)
    base = {1: float(pose["q2"]), 2: float(pose["q3"])}
    for jj in (1, 2):
        off = adir[jj] * APPROACH_OFF * (1.0 - ramp)      # 进场偏置 → 0
        dith = pose["dither"] * np.sin(2 * np.pi * hz * t + ph0 + (1.0 if jj == 2 else 0.0))
        cmd7[:, jj] = base[jj] + off + ramp * dith
    cmd7[:, 6] = pose["grip"]

    da = np.zeros((Sn, 6), np.float32)
    wv, _, _ = MB.schroeder(np.asarray(freqs, float), NEXC)
    da[w0:w0 + NEXC, joint] = sign * wv * amp
    meta = dict(w0=int(w0), nexc=int(NEXC), n_appr=N_APPR, n_stab=N_STAB,
                hz=hz, dither_rad=float(pose["dither"]), phase0_rad=float(ph0),
                approach_off_rad=float(APPROACH_OFF),
                approach_dir_planned={f"j{k+1}": int(v) for k, v in adir.items()})
    return cmd7, da, meta


def gate0_on_sequence(cmd7, da, p99_deg):
    """对**实际序列**重算 ⓪（fail-closed）。不是对设计值，是对要发出去的数组。"""
    lo, hi = MB.reach_limits()
    p99 = np.radians(p99_deg)
    q = cmd7[:, :6] + da
    out = {}
    for jj in (1, 2):
        m = min((q[:, jj] - lo[jj]).min(), (hi[jj] - q[:, jj]).min())
        out[f"j{jj+1}"] = float(np.degrees(m - p99[jj] - SAFE_RESERVE))
    return out


# ======================= 身份校验（fail-closed） =======================

def freeze():
    tr, ho, cands = S.train_and_holdout(16, 5)
    S.assert_train_only(tr, ho, where="freeze")
    man = {
        "git_hash": sh("git", "rev-parse", "HEAD"),
        "git_tag": sh("git", "describe", "--tags", "--exact-match") or "UNTAGGED",
        "pose_table_hash": pose_table_hash(tr, ho, cands),
        "train": sorted(tr), "holdout_sealed": sorted(ho),
        "seed_order": 20260729, "seed_sign": 731,
        "approach_off_rad": float(APPROACH_OFF),
        "n_appr": N_APPR, "n_stab": N_STAB, "n_tail": N_TAIL,
        "units": {"u_peak_per_joint": "rad", "seq_energy": "rad^2*s",
                  "approach_dir": "+1=从上方进场 / -1=从下方",
                  "dither_phase0": "rad, 命令相位(非实测)"},
    }
    PC.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"→ 冻结 manifest: {MANIFEST}")
    for k in ("git_hash", "git_tag", "pose_table_hash", "seed_order", "seed_sign"):
        print(f"   {k} = {man[k]}")
    return man


def verify_identity(pose_idx):
    """fail-closed：任一不符**直接退出**，不上机。"""
    if not MANIFEST.exists():
        raise SystemExit("✗ 无 manifest —— 先跑 --freeze")
    man = json.loads(MANIFEST.read_text())
    head = sh("git", "rev-parse", "HEAD")
    tag = sh("git", "describe", "--tags", "--exact-match") or "UNTAGGED"
    dirty = bool(sh("git", "status", "--porcelain"))
    tr, ho, cands = S.train_and_holdout(16, 5)
    S.assert_train_only(tr, ho, where="verify_identity")
    pth = pose_table_hash(tr, ho, cands)
    bad = []
    if head != man["git_hash"]:
        bad.append(f"HEAD {head[:8]} ≠ manifest {man['git_hash'][:8]}")
    if tag != man["git_tag"]:
        bad.append(f"tag {tag} ≠ manifest {man['git_tag']}")
    if dirty:
        bad.append("工作区不干净")
    if pth != man["pose_table_hash"]:
        bad.append(f"位姿表 hash {pth} ≠ {man['pose_table_hash']}")
    if sorted(tr) != man["train"] or sorted(ho) != man["holdout_sealed"]:
        bad.append("train/holdout 集合与 manifest 不一致")
    if pose_idx not in man["train"]:
        bad.append(f"位姿 {pose_idx} 不在 train 集合中")
    if pose_idx in man["holdout_sealed"]:
        bad.append(f"**位姿 {pose_idx} 属于 holdout 封存集合**")
    if bad:
        for b in bad:
            print(f"  ✗ {b}")
        raise SystemExit("✗ 身份校验失败 —— 不上机")
    print(f"  ✓ 身份校验通过 (HEAD {head[:8]}, tag {tag}, 位姿表 {pth})")
    return man, tr, ho, cands


# ======================= 主流程 =======================

def run_c(pose_idx, joint, arm, dry_run=False, rep=1):
    man, tr, ho, cands = verify_identity(pose_idx)
    # pilot 索引是 **tr 的顺序位置**(greedy 选择顺序), 与 gate0_C 的 `poses` 同序。
    # 用 sorted(tr) 会取到别的位姿(2026-07-29: 曾使 rm 从 0.33° 变成 12.2°)。
    if pose_idx not in tr:
        raise SystemExit(f"✗ 位姿 {pose_idx} 不在 train 集合")
    k = list(tr).index(pose_idx)
    pose = cands[pose_idx]
    b = json.loads((PC / "gate0_C_budget.json").read_text())
    anch = {b["pilot"]["P1"], b["pilot"]["P2"]}
    F = F_WIDE if k in anch else F_NARROW
    import gate0_C as G0
    E_star = min(G0._seq_energy(F_WIDE), G0._seq_energy(F_NARROW))
    amp = 0.002 * np.sqrt(E_star / G0._seq_energy(F))
    sign = {"A": 0, "Bp": +1, "Bm": -1}[arm]

    cmd7, da, meta = build_sequence(pose, joint, F, amp, sign if sign else 1)
    if arm == "A":
        da = np.zeros_like(da)                       # A 臂: 探针全零
    rm = gate0_on_sequence(cmd7, da, b["p99_quasistatic_deg"])
    print(f"  ⓪(对实际序列重算) {rm}")
    if min(rm.values()) <= 0:
        raise SystemExit(f"✗ ⓪ 不过: {rm} —— 不上机")

    seqname = f"c_p{pose_idx}_j{joint+1}_{arm}.npz"
    (RP.PROBE).mkdir(parents=True, exist_ok=True)
    np.savez(RP.PROBE / seqname, cmd7=cmd7, da=da,
             wins=np.array([meta["w0"]]), freqs=np.asarray(F, float))
    seq_hash = h16(np.concatenate([cmd7.ravel(), da.ravel()]))
    print(f"  序列 {seqname}  hash {seq_hash}  w0={meta['w0']} 帧数={len(cmd7)}")

    # rep 后缀: 臂序是 A → B+ → B− → A, 两条 A 若同名会**覆盖前一条**
    # (B′ 当初加轮次后缀正是为此)。rep 进入文件名与记录。
    armname = f"{arm}{rep}_c{pose_idx}_{joint+1}"
    tag = f"c_p{pose_idx}_j{joint+1}_"
    aborted, reason = False, None
    try:
        rc = RP.run(armname, dry_run=dry_run, seqfile=seqname, tag=tag, outdir=PC)
    except BaseException as e:                        # 含 KeyboardInterrupt
        aborted, reason, rc = True, f"{type(e).__name__}: {e}", 1
        print(f"\n✗ 异常: {reason}")
        traceback.print_exc()
    if dry_run:
        return rc

    # ---- schema 后处理: 12 个字段 + 完整实际发送序列 ----
    p = PC / f"replay_{tag}{armname}.json"
    if not p.exists():
        # 异常发生在 replay_probe 写盘**之前**(如连接阶段) ⇒ 这里必须自己落盘,
        # 否则"任意异常都要留下 aborted=true/reason 的部分日志"这条就落空了。
        print(f"  ⚠ replay_probe 未落盘 ⇒ 写最小 abort 日志")
        rec = {"arm": armname, "t": [], "t_send": [], "imu": [], "state": [],
               "imu100": [], "stopped": reason or "未知(无记录)"}
    else:
        rec = json.loads(p.read_text())
    # abort 路径可能是**零帧**记录 ⇒ 不能无条件按 2 维索引
    _st = np.asarray(rec.get("state") or [], float)
    st = _st[:, :6] * np.pi / 180 / 1000 if _st.ndim == 2 and _st.shape[0] else np.zeros((0, 6))
    adir_meas = {}
    if len(st) > N_APPR:
        for jj in (1, 2):
            d = st[N_APPR - 1, jj] - st[0, jj]
            adir_meas[f"j{jj+1}"] = int(np.sign(d)) if abs(d) > 1e-6 else 0
    rec.update({
        "git_hash": man["git_hash"], "git_tag": man["git_tag"],
        "pose_table_hash": man["pose_table_hash"],
        "seed_order": man["seed_order"], "seed_sign": man["seed_sign"],
        "session_id": sh("date", "+%Y%m%d"), "session_index": None,   # 由采集编排填
        "injected_joint": int(joint) if arm != "A" else None,
        "u_peak_per_joint": np.abs(da).max(0).tolist(),               # rad
        "seq_energy": float(np.sum(da ** 2) / FPS),                   # rad^2*s
        "approach_dir": {"planned": meta["approach_dir_planned"],
                         "measured_from_encoder": adir_meas},
        "dither_phase0": {"cmd_phase_rad": meta["phase0_rad"],
                          "ref_frame": 0,
                          "ref_t_send": (rec["t_send"][0] if rec.get("t_send") else None),
                          "note": "命令相位; **实测相位须由分析器拟合后另存**"},
        "seq_hash": seq_hash, "seq_meta": meta, "rep": int(rep),
        "sent_cmd7": cmd7.tolist(), "sent_da": da.tolist(),           # 完整实际发送序列
        "gate0_rm_deg": rm, "aborted": bool(aborted or rec.get("stopped")),
        "abort_reason": reason or rec.get("stopped"),
    })
    p.write_text(json.dumps(rec))
    print(f"  ✓ schema 字段已补齐 → {p}")
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="生成/重冻 manifest")
    ap.add_argument("--pose", type=int, help="train 位姿索引")
    ap.add_argument("--joint", type=int, default=2, choices=[1, 2], help="1=j2, 2=j3")
    ap.add_argument("--arm", default="A", choices=["A", "Bp", "Bm"])
    ap.add_argument("--rep", type=int, default=1,
                    help="同一臂型的第几条(臂序 A1→Bp1→Bm1→A2 的两条 A 必须不同)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze(); raise SystemExit(0)
    if a.pose is None:
        raise SystemExit("需要 --pose")
    raise SystemExit(run_c(a.pose, a.joint, a.arm, a.dry_run, a.rep))
