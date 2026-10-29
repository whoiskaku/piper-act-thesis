#!/usr/bin/env python3
"""**Optuna 参数同定** —— 24 个 MuJoCo 物理参数的自动搜索。

2026-08-26 面谈的唯一技术指示：
> 「どんなアルゴリズムを使って調整したんですか」→「人間が目で見て調整している」
> → **「そんなのさっさと Optuna か何かに任せましょうよ」**
> 补充理由（28:54）：**「相互作用とかあったらと思ってるんですけど」**

⇒ 关键不只是「自动化」，而是**24 个参数联合搜索、把参数间耦合算进去**。
此前的 `calibrate.py --grid` 是 `(kp×N, damping×N)` 二维**等比例**扫描，
本质仍是坐标扫描，正撞在老师给的理由上；本模块取代它。

## G 的定义（面谈澄清后的口径）

```
入力: ACT の関節位置命令列 → [G] → 出力: 関節軌道 + 手首 IMU ジャイロ
G   = パラメータ同定した Piper の MuJoCo モデル（ニューラルネットではない）
24 パラメータ = G の内部の同定対象（**入力でも出力でもない**）
Optuna = G の校正ツール（**G 本体ではない**）
```

## 同定对象（24 = 6 关节 × 4 类）

| 类 | MuJoCo 字段 | 现值 | 搜索范围（log 均匀） |
|---|---|---|---|
| `kp` | `actuator_gainprm[j,0]` / `biasprm[j,1]=-kp` | 500/200/200/100/50/50 | XML × [0.2, 20] |
| `damping` | `dof_damping[j]` | 500/5/20/5/5/5 | XML × [0.05, 20] |
| **`armature`** | `dof_armature[j]` | **全 0** | [1e-4, 1.0] kg·m² |
| **`frictionloss`** | `dof_frictionloss[j]` | **全 0** | [1e-3, 5.0] N·m |

后两类**从未被拟合过、且恒等于 0**，而我们有独立证据说明它们非零：
armature = 减速机换算惯量（j1 实测有效惯量与刚体模型差 115 倍量级）；
frictionloss = 静摩擦（δa≈0.7 mrad 的小指令在静止段被吃掉约 40%）。
**这是本条路线上真正没挖过的地方。**

> log 均匀下界不含 0，故「现模型」严格说不在搜索空间内；
> 但 armature 1e-4 kg·m²、frictionloss 1e-3 N·m 相对连杆惯量与 4.44 N·m
> 静态力矩可忽略，**等效于 0**。基线仍以真 0 单独评估。

## 预注册设计（**跑第一个 trial 之前定死，事后不得改**）

**数据切分**（90 条激励套件，`calibration` 角色；pp2 的 17 集**完全不参与**）:

| 集 | 条数 | 内容 | 用途 |
|---|---|---|---|
| 训练 | 62 | 位形 p0 + p1 | Optuna 优化 |
| **留出 A** | 25 | **位形 p2** | **位形泛化**：参数若是物理量，必须迁移到没见过的位形 |
| **留出 B** | 3 | `M_all_s*` 多正弦 | **多关节泛化**：逐关节同定的参数能否复合 |

留出集在优化过程中**一次都不评估**。

**目标函数** = 训练 62 条的**逐条相对残差的等权均值** `mean_i(‖bp(real−sim)‖² / ‖bp(real)‖²)`，
带通 0.5–9 Hz（与全项目同口径）。
> 用**等权**而非能量加权，是因为本项目已被单集主导坑过一次（pp2 的 `S1s07`
> 占 16.6% 能量、单集翻转结论）。能量加权值仍然记录，供与历史 0.5198 对照。

**固定不搜的部分**：固件 MOVE J 加速度限幅器（`apply_limiter(·,"a")`，
其 `a_j` 已由阶跃响应无模型定出并冻结）、`integrator=implicitfast`、
`timestep=0.002`、接触关闭、IMU 标定旋转 `R`。

**采样器**：TPE `multivariate=True`（联合建模参数耦合 ⇒ 直接回应「相互作用」），
`seed=20260826`。前两个 trial 用 `enqueue` 固定为现模型与 G-π-phys-v1。

**判否条件（预注册）**：
- **通过**：留出 A 等权均值 **≤ 0.35** 且优于现模型，**且**留出 B 同向改善；
- **否则**：判定**伺服/刚体参数层已到顶**，证据交给下一次面谈，
  路线转「命令→关节轨迹」层（A/B 分解已给出该层 +80.2%、17/17 的上限证据）。

用法:
    python sim/step2_g/optuna_ident.py --baseline
    python sim/step2_g/optuna_ident.py --run --trials 400 --workers 14
    python sim/step2_g/optuna_ident.py --report
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "sim", ROOT / "sim/step2_g"):
    sys.path.insert(0, str(p))

FPS = 20.0
SEED = 20260826
COL = ROOT / "outputs/step2_g/collect"
OUT = ROOT / "outputs/step2_g/optuna"
STUDY = "g_ident_24p"
PENALTY = 10.0                      # 发散/非有限 ⇒ 有限大惩罚(inf 会毒化 TPE)

XML_KP = np.array([500.0, 200.0, 200.0, 100.0, 50.0, 50.0])
XML_DAMP = np.array([500.0, 5.0, 20.0, 5.0, 5.0, 5.0])
BOUNDS = {"kp": (XML_KP * 0.2, XML_KP * 20.0),
          "damp": (XML_DAMP * 0.05, XML_DAMP * 20.0),
          "arm": (np.full(6, 1e-4), np.full(6, 1.0)),
          "fric": (np.full(6, 1e-3), np.full(6, 5.0))}
CLASSES = ("kp", "damp", "arm", "fric")

# 基线两档（真 0 的 armature/frictionloss，不走搜索空间下界）
BASE_XML = {"kp": XML_KP, "damp": XML_DAMP, "arm": np.zeros(6), "fric": np.zeros(6)}
BASE_PHYS_V1 = {**BASE_XML, "kp": np.array([5000.0, 200, 200, 100, 50, 50])}


# ── 数据 ────────────────────────────────────────────────────────────────
def split() -> tuple[list[str], list[str], list[str]]:
    done = [r["name"] for r in json.loads((COL / "state.json").read_text())["done"]]
    tr = sorted(n for n in done if re.search(r"_p[01](_|$)", n))
    ha = sorted(n for n in done if re.search(r"_p2(_|$)", n))
    hb = sorted(n for n in done if n.startswith("M_"))
    assert len(set(tr) | set(ha) | set(hb)) == len(done) == len(tr) + len(ha) + len(hb)
    return tr, ha, hb


def prep(names: list[str], limiter: str = "a") -> list[dict]:
    """限幅器与带通**与 24 参数无关** ⇒ 只算一次，进循环前缓存。"""
    from sim.step1_v2_loss import bandpass41
    from sim.step2_fidelity import apply_limiter
    out = []
    for nm in names:
        d = json.loads((COL / f"replay_{nm}.json").read_text())
        cmd = np.asarray(d["cmd"], float)
        real = np.asarray(d["imu"], float)[:, 3:6]
        n = min(len(cmd), len(real))
        a = bandpass41(real[:n])
        out.append({"name": nm, "act": apply_limiter(cmd, limiter), "n": n,
                    "abp": a, "ten": float(np.square(a).sum())})
    out.sort(key=lambda x: -x["n"])          # 长序列先跑, 便于并行填谷
    return out


# ── 前向 ────────────────────────────────────────────────────────────────
def set_params(m, p: dict) -> None:
    for j in range(6):
        kp = float(p["kp"][j])
        m.actuator_gainprm[j, 0] = kp
        m.actuator_biasprm[j, 1] = -kp
        m.dof_damping[j] = float(p["damp"][j])
        m.dof_armature[j] = float(p["arm"][j])
        m.dof_frictionloss[j] = float(p["fric"][j])


def roll(m, act: np.ndarray, T: int) -> np.ndarray:
    """→ 20 Hz 陀螺三轴。与 `calibrate.sim_gyro` 逐字同构（含夹爪初始化）。"""
    import mujoco
    from sim.step2_gap_baseline import aa_down
    dt = m.opt.timestep
    nsub = int(round(1 / FPS / dt))
    d = mujoco.MjData(m)
    g0 = float(np.clip(act[0, 6], 0, 0.035)) if act.shape[1] > 6 else 0.0
    d.qpos[:6] = act[0, :6]
    if m.nq > 7:
        d.qpos[6], d.qpos[7] = g0, -g0
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = act[0, :6]
    if m.nu > 7:
        d.ctrl[6], d.ctrl[7] = g0, -g0
    for _ in range(int(0.3 / dt)):
        mujoco.mj_step(m, d)
    hi = np.zeros((len(act) * nsub, 6))
    for t in range(len(act)):
        d.ctrl[:6] = act[t, :6]
        for k in range(nsub):
            mujoco.mj_step(m, d)
            hi[t * nsub + k] = d.sensordata[:6]
    return aa_down(hi, 1 / dt, T)[:, 3:6]


def evaluate(m, items, R, p) -> tuple[float, float, float, np.ndarray]:
    """→ (等权均值, 中位, 能量加权, 逐条相对残差)。发散 ⇒ 全 PENALTY。"""
    from sim.step1_v2_loss import bandpass41
    set_params(m, p)
    per, num, den = [], 0.0, 0.0
    for it in items:
        try:
            sim = roll(m, it["act"], it["n"])
        except Exception:                                    # noqa: BLE001
            return PENALTY, PENALTY, PENALTY, np.full(len(items), PENALTY)
        k = min(len(sim), it["n"])
        b = bandpass41(sim[:k] @ R.T)
        if not np.isfinite(b).all():
            return PENALTY, PENALTY, PENALTY, np.full(len(items), PENALTY)
        e = float(np.square(it["abp"][:k] - b).sum())
        per.append(min(e / max(it["ten"], 1e-12), PENALTY))
        num += e
        den += it["ten"]
    per = np.array(per)
    return float(per.mean()), float(np.median(per)), float(num / max(den, 1e-12)), per


# ── Optuna ──────────────────────────────────────────────────────────────
def suggest(trial) -> dict:
    p = {}
    for c in CLASSES:
        lo, hi = BOUNDS[c]
        p[c] = np.array([trial.suggest_float(f"{c}{j+1}", float(lo[j]), float(hi[j]),
                                             log=True) for j in range(6)])
    return p


def flat(p: dict) -> dict:
    """参数字典 → Optuna 的扁平命名（用于 enqueue_trial）。"""
    return {f"{c}{j+1}": float(np.clip(p[c][j], *[b[j] for b in BOUNDS[c]]))
            for c in CLASSES for j in range(6)}


def storage():
    import optuna
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        from optuna.storages import JournalStorage
        from optuna.storages.journal import JournalFileBackend
        return JournalStorage(JournalFileBackend(str(OUT / "journal.log")))
    except Exception:                                        # noqa: BLE001
        return f"sqlite:///{OUT/'ident.db'}"


def make_study(create: bool, seed: int = SEED):
    """`seed` **必须逐 worker 不同**：spawn 出来的进程若共用一个 seed，
    TPE 启动期(随机采样)会逐字生成同一组参数 —— 并行预算全被复制品吃掉。
    (已实测：worker 0/1 的 trial 2 与 3 值完全相同 0.6970535830799548。)"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(
        study_name=STUDY, storage=storage(), direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True,
                                           group=True, n_startup_trials=40))
    if create and not s.trials:
        s.enqueue_trial(flat(BASE_XML))          # trial 0 = 现 XML 模型
        s.enqueue_trial(flat(BASE_PHYS_V1))      # trial 1 = G-π-phys-v1
    return s


def _worker(rank: int, n_trials: int) -> None:
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    R, _ = load_calib(CALIB)
    items = prep(split()[0])
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    st = make_study(create=False, seed=SEED + 1009 * rank)
    t0 = time.time()

    def obj(trial):
        mu, md, en, _ = evaluate(m, items, R, suggest(trial))
        trial.set_user_attr("median", md)
        trial.set_user_attr("energy", en)
        return mu

    def cb(study, trial):
        if rank == 0:
            b = study.best_trial
            print(f"  [{len(study.trials):4d}] 本 trial {trial.value:.4f}   "
                  f"最优 {b.value:.4f} (能量加权 {b.user_attrs.get('energy', float('nan')):.4f})"
                  f"   {time.time()-t0:.0f}s", flush=True)

    st.optimize(obj, n_trials=n_trials, callbacks=[cb], gc_after_trial=True)


def cmd_run(a) -> int:
    import multiprocessing as mp
    make_study(create=True)
    per = int(np.ceil(a.trials / a.workers))
    print(f"**Optuna 24 参数同定** —— TPE(multivariate) seed={SEED}")
    print(f"  训练 {len(split()[0])} 条 · 留出 A(p2) {len(split()[1])} · "
          f"留出 B(multisine) {len(split()[2])}  ← 留出集本阶段**不评估**")
    print(f"  {a.workers} 进程 × {per} trial = {a.workers*per}\n")
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_worker, args=(i, per)) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return cmd_report(a)


def _fmt(p: dict) -> str:
    return "\n".join(
        f"    {c:5s} " + " ".join(f"{v:9.4g}" for v in p[c]) for c in CLASSES)


def cmd_baseline(a) -> int:
    import mujoco
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    R, _ = load_calib(CALIB)
    tr, ha, hb = split()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    sets = [("训练 p0+p1", prep(tr)), ("留出A p2", prep(ha)), ("留出B 多正弦", prep(hb))]
    print(f"**基线**（`armature`/`frictionloss` 真 0，不走搜索下界）\n")
    print(f"  {'配置':16s}{'数据集':14s}{'等权均值':>10s}{'中位':>9s}{'能量加权':>10s}")
    for nm, p in (("现 XML 模型", BASE_XML), ("G-π-phys-v1", BASE_PHYS_V1)):
        for sn, items in sets:
            mu, md, en, _ = evaluate(m, items, R, p)
            print(f"  {nm:16s}{sn:14s}{mu:>10.4f}{md:>9.4f}{en:>10.4f}", flush=True)
    print("\n  (1.0 = 「预测全零」的水平；<1 才算解释了东西)")
    return 0


def cmd_report(a) -> int:
    import mujoco, optuna
    from sim.step1_v2_loss import load_calib
    from sim.step2_fidelity import CALIB
    from sim.step2_gap_baseline import MODEL
    st = make_study(create=False)
    done = [t for t in st.trials if t.value is not None]
    print(f"\n{'='*78}\n**结果** —— {len(done)} 个完成的 trial")
    b = st.best_trial
    best = {c: np.array([b.params[f"{c}{j+1}"] for j in range(6)]) for c in CLASSES}
    print(f"\n  最优 trial #{b.number}  训练等权均值 **{b.value:.4f}**"
          f"  (中位 {b.user_attrs.get('median', float('nan')):.4f} · "
          f"能量加权 {b.user_attrs.get('energy', float('nan')):.4f})")
    print(f"\n  最优参数:\n{_fmt(best)}")
    print(f"\n  相对 XML 原值:  kp × " +
          " ".join(f"{v:.2f}" for v in best['kp'] / XML_KP) +
          "\n                  damp × " +
          " ".join(f"{v:.2f}" for v in best['damp'] / XML_DAMP))

    R, _ = load_calib(CALIB)
    tr, ha, hb = split()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    print(f"\n  {'数据集':16s}{'现 XML':>10s}{'phys-v1':>10s}{'**最优**':>12s}"
          f"{'改善':>9s}")
    verdict = {}
    for sn, names in (("训练 p0+p1", tr), ("**留出A p2**", ha), ("**留出B 多正弦**", hb)):
        items = prep(names)
        v = [evaluate(m, items, R, p)[0] for p in (BASE_XML, BASE_PHYS_V1, best)]
        verdict[sn] = v
        print(f"  {sn:16s}{v[0]:>10.4f}{v[1]:>10.4f}{v[2]:>12.4f}"
              f"{(1-v[2]/v[0])*100:>8.1f}%", flush=True)

    if len(done) >= 20:
        try:
            imp = optuna.importance.get_param_importances(
                st, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
            agg = {c: sum(v for k, v in imp.items() if k.startswith(c)) for c in CLASSES}
            print(f"\n  **参数重要度**（fANOVA，回应「相互作用」质疑）")
            for c, v in sorted(agg.items(), key=lambda x: -x[1]):
                print(f"    {c:6s} {v*100:5.1f}%")
            print("    逐参数 top6: " + ", ".join(
                f"{k} {v*100:.1f}%" for k, v in
                sorted(imp.items(), key=lambda x: -x[1])[:6]))
        except Exception as e:                               # noqa: BLE001
            print(f"\n  (重要度计算失败: {e})")

    ha_v = verdict["**留出A p2**"]
    hb_v = verdict["**留出B 多正弦**"]
    ok = ha_v[2] <= 0.35 and ha_v[2] < ha_v[0] and hb_v[2] < hb_v[0]
    print(f"\n{'='*78}\n**预注册判定**: 留出A ≤ 0.35 且优于现模型，且留出B 同向")
    print("⇒ " + ("**通过** —— 冻结为新的 G 物理底座" if ok else
                  "**未通过** —— 判定伺服/刚体参数层已到顶，转命令→关节轨迹层"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "best_params.json").write_text(json.dumps(
        {"trial": b.number, "train_mean": b.value,
         "params": {c: [float(x) for x in best[c]] for c in CLASSES},
         "splits": {k: [float(x) for x in v] for k, v in verdict.items()},
         "pass": bool(ok), "n_trials": len(done)}, ensure_ascii=False, indent=2))
    print(f"\n  → {OUT/'best_params.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    if a.baseline:
        return cmd_baseline(a)
    if a.run:
        return cmd_run(a)
    if a.report:
        return cmd_report(a)
    print(__doc__.split("用法:")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
