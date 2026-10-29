#!/usr/bin/env bash
# Step 1 阶段 1 采集: P0(5) / P1(8) / P2(6)  —— PREREG_STEP1.md §8
#
# 用法(在你终端, 先 conda activate lerobot):
#     bash scripts/collect_step1_stage1.sh              # 全跑
#     bash scripts/collect_step1_stage1.sh --preflight  # 只做开跑前自检, 不碰机器人
#     bash scripts/collect_step1_stage1.sh P1           # 只重开某一档(整档, 原 seed 表)
#     bash scripts/collect_step1_stage1.sh P3           # 阶段 2: P2 闸门 PASS 后才跑
#
# 为什么是脚本而不是直接粘命令: `set -e` 粘进交互式终端后, 任何一条命令返回非零
# 都会让**你的终端自己退出**。放进脚本里, 出错只退出脚本, 终端还在, 且有日志可看。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG_DIR="outputs/ab4/_logs"
mkdir -p outputs/ab4 outputs/step1_candidate "$LOG_DIR"

# ── libstdc++ 冲突修正 ──────────────────────────
# 症状: control_robot 里 `import cv2` 报
#   /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
# 成因: ROS noetic 的 setup 把 /usr/lib/x86_64-linux-gnu 放在 LD_LIBRARY_PATH 最前,
#   lerobot 的依赖先把**系统**旧 libstdc++(只到 CXXABI_1.3.12)载入进程; 之后 cv2
#   再要 libstdc++ 时复用已加载的那个 ⇒ 缺符号。单独 `import cv2` 反而没事,
#   因为那时系统库还没被载进来。
# 修法: 预加载 conda env 自带的 libstdc++(CXXABI_1.3.15), 让它先占位。
#   新版 libstdc++ 向后兼容, 已实测 control_robot 全链 + piper_sdk 均正常。
#   只动 libstdc++, 不改 LD_LIBRARY_PATH ⇒ 不遮蔽 ROS 需要的其他系统库。
_CONDA_LIBSTDCXX="${CONDA_PREFIX:-}/lib/libstdc++.so.6"
if [ -e "$_CONDA_LIBSTDCXX" ]; then
  export LD_PRELOAD="${_CONDA_LIBSTDCXX}${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "  [env] LD_PRELOAD=$_CONDA_LIBSTDCXX"
else
  echo "  [env] ⚠ 找不到 $_CONDA_LIBSTDCXX —— 没 conda activate lerobot?"
fi

POLICY=outputs/train/2026-07-01/ACT_pickplace2_noimu_seed42/checkpoints/100000/pretrained_model
CALIB=outputs/p2_labels/pp2a-deploy/manifest.json
PROBE_CKPT=outputs/p2_pi/pi_direct_bare_s0.pt

# ── 开跑前自检(不碰机器人) ────────────────────────────────────────────────
preflight () {
  echo "=== 开跑前自检 ==="
  local ok=0
  for f in "$POLICY" "$CALIB" "$PROBE_CKPT"; do
    if [ -e "$f" ]; then echo "  ✓ $f"; else echo "  ✗ 缺失: $f"; ok=1; fi
  done
  echo "  python: $(command -v python)"
  python - <<'PY' || ok=1
import sys
try:
    import torch, numpy, scipy, draccus            # noqa: F401
    from lerobot.common.policies.act.b_correction_deploy import MODES
    from lerobot.common.policies.act.step1_correction_runtime import EXPECTED_HP_ALPHA
except Exception as exc:                            # noqa: BLE001
    print(f"  ✗ import 失败: {type(exc).__name__}: {exc}")
    print("     → 多半是没 conda activate lerobot")
    sys.exit(1)
print(f"  ✓ import OK; MODES={MODES}")
assert "probe" in MODES, "probe 模式不在 MODES 里 —— 代码版本不对"
PY
  # **走真实的 control_robot import 链** —— cv2/libstdc++ 冲突只在这条链上暴露,
  # 单独 import cv2 是好的, 所以自检必须按真实顺序走一遍。
  python - <<'PYCHAIN' || ok=1
import sys
try:
    import lerobot.scripts.control_robot            # noqa: F401
    import cv2                                      # noqa: F401
except Exception as exc:                             # noqa: BLE001
    print(f"  ✗ control_robot 导入链失败: {type(exc).__name__}: {exc}")
    if "CXXABI" in str(exc):
        print("     → libstdc++ 冲突; 本脚本已设 LD_PRELOAD, 若仍失败请检查 CONDA_PREFIX")
    sys.exit(1)
print("  ✓ control_robot 全链 import OK (含 cv2)")
PYCHAIN

  # seed 表能否正确解析(纯离线, 不建机器人)
  python - <<'PY' || ok=1
import importlib.util, sys
from pathlib import Path
p = Path("lerobot/common/policies/act/b_correction_deploy.py")
spec = importlib.util.spec_from_file_location("_bc", p)
m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(m)
except Exception:
    sys.path.insert(0, ".")
    from lerobot.common.policies.act import b_correction_deploy as m
for tier, spec_s, n in (("P1", "1000,1001,1002,1003,1004,1005,1006,1007", 8),
                        ("P2", "2000,2000,2001,2001,2002,2002", 6)):
    seeds = m.parse_seed_schedule(spec_s, 0)
    hashes = [m.probe_hash(s) for s in seeds]
    assert len(seeds) == n, f"{tier} seed 数 {len(seeds)} ≠ {n}"
    print(f"  ✓ {tier}: {len(seeds)} seeds, {len(set(hashes))} 种波形 "
          f"{'(P2 应为 3 种)' if tier=='P2' else '(P1 应为 8 种)'}")
PY
  [ "$ok" -eq 0 ] && echo "=== 自检通过, 可以开跑 ===" || { echo "=== 自检失败, 先修上面的问题 ==="; return 1; }
}

# ── 单档采集 ──────────────────────────────────────────────────────────────
run_probe () {
  local tier="$1" mode="$2" mrad="$3" seeds="$4" episodes="$5"
  local log="$LOG_DIR/${tier}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "════ $tier : mode=$mode  mrad=$mrad  集数=$episodes ════"
  echo "  seeds: $seeds"
  echo "  终端输出同时写入: $log"
  echo

  # 已有同名日志 ⇒ 提示走整档重开的流程, 不静默覆盖
  if compgen -G "outputs/ab4/${tier}_ep*.json" > /dev/null; then
    echo "  ⚠ outputs/ab4/${tier}_ep*.json 已存在。"
    echo "    预注册禁止单集 rerecord 补采; 要重来请**整档**换名重开, 例如"
    echo "      mv outputs/ab4/${tier}_ep*.json outputs/ab4/_discarded_${tier}_$(date +%H%M%S)/"
    read -r -p "    仍要继续并覆盖? [y/N] " a
    [ "$a" = "y" ] || { echo "    已跳过 $tier"; return 0; }
  fi

  BCORR_CKPT="$PROBE_CKPT" \
  BCORR_CALIB="$CALIB" \
  BCORR_MODE="$mode" \
  BCORR_PROBE_MRAD="$mrad" \
  BCORR_PROBE_SEEDS="$seeds" \
  BCORR_LOG="outputs/ab4/${tier}.json" \
  python lerobot/scripts/control_robot.py \
    --robot.type=piper --robot.inference_time=true --robot.enable_imu=true \
    --control.type=record --control.fps=20 --control.single_task=pp2 \
    --control.warmup_time_s=2 --control.episode_time_s=30 --control.reset_time_s=10 \
    --control.push_to_hub=false --control.policy.path="$POLICY" \
    --control.repo_id="$USER/eval_ab4_${tier}" \
    --control.num_episodes="$episodes" 2>&1 | tee "$log"

  echo
  echo "──── $tier 采完, 自动核对 ────"
  python scripts/check_probe_logs.py "$tier" || {
    echo "  ⚠ 核对未通过 —— 先看上面的项, 不要接着采下一档"; return 1; }
}

# ── 入口 ──────────────────────────────────────────────────────────────────
case "${1:-all}" in
  --preflight) preflight; exit $? ;;
  P0) preflight && run_probe P0 off   1.5 0                                       5 ;;
  P1) preflight && run_probe P1 probe 1.5 1000,1001,1002,1003,1004,1005,1006,1007 8 ;;
  P2) preflight && run_probe P2 probe 1.5 2000,2000,2001,2001,2002,2002           6 ;;
  # P3 只有在 --stage p2 明确 PASS 之后才该跑(幅值线性检验, 复用 P1 前 4 个 seed)
  P3) preflight && run_probe P3 probe 3.0 1000,1001,1002,1003                     4 ;;
  all)
    preflight
    run_probe P0 off   1.5 0                                       5
    run_probe P1 probe 1.5 1000,1001,1002,1003,1004,1005,1006,1007 8
    run_probe P2 probe 1.5 2000,2000,2001,2001,2002,2002           6
    echo
    echo "════════════════════════════════════════════════════════"
    echo "P0/P1/P2 采完。**先不要采 P3。** 下一步跑 P2 跨 seed 闸门:"
    echo
    echo "  python sim/fit_step1_candidate.py \\"
    echo "      --data-root outputs/ab4 --calib $CALIB \\"
    echo "      --output outputs/step1_candidate/step1_linear_v1.npz --stage p2"
    echo
    echo "PASS 才采 P3; FAIL(退出码 2) 则整条路径作废。"
    echo "════════════════════════════════════════════════════════"
    ;;
  *) echo "用法: $0 [all|P0|P1|P2|P3|--preflight]"; exit 64 ;;
esac
