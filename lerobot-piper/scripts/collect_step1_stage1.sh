#!/usr/bin/env bash
# Step 1 阶段 1 采集: P0(5) / P1(8) / P2(6)  —— PREREG_STEP1.md §8
#
# 用法(在你终端, 先 conda activate lerobot):
#     bash scripts/collect_step1_stage1.sh              # 全跑
#     bash scripts/collect_step1_stage1.sh --preflight  # 只做开跑前自检, 不碰机器人
#     bash scripts/collect_step1_stage1.sh P1           # 只重开某一档(整档, 原 seed 表)
#
# 为什么是脚本而不是直接粘命令: `set -e` 粘进交互式终端后, 任何一条命令返回非零
# 都会让**你的终端自己退出**。放进脚本里, 出错只退出脚本, 终端还在, 且有日志可看。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
LOG_DIR="outputs/ab4/_logs"
mkdir -p outputs/ab4 outputs/step1_candidate "$LOG_DIR"

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
  *) echo "用法: $0 [all|P0|P1|P2|--preflight]"; exit 64 ;;
esac
