#!/usr/bin/env bash
# 弹性训练: 绕过本机"偶发启动期 SIGSEGV"(随机, 重试即可) + 任何中途崩溃从 last 自动续训。
# 顺序: insert 续训到 100K → pickplace2 从头训到 100K。全程 num_workers=0 (无 DataLoader worker → mid-run 稳)。
#
# 用法(在你终端, conda activate lerobot 后):
#   nohup bash scripts/train_resilient.sh > outputs/train/2026-07-01/resilient.log 2>&1 &
#   tail -f outputs/train/2026-07-01/resilient.log        # 看重试/接力节奏
#   tail -f outputs/train/2026-07-01/ACT_insert_noimu_seed42.log   # 看训练进度
set -u
cd ~/claude-test-2/lerobot-piper || exit 1
DATE=2026-07-01
NW="${NW:-0}"   # DataLoader workers. 0=最稳(无worker崩,但慢); 设 NW=2 提速(偶发worker崩由本脚本自动续训)
echo "[$(date)] num_workers=$NW"

run_until_100k () {
  local OUT="$1"; shift
  local LOG="${OUT}.log"
  local -a FRESH=("$@")
  local try
  for try in $(seq 1 80); do
    if [ -d "$OUT/checkpoints/100000" ]; then
      echo "[$(date)] OK $(basename "$OUT") 已达 100K"; return 0
    fi
    if [ -d "$OUT/checkpoints/last" ]; then
      local st; st=$(tr -d ' \n' < "$OUT/checkpoints/last/training_state/training_step.json" 2>/dev/null)
      echo "[$(date)] [$(basename "$OUT")] try#$try RESUME ($st)"
      PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py \
        --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true --num_workers=$NW --steps=100000 --log_freq=500 >> "$LOG" 2>&1
    else
      [ -d "$OUT" ] && rm -rf "$OUT"          # 清掉启动崩留下的半成品(无 checkpoint, 无损失)
      echo "[$(date)] [$(basename "$OUT")] try#$try FRESH"
      PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py "${FRESH[@]}" >> "$LOG" 2>&1
    fi
    echo "[$(date)] [$(basename "$OUT")] 退出码 $? — 3s 后检查/重试"
    sleep 3
  done
  echo "[$(date)] FAIL $(basename "$OUT") 试 80 次仍未到 100K"; return 1
}

echo "[$(date)] ===== 弹性训练开始 ====="
run_until_100k "outputs/train/$DATE/ACT_insert_noimu_seed42" || { echo "insert 未完成, 退出"; exit 1; }

run_until_100k "outputs/train/$DATE/ACT_pickplace2_noimu_seed42" \
  --dataset.repo_id=enolab-guo/pickplace2_train \
  --policy.type=act --policy.enable_imu_input=false \
  --batch_size=16 --steps=100000 --num_workers=$NW --use_amp=false --seed=42 \
  --save_freq=10000 --log_freq=500 \
  --output_dir="outputs/train/$DATE/ACT_pickplace2_noimu_seed42" \
  --job_name=ACT_pickplace2_noimu_seed42 \
  --wandb.enable=true --wandb.project=lerobot-piper-pickplace2 --wandb.disable_artifact=true \
  || { echo "pickplace2 未完成, 退出"; exit 1; }

echo "[$(date)] ===== 全部完成: insert + pickplace2 均到 100K ====="
