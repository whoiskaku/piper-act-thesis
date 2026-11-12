#!/usr/bin/env bash
# 新桌面两套示教的 ACT 训练：pickcube_raw(36 条) → insert39(39 条)，各 100K 步，依次跑。
# 配置同旧桌面 pickplace2（batch 16、关 AMP、seed 42、不用 IMU 输入、每 10K 存档、开 wandb）。
# 弹性部分照抄 train_resilient.sh：启动期偶发 SIGSEGV 就重试，中途崩溃从 last 自动续训。
#
# 用法（在你的终端，conda activate lerobot 后）：
#   nohup bash scripts/train_new_table.sh > outputs/train/2026-10-17/resilient.log 2>&1 &
#   tail -f outputs/train/2026-10-17/resilient.log                      # 看重试/接力
#   tail -f outputs/train/2026-10-17/ACT_pickcube_noimu_seed42.log      # 看训练进度
set -u
cd ~/claude-test-2/lerobot-piper || exit 1
DATE=2026-10-17
NW="${NW:-2}"   # DataLoader workers；偶发 worker 崩由本脚本续训。最稳是 NW=0（慢）
mkdir -p "outputs/train/$DATE"
echo "[$(date)] num_workers=$NW"

run_until_100k () {
  local OUT="$1"; shift
  local LOG="${OUT}.log"
  local try
  for try in $(seq 1 80); do
    if [ -d "$OUT/checkpoints/100000" ]; then
      echo "[$(date)] OK $(basename "$OUT") 已达 100K"; return 0
    fi
    if [ -d "$OUT/checkpoints/last" ]; then
      echo "[$(date)] [$(basename "$OUT")] try#$try RESUME"
      PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py \
        --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
        --resume=true --num_workers=$NW --steps=100000 --log_freq=500 >> "$LOG" 2>&1
    else
      [ -d "$OUT" ] && rm -rf "$OUT"          # 启动崩留下的半成品（没有 checkpoint）
      echo "[$(date)] [$(basename "$OUT")] try#$try FRESH"
      PYTHONFAULTHANDLER=1 python -u lerobot/scripts/train.py "$@" >> "$LOG" 2>&1
    fi
    echo "[$(date)] [$(basename "$OUT")] 退出码 $? — 3s 后检查/重试"
    sleep 3
  done
  echo "[$(date)] FAIL $(basename "$OUT") 试 80 次仍未到 100K"; return 1
}

train () {   # $1=数据集  $2=名字
  local OUT="outputs/train/$DATE/ACT_$2_noimu_seed42"
  run_until_100k "$OUT" \
    --dataset.repo_id="enolab-guo/$1" \
    --policy.type=act --policy.enable_imu_input=false \
    --batch_size=16 --steps=100000 --num_workers=$NW --use_amp=false --seed=42 \
    --save_freq=10000 --log_freq=500 \
    --output_dir="$OUT" --job_name="ACT_$2_noimu_seed42" \
    --wandb.enable=true --wandb.project=lerobot-piper-newtable --wandb.disable_artifact=true
}

echo "[$(date)] ===== 弹性训练开始 ====="
train pickcube_raw pickcube || { echo "pickcube 未完成, 退出"; exit 1; }
train insert39 insert || { echo "insert 未完成, 退出"; exit 1; }
echo "[$(date)] ===== 全部完成: pickcube + insert 均到 100K ====="
