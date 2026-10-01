#!/usr/bin/env bash
# 阶段一（Q1）正式运行 —— **只在 reports/PREREG_Q1_STAGE1.md 冻结后**，在用户自己的终端里运行。
# 可重入：配置/代码指纹一致的已完成作业会被跳过；中断后直接重跑本脚本即可。
#   bash sim/q1/run_q1_stage1.sh            # 默认 OUT=outputs/q1_stage1 JOBS=15
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-/home/enolab-guo/miniconda3/envs/lerobot/bin/python}
OUT=${OUT:-outputs/q1_stage1}
JOBS=${JOBS:-15}
MODELS="1a 1b 2a 2b 3a 3b"
mkdir -p "$OUT/logs"
"$PY" -u sim/q1/q1_run.py prepare --out "$OUT" 2>&1 | tee "$OUT/logs/prepare.log"
# 参照先跑：它在真实数据上出错时，还没花掉 90 个作业
"$PY" -u sim/q1/q1_run.py refs --out "$OUT" 2>&1 | tee "$OUT/logs/refs.log"
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" > "$OUT/logs/todo.txt"
echo "待跑作业 $(wc -l < "$OUT/logs/todo.txt") 个，并行 $JOBS"
# -r：作业已全部完成（todo 为空）时不调用 job
xargs -r -P "$JOBS" -L 1 sh -c \
  '"$0" -u sim/q1/q1_run.py job --out "$1" "$2" "$3" "$4" > "$1/logs/job_$2_$(echo "$3" | tr / _)_$4.log" 2>&1' \
  "$PY" "$OUT" < "$OUT/logs/todo.txt"
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" > "$OUT/logs/todo_after.txt"
if [ -s "$OUT/logs/todo_after.txt" ]; then
  echo "仍有未完成作业（见 $OUT/logs/todo_after.txt 与 job_*.log）—— 修复后重跑本脚本"; exit 1
fi
pids=()
for m in $MODELS; do
  "$PY" -u sim/q1/q1_run.py fullfit --out "$OUT" "$m" > "$OUT/logs/fullfit_$m.log" 2>&1 &
  pids+=("$!")
done
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "全数据拟合有失败（见 $OUT/logs/fullfit_*.log）—— 修复后重跑本脚本"; exit 1
fi
"$PY" -u sim/q1/q1_report.py --out "$OUT" | tee "$OUT/logs/report.md"
for m in $MODELS; do
  "$PY" -u sim/q1/fig_pp2_q1.py --out "$OUT" --model "$m" | tee "$OUT/logs/fig_$m.log"
done
echo "完成：$OUT/summary.json、$OUT/logs/report.md、$OUT/fig/"
