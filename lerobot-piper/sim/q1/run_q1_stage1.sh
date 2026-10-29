#!/usr/bin/env bash
# 阶段一（Q1）正式运行 —— **只在 reports/PREREG_Q1_STAGE1.md 冻结后**，在用户自己的终端里运行。
# 两阶段：内层（每个 模型×外层折×验证会话×种子 训到上限并给检查点评分）→ 终训（选 step*、训到 step*、
# 外层留出会话只评一次；fold=all 的终训只供 pp2 图）。
# 可重入：配置指纹与输入指纹都与当前一致的已完成作业会被跳过；中断后直接重跑本脚本即可。
# 墙钟上限：第一次 prepare 记录截止时间（+24 小时），续跑不重置；到点后作业拒绝/中止并写 INCOMPLETE.json，
# 汇总只写「未完成」、不作通过判决。
#   bash sim/q1/run_q1_stage1.sh            # 默认 OUT=outputs/q1_stage1 JOBS=15
set -euo pipefail
cd "$(dirname "$0")/../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1     # 闭式拟合逐位稳定
PY=${PY:-/home/enolab-guo/miniconda3/envs/lerobot/bin/python}
OUT=${OUT:-outputs/q1_stage1}
JOBS=${JOBS:-15}
MODELS="1a 1b 2a 2b 3a 3b"
mkdir -p "$OUT/logs"
"$PY" -u sim/q1/q1_run.py prepare --out "$OUT" 2>&1 | tee "$OUT/logs/prepare.log"
# 参照先跑：它在真实数据上出错时，还没花掉内层作业
"$PY" -u sim/q1/q1_run.py refs --out "$OUT" 2>&1 | tee "$OUT/logs/refs.log"

"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" --phase inner > "$OUT/logs/todo_inner.txt"
echo "待跑内层作业 $(wc -l < "$OUT/logs/todo_inner.txt") 个，并行 $JOBS"
# -r：没有待跑作业时不调用
xargs -r -P "$JOBS" -L 1 sh -c \
  '"$0" -u sim/q1/q1_run.py inner --out "$1" "$2" "$3" "$4" "$5" > "$1/logs/inner_$2_$(echo "$3" | tr / _)_$(echo "$4" | tr / _)_$5.log" 2>&1' \
  "$PY" "$OUT" < "$OUT/logs/todo_inner.txt" \
  || { echo "有内层作业失败或已到本轮截止时间（见 $OUT/logs/inner_*.log；超时会写 $OUT/INCOMPLETE.json）"; exit 1; }
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" --phase inner > "$OUT/logs/todo_inner_after.txt"
if [ -s "$OUT/logs/todo_inner_after.txt" ]; then
  echo "仍有未完成的内层作业（见 $OUT/logs/todo_inner_after.txt 与 inner_*.log）—— 修复后重跑本脚本"; exit 1
fi

"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" --phase final > "$OUT/logs/todo_final.txt"
echo "待跑终训作业 $(wc -l < "$OUT/logs/todo_final.txt") 个，并行 $JOBS"
xargs -r -P "$JOBS" -L 1 sh -c \
  '"$0" -u sim/q1/q1_run.py final --out "$1" "$2" "$3" "$4" > "$1/logs/final_$2_$(echo "$3" | tr / _)_$4.log" 2>&1' \
  "$PY" "$OUT" < "$OUT/logs/todo_final.txt" \
  || { echo "有终训作业失败或已到本轮截止时间（见 $OUT/logs/final_*.log；超时会写 $OUT/INCOMPLETE.json）"; exit 1; }
"$PY" -u sim/q1/q1_run.py list-jobs --out "$OUT" --phase final > "$OUT/logs/todo_final_after.txt"
if [ -s "$OUT/logs/todo_final_after.txt" ]; then
  echo "仍有未完成的终训作业（见 $OUT/logs/todo_final_after.txt 与 final_*.log）—— 修复后重跑本脚本"; exit 1
fi

"$PY" -u sim/q1/q1_report.py --out "$OUT" | tee "$OUT/logs/report.md"
for m in $MODELS; do
  "$PY" -u sim/q1/fig_pp2_q1.py --out "$OUT" --model "$m" | tee "$OUT/logs/fig_$m.log"
done
echo "完成：$OUT/summary.json、$OUT/logs/report.md、$OUT/fig/"
