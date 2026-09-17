#!/bin/bash
# The reference recipe, end to end.  One GH200 / H100-class GPU; PY points at the project venv.
#   scripts/train.sh full     Qwen3.5-2B-Base -> decider, one epoch over the full mixture (about 1.3M examples)
#   scripts/train.sh delta M  continue from an existing decider checkpoint M on the new formats + a replay sample
set -e; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
PY=${PY:-.venv312/bin/python}; MODE=${1:-full}; INIT=${2:-Qwen/Qwen3.5-2B-Base}; OUT=${OUT:-runs/decider_$MODE}
mkdir -p data logs
[ -f data/tasks.pkl ] || $PY -m decider.data.core --out data/tasks.pkl                 # download + convert ~95 public datasets (teacher_data/ is in the repo)
[ -f data/mixture_$MODE.pkl ] || $PY -m decider.data.mixture --base data/tasks.pkl --mode $MODE --out data/mixture_$MODE.pkl --probes data/probes.pkl
LR=1e-5; [ "$MODE" = delta ] && LR=8e-6
$PY -m decider.train --model "$INIT" --data data/mixture_$MODE.pkl --out "$OUT" --epochs 1 --lr $LR --warmup 150 --max_tokens 16384 --accum 2 \
    --max_options 255 --max_ctx 16384 --none_prob 0.1 --schema_first_prob 0.5 --eval_every 100000 --eval_limit 150 2>&1 | tee logs/train_$MODE.log
scripts/evaluate.sh "$OUT/model"
