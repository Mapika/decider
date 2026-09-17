#!/bin/bash
# v7: continue from v6 (r11): teacher-written custom questions (generic vs catch-all, free-form noul), index-annotated JSON arrays,
# situation data, replay; every example rendered state-first or schema-first (cacheable question prefix) at random
cd /lambda/nfs/new-fs/longshots/test
export PYTHONUNBUFFERED=1
.venv312/bin/python -m decider.train --model runs/r11_v6/model --data data/tasks_v7_delta.pkl --out runs/r12_v7 --epochs 1 --lr 8e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 150 --none_prob 0.1 --max_options 255 --max_ctx 16384 --schema_first_prob 0.5 > logs/r12_v7.log 2>&1
echo TRAIN_DONE >> logs/r12_v7.log
