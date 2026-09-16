#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
.venv312/bin/python -m decider.train --model Qwen/Qwen3.5-2B-Base --data data/tasks_v4.pkl --out runs/r10_v5 --epochs 1 --lr 1e-5 --max_tokens 16384 --accum 2 --warmup 150 --eval_every 2000 --eval_limit 150 --none_prob 0.1 > logs/r10_v5.log 2>&1
.venv312/bin/python -m decider.none_battery runs/r10_v5/model > logs/r9_battery.log 2>&1
.venv312/bin/python -m decider.evaluate --model runs/r10_v5/model --data data/tasks_v4.pkl --out runs/r10_v5/final --engine compile --bs 32 > runs/r10_v5/final_eval.log 2>&1
echo EVAL_DONE >> runs/r10_v5/final_eval.log
echo R10_DONE >> logs/r9_battery.log
