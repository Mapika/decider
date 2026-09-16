#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
until grep -q "EVAL_DONE" runs/r2_full/final_eval.log 2>/dev/null || [ -f runs/r2_full/EVAL_FAILED ]; do sleep 30; done
nohup .venv312/bin/python -m decider.train --data data/tasks_v2.pkl --out runs/r3_v2 --epochs 1 --lr 1e-5 --max_tokens 16384 --accum 2 --warmup 150 --eval_every 1500 --eval_limit 150 --none_prob 0.1 > logs/r3_v2.log 2>&1 &
sleep 5
setsid nohup ./scripts_wait_eval.sh runs/r3_v2 logs/r3_v2.log data/tasks_v2.pkl > /dev/null 2>&1 &
.venv312/bin/python -m decider.evaluate --model runs/r2_full/model --data data/tasks_v2.pkl --out runs/r2_full/final_v2 --bs 32 > runs/r2_full/final_v2_eval.log 2>&1
echo EVAL_DONE >> runs/r2_full/final_v2_eval.log
.venv312/bin/python -m decider.evaluate --model Qwen/Qwen3.5-2B-Base --data data/tasks_v2.pkl --out runs/zs_2b_v2 --bs 32 > logs/zs_2b_v2.log 2>&1
echo EVAL_DONE >> logs/zs_2b_v2.log
