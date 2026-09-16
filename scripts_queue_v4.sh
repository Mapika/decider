#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
.venv312/bin/python -m decider.train --model runs/r3_v2/model --data data/tasks_v4_delta.pkl --out runs/r7_v4 --epochs 1 --lr 8e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 300 --none_prob 0.1 > logs/r7_v4.log 2>&1
.venv312/bin/python -m decider.evaluate --model runs/r7_v4/model --data data/tasks_v4.pkl --out runs/r7_v4/final --engine compile --bs 32 > runs/r7_v4/final_eval.log 2>&1
echo EVAL_DONE >> runs/r7_v4/final_eval.log
.venv312/bin/python -m decider.games_eval runs/r7_v4/model --episodes 3 --out runs/games/r7_v4.json > logs/games_r7.log 2>&1
echo GAMES_DONE >> logs/games_r7.log
