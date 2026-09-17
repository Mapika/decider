#!/bin/bash
# v6: continue from v5 (r10) on the Jev-gap formats (described options, up to 255 options, JSON states with paths, long inputs) + replay
cd /lambda/nfs/new-fs/longshots/test
export PYTHONUNBUFFERED=1
.venv312/bin/python -m decider.train --model runs/r10_v5/model --data data/tasks_v6_delta.pkl --out runs/r11_v6 --epochs 1 --lr 8e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 150 --none_prob 0.1 --max_options 255 --max_ctx 16384 > logs/r11_v6.log 2>&1
echo TRAIN_DONE >> logs/r11_v6.log
