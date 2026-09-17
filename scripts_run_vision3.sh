#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
until grep -q EVAL_DONE runs/r10_v5/final_eval.log 2>/dev/null; do sleep 30; done
.venv312/bin/python -m decider.transplant runs/r10_v5/model runs/r10_v5/vlm > logs/r10_transplant.log 2>&1
.venv312/bin/python -m decider.build_vision2 > logs/build_vision3.log 2>&1
.venv312/bin/python -m decider.train_vision --model runs/r10_v5/vlm --data data/vision_mix2.pkl --out runs/v3_vision --epochs 1 --lr 1e-5 --warmup 100 --bs_img 24 --bs_txt 32 --eval_limit 300 --save_every 500 > logs/v3_vision.log 2>&1
.venv312/bin/python -m decider.games_rl --vision --init runs/v3_vision/model --games breakout,pong --eval_games breakout,pong,freeway,minigrid_empty --out runs/rl_pixels3 --iters 12 --envs_per_game 16 --max_t 300 --lr 3e-6 --batch 16 --update_samples 4000 --eval_every 4 --eval_episodes 1 > logs/rl_pixels3.log 2>&1
.venv312/bin/python -m decider.games_pixels runs/rl_pixels3/model --episodes 3 --mario --out runs/games/pixels_v3.json > logs/games_pixels_v3.log 2>&1
echo PIXELS_DONE >> logs/games_pixels_v3.log
