#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
until grep -q DAGGER_DONE logs/frames_dagger.log 2>/dev/null && grep -q R8_DONE logs/r8_battery.log 2>/dev/null; do sleep 30; done
.venv312/bin/python -m decider.build_vision2 > logs/build_vision2.log 2>&1
.venv312/bin/python -m decider.train_vision --model runs/r8_v5/vlm --data data/vision_mix2.pkl --out runs/v2_vision --epochs 1 --lr 1e-5 --warmup 100 --bs_img 24 --bs_txt 32 --eval_limit 300 --save_every 500 > logs/v2_vision.log 2>&1
.venv312/bin/python -m decider.games_pixels runs/v2_vision/model --episodes 3 --mario --out runs/games/pixels_v2.json > logs/games_pixels_v2.log 2>&1
echo PIXELS_DONE >> logs/games_pixels_v2.log
