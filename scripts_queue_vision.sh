#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
until [ -f data/cauldron.pkl ] && ! pgrep -f "decider.data_vision" >/dev/null; do sleep 30; done
.venv312/bin/python -m decider.build_vision > logs/build_vision.log 2>&1
.venv312/bin/python -m decider.train_vision --data data/vision_mix.pkl --out runs/v1_vision --epochs 1 --lr 1e-5 --warmup 100 --bs_img 24 --bs_txt 48 --eval_limit 300 > logs/v1_vision.log 2>&1
.venv312/bin/python -m decider.games_pixels runs/v1_vision/model --episodes 3 --mario --out runs/games/pixels_v1.json > logs/games_pixels_v1.log 2>&1
echo PIXELS_DONE >> logs/games_pixels_v1.log
