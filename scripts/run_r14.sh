#!/bin/bash
# v9: continue from v8 on terse-bucket routing + shell-command safety (teacher-written) + replay of the v8 mix; then the checks
cd /lambda/nfs/new-fs/longshots/test; export PYTHONUNBUFFERED=1 PYTHONPATH=.; PY=.venv312/bin/python; M=runs/r14_v9/model
$PY -m decider.train --model runs/r13_v8/model --data data/tasks_v9_delta.pkl --out runs/r14_v9 --epochs 1 --lr 5e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 150 --none_prob 0.1 --max_options 255 --max_ctx 16384 --schema_first_prob 0.5 > logs/r14_v9.log 2>&1
echo TRAIN_DONE >> logs/r14_v9.log
cp runs/r13_v8/model/decider_config.json $M/decider_config.json; sed -i 's/"version": "v8"/"version": "v9"/; s/2026-09-17/2026-09-18/' $M/decider_config.json
$PY -m decider.probes.batteries $M --layout=state_first --layout=schema_first > logs/battery_v9.log 2>&1
$PY -m decider.probes.applications $M > logs/applications_v9.log 2>&1
$PY -m decider.evaluate --model $M --data data/probes_v9.pkl --out runs/probes_v9/v9 --max_options 255 --max_ctx 4096 --bs 32 > logs/probes9_v9.log 2>&1
$PY -m decider.evaluate --model $M --data data/probes_v7.pkl --out runs/probes_v9/v7probes --max_options 255 --max_ctx 8192 --bs 32 > logs/probes7_v9.log 2>&1
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r14_v9/final --engine compile --bs 32 > runs/r14_v9/final_eval.log 2>&1
$PY -m decider.probes.isolated $M --n 300 --out runs/probes_v9/isolated.json > logs/isolated_v9.log 2>&1
echo ALL_DONE >> logs/isolated_v9.log
