#!/bin/bash
# v9b: same as v9 but the catch-all cases weighted x3 instead of generic x2 (v9 over-reached into the generic bucket)
cd /lambda/nfs/new-fs/longshots/test; export PYTHONUNBUFFERED=1 PYTHONPATH=.; PY=.venv312/bin/python; M=runs/r15_v9b/model
$PY -m decider.train --model runs/r13_v8/model --data data/tasks_v9b_delta.pkl --out runs/r15_v9b --epochs 1 --lr 5e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 150 --none_prob 0.1 --max_options 255 --max_ctx 16384 --schema_first_prob 0.5 > logs/r15_v9b.log 2>&1
echo TRAIN_DONE >> logs/r15_v9b.log
cp runs/r14_v9/model/decider_config.json $M/decider_config.json
$PY -m decider.probes.batteries $M --layout=state_first > logs/battery_v9b.log 2>&1
$PY -m decider.probes.applications $M > logs/applications_v9b.log 2>&1
$PY -m decider.evaluate --model $M --data data/probes_v9.pkl --out runs/probes_v9/v9b --max_options 255 --max_ctx 4096 --bs 32 > logs/probes9_v9b.log 2>&1
$PY -m decider.evaluate --model $M --data data/probes_v7.pkl --out runs/probes_v9/v9b_v7probes --max_options 255 --max_ctx 8192 --bs 32 --tasks routing_generic,routing_specific,routing_catchall,custom_choice_catchall > logs/probes7_v9b.log 2>&1
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r15_v9b/probes --tasks abstain_probe,offtopic_probe,trec,support_tickets --bs 32 > logs/abst_v9b.log 2>&1
echo ALL_DONE >> logs/abst_v9b.log
