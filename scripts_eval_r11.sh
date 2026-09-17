#!/bin/bash
# v6 evaluation: the 94-task regression set (original protocol), the v6 probes, full label sets, independence, abstention battery, games
cd /lambda/nfs/new-fs/longshots/test
export PYTHONUNBUFFERED=1
until grep -q TRAIN_DONE logs/r11_v6.log 2>/dev/null; do sleep 30; done
M=runs/r11_v6/model; PY=.venv312/bin/python
$PY -m decider.evaluate --model $M --data data/probes_v6.pkl --out runs/probes_v6/v6 --max_options 255 --max_ctx 32768 --bs 16 > logs/probes_v6.log 2>&1
WIDE=clinc_oos,banking77,massive_intent,go_emotions,bias_in_bios,bitext_support,newsgroups,dbpedia,massive_scenario
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r11_v6/wide --tasks $WIDE --max_options 255 --bs 16 > logs/wide_v6.log 2>&1
$PY -m decider.evaluate --model runs/r10_v5/model --data data/tasks_v4.pkl --out runs/r10_v5/wide --tasks $WIDE --max_options 255 --bs 16 > logs/wide_v5.log 2>&1
$PY -m decider.independence_probe $M --n 300 --out runs/probes_v6/independence_v6.json > logs/independence_v6.log 2>&1
$PY -m decider.none_battery $M > logs/r11_battery.log 2>&1
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r11_v6/final --engine compile --bs 32 > runs/r11_v6/final_eval.log 2>&1
echo EVAL_DONE >> runs/r11_v6/final_eval.log
$PY -m decider.games_eval $M --episodes 3 --out runs/games/r11_v6.json > logs/games_r11.log 2>&1
echo ALL_DONE >> logs/games_r11.log
