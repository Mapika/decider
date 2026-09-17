#!/bin/bash
# v7 evaluation in both prompt layouts (state-first, and schema-first = the cacheable one)
cd /lambda/nfs/new-fs/longshots/test
export PYTHONUNBUFFERED=1
until grep -q TRAIN_DONE logs/r12_v7.log 2>/dev/null; do sleep 30; done
M=runs/r12_v7/model; PY=.venv312/bin/python; mkdir -p runs/probes_v7
echo '{"temperature": 1.15, "neutralize_none": false, "version": "v7", "base": "Qwen/Qwen3.5-2B-Base", "max_options": 255, "max_state_tokens": 32768, "schema_first": true, "release_date": "2026-09-17"}' > $M/decider_config.json
$PY -m decider.custom_battery $M --layout=state_first --layout=schema_first > logs/battery_v7.log 2>&1
for L in state_first schema_first; do
  $PY -m decider.evaluate --model $M --data data/probes_v7.pkl --out runs/probes_v7/v7_$L --max_options 255 --max_ctx 8192 --bs 32 --layout $L > logs/probes7_v7_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/probes_v6.pkl --out runs/probes_v7/v6probes_$L --max_options 255 --max_ctx 32768 --bs 16 --layout $L > logs/probes6_v7_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/probes_v6_idx.pkl --out runs/probes_v7/idx_$L --max_options 255 --max_ctx 32768 --bs 8 --layout $L > logs/probesidx_v7_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r12_v7/final_$L --engine compile --bs 32 --layout $L > runs/r12_v7/final_eval_$L.log 2>&1
done
$PY -m decider.evaluate --model $M --data data/probes_v6_xl.pkl --out runs/probes_v7/xl_state_first --max_options 255 --max_ctx 32768 --bs 2 > logs/probesxl_v7.log 2>&1
WIDE=clinc_oos,banking77,massive_intent,go_emotions,bias_in_bios,bitext_support,newsgroups,dbpedia,massive_scenario
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r12_v7/wide --tasks $WIDE --max_options 255 --bs 16 > logs/wide_v7.log 2>&1
$PY -m decider.independence_probe $M --n 300 --out runs/probes_v7/independence_v7.json > logs/independence_v7.log 2>&1
$PY -m decider.none_battery $M > logs/r12_battery.log 2>&1
$PY -m decider.games_eval $M --episodes 3 --out runs/games/r12_v7.json > logs/games_r12.log 2>&1
echo ALL_DONE >> logs/games_r12.log
