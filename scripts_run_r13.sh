#!/bin/bash
# after v7 (r12): quick weakness checks on v7, then v8 = v7 + isolated level scoring (r13), then the full evaluation of v8 in both layouts
cd /lambda/nfs/new-fs/longshots/test
export PYTHONUNBUFFERED=1
until grep -q TRAIN_DONE logs/r12_v7.log 2>/dev/null; do sleep 30; done
PY=.venv312/bin/python; M7=runs/r12_v7/model; mkdir -p runs/probes_v7 runs/probes_v8
echo '{"temperature": 1.15, "neutralize_none": false, "version": "v7", "base": "Qwen/Qwen3.5-2B-Base", "max_options": 255, "max_state_tokens": 32768, "schema_first": true, "release_date": "2026-09-17"}' > $M7/decider_config.json
$PY -m decider.custom_battery $M7 --layout=state_first --layout=schema_first > logs/battery_v7.log 2>&1
for L in state_first schema_first; do
  $PY -m decider.evaluate --model $M7 --data data/probes_v7.pkl --out runs/probes_v7/v7_$L --max_options 255 --max_ctx 8192 --bs 32 --layout $L > logs/probes7_v7_$L.log 2>&1
done
echo V7_QUICK_DONE >> logs/battery_v7.log
$PY -m decider.train --model $M7 --data data/tasks_v8_delta.pkl --out runs/r13_v8 --epochs 1 --lr 6e-6 --max_tokens 16384 --accum 2 --warmup 50 --eval_every 100000 --eval_limit 150 --none_prob 0.1 --max_options 255 --max_ctx 16384 --schema_first_prob 0.5 > logs/r13_v8.log 2>&1
echo TRAIN_DONE >> logs/r13_v8.log
M=runs/r13_v8/model
echo '{"temperature": 1.15, "neutralize_none": false, "version": "v8", "base": "Qwen/Qwen3.5-2B-Base", "max_options": 255, "max_state_tokens": 32768, "schema_first": true, "isolated_levels": true, "release_date": "2026-09-17"}' > $M/decider_config.json
$PY -m decider.custom_battery $M --layout=state_first --layout=schema_first > logs/battery_v8.log 2>&1
for L in state_first schema_first; do
  $PY -m decider.isolated_probe $M --n 300 --layout $L --out runs/probes_v8/isolated_$L.json > logs/isolated_v8_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/probes_v7.pkl --out runs/probes_v8/v7probes_$L --max_options 255 --max_ctx 8192 --bs 32 --layout $L > logs/probes7_v8_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r13_v8/final_$L --engine compile --bs 32 --layout $L > runs/r13_v8/final_eval_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/probes_v6.pkl --out runs/probes_v8/v6probes_$L --max_options 255 --max_ctx 32768 --bs 16 --layout $L > logs/probes6_v8_$L.log 2>&1
  $PY -m decider.evaluate --model $M --data data/probes_v6_idx.pkl --out runs/probes_v8/idx_$L --max_options 255 --max_ctx 32768 --bs 8 --layout $L > logs/probesidx_v8_$L.log 2>&1
done
$PY -m decider.evaluate --model $M --data data/probes_v6_xl.pkl --out runs/probes_v8/xl_state_first --max_options 255 --max_ctx 32768 --bs 2 > logs/probesxl_v8.log 2>&1
WIDE=clinc_oos,banking77,massive_intent,go_emotions,bias_in_bios,bitext_support,newsgroups,dbpedia,massive_scenario
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r13_v8/wide --tasks $WIDE --max_options 255 --bs 16 > logs/wide_v8.log 2>&1
$PY -m decider.independence_probe $M --n 300 --out runs/probes_v8/independence_v8.json > logs/independence_v8.log 2>&1
$PY -m decider.none_battery $M > logs/r13_battery.log 2>&1
$PY -m decider.games_eval $M --episodes 3 --out runs/games/r13_v8.json > logs/games_r13.log 2>&1
echo ALL_DONE >> logs/games_r13.log
