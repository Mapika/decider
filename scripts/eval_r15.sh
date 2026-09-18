#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test; export PYTHONUNBUFFERED=1 PYTHONPATH=.; PY=.venv312/bin/python; M=runs/r15_v9b/model
$PY -m decider.evaluate --model $M --data data/tasks_v4.pkl --out runs/r15_v9b/final --engine compile --bs 32 > runs/r15_v9b/final_eval.log 2>&1
$PY -m decider.probes.isolated $M --n 300 --out runs/probes_v9/isolated_v9b.json > logs/isolated_v9b.log 2>&1
$PY -m decider.evaluate --model $M --data data/probes_v6.pkl --out runs/probes_v9/v9b_v6probes --max_options 255 --max_ctx 32768 --bs 16 --tasks hwu64,trec_fine,dbpedia_l3,json_k16,json_long,opaque:emotion,plain:emotion > logs/probes6_v9b.log 2>&1
$PY -m decider.probes.batteries $M --layout=schema_first > logs/battery_v9b_schema.log 2>&1
echo ALL_DONE >> logs/battery_v9b_schema.log
