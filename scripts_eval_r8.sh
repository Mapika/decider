#!/bin/bash
cd /lambda/nfs/new-fs/longshots/test
until grep -q TRAINDONE logs/r8_v5.log 2>/dev/null; do sleep 30; done
.venv312/bin/python -m decider.none_battery runs/r8_v5/model > logs/r8_battery.log 2>&1
.venv312/bin/python -m decider.evaluate --model runs/r8_v5/model --data data/tasks_v4.pkl --out runs/r8_v5/final --engine compile --bs 32 > runs/r8_v5/final_eval.log 2>&1
echo EVAL_DONE >> runs/r8_v5/final_eval.log
.venv312/bin/python -m decider.transplant runs/r8_v5/model runs/r8_v5/vlm > logs/r8_transplant.log 2>&1
echo R8_DONE >> logs/r8_battery.log
