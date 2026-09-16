#!/bin/bash
# usage: scripts_wait_eval.sh <run_dir> <train_log> [data_pkl]
cd /lambda/nfs/new-fs/longshots/test
DATA=${3:-data/tasks.pkl}
until grep -q "\[done\]\|Traceback" $2; do sleep 30; done
grep -q Traceback $2 && { echo "TRAINING FAILED" > $1/EVAL_FAILED; exit 1; }
.venv312/bin/python -m decider.evaluate --model $1/model --data $DATA --out $1/final --bs 32 > $1/final_eval.log 2>&1
echo EVAL_DONE >> $1/final_eval.log
