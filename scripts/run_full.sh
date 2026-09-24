#!/bin/bash
# The reference recipe, run for real: fresh task cache -> full mixture -> one epoch from Qwen3.5-2B-Base -> scripts/evaluate.sh
cd /lambda/nfs/new-fs/longshots/test; export PYTHONUNBUFFERED=1
if OUT=runs/decider_full BASE=data/tasks_full.pkl scripts/train.sh full > logs/full_run.log 2>&1; then
  echo FULL_RUN_DONE >> logs/full_run.log
else
  rc=$?; echo "FULL_RUN_FAILED exit $rc" >> logs/full_run.log; exit $rc
fi
