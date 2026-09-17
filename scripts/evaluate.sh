#!/bin/bash
# Everything we report, for one checkpoint: the regression set and the input-shape probes in both prompt layouts, full label
# sets, the hand-written batteries, independence, isolated levels, text games.   scripts/evaluate.sh runs/decider_full/model
set -e; cd "$(dirname "$0")/.."; export PYTHONUNBUFFERED=1
PY=${PY:-.venv312/bin/python}; M=$1; R=$(dirname "$M")/eval; mkdir -p "$R" logs
DATA=${DATA:-$(ls data/mixture_*.pkl | head -1)}                                     # its eval half is the regression set
[ -f "$M/decider_config.json" ] || echo '{"temperature": 1.15, "neutralize_none": false, "version": "dev", "max_options": 255, "max_state_tokens": 32768, "schema_first": true, "isolated_levels": true}' > "$M/decider_config.json"
for L in state_first schema_first; do
  $PY -m decider.evaluate --model "$M" --data "$DATA" --out "$R/regression_$L" --engine compile --bs 32 --layout $L
  $PY -m decider.evaluate --model "$M" --data data/probes.pkl --out "$R/probes_$L" --max_options 255 --max_ctx 32768 --bs 16 --layout $L
  $PY -m decider.probes.isolated "$M" --layout $L --out "$R/isolated_$L.json"
done
$PY -m decider.report "$R/regression_state_first"                                     # fits the temperature on in-task data; put it into decider_config.json
$PY -m decider.evaluate --model "$M" --data "$DATA" --out "$R/full_label_sets" --max_options 255 --bs 16 \
    --tasks clinc_oos,banking77,massive_intent,go_emotions,bias_in_bios,bitext_support,newsgroups,dbpedia,massive_scenario
$PY -m decider.probes.batteries "$M" --layout=state_first --layout=schema_first
$PY -m decider.probes.independence "$M" --out "$R/independence.json"
$PY -m decider.games.play "$M" --episodes 3 --out "$R/games.json"
