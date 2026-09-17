#!/bin/bash
# POST /v1/systemone (TypeSafe wire format) and POST /decide.   scripts/serve.sh runs/decider_full/model [port]
# DECIDER_SCHEMAS=schemas.json preloads fixed question schemas (prefix cached, graphs compiled before traffic); DECIDER_COMPILE=0 for a fast start.
cd "$(dirname "$0")/.."; DECIDER_MODEL=${1:-Mapika/decider-2b} exec ${UVICORN:-.venv312/bin/uvicorn} decider.serve:app --host 0.0.0.0 --port ${2:-8000}
