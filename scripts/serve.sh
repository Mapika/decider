#!/bin/bash
# POST /v1/systemone (TypeSafe wire format) and POST /decide.   scripts/serve.sh runs/decider_full/model [port]
# Listens on 127.0.0.1 only; DECIDER_HOST=0.0.0.0 opens it to the network (there is no authentication).
# DECIDER_SCHEMAS=schemas.json preloads fixed question schemas (prefix cached, graphs compiled before traffic); DECIDER_COMPILE=0 for a fast start.
cd "$(dirname "$0")/.."; DECIDER_MODEL=${1:-Mapika/decider-2b} exec ${UVICORN:-.venv312/bin/uvicorn} decider.serve:app --host ${DECIDER_HOST:-127.0.0.1} --port ${2:-8000}
