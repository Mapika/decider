"""Micro-batching HTTP server.
  POST /decide        {"context": str, "schema": {...}}                     -> typed JSON decisions (all questions packed in one row)
  POST /v1/systemone  {"state": str|object|array, "questions": {id: {...}}} -> the TypeSafe/Jev wire format (decider.systemone):
                      Choice (up to 255 described options), Score, Noul; every question is scored in its own row, so answers
                      are independent of each other ("independent": false packs them behind one copy of the state instead).
Requests arriving within `max_wait_ms` are scored in one forward pass (grouped by length bucket).
   uvicorn decider.serve:app --host 0.0.0.0 --port 8000     (env: DECIDER_MODEL, DECIDER_MAX_BATCH, DECIDER_MAX_WAIT_MS)
"""
import asyncio, os, random, time, threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from decider.engine import Engine, T_BUCKETS, _bucket
from decider.prompt import build, MAX_OPTIONS
from decider.infer import Decider, Example, Q, neutralize_options
from decider import systemone as S1

MODEL = os.environ.get("DECIDER_MODEL", "runs/r3_v2/model")
MAX_BATCH = int(os.environ.get("DECIDER_MAX_BATCH", "32"))
MAX_WAIT_MS = float(os.environ.get("DECIDER_MAX_WAIT_MS", "8"))
BATCH_WAIT_MS = float(os.environ.get("DECIDER_BATCH_WAIT_MS", "0"))
MAX_STATE_TOKENS = int(os.environ.get("DECIDER_MAX_STATE_TOKENS", "32768"))
MAX_FWD_TOKENS = int(os.environ.get("DECIDER_MAX_FWD_TOKENS", "65536"))     # padded tokens per forward pass
COMPILE = os.environ.get("DECIDER_COMPILE", "1") == "1"
FP8 = os.environ.get("DECIDER_FP8", "1") == "1"
app = FastAPI(title="decider")
MODEL_NAME = "decider"; TEMP = 1.0; TEMP_SCHEMA = 1.0; RELEASE_DATE = "2026-09-17"
gpu_lock = threading.Lock()            # one GPU job at a time: batched graph replays and shared-prefix requests must not interleave
SHARED_MIN_TOKENS = int(os.environ.get("DECIDER_SHARED_MIN_TOKENS", "768"))   # independent rows over a state this long share one prefix pass
eng = None; queue = None; stats = dict(requests=0, batches=0, decisions=0, batch_hist={})


class Req(BaseModel):
    context: str
    schema_: dict = None
    model_config = {"populate_by_name": True}
    def __init__(self, **kw):
        if "schema" in kw: kw["schema_"] = kw.pop("schema")
        super().__init__(**kw)


class _NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def _prepare(context, schema):
    qs = Decider._schema_to_questions(schema)
    for q in qs:
        if getattr(eng, "neutralize_none", True):
            q["options"], q["_back"] = neutralize_options(q["options"])
    ex = Example(context, [Q(q["question"], list(q["options"]), 0) for q in qs])
    it = build(ex, eng.tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=eng.max_ctx)
    return qs, it


def _format(schema, qs, probs):
    o = {}
    for (qtext, spec), q, p in zip(schema.items(), qs, probs):
        p = p[:len(q["options"])].tolist(); t = spec.get("type", "choice"); j = max(range(len(p)), key=p.__getitem__)
        back = q.get("_back", {}); names = [back.get(x, x) for x in q["options"]]
        if t == "bool":
            o[qtext] = {"noul": round(p[1], 4), "type": "noul"}
        elif t == "choice":
            o[qtext] = {"choice": names[j], "confidence": round(p[j], 4), "type": "choice",
                        "probabilities": {k: round(v, 4) for k, v in zip(names, p)}}
        else:
            keys = q["_keys"]; score = sum(float(k) * pi for k, pi in zip(keys, p))
            o[qtext] = {"score": round(score, 2), "confidence": round(p[j], 4), "type": "scale", "legend": q["_legend"],
                        "probabilities": {str(keys[i]): round(pi, 4) for i, pi in enumerate(p)}}
    return o


async def _collect(q):
    """Continuous batching: take what is already queued and go.  While a forward pass runs, new requests pile up and form the
    next batch, so there is no fixed wait at low load (it cost 1.5 ms per request) and full batches at high load.
    DECIDER_BATCH_WAIT_MS > 0 restores a short collection window."""
    batch = [await q.get()]; deadline = time.monotonic() + BATCH_WAIT_MS / 1000
    while len(batch) < MAX_BATCH:
        try:
            batch.append(q.get_nowait())
        except asyncio.QueueEmpty:
            timeout = deadline - time.monotonic()
            if timeout <= 0: break
            try: batch.append(await asyncio.wait_for(q.get(), timeout))
            except asyncio.TimeoutError: break
    return batch


async def batcher():
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect(queue)
        # sort by length; split into at most two groups when the spread is large (keeps padding small)
        batch.sort(key=lambda x: len(x[2]["ids"]))
        groups = [batch]
        if len(batch) >= 4:
            lo, hi = len(batch[0][2]["ids"]), len(batch[-1][2]["ids"])
            if _bucket(hi, T_BUCKETS) != _bucket(lo, T_BUCKETS) and hi > 1.5 * lo:
                cut = len(batch) // 2; groups = [batch[:cut], batch[cut:]]
        capped = []                                       # long rows: keep every forward under MAX_FWD_TOKENS padded tokens
        for g in groups:
            cur = []
            for x in g:
                if cur and (len(cur) + 1) * len(x[2]["ids"]) > MAX_FWD_TOKENS:
                    capped.append(cur); cur = []
                cur.append(x)
            capped.append(cur)
        for g in capped:
            items = [it for _, _, it in g]
            try:
                probs = await loop.run_in_executor(None, _locked, eng.score_items, items)
                for (fut, qs, it), p in zip(g, probs):
                    if not fut.done(): fut.set_result(p)
            except Exception as e:
                for fut, _, _ in g:
                    if not fut.done(): fut.set_exception(e)
            stats["batches"] += 1; stats["batch_hist"][len(g)] = stats["batch_hist"].get(len(g), 0) + 1


@app.on_event("startup")
async def _start():
    global eng, queue
    eng = Engine(MODEL, compile=COMPILE, fp8=FP8, conv_patch=COMPILE); print("[serve] engine", eng.cfg, flush=True)
    import json
    global MODEL_NAME, TEMP
    try: cfg = json.load(open(os.path.join(MODEL, "decider_config.json")))
    except Exception: cfg = {}
    eng.neutralize_none = bool(cfg.get("neutralize_none", True)); MODEL_NAME = "decider-" + str(cfg.get("version", "dev"))
    TEMP = float(os.environ.get("DECIDER_TEMPERATURE", cfg.get("temperature", 1.0)))
    global RELEASE_DATE; RELEASE_DATE = str(cfg.get("release_date", RELEASE_DATE))
    global SCHEMA_FIRST, se, squeue, ISOLATED
    ISOLATED = bool(cfg.get("isolated_levels", False))
    # the schema cache needs the questions-first layout, which costs accuracy (about 1.5 points on fixed label sets, more on large
    # label sets and long states): on when the model's config makes it the default, or with DECIDER_SCHEMA_CACHE=1
    trained = bool(cfg.get("schema_first", False) or cfg.get("schema_first_trained", False))
    SCHEMA_FIRST = trained and (bool(cfg.get("schema_first", False)) or os.environ.get("DECIDER_SCHEMA_CACHE", "0") == "1")
    global TEMP_SCHEMA; TEMP_SCHEMA = float(cfg.get("temperature_schema_first", TEMP))
    if SCHEMA_FIRST:
        from decider.schema_engine import SchemaEngine
        se = SchemaEngine(eng); squeue = asyncio.Queue(); asyncio.create_task(schema_batcher()); print("[serve] schema cache on", flush=True)
        pre = os.environ.get("DECIDER_SCHEMAS")             # JSON file: [{"questions": {...}, "independent": true, "batch_sizes": [1, 8, 32], "state_tokens": [64, 256]}]
        for spec in (json.load(open(pre)) if pre else []):  # known schemas: prefix computed, graphs compiled and captured before traffic
            _, h, _ = _schema_handle(spec["questions"], spec.get("independent", True), compile=COMPILE)
            t = se.warmup(h, spec.get("batch_sizes", (1, 8, 32)), spec.get("state_tokens", (64, 128, 256)))
            print(f"[serve] preloaded schema with {h.nq} rows, prefix {sum(h.tps)} tokens, graphs ready in {t:.0f}s", flush=True)
    shapes = [(B, T) for B in (1, 2, 4, 8, 16, 32) for T in T_BUCKETS if T <= eng.max_ctx + 256]
    if MAX_BATCH > 32: shapes += [(64, T) for T in T_BUCKETS if T <= 512]
    t = eng.warmup(shapes); print(f"[serve] captured {len(shapes)} graphs in {t:.0f}s", flush=True)
    queue = asyncio.Queue()
    asyncio.create_task(batcher())


@app.post("/decide")
async def decide(r: Req):
    qs, it = await asyncio.get_running_loop().run_in_executor(None, _prepare, r.context, r.schema_)
    fut = asyncio.get_running_loop().create_future()
    await queue.put((fut, qs, it))
    probs = await fut
    stats["requests"] += 1; stats["decisions"] += len(qs)
    return _format(r.schema_, qs, probs)


SCHEMA_FIRST = False; se = None; squeue = None; schemas = {}            # schema cache (models trained on the questions-first layout, v7+)


ISOLATED = False


def _schema_handle(questions, independent, compile=False):
    """Compile (or look up) the question schema: its prefix is run once, requests then only run the state."""
    import json
    key = (json.dumps(questions, sort_keys=True, ensure_ascii=False), independent)
    if key not in schemas:
        rqs = {k: S1.render_question(v) for k, v in questions.items()}; rows, index = S1.plan_rows(rqs, ISOLATED and independent)
        with gpu_lock:
            if len(schemas) >= 128:
                old = next(iter(schemas)); hid = schemas.pop(old)[1].id
                for k in [k for k in se.graphs if k[0] == hid]: del se.graphs[k]
            h = se.prepare(rows, independent=independent, compile=compile)
        schemas[key] = (rqs, h, index)
    return schemas[key]


seen = {}


def _worth_caching(questions, independent):
    """A schema gets a cached prefix and CUDA graphs from its second request on: one-off schemas go through the generic
    state-first engine, whose graphs do not depend on the questions, so ad-hoc traffic cannot thrash graph captures."""
    import json
    key = (json.dumps(questions, sort_keys=True, ensure_ascii=False), independent)
    if key in schemas: return True
    if len(seen) > 50000: seen.clear()
    seen[key] = seen.get(key, 0) + 1
    return seen[key] >= int(os.environ.get("DECIDER_SCHEMA_MIN_SEEN", "2"))


def _score_schema(h, rows):
    with gpu_lock:
        return se.score_rows(h, rows, temperature=TEMP_SCHEMA)


async def schema_batcher():
    """Requests that share a schema and arrive within the window are scored in one forward pass over their states."""
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect(squeue)
        groups = {}                                        # one forward per (schema, length bucket): short states are not padded to long ones
        for fut, h, row in batch: groups.setdefault((h.id, se.bucket(len(row[0]))), (h, []))[1].append((fut, row))
        for h, items in groups.values():
            step = max(1, MAX_BATCH // h.P)
            for i in range(0, len(items), step):
                chunk = items[i:i + step]
                try:
                    probs = await loop.run_in_executor(None, _score_schema, h, [c for _, c in chunk])
                    for (fut, _), p in zip(chunk, probs):
                        if not fut.done(): fut.set_result(p)
                except Exception as e:
                    for fut, _ in chunk:
                        if not fut.done(): fut.set_exception(e)
            stats["schema_batches"] = stats.get("schema_batches", 0) + 1


def _locked(fn, items):
    with gpu_lock:
        return fn(items, temperature=TEMP)          # fitted temperature from decider_config.json


class S1Req(BaseModel):
    state: object
    questions: dict
    model: str | None = None
    independent: bool = True
    layout: str | None = None            # "state_first" forces the uncached layout on a schema-first model


def _prepare_s1(state, questions, independent):
    ctx = S1.render_state(state); rqs = {k: S1.render_question(v) for k, v in questions.items()}
    flat, index = S1.plan_rows(rqs, ISOLATED and independent)
    rows = [[r] for r in flat] if independent else [flat]
    items = [build(Example(ctx, [Q(r["question"], list(r["options"]), 0) for r in row]), eng.tok, _NoShuffle(), max_options=MAX_OPTIONS,
                   max_ctx_tokens=MAX_STATE_TOKENS) for row in rows]
    return (rqs, index), items


@app.post("/v1/systemone")
async def systemone(r: S1Req):
    loop = asyncio.get_running_loop()
    if SCHEMA_FIRST and r.layout != "state_first" and _worth_caching(r.questions, r.independent):
        try:
            rqs, h, index = await loop.run_in_executor(None, _schema_handle, r.questions, r.independent)
        except ValueError as e:
            raise HTTPException(422, str(e))
        row = await loop.run_in_executor(None, lambda: se.tokenize(h, S1.render_state(r.state), MAX_STATE_TOKENS))      # CPU work stays off the GPU lock
        fut = loop.create_future(); await squeue.put((fut, h, row)); p = await fut
        stats["requests"] += 1; stats["decisions"] += len(rqs); stats["schema_requests"] = stats.get("schema_requests", 0) + 1
        return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, [pk.tolist() for pk in p]),
                "usage": {"input_tokens": len(row[0]) * h.P, "cached_tokens": sum(h.tps), "output_tokens": 0}}
    try:
        (rqs, index), items = await loop.run_in_executor(None, _prepare_s1, r.state, r.questions, r.independent)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if len(items) > 1 and min(len(it["ids"]) for it in items) >= SHARED_MIN_TOKENS:
        res = await loop.run_in_executor(None, _locked, eng.score_shared, items)      # long state: run it once, fork the cache per question
        stats["shared_prefix_requests"] = stats.get("shared_prefix_requests", 0) + 1
    else:
        futs = []
        for it in items:
            f = loop.create_future(); futs.append(f); await queue.put((f, None, it))
        res = await asyncio.gather(*futs)
    probs = [p for ps in res for p in ps]                                   # one prob row per question, request order
    stats["requests"] += 1; stats["decisions"] += len(rqs)
    return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, [p.tolist() for p in probs]),
            "usage": {"input_tokens": S1.unique_tokens(items), "output_tokens": 0}}


@app.get("/v1/models")
async def models():
    return {"models": [{"name": MODEL_NAME, "description": "decider: one-pass typed decisions with calibrated probabilities", "release_date": RELEASE_DATE}]}


@app.get("/health")
async def health():
    return {"ok": eng is not None, "model": MODEL}


@app.get("/stats")
async def get_stats():
    return dict(stats, engine=eng.stats if eng else None, graphs=len(eng.graphs) if eng else 0)
