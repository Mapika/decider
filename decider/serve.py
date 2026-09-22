"""HTTP server.
  POST /decide        {"context": str, "schema": {...}}                     -> typed JSON decisions (all questions packed in one row)
  POST /v1/systemone  {"state": str|object|array, "questions": {id: {...}}} -> the TypeSafe/Jev wire format (decider.systemone):
                      Choice (up to 255 described options), Score, Noul; every question is scored in its own row, so answers
                      are independent of each other ("independent": false packs them behind one copy of the state instead).
  GET  /v1/models  /health  /stats

   uvicorn decider.serve:app --host 0.0.0.0 --port 8000        (env: DECIDER_MODEL and the variables below)

Execution (1.1.0; docs/SERVING.md has the design and the measurements):
  * decider.engine_v2.EngineV2: CUDA graphs keyed on (batch bucket, padded length bucket) only.  The whole grid is captured
    during start-up and the engine is then sealed, so no request pays a graph capture or a torch.compile.  Rows longer than the
    last length bucket (8,192 tokens) run eager, in chunks bounded by DECIDER_GRAPH_TOKEN_BUDGET padded tokens.
  * one GPU thread runs every forward; tokenisation runs on a CPU pool; rows waiting at the same moment are grouped by length
    bucket and replayed together.
  * an independent request with more than one question over a state of at least DECIDER_SHARED_MIN_TOKENS tokens runs the state
    once and forks its cache per question (EngineV2.score_shared).  The cuDNN SDPA backend is off (decider.engine
    .set_attention_backend_policy), which is what makes that path agree with the full forward.
  * the schema cache (questions-first layout, prefix cached per question set) is honoured exactly as before: on when
    decider_config.json has "schema_first": true, or with DECIDER_SCHEMA_CACHE=1 on a model trained for that layout.  It is
    the one path that does GPU work after start-up that is not a graph replay: the first request with a new schema runs its
    prefix and captures one graph per (schema, batch bucket, state bucket) as in 1.0.x (/stats -> schema_cache.captures).
    Its requests are bounded and admitted like the others, on prefix plus suffix tokens, before any GPU preparation.
  * requests are bounded before they reach the GPU: HTTP 413 when a row, the request or the expanded question count exceeds the
    limits below, HTTP 503 when the outstanding work exceeds DECIDER_MAX_QUEUE_ROWS.

Variables (default):  DECIDER_COMPILE (0)  DECIDER_FP8 (0)  DECIDER_SHARED (1)  DECIDER_SHARED_MIN_TOKENS (768)
  DECIDER_MAX_BATCH (32)  DECIDER_BATCH_WAIT_MS (0; DECIDER_MAX_WAIT_MS is an alias)  DECIDER_MAX_STATE_TOKENS (32768)
  DECIDER_T_BUCKETS  DECIDER_B_BUCKETS  DECIDER_GRAPH_TOKEN_BUDGET (32768)  DECIDER_WARMUP (1)  DECIDER_TOKENIZE_THREADS (8)
  DECIDER_MAX_ROWS (1024)  DECIDER_MAX_ROW_TOKENS (DECIDER_MAX_STATE_TOKENS + 4096)  DECIDER_MAX_REQUEST_TOKENS (1048576)
  DECIDER_MAX_QUEUE_ROWS (4096)  DECIDER_TEMPERATURE  DECIDER_SCHEMA_CACHE (0)  DECIDER_SCHEMA_MIN_SEEN (2)  DECIDER_SCHEMAS
"""
import asyncio, json, os, time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from decider import systemone as S1
from decider.prompt import build, MAX_OPTIONS
from decider.prompt_fast import build_rows, unique_tokens


def _env_int(name, default):
    return int(os.environ.get(name, default))


MODEL = os.environ.get("DECIDER_MODEL", "runs/r3_v2/model")
MAX_BATCH = _env_int("DECIDER_MAX_BATCH", 32)
BATCH_WAIT_MS = float(os.environ.get("DECIDER_BATCH_WAIT_MS", os.environ.get("DECIDER_MAX_WAIT_MS", "0")))
MAX_STATE_TOKENS = _env_int("DECIDER_MAX_STATE_TOKENS", 32768)
SHARED = os.environ.get("DECIDER_SHARED", "1") == "1"
SHARED_MIN_TOKENS = _env_int("DECIDER_SHARED_MIN_TOKENS", 768)
COMPILE = os.environ.get("DECIDER_COMPILE", "0") == "1"
FP8 = os.environ.get("DECIDER_FP8", "0") == "1"
WARMUP = os.environ.get("DECIDER_WARMUP", "1") == "1"
TOKENIZE_THREADS = _env_int("DECIDER_TOKENIZE_THREADS", 8)
GRAPH_TOKEN_BUDGET = _env_int("DECIDER_GRAPH_TOKEN_BUDGET", 32768)
# request bounds, checked after tokenisation and before anything is queued
MAX_ROWS = _env_int("DECIDER_MAX_ROWS", 1024)                               # scoring rows per request (questions, expanded score levels)
MAX_ROW_TOKENS = _env_int("DECIDER_MAX_ROW_TOKENS", MAX_STATE_TOKENS + 4096)  # tokens in one row: truncated state + question block
MAX_REQUEST_TOKENS = _env_int("DECIDER_MAX_REQUEST_TOKENS", 1 << 20)       # sum of row lengths of one request
MAX_QUEUE_ROWS = _env_int("DECIDER_MAX_QUEUE_ROWS", 4096)                   # rows admitted and not yet scored, over all requests
DECIDE_MAX_CTX_TOKENS = 1536                                                # /decide context cap, unchanged from 1.0.x

MODEL_NAME = "decider"; TEMP = 1.0; TEMP_SCHEMA = 1.0; RELEASE_DATE = "2026-09-17"; ISOLATED = False; NEUTRALIZE_NONE = True
SCHEMA_FIRST = False; se = None; squeue = None; schemas = {}; seen = {}
eng = None; queue = None; gpu = None; cpu = None; batcher_task = None; schema_task = None
outstanding = 0
stats = dict(requests=0, batches=0, decisions=0, rows=0, shared_prefix_requests=0, errors=0, rejected_too_large=0,
             rejected_overloaded=0, batch_hist={}, bucket_hist={})


def _ints(name):
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else None


def load_config(path):
    """decider_config.json from a model folder or a Hub repository (the way decider.infer.Decider resolves it)."""
    try:
        if os.path.isdir(path):
            return json.load(open(os.path.join(path, "decider_config.json")))
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(path, "decider_config.json")))
    except Exception:
        return {}


# ---- request preparation (CPU) -------------------------------------------
class _NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def prepare(tok, state, questions, independent, isolated=False, max_state_tokens=32768):
    """Render, plan the rows, tokenize the state once.  -> (rqs, index, items, ctx_len).  The rows are those
    `prompt.build` produces for the same request (tests/test_prompt_fast.py, tests/test_serve_prepare.py)."""
    ctx = S1.render_state(state)
    rqs = {k: S1.render_question(v) for k, v in questions.items()}
    flat, index = S1.plan_rows(rqs, isolated and independent)
    pairs = [(r["question"], list(r["options"])) for r in flat]
    rows = [[p] for p in pairs] if independent else [pairs]
    items, ctx_len = build_rows(tok, ctx, rows, max_ctx_tokens=max_state_tokens)
    return rqs, index, items, ctx_len


def _prepare_s1(state, questions, independent):
    return prepare(eng.tok, state, questions, independent, ISOLATED, MAX_STATE_TOKENS)


def _prepare_decide(context, schema):
    from decider.infer import Decider, Example, Q, neutralize_options
    qs = Decider._schema_to_questions(schema)
    for q in qs:
        if NEUTRALIZE_NONE:
            q["options"], q["_back"] = neutralize_options(q["options"])
    ex = Example(context, [Q(q["question"], list(q["options"]), 0) for q in qs])
    it = build(ex, eng.tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=DECIDE_MAX_CTX_TOKENS)
    return qs, it


def _format_decide(schema, qs, probs):
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


# ---- limits ---------------------------------------------------------------
def check_size(row_lengths, max_rows=None, max_row_tokens=None, max_request_tokens=None):
    """Raise HTTPException(413) when a request exceeds the row count, per-row token or total token limit."""
    max_rows = MAX_ROWS if max_rows is None else max_rows
    max_row_tokens = MAX_ROW_TOKENS if max_row_tokens is None else max_row_tokens
    max_request_tokens = MAX_REQUEST_TOKENS if max_request_tokens is None else max_request_tokens
    n, total, longest = len(row_lengths), sum(row_lengths), max(row_lengths, default=0)
    if n > max_rows:
        msg = f"too many questions: the request expands to {n} scoring rows, the limit is {max_rows} (DECIDER_MAX_ROWS)"
    elif longest > max_row_tokens:
        msg = f"too many tokens: one row has {longest} tokens, the limit is {max_row_tokens} per row (DECIDER_MAX_ROW_TOKENS)"
    elif total > max_request_tokens:
        msg = f"too many tokens: the request has {total} tokens over {n} rows, the limit is {max_request_tokens} (DECIDER_MAX_REQUEST_TOKENS)"
    else:
        return
    stats["rejected_too_large"] += 1
    raise HTTPException(413, msg)


def _admit(n):
    """Reserve n rows of outstanding work or raise HTTPException(503)."""
    global outstanding
    if outstanding + n > MAX_QUEUE_ROWS:
        stats["rejected_overloaded"] += 1
        raise HTTPException(503, f"server busy: {outstanding} rows queued, the limit is {MAX_QUEUE_ROWS} (DECIDER_MAX_QUEUE_ROWS); retry later")
    outstanding += n


def _release(n):
    global outstanding
    outstanding -= n


# ---- GPU work ----------------------------------------------------------------
def _score_items(items):
    return eng.score_items(items, temperature=TEMP)


def _score_shared(items):
    return eng.score_shared(items, temperature=TEMP)


async def _collect(q):
    """Take what is already queued and go; DECIDER_BATCH_WAIT_MS > 0 adds a collection window."""
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
    """One forward per (length bucket, chunk): rows queued at the same moment share a replay when their bucket matches."""
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect(queue)
        groups = {}
        for fut, it in batch:
            groups.setdefault(eng.pad_len(len(it["ids"])), []).append((fut, it))
        for T, g in groups.items():
            cap = max(1, min(MAX_BATCH, eng.max_rows(T)))
            stats["bucket_hist"][T] = stats["bucket_hist"].get(T, 0) + len(g)
            for i in range(0, len(g), cap):
                part = g[i:i + cap]
                try:
                    probs = await loop.run_in_executor(gpu, _score_items, [it for _, it in part])
                    for (fut, _), p in zip(part, probs):
                        if not fut.done(): fut.set_result(p)
                except Exception as e:
                    for fut, _ in part:
                        if not fut.done(): fut.set_exception(e)
                stats["batches"] += 1
                stats["batch_hist"][len(part)] = stats["batch_hist"].get(len(part), 0) + 1


# ---- schema cache (questions-first layout; opt-in) --------------------------------
def _schema_key(questions, independent):
    return (json.dumps(questions, sort_keys=True, ensure_ascii=False), independent)


class _SQ:
    def __init__(self, text, options): self.text, self.options = text, options


def _plan_schema(questions, independent, state):
    """CPU part of a schema-cache request: prefix token lengths (from the cached handle when the schema is known, otherwise
    tokenised here, without touching the GPU) and the suffix row (context plus answer slots).  -> (tps, (suffix ids, slots)).
    Raises ValueError for an invalid question."""
    from decider.prompt import schema_prefix_ids, schema_suffix_ids
    rqs = {k: S1.render_question(v) for k, v in questions.items()}; rows, _ = S1.plan_rows(rqs, ISOLATED and independent)
    cached = schemas.get(_schema_key(questions, independent))
    if cached is not None:
        tps = list(cached[1].tps)
    else:
        qs = [_SQ(x["question"], list(x["options"])) for x in rows]
        tps = [len(schema_prefix_ids(eng.tok, g)) for g in ([[q] for q in qs] if independent else [qs])]
    row = schema_suffix_ids(eng.tok, S1.render_state(state), 1 if independent else len(rows), MAX_STATE_TOKENS)
    return tps, row


def _schema_handle(questions, independent, compile=False):
    """Compile (or look up) the question schema: its prefix is run once, requests then only run the state.  GPU thread."""
    key = _schema_key(questions, independent)
    if key not in schemas:
        rqs = {k: S1.render_question(v) for k, v in questions.items()}; rows, index = S1.plan_rows(rqs, ISOLATED and independent)
        if len(schemas) >= 128:
            old = next(iter(schemas)); hid = schemas.pop(old)[1].id
            for k in [k for k in se.graphs if k[0] == hid]: del se.graphs[k]
        h = se.prepare(rows, independent=independent, compile=compile)
        schemas[key] = (rqs, h, index)
    return schemas[key]


def _worth_caching(questions, independent):
    """A schema gets a cached prefix and CUDA graphs from its second request on."""
    key = _schema_key(questions, independent)
    if key in schemas: return True
    if len(seen) > 50000: seen.clear()
    seen[key] = seen.get(key, 0) + 1
    return seen[key] >= _env_int("DECIDER_SCHEMA_MIN_SEEN", 2)


def _score_schema(h, rows):
    return se.score_rows(h, rows, temperature=TEMP_SCHEMA)


async def schema_batcher():
    loop = asyncio.get_running_loop()
    while True:
        batch = await _collect(squeue)
        groups = {}
        for fut, h, row in batch: groups.setdefault((h.id, se.bucket(len(row[0]))), (h, []))[1].append((fut, row))
        for h, items in groups.values():
            step = max(1, MAX_BATCH // h.P)
            for i in range(0, len(items), step):
                chunk = items[i:i + step]
                try:
                    probs = await loop.run_in_executor(gpu, _score_schema, h, [c for _, c in chunk])
                    for (fut, _), p in zip(chunk, probs):
                        if not fut.done(): fut.set_result(p)
                except Exception as e:
                    for fut, _ in chunk:
                        if not fut.done(): fut.set_exception(e)
            stats["schema_batches"] = stats.get("schema_batches", 0) + 1


# ---- start-up / shutdown -------------------------------------------------------
def apply_config(cfg):
    global MODEL_NAME, TEMP, TEMP_SCHEMA, RELEASE_DATE, ISOLATED, NEUTRALIZE_NONE, SCHEMA_FIRST
    NEUTRALIZE_NONE = bool(cfg.get("neutralize_none", True))
    MODEL_NAME = "decider-" + str(cfg.get("version", "dev"))
    TEMP = float(os.environ.get("DECIDER_TEMPERATURE", cfg.get("temperature", 1.0)))
    TEMP_SCHEMA = float(cfg.get("temperature_schema_first", TEMP))
    RELEASE_DATE = str(cfg.get("release_date", RELEASE_DATE))
    ISOLATED = bool(cfg.get("isolated_levels", False))
    trained = bool(cfg.get("schema_first", False) or cfg.get("schema_first_trained", False))
    SCHEMA_FIRST = trained and (bool(cfg.get("schema_first", False)) or os.environ.get("DECIDER_SCHEMA_CACHE", "0") == "1")


def start_workers(loop=None):
    """Queues, executors and batcher tasks.  Needs `eng` set; a test can set a stand-in engine and call this directly."""
    global queue, gpu, cpu, batcher_task, squeue, schema_task
    loop = loop or asyncio.get_running_loop()
    if gpu is None: gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
    if cpu is None: cpu = ThreadPoolExecutor(max_workers=TOKENIZE_THREADS, thread_name_prefix="tok")
    queue = asyncio.Queue(); batcher_task = loop.create_task(batcher())
    if SCHEMA_FIRST:
        squeue = asyncio.Queue(); schema_task = loop.create_task(schema_batcher())


async def _start():
    global eng, se, gpu
    from decider.engine_v2 import EngineV2
    apply_config(load_config(MODEL))
    gpu = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
    loop = asyncio.get_running_loop()
    eng = await loop.run_in_executor(gpu, lambda: EngineV2(
        MODEL, compile=COMPILE, fp8=FP8, max_ctx_tokens=MAX_STATE_TOKENS, t_buckets=_ints("DECIDER_T_BUCKETS"),
        b_buckets=_ints("DECIDER_B_BUCKETS"), token_budget=GRAPH_TOKEN_BUDGET))
    print("[serve] engine", {k: v for k, v in eng.cfg.items() if k not in ("t_buckets", "b_buckets")}, flush=True)
    if SCHEMA_FIRST:
        from decider.schema_engine import SchemaEngine
        se = SchemaEngine(eng); print("[serve] schema cache on", flush=True)
        pre = os.environ.get("DECIDER_SCHEMAS")      # JSON: [{"questions": {...}, "independent": true, "batch_sizes": [1, 8, 32], "state_tokens": [64, 256]}]
        for spec in (json.load(open(pre)) if pre else []):
            _, h, _ = await loop.run_in_executor(gpu, _schema_handle, spec["questions"], spec.get("independent", True), COMPILE)
            t = await loop.run_in_executor(gpu, se.warmup, h, spec.get("batch_sizes", (1, 8, 32)), spec.get("state_tokens", (64, 128, 256)))
            print(f"[serve] preloaded schema with {h.nq} rows, prefix {sum(h.tps)} tokens, graphs ready in {t:.0f}s", flush=True)
    if WARMUP:
        t = await loop.run_in_executor(gpu, lambda: eng.warmup(log=lambda s: print(s, flush=True)))
        print(f"[serve] captured {len(eng.graphs)} graphs in {t:.0f}s", flush=True)
    eng.seal()
    print("[serve] ready", json.dumps(dict(model=MODEL_NAME, temperature=TEMP, isolated_levels=ISOLATED, schema_first=SCHEMA_FIRST,
                                          shared=SHARED, graphs=len(eng.graphs), limits=dict(
                                              max_rows=MAX_ROWS, max_row_tokens=MAX_ROW_TOKENS, max_request_tokens=MAX_REQUEST_TOKENS,
                                              max_queue_rows=MAX_QUEUE_ROWS))), flush=True)
    start_workers(loop)


def _stop():
    global gpu, cpu, batcher_task, schema_task
    for t in (batcher_task, schema_task):
        if t is not None: t.cancel()
    for ex in (gpu, cpu):
        if ex is not None: ex.shutdown(wait=False, cancel_futures=True)
    gpu = cpu = batcher_task = schema_task = None


@asynccontextmanager
async def lifespan(app):
    await _start()
    yield
    _stop()


app = FastAPI(title="decider", lifespan=lifespan)


# ---- routes -------------------------------------------------------------------
class Req(BaseModel):
    context: str
    schema_: dict = None
    model_config = {"populate_by_name": True}
    def __init__(self, **kw):
        if "schema" in kw: kw["schema_"] = kw.pop("schema")
        super().__init__(**kw)


class S1Req(BaseModel):
    state: object
    questions: dict
    model: str | None = None
    independent: bool = True
    layout: str | None = None            # "state_first" forces the uncached layout on a schema-first model


def _alive():
    return batcher_task is not None and not batcher_task.done() and (schema_task is None or not schema_task.done())


async def _queued(items):
    loop = asyncio.get_running_loop(); futs = []
    for it in items:
        f = loop.create_future(); futs.append(f); queue.put_nowait((f, it))
    return await asyncio.gather(*futs)


@app.post("/decide")
async def decide(r: Req):
    loop = asyncio.get_running_loop()
    try:
        qs, it = await loop.run_in_executor(cpu, _prepare_decide, r.context, r.schema_)
    except (ValueError, KeyError) as e:
        stats["errors"] += 1
        raise HTTPException(422, str(e))
    check_size([len(it["ids"])])
    _admit(1)
    try:
        probs = (await _queued([it]))[0]
    except Exception:
        stats["errors"] += 1; raise
    finally:
        _release(1)
    stats["requests"] += 1; stats["decisions"] += len(qs); stats["rows"] += 1
    return _format_decide(r.schema_, qs, probs)


@app.post("/v1/systemone")
async def systemone(r: S1Req):
    loop = asyncio.get_running_loop()
    if SCHEMA_FIRST and r.questions and r.layout != "state_first" and _worth_caching(r.questions, r.independent):
        try:                                       # CPU: rows, prefix lengths, suffix ids -> the complete cost before any GPU work
            tps, row = await loop.run_in_executor(cpu, _plan_schema, r.questions, r.independent, r.state)
        except ValueError as e:
            stats["errors"] += 1
            raise HTTPException(422, str(e))
        check_size([tp + len(row[0]) for tp in tps])
        _admit(len(tps))
        try:
            rqs, h, index = await loop.run_in_executor(gpu, _schema_handle, r.questions, r.independent)
            fut = loop.create_future(); await squeue.put((fut, h, row)); p = await fut
        except Exception:
            stats["errors"] += 1; raise
        finally:
            _release(len(tps))
        stats["requests"] += 1; stats["decisions"] += len(rqs); stats["schema_requests"] = stats.get("schema_requests", 0) + 1
        return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, [pk.tolist() for pk in p]),
                "usage": {"input_tokens": len(row[0]) * h.P, "cached_tokens": sum(h.tps), "output_tokens": 0}}
    try:
        rqs, index, items, ctx_len = await loop.run_in_executor(cpu, _prepare_s1, r.state, r.questions, r.independent)
    except ValueError as e:
        stats["errors"] += 1
        raise HTTPException(422, str(e))
    check_size([len(it["ids"]) for it in items])
    _admit(len(items))
    try:
        if SHARED and len(items) > 1 and min(len(it["ids"]) for it in items) >= SHARED_MIN_TOKENS:
            res = await loop.run_in_executor(gpu, _score_shared, items)      # long state: run it once, fork the cache per question
            stats["shared_prefix_requests"] += 1
        else:
            res = await _queued(items)
    except Exception:
        stats["errors"] += 1; raise
    finally:
        _release(len(items))
    probs = [p for ps in res for p in ps]                                   # one prob row per question, request order
    stats["requests"] += 1; stats["decisions"] += len(rqs); stats["rows"] += len(items)
    return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, [p.tolist() for p in probs]),
            "usage": {"input_tokens": unique_tokens(items, ctx_len), "output_tokens": 0}}


@app.get("/v1/models")
async def models():
    return {"models": [{"name": MODEL_NAME, "description": "decider: one-pass typed decisions with calibrated probabilities", "release_date": RELEASE_DATE}]}


@app.get("/health")
async def health():
    return {"ok": eng is not None and bool(getattr(eng, "sealed", True)) and _alive(), "model": MODEL}


@app.get("/stats")
async def get_stats():
    return dict(stats, outstanding_rows=outstanding, engine=eng.stats if eng else None, graphs=len(eng.graphs) if eng else 0,
                sealed=bool(eng and getattr(eng, "sealed", False)), schema_cache=dict(se.stats, schemas=len(schemas)) if se else None,
                limits=dict(max_rows=MAX_ROWS, max_row_tokens=MAX_ROW_TOKENS, max_request_tokens=MAX_REQUEST_TOKENS, max_queue_rows=MAX_QUEUE_ROWS))
