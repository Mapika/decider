"""Micro-batching HTTP server.  POST /decide {"context": str, "schema": {...}} -> typed JSON decisions.
Requests arriving within `max_wait_ms` are scored in one forward pass (grouped by length bucket).
   uvicorn decider.serve:app --host 0.0.0.0 --port 8000     (env: DECIDER_MODEL, DECIDER_MAX_BATCH, DECIDER_MAX_WAIT_MS)
"""
import asyncio, os, random, time, threading
from fastapi import FastAPI
from pydantic import BaseModel
from .engine import Engine, T_BUCKETS, _bucket
from .prompt import build
from .infer import Decider, Example, Q

MODEL = os.environ.get("DECIDER_MODEL", "runs/r3_v2/model")
MAX_BATCH = int(os.environ.get("DECIDER_MAX_BATCH", "32"))
MAX_WAIT_MS = float(os.environ.get("DECIDER_MAX_WAIT_MS", "8"))
COMPILE = os.environ.get("DECIDER_COMPILE", "1") == "1"
FP8 = os.environ.get("DECIDER_FP8", "0") == "1"
app = FastAPI(title="decider")
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
    ex = Example(context, [Q(q["question"], list(q["options"]), 0) for q in qs])
    it = build(ex, eng.tok, _NoShuffle(), max_ctx_tokens=eng.max_ctx)
    return qs, it


def _format(schema, qs, probs):
    o = {}
    for (qtext, spec), q, p in zip(schema.items(), qs, probs):
        p = p[:len(q["options"])].tolist(); t = spec.get("type", "choice"); j = max(range(len(p)), key=p.__getitem__)
        if t == "bool":
            o[qtext] = {"noul": round(p[1], 4), "type": "noul"}
        elif t == "choice":
            o[qtext] = {"choice": q["options"][j], "confidence": round(p[j], 4), "type": "choice",
                        "probabilities": {k: round(v, 4) for k, v in zip(q["options"], p)}}
        else:
            keys = q["_keys"]; score = sum(float(k) * pi for k, pi in zip(keys, p))
            o[qtext] = {"score": round(score, 2), "confidence": round(p[j], 4), "type": "scale", "legend": q["_legend"],
                        "probabilities": {str(keys[i]): round(pi, 4) for i, pi in enumerate(p)}}
    return o


async def batcher():
    loop = asyncio.get_running_loop()
    while True:
        first = await queue.get(); batch = [first]; t0 = time.monotonic()
        # adaptive window: wait briefly; only keep waiting (up to MAX_WAIT_MS) while more requests keep arriving
        deadline = time.monotonic() + min(MAX_WAIT_MS, 1.5) / 1000
        while len(batch) < MAX_BATCH:
            timeout = deadline - time.monotonic()
            if timeout <= 0: break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout))
                deadline = min(time.monotonic() + 2.0 / 1000, t0 + MAX_WAIT_MS / 1000)
            except asyncio.TimeoutError:
                break
        # sort by length; split into at most two groups when the spread is large (keeps padding small)
        batch.sort(key=lambda x: len(x[2]["ids"]))
        groups = [batch]
        if len(batch) >= 4:
            lo, hi = len(batch[0][2]["ids"]), len(batch[-1][2]["ids"])
            if _bucket(hi, T_BUCKETS) != _bucket(lo, T_BUCKETS) and hi > 1.5 * lo:
                cut = len(batch) // 2; groups = [batch[:cut], batch[cut:]]
        for g in groups:
            items = [it for _, _, it in g]
            try:
                probs = await loop.run_in_executor(None, eng.score_items, items)
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


@app.get("/health")
async def health():
    return {"ok": eng is not None, "model": MODEL}


@app.get("/stats")
async def get_stats():
    return dict(stats, engine=eng.stats if eng else None, graphs=len(eng.graphs) if eng else 0)
