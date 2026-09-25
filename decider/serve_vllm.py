"""HTTP server on vLLM for large stock or chat-layout models: the /v1/systemone readout of decider.serve, run by vLLM.

   (install: see "Install" below)
   DECIDER_MODEL=Qwen/Qwen3.6-27B DECIDER_LAYOUT=chat DECIDER_TEMPERATURE=1.943 \
       uvicorn decider.serve_vllm:app --host 0.0.0.0 --port 8000

What is the same as decider.serve.  The prompt rows are built by the same code (decider.serve.prepare: decider.prompt_fast
rows, the chat template of decider.prompt.ChatTemplate in the chat layout, thinking off, options in request order, the state
cut at DECIDER_MAX_STATE_TOKENS), and every row ends at its answer slot, the " (" token of "Answer: (".  The answer
distribution is softmax(letter logits / T) over the question's option letters.  vLLM returns log-softmax values over the
whole vocabulary for the requested letter ids (`logprob_token_ids`, raw logprobs, before any sampling transform); a
log-softmax differs from the logit by one constant per position, so softmax(logprob / T) over the letters equals
softmax(logit / T).  Temperatures come from decider_config.json ("temperature", "temperature_by_type") or
DECIDER_TEMPERATURE, as in decider.serve (decider.temperature).

What vLLM changes.  vLLM batches every row of a request (and of concurrent requests) into its scheduler steps with its own
CUDA graphs and kernels, and reuses shared prefixes through its prefix cache.  For the hybrid Qwen3.5-family models the
recurrent state is cached only at cache-block boundaries (mamba cache mode "align"), so a row that shares a prefix with an
earlier one recomputes from the last block boundary inside the shared part.  When a request has several rows over a shared
prefix that holds at least one whole cache block (and at least DECIDER_VLLM_PREFILL_FIRST_TOKENS tokens), the first row is run
alone so that the others find the prefix in the cache; otherwise all rows are submitted together in one scheduler step.  Numbers differ from decider.serve by kernel round-off (different
attention, GEMM and linear-attention kernels, and a prefix cache for the recurrent layers), not by the prompt or the readout.

Rows with more than 128 options: vLLM caps `logprob_token_ids` at 128 ids per request (vllm.sampling_params.
MAX_LOGPROB_TOKEN_IDS).  With DECIDER_VLLM_WIDE_LOGPROBS=1 (default) the cap is raised to 256 in this process and in the
workers (decider.vllm_worker), so every question (at most 255 options) is one request.  When a worker still reports 128,
a wider row is sent as ceil(n / cap) requests with the same prompt, one per slice of the letter ids, one after the other
(the later ones reuse the cached prefix), and the log-softmax values are joined before the softmax (they share the
normaliser, so this is exact).

Variables (default):  DECIDER_MODEL  DECIDER_LAYOUT (from decider_config.json; "chat" or "plain" overrides it, for a stock
  model without a decider_config.json)  DECIDER_TEMPERATURE  DECIDER_MAX_STATE_TOKENS (32768)  DECIDER_VLLM_MAX_MODEL_LEN
  (40960)  DECIDER_VLLM_GPU_MEMORY_UTILIZATION (0.90)  DECIDER_VLLM_KV_CACHE_DTYPE (auto)  DECIDER_VLLM_MAX_NUM_SEQS (128)
  DECIDER_VLLM_MAX_BATCHED_TOKENS (16384)  DECIDER_VLLM_PREFILL_FIRST_TOKENS (512)  DECIDER_VLLM_QUANTIZATION (from the
  checkpoint)  DECIDER_VLLM_ENFORCE_EAGER (0)  DECIDER_VLLM_WIDE_LOGPROBS (1)  DECIDER_MAX_ROWS (1024)  DECIDER_MAX_ROW_TOKENS
  (DECIDER_MAX_STATE_TOKENS + 4096)  DECIDER_MAX_REQUEST_TOKENS (1048576)  DECIDER_MAX_QUEUE_ROWS (4096)  DECIDER_TOKENIZE_THREADS (8)

Install.  vLLM 0.29.0 needs numpy 2 and pins its own torch, while decider-ai's base requirements pin numpy < 2, so the two are
not installed as one set of requirements.  Use a separate environment:
   pip install vllm==0.29.0 fastapi "uvicorn[standard]" jinja2 huggingface_hub && pip install --no-deps decider-ai
Only /v1/systemone (independent questions), /v1/models, /health and /stats are served; /decide and the schema cache are
decider.serve features.
"""
import asyncio, json, math, os, time, uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from decider import systemone as S1
from decider import temperature as TT
from decider.prompt import MAX_OPTIONS, resolve_layout, chat_template, label_table, with_layout
from decider.prompt_fast import unique_tokens


def _env_int(name, default):
    return int(os.environ.get(name, default))


MODEL = os.environ.get("DECIDER_MODEL", "Qwen/Qwen3.6-27B")
LAYOUT_ENV = os.environ.get("DECIDER_LAYOUT") or None
MAX_STATE_TOKENS = _env_int("DECIDER_MAX_STATE_TOKENS", 32768)
MAX_MODEL_LEN = _env_int("DECIDER_VLLM_MAX_MODEL_LEN", 40960)
GPU_MEM = float(os.environ.get("DECIDER_VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
KV_DTYPE = os.environ.get("DECIDER_VLLM_KV_CACHE_DTYPE", "auto")
MAX_NUM_SEQS = _env_int("DECIDER_VLLM_MAX_NUM_SEQS", 128)
MAX_BATCHED_TOKENS = _env_int("DECIDER_VLLM_MAX_BATCHED_TOKENS", 16384)
PREFILL_FIRST = _env_int("DECIDER_VLLM_PREFILL_FIRST_TOKENS", 512)
QUANT = os.environ.get("DECIDER_VLLM_QUANTIZATION") or None
ENFORCE_EAGER = os.environ.get("DECIDER_VLLM_ENFORCE_EAGER", "0") == "1"
MAX_ROWS = _env_int("DECIDER_MAX_ROWS", 1024)
MAX_ROW_TOKENS = _env_int("DECIDER_MAX_ROW_TOKENS", MAX_STATE_TOKENS + 4096)
MAX_REQUEST_TOKENS = _env_int("DECIDER_MAX_REQUEST_TOKENS", 1 << 20)
MAX_QUEUE_ROWS = _env_int("DECIDER_MAX_QUEUE_ROWS", 4096)
WIDE_LOGPROBS = os.environ.get("DECIDER_VLLM_WIDE_LOGPROBS", "1") == "1"
LOGPROB_IDS_CAP = 128                      # letter ids per vLLM request; raised at start-up when every worker allows more

MODEL_NAME = "decider"; TEMP = 1.0; TEMP_BY_TYPE = {}; LAYOUT = "plain"; CHAT = None; ISOLATED = False
engine = None; tok = None; LETTER_IDS = None; outstanding = 0; BLOCK = 16; cpu = None
stats = dict(requests=0, decisions=0, rows=0, vllm_requests=0, prefill_first=0, errors=0, rejected_too_large=0,
             rejected_overloaded=0)


def load_config(path):
    from decider.serve import load_config as lc
    return lc(path)


def apply_config(cfg):
    """Layout, temperatures and name from decider_config.json, with DECIDER_LAYOUT / DECIDER_TEMPERATURE overrides."""
    global MODEL_NAME, TEMP, TEMP_BY_TYPE, LAYOUT, ISOLATED
    cfg = with_layout(cfg, LAYOUT_ENV)
    LAYOUT = resolve_layout(cfg)
    (TEMP, TEMP_BY_TYPE), _ = TT.from_config(cfg, os.environ.get("DECIDER_TEMPERATURE"))
    MODEL_NAME = "decider-" + str(cfg.get("version", "dev"))
    ISOLATED = bool(cfg.get("isolated_levels", False))


def _engine_args():
    from vllm import AsyncEngineArgs
    kw = dict(model=MODEL, tokenizer=MODEL, dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEM,
              enable_prefix_caching=True, max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=MAX_BATCHED_TOKENS,
              kv_cache_dtype=KV_DTYPE, enforce_eager=ENFORCE_EAGER, language_model_only=True, disable_log_stats=True,
              worker_extension_cls="decider.vllm_worker.WorkerExtension", seed=0)
    if QUANT:
        kw["quantization"] = QUANT
    return AsyncEngineArgs(**kw)


def softmax_T(lp, T):
    m = max(lp); e = [math.exp((x - m) / T) for x in lp]; s = sum(e)
    return [x / s for x in e]


async def _logprobs(ids, n):
    """Log-softmax values of the first n letter ids at the position after `ids` (the answer slot)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    parts = [LETTER_IDS[i:i + LOGPROB_IDS_CAP] for i in range(0, n, LOGPROB_IDS_CAP)]
    parts[-1] = parts[-1][:n - LOGPROB_IDS_CAP * (len(parts) - 1)]

    async def one(tids):
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(tids), detokenize=False)
        stats["vllm_requests"] += 1
        final = None
        async for out in engine.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=uuid.uuid4().hex):
            final = out
        lp = final.outputs[0].logprobs[0]
        return [lp[t].logprob for t in tids]
    out = []
    for p in parts:                         # one after the other: the later slices find the prompt in the prefix cache
        out += await one(p)
    return out


async def score_rows(items, lcp=None):
    """-> one probability list per row, request order.  Every row has one slot, its last token (independent rows).
    lcp: the rows' common prefix length when the caller has it (computed on the CPU pool in _prepare_s1), else computed here.
    The rows run as tasks of one TaskGroup: when one fails or the request is cancelled, the others are cancelled and
    awaited (vLLM aborts a cancelled generate) before this returns or raises."""
    temps = TT.for_items(TEMP, TEMP_BY_TYPE, items)
    jobs = []
    for i, it in enumerate(items):
        if it["slots"] != [len(it["ids"]) - 1]:
            raise ValueError("serve_vllm scores rows with one answer slot at the end (independent questions)")
        T = temps[i][0] if isinstance(temps, list) else temps
        jobs.append((it["ids"], it["nopts"][0], T))

    async def run(job):
        ids, n, T = job
        return [softmax_T(await _logprobs(ids, n), T)]

    async def run_all(js):
        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(run(j)) for j in js]
        return [t.result() for t in tasks]
    if len(jobs) > 1:
        if lcp is None:
            lcp = prefix_len([j[0] for j in jobs])
        if lcp >= max(PREFILL_FIRST, BLOCK):             # at least one whole cache block of the shared prefix can be reused
            stats["prefill_first"] += 1
            first = await run(jobs[0])
            return [first] + await run_all(jobs[1:])
    return await run_all(jobs)


def prefix_len(ids):
    """Common prefix length of the rows (decider.shared_prefix.common_prefix_len, capped one token below the shortest row)."""
    from decider.shared_prefix import common_prefix_len
    return common_prefix_len(ids) if len(ids) > 1 else 0


def _prepare_s1(state, questions, independent):
    """CPU pool: decider.serve.prepare (render, plan, tokenize) plus the rows' common prefix length.
    -> (rqs, index, items, ctx_len, lcp)."""
    from decider.serve import prepare
    rqs, index, items, ctx_len = prepare(tok, state, questions, independent, ISOLATED, MAX_STATE_TOKENS, chat=CHAT)
    return rqs, index, items, ctx_len, prefix_len([it["ids"] for it in items])


async def _warmup():
    """One row per option-count width (the sampler compiles one Triton kernel per padded width of the letter-id list, which
    would otherwise land on the first request of each width), then a batch of rows over a shared prefix.  A warm-up row that
    does not fit the model length (DECIDER_VLLM_MAX_MODEL_LEN) is skipped, and the shared state is shortened until it fits."""
    fits = lambda items: max(len(it["ids"]) for it in items) < MAX_MODEL_LEN
    probe = S1.render_state("warm-up")
    for n in (2, 3, 5, 9, 17, 33, 65, 129, MAX_OPTIONS):
        items = _prepare_s1(probe, {"q": {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(n)}}}, True)[2]
        if fits(items):
            await score_rows(items)
    reps = 3000
    while reps >= 1:
        items = _prepare_s1("x " * reps, {f"q{i}": {"type": "choice", "instructions": f"q {i}", "criteria": {"a": None, "b": None}} for i in range(8)}, True)[2]
        if fits(items):
            await score_rows(items)
            break
        reps //= 2


async def _start():
    global engine, tok, LETTER_IDS, CHAT, BLOCK, LOGPROB_IDS_CAP, cpu
    cpu = ThreadPoolExecutor(max_workers=_env_int("DECIDER_TOKENIZE_THREADS", 8), thread_name_prefix="tok")
    import vllm.sampling_params as VSP
    from vllm.v1.engine.async_llm import AsyncLLM
    from transformers import AutoTokenizer
    apply_config(load_config(MODEL))
    tok = AutoTokenizer.from_pretrained(MODEL)
    LETTER_IDS = label_table(tok)[1]
    CHAT = chat_template(tok) if LAYOUT == "chat" else None
    if WIDE_LOGPROBS:
        from decider.vllm_worker import raise_logprob_cap
        raise_logprob_cap()
    engine = AsyncLLM.from_engine_args(_engine_args())
    LOGPROB_IDS_CAP = min([int(VSP.MAX_LOGPROB_TOKEN_IDS)] + list(await engine.collective_rpc("logprob_ids_cap")))
    sdp = await engine.collective_rpc("cudnn_sdp_enabled")
    BLOCK = max(await engine.collective_rpc("cache_block_size"))
    print("[serve_vllm] worker cuDNN SDPA enabled:", sdp, "cache block size:", BLOCK, "letter ids per request:", LOGPROB_IDS_CAP, flush=True)
    await _warmup()
    print("[serve_vllm] ready", json.dumps(dict(model=MODEL_NAME, path=MODEL, layout=LAYOUT, temperature=TEMP,
                                                temperature_by_type=TT.effective(TEMP, TEMP_BY_TYPE), max_model_len=MAX_MODEL_LEN,
                                                gpu_memory_utilization=GPU_MEM, kv_cache_dtype=KV_DTYPE, quantization=QUANT,
                                                vllm=VSP.__name__ and __import__("vllm").__version__)), flush=True)


@asynccontextmanager
async def lifespan(app):
    await _start()
    yield
    if engine is not None:
        engine.shutdown()
    if cpu is not None:
        cpu.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="decider-vllm", lifespan=lifespan)


class S1Req(BaseModel):
    state: object
    questions: dict
    model: str | None = None
    independent: bool = True
    layout: str | None = None


def check_size(lens):
    """HTTP 413 for a request over DECIDER_MAX_ROWS, DECIDER_MAX_ROW_TOKENS, DECIDER_MAX_REQUEST_TOKENS or the engine's maximum
    model length; the messages carry the markers the Decision Index kit reads as a capacity limit ("too many tokens",
    "maximum model length")."""
    n, total, longest = len(lens), sum(lens), max(lens, default=0)
    if n > MAX_ROWS:
        msg = f"too many questions: the request expands to {n} scoring rows, the limit is {MAX_ROWS} (DECIDER_MAX_ROWS)"
    elif longest > MAX_ROW_TOKENS:
        msg = f"too many tokens: one row has {longest} tokens, the limit is {MAX_ROW_TOKENS} per row (DECIDER_MAX_ROW_TOKENS)"
    elif longest >= MAX_MODEL_LEN:
        msg = f"too many tokens: one row has {longest} tokens; the maximum model length is {MAX_MODEL_LEN} (DECIDER_VLLM_MAX_MODEL_LEN)"
    elif total > MAX_REQUEST_TOKENS:
        msg = f"too many tokens: the request has {total} tokens over {n} rows, the limit is {MAX_REQUEST_TOKENS} (DECIDER_MAX_REQUEST_TOKENS)"
    else:
        return
    stats["rejected_too_large"] += 1
    raise HTTPException(413, msg)


async def _disconnected(request):
    """Returns when the client has gone away (the body has been read already, so the next ASGI message is the disconnect)."""
    while True:
        msg = await request.receive()
        if msg.get("type") == "http.disconnect":
            return


async def _release(tasks, n):
    """Cancel the tasks that are still running, wait until all have stopped, then release n admitted rows."""
    global outstanding
    try:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        outstanding -= n


async def _finish(cleanup):
    """Wait for the cleanup task even when this coroutine is cancelled meanwhile; re-raise the cancellation after it is done."""
    cancelled = False
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    cleanup.result()


@app.post("/v1/systemone")
async def systemone(r: S1Req, request: Request):
    global outstanding
    if not r.independent:
        raise HTTPException(422, "serve_vllm answers independent questions only (\"independent\": true)")
    try:                                            # render, tokenize and the prefix scan on a CPU thread, off the event loop
        rqs, index, items, ctx_len, lcp = await asyncio.get_running_loop().run_in_executor(cpu, _prepare_s1, r.state, r.questions, True)
    except ValueError as e:
        stats["errors"] += 1
        raise HTTPException(422, str(e))
    check_size([len(it["ids"]) for it in items])
    if outstanding + len(items) > MAX_QUEUE_ROWS:
        stats["rejected_overloaded"] += 1
        raise HTTPException(503, f"server busy: {outstanding} rows queued, the limit is {MAX_QUEUE_ROWS} (DECIDER_MAX_QUEUE_ROWS); retry later")
    outstanding += len(items)
    work = asyncio.ensure_future(score_rows(items, lcp)) if items else None
    gone = asyncio.ensure_future(_disconnected(request))
    try:
        if work is not None:
            await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
        if work is not None and not work.done():    # the client disconnected: cancelled in the cleanup below
            stats["client_disconnects"] = stats.get("client_disconnects", 0) + 1
            raise HTTPException(499, "client disconnected")
        res = work.result() if work is not None else []
    except HTTPException:
        raise
    except asyncio.CancelledError:                  # the handler itself was cancelled (shutdown, middleware): cleanup below
        raise
    except Exception:
        stats["errors"] += 1; raise
    finally:
        # Unconditional cleanup, also when the handler is cancelled (once or several times, also during this cleanup): a
        # separate task stops the watcher, cancels the rows that are still running (vLLM aborts them), waits for them and only
        # then releases the admission.  The handler waits for that task through any further cancellation and re-raises the
        # cancellation afterwards.
        await _finish(asyncio.ensure_future(_release([t for t in (gone, work) if t is not None], len(items))))
    probs = [p for ps in res for p in ps]
    stats["requests"] += 1; stats["decisions"] += len(rqs); stats["rows"] += len(items)
    return {"model": MODEL_NAME, "answers": S1.assemble(rqs, index, probs),
            "usage": {"input_tokens": unique_tokens(items, ctx_len), "output_tokens": 0}}


@app.get("/v1/models")
async def models():
    return {"models": [{"name": MODEL_NAME, "description": "decider readout on vLLM"}]}


@app.get("/health")
async def health():
    return {"ok": engine is not None, "model": MODEL, "layout": LAYOUT, "temperature": TEMP,
            "temperature_by_type": TT.effective(TEMP, TEMP_BY_TYPE), "engine": "vllm"}


@app.get("/stats")
async def get_stats():
    return dict(stats, outstanding_rows=outstanding)
