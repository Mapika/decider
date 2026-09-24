# Serving: the 1.1.0 HTTP server

`decider.serve` (1.1.0) replaces the 1.0.x server, which is kept for one release as `decider.serve_v1`. This document is
what a maintainer needs: the design, the defaults, the limits, the measurements and how to repeat them. The investigation
that led here is summarised in the appendix.

## 1. Why

The Decision Index (github.com/apolinario/decision-index) measured the 1.0.x server at a median of 50 ms and a p95 of
1,641 ms for decider-2b. The tail came from shape-dependence, not from the question set: the graph grid stopped at 2,048
tokens and was only warmed to 1,536, so the first request at a new `(batch, length)` shape paid a torch.compile and a CUDA
graph capture with the GPU lock held, and rows above 2,048 tokens ran eager at request-specific shapes. Separately, the
cached shared-state path returned wrong answers on some long requests; 1.0.2 traced that to the cuDNN SDPA backend and
turned it off (`decider.engine.set_attention_backend_policy`).

## 2. Design

`decider/engine_v2.py` (`EngineV2`), `decider/prompt_fast.py`, `decider/serve.py`.

* **Graphs keyed on `(batch bucket, length bucket)` only.** Length buckets `64, 128, 192, 256, 320, 384, 512, 640, 768,
  1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192`; batch buckets `1, 2, 4, 8, 16, 32`; a shape is captured when `B == 1`
  or `B * T <= DECIDER_GRAPH_TOKEN_BUDGET` (32,768). That is 89 graphs.
* **The whole grid is captured at start-up, then the engine is sealed.** A sealed engine never captures and never compiles:
  a shape with no graph runs the uncompiled eager forward and is counted in `/stats -> engine.eager_forwards`. A batch wider
  than the widest captured bucket at its length is split into captured buckets. Rows longer than 8,192 tokens run eager in
  chunks of at most `DECIDER_GRAPH_TOKEN_BUDGET` padded tokens (`engine.unbucketed_requests`).
* **Attention backend policy first.** `EngineV2.__init__` calls `set_attention_backend_policy()` (cuDNN SDPA off) before
  any compile or capture; captured graphs keep the backend they were captured with.
* **Shared-state path on.** An independent request with more than one question over a state of at least
  `DECIDER_SHARED_MIN_TOKENS` (768) tokens runs the state once, forks the cache and scores only the question suffixes
  (`EngineV2.score_shared`, the algorithm of `Engine.score_shared`, implemented once in `decider/shared_prefix.py` since
  1.1.1). The fork is made in chunks that fit `DECIDER_SHARED_FORK_GB`, so the peak memory does not grow with the question
  count (section 4.2). It is eager at request-specific shapes and is correct with the backend policy in place (section 5).
* **One GPU thread.** Every forward, including graph capture at start-up, runs on a single-thread executor; tokenisation
  runs on a CPU pool (`DECIDER_TOKENIZE_THREADS`, 8). Rows queued at the same moment are partitioned by
  `decider.batching.plan_batches`, which pads a shorter row into a longer row's bucket when that costs less than a second
  forward (section 4.1); `DECIDER_BATCH_WAIT_MS` (0) adds a collection window, and after a collection that held more than
  one request the next one waits up to `DECIDER_BATCH_ADAPTIVE_WAIT_MS` (2 ms).
* **The state is tokenised once per request.** `prompt.build` encodes `"Context:\n" + state` and appends a separately
  encoded question block, so a row's ids are `ctx_ids + question_piece`; `prompt_fast.build_rows` shares `ctx_ids` across
  the rows. `tests/test_prompt_fast.py` checks id-for-id equality with `prompt.build`.
* **Wire format unchanged.** Same routes, same `answers` (built by `systemone.assemble`), same `usage`, same 422 bodies,
  same field order as 1.0.x. `tests/test_serve_http.py` posts the same requests to `decider.serve` and `decider.serve_v1`
  with one stand-in engine and requires identical bytes. `/decide` is served as before (context cap 1,536 tokens). The
  schema cache (questions-first layout) is honoured as before: on when `decider_config.json` has `schema_first: true`, or
  with `DECIDER_SCHEMA_CACHE=1` on a model trained for that layout. When on, the first request with a new schema runs the
  schema's prefix and captures one graph per (schema, batch bucket, state bucket) at runtime, as in 1.0.x; that is the one
  exception to "no GPU work after start-up that is not a replay", and `/stats -> schema_cache` counts it. Its requests are
  planned and tokenised on CPU (prefix and suffix), checked against the limits and admitted before any GPU preparation.
* **Configuration is read from a Hub id as well as a folder** (`hf_hub_download`), the way `decider.infer.Decider` does.

## 3. Defaults and limits

| variable | default | meaning |
| --- | --- | --- |
| `DECIDER_MODEL` | `runs/r3_v2/model` | model folder or Hub id |
| `DECIDER_DEVICE` | `auto` | `auto` (CUDA, else MPS, else CPU, as `Decider`), `cuda`, `cuda:<i>`, `mps` or `cpu`. Off CUDA: no graphs, no warm-up, every request eager; FP8 and compile refuse to start. On a CPU run of a CUDA machine, uninstall `causal-conv1d` or the Qwen3.5 layers call its CUDA kernel on CPU tensors |
| `DECIDER_COMPILE` | `0` | torch.compile the forward during warm-up (never at runtime) |
| `DECIDER_FP8` | `0` | e4m3 weights with per-token activation scaling |
| `DECIDER_SHARED` | `1` | shared-state path for multi-question requests over long states |
| `DECIDER_SHARED_MIN_TOKENS` | `768` | shortest row length at which the shared path applies |
| `DECIDER_MAX_BATCH` | `32` | rows per dispatch |
| `DECIDER_BATCH_WAIT_MS` | `0` | collection window; `DECIDER_MAX_WAIT_MS` is an alias |
| `DECIDER_BATCH_ADAPTIVE_WAIT_MS` | `2` | extra window after a collection that held more than one request; `0` disables |
| `DECIDER_MERGE_OVERHEAD_TOKENS` | `512` | fixed cost of a forward, in padded tokens, in the batch partition's cost model |
| `DECIDER_SHARED_FORK_GB` | `8` | byte budget for one fork of the shared prefix cache; also capped at half the free memory |
| `DECIDER_MAX_STATE_TOKENS` | `32768` | state truncation |
| `DECIDER_T_BUCKETS`, `DECIDER_B_BUCKETS` | the ladders above | comma-separated |
| `DECIDER_GRAPH_TOKEN_BUDGET` | `32768` | `(B, T)` captured when `B == 1` or `B * T` fits; also the eager chunk size |
| `DECIDER_WARMUP` | `1` | `0` skips capture (everything runs eager) |
| `DECIDER_TOKENIZE_THREADS` | `8` | CPU pool |
| `DECIDER_MAX_ROWS` | `1024` | scoring rows one request may expand to (questions, isolated score levels) |
| `DECIDER_MAX_ROW_TOKENS` | `MAX_STATE_TOKENS + 4096` | tokens in one row: truncated state plus question block |
| `DECIDER_MAX_REQUEST_TOKENS` | `1048576` | sum of row lengths of one request |
| `DECIDER_MAX_QUEUE_ROWS` | `4096` | rows admitted and not yet scored, over all requests |
| `DECIDER_TEMPERATURE` | config `temperature`, else 1.0 | softmax temperature; when set it replaces `temperature` and switches the config's `temperature_by_type` off (1.4.0) |
| `DECIDER_SCHEMA_CACHE`, `DECIDER_SCHEMA_MIN_SEEN`, `DECIDER_SCHEMAS` | `0`, `2`, unset | schema cache, as in 1.0.x |

Responses at the limits, all decided after tokenisation and before anything is queued:

* HTTP 413 `{"detail": "too many questions: ..."}` when the request expands to more than `DECIDER_MAX_ROWS` rows;
  `"too many tokens: one row has N tokens, ..."` when a row exceeds `DECIDER_MAX_ROW_TOKENS`; `"too many tokens: the request
  has N tokens over M rows, ..."` when the sum exceeds `DECIDER_MAX_REQUEST_TOKENS`. Each message names the variable.
* HTTP 503 `{"detail": "server busy: N rows queued, the limit is ..."}` when admitting the request would take the
  outstanding rows over `DECIDER_MAX_QUEUE_ROWS`. Outstanding rows are released when the request finishes, on success or
  failure.
* HTTP 422 `{"detail": <message>}` for a question that fails `systemone.render_question`, as before.

`/health` is `{"ok": true}` only once the grid is captured and the batcher task is alive; the HTTP port does not accept
requests before the lifespan start-up finishes. `/health` and the ready line also report `temperature` and `temperature_by_type`, the temperature every
answer type gets after the fallback to `temperature` (1.4.0; README, "Temperatures in `decider_config.json`"). `/stats` reports `requests`, `decisions`, `rows`, `batches`,
`shared_prefix_requests`, `errors`, `rejected_too_large`, `rejected_overloaded`, `outstanding_rows`, the batch and bucket
histograms, `limits`, the engine counters `graph_captures`, `forwards`, `replays`, `eager_forwards`, `eager_rows`,
`shared_calls`, `unbucketed_requests`, and, when the schema cache is on, `schema_cache` with `prepared` (schemas whose prefix
was run), `captures` (graphs captured for them after start-up), `replays`, `eager` and `schemas` (handles held). `errors`
counts requests that failed after parsing: question validation (422 from `render_question`) and inference failures (500);
Pydantic 422s, 413 and 503 are not in it (413 and 503 have their own counters). After start-up `engine.graph_captures` must
stay at 89 and `eager_forwards` should equal the number of forwards over rows above 8,192 tokens; `schema_cache.captures`
grows by one graph set per new schema, which is the one place the server does GPU work after start-up that is not a replay.

Memory: the 89 graphs reserve about 25 GB on the 4B (graph pool plus `[B, T, 255]` float32 outputs). Nothing checks that
the grid fits before capturing; a smaller card needs a shorter `DECIDER_T_BUCKETS` or a lower budget (a start-up that runs out
of memory fails with `torch.OutOfMemoryError` and exits, it does not hang). Start-up is 27 to 45 s for the grid (section 6).

Prompt layout (1.2.0): the server reads `"layout"` from `decider_config.json` at start-up. `"layout": "chat"` (or
`"chat_template": true`), which the chat-trained research checkpoint decider-2b v11 has (not released), wraps every row in the tokenizer's chat template (`decider.prompt.build_chat`).
This applies to `/decide`, `/v1/systemone`, the shared-prefix fork and the schema cache. A config with no `"layout"` key is
the plain layout of every earlier model; its rows have the same token ids as in 1.1.x. An unknown layout stops start-up with a
`ValueError` that names it. The start-up line and `/health` report `"layout"`. There is no environment variable that
overrides the layout, because a model gives wrong probabilities when it is read in a layout it was not trained on. The chat
layout adds 12 tokens to each Qwen3.5 row (3 before the context and 9 before the first answer piece). `DECIDER_MAX_STATE_TOKENS`
still caps only `Context:\n<state>`, and `DECIDER_MAX_ROW_TOKENS` counts the whole row, template tokens included.
`/decide` keeps its 1,536-token context cap in both layouts. The research server that v11 was evaluated with used the
state cap there instead, so `/decide` answers on longer contexts can differ from it.
`decider.serve_v1` renders only the plain layout and refuses a chat-layout model at start-up.

### 3.1 Temperatures per answer type (1.4.0)

The server reads the temperatures from `decider_config.json` as `Decider` does (README, "Temperatures in
`decider_config.json`"): `temperature`, and optionally `temperature_by_type` `{"choice": T, "noul": T, "score": T}`, where a
missing type uses `temperature`. A `/decide` field of type `choice` is a `choice` answer, `bool` is `noul` and `scale` is
`score`; every row of a batch carries the answer type of each of its slots, so a batch that mixes requests and types still gives
every answer its own temperature. An invalid value or key stops start-up with a `ValueError` before the weights are loaded.
`DECIDER_TEMPERATURE` replaces `temperature` and switches the map off. decider-ai 1.3.0 and earlier ignore the map.

Checked on decider-4b v2.1 (map choice 1.110, noul 1.560, score 1.287) with 380 requests built from validation rows (single
Choice, noul and isolated-level Score questions, mixed requests and `/decide` requests with a choice, a bool and a scale field):
the server with CUDA graphs (257 graph replays during the check, 0 eager forwards, 123 requests through the shared-prefix fork)
was compared with a second server of the same folder at `DECIDER_TEMPERATURE=1`. For Choice and yes/no answers the reference is
softmax(log p₁ / T_type) of that server's probabilities p₁; for an isolated-level Score answer each level's yes/no pair
[1 − f, f], with f the level's `level_fit` at temperature 1, is rescaled with T_score and the yes probabilities are normalised
over the levels. The difference is at most 1.3e-4 on the 196 answers whose temperature-1 values are all at least 0.01 (and, for
level fits, at most 0.99), and 0.0016 over all 540 answers: the responses are rounded to four decimals, and the log of a rounded
probability near 0 or 1 is imprecise. In process, eager and on the graph path, the map was applied to the path's own
logits to 1.9e-7.

Start the server from a directory that does not contain another `decider/` package, or pass `--app-dir` pointing at the
installed package: uvicorn's default `--app-dir .` and `python -m` put the current directory first on `sys.path`, so a checkout
of another version in the working directory is imported instead of the installed one. `PYTHONSAFEPATH=1` keeps `python -m`
from adding the current directory.

## 4. Batching policy and the shared-prefix memory bound (1.1.1)

Two changes to `decider.serve` and the two engines. Neither changes a route, a request field or a response field.

### 4.1 Cross-request batching that merges

Until 1.1.1 the batcher grouped the rows it collected by exact padded length (`EngineV2.pad_len`), so two requests that
arrived together but landed in different length buckets ran as separate forwards. In the 1.1.0 benchmark at concurrency 8
that produced 2,238 batches for 2,000 requests, 1,250 of them a single row.

`decider/batching.py` replaces the grouping with a partition that may pad a shorter row up to a longer row's bucket when
that is cheaper than a second forward. The cost model is `overhead + B * T` token-units for a forward of `B` rows padded
to length `T`: `overhead` is the fixed cost of a forward expressed as the number of padded tokens that take the same
time. Rows are sorted by padded length descending (stable, so rows of the same bucket keep their arrival order) and split
into consecutive groups; a group runs at the bucket of its longest member, so a group starting at position `k` costs
`overhead + g * T[k]`. A dynamic program takes the cheapest split subject to `g <= min(DECIDER_MAX_BATCH,
EngineV2.max_rows(T))`, ties going to the larger group, so `DECIDER_MERGE_OVERHEAD_TOKENS=0` reproduces the 1.1.0
grouping exactly. Rows above the last captured length bucket (8,192 tokens) run eager at a request-specific shape
and are grouped by exact padded length as before, not merged into another row's bucket. The 1.1.0 grouping is one of the
partitions the program may choose, so the planned cost is never above it; `tests/test_batching.py` checks that, plus
coverage, the padded-length bound and the group cap, over 400 random row sets.

`DECIDER_MERGE_OVERHEAD_TOKENS` defaults to 512. The measurement: on decider-2b, `EngineV2.score_items` on a single row at
each captured length bucket (graph replay, slot gather and the device-to-host copy, which is what a forward costs the
server) takes 12.3 ms at 64 tokens and 67.4 ms at 4,096; a least-squares fit over the ladder gives `t(T) = 9.20 ms +
13.80 us/token`, so the fixed cost is worth 667 tokens; a fit over `T <= 1024` alone gives 1,061 tokens. 512 is the nearest
power of two to the full-ladder figure. The same run timed batches of 2 to 32 rows: the measured time is below
`overhead + B * T` at every width (102.9 ms against a modelled 122.3 ms at 32 rows of 256 tokens), because the per-token
cost falls as the batch grows, so the model is conservative about merging rather than optimistic. The card was shared with
a training run, which inflates both terms; the ratio is what the constant uses.

The collection window is unchanged (`DECIDER_BATCH_WAIT_MS`, 0) with one addition: when the previous collection held rows
from more than one request, the next one waits up to `DECIDER_BATCH_ADAPTIVE_WAIT_MS` (2 ms) for more rows; after a
collection that held one request it does not wait. A request's rows carry its id through the queue, which is what the
batcher reads. Two details of the rule:

* Rows whose future is already done are not counted. A cancelled or failed request leaves its rows in the queue, and
  they are not evidence that requests are overlapping.
* The window is dropped again when the queue was idle. If the first row of a collection took more than 50 ms
  (`serve.ADAPTIVE_IDLE_RESET_MS`) to arrive, the burst that earned the window is over, and an isolated request after a
  quiet period pays nothing.

Neither the window nor the merge changes what a row is scored against, only which rows share a forward.

### 4.2 Shared-prefix memory bound

`Engine.score_shared` and `EngineV2.score_shared` ran the common prefix once and then `cache.reorder_cache(zeros(n))`,
which copies the prefix cache to all `n` question rows at once. The cost is `n` times the prefix cache: a 31k-token state
with 32 questions is 133 GB on a 31B model, and it is wasteful on the 2B.

`decider/shared_prefix.py` holds the implementation both engines now call, so the two cannot diverge; `min_prefix` (192)
and the cuDNN SDPA policy are unchanged, and `Engine.score_shared`'s signature is unchanged (`decider2/serve_chat.py`
subclasses it). The prefix cache is forked in chunks: `cache_row_bytes` sums the bytes one row holds over the layers,
`fork_cache` builds a new cache of `m` rows from the un-expanded prefix cache without mutating it, and the loop scores the
rows `m` at a time, each chunk's suffixes padded to that chunk's own longest suffix, dropping the fork before the next
chunk. `m = clamp(budget // prefix_bytes, 1, n)`, the budget being `DECIDER_SHARED_FORK_GB` (8) capped at half of the
memory free on the device at that moment. The cap applies only where the CUDA memory queries succeed: on CPU, on MPS and
when the driver does not know the device string, the configured budget is kept as it is. The lower clamp is one row, so
a request whose single prefix copy is already over the budget still runs -- the budget bounds the fork, it cannot make
it free.

A cache layout the helpers do not recognise is not chunked at all. If a layer holds a tensor, or a dict of tensors,
under a name outside `keys`, `values`, `indexer_keys`, `conv_states`, `recurrent_states`, there is no way to tell
whether it carries a batch dimension, so `score_shared` falls back to one fork of all `n` rows -- the 1.1.0 behaviour,
correct but unbounded -- and counts it in `/stats -> engine.shared_unchunked_layout`. On decider-2b under transformers
5.17 that counter stays at zero: its `DynamicCache` holds `DynamicLayer` and `LinearAttentionLayer` objects whose only
tensors are the enumerated ones.

The helpers do not assume one cache layout: they walk `cache.layers` and take whatever of `keys`, `values`, `conv_states`
and `recurrent_states` is present, as tensors or as dicts of tensors, which is what transformers 5.17 gives for the
Qwen3.5 hybrid layers (attention KV in the same layer object as the delta-net conv and recurrent states). The fork's
tensors are fresh `index_select` allocations, never views: the linear-attention layers write their states back with
`copy_()`, so a fork that shared storage would corrupt the prefix for the next chunk.

Answers. With every suffix the same length, chunking is bit-identical to the single fork at `m = 1, 2, 3` on decider-2b:
the batch size of the suffix forward on its own changes nothing. With suffixes of different lengths a chunk pads to its
own longest suffix instead of the request's, the kernels reduce in a different order, and the answers move: at most
2.9e-5 of probability on a four-row request with 223 to 450-token suffixes and 4.5e-5 on a 32-row request with 223 to
471-token suffixes, argmax unchanged. A request whose suffixes are further apart moves further: 1.5e-2 on the 32-row
synthetic row of the memory table, whose suffixes run from 182 to 932 tokens, still well inside the server's 0.05
tolerance against the eager reference. `tests/test_engine_v2_cuda.py` covers the equal-length case and both
mixed-length cases (4 rows and 32); the 182 to 932-token spread is measured, not asserted.

Peak reserved memory on decider-2b, measured around the call after `empty_cache`, over what was already held:

| request | prefix cache | old (n-row fork) | new, 8 GB budget | new, 1 GB budget | new, m = 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ContractNLI:test:58`, 17 questions, 7,988-token state | 112.5 MB/row | 3.75 GB | 3.84 GB | 1.94 GB | 0.67 GB |
| `BRIGHT-retrieval:robotics:7:chunk0`, 32 questions, 1,489-token state | 36.3 MB/row | 3.77 GB | 3.85 GB | 3.38 GB | 0.15 GB |
| synthetic 16,000-token state, 32 questions | 206.3 MB/row | 15.38 GB | 15.53 GB | 2.17 GB | 1.34 GB |

Reading the table: on decider-2b the default 8 GB budget does not bind on anything in the 4,000-row Decision Index
sample, whose heaviest shared-prefix request forks 1.87 GB; it bounds the case the change is for. The peak is not the fork
alone: the suffix forward's activations scale with the chunk as well, so the observed peak is about 2.4 times the fork
(6.45 GB of fork, 15.4 GB of peak, on the synthetic row). A deployment that wants a hard ceiling should set
`DECIDER_SHARED_FORK_GB` to about 40% of what it can afford. The prefix forward itself is linear and cheap (1.31 GB at
16,000 tokens, 1.99 GB at 24,000) and is not chunked.

## 5. Correctness

* `tests/test_engine_v2_cuda.py` (marker `cuda`, skips without a GPU; `DECIDER_TEST_MODEL` selects the checkpoint) loads a
  real checkpoint, captures one graph, and checks on a synthetic two-row prompt with a 4,000-token shared prefix and
  223/240-token suffixes (the shape class of Decision Index row `RouterBench-5shot:26878`) that the graph path and the
  shared path are finite, have the argmax of the masked eager forward, and are within 0.05 of it. It also checks the exact
  configuration of `decider/bench/probe_cache_split.py` that fails with cuDNN SDPA on (seed-0 random sequence of 4,300
  tokens, splits 3,840 / 3,968 / 4,096): relative error below 0.05 at the last position. Passes on decider-2b and the 4B.
* On the RouterBench row itself (2 questions, 4,229 / 4,246-token rows, prefix 4,006): with the policy the shared path
  answers `option_4` for both questions on both checkpoints, max |dp| against the full forward 1.6e-3 (4B) and 1.2e-2 (2B);
  with cuDNN forced back on it answers options 7/10 (4B) and 2/1 (2B) at high confidence. Log:
  `decider2/serving_v2_fix_evidence/row_shared_check_{2b,4b}.log` in the research notes.
* `tests/test_batching.py` checks the batch partition on CPU: coverage, the padded-length bound, the group cap and
  "never more expensive than the per-bucket grouping", over 400 random row sets. `tests/test_shared_prefix.py` checks the
  cache helpers against stand-in layers of every layout (which tensors are found, the per-row byte count, the chunk size a
  budget gives, that a fork has its own storage and leaves the original alone, that `indexer_keys` is forked and that an
  unrecognised state name turns chunking off) and then runs the whole scoring loop against a cache-aware stand-in
  backbone whose answers depend on the position: per-chunk slot offsets with unequal suffix lengths, multi-question
  rows, the fallbacks that return `None`, the chunk-size clamps and the budget when the CUDA query raises. A fork left
  at one row fails that backbone outright, and a fork sharing storage with the prefix moves its answers by 0.77.
  `tests/test_serve_http.py` checks the queue side: which rows share a forward, that every row gets its own answer back
  across split collections, a cancelled row, a forward that raises, and the adaptive window's rule.
* `decider/bench/verify_engine_v2.py` compares `EngineV2.score_items`, `EngineV2.score_shared` and `Engine.score_items`
  against the masked eager forward over a row file and exits 1 on any argmax mismatch, non-finite output or max |dp| above
  `--tol`. `decider/bench/probe_cache_split.py` sweeps split points of a random sequence and exits 1 on any failure; `--cudnn`
  reproduces the fault.
* Answers are deterministic for a fixed batch shape and not across batch shapes (bf16 reduction order): against the eager
  reference the server differs on 0.1% to 0.16% of argmaxes, all at near-ties (reference top probability at most 0.53),
  with per-answer probability differences up to 0.05 on the 4B and 0.12 on the 2B (section 6). Pin
  `DECIDER_MAX_BATCH=1 DECIDER_B_BUCKETS=1` for bit-reproducible answers.

## 6. Measurements

Setup: one B300 (`CUDA_VISIBLE_DEVICES=2`, no other process on the card during the runs, checked every 30 s), 1,000 rows in
file order from the Decision Index 4,000-row sample (`work/sample-4k.jsonl.gz`; all questions are `choice`, median 1
question per request, p95 64, longest row median 224 tokens, p95 4,399, max 8,822), posted as
`decision_index/engines/http.py` posts them by `decider/bench/replay_systemone.py`, first at concurrency 1 and then at
concurrency 8 against the same freshly started server (`decider/bench/run_serving_matrix.py`). "old" is
`decider.serve_v1` at its defaults (`DECIDER_COMPILE=1 DECIDER_FP8=1`, shared path on), "new" is `decider.serve` at its
defaults. Effective configuration, from the model folders' `decider_config.json` and the responses: decider-2b ran as
`decider-v10`, temperature 1.3, isolated levels on, in both servers (both response files report `"model": "decider-v10"`,
which the old server's folder-only loader produces only when it has read the config; no `DECIDER_TEMPERATURE` was set in the
runner's environment); the 4B folder has no config, so both servers ran as `decider-dev`, temperature 1.0, isolated levels
off. Old: `compile=True, fp8=True, conv_patch=True`, 72 graphs at warm-up; new: `compile=False, fp8=False,
conv_patch=False`, 89 graphs. Latencies are client-side wall time in milliseconds. "eager disagreements" is the number of
answers (of all answers in the 1,000 requests) whose argmax differs from `decider/bench/eager_reference.py`, the masked eager
forward at the same temperature; "max |dp|" is the largest per-answer probability difference against it.

### 5.1 decider-2b (Mapika/decider-2b v10, temperature 1.3, isolated levels), 1,000 requests = 8,247 answers

| server | conc | median | p95 | p99 | mean | max | req/s | errors | start-up | eager disagreements | max \|dp\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old (1.0.x, `serve_v1`) | 1 | **5.8** | 103.4 | 372.7 | 31.7 | 1477.1 | 31.5 | 0 | 114 s | 232 (2.8%) | 0.561 |
| new (1.1.0, `serve`) | 1 | 9.1 | **82.4** | **172.1** | **26.6** | **282.6** | **37.5** | 0 | **27 s** | **9 (0.11%)** | **0.116** |
| old | 8 | 243.3 | 896.8 | 1171.0 | 310.4 | 1513.6 | 25.7 | 0 | 114 s | 220 (2.7%) | 0.561 |
| new | 8 | **136.3** | **442.6** | **592.1** | **186.9** | **775.3** | **42.8** | 0 | **27 s** | **12 (0.15%)** | **0.116** |

### 5.2 decider-4b (`runs/decider_4b_v1/baseline_4b_bf16`, temperature 1.0), 1,000 requests = 8,247 answers

| server | conc | median | p95 | p99 | mean | max | req/s | errors | start-up | eager disagreements | max \|dp\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old (1.0.x, `serve_v1`) | 1 | **8.7** | 203.0 | 775.7 | 57.0 | 3040.9 | 17.6 | 0 | 126 s | 109 (1.3%) | 0.651 |
| new (1.1.0, `serve`) | 1 | 14.7 | **170.5** | **331.4** | **50.8** | **628.4** | **19.7** | 0 | **45 s** | **8 (0.10%)** | **0.053** |
| old | 8 | 483.9 | 1566.9 | 2311.5 | 605.4 | 2622.6 | 13.2 | 0 | 126 s | 117 (1.4%) | 0.651 |
| new | 8 | **258.5** | **914.0** | **1259.2** | **378.6** | **1433.8** | **21.1** | 0 | **45 s** | **13 (0.16%)** | **0.053** |

Reading the tables:

* The new server loses the median (9.1 against 5.8 ms on the 2B, 14.7 against 8.7 ms on the 4B) and wins every tail
  statistic, the mean, the maximum and the throughput at both concurrencies. On the 2B at concurrency 1, p99 172 against
  373 ms and maximum 283 against 1,477 ms; at concurrency 8, p95 443 against 897 ms. This compares the two servers at their
  shipped defaults, which differ in scheduler, precision (FP8 against bf16) and compilation together; the effect of each
  switch on its own was not isolated in this run.
* `/stats` after each run: the new server captured 89 graphs at start-up and none afterwards (`graph_captures` 89 at the
  end of both replays); its only eager forwards were the 5 requests per replay with a row above 8,192 tokens
  (`eager_forwards` 5 after concurrency 1, 10 after both; `unbucketed_requests` the same). The old server captured 72 graphs
  at warm-up (its log), had 74 after the concurrency-1 replay and 76 (2B) or 77 (4B) after the concurrency-8 replay: four to
  five shapes compiled and captured on the request path, with the GPU lock held, plus 56 requests per replay that left the
  graph path (`long_forwards`). Both servers took the shared-state path on the same 118 requests per replay.
* Agreement with the eager forward is a tolerance, not equality. The new server's argmax differs from the masked eager
  reference on 8 to 13 of 8,247 answers per run (0.10% to 0.16%); at every one of them the reference's top probability is at
  most 0.53 and its top-two margin at most 0.071 (2B) or 0.031 (4B). The largest per-answer probability difference is 0.053
  on the 4B and 0.116 on the 2B (one answer of a home-appliance row, argmax unchanged). A fresh 200-request replay against a
  live server (second review) gave 1,505 of 1,506 answers in agreement, max |dp| 0.023. The likely cause is bf16 reduction
  order changing with the batch shape (the same server disagrees with itself at the same rate between concurrency 1 and 8),
  but the raw files do not isolate it. The old server at its FP8 defaults differs on 109 to 232 answers (1.3% to 2.8%), 10
  to 28 of them at a reference confidence above 0.6, with probability differences up to 0.65: its default numerics are not
  those of the released bf16 weights. `usage` blocks match the reference in every run.
* Start-up: 27 s (2B) and 45 s (4B) for the 89 graphs, against 114 s and 126 s for the old server's 72 compiled shapes with a
  warm inductor cache (the first write-up measured 309 s and 393 s with a cold one).

### 6.3 The 1.1.1 batching change (decider-2b, 1,000 requests, shared card)

Setup: the same 1,000 rows and the same harness as sections 6.1 and 6.2, decider-2b, `CUDA_VISIBLE_DEVICES=3` on a B300
that a training run was using at the same time, so the absolute latencies here are about twice those of section 6.1 and
are only comparable within this table. "1.1.0" is `decider.serve` from the released checkout, "1.1.1" is the same module
from the working tree; the two cases ran back to back on the same card, each as its own server process, concurrency 1
first and then concurrency 8. "batches" and "single-row batches" are that replay's share of `/stats`, which is cumulative
over a case. Two 1.1.1 controls are included: `DECIDER_BATCH_ADAPTIVE_WAIT_MS=0`, and `DECIDER_MERGE_OVERHEAD_TOKENS=0`,
which turns the partition back into the 1.1.0 per-bucket grouping inside the new code. "eager disagreements" is the
number of the 8,247 answers whose argmax differs from `decider/bench/eager_reference.py`; every case has exactly one
answer above the 0.05 probability tolerance and a maximum difference of 0.116, the same answer as in section 6.1, and no
`usage` mismatches.

| server | conc | median | p95 | p99 | mean | max | req/s | errors | batches / 1,000 requests | single-row batches | eager disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.1.0 | 1 | 18.5 | 187.0 | 400.5 | 57.9 | 639.2 | 17.25 | 0 | 1,207 | 783 | 9 (0.11%) |
| 1.1.1 | 1 | 18.5 | 185.9 | 379.1 | 57.8 | 698.3 | 17.30 | 0 | 1,134 | 745 | 8 (0.10%) |
| 1.1.1, adaptive wait off | 1 | 18.8 | 184.7 | 360.5 | 57.5 | 644.2 | 17.38 | 0 | 1,134 | 745 | 8 (0.10%) |
| 1.1.1, merge off | 1 | 18.5 | 190.5 | 371.1 | 57.0 | 638.9 | 17.53 | 0 | 1,207 | 783 | 9 (0.11%) |
| 1.1.0 | 8 | 304.4 | 1018.6 | 1315.8 | 432.8 | 1619.3 | 18.46 | 0 | 1,027 | 476 | 13 (0.16%) |
| 1.1.1 | 8 | **300.8** | 1030.6 | 1299.7 | **420.3** | 1565.9 | **19.02** | 0 | **748** | **210** | 12 (0.15%) |
| 1.1.1, adaptive wait off | 8 | 305.3 | 1057.9 | 1267.4 | 421.4 | 1459.1 | 18.97 | 0 | 749 | 208 | 13 (0.16%) |
| 1.1.1, merge off | 8 | 316.7 | 1024.3 | 1256.3 | 425.3 | 1522.6 | 18.79 | 0 | 1,017 | 464 | 13 (0.16%) |

Reading the table:

* The partition does what it was written for. At concurrency 8 the server runs 748 forwards for 1,000 requests instead of
  1,027, and single-row forwards fall from 476 to 210. Throughput rises 3% (19.02 against 18.46 req/s), the mean falls
  from 433 to 420 ms and the median from 304 to 301 ms; p95 is 1% worse and p99 1% better, which is inside the run-to-run
  spread on a shared card. At concurrency 1 the median is unchanged at 18.5 ms and the forward count still falls (1,207
  to 1,134), because a single request whose rows land in several length buckets is now one forward instead of several.
* The merge is what produces the gain, not the adaptive wait or anything else in the release. With the merge off the new
  code reproduces the 1.1.0 forward counts (1,207 and 1,017 against 1,207 and 1,027) and lands between the two servers on
  latency (concurrency-8 median 316.7 ms, 18.79 req/s). Turning only the adaptive wait off leaves the forward count
  unchanged (749 against 748) and moves throughput by 0.3%. The adaptive wait is kept on at 2 ms because it does not cost
  the concurrency-1 median and gives the collection a chance to fill when requests are genuinely overlapping; on this
  sample its effect is inside the noise.
* The shared-state path took the same 118 requests per replay in both servers, and both captured 89 graphs at start-up
  and ran the same 5 eager forwards per replay for rows above 8,192 tokens.
* This is one measurement on one shared card. The 1.1.0 numbers in section 6.1 (median 9.1 ms, 42.8 req/s at concurrency
  8) were taken on an idle card; nothing here contradicts them, and nothing here should be compared to them.

Raw files: `summary.json`, per-case responses `{old,new}_c{1,8}.jsonl`, `eager.jsonl` and the server logs under
`runs/serving_v2_release/{2b,4b}` in the research checkout; `gpu2_monitor.log` there records the processes on the card
every 30 s during the runs (only the server under test, or the reference, at any time).

Not measured here: the 35B MoE checkpoint, `independent: false` requests end to end, Score questions under a running
server (the sample is all Choice; the 2B ran with isolated levels on but no row exercised them), the schema cache's latency
(its admission and wire format are tested, `tests/test_serve_http.py`), `DECIDER_COMPILE=1` or `DECIDER_FP8=1` on the new
server, and the
Decision Index harness itself (its 132k requests at unknown concurrency against decider-2b gave 50 ms / 1,641 ms; the
numbers above are comparable server to server, not to theirs).

## 7. Reproducing

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.
D=/path/to/sample.jsonl.gz     # {"id", "state", "questions"} rows
python -m decider.bench.run_serving_matrix --model /path/to/model --data $D --out-dir runs/serving --rows 1000 --conc 1,8 \
    --case old:decider.serve_v1: --case new:decider.serve:
python -m decider.bench.eager_reference --model /path/to/model --data $D --rows 1000 --out runs/serving/eager.jsonl
python -m decider.bench.replay_systemone compare runs/serving/new_c1.jsonl runs/serving/eager.jsonl --tol 0.05
python -m decider.bench.verify_engine_v2 --model /path/to/model --data $D --rows 200 --engine1
python -m pytest -q tests                                   # CPU; add -m cuda with a GPU for the real-checkpoint tests
```

`run_serving_matrix` starts each server as its own process, waits for `/health`, replays, and terminates it by pid.
`summary.json` in the output directory holds the latency summaries, start-up times and `/stats` of every case.

## Appendix: investigation history (2026-09-22)

1. Diagnosis on the 1.0.x code: the schema cache was off by default on both released checkpoints, so the Decision Index
   tail was not schema-dependence. It was the six `(B, 2048)` shapes compiled and captured on the request path under the GPU
   lock (about 5 s each with `DECIDER_COMPILE=1`), rows above 2,048 tokens running eager at request-specific shapes (10.6% of
   the sample), and the eager shared-state pass (11.5%). FP8 and the unrolled conv patch, both on by default, are wins only
   inside compiled regions and losses on the eager paths.
2. First measurement of the redesign on the 4B and 2B (docs/SERVING.md as first written): the shape-keyed grid removed all
   request-path captures, cut the 4B p99 from 812 to 316 ms with the shared pass, and showed FP8 changing 1.8% of the 2B's
   answers against bf16 and torch.compile measuring faster in the bare engine but slower through the server (unexplained;
   the leading candidate is the conv patch on the eager paths).
3. The cached two-pass forward was found to disagree with the single prefill at some split points. The first write-up
   attributed it to the model's cached continuation and turned the shared path off. The independent review
   (`decider2/SERVING_V2_REVIEW.md`) isolated it below the model to the cuDNN SDPA backend on this installation (torch
   2.14 / CUDA 13, B300) with the same Q, K, V and a verified mask, and showed the math and memory-efficient backends
   correct; 1.0.2 shipped the backend policy in `Engine`, and this release applies it in `EngineV2` and turns the shared
   path back on.
4. The review's other blocking items, fixed here: the empty-question usage regression (`unique_tokens` returned the
   context length for zero rows), the missing `/decide` route and schema-first handling, no total-request bound (a 16-token
   state cap still admitted a 40,034-token row), no queue bound, tests importing torch and fastapi unconditionally with
   author-specific tokenizer paths, and the comparison tools counting options rather than answers, ignoring missing ids,
   and treating NaN as agreement.
5. Second review (`decider2/SERVING_V2_REVIEW_2.md`): the schema-cache path checked its limits only on the state suffix and
   ran the schema's prefixes on the GPU before the check and before admission (a 222-token request got 413 in state-first and
   200 in schema mode under a 128-token bound). Fixed by planning and tokenising prefix and suffix on CPU, checking the full
   cost and reserving queue work before any GPU preparation; regression test in `tests/test_serve_http.py`. The public
   "no runtime capture" claim was qualified to the default path and the schema cache's captures are counted in `/stats`.
