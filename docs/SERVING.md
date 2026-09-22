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
  `DECIDER_SHARED_MIN_TOKENS` (768) tokens runs the state once, forks the cache to every row and scores only the question
  suffixes (`EngineV2.score_shared`, the algorithm of `Engine.score_shared`). It is eager at request-specific shapes and is
  correct with the backend policy in place (section 4).
* **One GPU thread.** Every forward, including graph capture at start-up, runs on a single-thread executor; tokenisation
  runs on a CPU pool (`DECIDER_TOKENIZE_THREADS`, 8). Rows queued at the same moment are grouped by length bucket and
  replayed together; `DECIDER_BATCH_WAIT_MS` (0) adds a collection window.
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
| `DECIDER_COMPILE` | `0` | torch.compile the forward during warm-up (never at runtime) |
| `DECIDER_FP8` | `0` | e4m3 weights with per-token activation scaling |
| `DECIDER_SHARED` | `1` | shared-state path for multi-question requests over long states |
| `DECIDER_SHARED_MIN_TOKENS` | `768` | shortest row length at which the shared path applies |
| `DECIDER_MAX_BATCH` | `32` | rows per dispatch |
| `DECIDER_BATCH_WAIT_MS` | `0` | collection window; `DECIDER_MAX_WAIT_MS` is an alias |
| `DECIDER_MAX_STATE_TOKENS` | `32768` | state truncation |
| `DECIDER_T_BUCKETS`, `DECIDER_B_BUCKETS` | the ladders above | comma-separated |
| `DECIDER_GRAPH_TOKEN_BUDGET` | `32768` | `(B, T)` captured when `B == 1` or `B * T` fits; also the eager chunk size |
| `DECIDER_WARMUP` | `1` | `0` skips capture (everything runs eager) |
| `DECIDER_TOKENIZE_THREADS` | `8` | CPU pool |
| `DECIDER_MAX_ROWS` | `1024` | scoring rows one request may expand to (questions, isolated score levels) |
| `DECIDER_MAX_ROW_TOKENS` | `MAX_STATE_TOKENS + 4096` | tokens in one row: truncated state plus question block |
| `DECIDER_MAX_REQUEST_TOKENS` | `1048576` | sum of row lengths of one request |
| `DECIDER_MAX_QUEUE_ROWS` | `4096` | rows admitted and not yet scored, over all requests |
| `DECIDER_TEMPERATURE` | config `temperature`, else 1.0 | softmax temperature |
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
requests before the lifespan start-up finishes. `/stats` reports `requests`, `decisions`, `rows`, `batches`,
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
of memory fails with `torch.OutOfMemoryError` and exits, it does not hang). Start-up is 27 to 45 s for the grid (section 5).

## 4. Correctness

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
* `decider/bench/verify_engine_v2.py` compares `EngineV2.score_items`, `EngineV2.score_shared` and `Engine.score_items`
  against the masked eager forward over a row file and exits 1 on any argmax mismatch, non-finite output or max |dp| above
  `--tol`. `decider/bench/probe_cache_split.py` sweeps split points of a random sequence and exits 1 on any failure; `--cudnn`
  reproduces the fault.
* Answers are deterministic for a fixed batch shape and not across batch shapes (bf16 reduction order): against the eager
  reference the server differs on 0.1% to 0.16% of argmaxes, all at near-ties (reference top probability at most 0.53),
  with per-answer probability differences up to 0.05 on the 4B and 0.12 on the 2B (section 5). Pin
  `DECIDER_MAX_BATCH=1 DECIDER_B_BUCKETS=1` for bit-reproducible answers.

## 5. Measurements

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

Raw files: `summary.json`, per-case responses `{old,new}_c{1,8}.jsonl`, `eager.jsonl` and the server logs under
`runs/serving_v2_release/{2b,4b}` in the research checkout; `gpu2_monitor.log` there records the processes on the card
every 30 s during the runs (only the server under test, or the reference, at any time).

Not measured here: the 35B MoE checkpoint, `independent: false` requests end to end, Score questions under a running
server (the sample is all Choice; the 2B ran with isolated levels on but no row exercised them), the schema cache's latency
(its admission and wire format are tested, `tests/test_serve_http.py`), `DECIDER_COMPILE=1` or `DECIDER_FP8=1` on the new
server, and the
Decision Index harness itself (its 132k requests at unknown concurrency against decider-2b gave 50 ms / 1,641 ms; the
numbers above are comparable server to server, not to theirs).

## 6. Reproducing

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
