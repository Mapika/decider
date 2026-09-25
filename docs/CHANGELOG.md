# Changelog

Newest first. Every entry names the weights it applies to; the Hub repositories keep earlier weights under tags where noted.
`HISTORY.md` is the long form: how each stage was trained and what was measured.

## 1.5.0 (2026-09-25): `decider.serve_vllm` and `DECIDER_LAYOUT`

* **`decider.serve_vllm`**: the `/v1/systemone` readout on vLLM 0.29.0 for large stock or chat-layout checkpoints. Same rows
  (`decider.serve.prepare`), same slot, `softmax(letter logits / T)` read from vLLM's raw log-softmax values of the option
  letters; up to 255 options in one request (`decider.vllm_worker` raises vLLM's 128-id cap in the worker); rows sharing a
  prefix of at least one cache block run the first row alone so the others hit vLLM's prefix cache; the request limits and
  capacity messages of `decider.serve`; rows cancelled and aborted when a row fails, the client disconnects or the handler is
  cancelled, and the admission is released only after they have stopped. Qwen/Qwen3.6-27B bf16 at T 1.943 on one idle B300,
  Decision Index 0.2 sample (4,490 requests), one request at a time: median 32.3 ms, p95 430 ms, against 40.1 / 712 ms for
  `decider.serve` 1.4.0 and 50.8 / 716 ms for the research server of the submitted run; 99.62% argmax agreement with the
  submitted run (docs/SERVING.md section 8). Install it in its own environment (vLLM 0.29.0 needs numpy 2):
  `pip install vllm==0.29.0 fastapi "uvicorn[standard]" jinja2 huggingface_hub && pip install --no-deps decider-ai`.
  No weights change.
* **`DECIDER_LAYOUT`** (`decider.serve`, `decider.serve_vllm`): `chat` or `plain` replaces the layout of `decider_config.json`,
  so a stock checkpoint without one can be read in the chat layout (`prompt.with_layout`).
* **`scripts/stage_release.py`** also copies `serve_vllm.py` and `vllm_worker.py` into a model release folder.

## decider-4b v2.1 and decider-2b v11 (2026-09-24): replay toward the parent, one temperature per answer type

[Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) now holds v2.1 (v2 under the Hub tag `v2`, v1 under `v1`) and
[Mapika/decider-2b](https://huggingface.co/Mapika/decider-2b) holds v11 (v10 under the tag `v10`, v8 under `v8`). Both are their
parent plus a LoRA of rank 64 on the attention and MLP weights, 2 epochs, merged: the generated decision families, document
questions and human-labelled rows of decider-4b v2's stage 2, plus replay rows of the parent's mixture that are trained toward
the parent's own answer distribution (KL to the parent) instead of their labels. Both `decider_config.json` files carry a
`temperature_by_type` map (decider-ai 1.4.0, below); decider-ai 1.3.0 and earlier ignore the map and use `temperature`.
Neither passed its pre-registered release rules; both were released on the full comparison, which the model cards give with
every failure.

* **decider-4b v2.1** (v1 + LoRA on 29,325 rows; `temperature` 1.099, map choice 1.110, noul 1.560, score 1.287). Against v2 on
  the same rows: bag-draw games in sampled play 52.0% against 37.9% wins (v1 56.6%), zero-shot games sampled 26.9% against 22.4%,
  live browser sampled 93.2% against 88.1%, CliffWalking −13 against −60, regression set 0.831 / 0.784 against 0.824 / 0.779;
  held-out generated families 0.556 against 0.560, JevBench public hard tier 0.649 against 0.676. Worse: calibration on hard
  items (held-out generated families 0.147 against v2's 0.046, JevBench hard tier 0.184 against 0.104), BabyAI-GoTo 0.19 (v1
  0.54), greedy bag-draw play 9.4 points under v1, needs-live-data and touches-outside-project probes one item under v1 each, and
  issue #9 form case c_1 is still answered wrongly. The rule failed on c_1, and the map on the 0.08 calibration limit.
* **decider-2b v11** (v10 + LoRA on 42,749 rows; `temperature` 1.145, map choice 1.164, noul 1.624, score 1.124). Against v10 on
  the same rows: held-out generated families 0.429 against 0.324, held-out document questions 0.753 against 0.646, JevBench public
  hard tier 0.577 against 0.459, calibration error 0.156 against 0.226 on the held-out generated families and 0.175 against 0.307
  on the JevBench hard tier. Worse: human-labelled public sets −2.2 points, a knowledge guard set −1.6, greedy bag-draw play −10.9,
  sampled slippery-grid play −4.3, sampled browser play −2.8 (interval includes zero), TypeSafe −4.9 (interval includes zero). No
  checkpoint met the rule's eligibility condition (at most 1 point lost on the human-labelled sets); the fallback candidate fails
  the 0.08 calibration limit, which v10 (0.226) also fails. The model name in answers is `decider-2b-v11` (v10: `decider-v10`),
  and the config no longer marks the model as trained for the schema-first layout.

## 1.4.0 (2026-09-24): one temperature per answer type

No weights change for the models released before this version: their `decider_config.json` files have no map, so they answer
exactly as with 1.3.0. decider-4b v2.1 and decider-2b v11, released with 1.4.0, carry a map (entry above).

* **`decider_config.json` may set a temperature per answer type.** Next to `temperature` (one value for every answer, the only
  form before 1.4.0), a config may carry `temperature_by_type`, for example
  `{"choice": 1.48, "noul": 2.22, "score": 1.38}`. A type that is missing uses `temperature`. The keys are the answer types
  of `POST /v1/systemone`: `choice`, `noul` and `score`. A `POST /decide` / `decide_json` field maps onto them as `choice` ->
  `choice`, `bool` -> `noul`, `scale` -> `score`, and a `/v1/systemone` question of type `bool` is a `noul`. Questions given to
  `Decider.decide()` / `decide_batch()` (a question and its options, no type) are `choice`.
* **Isolated Score levels use the `score` temperature.** A Score question read with isolated levels (`isolated_levels`: one
  yes/no row per level, combined into the level distribution) divides every one of its level rows by the `score` temperature,
  not the `noul` one. The rows are yes/no readings, but together they form one Score answer, and the temperature is fitted on
  that answer: `decider.calibrate` fits `score` by the NLL of the combined level distribution, so the fitted value and the
  served readout are the same computation. Fit it on answers read with the same `isolated_levels` setting the model serves
  with.
* **Schema cache.** The schema cache (questions-first layout) keeps its own `temperature_schema_first` and may also have
  `temperature_schema_first_by_type`. Its temperature for type t is `temperature_schema_first_by_type[t]`, else
  `temperature_schema_first`, else, when the config has no schema-first value at all, the state-first temperature of t.
* **Every scoring path applies it:** `Decider` on the CUDA-graph engine and on the eager model (MPS, CPU), the shared-prefix
  path, the schema cache (`Decider.schema()` and the server's), `decide_json`, `system_one`, and both servers
  (`decider.serve` and `decider.serve_v1`). Each prompt row carries the answer type of each of its answer slots, so a server
  batch that mixes requests and types still gives every answer its own temperature.
* **Overrides.** `Decider(path, temperature=T)` and `DECIDER_TEMPERATURE` replace `temperature` and switch
  `temperature_by_type` off, so the override is the one temperature of every state-first answer, as before 1.4.0.
  `Decider(path, temperature_by_type={...})` sets the map in code.
* **Validation at load time.** Every temperature (`temperature`, `temperature_schema_first` and each map value) must be a finite
  number > 0, and a map may only have the keys `choice`, `noul` and `score`. Anything else stops `Decider()` and the server's
  start-up with a `ValueError` that names the key and the allowed keys (for example `"bool"` is refused with the hint that
  it is `"noul"`). Before 1.4.0 a temperature of 0 or a negative one was accepted and gave NaN or inverted probabilities.
* **Reporting.** `GET /health` and the server's ready line report `temperature` and `temperature_by_type` (the temperature
  every type gets, after the fallback), and `temperature_schema_first_by_type` when the schema cache is on.
* **Fitting: `python -m decider.calibrate records.jsonl`** fits the map by NLL from answers read at temperature 1, grouped by
  type (grid search on 0.05 to 20, then a golden-section refinement), and prints it with one pooled temperature for comparison
  and the NLL before and after. A malformed record (gold out of range, wrong shape, NaN) is refused with its position; the
  isolated-levels objective is computed in log space, so confident level rows do not underflow.
  `decider.calibrate.collect(decider, examples)` produces the records from a `Decider` and
  labelled `/v1/systemone`-shaped examples.
* **Unchanged without the map.** A config without `temperature_by_type` gives every path the same Python float as 1.3.0, so
  the probabilities are bit-identical. `tests/test_temperature.py` checks that the engines receive that float and compares
  every CPU scoring path against numbers produced by 1.3.0 (`tests/data/temperature_1_3_0_pins.json`); decider-0.8b (CUDA,
  eager) gave byte-identical `system_one`, `decide_json` and `decide_batch` output under 1.3.0 and 1.4.0.
* **Older versions ignore the map.** decider-ai 1.3.0 and earlier read only `temperature` and ignore `temperature_by_type`, so a
  model with a map serves every answer at its `temperature` there. A temperature does not change which option is most probable,
  so the answers are the same, except exact ties between the level rows of an isolated Score answer, which rounding can break
  differently (5 of 34,858 rows over the eight measurement sets of both models, all isolated Score rows). Checked on decider-4b v2.1 and decider-2b v11,
  CUDA, eager: 1.3.0 gives exactly the probabilities of 1.4.0 with the map switched off.
* `scripts/stage_release.py` copies the new modules `decider/temperature.py` and `decider/calibrate.py` into a release folder.
* Tests: the CUDA tests (`-m cuda`) default to `Mapika/decider-2b` at the Hub tag `v10`, the checkpoint their tolerances were
  measured on; the chunked shared-prefix fork is bit-identical to the single fork on v10 but differs by up to 4.4e-4 in probability
  on v11 (argmax unchanged). `DECIDER_TEST_MODEL` still selects another checkpoint.

## 1.3.0 (2026-09-24): TypeSafe's confidence, noul questions without instructions

Code only; no weights change. Both changes follow the conformance report in #15.

* **`confidence` values change.** In `POST /v1/systemone` answers and in the Python API that builds the same answers
  (`Decider.system_one()` and `Decider.schema()`), `confidence` on a Choice or Score answer now follows TypeSafe's definition.
  Before 1.3.0 it was the largest probability, p_max.
  * Choice with n options: `(n·p_max − 1)/(n − 1)`, clipped to [0, 1]. A uniform distribution gives 0, all probability on one
    option gives 1. Example from #15: probabilities 0.195, 0.1994, 0.2252, 0.2286, 0.1518 gave `confidence` 0.2286 and now give
    0.0358.
  * Score with n levels: `max(0, 1 − Σ pᵢ·|i − k| / D)`, where k is the most likely level and
    `D = (1/n)·Σ |i − (n − 1)/2|` is the mean distance of the n levels from the middle of the scale. This is the formula of TypeSafe's `system-one-adapter-python` (`score_confidence`). For
    TypeSafe's documented Score example (probabilities 0, 0.95, 0.05) it gives 0.925, as does the Choice formula applied to the
    levels; the two differ on other distributions, and we use the adapter's because it is TypeSafe's own code. With two levels
    the two formulas are equal.
  * Noul answers have no `confidence`, as before and as TypeSafe documents.
  * **To keep the old value, read `x_p_max`.** Every Choice and Score answer now also has `x_p_max`, the largest probability,
    which is exactly the old `confidence`. A threshold tuned on `confidence` before 1.3.0 can be moved to `x_p_max` unchanged.
    For a Choice with a fixed number of options n, the new `confidence` is an increasing function of `x_p_max`, so a threshold t
    on the old value corresponds to about `(n·t − 1)/(n − 1)` on the new one (the two fields are rounded separately, so values
    at the boundary can fall on different sides). For a Score there is no such conversion: two answers with the same `x_p_max`
    can have different `confidence`. `examples/routing_with_confidence.py` and `examples/composite_scoring.py` now read `x_p_max`,
    so their routing decisions and printed probabilities are as before; the routing example also prints `confidence`.
  * Unchanged: `POST /decide` and `Decider.decide()` / `decide_batch()` (the plain form, which is not TypeSafe's format) still
    report the top probability as `confidence`. `certainty`, `probabilities`, `score`, `level_fit` and `fit_mass` are unchanged.
  * `decider.bench.public_suite` computes its calibration error from `x_p_max`, so its numbers are comparable with earlier runs.
* **A noul question may omit `instructions`.** TypeSafe's OpenAPI file marks `instructions` optional; 1.2.2 answered such a
  question with HTTP 422 (`question without instructions`). A noul (or bool) question whose `instructions` is missing, `null` or
  empty is now answered when its `criteria` describe true or false. The question text shown to the model is then the fixed
  sentence `Which answer fits the context?` and the options are rendered from the criteria as before (`no: <false description>`,
  `yes: <true description>`). The question id is not used, because ids are never shown to the model. A noul question with
  neither instructions nor a true or false description still gets 422, now with the message
  `noul question without instructions: criteria must describe true or false`; this includes `"instructions": null` without
  criteria, which 1.2.2 answered with the text `null` as the question. Questions that give instructions are rendered exactly as
  before. Choice and Score questions still require `instructions`. On 12 hand-written noul questions, decider-2b answered 12 of
  12 correctly with instructions and 11 of 12 with the criteria alone.

## decider-4b v2 (2026-09-24): a LoRA stage on harder decisions

[Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) (bf16, 8.4 GB) now holds v2; the v1 weights stay under the Hub tag
`v1`. v2 is v1 plus a LoRA of rank 64 on the attention and MLP weights, trained for 2 epochs on 29,356 rows and merged: generated
decision families with code-computed answers, questions over business documents written by Qwen3.6-27B and kept only when two
independent answers agreed, human-labelled public sets, and replay of v1's mixture v2. No JevBench or Decision Index item was
used for training, selection or temperature. Temperature 1.935 (v1 1.05), fitted on 61 in-task regression tasks. Plain layout,
as v1: decider-ai 1.0.2, 1.1.4, 1.2.1 and 1.2.2 were checked to load it and give the same answers; no package change is needed.

Against v1 on the same rows: JevBench public hard tier 0.676 against 0.550 (read once, after selection) with hard-tier ECE 0.071
against 0.288 (recomputed at 1.935 from the stored probabilities, 6 Score items kept at 1.719); OpenJev +2.8 points; TypeSafe +4.9 (interval includes zero); Bespoke's suite 0.773 against 0.757 macro; Decision
Index sample calibration error 0.074 against 0.086. Worse: regression set −1.0 in-task and −0.9 held-out points (0.824 / 0.779),
in-task ECE 0.041 against 0.027; bag-draw games in sampled play 37.9% against 56.6% wins and zero-shot games sampled 22.4%
against 27.8%; sampled browser play −2.8 points; CliffWalking −60 against −13 and BabyAI-GoTo 0.35 against 0.54; model-router
probe 0.935 against 0.968. The pre-registered rule of the training run (beat an earlier candidate on our own held-out hard sets)
was not met by 0.5 to 0.8 points; the release was decided on the full comparison with v1. If you rely on sampled play, load
revision `v1` (`Decider(snapshot_download("Mapika/decider-4b", revision="v1"))`). The model card has every row with its interval.

## 1.2.2 (2026-09-24): validation and release-script fixes

Code only; no weights change.

* `/v1/systemone`: a noul (or bool) question whose `criteria` is not a map or null, for example `["bad"]`, returned HTTP 500.
  It now returns 422 with `noul criteria: a map of optional true/false descriptions`, before any forward pass. Empty non-map
  values (`[]`, `""`, `false`, `0`) were read as "no criteria" before and are now also rejected. Omitted or `null` criteria
  and one-sided maps are accepted as before. The same check is in the shared question renderer, so in the Python API
  `Decider.system_one()` and `Decider.schema()` now raise `ValueError` for these values. Contributed by @dajiaohuang (#11,
  fixes #10).
* `scripts/stage_release.py` now copies the four modules the server imports (`batching`, `prompt_fast`, `engine_v2`,
  `shared_prefix`) and the two Apple Silicon modules the engine loads on MPS (`mps_ops`, `mps_moe`). Before this, a staged
  bundle could not import `decider.serve`. A test stages a bundle and imports the server. Contributed by @dajiaohuang (#13,
  fixes #12).
* `scripts/train.sh` sets `pipefail`, so a failed training step stops the script before evaluation. `scripts/run_full.sh`
  writes `FULL_RUN_DONE` only when training and evaluation both succeed; otherwise it writes `FULL_RUN_FAILED` with the
  exit status (fixes #14, reported by @dajiaohuang).

## 1.2.1 (2026-09-23): jinja2 is a dependency

Code only; no weights change. The chat layout of 1.2.0 builds its prompts with the tokenizer's chat template
(`tok.apply_chat_template`), which needs `jinja2`; transformers does not install it. 1.2.0 did not list it, so loading a
chat-layout model in an environment without `jinja2` stopped with an `ImportError`. No released model uses the chat
layout, so no released model was affected. `jinja2` is now a dependency of the package, and the test workflow installs it.

## 1.2.0 (2026-09-23): the chat prompt layout

Code only; no weights change: `Mapika/decider-2b` stays v10. decider-2b v11, a research checkpoint that is not released, was
trained with every prompt wrapped in its tokenizer's chat template, and it gives wrong probabilities when it is read in the
plain layout that every released model uses. 1.2.0 reads
the layout from the model's `decider_config.json`. `"layout": "chat"` (or `"chat_template": true`) selects the chat
layout. A config with no `"layout"` key selects the plain layout, which is every released model up to and including
decider-2b v10, decider-4b, decider-35b-a3b, decider-0.8b and decider-2b-vision.

The chat layout, state-first, for Qwen3.5:

    <|im_start|>user\n  Context:\n<state>  \n\nQuestion: <q>\nOptions:\n(A) ..\n(B) ..  <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n  Answer: (

The head and tail of the template come from `tok.apply_chat_template` on a single user turn, with no system prompt and with
thinking switched off (`enable_thinking=False`). When a row holds several questions, all question blocks come first. The
answer pieces (`Answer 1: (`, `\nAnswer 2: (`, ...) follow the tail. In the schema-first layout the user turn holds the
question blocks and then `\n\nContext:\n<state>`. Each part is tokenized separately, and the letter is read at the final
` (` token, as in the plain layout. The state cap (`max_ctx_tokens`, `DECIDER_MAX_STATE_TOKENS`) applies to
`Context:\n<state>`; the template tokens do not count against it (3 head tokens and 9 tail tokens for Qwen3.5).

Every path that builds prompts uses the layout: `decider.infer.Decider` (`decide`, `decide_batch`, `decide_json`,
`system_one`, `schema()`), on CUDA with graphs and on MPS and CPU; `decider.serve` (`/decide`, `/v1/systemone`, the
shared-prefix fork, the schema cache); and `decider.bench.eager_reference`. `decider.prompt` has the new functions
`resolve_layout`, `chat_template`, `chat_for` and `build_chat`. `build`, `schema_prefix_ids`, `schema_suffix_ids`,
`prompt_fast.build_rows`, `serve.prepare` and `SchemaEngine` take an optional `chat=` argument. The default `None` is the
plain layout. The engines are unchanged: they score token ids and do not depend on the layout. `/health` and the start-up
line of the server report the layout. A config that names a layout this version does not know (anything except `"plain"` and
`"chat"`), or that says both `"layout": "plain"` and `"chat_template": true`, raises a `ValueError` that names the value.
`Decider` raises it before loading the weights and the server raises it at start-up. `decider.serve_v1`, the 1.0.x server,
renders only the plain layout, and it now refuses a chat-layout model at start-up. Versions 1.1.x and earlier do not read
the `layout` key, so they read a chat-layout model in the plain layout without an error. A chat-layout model needs 1.2.0
or later.

Checks:

- **Plain layout.** Token ids are identical to 1.1.3/1.1.4. `tests/test_layout.py` pins sha256 digests of 359 prompt sets
  produced by 1.1.3: `prompt.build` (state-first and schema-first, several state caps, shuffled and unshuffled),
  `schema_prefix_ids`, `schema_suffix_ids`, `build_rows`, `serve.prepare`, the `/decide` preparation, and the prompt
  building of `Decider.decide_batch` and `Decider.system_one` with `neutralize_none` and `isolated_levels` on and off. On
  a GPU, decider-2b v10 under 1.1.3 and under 1.2.0 gives identical answers on 121 Decision Index rows (`system_one`),
  identical `decide_json` answers, and identical raw `decide_batch` probabilities (maximum absolute difference 0.0).
- **Chat layout, token ids.** The ids equal those of the chat-layout research server that v11 was evaluated with, and those
  of the builder v11 was trained with. The tests hold copies of both renderers. Against the originals, with the v11
  tokenizer: 802 `/v1/systemone` requests (401 Decision Index rows plus a 40,000-event log state with a 200-option
  question, each at state caps 32,768 and 512; 7,468 rows, of which 624 are truncated and 150 have more than 10 options;
  packed rows as well); 200 `/decide` requests; and 2,700 training-builder items (1,350 state-first and 1,350
  schema-first, with random option order, sub-sampling from 300 options, and state caps of 64, 1,536 and 16,384 tokens).
  All are equal.
- **Chat layout, probabilities** (decider-2b v11 candidate, T 1.0257, GPU, 301 Decision Index rows, 2,848 questions,
  compared with the research server's engine). The package's `Engine` with the server's settings (compile off, conv patch
  off) gives identical probabilities (maximum absolute difference 0.0, no argmax changes). The HTTP server (`EngineV2`,
  answers rounded to 4 decimals) gives a mean per-question maximum difference of 0.00027 and a maximum of 0.026, with 1
  argmax change. The library `Decider` (`torch.compile` and conv patch on, which is the library default for every model)
  gives a mean of 0.0044 and a maximum of 0.040, with 16 argmax changes, all at a top-2 margin of 0.06 or less. For
  comparison, the difference between the two engines on decider-2b v10 in the plain layout is a mean of 0.0028 and a
  maximum of 0.069, with 7 argmax changes. On CPU (the eager path that MPS also uses), 12 rows with 19 questions have
  equal ids, a maximum difference of 0.014 and no argmax changes.
- **Regression readout** (95-task public regression set, 144,226 rows, per-task RNG 1234, state cap 1,536, state-first,
  temperature fitted on the 67 in-task tasks). The package builds the rows with `prompt.build(..., chat=...)`. Through
  the eager model path, the probabilities are identical to the research readout of the v11 candidate on every row
  (maximum difference 0.0). The fitted temperature is 1.0257, and held-out accuracy, NLL and ECE (Decision Index
  definition) are 0.7619, 0.6161 and 0.0501, the same as the research readout. Through the library's CUDA-graph engine
  (compiled), the mean per-row maximum difference is 0.0035, 748 of 144,226 argmaxes change, and the held-out numbers
  are 0.7607, 0.6162 and 0.0512.
- **Evaluation, probes and benchmarks.** `decider.evaluate` (`run_eval(..., chat=)` and its CLI), `decider.probes.isolated`,
  `decider.probes.independence`, `decider.bench.verify_engine_v2`, `decider.bench.schema`, `decider.bench.latency` and the
  `decider.engine` self-check and `moe/vllm_check.py` read the model's layout (`decider.prompt.chat_for_model`, which reads `decider_config.json`
  from a folder or a Hub id) and build chat prompts for a chat model. `decider.serve_v1` reads the config the same way,
  so it also refuses a chat model given as a Hub id. The training and data-building code (`decider.train`,
  `decider/data/`, `decider.games.rl`, `decider.vision`) renders the plain layout only.
- **cuDNN attention off on the eager path too.** Since 1.0.2 the CUDA-graph engines switch off PyTorch's cuDNN
  scaled-dot-product-attention backend, which returns wrong output for masked attention on Blackwell with torch 2.14.
  `DecisionModel`, which `Decider(..., use_graphs=False)`, `decider.evaluate` and the probes use, now switches it off
  as well. The setting is process-wide, as it already was once an engine was built.

## 1.1.4 (2026-09-23): decider-35b-a3b on Apple Silicon, two MPS replacements

Code only; no weights change. Contributed by @nassersala in issue #6. On MPS, decider-35b-a3b ran transformers' Qwen3.5-MoE
reference code, and two operations in it took most of each decision on an M5 Max: `torch.histc` (tokens per expert, about
45 ms a call, once per MoE layer) and `torch.linalg.solve_triangular` (about 17 ms a call, twice per linear-attention layer).
The library's MPS patch (`decider.mps_ops.patch_mps`), which until now covered only the dense Qwen3.5 module, also installs
`decider/mps_moe.py`: an exact count in place of `histc`, and a block-doubling inverse in place of the unit lower-triangular
solve. They are installed only into the two transformers modules that make these calls (`transformers.integrations.moe`
and `modeling_qwen3_5_moe`, each given its own view of `torch`); global `torch` and every other caller are untouched, and
inside those modules they act only on MPS tensors of the matching call shape. The count drops out-of-range expert ids as
`histc` does. Reported effect: 2.8-4 s a decision becomes 0.23-0.5 s for typical inputs and 2.5 s for a 3,900-token input
(was 11 s); the JevBench public-item counts are unchanged (48/48, 70/72, 75/111). `DECIDER_MPS_MOE_PATCH=0` before the first
`patch_mps()` call leaves them out. Tested here on CPU tensors (the count equals `histc` including dropped ids; the solve matches torch to 1e-10
in float64). Tested on MPS by @nassersala at commit ac3183d (M5 Max, 128 GB, torch 2.14.0, transformers 5.17.0), JevBench
public items, 231 items:

| | easy / standard / hard | same answer as before, items | seconds a decision, median / p90 |
|---|---|---|---|
| decider-35b-a3b, 1.1.4 | 48/48, 70/72, 75/111 | 231 of 231 | 0.33 / 2.2 |
| decider-35b-a3b, `DECIDER_MPS_MOE_PATCH=0` | 48/48, 70/72, 75/111 | 231 of 231 | 3.16 / 7.8 |
| decider-2b, 1.1.4 (dense model; replacements not used) | 48/48, 63/72, 51/111 | 231 of 231 | 0.05 / 0.35 |

With the replacements the 35B's probabilities move by a median of 0.002 (largest 0.29, on a 2,338-token item); rerun with the
solve done exactly on CPU in float64, the block inverse was closer to the exact result than the MPS solver on 3 of the 4
most-shifted items. The 35B card now lists the Mac result.

## 1.1.3 (2026-09-23): `/decide` answers a malformed schema with 422

Code only; no weights change. Reported in issue #8: `POST /decide` with no `schema`, or with a question mapped straight to a
list of options (`{"Which team?": ["billing", "technical"]}`), returned 500 with an unhandled traceback, because the request
model accepted `None` and any dict and the first code to look inside the schema was the question conversion. The schema is
now checked before any work (`decider.infer.Decider._check_schema`). A missing schema, a schema that is not an object (one
holding NaN or Infinity was a 500, because the validation error echoed it), a field that is not an object, choice
options that are not a non-empty list of strings, a scale legend that is empty, has keys that are not finite numbers or holds
NaN or Infinity, and an unknown type are 422s whose message names the expected form. Legends with non-finite keys or values
were also a 500 (the answer could not be serialised), and so were non-string options under the default settings. Every
other schema 1.1.2 answered is still answered the same way, with two exceptions: options or a legend given as a bare string,
which 1.1.2 split into single characters, and non-string options on a model whose `decider_config.json` sets `neutralize_none` to false, are now 422s. An empty
schema returns `{}` without a forward pass. The library's `decide_json` raises the same `ValueError`. `/v1/systemone` with an
empty question map still returns 200 with an empty answer map, as in 1.0.x.

## 1.1.2 (2026-09-22): the HTTP server runs on MPS and CPU

Code only; no weights change. Reported in issue #5: the server always built its engine on `cuda`, so on a machine without CUDA
it stopped at start-up with torch's "Torch not compiled with CUDA enabled" assertion, while the README lists MPS and CPU. The
server now picks its device the way `decider.infer.Decider` does (CUDA, else MPS, else CPU; float16 on MPS, bfloat16
elsewhere), and `DECIDER_DEVICE=auto|cuda|cuda:<i>|mps|cpu` overrides it. Off CUDA there are no graphs to capture, so start-up
skips the warm-up and every request runs eager; on MPS the engine applies the same MPS patch as the library. An explicit
device that is not available, or `DECIDER_FP8` / `DECIDER_COMPILE` off CUDA, stops start-up with a message naming the
requirement. `/health` reports the device. Checked on CPU with no GPU visible: the server answers the README example with the
same probabilities as `Decider(device="cpu")` (billing 0.5706, refund 0.987, frustration 0.354 / 0.540 / 0.106), 685 ms per
request. MPS is not tested here; it uses the code path of the merged Apple Silicon work. Nothing changes on CUDA.

## 1.1.1 (2026-09-22): cross-request batch merging and a bounded shared-prefix fork

Code only; no weights change. Same routes, same request and response format, same limits. Two changes to `decider.serve`
and the two engines, both measured on decider-2b. Design and measurements: `docs/SERVING.md`, section 4.

**Batching.** 1.1.0 grouped the rows a collection pulled off the queue by exact padded length, so two requests that
arrived together but landed in different length buckets ran as separate forwards; at concurrency 8 that was 2,238 batches
for 2,000 requests, 1,250 of them a single row. `decider/batching.py` replaces the grouping with a partition that pads a
shorter row into a longer row's bucket when that costs less than a second forward. A forward of `B` rows at padded length
`T` is modelled as `DECIDER_MERGE_OVERHEAD_TOKENS + B * T` token-units; rows are sorted by padded length descending and a
dynamic program takes the cheapest split into consecutive groups, each capped at the engine's widest captured batch
bucket for its length and at `DECIDER_MAX_BATCH`. Rows above the last captured length bucket still run alone in their own
padded length. The 1.1.0 grouping is one of the partitions the program may choose, so the planned cost is never above it.
`DECIDER_MERGE_OVERHEAD_TOKENS` defaults to 512: on the 2B a single-row forward fits `t(T) = 9.20 ms + 13.80 us/token`
over the bucket ladder, so the fixed cost is worth 667 tokens, and 512 is the nearest power of two.
`DECIDER_BATCH_ADAPTIVE_WAIT_MS` (2 ms) is a new collection window that applies only after a collection that held live
rows from more than one request, and only while the queue stays busy: a first row that took more than 50 ms to arrive
drops the window again, and rows whose request has already gone away are not counted. After a single-request collection
the batcher does not wait, as before.

**Shared prefix.** `Engine.score_shared` and `EngineV2.score_shared` ran the state once and then
`cache.reorder_cache(zeros(n))`, which copies the prefix cache to all `n` question rows at once: `n` times the prefix
cache, which is 133 GB for a 31k-token state with 32 questions on a 31B model. The implementation now lives in
`decider/shared_prefix.py`, used by both engines, and forks the prefix cache in chunks of `m` rows with
`m = clamp(DECIDER_SHARED_FORK_GB // prefix_bytes, 1, n)`, the budget also capped at half the memory free on the device.
`min_prefix`, the cuDNN SDPA policy and `Engine.score_shared`'s signature are unchanged. The fork helpers walk
`cache.layers` and handle `keys`, `values`, `indexer_keys`, `conv_states` and `recurrent_states` without assuming one
layout, and always allocate fresh tensors, because the linear-attention layers write their states back in place. A layer
holding a tensor under any other name is not chunked at all: `score_shared` falls back to the 1.1.0 single fork of every
row and counts it in `/stats -> engine.shared_unchunked_layout`, which stays at zero on the released checkpoints. Answers: with equal-length suffixes the chunked path is bit-identical to the single fork at every chunk size, so
the batch size on its own changes nothing; with suffixes of different lengths a chunk pads to its own longest suffix and
the answers move by at most 2.9e-5 (four rows, 223 to 450-token suffixes) to 1.5e-2 (32 rows, 182 to 932-token suffixes)
of probability, argmax unchanged. Peak reserved memory on the 2B for the heaviest shared-prefix request in the 4,000-row
Decision Index sample (17 questions over a 7,988-token state): 3.75 GB before, 1.94 GB at a 1 GB budget, 0.67 GB at one
row per fork. On a synthetic 16,000-token state with 32 questions, 15.38 GB before and 2.17 GB at a 1 GB budget. The
default 8 GB budget does not bind on anything in that sample; it bounds the long-state case.

**Measured** (1,000 Decision Index rows, decider-2b, one B300 shared with a training run, so the absolute latencies are
inflated; 1.1.0 from the released checkout, 1.1.1 from the working tree, back to back on the same card): at concurrency 1
the median is unchanged at 18.5 ms, p99 379 against 401 ms, throughput 17.30 against 17.25 req/s. At concurrency 8 the
server runs 748 batches for 1,000 requests instead of 1,027 (single-row batches 210 against 476), throughput 19.02
against 18.46 req/s, median 301 against 304 ms, mean 420 against 433 ms, p99 1,300 against 1,316 ms. No errors in any
run. Two controls on the new code: with `DECIDER_MERGE_OVERHEAD_TOKENS=0` the batching is the 1.1.0 grouping again (1,207
and 1,017 batches) and the gain disappears; with `DECIDER_BATCH_ADAPTIVE_WAIT_MS=0` nothing measurable changes. Agreement
with the masked eager forward is unchanged: 8 to 13 of 8,247 answers differ in argmax in every case, old and new, one
answer above a 0.05 probability tolerance, `usage` identical.

## decider-4b v1 (2026-09-22): the supervised recipe on Qwen3.5-4B-Base with mixture v2

[Mapika/decider-4b](https://huggingface.co/Mapika/decider-4b) (bf16, 8.4 GB). One supervised pass over mixture v2, 742M tokens: the
public decision mixture (60% of tokens) plus 26 further public decision datasets (code, logs, legal, tables, finance, medical,
science, multilingual, temporal and rule reasoning) and ten programmatic families with verifiable gold, each with a held-out
variant. `torch.optim.AdamW` on the bf16 parameters, no FP32 master copy (the public trainer's optimizer), peak LR 1e-5, cosine, 26,729
steps of 32,768 tokens, 577 minutes on two B300s. No RL stage. Temperature 1.05 fitted on the in-task half of the regression set.
Against decider-2b v10 on the same rows: accuracy higher on 87 of 95 regression tasks (in-task / held-out 0.834 / 0.788 against
0.805 / 0.755, NLL −0.07), +2.7 points on the validation rows, +5.7 on Mind2Web, +0.6 on OpenJev, level on TypeSafe, JevBench hard
tier 0.541 against 0.459, Bespoke's suite 0.757 against 0.704 macro; below decider-35b-a3b by 2.1 to 2.3 regression points and on
every fixture. Live browser: greedy play level with v10 over all tasks (91.5% against 90.9%) and 17 points below it on the six
tasks v10 never used for reward (75.0% against 91.7%); sampled play 90.9% against 93.2%. Zero-shot on the ten text games (no game
rows in mixture v2): CliffWalking and Blackjack at teacher level, Breakout 18 against the teacher's 22, Pong not learned. Decision
Index 4,000-request sample: index 48.5 (2B 42.3, 35B 50.4), calibration error 0.086 against the 2B's 0.093 and the 35B's 0.027;
the 4B is overconfident outside its regression set (JevBench hard-tier ECE 0.29). Tables in `docs/RESULTS.md`, the model card on the Hub.

## 1.1.0 (2026-09-22): the HTTP server captures its CUDA graphs at start-up

Code only; no weights change. `decider.serve` is a new implementation with the same module name, routes, request format and
response format (fields, order, `usage`, 422 bodies); the environment variables are the same except where listed below. The
1.0.x server is kept for one release as `decider.serve_v1` (its code unchanged, its module docstring replaced) and is removed in
1.2.0. Probabilities are not byte-identical across the two servers: the defaults changed from FP8 to bf16 and a Hub-id
configuration is now read (below). Design, limits and measurements: `docs/SERVING.md`.

What changed and why. On the Decision Index the 1.0.x server gave decider-2b a p95 of 1,641 ms against a 50 ms median. The
cause was shape-dependence: the graph grid stopped at 2,048 tokens and was only warmed to 1,536, so the first request at a new
`(batch, length)` shape paid a torch.compile and a graph capture with the GPU lock held, and rows above 2,048 tokens ran eager
at request-specific shapes. The new server:

* captures every CUDA graph of a fixed `(batch bucket, length bucket)` grid during start-up (lengths 64 to 8,192, batches 1 to
  32, 89 graphs) and then seals the engine: on the default path no request can trigger a capture or a compile. Rows above
  8,192 tokens run eager in bounded chunks. The opt-in schema cache is the exception: as in 1.0.x, the first request with a new
  schema runs its prefix and captures one graph per (schema, batch bucket, state bucket); `/stats -> schema_cache` counts them.
  Engine: `decider.engine_v2.EngineV2`.
* defaults to `DECIDER_COMPILE=0` and `DECIDER_FP8=0` (were 1). At the old defaults the served probabilities differ from the
  bf16 weights on 1.3% to 2.8% of the measured answers (the 1.0.x row of the `docs/SERVING.md` table); the new defaults serve the
  bf16 numerics. The two switches were not measured in isolation against the new server; both remain available.
* keeps the shared-state path on (`DECIDER_SHARED=1`): an independent request with several questions over a state of at least
  768 tokens runs the state once and forks the cache per question. It is correct because the engine applies the attention
  backend policy of 1.0.2 before any capture; `tests/test_engine_v2_cuda.py` checks it against the full forward on the shape
  that failed.
* runs every forward on one GPU thread, tokenises the state once per request instead of once per question, and groups rows
  waiting at the same moment by length bucket.
* bounds requests before they reach the GPU: HTTP 413 when a request expands to more than `DECIDER_MAX_ROWS` (1,024) scoring
  rows, when a row exceeds `DECIDER_MAX_ROW_TOKENS` (state cap + 4,096 = 36,864) tokens, or when the request exceeds
  `DECIDER_MAX_REQUEST_TOKENS` (1,048,576) tokens in total; HTTP 503 when more than `DECIDER_MAX_QUEUE_ROWS` (4,096) rows are
  admitted and not yet scored. 1.0.x had no such limits.
* reads `decider_config.json` from a Hub id as well as from a folder. 1.0.x only read it from a folder, so
  `scripts/serve.sh Mapika/decider-2b` served temperature 1.0 with isolated levels off and the model name `decider-dev`; it now
  serves the config's temperature 1.3, isolated levels and `decider-v10`, as the library does.
* `/decide` returns 422 with the message for an invalid schema (1.0.x returned 500).
* `/health` is true only once the graphs are captured and the batcher is running; `/stats` adds `rows`, `rejected_too_large`,
  `rejected_overloaded`, `outstanding_rows`, `limits`, `schema_cache` and the engine's capture/replay/eager counters
  (`graph_captures`, `replays`, `eager_forwards`, `shared_calls` replace the 1.0.x `long_forwards` and `shared_prefix_calls`).
  `errors` counts requests that failed after parsing (question validation, inference failures); Pydantic 422s, 413 and 503
  are not in it.
* schema-cache requests are bounded and admitted like the others, on prefix plus suffix tokens, before any prefix is run on
  the GPU; a schema-cache request with no questions returns the empty answer set instead of an error.

Unchanged: the wire format of `/v1/systemone` and `/decide` (answers, `usage`, field order, 422 bodies), the schema cache and
its switches (`schema_first` in the config, `DECIDER_SCHEMA_CACHE`, `DECIDER_SCHEMAS`, `DECIDER_SCHEMA_MIN_SEEN`),
`DECIDER_MAX_BATCH`, `DECIDER_MAX_STATE_TOKENS`, `DECIDER_TEMPERATURE`. `DECIDER_BATCH_WAIT_MS` is the collection window as
before, default 0; `DECIDER_MAX_WAIT_MS`, declared but unused in 1.0.x, is now an alias for it. `DECIDER_MAX_FWD_TOKENS` is
replaced, not aliased, by `DECIDER_GRAPH_TOKEN_BUDGET` (32,768 padded tokens per forward). Start-up took 27 s (decider-2b) and
45 s (4B) for the graph grid on a B300.

Bench and test tooling: `decider/bench/replay_systemone.py` (replay a row file against a server, compare two runs),
`run_serving_matrix.py`, `verify_engine_v2.py`, `probe_cache_split.py`, `eager_reference.py`; `tests/test_serve_http.py` checks
the new server byte for byte against `serve_v1` on the same requests with a stand-in engine.

## 1.0.2 (2026-09-22): wrong answers from the cached shared-state path on Blackwell

Code only; no weights change. `decider.serve` scores the questions of one request against a cached prefix of the shared state
(`Engine.score_shared`), and the optional schema cache does the same for a cached schema. On a B300 with torch 2.14 / CUDA 13 the
cuDNN scaled-dot-product-attention backend that PyTorch selects for that masked rectangular attention (a suffix of a few hundred
tokens attending to a cached prefix of about 4,000) returns wrong, finite output in the first full-attention block; the math and
memory-efficient backends are correct, and the plain single-pass forward is unaffected. Effect: some long shared-state requests
got a wrong option with high confidence and the answer changed between identical requests. Reproduced on Decision Index row
RouterBench-5shot:26878 (prefix 4,006 tokens, suffixes 223 and 240): the cached path answered option_7 and option_10, the full
forward and the corrected path answer option_4 at p=0.95 and 0.93. Fix: `Engine` now disables the cuDNN SDPA backend before
compile and graph capture (`decider.engine.set_attention_backend_policy`). Found by an independent review of the serving path;
a review write-up with the reproduction commands is in the research notes. If you run the server from an earlier version on
Hopper or Blackwell, upgrade or set `torch.backends.cuda.enable_cudnn_sdp(False)` before constructing `Decider` or `Engine`.

## decider-35b-a3b v1 (2026-09-20): the supervised recipe on a 35B mixture-of-experts base

[Mapika/decider-35b-a3b](https://huggingface.co/Mapika/decider-35b-a3b) (bf16, 65 GB) and
[Mapika/decider-35b-a3b-nvfp4](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4) (NVFP4 for vLLM and TensorRT-LLM, 19.6 GB).
One epoch of the public mixture on Qwen3.5-35B-A3B-Base with the routed experts frozen and Muon on the block matrices, 394
minutes on four B300s, no RL stage. Above decider-2b v10 on 93 of 95 regression tasks (in-task / held-out accuracy 0.855 / 0.810
against 0.805 / 0.755), +6.7 points on the validation rows, +5.0 on OpenJev, +6.9 on Mind2Web, JevBench hard tier 0.676 against
0.459, Bespoke's suite 0.774 against 0.704 macro. Greedy browser play 97.2% against 90.9%, sampled play 86.4% against 93.2%: the
argmax is right more often, the served distribution is less sharp, which is what the RL stage of v10 trains. The NVFP4 build
loses 1.0 to 1.5 accuracy points against bf16 in vLLM. Tables in the README ("Results"), training details and the AdamW
comparison in `HISTORY.md`, scripts in `moe/`.

## decider-2b v10 (2026-09-19): calibration-aware RL on live browser tasks and exact games

v10 is the v8 weights continued for 384 steps of reinforcement learning whose only rewards are outcomes: whether a browser task's
own checker reports success, whether a game is won, and how well the model's stated belief about the next outcome of an action
matches the exact probability law of the game. No gold labels enter. A hard KL limit to the v8 weights on replayed training rows
keeps the model's answers on its original tasks in place. The recipe, gates and every measurement are in [docs/RL.md](RL.md).

### Browser

![v8 (left) and v10 (right) solving live MiniWoB++ click tasks in Chrome; each frame shows the chosen element and its served probability](../media/v10_browser_montage.gif)

*v8 (left) and v10 (right) on eight live browser tasks, same pages and seeds. Each click is one typed decision: the clickable
elements on the page are the options, the model returns a probability for each, and the task's own checker grades the result.
Six of the eight tasks were never used for training. Per-task recordings: `media/v10_browser_*.gif`.*

The browser gain is in the served distribution: greedy play is 90.9% against 90.3%, sampled play is where the ten points are, and
the six tasks that were never rewarded gain the most.

![per-task browser success, v8 against v10](../media/v10_browser_tasks.png)

### Games

![v8 (left) and v10 (right) on the same grid, tic-tac-toe, bag-draw and minesweeper boards, with the served action distribution written on every cell](../media/v10_games_montage.gif)

*Same boards, same dice for both versions. The numbers on the cells are the probabilities the model serves for each move; in
minesweeper the shading is the exact mine risk of each hidden cell, computed from all placements consistent with the revealed
numbers. Per-game recordings: `media/v10_game_*.gif`.*

A 2B model without search loses most of these games before and after RL. What moves is where the probability mass sits: in the
bag draws v10 puts 67% on the best bag where v8 spread 7% across many, and it wins 6 points more of them. On the grid, v10 puts 90%
on the right move where v8 put 37%, which helps when the move is right and hurts when the dice slip. Win rates on the same 234
boards, sampled play, with 95% intervals over boards:

| game | v8 | v10 | difference |
|---|---|---|---|
| bag draws (64 boards x 4) | 35.2% | 41.4% | +6.2 (+0.8 to +11.7) |
| 5x5 slippery grid (64 x 4) | 14.1% | 18.8% | +4.7 (−2.0 to +11.3) |
| tic-tac-toe against minimax with 25% random moves (74 x 4) | 23.6% | 23.0% | −0.7 (−4.7 to +3.0) |
| 4x4 minesweeper, 4 mines (32 x 4) | 2.3% | 0.0% | −2.3 (−4.7 to 0.0) |

### Calibration

![belief excess over the exact laws, and click-outcome prediction, v8 against v10](../media/v10_calibration.png)

Calibration is what the RL objective trains directly. For every action in a game with a known probability law, the model is
asked what will happen next, and its answer is scored against the exact law with a log score. v10 is 0.22 nats above the law
where v8 was 0.47. In the browser it predicts the outcome of its own click (success, failure, continue) at a log score of −0.03
against −0.35.

### Everything on the same rows

![v10 minus v8 on the same rows, with 95% intervals](../media/v10_vs_v8.png)

| on the same rows, v10 against v8 | v8 | v10 | difference (95% interval) |
|---|---|---|---|
| live MiniWoB++ click tasks, 22 tasks x 8 seeds, sampled play | 83.0% | 93.2% | +10.2 (+5.1 to +15.9) |
| the 6 tasks never used for reward | 72.9% | 91.7% | +18.8 (+6.2 to +31.2) |
| bag-draw games, win rate | 35.2% | 41.4% | +6.2 (+0.8 to +11.7) |
| Mind2Web element and action choice, 1,770 rows | 81.1% | 82.7% | +1.5 (+0.7 to +2.4) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 78.4% / 0.594 | 80.4% / 0.585 | +2.0 (−2.0 to +5.9) |
| 847 in-task validation rows, accuracy / NLL | 83.6% / 0.443 | 83.2% / 0.444 | −0.4 (−1.3 to +0.6) |
| Bespoke's public suite, 13 subsets, macro | 0.706 | 0.704 | |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.861 / 0.459 | 1.000 / 0.847 / 0.459 | |
| the regression set rebuilt here, 67 in-task / 28 held-out tasks, accuracy | 0.806 / 0.757 | 0.805 / 0.755 | within noise |
| OpenJev, 5,252 rows | 64.1% | 63.3% | −0.8 (−1.3 to −0.3) |

What v10 does not change: general accuracy on its training tasks, calibration on Bespoke's suite, tic-tac-toe and minesweeper
play, and speed (same architecture, same readout, same temperature). The one measured regression is OpenJev, under one point.
v10 continues the v8 weights that were on the Hub; the v9 terse-bucket data described in the README is not in it.

## decider-2b v9: terse buckets and command safety

Teacher-written routing messages over plain option lists (`support`, `help`, `account`, no descriptions) and labelled shell
commands. Held-out terse-bucket routing, generic / specific / catch-all: 0.86 / 0.95 / 0.88 (v8: 0.59 / 0.96 / 0.93); the 94-task
set unchanged. v9 was described in the README but the Hub weights stayed v8, so v10 continues v8 and does not contain this data.

## decider-2b v8: isolated Score levels, generic options, second prompt layout

Every Score level judged in its own row with normalised fits; teacher-written custom questions with a generic option next to a
catch-all; the schema-first (cacheable) layout trained 50/50 with state-first. The weights are kept under the Hub tag `v8`.

## decider-2b v6 to v7: the input shapes Jev accepts

Described options, up to 255 options with one label token each, JSON states with path references, long inputs, and the
`POST /v1/systemone` request shape. Details in `HISTORY.md`, section "v6".

## decider-2b v5: the proper abstention fix

An abstain option is added to 10% of questions with three or more options; in a quarter of those the option list is replaced by
labels from an unrelated task so that the abstain option is correct.

## decider-2b v4: situation-to-action data and ten games

Next-action choice from agent trajectories and game states behind one typed interface (`decider/games/`), plus the Super Mario
Bros demo; the montage at the top of the README is this version.

## decider-2b v1 to v3: the one-pass readout

Cross-entropy on the letter logits at the answer slot over the public decision mixture, one temperature fitted on in-task
data. `HISTORY.md` has the per-stage numbers.
