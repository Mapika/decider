# decider — one-pass typed decisions with calibrated probabilities

Goal: reproduce the shape of TypeSafe AI's *Jev* (typed decisions with calibrated
probabilities, all fields from one forward pass, no token generation) on a
single GH200, starting from a modern small open base model.

## Design
* **Backbone**: `Qwen/Qwen3.5-2B-Base` (Feb 2026, hybrid linear-attention).
* **Interface**: `Context` + N typed `Question`s, each with an explicit option list.
  Every question gets an answer slot `Answer k: (`; the logits at that slot are
  restricted to the option-letter tokens `A..J` and softmaxed. All N decisions
  come from one forward pass (no decoding).
* **Candidate conditioning**: label sets larger than 10 are sub-sampled per
  example (gold always kept) and shuffled, so the model must read the options.
* **Training objective**: cross-entropy (a proper scoring rule) on ~60 public
  decision datasets (intents, routing, moderation, NLI, MCQ, sentiment, ...).
  Optional Brier term. Post-hoc temperature is fitted on in-task data and
  tested on held-out tasks.
* **Evaluation**: 47 in-task + 16 held-out (never trained) tasks. Metrics:
  accuracy, NLL, Brier, ECE, AURC, selective accuracy at 80% coverage.
  The held-out set is the important one: does calibration transfer to tasks
  the model never saw?

## Serving

```bash
DECIDER_MODEL=runs/r3_v2/model .venv312/bin/uvicorn decider.serve:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/decide -H 'content-type: application/json' -d '{"context": "My card was charged twice.",
  "schema": {"Which department?": {"type": "choice", "options": ["billing", "technical", "sales"]},
             "Refund needed?": {"type": "bool"},
             "Urgency?": {"type": "scale", "legend": {"0": "low", "1": "medium", "2": "high"}}}}'
```
Requests arriving within a few milliseconds are scored in one forward pass. `decider.infer.Decider`
uses the same CUDA-graph engine in-process.

Measured on one GH200 (support-ticket contexts, ~230 tokens, 5 typed fields per request):

| | latency p50 | throughput |
|---|---|---|
| in-process `Decider.decide`, eager PyTorch | 49 ms | |
| in-process `Decider.decide`, CUDA-graph engine | 6.6 ms | |
| HTTP, 1 client | 10 ms | 97 req/s |
| HTTP, 4 clients | 25 ms | 150 req/s |
| HTTP, 64 clients | 231 ms | 263 req/s = 1314 decisions/s |

The single-request cost was launch overhead (1742 kernels per forward, GPU busy 5 ms of 44);
CUDA graphs remove it. Batched throughput (~90k tokens/s) is compute-bound; torch.compile
did not help (graph breaks inside transformers).

## Layout
```
decider/data.py        task registry -> Example(context, [Q(text, options, gold)])
decider/prompt.py      prompt/slot construction, multi-question packing
decider/model.py       DecisionModel: hidden state at slots -> letter logits
decider/train.py       finetune (bucketed shapes, bf16, grad-ckpt)
decider/evaluate.py    per-task metrics, saves probs
decider/report.py      side-by-side comparison + temperature scaling
decider/engine.py      Engine: shape-bucketed CUDA graphs (7x lower single-request latency than eager)
decider/serve.py       micro-batching HTTP server (POST /decide {context, schema})
decider/loadtest.py    closed-loop load test against the server
decider/bench_engine.py  eager vs CUDA graph vs torch.compile on fixed shapes
decider/bench_latency.py  decisions/s and latency of the one-pass interface (eager)
data/tasks.pkl    cached examples (python -m decider.data)
runs/             zs_2b, zs_4b (zero-shot baselines), r1_200k, ...
```
Model weights: https://huggingface.co/Mapika/decider-2b (after upload).
Setup: `uv venv --python 3.12 .venv312 && uv pip install -p .venv312/bin/python torch transformers peft accelerate datasets pillow "numpy<2" scikit-learn flash-linear-attention fastapi "uvicorn[standard]" httpx`.
