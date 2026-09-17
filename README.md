# decider: one-pass typed decisions with calibrated probabilities

![the model playing ten games from text state descriptions, plus Mario with the RL checkpoint](media/montage.gif)

A language model that does not generate text. It reads a **state** and a set of **typed questions** and returns, from a
single forward pass, a probability distribution for every question: no decoding, no parsing, no output outside the
options you defined. It is an open reproduction of the "System One" model class (TypeSafe AI's *Jev*), built on
`Qwen/Qwen3.5-2B-Base` and trained on one GH200. Weights: https://huggingface.co/Mapika/decider-2b

```python
from decider.infer import Decider
d = Decider("Mapika/decider-2b")
d.system_one(
    {"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
     "refund_policy": "Duplicate charges are eligible for a refund."},
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
     "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
     "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
# {"answers": {"department": {"choice": "billing", "confidence": 0.97, "certainty": ..., "probabilities": {...}},
#              "refund_requested": {"noul": 0.93}, "frustration": {"score": 1.2, "probabilities": {...}, "level_fit": {...}}}}
```

## What it does

| | |
|---|---|
| question types | **Choice** (2-255 options, each optionally with a description or a JSON rubric), **Score** (2-10 described levels, returns the expected level), **Noul** (probability of yes) |
| state | a string, or any JSON value; questions can name a part of it by path (`` `tickets[3].text` ``); up to 32k tokens |
| independence | every question is scored on its own: adding, removing or reordering questions cannot change another answer |
| isolated levels | every Score level is judged alone (it sees neither its number nor its neighbours); the per-level fits are normalised |
| abstention | a catch-all option ("other", "none of the above", ...) is chosen when nothing on offer fits |
| calibration | trained with a proper scoring rule; one temperature fitted on in-task data, checked on held-out tasks |
| wire format | `POST /v1/systemone` is TypeSafe's format; their SDKs work unchanged with `TYPESAFE_BASE_URL` pointing at `decider.serve` |
| speed | CUDA-graph engine, FP8, and a **schema cache**: a fixed question set is computed once, requests run only the state |

## How it works

`decider/prompt.py` renders a request as text with one answer slot per question; `decider/model.py` reads the hidden state at
each slot, projects it onto one label token per option (A-J, then K-Z and two-letter tokens up to 255) and softmaxes over the
valid ones. Letters are never generated, so all slots come out of one pass.

Two layouts are trained, 50/50. **State-first** (`Context ... Question ... Options ... Answer: (`) is the original.
**Schema-first** puts the question/option blocks before the state, so they are a prefix that does not depend on the state:
`decider/schema_engine.py` runs that prefix once per schema, keeps its cache (attention K/V of the 6 full-attention layers, conv
and recurrent state of the 18 delta-net layers) read-only, and a request runs only `Context: <state>` plus the slots, as a CUDA
graph per (batch, length) bucket. Independent scoring uses one cached prefix per question (or per Score level) and one row each.
For the state-first layout the same idea works the other way round (`Engine.score_shared`): the state is run once and its
cache forked to every question, which is the delta-net equivalent of a block attention mask.

## Layout

```
decider/prompt.py        the two prompt layouts, label table, answer slots
decider/model.py         DecisionModel: backbone -> slot hidden states -> option logits
decider/systemone.py     Choice / Score / Noul with criteria -> prompt rows; typed answers; isolated levels; index annotation
decider/infer.py         Decider: system_one(), schema() (compiled, cached question sets), decide()
decider/engine.py        shape-bucketed CUDA graphs, torch.compile, shared-prefix scoring;  fp8.py  e4m3 linears
decider/schema_engine.py schema cache (read-only prefix cache + suffix graphs)
decider/serve.py         HTTP server: /v1/systemone, /decide, continuous batching per schema and length bucket
decider/data/            task registry (~95 public datasets), augment.py (all input-shape augmentations), mixture.py (the mixture
                         and the probes), teacher_*.py (label descriptions, custom questions, situations from a local 27B teacher)
decider/train.py         cross-entropy fine-tune, token-bucketed batches, random layout per example, abstain augmentation
decider/evaluate.py      accuracy / NLL / Brier / ECE / AURC / selective accuracy per task;  report.py  comparisons, temperature fit
decider/probes/          hand-written batteries, question independence, isolated levels
decider/bench/           engine and schema-cache benchmarks, HTTP load test
decider/games/           ten text games + Super Mario Bros behind the same interface, imitation and PPO
decider/vision/          the vision-language variant (decisions from pixels)
teacher_data/            the teacher-written data the mixture needs (label descriptions, custom questions, routing messages, situations)
scripts/                 train.sh, evaluate.sh, serve.sh, stage_release.py, upload_hf.py
examples/                routing with confidence gates, composite scoring, hierarchical beam over Choice probabilities
docs/HISTORY.md          how the released weights were actually produced (v1 to v8) and what was measured at each stage
```

## Train

```bash
uv venv --python 3.12 .venv312 && uv pip install -p .venv312/bin/python -e ".[serve,train]"
scripts/train.sh full                       # datasets -> data/tasks.pkl -> data/mixture_full.pkl -> one epoch from Qwen3.5-2B-Base -> scripts/evaluate.sh
scripts/train.sh delta runs/some/model      # or: continue an existing decider checkpoint on the new formats + a replay sample
```

`decider/data/mixture.py` lists every component of the mixture with its size. The released weights were produced in stages
(`delta` runs on top of each other, see `docs/HISTORY.md`); `full` is the same data as a single run.

## Serve

```bash
scripts/serve.sh Mapika/decider-2b 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{"state": "My card was charged twice.",
  "questions": {"team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges, refunds", "technical": "bugs, outages"}},
                "refund": {"type": "noul", "instructions": "Is a refund needed?"}}}'
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local python your_typesafe_sdk_script.py
```

A schema seen twice gets a cached prefix and its own graphs; `DECIDER_SCHEMAS=schemas.json` preloads and compiles known schemas
before traffic. In process: `s = d.schema(questions, compile=True); s(state); s.batch(states)`.

## Results

RESULTS_PLACEHOLDER

## Limitations

LIMITATIONS_PLACEHOLDER
