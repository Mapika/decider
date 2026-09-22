---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass]
---

# decider-0.8b: typed decisions with calibrated probabilities in one forward pass

The smallest decider: a language model that does not generate text. It reads
a **state** (a string or any JSON value) and a set of **typed questions** and returns a probability distribution for every question
from a single forward pass: Choice (2-255 options, optionally described), Score (2-10 described levels), Noul (probability of yes).
No decoding, no parsing, no output outside the options you defined. Same code, same wire format (`POST /v1/systemone`, TypeSafe
Jev's format), same training recipe as the 2B: one epoch of `scripts/train.sh full` from `Qwen/Qwen3.5-0.8B-Base` over the full
mixture (1.47M examples, 455M tokens, 4.5 h on one GH200). 

**Contents:** [The decider family](#the-decider-family) · [Usage](#usage) · [How it compares with the 2B](#how-it-compares-with-the-2b) · [Limitations](#limitations) · [Changelog](#changelog)

## The decider family

All six repositories share one interface (`decider.infer.Decider`, `POST /v1/systemone` in TypeSafe's format) and one
readout: the letter logits at an answer slot, softmaxed over the options. Pick by size and input.

| model | base | weights | use it for | numbers |
|---|---|---|---|---|
| [decider-2b](https://huggingface.co/Mapika/decider-2b) v10 | Qwen3.5-2B-Base | 3.5 GB bf16 | the default: routing, classification, judgments, browser agents; 4 ms per request with CUDA graphs on one GPU | regression set 0.805 in-task / 0.755 held-out; live browser 93%; Bespoke suite 0.704 |
| [decider-4b](https://huggingface.co/Mapika/decider-4b) v1 | Qwen3.5-4B-Base | 8.4 GB bf16 | the middle point: knowledge and reasoning questions above the 2B in a dense 8.4 GB model; no RL stage | 0.834 / 0.788, above the 2B on 87 of 95 tasks; JevBench hard 0.541; Bespoke 0.757 |
| [decider-35b-a3b](https://huggingface.co/Mapika/decider-35b-a3b) v1 | Qwen3.5-35B-A3B-Base (3B active) | 65 GB bf16 | when accuracy is worth 3 to 4 times the cost per decision: knowledge and multi-step questions, long policies | 0.855 / 0.810, above the 2B on 93 of 95 tasks; JevBench hard 0.676; Bespoke 0.774; no RL stage |
| [decider-35b-a3b-nvfp4](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4) | the 35B in NVFP4 | 19.6 GB | the 35B on Blackwell through vLLM or TensorRT-LLM | 1.0 to 1.5 points under bf16 on the measured fixtures |
| [decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | Qwen3.5-0.8B-Base | 1.4 GB bf16 | the smallest: routing, yes/no and short-state lookups within 1 to 4 points of the 2B, 1.5x faster | 0.776 / 0.707 on the single-run protocol (2B: 0.809 / 0.739) |
| [decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) | Qwen3.5-2B vision-language, v5 text weights | 4.1 GB bf16 | decisions from an image plus a question; game frames | Visual7W 0.89; Breakout 41 from pixels |

Code, data registry, training scripts, the changelog and the per-version history: https://github.com/Mapika/decider.

## Usage

```python
# pip install git+https://github.com/Mapika/decider
from decider.infer import Decider
d = Decider("Mapika/decider-0.8b")
d.system_one(
    {"ticket": "I was charged twice for order A-104. Please refund the duplicate."},
    {"team": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs, outages", "other": None}},
     "refund_requested": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
     "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
```

`decide_batch`, `schema` (the cached question set), `decider.serve` and the TypeSafe request shape work as in the decider-2b
card; `decider/` in this repository is the inference subset of the GitHub package.

## How it compares with the 2B

Same 93 public tasks, same protocol, one temperature fitted on in-task data (it came out at 1.03 for both: the recipe calibrates
at every size). "Held-out" means no example of that dataset was trained on.

| | decider-0.8b | decider-2b (same single-run recipe) |
|---|---|---|
| in-task accuracy / ECE, 69 tasks | 0.776 / 0.032 | 0.809 / 0.030 |
| held-out accuracy / ECE, 24 tasks | 0.707 / 0.096 | 0.739 / 0.086 |
| schema-first layout (the cacheable one), in-task / held-out accuracy | 0.770 / 0.699 | 0.790 / 0.707 |
| teacher-written custom questions, held-out domains: noul / choice / score | 0.94 / 0.95 / 0.81 | 0.98 / 0.97 / 0.83 |
| terse-bucket routing, held-out domains: generic / specific / catch-all | 0.87 / 0.93 / 0.84 | 0.91 / 0.94 / 0.92 |
| JSON state, one of 16 / 64 records named by path (64 with indices written in) | 0.58 / 0.52 (0.62) | 0.59 / 0.53 (0.63) |
| all 64 / 50 / 70 / 219 labels at once: HWU64, TREC-fine, DBpedia L2, L3 (held-out) | 0.81 / 0.63 / 0.66 / 0.83 | 0.85 / 0.76 / 0.72 / 0.86 |
| QuALITY, whole article (5-8k tokens) | 0.63 | 0.68 |
| isolated Score levels vs listwise (teacher-written, held-out domains) | 0.83 vs 0.82, fits sum to 1.02 | 0.84 vs 0.84 |
| hand-written battery: generic option right / catch-all right | 0.95 / 0.95 | 0.90 / 0.85 |

What the smaller model gives up is knowledge, not the decision format: the largest drops are TruthfulQA (0.41 vs 0.55), OpenBookQA
(0.67 vs 0.81), HellaSwag (0.76 vs 0.88) and ARC (0.74 vs 0.86), and wide label sets that need fine distinctions (TREC-fine). Routing,
classification, yes/no judgments and JSON lookups on short states are within one to four points of the 2B. It is a weaker player:
Pong and Breakout stay at the scripted teacher's level, CliffWalking fails (it walks off the cliff), held-out Freeway scores 0.
The regression set runs about 1.5x faster than on the 2B; bf16 weights are 1.5 GB.

## Limitations

Those of decider-2b, more so: a small model without reasoning; English only; rules written into the question ("fill if empty,
otherwise skip") are not followed reliably, so state the decision as a plain question with described options; knowledge-heavy
multiple choice is close to the base model; calibration is measured on public datasets and teacher-labelled probes, not on your
traffic. The teacher-written training data comes from Qwen3.5-27B and carries its biases.

## Changelog

| version | what changed |
|---|---|
| **v1** (these weights) | one epoch of `scripts/train.sh full` from Qwen3.5-0.8B-Base, the single-run form of the supervised recipe that produced decider-2b v8 to v9 |

Every decider release is listed in [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) of the
GitHub repository. Code and the training recipe: https://github.com/Mapika/decider.
