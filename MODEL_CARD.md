---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass]
---

# decider-decider-2B: typed decisions with calibrated probabilities in one forward pass

An open replication of the "System One model" idea: a language model that does not
generate text. It reads a context plus one or more typed questions, each with an
explicit option list, and returns a probability distribution over the options for
every question from a single forward pass. No decoding, no JSON parsing, no
schema violations. It is meant to be called from software, not chatted with.

Base model: [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (1.9B parameters),
fully fine-tuned for one epoch (942k examples, 183M tokens, 2.5 hours on one
NVIDIA GH200) with cross-entropy, a proper scoring rule, on a mixture of 64
public decision datasets, then continued for one epoch on 45k situation-to-action
examples (agent trajectories, web element choice, synthetic situations, game states)
with a replay of the general mixture (v4). v6 to v8 continue on the input shapes of TypeSafe's Jev API (described options, up to
255 options, JSON states with path references, long inputs, questions scored independently), on teacher-written custom questions
(free-form yes/no, user-named options, a generic option next to a catch-all), on a second, cacheable prompt layout, and on
isolated Score levels. This card describes v9 (v8 plus teacher-written terse-bucket routing messages and shell-command safety data).

## Usage

```python
from decider.infer import Decider          # decider/ is included in this repo
d = Decider("<this repo>")
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': 0.99, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.99, 'probs': {...}}]
```

`decide_batch` scores many contexts, each with many questions, in one call.
Set `abstain_below=t` to return `None` for decisions with confidence under `t`
(route to a human). 2 to 255 options per question (v6; more than 10 options use one label token per
option, see `decider/prompt.py`).

The same request shape as TypeSafe's Jev (`POST /v1/systemone`), in process or over HTTP:

```python
d.system_one({"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
              "refund_policy": "Duplicate charges are eligible for a refund."},
             {"department": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                          "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
              "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
              "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
# {"model": "decider-v6", "answers": {"department": {"type": "choice", "choice": "billing", "confidence": ..., "certainty": ..., "probabilities": {...}},
#  "refund_requested": {"type": "noul", "noul": ...}, "frustration": {"type": "score", "score": ..., "legend": {...}, ...}}, "usage": {...}}
```

State may be a string, object or array (up to 32k tokens with the questions); `instructions` and every option
description may be a string or any JSON value; question ids are never shown to the model. Each question is scored in
its own row, so answers do not depend on which other questions are asked (`independent=False` packs them into one row,
about half the latency for short states). Each Score level is likewise judged in its own row, without its number or its
neighbours, and the per-level fits are normalised (`"isolated": false` on a question restores listwise scoring); the answer
also reports `level_fit` and their sum `fit_mass` (near 1 when exactly one level fits).

For a fixed set of questions, `s = d.schema(questions)` computes the question prefix once and `s(state)` / `s.batch(states)`
then run only the state (1.2-2.4x faster per request, up to 19x per batch). It uses a questions-first prompt layout that costs
accuracy: about 1.5 points on fixed label sets, 5 on per-example options, more on 50+ options and multi-thousand-token states. `decider.serve` exposes the same thing as `POST /v1/systemone`; the official
`typesafe-sdk` works against it unchanged with `TYPESAFE_BASE_URL` pointing at the server.

Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton
kernels for the Qwen3.5 linear-attention layers; the model runs without it but
several times slower). Python 3.11+ recommended so those kernels can use
`torch.compile`.

Without the helper package, the same computation in plain `transformers`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(REPO); m = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).cuda().eval()
prompt = ("Context:\nMy card was charged twice for the same purchase.\n\n"
          "Question: Which department should handle this?\nOptions:\n(A) billing\n(B) technical support\n(C) sales\nAnswer: (")
ids = tok(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    logits = m(**ids).logits[0, -1]
letters = [tok.encode(L, add_special_tokens=False)[0] for L in "ABC"]
probs = torch.softmax(logits[letters].float(), -1)      # -> P(billing), P(technical support), P(sales)
```

For several questions in one pass, append further `Question k: ... Answer k: (`
blocks and read the logits at each `(` position (see `decider/prompt.py`).

## How it works

Prompt: `Context: ...` followed by, for each question, the question text, the
numbered options `(A) ... (B) ...`, and an answer slot `Answer k: (`. The hidden
state at each slot is projected with the option-letter rows of the LM head and
softmaxed over the valid letters. Letters are never generated, so all slots are
read from one pass. Large label sets were sub-sampled to at most 10 options per
training example (gold always kept, order shuffled), so the model conditions on
the supplied candidates rather than a fixed head.

## Field types

* **bool** (`noul`): probability of "yes".
* **choice**: argmax option, its probability, and the full distribution.
* Jev names: `noul` (bool), `choice` with `criteria` {name: description | JSON | null}, `score` with `criteria`
  [level descriptions]. Choice and score answers carry `confidence` (top probability, the calibrated number) and
  `certainty` (1 - normalised entropy of the distribution).
* **scale**: an ordered legend (e.g. 0: none ... 3: high); returns the expected
  level (`score`), the probability of the most likely level, and the distribution.

## Training data

69 public datasets plus synthetic situations, up to 20k examples each (`decider/data.py`, `decider/data2.py`, `decider/data3.py`):
intent detection, ticket routing, topic classification, sentiment, emotion,
moderation (toxicity, hate, spam, jailbreak, safety), NLI, paraphrase, fact
verification, passage relevance, reading comprehension, multiple-choice QA,
ordinal rating scales (HelpSteer2 attributes, STS-B, hate-speech intensity,
LIAR2 truthfulness), pairwise response preference (HelpSteer3, UltraFeedback,
SHP, HH-RLHF) and tool selection (Glaive, ToolACE).
v4 adds next-action choice from agent trajectories (AgentGym AgentTraj-L), web element
choice (Mind2Web), 1.5k synthetic situations written by Qwen3.5-27B, and teacher-labelled
states from Pong, Breakout, CliffWalking, MiniGrid and Super Mario Bros.
Abstention augmentation: in 10% of questions with three or more options an abstain
option with one of twelve wordings is added; in a quarter of those the whole option list is
replaced by labels from an unrelated task, making the abstain option correct.

## Evaluation

| Model | Split | Acc | NLL | Brier | ECE | AURC | Acc@80% |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B-Base, zero-shot | in-task (64) | 0.620 | 0.908 | 0.493 | 0.121 | 0.280 | 0.663 |
| Qwen3.5-2B-Base, zero-shot | held-out (23) | 0.642 | 0.853 | 0.460 | 0.105 | 0.242 | 0.685 |
| Qwen3.5-4B-Base, zero-shot | in-task (64) | 0.695 | 0.768 | 0.405 | 0.090 | 0.206 | 0.742 |
| Qwen3.5-4B-Base, zero-shot | held-out (23) | 0.711 | 0.734 | 0.390 | 0.089 | 0.169 | 0.761 |
| **this model (v5)** | in-task (69) | 0.815 | 0.445 | 0.248 | 0.028 | 0.093 | 0.866 |
| **this model (v5)** | held-out (24) | 0.738 | 0.678 | 0.360 | 0.084 | 0.145 | 0.793 |
| **this model (v6, T=1.15)** | in-task (69) | 0.813 | 0.450 | 0.251 | 0.032 | 0.094 | 0.864 |
| **this model (v6, T=1.15)** | held-out (24) | 0.736 | 0.664 | 0.358 | 0.084 | 0.145 | 0.793 |
| **this model (v8, T=1.30)** | in-task (69) | 0.811 | 0.460 | | 0.037 | | |
| **this model (v8, T=1.30)** | held-out (24) | 0.741 | 0.655 | | 0.088 | | |
| **this model (v9, T=1.36)** | in-task (69) | 0.812 | 0.464 | | 0.041 | | |
| **this model (v9, T=1.36)** | held-out (24) | 0.741 | 0.655 | | 0.087 | | |
| v8, questions-first layout (schema cache), T=1.18 | held-out (24) | 0.707 | 0.757 | | 0.104 | | |


Per-task accuracy / ECE on the held-out datasets:

| Task | Qwen3.5-2B-Base, zero-shot | Qwen3.5-4B-Base, zero-shot | this model |
|---|---|---|---|
| abstain_probe | 0.377 / 0.193 | 0.453 / 0.242 | 0.785 / 0.037 |
| ade | 0.794 / 0.117 | 0.746 / 0.101 | 0.808 / 0.038 |
| arena_pref | 0.382 / 0.228 | 0.434 / 0.159 | 0.474 / 0.144 |
| bbc_news | 0.910 / 0.010 | 0.927 / 0.024 | 0.941 / 0.016 |
| cb | 0.554 / 0.115 | 0.696 / 0.087 | 0.893 / 0.115 |
| cr_reviews | 0.882 / 0.073 | 0.923 / 0.015 | 0.902 / 0.020 |
| dolly_category | 0.269 / 0.137 | 0.351 / 0.149 | 0.318 / 0.211 |
| fin_phrasebank | 0.560 / 0.032 | 0.713 / 0.050 | 0.660 / 0.082 |
| fin_sentiment | 0.345 / 0.335 | 0.437 / 0.303 | 0.769 / 0.031 |
| hermes_tools | 0.704 / 0.067 | 0.702 / 0.160 | 0.737 / 0.155 |
| massive_scenario | 0.706 / 0.094 | 0.707 / 0.032 | 0.815 / 0.023 |
| paws | 0.759 / 0.123 | 0.834 / 0.018 | 0.691 / 0.221 |
| pubmedqa | 0.728 / 0.042 | 0.768 / 0.082 | 0.790 / 0.063 |
| quality | 0.457 / 0.205 | 0.519 / 0.184 | 0.513 / 0.137 |
| reward_bench | 0.624 / 0.074 | 0.772 / 0.045 | 0.799 / 0.043 |
| sciq | 0.978 / 0.011 | 0.988 / 0.019 | 0.982 / 0.017 |
| social_iqa | 0.659 / 0.089 | 0.739 / 0.053 | 0.712 / 0.060 |
| strategyqa | 0.566 / 0.029 | 0.646 / 0.037 | 0.613 / 0.069 |
| student_questions | 0.903 / 0.059 | 0.933 / 0.021 | 0.927 / 0.030 |
| trec | 0.696 / 0.062 | 0.822 / 0.050 | 0.766 / 0.036 |
| truthfulqa | 0.460 / 0.111 | 0.591 / 0.113 | 0.497 / 0.091 |
| tweet_irony | 0.511 / 0.158 | 0.676 / 0.073 | 0.779 / 0.059 |
| xstory_cloze | 0.941 / 0.061 | 0.981 / 0.041 | 0.965 / 0.028 |

| Task | Qwen3.5-2B-Base, zero-shot | Qwen3.5-4B-Base, zero-shot | this model |
|---|---|---|---|
| abstain_probe | 0.377 / 0.193 | 0.453 / 0.242 | 0.785 / 0.037 |
| ade | 0.794 / 0.117 | 0.746 / 0.101 | 0.808 / 0.038 |
| arena_pref | 0.382 / 0.228 | 0.434 / 0.159 | 0.474 / 0.144 |
| bbc_news | 0.910 / 0.010 | 0.927 / 0.024 | 0.941 / 0.016 |
| cb | 0.554 / 0.115 | 0.696 / 0.087 | 0.893 / 0.115 |
| cr_reviews | 0.882 / 0.073 | 0.923 / 0.015 | 0.902 / 0.020 |
| dolly_category | 0.269 / 0.137 | 0.351 / 0.149 | 0.318 / 0.211 |
| fin_phrasebank | 0.560 / 0.032 | 0.713 / 0.050 | 0.660 / 0.082 |
| fin_sentiment | 0.345 / 0.335 | 0.437 / 0.303 | 0.769 / 0.031 |
| hermes_tools | 0.704 / 0.067 | 0.702 / 0.160 | 0.737 / 0.155 |
| massive_scenario | 0.706 / 0.094 | 0.707 / 0.032 | 0.815 / 0.023 |
| paws | 0.759 / 0.123 | 0.834 / 0.018 | 0.691 / 0.221 |
| pubmedqa | 0.728 / 0.042 | 0.768 / 0.082 | 0.790 / 0.063 |
| quality | 0.457 / 0.205 | 0.519 / 0.184 | 0.513 / 0.137 |
| reward_bench | 0.624 / 0.074 | 0.772 / 0.045 | 0.799 / 0.043 |
| sciq | 0.978 / 0.011 | 0.988 / 0.019 | 0.982 / 0.017 |
| social_iqa | 0.659 / 0.089 | 0.739 / 0.053 | 0.712 / 0.060 |
| strategyqa | 0.566 / 0.029 | 0.646 / 0.037 | 0.613 / 0.069 |
| student_questions | 0.903 / 0.059 | 0.933 / 0.021 | 0.927 / 0.030 |
| trec | 0.696 / 0.062 | 0.822 / 0.050 | 0.766 / 0.036 |
| truthfulqa | 0.460 / 0.111 | 0.591 / 0.113 | 0.497 / 0.091 |
| tweet_irony | 0.511 / 0.158 | 0.676 / 0.073 | 0.779 / 0.059 |
| xstory_cloze | 0.941 / 0.061 | 0.981 / 0.041 | 0.965 / 0.028 |

| Task | Qwen3.5-2B-Base, zero-shot | Qwen3.5-4B-Base, zero-shot | this model |
|---|---|---|---|
| ade | 0.794 / 0.117 | 0.746 / 0.101 | 0.827 / 0.023 |
| bbc_news | 0.910 / 0.010 | 0.927 / 0.024 | 0.919 / 0.025 |
| cr_reviews | 0.882 / 0.073 | 0.923 / 0.015 | 0.911 / 0.019 |
| dolly_category | 0.269 / 0.137 | 0.351 / 0.149 | 0.316 / 0.212 |
| fin_phrasebank | 0.560 / 0.032 | 0.713 / 0.050 | 0.640 / 0.125 |
| fin_sentiment | 0.345 / 0.335 | 0.437 / 0.303 | 0.773 / 0.029 |
| massive_scenario | 0.706 / 0.094 | 0.707 / 0.032 | 0.822 / 0.017 |
| paws | 0.759 / 0.123 | 0.834 / 0.018 | 0.693 / 0.189 |
| pubmedqa | 0.728 / 0.042 | 0.768 / 0.082 | 0.768 / 0.043 |
| sciq | 0.978 / 0.011 | 0.988 / 0.019 | 0.985 / 0.016 |
| social_iqa | 0.659 / 0.089 | 0.739 / 0.053 | 0.712 / 0.055 |
| strategyqa | 0.566 / 0.029 | 0.646 / 0.037 | 0.594 / 0.095 |
| student_questions | 0.903 / 0.059 | 0.933 / 0.021 | 0.930 / 0.017 |
| trec | 0.696 / 0.062 | 0.822 / 0.050 | 0.772 / 0.044 |
| truthfulqa | 0.460 / 0.111 | 0.591 / 0.113 | 0.541 / 0.062 |
| tweet_irony | 0.511 / 0.158 | 0.676 / 0.073 | 0.754 / 0.054 |

| Task | Qwen3.5-2B-Base, zero-shot | Qwen3.5-4B-Base, zero-shot | this model (200k-example run) |
|---|---|---|---|
| ade | 0.794 / 0.117 | 0.746 / 0.101 | 0.808 / 0.046 |
| bbc_news | 0.910 / 0.010 | 0.927 / 0.024 | 0.928 / 0.014 |
| cr_reviews | 0.882 / 0.073 | 0.923 / 0.015 | 0.915 / 0.012 |
| dolly_category | 0.269 / 0.137 | 0.351 / 0.149 | 0.325 / 0.193 |
| fin_phrasebank | 0.560 / 0.032 | 0.713 / 0.050 | 0.652 / 0.108 |
| fin_sentiment | 0.345 / 0.335 | 0.437 / 0.303 | 0.796 / 0.042 |
| massive_scenario | 0.706 / 0.094 | 0.707 / 0.032 | 0.808 / 0.022 |
| paws | 0.759 / 0.123 | 0.834 / 0.018 | 0.713 / 0.139 |
| pubmedqa | 0.728 / 0.042 | 0.768 / 0.082 | 0.772 / 0.048 |
| sciq | 0.978 / 0.011 | 0.988 / 0.019 | 0.983 / 0.021 |
| social_iqa | 0.659 / 0.089 | 0.739 / 0.053 | 0.703 / 0.060 |
| strategyqa | 0.566 / 0.029 | 0.646 / 0.037 | 0.597 / 0.089 |
| student_questions | 0.903 / 0.059 | 0.933 / 0.021 | 0.927 / 0.029 |
| trec | 0.696 / 0.062 | 0.822 / 0.050 | 0.816 / 0.039 |
| truthfulqa | 0.460 / 0.111 | 0.591 / 0.113 | 0.529 / 0.058 |
| tweet_irony | 0.511 / 0.158 | 0.676 / 0.073 | 0.769 / 0.048 |


Probabilities use a temperature of 1.05 fitted on in-task data (stored in `decider_config.json`, applied by the helper).
*In-task* = test splits of the training datasets (in-task rows for the zero-shot baselines cover the original 64). *Held-out* = 23 datasets never
seen in training: TREC, BBC news, PAWS, SciQ, Social IQa, StrategyQA, PubMedQA,
TruthfulQA, tweet irony, financial sentiment, ADE, MASSIVE scenario, student
question categories, Dolly categories, CR reviews, Financial PhraseBank,
CommitmentBank, QuALITY, XStoryCloze, RewardBench, Arena preferences (3-way),
Hermes tool selection, and an abstention probe (held-out classification tasks
where in half the cases the correct option is absent and "none of the above" is
right). Chance accuracy is 0.33 on both sets. ECE = expected calibration error
(15 bins), AURC = area under the risk-coverage curve, acc@80 = accuracy on
the 80% most confident decisions.

## Speed

One NVIDIA GH200, bf16. `decider.infer.Decider` uses shape-bucketed CUDA
graphs (`decider/engine.py`); the micro-batching server is `decider/serve.py`
in the GitHub repo. Support-ticket contexts of ~230 tokens with 3 to 5 typed
questions each:

| setting | p50 latency | throughput |
|---|---|---|
| single request, eager PyTorch | 49 ms | |
| single request, CUDA graphs + torch.compile (helper default) | 4.0 ms | |
| batch of 32, in-process, bf16 | 70 ms | ~1370 decisions/s |
| batch of 32, in-process, FP8 linears | 58 ms | ~1670 decisions/s |
| HTTP server (FP8), 1 client | 6.8 ms | 134 req/s |
| HTTP server (FP8), 64 clients | 126 ms | 431 req/s, 2152 decisions/s |

FP8 (e4m3 weights, per-token activation scales) changes accuracy and calibration by
less than the evaluation noise (18-task check: accuracy 0.833 vs 0.835, ECE equal).

## Limitations

* English only. Options must be short phrases; free-text fields are not supported.
* Calibration is measured on public datasets; verify it on your own labelled
  data before using confidence for routing.
* No reasoning: this is a fast pattern-matching decision model, not a chat model.
* Inputs up to 32k tokens are accepted (v6 trained to 16k, probed to 30k). Plain long reading works (QuALITY, whole
  5-8k-token article: 0.71, against 0.51 clipped). Picking one record out of a long JSON array by position is the weak
  case: 0.70 with 4 records, 0.64 with 16, 0.49 with 64 (single-record ceiling 0.72); address records by key where
  possible. The helper writes `"_index": i` into arrays of 8 or more elements, which recovers part of it (0.57 at 64).
* Full label sets (v6): 0.84 on held-out HWU64 (64 options), 0.76 TREC-fine (50), 0.87 DBpedia level 3 (219), 0.70 DBpedia
  level 2 (70, ECE 0.14: the least calibrated of these). In-task CLINC 151-way 0.88 against 0.98 with 10 sampled options.
* Questions packed into one row (`independent=False`, or `decide` with several questions) still see earlier question
  texts: reversing their order changes up to 12% of answers on multi-question tasks. The default `system_one` path
  scores each question alone and has no such dependence.
* Held-out text game Freeway fell from 6 (v4) to 3 (v5) to 0 (v6) over three episodes; trained games are unchanged.
* Knowledge-heavy multiple choice (MMLU, MedQA, ARC) improves only modestly over the
  base model; fine-tuning on decisions does not add world knowledge.
* Compared with a run on the 47-dataset v1 mixture, adding the v2 datasets
  raised held-out accuracy but lowered two held-out tasks: Hermes tool selection
  (0.80 to 0.74) and TruthfulQA (0.54 to 0.50).
* Scale fields are the least trained type; expect wider distributions there.
* Abstention (v5): an option such as "none of the above", "other" or "unsure" is chosen when
  nothing on offer fits the situation, not when the exact fine-grained label is merely absent
  (it then takes the best available option). Trained with twelve abstain wordings and
  off-topic option lists; on a held-out probe with off-topic option lists it scores 0.83
  (v4: 0.68). Earlier versions (v4 and before) had learned the literal phrase as an abstain
  signal; the bundled helper's rewrite for that is disabled for v5 via `decider_config.json`.
* Catch-all options next to generic ones ("support" vs "other"): v6 sent in-scope messages that fit only the generic option to
  the catch-all (0.60 on a hand-written battery, 0.50 on held-out teacher-written routing messages). v8: 0.85 and 0.94, with the
  catch-all cases at 0.95 and 0.90. Question wordings far from the training data (public datasets plus 24k teacher-written
  questions) remain the main risk; verify on your own examples.
* Generic buckets with plain names (`support`, `help`, `account`) next to a catch-all: v9 picks the bucket when it should (held-out
  terse-bucket messages 0.59 to 0.86; the `check_balance / approve_transfer / support / other` case that v8 got wrong now goes to
  `support` at 0.79-0.85) at a small cost on the catch-all side (0.93 to 0.88 on that probe; 0.62 to 0.58 on the abstention probe).
* Isolated Score levels match listwise scoring within about a point (LIAR2: 3 points lower). Levels should describe situations,
  not degrees.
* One in-task dataset, `tweet_hate` (SemEval-2019 HatEval), stays near chance on its
  test split. That split is known to differ from its training split in collection
  and label definition; the number is reported as measured.

## Reproduction

Code, data registry, training and evaluation scripts: https://github.com/Mapika/decider
(`decider/` in this model repo is the inference subset of that package).
