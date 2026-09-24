---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass]
---

# decider-2b: typed decisions with calibrated probabilities in one forward pass

A language model that does not generate text. It reads a state and one or more typed questions, each with an explicit option
list, and returns a probability distribution over the options for every question from one forward pass. There is no decoding,
no parsing and no output outside the options you defined. It is called from software, not chatted with. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev).

Base model: [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) (1.9B parameters). The supervised stages
(v1 to v8) fine-tune it with cross-entropy, a proper scoring rule, on a mixture of about 95 public decision datasets, agent
trajectories, web element choice, game states and teacher-written custom questions, in two prompt layouts and with isolated
Score levels. **This repository holds v10**: the v8 weights continued for 384 steps of calibration-aware reinforcement learning
whose only rewards are outcomes (live browser task checkers and the exact probability laws of games), with a hard KL limit to
the v8 weights on replayed training rows. Code, data registry, training scripts and the recipe are at
https://github.com/Mapika/decider; `decider/` in this repository is the inference subset of that package. The other sizes and the vision variant are listed under The decider family.

**Contents:** [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Field types](#field-types) · [Training](#training) · [Evaluation](#evaluation) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

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
from decider.infer import Decider          # decider/ is included in this repo
d = Decider("Mapika/decider-2b")
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': 0.99, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.99, 'probs': {...}}]
```

`decide_batch` scores many states, each with many questions, in one call. `abstain_below=t` returns `None` for decisions with
confidence under `t`. A question can have 2 to 255 options (more than 10 options use one label token per option, see
`decider/prompt.py`).

The same request shape as TypeSafe's Jev (`POST /v1/systemone`), in process or over HTTP:

```python
d.system_one({"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
              "refund_policy": "Duplicate charges are eligible for a refund."},
             {"department": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                          "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
              "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
              "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
# {"model": "decider-v10", "answers": {"department": {"type": "choice", "choice": "billing", "confidence": ..., "certainty": ..., "probabilities": {...}},
#  "refund_requested": {"type": "noul", "noul": ...}, "frustration": {"type": "score", "score": ..., "legend": {...}, ...}}, "usage": {...}}
```

The state may be a string, object or array (up to 32k tokens with the questions). `instructions` and every option
description may be a string or any JSON value. Question ids are never shown to the model. Each question is scored in its own
row, so an answer does not depend on which other questions are asked (`independent=False` packs them into one row, about half
the latency for short states). Each Score level is likewise judged in its own row, without its number or its neighbours, and the
per-level fits are normalised (`"isolated": false` restores listwise scoring). The answer also reports `level_fit` and their
sum `fit_mass`, which is near 1 when exactly one level fits.

For a fixed set of questions, `s = d.schema(questions)` computes the question prefix once and `s(state)` / `s.batch(states)`
then run only the state (1.2 to 2.4x faster per request, up to 19x per batch). It uses a questions-first prompt layout that
costs accuracy: about 1.5 points on fixed label sets, 5 on per-example options, more on 50 or more options and on states of
several thousand tokens. `decider.serve` exposes the same thing as `POST /v1/systemone`; the official `typesafe-sdk` works
against it unchanged with `TYPESAFE_BASE_URL` pointing at the server.

Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton kernels for the Qwen3.5 linear-attention
layers; the model runs without it but several times slower). Python 3.11 or newer lets those kernels use `torch.compile`.

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
probs = torch.softmax(logits[letters].float() / 1.30, -1)      # -> P(billing), P(technical support), P(sales); 1.30 is the stored temperature
```

For several questions in one pass, append further `Question k: ... Answer k: (` blocks and read the logits at each `(`
position (see `decider/prompt.py`).

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled),
so the model conditions on the supplied candidates rather than on a fixed head.

## Field types

* **noul**: probability of "yes".
* **choice** with `criteria` {name: description | JSON | null}: the argmax option, its probability (`x_p_max`, the calibrated
  number), `confidence` (TypeSafe's definition, `(n·x_p_max − 1)/(n − 1)` for n options), `certainty` (1 minus the normalised
  entropy) and the full distribution.
* **score** with `criteria` [level descriptions]: the expected level, the probability of the most likely level (`x_p_max`),
  `confidence` (TypeSafe's definition: 1 minus the expected distance from the most likely level, divided by the mean distance of
  the levels from the middle of the scale, floored at 0), the distribution, and the per-level fits.

Before decider-ai 1.3.0, `confidence` in `system_one` and `POST /v1/systemone` answers was the probability that is now `x_p_max`.
The plain form (`decide`, `POST /decide`) still reports the top probability as `confidence`.

## Training

**Supervised stages (v1 to v8).** One epoch on a mixture of public decision datasets (intent detection, ticket routing, topic
classification, sentiment, emotion, moderation, NLI, paraphrase, fact verification, passage relevance, reading comprehension,
multiple-choice QA, ordinal rating scales, pairwise response preference, tool selection), then continuation epochs that added
next-action choice from agent trajectories (AgentGym), web element choice (Mind2Web), teacher-written situations and game states,
the input shapes of the Jev API (described options, up to 255 options, JSON states with path references, long inputs), teacher-
written custom questions with a generic option next to a catch-all, a second cacheable prompt layout, and isolated Score levels.
In 10% of questions with three or more options an abstain option is added; in a quarter of those the option list is replaced by
labels from an unrelated task so that the abstain option is correct. The full list of components with sizes is in
`decider/data/mixture.py` of the GitHub repository; `scripts/train.sh full` reproduces the supervised stages in one run.

**Reinforcement learning stage (v8 to v10).** 384 optimizer steps at a peak learning rate of 1e-6 (cosine, 16 warm-up steps),
selected among the checkpoints of a 576-step run. Each of the 48 iterations plays 4 live MiniWoB++ click tasks, 4 minesweeper
boards and 4 game boards (a 5x5 grid with a slippery move, draws from bags of known composition), 4 repeats each, through the
same one-pass readout that serves requests. Three loss terms use those rollouts: a PPO clipped surrogate (clip 0.2) on the
terminal outcome with a leave-one-replicate-out baseline; a proper log score of the model's stated belief about the immediate
outcome of its action against the exact law (games, minesweeper) or the realised outcome (browser); and a rendering-consistency
term that pulls the model's answer in the other prompt layout and the reversed option order toward its served answer. A fourth
term keeps the model where it was: on 8 replayed supervised rows per step, KL(v8 ‖ student) on the served distribution must
stay under 0.01 nats on average and 0.05 on any row, otherwise the step drops the reward terms and follows only the KL
gradient. Six browser tasks were held out from reward and used for validation only. No gold labels were used. The recipe and
every measurement are in `docs/RL.md` of the GitHub repository.

## Evaluation

**94 public tasks, original protocol.** Large label sets sub-sampled to 10 options; one temperature fitted on in-task data
and stored in `decider_config.json`. "In-task" means the test splits of the training datasets; "held-out" means datasets never
seen in training (TREC, BBC news, PAWS, SciQ, Social IQa, StrategyQA, PubMedQA, TruthfulQA, tweet irony, financial sentiment,
ADE, MASSIVE scenario, student question categories, Dolly categories, CR reviews, Financial PhraseBank, CommitmentBank,
QuALITY, XStoryCloze, RewardBench, Arena preferences, Hermes tool selection, and an abstention probe). ECE is the expected
calibration error with 15 bins.

| model | in-task (69 tasks) acc / NLL / ECE | held-out (24 tasks) acc / NLL / ECE |
|---|---|---|
| Qwen3.5-2B-Base, zero-shot | 0.620 / 0.908 / 0.121 | 0.642 / 0.853 / 0.105 |
| decider-2b v8, T=1.30 | 0.811 / 0.460 / 0.037 | 0.741 / 0.655 / 0.088 |
| decider-2b v9, T=1.36 | 0.812 / 0.464 / 0.041 | 0.741 / 0.655 / 0.087 |
| decider-2b v8, rebuilt set (67 / 28 tasks, see note), T=1.30 | 0.806 / 0.473 / 0.038 | 0.757 / 0.622 / 0.083 |
| **decider-2b v10 (this repository), rebuilt set, T=1.30** | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| v8, questions-first layout (schema cache), T=1.18 | 0.790 / 0.500 / 0.038 | 0.707 / 0.757 / 0.104 |

The two "rebuilt set" rows were measured after the data pipeline was rebuilt on another machine: two datasets no longer download
(TREC-fine, the game states) and the current mixture adds held-out probes, so that set has 67 in-task and 28 held-out tasks. Its
numbers are comparable to each other, not to the rows above. v10 matches v8 on it.

<details>
<summary><b>Per-task accuracy / ECE on the 28 held-out datasets, v8 against v10</b></summary>

Per-task accuracy / ECE on the held-out datasets of the rebuilt set, v8 against v10:

| task | v8 acc / ECE | v10 acc / ECE |
|---|---|---|
| abstain_probe | 0.633 / 0.112 | 0.606 / 0.134 |
| ade | 0.811 / 0.044 | 0.817 / 0.038 |
| arena_pref | 0.487 / 0.173 | 0.483 / 0.189 |
| bbc_news | 0.924 / 0.014 | 0.927 / 0.013 |
| cb | 0.911 / 0.090 | 0.857 / 0.093 |
| cr_reviews | 0.900 / 0.027 | 0.903 / 0.031 |
| dbpedia_l2 | 0.948 / 0.017 | 0.950 / 0.018 |
| dbpedia_l3 | 0.989 / 0.007 | 0.987 / 0.005 |
| dolly_category | 0.291 / 0.209 | 0.299 / 0.203 |
| fin_phrasebank | 0.684 / 0.043 | 0.694 / 0.042 |
| fin_sentiment | 0.794 / 0.069 | 0.793 / 0.058 |
| hermes_tools | 0.718 / 0.209 | 0.723 / 0.208 |
| hwu64 | 0.964 / 0.031 | 0.961 / 0.030 |
| massive_scenario | 0.766 / 0.040 | 0.756 / 0.041 |
| offtopic_probe | 0.841 / 0.033 | 0.841 / 0.027 |
| paws | 0.707 / 0.169 | 0.724 / 0.145 |
| pubmedqa | 0.752 / 0.083 | 0.756 / 0.085 |
| quality | 0.495 / 0.236 | 0.494 / 0.233 |
| quality_full | 0.505 / 0.205 | 0.508 / 0.198 |
| reward_bench | 0.825 / 0.042 | 0.819 / 0.045 |
| sciq | 0.982 / 0.022 | 0.982 / 0.024 |
| social_iqa | 0.698 / 0.072 | 0.708 / 0.077 |
| strategyqa | 0.559 / 0.123 | 0.552 / 0.138 |
| student_questions | 0.927 / 0.036 | 0.925 / 0.045 |
| trec | 0.792 / 0.057 | 0.784 / 0.066 |
| truthfulqa | 0.529 / 0.102 | 0.537 / 0.090 |
| tweet_irony | 0.801 / 0.048 | 0.795 / 0.052 |
| xstory_cloze | 0.962 / 0.017 | 0.962 / 0.017 |

</details>

**v10 against v8 on the same rows.** Every row below is scored by both models on identical inputs and seeds. Intervals are
95% bootstrap or paired intervals.

| | v8 | v10 | difference |
|---|---|---|---|
| live MiniWoB++ click tasks, 22 tasks x 8 seeds, sampled play | 83.0% | 93.2% | +10.2 (+5.1 to +15.9) |
| the 6 tasks never used for reward | 72.9% | 91.7% | +18.8 (+6.2 to +31.2) |
| same tasks, greedy play | 90.3% | 90.9% | +0.6 |
| Mind2Web element and action choice, 1,770 rows | 81.1% | 82.7% | +1.5 (+0.7 to +2.4) |
| bag-draw games, win rate, 64 boards x 4 | 35.2% | 41.4% | +6.2 (+0.8 to +11.7) |
| slippery-grid games, win rate, 64 boards x 4 | 14.1% | 18.8% | +4.7 (−2.0 to +11.3) |
| stated belief, nats above the exact law (lower is better) | 0.473 | 0.219 | |
| click-outcome prediction, log score (higher is better) | −0.349 | −0.034 | |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 78.4% / 0.594 | 80.4% / 0.585 | +2.0 (−2.0 to +5.9) |
| 847 in-task validation rows, accuracy / NLL | 83.6% / 0.443 | 83.2% / 0.444 | −0.4 (−1.3 to +0.6) |
| Bespoke's public suite, 13 subsets, macro accuracy | 0.706 | 0.704 | |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.875 / 0.441 | 1.000 / 0.889 / 0.459 | +1 / +2 items |
| OpenJev, 5,252 rows, accuracy / NLL | 64.1% / 0.906 | 63.3% / 0.916 | −0.8 (−1.3 to −0.3) |

The browser gain is in the served distribution rather than in the argmax: sampled play improves by ten points, greedy play by
under one. Tic-tac-toe and minesweeper play did not change; a 2B model without search loses most of those games either
way. The one measured regression is OpenJev, under one point. The JevBench row was read again for both versions on 2026-09-23, in process, bf16, decider-ai
1.2.1. The values first published here (standard / hard: v8 0.861 / 0.459, v10 0.847 / 0.459) came from the FP8 server of
2026-09-19 and do not reproduce item for item.

**Bespoke's public suite** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through `system_one` as
shipped). decider-2b v10 macro 0.704 / micro 0.711; v9 0.701 / 0.711; Nimble-9B 0.748 / 0.759; Jev 1.13.0 0.760 / 0.773
(the last two copied from Bespoke's report). Per-subset numbers, the JevBench public-item comparison (decider-2b v10 is at 1.000 / 0.889 / 0.459 on the easy / standard /
hard public items, against Jev 1.13.0 at 1.000 / 0.986 / 0.730) and recordings of both versions on the same browser pages and
game boards are in the GitHub README.

## Speed

One NVIDIA B300, decider-ai 1.2.1, measured 2026-09-23. Support-ticket states of about 230 tokens with 3 typed questions each
(the first 64 `support_tickets` examples). `decider.infer.Decider` uses shape-bucketed CUDA graphs; the batching server is
`decider/serve.py`, whose default since 1.1 is bf16.

| setting | p50 latency | throughput |
|---|---|---|
| single request, eager PyTorch | 18.9 ms | |
| single request, CUDA graphs + torch.compile (helper default) | 3.2 ms | |
| batch of 32, in-process, bf16 | 35.5 ms | about 2,700 decisions/s |
| batch of 32, in-process, FP8 linears | 32.3 ms | about 2,980 decisions/s |
| HTTP server `/decide` (bf16, default), 1 client | 6.1 ms | 158 req/s |
| HTTP server `/decide` (bf16, default), 64 clients | 134 ms | 436 req/s, 2,181 decisions/s |
| HTTP server `/decide` with `DECIDER_FP8=1`, 64 clients | 212 ms | 286 req/s, 1,429 decisions/s |

On the B300, FP8 is faster in process but slower through the server, so the server default is bf16. The schema-cache figures
were measured earlier on one GH200 and not repeated: with `Decider.schema`, 10 described questions on short chat messages ran at
11,180 decisions/s in a batch, and one question with 151 options at 19x the full-forward rate. FP8 (e4m3 weights, per-token
activation scales) changes accuracy and calibration by less than the evaluation noise.

## Limitations

* A 2B model without reasoning. Knowledge-heavy multiple choice (MMLU, MedQA, ARC) improves little over the base model, and a
  judgment that needs several steps should be split into several questions.
* English only. Calibration is measured on public datasets and teacher-labelled probes, not on your traffic. Check it on your
  own labels before using confidence for routing.
* v10 continues the v8 weights. The v9 data for terse bucket names (`support`, `help`, `account` next to `other`) is not in it:
  on held-out terse-bucket messages v8 chose the generic bucket correctly 59% of the time where v9 reached 86%. Name or
  describe the generic option as a bucket (`general_support`, or a description).
* Rules written into the question ("fill if empty, otherwise skip") are not followed at this size. State the decision as a
  plain question with described options.
* Picking one record out of a long JSON array by position is the least accurate input shape (0.51 with 64 records against
  0.70 with one). Address records by key, or let the helper write the index into the array (0.62).
* Full label sets cost accuracy against 10 sampled options: CLINC 151-way 0.88 against 0.98; DBpedia level 2 with 70 labels
  is the least calibrated case (ECE 0.14).
* Questions packed into one row (`independent=False`) see the earlier question texts, and reversing their order changes up to
  12% of answers. The default path scores each question alone.
* The v10 browser results are on 22 click-only MiniWoB++ tasks: small synthetic pages with the elements listed as text. Typing,
  scrolling and real websites were not tested. OpenJev accuracy is 0.8 points lower than v8.
* Abstention: a catch-all option ("none of the above", "other", "unsure") is chosen when nothing on offer fits, not when the
  exact fine-grained label is merely absent. Wordings far from the training data remain the main risk.
* One in-task dataset, `tweet_hate` (SemEval-2019 HatEval), stays near chance on its test split, whose collection and label
  definition differ from the training split. The number is reported as measured.

## Changelog

| version | what changed |
|---|---|
| **v10** (2026-09-19, these weights) | v8 plus 384 steps of calibration-aware RL on live browser tasks and exact games. Measured on the same rows: live browser click tasks 83% to 93% sampled success (held-out tasks 73% to 92%), stated beliefs about action outcomes 0.47 to 0.22 nats above the exact law, Mind2Web +1.5 points, general accuracy and Bespoke's public suite unchanged, OpenJev −0.8 points. |
| v9 | terse-bucket routing messages and labelled shell commands in the data; described in the GitHub README, but the Hub weights stayed v8, so v10 does not contain it |
| v8 (Hub tag `v8`) | isolated Score levels, teacher-written custom questions with a generic option next to a catch-all, the cacheable schema-first layout |
| v6 to v7 | the input shapes Jev accepts: described options, up to 255 options, JSON states with path references, long inputs |
| v4 to v5 | next-action choice from agent trajectories and game states; the proper abstention fix |
| v1 to v3 | the one-pass readout on the public decision mixture, one fitted temperature |

The full entries, with the browser and game recordings and the same-rows comparison against v8, are in
[docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) of the GitHub repository;
[docs/HISTORY.md](https://github.com/Mapika/decider/blob/main/docs/HISTORY.md) has how each stage was trained and measured.

## Reproduction

Code, data registry, training and evaluation scripts, the RL recipe and the per-version history:
https://github.com/Mapika/decider. Each release is staged with `scripts/stage_release.py` and uploaded with
`scripts/upload_hf.py`; the previous weights are kept under the tag `v8` in this repository.
