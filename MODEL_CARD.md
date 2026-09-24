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
Score levels. v10 continued the v8 weights for 384 steps of calibration-aware reinforcement learning whose only rewards are
outcomes (live browser task checkers and the exact probability laws of games), with a hard KL limit to the v8 weights on
replayed training rows. **This repository holds v11**: v10 plus a LoRA of rank 64, trained for 2 epochs on 42,749 rows of harder
decisions and replay, with the replay rows trained toward v10's own answer distribution, merged into the weights, and one
temperature per answer type. v10 stays available under the Hub tag `v10` and v8 under `v8`. Code, data registry, training
scripts and the recipe are at https://github.com/Mapika/decider; `decider/` in this repository is the inference subset of that
package. The other sizes and the vision variant are listed under The decider family.

v11 is better than v10 on hard decisions and worse on some everyday and game rows. On the same rows: held-out generated decision
families 0.429 against 0.324, held-out document questions 0.753 against 0.646, JevBench public hard tier 0.577 against 0.459, and
better calibrated on both hard sets than v10 (0.156 against 0.226 and 0.175 against 0.307), though still overconfident there.
Losses: human-labelled public sets −2.2 points, a knowledge guard set −1.6, greedy bag-draw play −10.9, sampled slippery-grid
play −4.3, sampled browser play −2.8 (interval includes zero), TypeSafe −4.9 (interval includes zero). Details under
[Changes from v10](#changes-from-v10).

**Contents:** [Changes from v10](#changes-from-v10) · [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Field types](#field-types) · [Training](#training) · [Evaluation](#evaluation) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

## Changes from v10

v11 is v10 plus one supervised LoRA stage (described under Training) and a per-type temperature map:

* **Harder decisions.** The same 22,649 labelled rows as the stage 2 of decider-4b v2.1: 8,000 rows from ten generated decision
  families whose answers the generating code computes, 11,356 questions over business documents written by Qwen3.6-27B and kept
  when two further independent answers agreed, and 3,293 human-labelled rows.
* **Replay trained toward v10's own distribution.** 20,100 replay rows from the public decision mixture, trained toward v10's
  answer distribution (loss KL(p_v10 ‖ p_model) over the options) instead of their labels, so that everyday answers stay close
  to v10's. On 1,000 held-out replay rows the mean KL at temperature 1 is 0.047 nats and 94.9% of the argmax answers equal
  v10's. The fitted temperature is 1.145 (v10 1.30).
* **One temperature per answer type.** `decider_config.json` has `temperature` 1.145 and `temperature_by_type`
  `{"choice": 1.164, "noul": 1.624, "score": 1.124}`, fitted by NLL with `decider.calibrate` (decider-ai 1.4.0). decider-ai 1.4.0
  and later use the map. **decider-ai 1.3.0 and earlier ignore the map and serve every answer at 1.145**; a temperature does not
  change which option is most probable, so the answers are the same either way (except exact ties between the level rows of an
  isolated Score answer: 1 row of 5,000 on the held-out generated families and 1 of 944 on the teacher validation rows changed);
  the probabilities differ, mostly on yes/no answers.
* The model name in answers is `decider-2b-v11` (v10 reported `decider-v10`).
* `decider_config.json` no longer marks the model as trained for the schema-first layout (`schema_first_trained: false`; v10:
  true), because the LoRA stage used only the state-first layout and no schema-first temperature was fitted. With
  `DECIDER_SCHEMA_CACHE=1` the server therefore does not turn the schema cache on for v11. `Decider.schema()` still runs; its
  accuracy was not measured on v11.

All rows below are on identical inputs and seeds: v11 through decider-ai 1.4.0 with its map, v10 through decider-ai 1.3.0 at its
stored temperature 1.30, in the same session on 2026-09-24 (so the v10 numbers can differ slightly from the ones first published
for v10 further down). Intervals are 95% paired bootstrap intervals (rows for the fixtures, boards for the games, task-seed pairs
for the browser). The regression set and our own sets were read from stored temperature-1 logits at each model's served
temperatures. v10's JevBench file was read earlier through decider-ai 1.2.1; accuracy does not depend on the temperature.

| set | v10 (T 1.30) | v11 (map) | v11 minus v10 |
|---|---|---|---|
| regression set, 67 in-task tasks, accuracy / NLL / ECE | 0.806 / 0.474 / 0.038 | 0.802 / 0.481 / 0.038 | −0.4 |
| regression set, 28 held-out tasks | 0.755 / 0.622 / 0.084 | 0.752 / 0.626 / 0.083 | −0.3 |
| held-out generated families (heldout_jb), 5,000 rows, accuracy / ECE | 0.324 / 0.226 | 0.429 / 0.156 | +10.5 |
| held-out document questions (test_teacher2), 449 rows, accuracy / ECE | 0.646 / 0.081 | 0.753 / 0.075 | +10.7 |
| human-labelled public sets (cal_human), 1,595 rows, accuracy | 0.803 | 0.781 | −2.2 |
| knowledge guard (MMLU, ARC and others), 2,994 rows, accuracy | 0.732 | 0.715 | −1.6 |
| 847 in-task validation rows, accuracy / NLL | 83.4% / 0.444 | 81.9% / 0.468 | −1.4 (−3.0 to +0.1); NLL +0.024 (+0.008 to +0.039) |
| OpenJev, 5,252 rows, accuracy / NLL | 63.2% / 0.917 | 64.6% / 0.860 | +1.4 (+0.6 to +2.2); NLL −0.057 (−0.068 to −0.046) |
| Mind2Web, 1,770 rows, accuracy / NLL | 82.6% / 0.543 | 83.9% / 0.495 | +1.3 (+0.1 to +2.5); NLL −0.048 (−0.074 to −0.023) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 80.4% / 0.585 | 75.5% / 0.604 | −4.9 (−10.8 to +1.0); NLL +0.018 (−0.088 to +0.135) |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.889 / 0.459 | 1.000 / 0.889 / 0.577 | hard +13 items |
| JevBench hard tier, top-label ECE | 0.307 | 0.175 | |
| Bespoke's public suite, macro / micro | 0.703 / 0.711 | 0.706 / 0.711 | +0.3 macro |
| live MiniWoB++, sampled, all 22 tasks | 93.2% | 90.3% | −2.8 (−7.4 to +1.1) |
| live MiniWoB++, sampled, 16 rewarded tasks | 93.8% | 89.8% | −3.9 (−9.4 to +0.8) |
| live MiniWoB++, sampled, 6 held-out tasks | 91.7% | 91.7% | 0.0 (−8.3 to +8.3) |
| live MiniWoB++, greedy, all 22 tasks | 91.5% | 92.0% | +0.6 (−3.4 to +4.5) |
| live MiniWoB++, greedy, 16 rewarded tasks | 91.4% | 92.2% | +0.8 (−4.7 to +6.2) |
| live MiniWoB++, greedy, 6 held-out tasks | 91.7% | 91.7% | 0.0 (−6.2 to +6.2) |
| zero-shot games, 234 boards, sampled, win rate | 23.9% | 22.6% | −1.3 (−3.4 to +0.9) |
| bag-draw games, 64 boards, sampled, win rate | 41.8% | 41.8% | 0.0 (−5.5 to +5.5) |
| slippery-grid games, 64 boards, sampled, win rate | 19.1% | 14.8% | −4.3 (−7.8 to −1.2) |
| zero-shot games, 234 boards, greedy, win rate | 26.9% | 24.8% | −2.1 (−5.6 to +1.3) |
| bag-draw games, 64 boards, greedy, win rate | 57.8% | 46.9% | −10.9 (−20.3 to −3.1) |
| slippery-grid games, 64 boards, greedy, win rate | 15.6% | 15.6% | 0.0 (0.0 to +0.0) |
| ten text games, greedy: Pong / Breakout / CliffWalking / BabyAI-GoTo / Freeway / Blackjack | 8 / 22 / −13 / 0.18 / 0 / −1 | 8 / 22 / −13 / 0.19 / 1 / −1 | |
| behaviour probes: model-router tier / needs-live-data (31 items) | 0.903 / 0.806 | 0.935 / 0.774 | +1 / −1 item |
| behaviour probes: command risk / touches-outside-project (45 items) | 0.733 / 0.556 | 0.822 / 0.556 | +4 / 0 items |
| behaviour probes: generic bucket / catch-all / abstention battery / browser element and action | 0.85 / 0.95 / 8 of 8 / 0.938 and 0.875 | 0.85 / 0.95 / 8 of 8 / 0.938 and 0.875 | |
| issue #9 form cases, right of 4 | 0 | 0 | |

**Gains.** On the two held-out hard sets, whose families and business domains are not in the training data, v11 is 10.5 and 10.7
points above v10, and on JevBench's public hard tier 13 items (11.7 points) above it. OpenJev +1.4 and Mind2Web +1.3 points
(intervals exclude zero), command-risk probe +4 items. Calibration on hard items is better than v10's: 0.156 against 0.226 on the
held-out generated families and 0.175 against 0.307 on the JevBench hard tier.

**Losses, stated plainly.**
* Human-labelled public sets (the validation halves of MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77 and others): 0.781
  against 0.803, −2.2 points. A knowledge guard set (MMLU, ARC and similar): 0.715 against 0.732, −1.6 points.
* Greedy bag-draw play: 46.9% against 57.8% wins, −10.9 points (interval −20.3 to −3.1). Sampled slippery-grid play: 14.8% against
  19.1%, −4.3 points (interval −7.8 to −1.2).
* Sampled browser play: 90.3% against 93.2%, −2.8 points (interval −7.4 to +1.1, includes zero); on the 16 rewarded tasks −3.9
  (interval −9.4 to +0.8). Greedy browser play and the six held-out tasks are level.
* TypeSafe workflow decisions −4.9 points (interval −10.8 to +1.0), 847 validation rows −1.4 (−3.0 to +0.1), regression set −0.4
  in-task and −0.3 held-out; needs-live-data probe one item lower.
* Still overconfident on hard items: calibration error 0.156 on the held-out generated families, where our release limit is 0.08.
* The issue #9 form cases are 0 of 4, as for v10.

**How v11 was chosen, and why it is released although it did not pass.** The run had a pre-registered release rule for the 2B.
A checkpoint was eligible only if it lost at most 1 point on the human-labelled sets against v10; every checkpoint lost more
(v11: −2.2), so no checkpoint was eligible. The rule's fallback candidate is v11 (two arms tied within 0.002 on the selection
score; the tie went to the higher regression in-task accuracy). It passes the numeric items (regression accuracy at most 1.0 point
in-task and 0.5 held-out under v10, held-out generated families at least 5 points and held-out document questions at least 3
points above v10, regression ECE at most 0.01 above v10's, sampled browser, zero-shot and bag-draw play not shown to be below
v10's: upper end of the 95% interval at least 0) and fails the calibration item:
calibration error on the held-out generated families at most 0.08. v11 is at 0.171 at its global temperature and 0.156 with the
map; v10 is at 0.226 and would fail the same item. A second pre-registered rule for the temperature map had the same 0.08 limit,
so the map did not pass either. v11 is released with the map on a decision made after reading the full comparison above. The
JevBench public items were read once for v11 at its global temperature and once with the map, after the rule decisions; they
were not used for training, selection or the temperatures.

**Which version to use.**
* v11 (this revision): the default. Hard judgments, long policies, document questions.
* v10 (`revision="v10"`): if you rely on sampled browser play, greedy bag-draw or slippery-grid play, knowledge multiple choice or
  TypeSafe-style workflow decisions, where v10 is ahead (see the losses above).

The package loads a local folder, so download the revision first:

```python
from huggingface_hub import snapshot_download
from decider.infer import Decider
d = Decider(snapshot_download("Mapika/decider-2b", revision="v10"))
```

For the HTTP server, set `DECIDER_MODEL` to the same downloaded folder.

## The decider family

All six repositories share one interface (`decider.infer.Decider`, `POST /v1/systemone` in TypeSafe's format) and one
readout: the letter logits at an answer slot, softmaxed over the options. Pick by size and input.

| model | base | weights | use it for | numbers |
|---|---|---|---|---|
| [decider-2b](https://huggingface.co/Mapika/decider-2b) v11 | Qwen3.5-2B-Base | 3.8 GB bf16 | the default: routing, classification, judgments, browser agents; 4 ms per request with CUDA graphs on one GPU; v10 under the tag `v10` | regression set 0.802 in-task / 0.752 held-out; JevBench hard 0.577; live browser 90% sampled; Bespoke suite 0.706 |
| [decider-4b](https://huggingface.co/Mapika/decider-4b) v2.1 | Qwen3.5-4B-Base | 8.4 GB bf16 | the middle point: knowledge questions and hard judgments above the 2B in a dense 8.4 GB model; no RL stage; v2 and v1 under the tags `v2` and `v1` | 0.831 / 0.784; JevBench hard 0.649; live browser 93% sampled; Bespoke 0.756 |
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
# [{'choice': 'billing', 'confidence': 0.93, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.70, 'probs': {...}}]   (v11, decider-ai 1.4.0, eager)
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
# {"model": "decider-2b-v11", "answers": {"department": {"type": "choice", "choice": "billing", "confidence": ..., "certainty": ..., "probabilities": {...}},
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
probs = torch.softmax(logits[letters].float() / 1.164, -1)     # -> P(billing), P(technical support), P(sales); 1.164 is the stored choice temperature
```

For several questions in one pass, append further `Question k: ... Answer k: (` blocks and read the logits at each `(`
position (see `decider/prompt.py`).

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. From decider-ai 1.4.0 the config may also hold `temperature_by_type`, one temperature per answer type
(`choice`, `noul`, `score`; a missing type uses `temperature`). This release's config has such a map: Choice answers use 1.164,
yes/no (noul) answers 1.624, Score answers 1.124 on each of their level rows. Package versions before 1.4.0 use `temperature`
(1.145) for every answer. `DECIDER_TEMPERATURE=T` or `Decider(path, temperature=T)` replaces `temperature` with T and switches the
map off, so every answer then uses T. Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled),
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

**LoRA stage (v10 to v11).** A LoRA of rank 64 (alpha 128) on the attention and MLP weights of v10, trained for 2 epochs over
42,749 rows in the plain state-first layout with isolated Score levels, then merged into the bf16 weights:

| source | rows | content | target |
|---|---|---|---|
| generated decision families | 8,000 | ten families (temporal and numeric decisions, subtle answer judgment, long policies, multi-hop lookup, abstention, probability, constrained trade-offs, safety judgment, paraphrase sensitivity, adversarial traps); the answers are computed by the generating code | the label |
| questions over business documents, written by Qwen3.6-27B with thinking on | 11,356 | one realistic business document plus three or four typed questions per writer call; each question kept only when two further independent answers by the same model agreed with the writer's | the label |
| human-labelled public sets (training halves) | 3,293 | MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77, RACE, OpenBookQA, LogiQA 2, MedQA, Winogrande | the label |
| replay of the public decision mixture | 20,100 | 100 rows from the training half of each of the 67 in-task regression tasks (6,700); 7,000 from the families closest to form filling, browser and agent actions, routing, tools, shell commands and situations (form rules 1,000, Mind2Web 800, agent trajectories 1,300, routing 1,000, custom questions 1,000, commands 600, tool selection 600, situations 700); 6,400 spread over every other family | v10's answer distribution: loss KL(p_v10 ‖ p_model) at temperature 1 |

| LoRA stage | |
|---|---|
| trainable parameters | LoRA rank 64, alpha 128, on the attention and MLP projections; merged after training |
| loss | cross-entropy on the slot readout for labelled rows; KL(p_v10 ‖ p_model) over the options for replay rows, with p_v10 from the frozen v10 weights |
| schedule | learning rate 1e-4, 5% warm-up then cosine, 1,676 steps of 65,536 tokens (2 epochs), seed 0 |
| hardware | one NVIDIA B300 shared with another job, 150 minutes |

The replay comes from the public mixture (`scripts/train.sh full`), which also holds the v9 additions (shell commands, terse
routing) that v8 and v10 were not trained on; those rows were trained toward v10's answers, not their labels. No JevBench item
and no Decision Index item was used for training, for writing the generators or the document questions, for selecting the
checkpoint, or for the temperatures. Every training row was checked against every evaluation file used for selection, the
evaluation halves of the regression tasks, the four request fixtures and the issue #9 cases. Two arms were trained: one with labels on a replay of 6,700 rows,
and this one, with v10's distribution on the three times larger replay described above; the rule's fallback selected this arm
after epoch 2.

**Temperatures.** `temperature` 1.145, fitted by NLL on the in-task half of the public regression set without Banking77,
CLINC-OOS, MMLU, ARC, Winogrande and HellaSwag (61 tasks, 102,804 rows). `temperature_by_type` fitted by NLL per answer type with
`decider.calibrate.fit_by_type` on a pool of those regression rows and our own validation rows: 108,910 Choice answers, 1,677 yes/no
answers and 535 Score answers. v10's temperature (1.30) was one value fitted on the 67 in-task tasks.

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
| decider-2b v10, rebuilt set, T=1.30 | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| **decider-2b v11 (this repository), rebuilt set, per-type map** | 0.802 / 0.481 / 0.038 | 0.752 / 0.626 / 0.083 |
| v8, questions-first layout (schema cache), T=1.18 | 0.790 / 0.500 / 0.038 | 0.707 / 0.757 / 0.104 |

The two "rebuilt set" rows were measured after the data pipeline was rebuilt on another machine: two datasets no longer download
(TREC-fine, the game states) and the current mixture adds held-out probes, so that set has 67 in-task and 28 held-out tasks. Its
numbers are comparable to each other, not to the rows above. v10 matches v8 on it. The v10 row was measured again on 2026-09-24
in the v11 session as 0.806 / 0.474 / 0.038 and 0.755 / 0.622 / 0.084. The regression rows are all Choice answers, so v11 reads
them at the Choice temperature 1.164.

<details>
<summary><b>Per-task accuracy / ECE on the 28 held-out datasets, v8 against v10 (not measured per task for v11)</b></summary>

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

**v11 against v10 on the same rows** is the table under [Changes from v10](#changes-from-v10).

**Bespoke's public suite, v10 against v11** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through
`system_one` as shipped; same session, v10 through decider-ai 1.3.0, v11 through 1.4.0 with the map):

| subset (type) | decider-2b v10 | decider-2b v11 |
|---|---|---|
| vitaminc-dev (choice) | 0.639 | 0.639 |
| massive-en-US (choice; trained) | 0.823 | 0.814 |
| massive-de-DE (choice) | 0.797 | 0.783 |
| boolq (noul; trained) | 0.803 | 0.850 |
| squad2 (noul) | 0.776 | 0.743 |
| paws (noul; trained) | 0.720 | 0.760 |
| multinli (choice; trained) | 0.856 | 0.853 |
| civil_comments (noul; trained) | 0.840 | 0.870 |
| aegis2 (noul) | 0.728 | 0.716 |
| helpsteer2 (score; trained) | 0.426 | 0.446 |
| summeval-relevance (score) | 0.354 | 0.229 |
| summeval-consistency (score) | 0.653 | 0.757 |
| pubmedqa (choice; trained) | 0.724 | 0.712 |
| **macro / micro** | 0.703 / 0.711 | 0.706 / 0.711 |
| macro over the subsets not trained on | 0.658 | 0.645 |

**Calibration of v11 on our own sets** (10-bin top-label ECE on stored temperature-1 logits, read at the served temperatures;
heldout_jb: held-out generated families, test_teacher2: held-out document questions, guard: knowledge guard, cal_human:
human-labelled validation rows):

| set | rows | accuracy | ECE at the global T / with the map | choice / noul / score ECE with the map |
|---|---|---|---|---|
| heldout_jb | 5,000 | 0.429 | 0.171 / 0.156 | 0.172 / 0.121 / 0.141 |
| test_teacher2 | 449 | 0.753 | 0.083 / 0.075 | 0.074 / 0.093 / 0.131 |
| guard | 2,994 | 0.715 | 0.050 / 0.046 | 0.046 / − / − |
| cal_human | 1,595 | 0.781 | 0.030 / 0.030 | 0.030 / − / − |

The map lowers the error of yes/no answers on the held-out generated families (0.171 to 0.121) and leaves Choice answers where
they were (0.175 to 0.172), because the Choice temperature is fitted on a pool that is 94% everyday regression rows. A map fitted
without the regression rows (Choice 1.385) would reach 0.126 there and would make the regression set less calibrated (in-task ECE
0.0510 against 0.0385); it was not used. On the JevBench hard tier, v11's top-label ECE with the map is 0.175 (0.195 at the global
temperature; by type Choice 0.205, yes/no 0.243, Score 0.366). The yes/no Brier score on the probe batteries is 0.048 with the
map and 0.043 at the global temperature (v10 0.046).

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

v11 has v10's architecture and size, and the temperature map is a division per answer, so the speed was not measured again.
The numbers below were measured on v10: one NVIDIA B300, decider-ai 1.2.1, 2026-09-23. Support-ticket states of about 230 tokens with 3 typed questions each
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

* v11's losses against v10 (see Changes from v10): human-labelled public sets −2.2 points, knowledge guard −1.6, greedy bag-draw
  play −10.9, sampled slippery-grid play −4.3, sampled browser play −2.8 (interval includes zero), TypeSafe −4.9 (interval includes
  zero).
* Overconfident on hard multi-step items: calibration error 0.156 on our held-out generated families and 0.175 on the JevBench
  public hard tier. Better than v10 (0.226, 0.307), not calibrated.
* The per-type map needs decider-ai 1.4.0 or later. With 1.3.0 or earlier, or with `DECIDER_TEMPERATURE` set, every answer uses
  1.145: the answers are the same and yes/no answers are sharper.
* The stated-belief and click-outcome calibration of v10's RL stage (below) was not measured again on v11.
* A 2B model without reasoning. Knowledge-heavy multiple choice (MMLU, MedQA, ARC) improves little over the base model, and a
  judgment that needs several steps should be split into several questions.
* English only. Calibration is measured on public datasets and teacher-labelled probes, not on your traffic. Check it on your
  own labels before using confidence for routing.
* v10 and v11 continue the v8 weights. The v9 data for terse bucket names (`support`, `help`, `account` next to `other`) is not
  in them as labels (v11's replay contains those rows but trains them toward v10's answers):
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
* The browser results are on 22 click-only MiniWoB++ tasks: small synthetic pages with the elements listed as text. Typing,
  scrolling and real websites were not tested.
* Abstention: a catch-all option ("none of the above", "other", "unsure") is chosen when nothing on offer fits, not when the
  exact fine-grained label is merely absent. Wordings far from the training data remain the main risk.
* One in-task dataset, `tweet_hate` (SemEval-2019 HatEval), stays near chance on its test split, whose collection and label
  definition differ from the training split. The number is reported as measured.

## Changelog

| version | what changed |
|---|---|
| **v11** (2026-09-24, these weights) | v10 plus a merged LoRA (rank 64, attention and MLP, 2 epochs, 42,749 rows: generated decision families, document questions written by Qwen3.6-27B and kept when two independent answers agreed, human-labelled public sets, and a replay of the public mixture trained toward v10's own answers); temperature 1.145 and `temperature_by_type` {choice 1.164, noul 1.624, score 1.124} (decider-ai 1.4.0; older versions use 1.145). Held-out generated families 0.429 against 0.324, held-out document questions 0.753 against 0.646, JevBench hard 0.577 against 0.459; human-labelled sets −2.2, knowledge guard −1.6, greedy bag-draw −10.9, sampled slippery grid −4.3, sampled browser −2.8 points. Did not pass its pre-registered rule; released on the full comparison |
| v10 (2026-09-19, Hub tag `v10`) | v8 plus 384 steps of calibration-aware RL on live browser tasks and exact games. Measured on the same rows: live browser click tasks 83% to 93% sampled success (held-out tasks 73% to 92%), stated beliefs about action outcomes 0.47 to 0.22 nats above the exact law, Mind2Web +1.5 points, general accuracy and Bespoke's public suite unchanged, OpenJev −0.8 points. |
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
`scripts/upload_hf.py`; the previous weights are kept under the tags `v10` and `v8` in this repository. The LoRA stage of v11 was
trained with a LoRA trainer in the research repository. `eval_results.json` has v11's regression metrics with the map and at the
global temperature, our held-out sets by answer type, the fixtures, games, browser and Bespoke results with the paired
comparisons against v10, the text games, the behaviour probes, the issue #9 cases and the JevBench public items.
