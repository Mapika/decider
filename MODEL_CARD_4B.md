---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B-Base
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass]
---

# decider-4b: typed decisions with calibrated probabilities in one forward pass, 4B dense

A language model that does not generate text. It reads a state and one or more typed questions, each with an explicit option
list, and returns a probability distribution over the options for every question from one forward pass. There is no decoding,
no parsing and no output outside the options you defined. It is called from software, not chatted with. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev).

Base model: [Qwen/Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base): 4.2B parameters, 32 layers, 8 with full
attention and 24 with gated delta-net linear attention, hidden size 2,560. Two supervised stages. Stage 1 (decider-4b v1): one
pass of cross-entropy on the slot readout over mixture v2, the public decision mixture of
[decider-2b](https://huggingface.co/Mapika/decider-2b) plus 26 further public decision datasets and ten programmatically
generated families with verifiable gold (742M tokens). Stage 2 (v2.1): a LoRA of rank 64 on the attention and MLP weights,
trained for 2 epochs on 29,325 rows of harder decisions and replay, with the replay rows trained toward v1's own answer
distribution, then merged into the weights. There is no reinforcement-learning stage. **This repository holds v2.1**, the bf16
weights (8.4 GB), with one temperature per answer type. v2 stays available under the Hub tag `v2` and v1 under the tag `v1`
(see [Changes from v2](#changes-from-v2) for who should keep using them). The other sizes are listed under The decider family.
`decider/` in this repository is the inference subset of the GitHub package.

v2.1 keeps most of v2's gain on hard decisions and gets back most of what v2 lost in sampled play. Against v2 on the same rows:
bag-draw games in sampled play 52.0% against 37.9% wins (v1 56.6%), zero-shot games sampled 26.9% against 22.4% (v1 27.8%), live
browser tasks sampled 93.2% against 88.1% (v1 90.9%), CliffWalking −13 against −60, the regression set 0.831 / 0.784 against 0.824
/ 0.779 (v1 0.834 / 0.788); held-out generated decision families 0.556 against 0.560 (v1 0.469) and the JevBench public hard tier
0.649 against 0.676 (v1 0.550). It is less well calibrated than v2 on hard items (details under
[Calibration](#calibration)), still answers one of the two form-filling cases of issue #9 wrongly, and is worse than v1 on
BabyAI-GoTo and on greedy bag-draw play. Against decider-35b-a3b it is 2.4 and 2.6 points lower on the regression set.

**Contents:** [Changes from v2](#changes-from-v2) · [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Training](#training) · [Evaluation](#evaluation) · [Calibration](#calibration) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

## Changes from v2

v2.1 differs from v2 in two things:

* **Replay rows are trained toward v1's own distribution.** Stage 2 of v2 trained every row, including 6,700 replay rows from v1's
  training mixture, on its hard label. v2.1 trains the replay rows toward v1's answer distribution instead, with the loss
  KL(p_v1 ‖ p_model) over the options of each answer, and keeps hard labels on every other row. Everything else is v2's recipe:
  the same rows (29,325 after removing 31 rows that shared a context with an evaluation file; v2 had 29,356), LoRA rank 64, learning
  rate 1e-4, 2 epochs. Training the replay rows on hard labels had sharpened the logits everywhere; the fitted temperature then
  rose to 1.935 and flattened every served answer, which is what cost v2 its sampled play. With the replay rows trained toward
  v1, the fitted temperature is 1.099 (v1 1.05). On 1,000 held-out replay rows the mean KL(p_v1 ‖ p_model) at temperature 1 is 0.021
  nats (v2: 0.112), and 96.3% of the argmax answers equal v1's (v2: 94.1%).
* **One temperature per answer type.** `decider_config.json` has `temperature` 1.099 and `temperature_by_type`
  `{"choice": 1.110, "noul": 1.560, "score": 1.287}`, fitted by NLL with `decider.calibrate` (decider-ai 1.4.0). decider-ai 1.4.0
  and later use the map. **decider-ai 1.3.0 and earlier ignore the map and serve every answer at 1.099**; a temperature does not
  change which option is most probable, so the answers are the same either way (except exact ties between the level rows of an
  isolated Score answer, where rounding can break the tie differently: 1 of 5,000 held-out generated-family rows and 2 of 447
  document-question validation rows changed); the
  probabilities differ, mostly on yes/no and Score answers. The map changes little in practice: sampled
  play, the fixtures and the probes are the same within noise with and without it, and it lowers the calibration error on
  held-out yes/no answers (0.146 to 0.101) and Score answers (0.140 to 0.108). It does not fix the overconfidence on hard
  Choice answers (see Calibration).

All rows below are on identical inputs and seeds: v2.1 through decider-ai 1.4.0 with its map, v1 and v2 through decider-ai 1.3.0
at their stored temperatures (1.05 and 1.935), measured in the same session on 2026-09-24. Intervals are 95% paired bootstrap
intervals (rows for the fixtures, boards for the games, task-seed pairs for the browser). The regression set and the two held-out
sets were read from stored temperature-1 logits at each model's served temperatures. The JevBench files of v1 and v2 were read
earlier through decider-ai 1.2.1, v2's at the candidate temperature 1.719; accuracy does not depend on the temperature.

| set | v1 (T 1.05) | v2 (T 1.935) | v2.1 (map) | v2.1 minus v1 | v2.1 minus v2 |
|---|---|---|---|---|---|
| regression set, 67 in-task tasks, accuracy / NLL / ECE | 0.834 / 0.404 / 0.027 | 0.824 / 0.441 / 0.041 | 0.831 / 0.414 / 0.031 | −0.3 | +0.7 |
| regression set, 28 held-out tasks | 0.788 / 0.558 / 0.071 | 0.779 / 0.566 / 0.080 | 0.784 / 0.569 / 0.077 | −0.4 | +0.5 |
| 847 in-task validation rows, accuracy / NLL | 86.1% / 0.417 | 85.0% / 0.419 | 85.0% / 0.432 | −1.1 (−2.1 to +0.0); NLL +0.015 (+0.003 to +0.028) | 0.0 (−1.5 to +1.5); NLL +0.014 (−0.006 to +0.035) |
| OpenJev, 5,252 rows, accuracy / NLL | 63.9% / 0.893 | 66.7% / 0.789 | 66.0% / 0.846 | +2.1 (+1.2 to +2.9); NLL −0.047 (−0.060 to −0.035) | −0.7 (−1.6 to +0.2); NLL +0.057 (+0.044 to +0.070) |
| Mind2Web, 1,770 rows, accuracy / NLL | 88.4% / 0.366 | 87.4% / 0.393 | 87.5% / 0.375 | −0.8 (−1.7 to +0.0); NLL +0.009 (−0.004 to +0.023) | +0.1 (−0.8 to +1.1); NLL −0.018 (−0.032 to −0.005) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 81.4% / 0.611 | 86.3% / 0.410 | 84.3% / 0.441 | +2.9 (−2.9 to +8.8); NLL −0.170 (−0.324 to −0.036) | −2.0 (−7.8 to +3.9); NLL +0.031 (−0.069 to +0.134) |
| held-out generated families (heldout_jb), 5,000 rows, accuracy / ECE | 0.469 / 0.260 | 0.560 / 0.046 | 0.556 / 0.147 | +8.7 | −0.4 |
| held-out document questions (test_teacher2), 449 rows, accuracy / ECE | 0.726 / 0.110 | 0.826 / 0.053 | 0.820 / 0.043 | +9.4 | −0.7 |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.958 / 0.550 | 1.000 / 0.986 / 0.676 | 1.000 / 0.986 / 0.649 | hard +11 items | hard −3 items |
| JevBench hard tier, top-label ECE (v1, v2: files read through 1.2.1, v2 at T 1.719) | 0.288 | 0.104 | 0.184 | | |
| Bespoke's public suite, macro / micro | 0.757 / 0.765 | 0.773 / 0.781 | 0.756 / 0.765 | −0.1 macro | −1.6 macro |
| live MiniWoB++, sampled, all 22 tasks | 90.9% | 88.1% | 93.2% | +2.3 (−1.7 to +6.2) | +5.1 (+0.6 to +9.7) |
| live MiniWoB++, sampled, 16 rewarded tasks | 96.1% | 93.0% | 95.3% | −0.8 (−3.9 to +2.3) | +2.3 (−2.3 to +7.0) |
| live MiniWoB++, sampled, 6 held-out tasks | 77.1% | 75.0% | 87.5% | +10.4 (0.0 to +20.8) | +12.5 (0.0 to +25.0) |
| live MiniWoB++, greedy, all 22 tasks | 91.5% | 92.6% | 93.8% | +2.3 (−0.6 to +5.7) | +1.1 (−1.7 to +4.5) |
| live MiniWoB++, greedy, 16 rewarded tasks | 97.7% | 96.9% | 96.1% | −1.6 (−3.9 to +0.0) | −0.8 (−2.3 to +0.0) |
| live MiniWoB++, greedy, 6 held-out tasks | 75.0% | 81.2% | 87.5% | +12.5 (+4.2 to +22.9) | +6.2 (−4.2 to +16.7) |
| zero-shot games, 234 boards, sampled, win rate | 27.8% | 22.4% | 26.9% | −0.9 (−2.6 to +1.0) | +4.5 (+2.5 to +6.6) |
| bag-draw games, 64 boards, sampled, win rate | 56.6% | 37.9% | 52.0% | −4.7 (−9.0 to −0.4) | +14.1 (+8.2 to +19.9) |
| slippery-grid games, 64 boards, sampled, win rate | 16.4% | 16.0% | 16.8% | +0.4 (−2.7 to +3.5) | +0.8 (−2.3 to +3.9) |
| zero-shot games, 234 boards, greedy, win rate | 29.1% | 27.8% | 25.2% | −3.8 (−7.3 to −0.4) | −2.6 (−6.4 to +1.3) |
| bag-draw games, 64 boards, greedy, win rate | 62.5% | 48.4% | 53.1% | −9.4 (−17.2 to −3.1) | +4.7 (−3.1 to +12.5) |
| slippery-grid games, 64 boards, greedy, win rate | 12.5% | 18.8% | 10.9% | −1.6 (−7.9 to +4.7) | −7.8 (−15.6 to +0.0) |
| ten text games, greedy: Pong / Breakout / CliffWalking / BabyAI-GoTo / Freeway / Blackjack | −21 / 14 / −13 / 0.54 / 0 / −0.6 | −21 / 12 / −60 / 0.35 / 1 / −0.6 | −5 / 35 / −13 / 0.19 / 0 / −0.6 | | |
| behaviour probes: model-router tier / needs-live-data (31 items) | 0.968 / 0.871 | 0.935 / 0.839 | 0.968 / 0.839 | 0 / −1 item | +1 / 0 items |
| behaviour probes: command risk / touches-outside-project (45 items) | 0.889 / 0.956 | 0.911 / 0.933 | 0.889 / 0.933 | 0 / −1 item | −1 / 0 items |
| behaviour probes: generic bucket / catch-all / abstention battery / browser element and action | 1.00 / 0.90 / 7 of 8 / 0.875 and 0.875 | 0.95 / 0.95 / 8 of 8 / 0.938 and 0.938 | 1.00 / 0.95 / 8 of 8 / 0.938 and 0.750 | | |
| issue #9 form cases c_1 / c_2 (probability of the gold option) | right (0.79) / right (0.98) | wrong (0.16) / right (0.28) | wrong (0.20) / right (0.41) | | |

**Regressions, stated plainly.**
* BabyAI-GoTo (greedy text game): 0.19 against v1's 0.54 and v2's 0.35.
* Greedy bag-draw play: 53.1% against v1's 62.5% wins (−9.4 points, interval −17.2 to −3.1); greedy zero-shot games overall 25.2%
  against 29.1% (−3.8, interval −7.3 to −0.4). Sampled bag-draw play is 4.7 points under v1 (interval −9.0 to −0.4).
* Behaviour probes: needs-live-data 0.839 against v1's 0.871 and touches-outside-project 0.933 against 0.956, one item each (as
  v2); the browser-agent action choice is 12 of 16 items (0.750), 2 fewer than v1 (0.875) and 3 fewer than v2
  (0.938).
* Form filling: issue #9 case c_1 is still answered wrongly. The field is "Degree earned" and the document entity is "Studied:
  Associate of Arts"; v2.1 chooses "skip" at 0.76 and gives the gold entity 0.20 (v1: gold 0.79; v2: gold 0.16). Case c_2 is
  answered correctly at 0.41. For form filling we suggest v1.
* Calibration on hard items: on the held-out generated families the calibration error is 0.147 against v2's 0.046 (see
  Calibration), and on the JevBench public hard tier 0.184 against v2's 0.104.
* JevBench public hard tier: 72 of 111 items (0.649) against v2's 75 (0.676).
* Against v2: TypeSafe −2.0 points and OpenJev −0.7 (intervals include zero), NLL higher on both; Bespoke's suite 0.756 against
  0.773 macro (v1 0.757); greedy slippery-grid play 10.9% against 18.8%.

**How v2.1 was chosen, and why it is released although it did not pass.** The run had pre-registered release rules. v2.1 passed
the numeric items (held-out document questions 0.820 against the required 0.816, held-out generated families 0.556 against 0.550,
regression accuracy and calibration not worse than v2's, sampled bag-draw above the midpoint of v1 and v2, sampled zero-shot
games and browser play not shown to be below v1's (upper end of the 95% interval at least 0), each probe at most one item below v1) and failed the last item, which required both
issue #9 form cases to be right: c_1 is wrong. By the rule it was not recommended. A second pre-registered rule for the
temperature map required the calibration error on the held-out generated families to be at most 0.08; the map reaches 0.147
(0.163 without it), so the map did not pass either. v2.1 is released with the map on a decision made after reading the full comparison above: it is better than v2 on sampled play, CliffWalking, the model-router probe and the regression set, at v2's
level on our held-out hard sets, and its failures are listed in this section. The JevBench public items were read once for the
candidate at its global temperature and once with the map, after the rule decisions; they were not used for training,
selection or the temperatures.

**Which version to use.**
* v2.1 (this revision): the default. Sampled play (games, browser agents that sample actions), hard decisions.
* v2 (`revision="v2"`): if you rely on confidence values on hard multi-step items, where v2 is better calibrated (held-out
  generated families 0.046, JevBench hard tier 0.104), or on its slightly higher JevBench hard tier and TypeSafe accuracy.
* v1 (`revision="v1"`): form filling (issue #9), BabyAI-GoTo-like grid navigation, greedy bag-draw play.

The package loads a local folder, so download the revision first:

```python
from huggingface_hub import snapshot_download
from decider.infer import Decider
d = Decider(snapshot_download("Mapika/decider-4b", revision="v2"))      # or revision="v1"
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
d = Decider("Mapika/decider-4b")
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': ..., 'probs': {...}}, {'choice': 'yes', 'confidence': ..., 'probs': {...}}]
```

The API is the same as decider-2b's: `decide_batch` scores many states with many questions in one call, `abstain_below=t`
returns `None` under a confidence threshold, a question can have 2 to 255 options, and `system_one` / `decider.serve` accept
TypeSafe's `POST /v1/systemone` request shape. Every question and every Score level is scored in its own row. The state may be a
string, object or array of up to 32k tokens. See the decider-2b card for the full description of the request shape, field types
and the schema cache.

Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton kernels for the Qwen3.5 linear-attention layers;
the model runs without it but several times slower). The weights take 8.4 GB in bf16. v2.1 uses the plain prompt layout, as v1 and
v2 do. The per-type temperatures need decider-ai 1.4.0 or later (or the `decider/` subset in this repository, taken from 1.4.0).
Checked on CUDA, eager path: decider-ai 1.3.0 loads v2.1 with the name `decider-4b-v2.1` and temperature 1.099 and gives exactly
the probabilities that 1.4.0 gives with the map switched off; 1.4.0 and the `decider/` subset in this repository load the map
and give the same probabilities as each other. For Choice and yes/no answers the 1.4.0 output with the map equals
softmax(log(p) · 1.099 / T_type) of the 1.3.0 output p (to 1e-7 on `decide`, to 8e-5 on the four-decimal `system_one` values);
an isolated-level Score answer is rescaled per level row (each level's yes/no pair at 1.287 instead of 1.099) before the
levels are normalised, so it cannot be recomputed from the combined Score probabilities. The 1.4.0 HTTP server with CUDA graphs was checked with `/v1/systemone` and `/decide` requests. Other package
versions, MPS and the FP8 path were not checked on v2.1. On Blackwell GPUs use 1.0.2 or later (1.0.0 and 1.0.1 have the cuDNN
attention fault fixed in 1.0.2). The model is dense, so the CUDA-graph engine, `torch.compile` and the FP8 path of the helper
package apply to it as to decider-2b (`use_graphs=False` selects eager PyTorch).

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
probs = torch.softmax(logits[letters].float() / 1.110, -1)     # 1.110 is the stored choice temperature (yes/no 1.560, Score 1.287)
```

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. From decider-ai 1.4.0 the config may also hold `temperature_by_type`, one temperature per answer type
(`choice`, `noul`, `score`; a missing type uses `temperature`). This release's config has such a map: Choice answers use 1.110,
yes/no (noul) answers 1.560, Score answers 1.287 on each of their level rows. Package versions before 1.4.0 use `temperature`
(1.099) for every answer. `DECIDER_TEMPERATURE=T` or `Decider(path, temperature=T)` replaces `temperature` with T and switches the
map off, so every answer then uses T. Large label sets were sub-sampled to at most 10 options per
training example (gold always kept, order shuffled), so the model conditions on the supplied candidates rather than on a fixed
head.

## Training

**Stage 1 (decider-4b v1).** One supervised pass over **mixture v2**, in the two prompt layouts (state-first and schema-first,
50/50), with isolated Score levels and 10% abstention rows, 1,892,408 items and 742M tokens, pre-tokenized once and read in the
same order by both ranks. Three sources:

| source | rows | share of tokens | content |
|---|---|---|---|
| the public decision mixture of decider-2b (`scripts/train.sh full` of the GitHub repository) | 1,539,860 | 60% | about 95 public decision datasets, agent trajectories, Mind2Web element choice, teacher-written custom questions, Jev's input shapes |
| 26 further public datasets with gold labels | 131,318 | 8% | code defect, clone and review-needed judgments; log anomaly and severity (HDFS, BGL); legal (LEDGAR, Unfair-ToS, CaseHOLD, SCOTUS, ECtHR); tables (TabFact, WikiTQ, FeTaQA, InfoTabs, TAT-QA); finance headlines; German credit; symptom diagnosis and medical specialty; SciTail, SciEntsBank, Climate-FEVER; XNLI, PAWS-X, MASSIVE (multilingual), Belebele, XCOPA; MC-TACO, TRACIE, TimeQA; ProofWriter, RuleTaker, LogiQA 2; essay scoring; agent next action |
| ten programmatic families with verifiable gold, each with a held-out variant | 221,230 | 32% | code, dates and times, logs, long documents, plans, policies, probability, schedules, tables, tools |

Ten of the 26 public datasets (Belebele, code clone, ECtHR, essay scoring, InfoTabs, PAWS-X, RuleTaker, SCOTUS, XCOPA, TRACIE)
and the held-out variant of every programmatic family were kept out of training; every evaluation row was checked against every
training row of all three sources and 4,168 overlapping training pairs were dropped. No game rows are in mixture v2 (the
public mixture rebuilt on this machine has none), so every game result below is zero-shot.

| stage 1 | |
|---|---|
| trainable parameters | all 4.2B (426 tensors) |
| optimizer | `torch.optim.AdamW` applied directly to the bf16 parameters, no FP32 master copy; betas 0.9 / 0.95, no weight decay |
| schedule | peak learning rate 1e-5, 150 warm-up steps, cosine to zero, 26,729 steps of 32,768 tokens, gradient clip 1.0 |
| hardware | 2 NVIDIA B300, data parallel, 16,384-token micro-batches per GPU, gradient checkpointing; 577 minutes at about 25,000 tokens per second, 39.6 GB peak per GPU |
| training cross-entropy | 0.97 over the first 200 steps, 0.42 at 25%, 0.35 at 50%, 0.35 over the last 300 steps |

Why this optimizer: on a controlled quarter-data comparison of the same 4B (`ref_4b` against `ref_4b_bf16opt`, identical data and
schedule, only the optimizer changed), AdamW on the bf16 parameters beat AdamW with FP32 master weights by 3.3 held-out points and
0.072 nats (0.791 against 0.758 held-out accuracy), and by 7 points on nine knowledge tasks. The master copy lets every small update
through and moves the weights further from the base model; without it, updates below the bf16 resolution round away and more of
the base model's knowledge is kept.

**Stage 2 (v2.1).** A LoRA of rank 64 (alpha 128) on the attention and MLP weights of v1, trained for 2 epochs over 29,325 rows
in v1's plain state-first layout with isolated Score levels, then merged into the bf16 weights:

| source | rows | content | target |
|---|---|---|---|
| generated decision families | 8,000 | ten families (temporal and numeric decisions, subtle answer judgment, long policies, multi-hop lookup, abstention, probability, constrained trade-offs, safety judgment, paraphrase sensitivity, adversarial traps); the answers are computed by the generating code | the label |
| questions over business documents, written by Qwen3.6-27B with thinking on | 11,356 | in two rounds, one realistic business document (up to 34 domains and 22 document kinds) plus three or four typed questions per writer call; each question was answered twice more by the same model in fresh contexts, with shuffled options and without the writer's answer, and kept only when both answers agreed with the writer's (89% and 91% kept) | the label |
| human-labelled public sets (training halves) | 3,293 | MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77, RACE, OpenBookQA, LogiQA 2, MedQA, Winogrande | the label |
| replay of mixture v2 | 6,676 | 100 rows from the training half of each of the 67 in-task regression tasks | v1's answer distribution: loss KL(p_v1 ‖ p_model) at temperature 1 |

These are v2's rows without 31 that the v2.1 checks found to share a context with an evaluation file (7 human-labelled rows and
24 replay rows).

| stage 2 | |
|---|---|
| trainable parameters | LoRA rank 64, alpha 128, on the attention and MLP projections; merged after training |
| loss | cross-entropy on the slot readout for labelled rows; KL(p_v1 ‖ p_model) over the options for replay rows, with p_v1 from the frozen v1 weights; the step loss is the sum over answers divided by the number of answers |
| schedule | learning rate 1e-4, 5% warm-up then cosine, 1,518 steps of 65,536 tokens (2 epochs), seed 0 |
| hardware | one NVIDIA B300, 94 minutes |

No JevBench item and no Decision Index item was used for training, for writing the generators or the document questions, for
selecting the checkpoint, or for the temperatures. The ten generated families and the skill list of the document questions were
written from the family names that JevBench publishes for its sealed set, not from its items. Every training row was checked
against every evaluation file used for selection, the canonical examples of the evaluation half of all 153 mixture-v2 evaluation
tasks, the four request fixtures and the issue #9 cases: there is no exact state-and-question overlap (a few contexts under 40
characters, such as a short utterance, occur under another task and question). The held-out sets used for selection are
generated families from held-out templates (heldout_jb) and document questions from business domains that are not in the
training data (test_teacher2). Two arms were trained to the end (B4 and C4, which differ in the size of the replay; two further arms were
stopped or cancelled at the user's request); the rule selected B4 after epoch 2.

**Temperatures.** `temperature` 1.099, fitted by NLL (all rows pooled) on the in-task half of the public regression set without
Banking77, CLINC-OOS, MMLU, ARC, Winogrande and HellaSwag: 61 tasks, 102,804 rows. `temperature_by_type` fitted by NLL per answer
type with `decider.calibrate.fit_by_type` on a pool of those regression rows and our own validation rows (the validation halves of
the generated families and document questions, human-labelled validation rows, held-out replay rows): 108,927 Choice answers,
1,677 yes/no answers and 535 Score answers. The Choice pool is 94% regression rows, so the Choice temperature (1.110) stays close
to the global one. v2's temperature (1.935) and v1's (1.05) were one value each.

## Evaluation

All v2.1 numbers are with the per-type map through decider-ai 1.4.0, except where a paragraph says otherwise. Greedy play and
accuracy do not depend on the temperature; sampled play and calibration do.

**Public regression set**, rebuilt on this machine (95 tasks: 67 in-task, 28 held-out; large label sets sub-sampled to 10
options; the temperatures fitted on in-task data). The rows are the same for every model. ECE is the expected calibration
error with 15 bins, per-task mean. The regression rows are all Choice answers, so v2.1 reads them at the Choice temperature 1.110.

| model | in-task acc / NLL / ECE (67 tasks) | held-out acc / NLL / ECE (28 tasks) |
|---|---|---|
| decider-2b v10, T=1.30 | 0.806 / 0.474 / 0.038 | 0.755 / 0.622 / 0.084 |
| decider-2b v11, per-type map | 0.802 / 0.481 / 0.038 | 0.752 / 0.626 / 0.083 |
| decider-4b v1, T=1.05 | 0.834 / 0.404 / 0.027 | 0.788 / 0.558 / 0.071 |
| decider-4b v2, T=1.935 | 0.824 / 0.441 / 0.041 | 0.779 / 0.566 / 0.080 |
| **decider-4b v2.1 (this repository), per-type map** | **0.831 / 0.414 / 0.031** | **0.784 / 0.569 / 0.077** |
| decider-35b-a3b v1, T=1.08 | 0.855 / 0.357 / 0.026 | 0.810 / 0.497 / 0.069 |

Per-task paired intervals were not recomputed for v2.1; the v2 card (tag `v2`) has them for v2 against v10 and the 35B, and v2.1
is within 0.7 points of v2 on both halves.

**The same rows as v1 and v2** are in the table under [Changes from v2](#changes-from-v2): the four fixtures (847 in-task
validation rows, OpenJev, Mind2Web, TypeSafe), our held-out hard sets, JevBench, Bespoke, live browser tasks, zero-shot games,
text games, behaviour probes and the issue #9 cases.

**Ten text games, zero-shot** (`decider.games.play`, five episodes per game, greedy, eager; the 2B was trained on the first four,
the 4B on none): Pong −5 (v1 −21, v2 −21; the 2B 8), Breakout 35 (v1 14, v2 12; the 2B 22), CliffWalking −13 (v1 −13, v2 −60;
teacher −13), BabyAI-GoTo 0.19 (v1 0.54, v2 0.35), Freeway 0 (v2 1), Blackjack −0.6, FrozenLake, MiniGrid-Empty, LavaGap and DoorKey
0 as for v1 and v2. Greedy play takes the most probable option, so these results do not depend on the temperature.

**JevBench public items** (231 items of [Benchmark Heaven](https://benchmarkheaven.com/jev-models); argmax over the exact
label set with the request the harness's TypeSafe adapter builds). Read once for v2.1 at its global temperature (1.3.0) and once
with the map (1.4.0), both after the rule decisions: easy 1.000, standard 0.986, hard 0.649 (72 of 111) in both reads. Top-label
ECE on the hard tier is 0.184 with the map (0.210 at the global temperature); by answer type with the map, Choice 0.156 (67 items),
yes/no 0.259 (38), Score 0.354 (6). Mean confidence on the hard tier is 0.799 at accuracy 0.649. For comparison on the same items:
v2 0.676 (ECE 0.104 as read at T 1.719; 0.071 recomputed at its release temperature 1.935 with 6 Score items kept at 1.719), v1
0.550 (ECE 0.288), decider-35b-a3b 0.676, Jev 1.13.0 0.730. The public hard tier is 111 items (95% interval about ±9 points).

**Bespoke's public suite** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through `system_one` as
shipped). v1 and v2 through decider-ai 1.3.0, v2.1 through 1.4.0 with the map, same session. Nimble-9B (0.748 / 0.759 macro /
micro) and Jev 1.13.0 (0.760 / 0.773) are in Bespoke's report.

| subset (type) | decider-4b v1 | decider-4b v2 | decider-4b v2.1 |
|---|---|---|---|
| vitaminc-dev (choice) | 0.756 | 0.778 | 0.743 |
| massive-en-US (choice; trained) | 0.860 | 0.869 | 0.863 |
| massive-de-DE (choice) | 0.843 | 0.837 | 0.840 |
| boolq (noul; trained) | 0.873 | 0.860 | 0.867 |
| squad2 (noul) | 0.706 | 0.793 | 0.789 |
| paws (noul; trained) | 0.716 | 0.832 | 0.764 |
| multinli (choice; trained) | 0.933 | 0.926 | 0.933 |
| civil_comments (noul; trained) | 0.880 | 0.857 | 0.873 |
| aegis2 (noul) | 0.820 | 0.812 | 0.804 |
| helpsteer2 (score; trained) | 0.466 | 0.466 | 0.450 |
| summeval-relevance (score) | 0.425 | 0.463 | 0.396 |
| summeval-consistency (score) | 0.833 | 0.826 | 0.799 |
| pubmedqa (choice; trained) | 0.728 | 0.724 | 0.708 |
| **macro / micro** | 0.757 / 0.765 | 0.773 / 0.781 | 0.756 / 0.765 |
| macro over the subsets not trained on | 0.731 | 0.751 | 0.728 |

v2.1 is at v1's level on this suite and 1.6 points under v2 (macro). The largest differences to v2 are PAWS (0.764 against 0.832),
SummEval relevance (0.396 against 0.463) and VitaminC (0.743 against 0.778).

**Behaviour probes** (teacher-labelled, same probes as the other releases; v1 / v2 in brackets): generic-versus-specific bucket
choice 1.00 / 1.00 (1.00 / 1.00; 0.95 / 1.00), catch-all when nothing fits 0.95 (0.90; 0.95), abstention battery 8 of 8 (7 of 8;
8 of 8); model-router tier 0.968 (0.968; 0.935) and needs-live-data 0.839 (0.871; 0.839); command-risk classification 0.889
(0.889; 0.911) with no destructive command called safe, touches-outside-project 0.933 (0.956; 0.933); browser-agent element and
action choice 0.938 / 0.750 (0.875 / 0.875; 0.938 / 0.938). The yes/no Brier score on the probe batteries is 0.010 with the map
and 0.006 at the global temperature: the yes/no temperature 1.560 flattens easy yes/no answers there.

## Calibration

The per-type map was fitted on everyday regression rows and our own validation rows. It helps where it can: yes/no and Score
answers on the held-out sets. It does not change Choice answers much, because the Choice temperature fitted on a pool that is
94% everyday rows stays at 1.110. On the held-out generated families, whose answers are two-thirds Choice, v2.1 is
overconfident: calibration error 0.147 with the map, where our own release limit is 0.08. v2 is at 0.046 there, only because its
one temperature of 1.935 flattens every answer, which is also what cost it sampled play. ECE below is 10-bin top-label ECE on
stored temperature-1 logits read at the served temperatures.

| set | rows | accuracy | ECE at the global T / with the map | choice / noul / score ECE with the map |
|---|---|---|---|---|
| heldout_jb | 5,000 | 0.556 | 0.163 / 0.147 | 0.170 / 0.101 / 0.108 |
| test_teacher2 | 449 | 0.820 | 0.050 / 0.043 | 0.057 / 0.048 / 0.086 |
| guard | 2,994 | 0.819 | 0.019 / 0.017 | 0.017 / − / − |
| cal_human | 1,595 | 0.856 | 0.019 / 0.021 | 0.021 / − / − |

On the held-out generated families: v1 (T 1.05) 0.260, v2 (T 1.935) 0.046, v2.1 at T = 1 0.182. A map fitted without the
regression rows (Choice 1.259) would reach 0.127 there and would make the regression set less calibrated (in-task ECE 0.0375
against 0.0309); it was not used. On the fixtures (10-bin ECE, map): 847 validation rows 0.035 (v1 0.037, v2 0.031), TypeSafe
0.063 (v1 0.124, v2 0.072), OpenJev 0.154 (v1 0.158, v2 0.131), Mind2Web 0.017 (v1 0.016, v2 0.051). The Decision Index
sample was not read for v2.1. If you route on confidence, calibrate on your own labels; `python -m decider.calibrate` fits a
per-type map from your own answers read at temperature 1.

## Speed

v2.1 has the same architecture and size as v1 and v2, and the temperature map is a division per answer; the speed was not
measured again. The measurements of v1 and v2 (the same weights layout): in one session on one unshared NVIDIA B300 (bf16), 34.6
ms (v2) and 32.4 ms (v1) median per decision over 200 game-state decisions of 156 tokens median, batch of one, eager PyTorch
without CUDA graphs or `torch.compile`, timed around the forward pass with `torch.cuda.synchronize()`. The eager path is
launch-bound, so host load changes it: v1's first measurement was 24.7 ms (10th to 90th percentile 24.6 to 43.5 ms), with
decider-2b at 17.9 ms and decider-35b-a3b at 41.4 ms on the same decisions and method. With the helper's CUDA graphs and
`torch.compile`, one support-ticket request (228 tokens, 3 questions) takes 5.2 ms (FP8 5.0 ms); a batch of 32 such states takes
81.5 ms, 1,178 decisions per second (FP8 71.2 ms, 1,349 per second). The HTTP server (`/decide`, bf16) answers 72.8 requests per
second at a median of 13.3 ms with one client and 190 requests per second with 64 clients.

## Limitations

* No reinforcement-learning stage: stated beliefs about action outcomes were not trained against exact laws. On live browser
  tasks v2.1 is at 93.2% sampled and 87.5% on the six held-out tasks; decider-2b v10, which has the RL stage, is at 93.2% and 91.7%.
* Overconfident on hard multi-step items: calibration error 0.147 on our held-out generated families and 0.184 on the JevBench
  public hard tier, against v2's 0.046 and 0.104. On those items, do not read a confidence of 0.8 as an 80% chance of being right.
* Issue #9 form case c_1 is answered wrongly (see Changes from v2); BabyAI-GoTo 0.19 against v1's 0.54; greedy bag-draw play 9.4
  points under v1.
* About 0.3 points under v1 on the regression set and 1.1 points under it on the 847 validation rows; needs-live-data and
  touches-outside-project probes one item under v1 each.
* The per-type map needs decider-ai 1.4.0 or later. With 1.3.0 or earlier, or with `DECIDER_TEMPERATURE` set, every answer uses
  1.099: the answers are the same, yes/no answers are sharper (on the held-out generated families their ECE is 0.146 instead of
  0.101) and Score answers are sharper.
* Below the 35B by 2.4 / 2.6 points on the regression set and 2.7 on the JevBench public hard tier (3 items).
* The JevBench public hard tier is 111 items; differences of a few items between versions are within its noise.
* The stage-2 LoRA was trained only in the plain state-first layout. The schema-first layout (the server's opt-in schema cache)
  was not measured on v2.1, and `decider_config.json` does not mark it as trained for that layout (`schema_first_trained:
  false`), so `DECIDER_SCHEMA_CACHE=1` does not turn the schema cache on for this model.
* Mixture v2's 26 additional public datasets and ten programmatic families, and stage 2's generators and document questions, are
  described above but their builders are not in the public package; `scripts/train.sh full` reproduces the 60% of stage 1's data
  that is the public mixture.
* English is the main language; the multilingual rows (XNLI, PAWS-X, MASSIVE, Belebele, XCOPA) are a small share of the data
  and were not measured beyond the mixture-v2 evaluation set and Bespoke's German MASSIVE subset.
* Everything else in the decider-2b card's limitations (packed questions see each other, long JSON arrays by position, full
  label sets against sampled options, abstention wording, rules in the question) applies; those shapes were not re-measured at
  this size.

## Changelog

| version | what changed |
|---|---|
| **v2.1** (2026-09-24, these weights) | v1 + a merged LoRA on v2's rows (29,325 after removing 31 that overlapped evaluation files), with the replay rows trained toward v1's own answer distribution; temperature 1.099 and `temperature_by_type` {choice 1.110, noul 1.560, score 1.287} (decider-ai 1.4.0; older versions use 1.099). Against v2: sampled play recovered (bag-draw 52.0% against 37.9%, browser 93.2% against 88.1%), CliffWalking −13 against −60, regression set +0.7 / +0.5; hard sets at v2's level; less calibrated on hard items (0.147 against 0.046); issue #9 c_1 still wrong. Did not pass its pre-registered rules; released on the full comparison |
| v2 (2026-09-24, Hub tag `v2`) | v1 + a merged LoRA (rank 64, attention and MLP, 2 epochs, 29,356 rows: generated decision families, document questions written by Qwen3.6-27B and kept when two independent answers agreed, human-labelled public sets, replay of mixture v2); temperature 1.935. Better on hard judgments, TypeSafe, OpenJev and Bespoke's suite; worse on sampled play, some text games and by about 1 point on everyday tasks |
| v1 (2026-09-22, Hub tag `v1`) | first release: one pass over mixture v2 on Qwen3.5-4B-Base with AdamW on bf16 parameters, no RL stage; temperature 1.05 |

The GitHub repository's [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) lists every
decider release.

## Reproduction

Code, data registry, training and evaluation scripts and the per-version history: https://github.com/Mapika/decider
(`docs/HISTORY.md`, section "decider-4b"). Stage 1 was trained with the data-parallel trainer of the architecture A/B study
(`arch_ab/train_dp_optvar.py` in the research repository, optimizer variant `bf16`); stage 2 with a LoRA trainer in the research
repository. Both were evaluated with the public `decider.evaluate` and the head-to-head tools. `eval_results.json` in this
repository has the regression metrics with the map and at the global temperature, our held-out sets by answer type, the
fixtures, games, browser and Bespoke results with the paired comparisons against v1 and v2, the text games, the behaviour
probes, the issue #9 cases and the JevBench public items.

**Independence.** This is an independent project. It is not affiliated with or endorsed by TypeSafe AI. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev); nothing was distilled from Jev. The training data is public
datasets, programmatically generated rows with verifiable gold, and data written or labelled by local Qwen3.5-27B and
Qwen3.6-27B models. License: Apache 2.0.
