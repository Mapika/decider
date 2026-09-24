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
generated families with verifiable gold (742M tokens). Stage 2 (v2): a LoRA of rank 64 on the attention and MLP weights,
trained for 2 epochs on 29,356 rows of harder decisions and replay, then merged into the weights. There is no
reinforcement-learning stage. **This repository holds v2**, the bf16 weights (8.4 GB). v1 stays available under the Hub tag
`v1` (see [Changes from v1](#changes-from-v1) for who should keep using it). The other sizes are listed under The decider
family. `decider/` in this repository is the inference subset of the GitHub package.

Against decider-2b v10 on the same rows: accuracy is higher on 66 of the 95 regression tasks (in-task 0.824 against 0.805,
held-out 0.779 against 0.755), +1.8 points on the 847 validation rows, +3.5 on OpenJev, +4.8 on Mind2Web, +5.9 on the TypeSafe
workflow rows (interval includes zero), JevBench hard tier 0.676 against 0.459, Bespoke's public suite 0.773 against 0.704 macro.
Against decider-35b-a3b it is 3.1 to 3.2 points lower on the regression set, 1.6 to 5.0 points lower on three fixtures, level on
TypeSafe and level on the JevBench hard tier (0.676 both), at 8.4 GB instead of 65 GB. On live browser tasks it plays at the level
of the 2B in greedy mode (92.6% against 90.9%) and below it on the six held-out tasks (81.2% against 91.7%), because it has no RL
stage. Details under Evaluation.

**Contents:** [Changes from v1](#changes-from-v1) · [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Training](#training) · [Evaluation](#evaluation) · [Calibration](#calibration) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

## Changes from v1

v2 is v1 plus the stage-2 LoRA described under Training. It is better than v1 on hard and document-based judgments and on
TypeSafe, OpenJev and Bespoke's suite, and worse on sampled game and browser play, on some text games and by about 1 point on
everyday tasks. All rows below are on identical inputs, v1 at its stored temperature 1.05 and v2 at 1.935, through decider-ai
1.2.1; intervals are 95% paired bootstrap intervals of v2 minus v1 (tasks, rows, items, boards or task-seed pairs).

| set | v1 | v2 | v2 minus v1 |
|---|---|---|---|
| regression set, 67 in-task tasks, accuracy / NLL / ECE | 0.834 / 0.404 / 0.027 | 0.824 / 0.441 / 0.041 | accuracy −1.0 (−1.2 to −0.7), NLL +0.037 (+0.029 to +0.046), ECE +0.014 (+0.008 to +0.021) |
| regression set, 28 held-out tasks | 0.788 / 0.558 / 0.071 | 0.779 / 0.566 / 0.080 | accuracy −0.9 (−2.0 to +0.1), NLL +0.008 (−0.024 to +0.035), ECE +0.009 (−0.003 to +0.021) |
| 847 in-task validation rows, accuracy / NLL | 86.1% / 0.417 | 85.0% / 0.419 | −1.1 (−2.6 to +0.5); NLL +0.002 (−0.019 to +0.021) |
| OpenJev, 5,252 rows | 63.9% / 0.893 | 66.7% / 0.789 | +2.8 (+1.8 to +3.9); NLL −0.104 (−0.121 to −0.087) |
| Mind2Web, 1,770 rows | 88.4% / 0.366 | 87.4% / 0.393 | −1.0 (−2.0 to +0.1); NLL +0.027 (+0.011 to +0.044) |
| TypeSafe workflow decisions, 102 rows | 81.4% / 0.611 | 86.3% / 0.410 | +4.9 (−1.0 to +10.8); NLL −0.201 (−0.348 to −0.072) |
| Bespoke's public suite, macro / micro | 0.757 / 0.765 | 0.773 / 0.781 | macro +1.6 |
| JevBench public items, easy / standard / hard | 1.000 / 0.958 / 0.550 | 1.000 / 0.986 / 0.676 | hard +12.6 (+4.5 to +20.7) |
| JevBench hard tier, top-label ECE (v2: recomputed at T 1.935 from stored probabilities, the 6 Score items at T 1.719) | 0.288 | 0.071 | |
| Decision Index 4,000-request sample, calibration error (site definition; v2 recomputed at T 1.935 from stored probabilities) | 0.086 | 0.074 | |
| live MiniWoB++, 22 tasks x 8 seeds, greedy (all / 6 held-out tasks) | 91.5% / 75.0% | 92.6% / 81.2% | +1.1 (−1.7 to +4.0) / +6.2 (−2.1 to +16.7) |
| live MiniWoB++, sampled (all / 16 rewarded / 6 held-out) | 90.9% / 96.1% / 77.1% | 88.1% / 93.0% / 75.0% | −2.8 (−7.4 to +1.7) / −3.1 (−7.8 to +0.8) / −2.1 (−14.6 to +10.4) |
| zero-shot games, win rate, sampled (234 boards) | 27.8% | 22.4% | −5.3 (−7.6 to −3.1) |
| zero-shot games, greedy | 29.1% | 27.8% | −1.3 (−5.6 to +2.6) |
| bag-draw games alone, sampled / greedy | 56.6% / 62.5% | 37.9% / 48.4% | −18.8 (−24.6 to −12.9) / −14.1 (−23.4 to −6.2) |
| ten text games, greedy: CliffWalking / BabyAI-GoTo / Breakout | −13 / 0.54 / 14 | −60 / 0.35 / 12 | |
| behaviour probes: model-router tier / needs-live-data / touches-outside-project | 0.968 / 0.871 / 0.956 | 0.935 / 0.839 / 0.933 | |
| behaviour probes: abstention battery / catch-all / command risk / browser element and action | 7 of 8 / 0.90 / 0.889 / 0.875 and 0.875 | 8 of 8 / 0.95 / 0.911 / 0.938 and 0.938 | |

The v1 column is v1 measured again on 2026-09-24 through decider-ai 1.2.1, so that both versions are read by the same code on
the same day. It differs from the numbers first published for v1 by at most 1.0 point on the regression set, the fixtures and
JevBench, by up to 2.1 points on the live browser and game rows (sampled held-out browser tasks 77.1% against 79.2%, bag-draw
greedy 62.5% against 60.9%), and on Breakout (14 against 18). The first-published values are in the v1 card under the tag `v1`.

**Regressions, stated plainly.**
* Sampled play in games: the bag-draw games (choose the bag with the highest expected value) fall from 56.6% to 37.9% wins in
  sampled play, and zero-shot games overall from 27.8% to 22.4%. An earlier candidate (the Qwen3.5-4B instruct model plus a LoRA on
  15,768 of these 29,356 rows) showed a drop of 12.5 points on the same bag-draw games; this points to the new data, but it was not tested
  directly.
* Sampled browser play: −2.8 points over all 22 tasks and −3.1 on the 16 rewarded tasks (intervals include zero); the losses
  are on click-dialog-2, click-tab, click-checkboxes-large, click-checkboxes-soft, click-checkboxes-transfer and click-option,
  the gains on click-tab-2, click-tab-2-hard and focus-text-2.
* Text games (greedy): CliffWalking −60 against −13, BabyAI-GoTo 0.35 against 0.54, Breakout 12 against 14.
* Behaviour probes: model-router tier 0.935 against 0.968, needs-live-data 0.839 against 0.871, touches-outside-project 0.933
  against 0.956, generic bucket choice 0.95 against 1.00.
* Everyday tasks: regression in-task accuracy −1.0 point and held-out −0.9 (the held-out interval includes zero), lower on 75
  of 95 tasks, mostly by under 3 points; the largest losses are CommitmentBank −8.9, PubMedQA −6.2, hate-speech tweets −4.7,
  emotion −4.3, LIAR −3.9, QuALITY (full) −3.3, MedQA −3.0. Regression-set calibration is worse (in-task ECE 0.041 against
  0.027), and rating (Score) tasks are the least calibrated in the probe (see Calibration).

If you rely on sampled play (games, browser agents that sample actions) or on the probes above, pin revision `v1`. The
package loads a local folder, so download the revision first:

```python
from huggingface_hub import snapshot_download
from decider.infer import Decider
d = Decider(snapshot_download("Mapika/decider-4b", revision="v1"))
```

For the HTTP server, set `DECIDER_MODEL` to the same downloaded folder.

**How v2 was chosen.** The training run had a pre-registered rule for replacing v1: the candidate had to beat an earlier
candidate (the Qwen3.5-4B instruct model plus a LoRA on 15,768 of the same hard-decision rows) on our own held-out hard sets. v2 did not meet it:
it was 0.8 points short on the held-out teacher-written set and 0.5 points short on the held-out generated families, that is, it
tied the earlier candidate there. The earlier candidate had lost v1's everyday skills and v2 keeps them within about 1 point, so
v2 was then measured on every set of this card, and the release was decided on that full comparison with v1. The JevBench public
items were read once for v2, after selection; they were not used for training, selection or the temperature.

## The decider family

All six repositories share one interface (`decider.infer.Decider`, `POST /v1/systemone` in TypeSafe's format) and one
readout: the letter logits at an answer slot, softmaxed over the options. Pick by size and input.

| model | base | weights | use it for | numbers |
|---|---|---|---|---|
| [decider-2b](https://huggingface.co/Mapika/decider-2b) v10 | Qwen3.5-2B-Base | 3.5 GB bf16 | the default: routing, classification, judgments, browser agents; 4 ms per request with CUDA graphs on one GPU | regression set 0.805 in-task / 0.755 held-out; live browser 93%; Bespoke suite 0.704 |
| [decider-4b](https://huggingface.co/Mapika/decider-4b) v2 | Qwen3.5-4B-Base | 8.4 GB bf16 | the middle point: knowledge questions and hard judgments above the 2B in a dense 8.4 GB model; no RL stage; v1 under the tag `v1` for sampled play | 0.824 / 0.779, above the 2B on 66 of 95 tasks; JevBench hard 0.676; Bespoke 0.773 |
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
TypeSafe's `POST /v1/systemone` request shape (the official `typesafe-sdk` works with `TYPESAFE_BASE_URL` pointing at the
server; checked with decider-2b, not again with this model). Every question and every Score level is scored in its own row. The state may be a string, object or array of up to
32k tokens. See the decider-2b card for the full description of the request shape, field types and the schema cache.

Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton kernels for the Qwen3.5 linear-attention layers;
the model runs without it but several times slower). The weights take 8.4 GB in bf16. v2 uses the plain prompt layout, as v1
does, so it needs no new package version. decider-ai 1.0.2, 1.1.4, 1.2.1 and 1.2.2 and the `decider/` subset in this repository
(taken from 1.2.2) were checked on CUDA: each loads v2 with the name `decider-4b-v2` and temperature 1.935 and gives the same
probabilities to four decimals on the usage example above, on the eager path and on the CUDA-graph path (the two paths differ in
the third decimal, as for every model). The 1.2.2 HTTP server was checked with one `/v1/systemone` request. Other versions,
devices and the FP8 path were not checked on v2. On Blackwell GPUs use 1.0.2 or later (1.0.0 and 1.0.1 have the cuDNN attention fault fixed
in 1.0.2). The model is dense, so the CUDA-graph engine, `torch.compile` and the FP8 path of the helper package apply to it as
to decider-2b (`use_graphs=False` selects eager PyTorch). The measurements below were taken with the eager path unless stated.

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
probs = torch.softmax(logits[letters].float() / 1.935, -1)     # 1.935 is the stored temperature (v1: 1.05)
```

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled),
so the model conditions on the supplied candidates rather than on a fixed head.

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

**Stage 2 (v2).** A LoRA of rank 64 (alpha 128) on the attention and MLP weights of v1, trained with cross-entropy on the slot
readout for 2 epochs over 29,356 rows in v1's plain state-first layout with isolated Score levels, then merged into the bf16
weights:

| source | rows | content |
|---|---|---|
| generated decision families | 8,000 | ten families (temporal and numeric decisions, subtle answer judgment, long policies, multi-hop lookup, abstention, probability, constrained trade-offs, safety judgment, paraphrase sensitivity, adversarial traps); the answers are computed by the generating code |
| questions over business documents, written by Qwen3.6-27B with thinking on | 11,356 | in two rounds, one realistic business document (up to 34 domains and 22 document kinds) plus three or four typed questions per writer call; each question was answered twice more by the same model in fresh contexts, with shuffled options and without the writer's answer, and kept only when both answers agreed with the writer's (89% and 91% kept) |
| human-labelled public sets (training halves) | 3,300 | MMLU, ARC, CommonsenseQA, BoolQ, MNLI, SNLI, Banking77, RACE, OpenBookQA, LogiQA 2, MedQA, Winogrande |
| replay of mixture v2 | 6,700 | 100 rows from the training half of each of the 67 in-task regression tasks |

| stage 2 | |
|---|---|
| trainable parameters | LoRA rank 64, alpha 128, on the attention and MLP projections; merged after training |
| schedule | learning rate 1e-4, 5% warm-up then cosine, 1,517 steps of 65,536 tokens (2 epochs), seed 0 |
| hardware | one NVIDIA B300, 87 minutes |

No JevBench item and no Decision Index item was used for training, for writing the generators or the document questions, for
selecting the checkpoint, or for the temperature. The ten generated families and the skill list of the document questions were
written from the family names that JevBench publishes for its sealed set, not from its items. Every training row was checked
against every evaluation file used for selection and against the evaluation half of all 95 regression tasks: there is no exact
state-and-question overlap. The held-out sets used for selection are generated families from held-out templates and document
questions from business domains that are not in the training data.

**Temperature.** 1.935, fitted by NLL (all rows pooled) on the in-task half of the public regression set without Banking77,
CLINC-OOS, MMLU, ARC, Winogrande and HellaSwag: 61 tasks, 102,804 rows. Fitted on all 67 in-task tasks the value is 1.942. v1's
temperature (1.05) was fitted on the 67 in-task tasks, the same set and method as decider-2b and decider-35b-a3b. No
reinforcement-learning stage was run on this model; the RL recipe of decider-2b v10 is documented in `docs/RL.md` of the GitHub
repository.

## Evaluation

All v2 numbers are at the stored temperature 1.935 through decider-ai 1.2.1, except where a paragraph says otherwise. Greedy
play and accuracy do not depend on the temperature; sampled play and calibration do, and those were measured again at 1.935.

**Public regression set**, rebuilt on this machine (95 tasks: 67 in-task, 28 held-out; large label sets sub-sampled to 10
options; one temperature per model fitted on in-task data). All four rows are the same rows. ECE is the expected calibration
error with 15 bins; intervals are 95% bootstrap intervals over tasks, paired.

| model | in-task acc / NLL / ECE (67 tasks) | held-out acc / NLL / ECE (28 tasks) |
|---|---|---|
| decider-2b v10, T=1.30 | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| decider-4b v1, T=1.05 | 0.834 / 0.404 / 0.027 | 0.788 / 0.558 / 0.071 |
| **decider-4b v2 (this repository), T=1.935** | **0.824 / 0.441 / 0.041** | **0.779 / 0.566 / 0.080** |
| decider-35b-a3b v1, T=1.08 | 0.855 / 0.357 / 0.026 | 0.810 / 0.497 / 0.069 |

Against v10: +1.9 in-task points (+1.0 to +2.8) and +2.4 held-out points (+1.1 to +3.7); NLL −0.032 (−0.054 to −0.011) and
−0.056 (−0.094 to −0.024); accuracy higher on 66 of the 95 tasks, equal on two, lower on 27 (hate-speech tweets −4.9 points,
emotion −3.6, ADE −2.9, abstention probe −2.3, SST-5 −2.1, HelpSteer3 preference −2.1, LIAR −2.1, the others under 2). The
largest gains are on knowledge and reasoning tasks: MedQA +13.4, MedMCQA +12.3, Winogrande +11.7, TruthfulQA +11.6, MMLU +11.3,
OpenBookQA +8.8, StrategyQA +8.0. Mean accuracy over nine knowledge tasks (MMLU, ARC, HellaSwag, MedQA, MedMCQA, Social IQa,
COPA, TruthfulQA, TREC): 0.797 (v1 0.801) against v10's 0.707 and the 35B's 0.862. Against the 35B: −3.1 in-task (−4.0 to −2.3)
and −3.2 held-out points (−4.5 to −1.8), lower on 85 of 95 tasks and higher on 7; the largest gaps are MedQA −17.7, HelpSteer3
preference −16.6, MedMCQA −11.9, StrategyQA −10.8, TruthfulQA −10.0.

<details>
<summary><b>Per-task accuracy / ECE on the 28 held-out datasets, decider-2b v10, this model, decider-35b-a3b</b></summary>

| task | decider-2b v10 | decider-4b v2 | decider-35b-a3b |
|---|---|---|---|
| abstain_probe | 0.606 / 0.134 | 0.582 / 0.133 | 0.622 / 0.085 |
| ade | 0.817 / 0.038 | 0.789 / 0.107 | 0.837 / 0.035 |
| arena_pref | 0.483 / 0.189 | 0.503 / 0.218 | 0.521 / 0.121 |
| bbc_news | 0.927 / 0.013 | 0.951 / 0.068 | 0.944 / 0.027 |
| cb | 0.857 / 0.093 | 0.839 / 0.118 | 0.893 / 0.084 |
| cr_reviews | 0.903 / 0.031 | 0.898 / 0.046 | 0.914 / 0.033 |
| dbpedia_l2 | 0.950 / 0.018 | 0.951 / 0.017 | 0.961 / 0.010 |
| dbpedia_l3 | 0.987 / 0.005 | 0.991 / 0.012 | 0.992 / 0.004 |
| dolly_category | 0.299 / 0.203 | 0.374 / 0.142 | 0.354 / 0.098 |
| fin_phrasebank | 0.694 / 0.042 | 0.699 / 0.077 | 0.759 / 0.110 |
| fin_sentiment | 0.793 / 0.058 | 0.826 / 0.051 | 0.839 / 0.136 |
| hermes_tools | 0.723 / 0.208 | 0.737 / 0.124 | 0.799 / 0.085 |
| hwu64 | 0.961 / 0.030 | 0.963 / 0.037 | 0.975 / 0.022 |
| massive_scenario | 0.756 / 0.041 | 0.793 / 0.052 | 0.799 / 0.027 |
| offtopic_probe | 0.841 / 0.027 | 0.823 / 0.043 | 0.870 / 0.038 |
| paws | 0.724 / 0.145 | 0.794 / 0.124 | 0.729 / 0.169 |
| pubmedqa | 0.756 / 0.085 | 0.758 / 0.055 | 0.820 / 0.078 |
| quality | 0.494 / 0.233 | 0.561 / 0.179 | 0.632 / 0.096 |
| quality_full | 0.508 / 0.198 | 0.505 / 0.188 | 0.565 / 0.112 |
| reward_bench | 0.819 / 0.045 | 0.853 / 0.032 | 0.919 / 0.024 |
| sciq | 0.982 / 0.024 | 0.991 / 0.021 | 0.993 / 0.011 |
| social_iqa | 0.708 / 0.077 | 0.759 / 0.062 | 0.823 / 0.025 |
| strategyqa | 0.552 / 0.138 | 0.632 / 0.157 | 0.739 / 0.036 |
| student_questions | 0.925 / 0.045 | 0.940 / 0.052 | 0.954 / 0.090 |
| trec | 0.784 / 0.066 | 0.842 / 0.027 | 0.832 / 0.160 |
| truthfulqa | 0.537 / 0.090 | 0.654 / 0.063 | 0.754 / 0.068 |
| tweet_irony | 0.795 / 0.052 | 0.827 / 0.027 | 0.861 / 0.129 |
| xstory_cloze | 0.962 / 0.017 | 0.971 / 0.017 | 0.995 / 0.016 |

</details>

**On the same rows as decider-2b v10 and decider-35b-a3b.** Every row below is scored by all three models on identical inputs
and seeds. Intervals are 95% paired bootstrap intervals (rows for the fixtures, boards for the games, task-seed pairs for the
browser). The table under [Changes from v1](#changes-from-v1) has v1 on the same rows.

| | decider-2b v10 | **decider-4b v2** | decider-35b-a3b | 4B minus 2B | 4B minus 35B |
|---|---|---|---|---|---|
| 847 in-task validation rows, accuracy / NLL | 83.2% / 0.444 | 85.0% / 0.419 | 90.0% / 0.329 | +1.8 (−0.5 to +3.9) | −5.0 (−7.1 to −2.8) |
| OpenJev, 5,252 rows, accuracy / NLL | 63.3% / 0.916 | 66.7% / 0.789 | 68.3% / 0.752 | +3.5 (+2.2 to +4.7) | −1.6 (−2.8 to −0.3) |
| Mind2Web element and action choice, 1,770 rows | 82.7% / 0.543 | 87.4% / 0.393 | 89.6% / 0.316 | +4.8 (+3.1 to +6.4) | −2.2 (−3.7 to −0.6) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 80.4% / 0.585 | 86.3% / 0.410 | 86.3% / 0.342 | +5.9 (−2.9 to +14.7) | 0.0 (−5.9 to +4.9) |
| Bespoke's public suite, 13 subsets, macro / micro | 0.704 / 0.711 | 0.773 / 0.781 | 0.774 / 0.787 | | |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.889 / 0.459 | 1.000 / 0.986 / 0.676 | 1.000 / 0.972 / 0.676 | | |
| live MiniWoB++ click tasks, 22 tasks x 8 seeds, greedy play | 90.9% | 92.6% | 97.2% | +1.7 (−4.0 to +7.4) | −4.5 (−8.0 to −1.7) |
| the same, 6 tasks v10 never used for reward, greedy | 91.7% | 81.2% | 97.9% | −10.4 (−22.9 to +2.1) | −16.7 (−27.1 to −6.2) |
| live MiniWoB++ click tasks, sampled play | 93.2% | 88.1% | 86.4% | −5.1 (−10.8 to +0.6) | +1.7 (−3.4 to +6.8) |
| the same, 6 held-out tasks, sampled | 91.7% | 75.0% | 79.2% | −16.7 (−31.2 to −4.2) | −4.2 (−16.7 to +8.3) |
| zero-shot games, win rate, sampled play (234 boards) | 23.7% | 22.4% | 24.1% | −1.3 (−4.2 to +1.5) | −1.7 (−3.7 to +0.2) |
| zero-shot games, greedy play | 26.5% | 27.8% | 37.2% | +1.3 (−4.3 to +6.8) | −9.4 (−15.0 to −3.8) |
| bag-draw games alone, sampled | 41.4% | 37.9% | 41.8% | −3.5 (−9.0 to +2.0) | −3.9 (−8.6 to +0.8) |

On the two fixtures whose rows are in neither model's training data (TypeSafe, OpenJev) v2 is above the 2B (OpenJev +3.5, interval
excludes zero; TypeSafe +5.9, interval includes zero) and on TypeSafe level with the 35B. Stage 2 contains teacher-written and
human-labelled judgment rows, not TypeSafe or OpenJev rows. The browser rows show the missing RL stage: greedy play is level
with v10 over all tasks and 10 points below it on the six tasks that v10's RL never rewarded (interval includes zero), where the
4B's weakest tasks are focus-text-2 (0.375) and click-collapsible-2 (0.5). On the bag-draw games v1 won about 15 points more
often than either other model in sampled play; v2 does not (37.9% against 41.4% and 41.8%). On tic-tac-toe, the slippery grid
and minesweeper every model is near the random floor.

**Ten text games, zero-shot** (`decider.games.play`, five episodes per game, greedy, eager; the 2B was trained on the first
four, the 4B and the 35B on none): Pong −21 (teacher 8, 2B 8, 35B −21), Breakout 12 (22, 22, 6), CliffWalking −60 (teacher −13,
2B −13, 35B −1,248), MiniGrid-Empty 0 (teacher 0.96, both others 0), Freeway 1 (teacher 5, 2B 0, 35B 1), FrozenLake 0 (teacher 1,
others 0), Blackjack −0.6 (teacher −0.6, 2B −1, 35B −0.6), MiniGrid-LavaGap 0, MiniGrid-DoorKey 0 (teacher 0), BabyAI-GoTo 0.35
(teacher 0.34, 2B 0, 35B 0.19). v1 reached the teacher on CliffWalking (−13) and read 0.54 on BabyAI-GoTo; v2 keeps neither.
Neither version learns Pong from the text state. Greedy play takes the most probable option, so these results do not depend on
the temperature.

**JevBench public items** (231 items of [Benchmark Heaven](https://benchmarkheaven.com/jev-models); argmax over the exact
label set with the request the harness's TypeSafe adapter builds). The items were read once for v2, after selection, at the
candidate temperature 1.719; the accuracies do not depend on the temperature, and the calibration numbers below are recomputed
at 1.935 from the stored probabilities. Jev 1.13.0 is at 1.000 / 0.986 / 0.730, SemIf (also a Qwen3.5-4B) at 1.000 / 0.986 /
0.613, decider-35b-a3b at 1.000 / 0.972 / 0.676 on the same items. The one standard-tier family with misses is adequacy (0.92).
Hard-tier families (v1 as first published in brackets): trap 1.00 (1.00), hard routing 1.00 (1.00), adversarial 1.00 (0.67), probability 0.90 (0.60),
ambiguous 0.86 (0.29), multi-hop 0.72 (0.78), trade-off 0.67 (0.33), long policy 0.63 (0.47), judge-hard 0.47 (0.47),
temporal-numeric 0.27 (0.13). Top-label ECE is 0.008 / 0.085 / 0.071 by tier (v1, read again on 2026-09-24: 0.001 / 0.047 / 0.288;
the 35B's hard tier 0.151). On the hard tier the mean confidence is 0.74 at accuracy 0.68; 31% of hard items are answered at confidence 0.9 or more,
with accuracy 0.94 on them (v1: 50% at 0.73). The 18 Score items were answered with isolated levels, whose per-level values are
not stored, so they keep their T 1.719 probabilities inside these ECE values. The public hard tier is 111 items (95% interval
about ±9 points), and the gain over v1 there (+12.6) is larger than the gain on our own held-out hard sets.

**Bespoke's public suite** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through `system_one` as
shipped). Nimble-9B and Jev 1.13.0 numbers are copied from Bespoke's report.

| subset (type) | decider-2b v10 | decider-4b v2 | decider-35b-a3b | Nimble-9B | Jev 1.13.0 |
|---|---|---|---|---|---|
| vitaminc-dev (choice) | 0.639 | 0.778 | 0.795 | 0.766 | 0.801 |
| massive-en-US (choice; trained) | 0.823 | 0.869 | 0.880 | 0.869 | 0.874 |
| massive-de-DE (choice, German) | 0.797 | 0.837 | 0.869 | 0.834 | 0.869 |
| boolq (noul; trained) | 0.803 | 0.860 | 0.887 | 0.860 | 0.897 |
| squad2 (noul) | 0.776 | 0.793 | 0.749 | 0.806 | 0.829 |
| paws (noul; trained) | 0.720 | 0.832 | 0.768 | 0.828 | 0.892 |
| multinli (choice; trained) | 0.856 | 0.926 | 0.910 | 0.853 | 0.829 |
| civil_comments (noul; trained) | 0.840 | 0.857 | 0.907 | 0.703 | 0.810 |
| aegis2 (noul) | 0.728 | 0.812 | 0.808 | 0.812 | 0.804 |
| helpsteer2 (score; trained) | 0.426 | 0.466 | 0.478 | 0.390 | 0.341 |
| summeval-relevance (score) | 0.354 | 0.463 | 0.483 | 0.492 | 0.350 |
| summeval-consistency (score) | 0.660 | 0.826 | 0.757 | 0.757 | 0.812 |
| pubmedqa (choice; trained) | 0.724 | 0.724 | 0.768 | 0.756 | 0.772 |
| **macro / micro** | 0.704 / 0.711 | **0.773 / 0.781** | 0.774 / 0.787 | 0.748 / 0.759 | 0.760 / 0.773 |

On the six subsets whose training split is not in the mixture the macro accuracy is 0.751 (v1 0.731, 2B 0.659, 35B 0.744). v2 is
1.3 points above Jev 1.13.0 and 2.5 above Nimble-9B on the average (v1: 0.757). It is still behind Jev where a claim has to be
checked against evidence that nearly matches it (PAWS 0.832 against 0.892, VitaminC 0.778 against 0.801) and on SQuAD2
answerability (0.793 against 0.829), where it is now 1.7 points above the 2B (v1 was 7 points under it).

**Behaviour probes** (teacher-labelled, same probes as the other releases; v1 in brackets): generic-versus-specific bucket choice
0.95 / 1.00 (1.00 / 1.00), catch-all when nothing fits 0.95 (0.90; 35B 0.95), abstention battery 8 of 8 (7 of 8); model-router
tier 0.935 (0.968) and needs-live-data 0.839 (0.871); command-risk classification 0.911 (0.889) with no destructive command
called safe (35B 0.933), touches-outside-project 0.933 (0.956); browser-agent element and action choice 0.938 / 0.938 (0.875 /
0.875; 35B 0.938). Scoring a Score level alone against scoring it with its neighbours changes accuracy by at most 2.0 points on
five rating datasets, and the per-level fits sum to between 0.92 and 1.00 (this probe reads the raw logits, T=1).

## Calibration

The stored temperature 1.935 was fitted on 61 of the 67 in-task regression tasks (see Training). On the regression set v2 is less
well calibrated than v1 and the 35B (ECE 0.041 / 0.080 against v1's 0.027 / 0.071 and the 35B's 0.026 / 0.069), with per-task
exceptions: hate-speech tweets 0.349 (in-task), Arena preferences 0.218, QuALITY (full) 0.188, QuALITY 0.179, StrategyQA 0.157,
Dolly categories 0.142, abstention probe 0.133; the in-task preference tasks HH-RLHF (0.113) and SHP (0.111) are also above 0.1.
On the five rating datasets of the Score-level probe, which reads the raw logits (T=1) and scores all levels together, the ECE
is 0.13 to 0.27 (v1 0.05 to 0.09 in the same probe); these sets were not measured at the served temperatures. One temperature
does not fit every task type. Outside the regression set v2 is better calibrated than v1 on most sets:

| set | measure | decider-2b v10 | decider-4b v1 (read again 2026-09-24) | decider-4b v2 | decider-35b-a3b |
|---|---|---|---|---|---|
| JevBench hard tier, 111 items | top-label ECE | 0.304 | 0.288 | 0.071 | 0.151 |
| TypeSafe, 102 rows | 10-bin ECE / mean total variation to the frontier reference | 0.091 / 0.242 | 0.124 / 0.232 | 0.072 / 0.196 | 0.065 / 0.165 |
| OpenJev, 5,252 rows | 10-bin ECE | 0.150 | 0.158 | 0.131 | 0.049 |
| 847 in-task validation rows | 10-bin ECE | | 0.037 | 0.031 | |
| Mind2Web, 1,770 rows | 10-bin ECE | | 0.016 | 0.051 | |
| Decision Index 4,000-request sample, 33 benchmarks | benchmark-weighted ECE (the site's definition) | 0.093 | 0.086 | 0.074 | 0.027 |
| the same | share of all answers given at 95% or more confidence and wrong | 0.010 | 0.022 | 0.012 | 0.005 |
| the same | mean confidence against accuracy | 0.658 / 0.566 | 0.722 / 0.637 | 0.690 / 0.637 | 0.684 / 0.690 |
| the same | sample index (about 4 points under a full run) | 42.3 | 48.5 | 48.3 | 50.4 |

The Decision Index rows are a readout of the 4,000-request sample (about 88 cases per benchmark) through the public server,
computed with the site's definition (benchmark-weighted pooled bins, ten bins; the same computation gives 0.027 on the 35B
against the site's published 0.031). v2 was read once at the candidate temperature 1.719 (ECE 0.090, sample index 48.3); the
chosen answers do not change with the temperature, and the calibration rows above are recomputed at 1.935 from the stored
probabilities. The JevBench hard-tier value is recomputed the same way, except its 6 Score items, which keep their T 1.719
probabilities. The sample index was not recomputed. Nothing was fitted on index rows. v2's confidence exceeds its accuracy by
0.053 on that sample (v1 0.085, the 2B 0.092; the 35B has no gap). The gap is per-benchmark heterogeneity, not a global scale: a
single temperature that removes it on the knowledge benchmarks would make the wide label sets underconfident. If you route on
confidence, calibrate on your own labels.

## Speed

v2 has the same architecture and size as v1, so it runs at v1's speed. These timings were taken with the same weights in the
candidate readout session of 2026-09-24 (the server was configured with the candidate temperature 1.719; the temperature does
not change the amount of computation). In one session on one unshared NVIDIA B300 (bf16), both models measured one after the
other: 34.6 ms (v1 32.4 ms) median per decision over 200 game-state decisions of 156 tokens median,
batch of one, eager PyTorch without CUDA graphs or `torch.compile`, timed around the forward pass with `torch.cuda.synchronize()`.
The eager path is launch-bound, so host load changes it: v1's first measurement was 24.7 ms (10th to 90th percentile 24.6 to
43.5 ms), with decider-2b at 17.9 ms and decider-35b-a3b at 41.4 ms on the same decisions and method. With the helper's CUDA
graphs and `torch.compile`, one support-ticket request (228 tokens, 3 questions) takes 5.2 ms (FP8 5.0 ms); a batch of 32 such
states takes 81.5 ms, 1,178 decisions per second (FP8 71.2 ms, 1,349 per second). The HTTP server (`/decide`, bf16) answers 72.8
requests per second at a median of 13.3 ms with one client and 190 requests per second with 64 clients.

## Limitations

* No reinforcement-learning stage: stated beliefs about action outcomes were not trained against exact laws, and on live
  browser tasks the model is 10 points below decider-2b v10 on the six held-out tasks in greedy play (81% against 92%, interval
  includes zero) and 17 points below it in sampled play (75% against 92%).
* Sampled play is worse than v1's: bag-draw games 37.9% against 56.6% wins, zero-shot games 22.4% against 27.8%, sampled browser
  play −2.8 points; CliffWalking −60 against −13. Pin revision `v1` if you depend on these (see Changes from v1).
* Less calibrated than v1 on the regression set (in-task ECE 0.041 against 0.027), on Mind2Web (0.051 against 0.016) and on
  rating tasks in the Score-level probe (ECE 0.13 to 0.27 on the five rating sets at T=1); still overconfident outside the regression set: OpenJev ECE 0.13,
  Decision Index sample ECE 0.074 where the 35B reads 0.027. On the Decision Index sample 1.2% of all answers are given at 95% or
  more confidence and are wrong, against 0.5% for the 35B. Calibration is measured on public datasets, not on your traffic.
* About 1 point under v1 on everyday tasks (regression set, validation rows, Mind2Web) and lower than v1 on the model-router and
  command-scope probes.
* Below the 35B on 85 of 95 regression tasks by 3.1 / 3.2 points and on three fixtures by 1.6 to 5.0 points; level on TypeSafe
  and on the JevBench hard tier. The largest gaps are on MedQA, MedMCQA, StrategyQA, TruthfulQA and preference judgments.
* The abstention probe is 2.4 points under the 2B (0.582 against 0.606), with ECE 0.133.
* The JevBench public hard tier is 111 items; v2's gain there (+12.6 points over v1) is larger than its gain on our own held-out
  hard sets, where it tied an earlier candidate. Do not read it as a gain of that size on hard items in general.
* The stage-2 LoRA was trained only in the plain state-first layout. The schema-first layout (the server's opt-in schema cache)
  was not measured on v2, and `decider_config.json` does not mark v2 as trained for it (`schema_first_trained: false`), so
  `DECIDER_SCHEMA_CACHE=1` does not turn the schema cache on for this model.
* Mixture v2's 26 additional public datasets and ten programmatic families, and stage 2's generators and document questions, are
  described above but their builders are not in the public package; `scripts/train.sh full` reproduces the 60% of stage 1's data
  that is the public mixture. On the held-out variants of mixture v2's programmatic families v1 was at 0.635 mean accuracy (logs
  0.28, plans 0.44, probability 0.43, code 0.88, policy 0.99); this was not measured on v2.
* English is the main language; the multilingual rows (XNLI, PAWS-X, MASSIVE, Belebele, XCOPA) are a small share of the data
  and were not measured beyond the mixture-v2 evaluation set.
* Everything else in the decider-2b card's limitations (packed questions see each other, long JSON arrays by position, full
  label sets against sampled options, abstention wording, rules in the question) applies; those shapes were not re-measured at
  this size.

## Changelog

| version | what changed |
|---|---|
| **v2** (2026-09-24, these weights) | v1 + a merged LoRA (rank 64, attention and MLP, 2 epochs, 29,356 rows: generated decision families, document questions written by Qwen3.6-27B and kept when two independent answers agreed, human-labelled public sets, replay of mixture v2); temperature 1.935; plain layout, checked with decider-ai 1.0.2, 1.1.4, 1.2.1 and 1.2.2. Better on hard judgments, TypeSafe, OpenJev and Bespoke's suite; worse on sampled play, some text games and by about 1 point on everyday tasks |
| v1 (2026-09-22, Hub tag `v1`) | first release: one pass over mixture v2 on Qwen3.5-4B-Base with AdamW on bf16 parameters, no RL stage; temperature 1.05 |

The GitHub repository's [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) lists every
decider release.

## Reproduction

Code, data registry, training and evaluation scripts and the per-version history: https://github.com/Mapika/decider
(`docs/HISTORY.md`, section "decider-4b"). Stage 1 was trained with the data-parallel trainer of the architecture A/B study
(`arch_ab/train_dp_optvar.py` in the research repository, optimizer variant `bf16`); stage 2 with a LoRA trainer in the research
repository. Both were evaluated with the public `decider.evaluate` and the head-to-head tools, and uploaded with
`scripts/upload_hf.py`. `eval_results.json` in this repository has the per-task regression metrics at T 1.935, the fixtures, JevBench, Bespoke's
suite, the games, the browser, the text games, the Decision Index calibration, the behaviour probes and speed, and the paired
comparisons with v1 (regression set, fixtures, JevBench, games, browser).

**Independence.** This is an independent project. It is not affiliated with or endorsed by TypeSafe AI. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev); nothing was distilled from Jev. The training data is public
datasets, programmatically generated rows with verifiable gold, and data written or labelled by local Qwen3.5-27B and
Qwen3.6-27B models. License: Apache 2.0.
