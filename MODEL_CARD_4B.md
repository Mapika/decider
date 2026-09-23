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
attention and 24 with gated delta-net linear attention, hidden size 2,560. One supervised pass of cross-entropy on the slot readout
over mixture v2: the public decision mixture of [decider-2b](https://huggingface.co/Mapika/decider-2b) plus 26 further public
decision datasets and ten programmatically generated families with verifiable gold (742M tokens in all). The optimizer is AdamW
applied directly to the bf16 parameters, the same as the public trainer of decider-2b. There is no reinforcement-learning stage.
**This repository holds v1**, the bf16 weights (8.4 GB). The other sizes are listed under The decider family. `decider/` in this
repository is the inference subset of the GitHub package.

Against decider-2b v10 on the same rows: accuracy is higher on 87 of the 95 regression tasks (in-task 0.834 against 0.805,
held-out 0.788 against 0.755), +2.7 points on the 847 validation rows, +0.6 on OpenJev, +5.7 on Mind2Web, level on the TypeSafe
workflow rows, JevBench hard tier 0.541 against 0.459, Bespoke's public suite 0.757 against 0.704 macro. Against
decider-35b-a3b it is 2.1 to 2.3 points lower on the regression set and 1.2 to 5.9 points lower on every fixture, at 8.4 GB
instead of 65 GB. On live browser tasks it plays at the level of the 2B in greedy mode (91.5% against 90.9%) and below it on the six
held-out tasks (75.0% against 91.7%), because it has no RL stage. Details under Evaluation.

**Contents:** [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Training](#training) · [Evaluation](#evaluation) · [Calibration](#calibration) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

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
d = Decider("Mapika/decider-4b")
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': ..., 'probs': {...}}, {'choice': 'yes', 'confidence': ..., 'probs': {...}}]
```

The API is the same as decider-2b's: `decide_batch` scores many states with many questions in one call, `abstain_below=t`
returns `None` under a confidence threshold, a question can have 2 to 255 options, and `system_one` / `decider.serve` accept
TypeSafe's `POST /v1/systemone` request shape (the official `typesafe-sdk` works with `TYPESAFE_BASE_URL` pointing at the
server). Every question and every Score level is scored in its own row. The state may be a string, object or array of up to
32k tokens. See the decider-2b card for the full description of the request shape, field types and the schema cache.

Requirements: `torch`, `transformers>=5`, and `flash-linear-attention` (Triton kernels for the Qwen3.5 linear-attention layers;
the model runs without it but several times slower). The weights take 8.4 GB in bf16. The model is dense, so the CUDA-graph
engine, `torch.compile` and the FP8 path of the helper package apply to it as to decider-2b (`use_graphs=False` selects eager PyTorch). The measurements below were taken with the eager path.

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
probs = torch.softmax(logits[letters].float() / 1.05, -1)      # 1.05 is the stored temperature
```

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled),
so the model conditions on the supplied candidates rather than on a fixed head.

## Training

One supervised pass over **mixture v2**, in the two prompt layouts (state-first and schema-first, 50/50), with isolated Score
levels and 10% abstention rows, 1,892,408 items and 742M tokens, pre-tokenized once and read in the same order by both ranks.
Three sources:

| source | rows | share of tokens | content |
|---|---|---|---|
| the public decision mixture of decider-2b (`scripts/train.sh full` of the GitHub repository) | 1,539,860 | 60% | about 95 public decision datasets, agent trajectories, Mind2Web element choice, teacher-written custom questions, Jev's input shapes |
| 26 further public datasets with gold labels | 131,318 | 8% | code defect, clone and review-needed judgments; log anomaly and severity (HDFS, BGL); legal (LEDGAR, Unfair-ToS, CaseHOLD, SCOTUS, ECtHR); tables (TabFact, WikiTQ, FeTaQA, InfoTabs, TAT-QA); finance headlines; German credit; symptom diagnosis and medical specialty; SciTail, SciEntsBank, Climate-FEVER; XNLI, PAWS-X, MASSIVE (multilingual), Belebele, XCOPA; MC-TACO, TRACIE, TimeQA; ProofWriter, RuleTaker, LogiQA 2; essay scoring; agent next action |
| ten programmatic families with verifiable gold, each with a held-out variant | 221,230 | 32% | code, dates and times, logs, long documents, plans, policies, probability, schedules, tables, tools |

Ten of the 26 public datasets (Belebele, code clone, ECtHR, essay scoring, InfoTabs, PAWS-X, RuleTaker, SCOTUS, XCOPA, TRACIE)
and the held-out variant of every programmatic family were kept out of training; every evaluation row was checked against every
training row of all three sources and 4,168 overlapping training pairs were dropped. No game rows are in mixture v2 (the
public mixture rebuilt on this machine has none), so every game result below is zero-shot.

| | |
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
the base model's knowledge is kept. The temperature 1.05 was fitted by NLL on the in-task half of the public regression set (67
tasks), the same set and method as for decider-2b and decider-35b-a3b. No reinforcement-learning stage was run on this model; the
RL recipe of decider-2b v10 is documented in `docs/RL.md` of the GitHub repository.

## Evaluation

**Public regression set**, rebuilt on this machine (95 tasks: 67 in-task, 28 held-out; large label sets sub-sampled to 10
options; one temperature per model fitted on in-task data). All three rows are the same rows. ECE is the expected calibration
error with 15 bins; intervals are 95% bootstrap intervals over tasks, paired.

| model | in-task acc / NLL / ECE (67 tasks) | held-out acc / NLL / ECE (28 tasks) |
|---|---|---|
| decider-2b v10, T=1.30 | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| **decider-4b v1 (this repository), T=1.05** | **0.834 / 0.404 / 0.027** | **0.788 / 0.558 / 0.071** |
| decider-35b-a3b v1, T=1.08 | 0.855 / 0.357 / 0.026 | 0.810 / 0.497 / 0.069 |

Against v10: +2.8 in-task points (+2.0 to +3.7) and +3.3 held-out points (+2.1 to +4.5); NLL −0.069 and −0.064; accuracy higher on
87 of the 95 tasks, equal on one, lower on seven (abstention probe −4.6 points, agent trajectories −1.3, HelpSteer3 preference
−1.1, offensive tweets −0.8, three others under 0.5). The largest gains are on knowledge and reasoning tasks: MedQA +16.5, MedMCQA
+13.7, Winogrande +12.2, TruthfulQA +12.0, MMLU +10.9, OpenBookQA +9.0, StrategyQA +8.4. Mean accuracy over nine knowledge tasks
(MMLU, ARC, HellaSwag, MedQA, MedMCQA, Social IQa, COPA, TruthfulQA, TREC): 0.800 against v10's 0.707 and the 35B's 0.862.
Against the 35B: −2.1 in-task (−2.9 to −1.5) and −2.3 held-out points (−3.5 to −1.2), lower on 75 of 95 tasks; the largest gaps
are HelpSteer3 preference −15.6, MedQA −14.5, MedMCQA −10.5, StrategyQA −10.3, TruthfulQA −9.7.

<details>
<summary><b>Per-task accuracy / ECE on the 28 held-out datasets, decider-2b v10, this model, decider-35b-a3b</b></summary>

| task | decider-2b v10 | decider-4b | decider-35b-a3b |
|---|---|---|---|
| abstain_probe | 0.606 / 0.134 | 0.559 / 0.205 | 0.622 / 0.085 |
| ade | 0.817 / 0.038 | 0.814 / 0.070 | 0.837 / 0.035 |
| arena_pref | 0.483 / 0.189 | 0.495 / 0.180 | 0.521 / 0.121 |
| bbc_news | 0.927 / 0.013 | 0.953 / 0.028 | 0.944 / 0.027 |
| cb | 0.857 / 0.093 | 0.929 / 0.089 | 0.893 / 0.084 |
| cr_reviews | 0.903 / 0.031 | 0.920 / 0.023 | 0.914 / 0.033 |
| dbpedia_l2 | 0.950 / 0.018 | 0.956 / 0.011 | 0.961 / 0.010 |
| dbpedia_l3 | 0.987 / 0.005 | 0.993 / 0.005 | 0.992 / 0.004 |
| dolly_category | 0.299 / 0.203 | 0.366 / 0.174 | 0.354 / 0.098 |
| fin_phrasebank | 0.694 / 0.042 | 0.721 / 0.037 | 0.759 / 0.110 |
| fin_sentiment | 0.793 / 0.058 | 0.850 / 0.092 | 0.839 / 0.136 |
| hermes_tools | 0.723 / 0.208 | 0.752 / 0.164 | 0.799 / 0.085 |
| hwu64 | 0.961 / 0.030 | 0.971 / 0.020 | 0.975 / 0.022 |
| massive_scenario | 0.756 / 0.041 | 0.791 / 0.067 | 0.799 / 0.027 |
| offtopic_probe | 0.841 / 0.027 | 0.852 / 0.024 | 0.870 / 0.038 |
| paws | 0.724 / 0.145 | 0.729 / 0.163 | 0.729 / 0.169 |
| pubmedqa | 0.756 / 0.085 | 0.820 / 0.048 | 0.820 / 0.078 |
| quality | 0.494 / 0.233 | 0.568 / 0.160 | 0.632 / 0.096 |
| quality_full | 0.508 / 0.198 | 0.538 / 0.140 | 0.565 / 0.112 |
| reward_bench | 0.819 / 0.045 | 0.868 / 0.017 | 0.919 / 0.024 |
| sciq | 0.982 / 0.024 | 0.993 / 0.011 | 0.993 / 0.011 |
| social_iqa | 0.708 / 0.077 | 0.779 / 0.031 | 0.823 / 0.025 |
| strategyqa | 0.552 / 0.138 | 0.636 / 0.072 | 0.739 / 0.036 |
| student_questions | 0.925 / 0.045 | 0.943 / 0.024 | 0.954 / 0.090 |
| trec | 0.784 / 0.066 | 0.802 / 0.025 | 0.832 / 0.160 |
| truthfulqa | 0.537 / 0.090 | 0.657 / 0.044 | 0.754 / 0.068 |
| tweet_irony | 0.795 / 0.052 | 0.821 / 0.024 | 0.861 / 0.129 |
| xstory_cloze | 0.962 / 0.017 | 0.973 / 0.028 | 0.995 / 0.016 |

</details>

**On the same rows as decider-2b v10 and decider-35b-a3b.** Every row below is scored by all three models on identical inputs
and seeds. Intervals are 95% paired bootstrap intervals (rows for the fixtures, boards for the games, task-seed pairs for the
browser).

| | decider-2b v10 | **decider-4b** | decider-35b-a3b | 4B minus 2B | 4B minus 35B |
|---|---|---|---|---|---|
| 847 in-task validation rows, accuracy / NLL | 83.2% / 0.444 | 86.0% / 0.417 | 90.0% / 0.329 | +2.7 (+0.7 to +4.7) | −4.0 (−6.1 to −2.1) |
| OpenJev, 5,252 rows, accuracy / NLL | 63.3% / 0.916 | 63.8% / 0.894 | 68.3% / 0.752 | +0.6 (−0.6 to +1.8) | −4.5 (−5.7 to −3.3) |
| Mind2Web element and action choice, 1,770 rows | 82.7% / 0.543 | 88.3% / 0.367 | 89.6% / 0.316 | +5.7 (+4.0 to +7.3) | −1.2 (−2.8 to +0.2) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 80.4% / 0.585 | 80.4% / 0.609 | 86.3% / 0.342 | 0.0 (−9.8 to +9.8) | −5.9 (−12.7 to +1.0) |
| Bespoke's public suite, 13 subsets, macro / micro | 0.704 / 0.711 | 0.757 / 0.765 | 0.774 / 0.787 | | |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.889 / 0.459 | 1.000 / 0.958 / 0.541 | 1.000 / 0.972 / 0.676 | | |
| live MiniWoB++ click tasks, 22 tasks x 8 seeds, greedy play | 90.9% | 91.5% | 97.2% | +0.6 (−4.5 to +5.7) | −5.7 (−9.7 to −2.3) |
| the same, 6 tasks v10 never used for reward, greedy | 91.7% | 75.0% | 97.9% | −16.7 (−27.1 to −6.2) | −22.9 (−35.4 to −12.5) |
| live MiniWoB++ click tasks, sampled play | 93.2% | 90.9% | 86.4% | −2.3 (−8.0 to +2.8) | +4.5 (−0.6 to +10.2) |
| the same, 6 held-out tasks, sampled | 91.7% | 79.2% | 79.2% | −12.5 (−25.0 to 0.0) | 0.0 (−12.5 to +12.5) |
| zero-shot games, win rate, sampled play (234 boards) | 23.7% | 27.7% | 24.1% | +4.0 (+1.1 to +7.1) | +3.5 (+1.2 to +5.9) |
| zero-shot games, greedy play | 26.5% | 28.6% | 37.2% | +2.1 (−3.0 to +7.3) | −8.5 (−13.7 to −3.0) |
| bag-draw games alone, sampled | 41.4% | 56.2% | 41.8% | +14.8 (+9.4 to +20.3) | +14.5 (+8.6 to +20.3) |

On the two fixtures whose rows are in neither model's training data (TypeSafe, OpenJev) the 4B is level with the 2B: the gain over
the 2B is on the in-task rows, on Mind2Web and on knowledge questions, not on these judgment rows. The browser rows show the
missing RL stage: greedy play matches the 2B over all tasks and is 17 points below it on the six tasks that v10's RL never
rewarded, where the 4B's weakest tasks are focus-text-2 (0.375), click-tab-2-hard (0.5) and click-collapsible-2 (0.625). On the
bag-draw games (choose the bag with the highest expected value from a list of known compositions) the 4B wins 15 points more often
than either other model in sampled play; on tic-tac-toe, the slippery grid and minesweeper every model is near the random floor.

**Ten text games, zero-shot** (`decider.games.play`, five episodes per game, greedy, eager; the 2B was trained on the first
four, the 4B and the 35B on none): Pong −21 (teacher 8, 2B 8, 35B −21), Breakout 18 (22, 22, 6), CliffWalking −13 (teacher −13,
2B −13, 35B −1,248), MiniGrid-Empty 0 (teacher 0.96, both others 0), Freeway 1 (teacher 5, 2B 0, 35B 1), FrozenLake 0 (teacher 1,
others 0), Blackjack −0.6 (teacher −0.6, 2B −1, 35B −0.6), MiniGrid-LavaGap 0, MiniGrid-DoorKey 0 (teacher 0), BabyAI-GoTo 0.54
(teacher 0.34, 2B 0, 35B 0.19). Without any game rows the 4B reaches the teacher on CliffWalking and Blackjack, is close on
Breakout and above the teacher on BabyAI-GoTo; it does not learn Pong from the text state.

**JevBench public items** (231 items of [Benchmark Heaven](https://benchmarkheaven.com/jev-models); argmax over the exact
label set with the request the harness's TypeSafe adapter builds). Jev 1.13.0 is at 1.000 / 0.986 / 0.730, SemIf (also a
Qwen3.5-4B) at 1.000 / 0.986 / 0.613, decider-35b-a3b at 1.000 / 0.972 / 0.676 on the same items. Standard-tier misses are on
adequacy (0.83) and policy (0.92) items. Hard-tier families: trap 1.00, hard routing 1.00, multi-hop 0.78, adversarial 0.67,
probability 0.60, long policy 0.47, judge-hard 0.47, trade-off 0.33, ambiguous 0.29, temporal-numeric 0.13. Top-label ECE is
0.001 / 0.038 / 0.287 by tier: the model is overconfident on the hard tier (mean confidence 0.83 at accuracy 0.54), more so than
the 35B (0.151) and about as much as the 2B (0.304).

**Bespoke's public suite** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through `system_one` as
shipped). Nimble-9B and Jev 1.13.0 numbers are copied from Bespoke's report.

| subset (type) | decider-2b v10 | decider-4b | decider-35b-a3b | Nimble-9B | Jev 1.13.0 |
|---|---|---|---|---|---|
| vitaminc-dev (choice) | 0.639 | 0.756 | 0.795 | 0.766 | 0.801 |
| massive-en-US (choice; trained) | 0.823 | 0.860 | 0.880 | 0.869 | 0.874 |
| massive-de-DE (choice, German) | 0.797 | 0.843 | 0.869 | 0.834 | 0.869 |
| boolq (noul; trained) | 0.803 | 0.873 | 0.887 | 0.860 | 0.897 |
| squad2 (noul) | 0.776 | 0.706 | 0.749 | 0.806 | 0.829 |
| paws (noul; trained) | 0.720 | 0.716 | 0.768 | 0.828 | 0.892 |
| multinli (choice; trained) | 0.856 | 0.933 | 0.910 | 0.853 | 0.829 |
| civil_comments (noul; trained) | 0.840 | 0.880 | 0.907 | 0.703 | 0.810 |
| aegis2 (noul) | 0.728 | 0.820 | 0.808 | 0.812 | 0.804 |
| helpsteer2 (score; trained) | 0.426 | 0.466 | 0.478 | 0.390 | 0.341 |
| summeval-relevance (score) | 0.354 | 0.425 | 0.483 | 0.492 | 0.350 |
| summeval-consistency (score) | 0.660 | 0.833 | 0.757 | 0.757 | 0.812 |
| pubmedqa (choice; trained) | 0.724 | 0.728 | 0.768 | 0.756 | 0.772 |
| **macro / micro** | 0.704 / 0.711 | **0.757 / 0.765** | 0.774 / 0.787 | 0.748 / 0.759 | 0.760 / 0.773 |

On the six subsets whose training split is not in the mixture the macro accuracy is 0.731 (2B 0.659, 35B 0.744). The 4B is above
Nimble-9B on the average and 0.3 points under Jev; it is behind Jev where a claim has to be checked against evidence that nearly
matches it (PAWS, VitaminC) and on SQuAD2 answerability, where it is also 7 points under the 2B.

**Behaviour probes** (teacher-labelled, same probes as the other releases): generic-versus-specific bucket choice 1.00 / 1.00,
catch-all when nothing fits 0.90 (35B 0.95), abstention battery 7 of 8; model-router tier 0.968 and needs-live-data 0.871;
command-risk classification 0.889 with no destructive command called safe (35B 0.933), touches-outside-project 0.956; browser-agent
element and action choice 0.875 / 0.875 (35B 0.938). Scoring a Score level alone against scoring it with its neighbours changes
accuracy by at most 2.3 points on five rating datasets, and the per-level fits sum to between 0.89 and 1.04.

## Calibration

The stored temperature 1.05 was fitted on the in-task half of the regression set. On that set and on the held-out half the
model is as well calibrated as the 35B (ECE 0.027 / 0.071 against 0.026 / 0.069), with per-task exceptions (abstention probe
0.205, Arena preferences 0.180, Dolly categories 0.174, Hermes tools 0.164, PAWS 0.163, QuALITY 0.160, hate-speech tweets 0.210
in-task). Outside the regression set it is overconfident:

| set | measure | decider-2b v10 | decider-4b | decider-35b-a3b |
|---|---|---|---|---|
| JevBench hard tier, 111 items | top-label ECE | 0.304 | 0.287 | 0.151 |
| TypeSafe, 102 rows | 10-bin ECE / mean total variation to the frontier reference | 0.091 / 0.242 | 0.122 / 0.232 | 0.065 / 0.165 |
| OpenJev, 5,252 rows | 10-bin ECE | 0.150 | 0.159 | 0.049 |
| Decision Index 4,000-request sample, 33 benchmarks | benchmark-weighted ECE (the site's definition) | 0.093 | 0.086 | 0.027 |
| the same | share of all answers given at 95% or more confidence and wrong | 0.010 | 0.022 | 0.005 |
| the same | mean confidence against accuracy | 0.658 / 0.566 | 0.722 / 0.637 | 0.684 / 0.690 |
| the same | sample index (about 4 points under a full run) | 42.3 | 48.5 | 50.4 |

The Decision Index row is a readout of the 4,000-request sample (about 88 cases per benchmark) through the public server with
this repository's `decider_config.json`, computed with the site's definition (benchmark-weighted pooled bins, ten bins; the same
computation gives 0.027 on the 35B against the site's published 0.031). Without the stored temperature (T=1) the same sample reads
0.095; the temperature moves the ECE by 0.009 and the index not at all. The 4B's confidence exceeds its accuracy by 0.085 on that
sample, about the same gap as the 2B's 0.092, where the 35B has none. Nothing was fitted on index rows. The gap is per-benchmark
heterogeneity, not a global scale: a single temperature that removes it on the knowledge benchmarks would make the wide label sets
underconfident. If you route on confidence, calibrate on your own labels.

## Speed

24.7 ms median per decision (10th to 90th percentile 24.6 to 43.5 ms) on one NVIDIA B300, bf16, batch of one, eager PyTorch
without CUDA graphs or `torch.compile`, over 200 game-state decisions of 156 tokens median, timed around the forward pass with
`torch.cuda.synchronize()`; decider-2b measures 17.9 ms and decider-35b-a3b 41.4 ms on the same 200 decisions with the same
method. The CUDA-graph and FP8 paths of the helper, which take decider-2b from 49 ms eager to 4 ms, were not measured on this
model.

## Limitations

* No reinforcement-learning stage: stated beliefs about action outcomes were not trained against exact laws, and on live
  browser tasks the model is 17 points below decider-2b v10 on the six held-out tasks (75% against 92% greedy) and level with it
  on the sixteen tasks v10 was rewarded on.
* Overconfident outside the regression set: JevBench hard-tier ECE 0.29, OpenJev ECE 0.16, Decision Index sample ECE 0.086
  where the 35B reads 0.027. On the Decision Index sample 2.2% of all answers are given at 95% or more confidence and are wrong,
  twice the 2B's share and four times the 35B's. Calibration is measured on public datasets, not on your traffic.
* Level with the 2B on the two clean judgment fixtures (TypeSafe 80.4% both, OpenJev +0.6 points) and 7 points under it on SQuAD2
  answerability; the gain over the 2B is on knowledge questions, Mind2Web and in-task rows.
* Below the 35B on 75 of 95 regression tasks and on every fixture, by 1 to 6 points; the largest gaps are on MedQA, MedMCQA,
  StrategyQA, TruthfulQA and preference judgments.
* The abstention probe is 4.6 points under the 2B (0.559 against 0.606) and its least calibrated held-out task (ECE 0.205).
* Optimizer: this model was trained with `torch.optim.AdamW` on the bf16 parameters and no FP32 master copy, so the finding
  reported for decider-35b-a3b (FP32 master weights move a model further from its base and cost knowledge-task accuracy) does
  not apply to it; the same schedule with a master copy was measured on the 4B and was 3.3 held-out points worse.
* Mixture v2's 26 additional public datasets and ten programmatic families are described above but their builders are not yet in
  the public package; `scripts/train.sh full` reproduces the 60% of the data that is the public mixture. On the held-out variants
  of the programmatic families the model is at 0.635 mean accuracy (logs 0.28, plans 0.44, probability 0.43, code 0.88, policy
  0.99), so the generated families transfer unevenly to their held-out variants.
* English is the main language; the multilingual rows (XNLI, PAWS-X, MASSIVE, Belebele, XCOPA) are a small share of the data
  and were not measured beyond the mixture-v2 evaluation set.
* Everything else in the decider-2b card's limitations (packed questions see each other, long JSON arrays by position, full
  label sets against sampled options, abstention wording, rules in the question) applies; those shapes were not re-measured at
  this size.

## Changelog

| version | what changed |
|---|---|
| **v1** (2026-09-22, these weights) | first release: one pass over mixture v2 on Qwen3.5-4B-Base with AdamW on bf16 parameters, no RL stage |

The GitHub repository's [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) lists every
decider release.

## Reproduction

Code, data registry, training and evaluation scripts and the per-version history: https://github.com/Mapika/decider
(`docs/HISTORY.md`, section "decider-4b"). Trained with the data-parallel trainer of the architecture A/B study
(`arch_ab/train_dp_optvar.py` in the research repository, optimizer variant `bf16`), evaluated with the public `decider.evaluate`
and the head-to-head tools, staged by hardlinks with `decider2/release_4b/stage.py` and uploaded with `scripts/upload_hf.py`.
`eval_results.json` in this repository has the per-task regression metrics at T 1.05, every other set above and the paired
comparisons.

**Independence.** This is an independent project. It is not affiliated with or endorsed by TypeSafe AI. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev); nothing was distilled from Jev. The training data is public
datasets, programmatically generated rows with verifiable gold, and data labelled by a local Qwen3.5-27B teacher. License:
Apache 2.0.
