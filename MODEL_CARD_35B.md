---
license: apache-2.0
base_model: Qwen/Qwen3.5-35B-A3B-Base
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass, mixture-of-experts]
---

# decider-35b-a3b: typed decisions with calibrated probabilities in one forward pass, 35B mixture of experts

A language model that does not generate text. It reads a state and one or more typed questions, each with an explicit option
list, and returns a probability distribution over the options for every question from one forward pass. There is no decoding,
no parsing and no output outside the options you defined. It is called from software, not chatted with. It is an open
reproduction of the "System One" model class (TypeSafe AI's Jev).

Base model: [Qwen/Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base): 34.7B parameters, of which 3B
are active per token (256 routed experts, 8 active, plus a shared expert; 40 layers, 10 with full attention and 30 with
gated delta-net linear attention). The supervised recipe of [decider-2b](https://huggingface.co/Mapika/decider-2b) (one epoch of
cross-entropy on the slot readout over the public decision mixture) was applied to it with the routed experts frozen and the
Muon optimizer on the block matrices. **This repository holds v1**, the bf16 weights (65 GB). An NVFP4 checkpoint of the
same weights for vLLM and TensorRT-LLM is at
[Mapika/decider-35b-a3b-nvfp4](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4); the smaller models are listed under
The decider family. `decider/` in this repository is the inference subset of the GitHub package.

Against decider-2b v10 on the same rows: accuracy is higher on 93 of the 95 regression tasks (in-task 0.855 against 0.805,
held-out 0.810 against 0.755), +6.7 points on the 847 validation rows, +5.0 on OpenJev, +6.9 on Mind2Web, +5.9 on the
TypeSafe workflow rows, JevBench hard tier 0.676 against 0.459, Bespoke's public suite 0.774 against 0.704 macro. Negative
log-likelihood drops by 0.12 to 0.24 nats on every fixture. The model was not RL-trained: on live browser tasks its greedy
play beats v10 (97.2% against 90.9%) and its sampled play is behind (86.4% against 93.2%). Details under Evaluation.

**Contents:** [The decider family](#the-decider-family) · [Usage](#usage) · [How it works](#how-it-works) · [Training](#training) · [Evaluation](#evaluation) · [Speed](#speed) · [Limitations](#limitations) · [Changelog](#changelog) · [Reproduction](#reproduction)

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
d = Decider("Mapika/decider-35b-a3b", use_graphs=False)
d.decide("My card was charged twice for the same purchase.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}])
# [{'choice': 'billing', 'confidence': 0.99, 'probs': {...}}, {'choice': 'yes', 'confidence': 0.98, 'probs': {...}}]
```

The API is the same as decider-2b's: `decide_batch` scores many states with many questions in one call, `abstain_below=t`
returns `None` under a confidence threshold, a question can have 2 to 255 options, and `system_one` / `decider.serve` accept
TypeSafe's `POST /v1/systemone` request shape (the official `typesafe-sdk` works with `TYPESAFE_BASE_URL` pointing at the
server). Every question and every Score level is scored in its own row. The state may be a string, object or array of up to
32k tokens. See the decider-2b card for the full description of the request shape, field types and the schema cache.

Requirements: one GPU with at least 80 GB of memory (the weights take 65 GB in bf16), `torch>=2.14`, `transformers>=5.17`
and `flash-linear-attention`. `config.json` sets `experts_implementation: grouped_mm`, which runs the 256 experts of a layer
as one grouped matrix multiplication; the eager expert loop that `transformers` falls back to on older versions is about
13x slower. `use_graphs=False` is required: the CUDA-graph engine and the FP8 path of the helper package were built for the
dense models and are untested with this architecture. Loading takes about 25 seconds from local disk.

Apple Silicon (reported by @nassersala in issue #6): on an M5 Max with 128 GB of unified memory, macOS 26.5, torch 2.14 and
transformers 5.17, the model loads in bf16 in about 60 s through `Decider(path, device="mps", dtype=torch.bfloat16,
use_graphs=False)` and reproduces this card's JevBench public-item counts exactly (48/48, 70/72, 75/111).
`flash-linear-attention` does not install there; install the package with `--no-deps` after `torch`, `transformers>=5.17`,
`numpy<2` and `huggingface_hub`. From decider-ai 1.1.4 the library's MPS patch replaces two slow MPS operations in the
MoE reference code (`decider/mps_moe.py`), which takes a decision from 2.8-4 s to 0.23-0.5 s for typical inputs.

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
probs = torch.softmax(logits[letters].float() / 1.08, -1)      # 1.08 is the stored temperature
```

## How it works

The prompt is `Context: ...` followed by, for each question, the question text, the lettered options `(A) ... (B) ...` and an
answer slot `Answer k: (`. The hidden state at each slot is projected with the option-letter rows of the LM head and softmaxed
over the valid letters, divided by the temperature in `decider_config.json`. Letters are never generated, so all slots are read
from one pass. Large label sets were sub-sampled to at most 10 options per training example (gold always kept, order shuffled),
so the model conditions on the supplied candidates rather than on a fixed head.

## Training

One epoch of the public supervised mixture of the GitHub repository (`scripts/train.sh full`: about 95 public decision
datasets, agent trajectories, Mind2Web element choice, game states, teacher-written custom questions, Jev's input shapes,
two prompt layouts, isolated Score levels, 10% abstention rows). 1,543,567 items, 463M tokens, pre-tokenized once and read in
the same order by every rank.

| | |
|---|---|
| trainable parameters | 2.45B: attention, delta-net, shared experts, routers, norms, embeddings, LM head. The 256 routed experts of every layer (32.2B parameters) stay at the base weights. |
| optimizer | Muon on the 250 block matrices (1.41B parameters): momentum 0.95, Nesterov, 5 Newton-Schulz steps, update scaled by 0.2 sqrt(max(rows, cols)) so the AdamW learning-rate schedule applies. AdamW (betas 0.9 / 0.95) on embeddings, head, norms, routers, shared-expert gates, convolutions and 1-D parameters (1.04B). FP32 master weights, bf16 forward. |
| schedule | peak learning rate 1e-5, 150 warm-up steps, cosine to zero, 16,287 steps of 32,768 tokens, gradient clip 1.0, no weight decay |
| hardware | 4 NVIDIA B300, data parallel, 8,192-token micro-batches per GPU, gradient checkpointing, grouped-GEMM experts; 394 minutes at 22,000 to 25,000 tokens per second, 100 GB peak per GPU |
| training cross-entropy | 0.93 over the first 200 steps, 0.50 at 25%, 0.46 at 50%, 0.43 over the last 300 steps |

Muon was chosen over AdamW on a same-data comparison stopped at 11% of the epoch: at every logged step both optimizers had
seen identical examples, and Muon's cross-entropy was lower in 75 of 94 logged windows, 0.537 against 0.605 at step 1,880
(mean over steps 1,500 to 1,880: 0.557 against 0.614). No AdamW model was trained to the end, so there is no AdamW row in the
evaluation tables. The temperature 1.08 was fitted on the in-task half of the regression set. No reinforcement-learning stage
was run on this model; the RL recipe of decider-2b v10 is documented in `docs/RL.md` of the GitHub repository.

## Evaluation

**Public regression set**, rebuilt on this machine (95 tasks: 67 in-task, 28 held-out; large label sets sub-sampled to 10
options; one temperature per model fitted on in-task data). The decider-2b rows are the same set, same rows. ECE is the
expected calibration error with 15 bins.

| model | in-task acc / NLL / ECE (67 tasks) | held-out acc / NLL / ECE (28 tasks) |
|---|---|---|
| Qwen3.5-35B-A3B-Base, zero-shot, T=1.27 | 0.732 / 0.693 / 0.096 | 0.749 / 0.650 / 0.082 |
| after 25% of the epoch, T=0.96 | 0.839 / 0.399 / 0.032 | 0.803 / 0.522 / 0.073 |
| after 50%, T=1.08 | 0.850 / 0.370 / 0.028 | 0.813 / 0.488 / 0.064 |
| **decider-35b-a3b v1 (this repository), T=1.08** | **0.855 / 0.357 / 0.026** | **0.810 / 0.497 / 0.069** |
| decider-2b v10, T=1.30 | 0.805 / 0.474 / 0.037 | 0.755 / 0.622 / 0.084 |
| decider-2b v8, T=1.30 | 0.806 / 0.473 / 0.038 | 0.757 / 0.622 / 0.083 |

Half the epoch reaches 99% of the final in-task accuracy; held-out accuracy is flat from 50% to 100%. Accuracy is above v10 on
93 of the 95 tasks and 0.6 points below on two (counterfactual detection, offensive-tweet detection). The largest gains are
on knowledge and reasoning tasks: MedQA +31 points, MedMCQA +24, TruthfulQA +22, Winogrande +20, MMLU +19, StrategyQA +19.

<details>
<summary><b>Per-task accuracy / ECE on the 28 held-out datasets, decider-2b v10 against this model</b></summary>

Per-task accuracy / ECE on the held-out datasets, v10 against this model:

| task | decider-2b v10 | decider-35b-a3b |
|---|---|---|
| abstain_probe | 0.606 / 0.134 | 0.622 / 0.085 |
| ade | 0.817 / 0.038 | 0.837 / 0.035 |
| arena_pref | 0.483 / 0.189 | 0.521 / 0.121 |
| bbc_news | 0.927 / 0.013 | 0.944 / 0.027 |
| cb | 0.857 / 0.093 | 0.893 / 0.084 |
| cr_reviews | 0.903 / 0.031 | 0.914 / 0.032 |
| dbpedia_l2 | 0.950 / 0.018 | 0.961 / 0.010 |
| dbpedia_l3 | 0.987 / 0.005 | 0.992 / 0.004 |
| dolly_category | 0.299 / 0.203 | 0.354 / 0.098 |
| fin_phrasebank | 0.694 / 0.042 | 0.759 / 0.110 |
| fin_sentiment | 0.793 / 0.058 | 0.839 / 0.136 |
| hermes_tools | 0.723 / 0.208 | 0.799 / 0.085 |
| hwu64 | 0.961 / 0.030 | 0.975 / 0.022 |
| massive_scenario | 0.756 / 0.041 | 0.799 / 0.028 |
| offtopic_probe | 0.841 / 0.027 | 0.870 / 0.038 |
| paws | 0.724 / 0.145 | 0.729 / 0.168 |
| pubmedqa | 0.756 / 0.085 | 0.820 / 0.078 |
| quality | 0.494 / 0.233 | 0.632 / 0.096 |
| quality_full | 0.508 / 0.198 | 0.565 / 0.112 |
| reward_bench | 0.819 / 0.045 | 0.919 / 0.024 |
| sciq | 0.982 / 0.024 | 0.993 / 0.011 |
| social_iqa | 0.708 / 0.077 | 0.823 / 0.025 |
| strategyqa | 0.552 / 0.138 | 0.739 / 0.036 |
| student_questions | 0.925 / 0.045 | 0.954 / 0.090 |
| trec | 0.784 / 0.066 | 0.832 / 0.160 |
| truthfulqa | 0.537 / 0.090 | 0.754 / 0.068 |
| tweet_irony | 0.795 / 0.052 | 0.861 / 0.129 |
| xstory_cloze | 0.962 / 0.017 | 0.995 / 0.016 |

</details>

**On the same rows as decider-2b.** Every row below is scored by both models on identical inputs and seeds. Intervals are
95% paired bootstrap intervals.

| | decider-2b v10 | decider-35b-a3b | difference |
|---|---|---|---|
| 847 in-task validation rows, accuracy / NLL | 83.2% / 0.444 | 90.0% / 0.329 | +6.7 (+4.5 to +9.0) |
| OpenJev, 5,252 rows, accuracy / NLL | 63.3% / 0.916 | 68.3% / 0.752 | +5.0 (+3.8 to +6.2) |
| Mind2Web element and action choice, 1,770 rows | 82.7% / 0.543 | 89.6% / 0.316 | +6.9 (+5.1 to +8.7) |
| TypeSafe workflow decisions, 102 rows, accuracy / NLL | 80.4% / 0.585 | 86.3% / 0.342 | +5.9 (−2.0 to +13.7) |
| Bespoke's public suite, 13 subsets, macro / micro | 0.704 / 0.711 | 0.774 / 0.787 | |
| JevBench public items, easy / standard / hard accuracy | 1.000 / 0.889 / 0.459 | 1.000 / 0.972 / 0.676 | |
| live MiniWoB++ click tasks, 22 tasks x 8 seeds, greedy play | 90.9% | 97.2% | +6.2 (+1.7 to +10.8) |
| the same, 6 tasks v10 never used for reward, greedy | 91.7% | 97.9% | +6.2 (0.0 to +14.6) |
| live MiniWoB++ click tasks, sampled play | 93.2% | 86.4% | −6.8 (−12.5 to −1.7) |
| the same, 6 held-out tasks, sampled | 91.7% | 79.2% | −12.5 (−25.0 to −2.1) |
| zero-shot games, win rate, greedy play (234 boards) | 26.5% | 37.2% | +10.7 (+5.6 to +15.8) |
| zero-shot games, sampled play | 23.7% | 24.1% | +0.4 (−2.4 to +3.3) |

The browser rows show what the RL stage of v10 does and this model lacks: v10's sampled play matches its greedy play because
RL sharpened the served distribution on those tasks; this model's argmax is right more often, but its distribution still puts
mass on wrong elements (its sampled play is +3.4 points against v8, which had no RL either). Among the games, the largest
greedy gains are on the slippery grid (+17 points) and tic-tac-toe (+14); minesweeper stays near zero for every model.

**JevBench public items** (231 items of [Benchmark Heaven](https://benchmarkheaven.com/jev-models); argmax over the exact
label set with the request the harness's TypeSafe adapter builds). Jev 1.13.0 is at 1.000 / 0.986 / 0.730, djev at 1.000 /
0.986 / 0.676, SemIf 4B at 1.000 / 0.986 / 0.613 on the same items, from their published per-item outcomes. This model's
hard-tier misses are on temporal-numeric items (0.33), long policies (0.63) and judge-hard items (0.65); adversarial, trap
and hard routing items are all correct. Top-label ECE is 0.001 / 0.059 / 0.151 by tier: the model is overconfident on the
hard tier.

**Bespoke's public suite** (13 human-labelled subsets, 3,880 records in Jev's wire format, answered through `system_one` as
shipped). Nimble-9B and Jev 1.13.0 numbers are copied from Bespoke's report.

| subset (type) | decider-2b v10 | decider-35b-a3b | Nimble-9B | Jev 1.13.0 |
|---|---|---|---|---|
| vitaminc-dev (choice) | 0.639 | 0.795 | 0.766 | 0.801 |
| massive-en-US (choice; trained) | 0.823 | 0.880 | 0.869 | 0.874 |
| massive-de-DE (choice, German) | 0.797 | 0.869 | 0.834 | 0.869 |
| boolq (noul; trained) | 0.803 | 0.887 | 0.860 | 0.897 |
| squad2 (noul) | 0.776 | 0.749 | 0.806 | 0.829 |
| paws (noul; trained) | 0.720 | 0.768 | 0.828 | 0.892 |
| multinli (choice; trained) | 0.856 | 0.910 | 0.853 | 0.829 |
| civil_comments (noul; trained) | 0.840 | 0.907 | 0.703 | 0.810 |
| aegis2 (noul) | 0.728 | 0.808 | 0.812 | 0.804 |
| helpsteer2 (score; trained) | 0.426 | 0.478 | 0.390 | 0.341 |
| summeval-relevance (score) | 0.354 | 0.483 | 0.492 | 0.350 |
| summeval-consistency (score) | 0.660 | 0.757 | 0.757 | 0.812 |
| pubmedqa (choice; trained) | 0.724 | 0.768 | 0.756 | 0.772 |
| **macro / micro** | 0.704 / 0.711 | **0.774 / 0.787** | 0.748 / 0.759 | 0.760 / 0.773 |

On the six subsets whose training split is not in the mixture the macro accuracy is 0.744. The model is behind Jev where a
claim has to be checked against evidence that nearly matches it (PAWS, SummEval consistency) and on SQuAD2 answerability.

**Behaviour probes** (teacher-labelled, same probes as the 2B releases): generic-versus-specific bucket choice 1.00 / 1.00,
catch-all when nothing fits 0.95; command-risk classification 0.933 with no destructive command called safe; browser-agent
element and action choice 0.938 / 0.938. Scoring a Score level alone against scoring it with its neighbours changes accuracy
by at most 2 points on five rating datasets, and the per-level fits sum to between 0.92 and 1.07.

## Speed

One NVIDIA B300, bf16, eager PyTorch (`use_graphs=False`), grouped-GEMM experts. A 92-token support ticket with three typed
questions, and a 5-token chat message with one question:

| setting | latency | throughput |
|---|---|---|
| single ticket request, 3 questions | 47 ms | |
| batch of 64 tickets, 3 questions each | 368 ms | 174 states/s, 522 decisions/s |
| batch of 64 short messages, 1 question | 111 ms | 575 decisions/s |

decider-2b serves the same tickets at 4 ms with CUDA graphs and about 1,400 to 2,100 decisions/s; this model is for
workloads where the accuracy gain is worth 3 to 4 times the cost per decision, and for the NVFP4 build on Blackwell (see the
`-nvfp4` repository).

## Limitations

* No reinforcement-learning stage: stated beliefs about action outcomes were not trained against exact laws, and the served
  distribution on live browser tasks is less sharp than decider-2b v10's (sampled play 86% against 93%).
* 65 GB of bf16 weights; one 80 GB GPU is the minimum on CUDA (a Mac with 128 GB of unified memory also runs it, see
  Requirements), and the CUDA-graph and FP8 paths of the helper are untested here.
* Overconfident on the hardest external items (JevBench hard-tier ECE 0.15) and on some held-out classification sets
  (TREC 0.16, financial sentiment 0.14) although the aggregate ECE is 0.03 / 0.07.
* English only. Calibration is measured on public datasets and teacher-labelled probes, not on your traffic. Check it on your
  own labels before using confidence for routing.
* The routed experts are the base model's: the fine-tuning changed 2.45B of the 34.7B parameters.
* Everything else in the decider-2b card's limitations (packed questions see each other, long JSON arrays by position, full
  label sets against sampled options, abstention wording) applies; those shapes were not re-measured at this size.

## Changelog

| version | what changed |
|---|---|
| **v1** (2026-09-20, these weights) | first release: one public-mixture epoch on Qwen3.5-35B-A3B-Base with the routed experts frozen and Muon on the block matrices; NVFP4 build in the sibling repository |

The GitHub repository's [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) lists every
decider release.

## Reproduction

Code, data registry, training and evaluation scripts and the per-version history: https://github.com/Mapika/decider
(`docs/HISTORY.md`, section "decider-35b-a3b"). The training code for the frozen-expert Muon run is in the repository's
history document; the merged checkpoint is this repository. Staged with `scripts/stage_release.py` and uploaded with
`scripts/upload_hf.py`.
