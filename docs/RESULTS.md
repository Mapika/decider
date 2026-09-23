# Results

Every measurement moved out of the README, with the conditions it was taken under.

**Where the numbers come from.** Unless a version is named, the numbers in this file were measured on the v8 weights on one
GH200; v9 is v8 plus the terse-bucket and command data, with the same numbers on the 94 tasks. Rows marked "rebuilt set" and
everything about decider-35b-a3b and decider-4b were measured on a B300. "Held-out" means no example of that dataset was trained on. The
external-leaderboard numbers were read from the leaderboards on the dates given and were not measured here. The JevBench
public-item and Bespoke-suite rows were run in this repository (`decider/bench/`). Dates: decider-2b v10 measurements
2026-09-19, decider-35b-a3b measurements 2026-09-20, decider-4b measurements 2026-09-22, JevBench and Decision Index standings read 2026-09-21 and 2026-09-22,
Apple Silicon MPS measurements 2026-09-21 (`docs/benchmarks/`). [docs/HISTORY.md](HISTORY.md) has the per-stage measurements
and [docs/RL.md](RL.md) the RL stage.

**Contents:** [External leaderboards](#external-leaderboards) · [decider-35b-a3b against decider-2b v10](#decider-35b-a3b-against-decider-2b-v10)
· [decider-4b against decider-2b v10 and decider-35b-a3b](#decider-4b-against-decider-2b-v10-and-decider-35b-a3b)
· [The 94 public tasks](#the-94-public-tasks) · [JevBench public items](#jevbench-public-items) ·
[Bespoke's public suite](#bespokes-public-suite) · [Speed](#speed) · [Input shapes](#input-shapes) ·
[Custom questions and catch-all options](#custom-questions-and-catch-all-options) ·
[Terse buckets and applications](#terse-buckets-and-applications-v9) · [Form filling](#form-filling-against-a-specialist) ·
[Isolated Score levels](#isolated-score-levels) · [Independence](#independence) ·
[Text games](#text-games-supervised-stages) · [Browser tasks](#browser-tasks)

## External leaderboards

Read on the dates given; not measured here.

**JevBench**, read 2026-09-21 ([Benchmark Heaven](https://benchmarkheaven.com/jev-models), harness at
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)). 36 entries. The total score combines four axes;
speed and cost are measured from the operator's server.

| system | score | intelligence | calibration | speed | cost |
|---|---|---|---|---|---|
| Jev 1.13.0 (TypeSafe AI) | 75.4 | — | — | — | — |
| SemIf | 74.7 | — | — | — | — |
| decider-35b-a3b (#10 of 36) | 68.9 | 86.3 | 71.5 | 80.8 | 45.3 |
| decider-2b (#21 of 36) | 64.6 | 73.8 | 46.6 | 83.2 | 61.0 |

Jev 1.13.0 and SemIf are the two leading entries; their per-axis scores are not reproduced here.

**Decision Index**, edition v0.1 dated 2026-09-22
([leaderboard](https://multimodalart-jev-decision-index.static.hf.space), kit at
[apolinario/decision-index](https://github.com/apolinario/decision-index)). 32 entries, 132,422 requests, 37 benchmarks,
scored on a 19-benchmark panel.

| system | score | rank |
|---|---|---|
| Jev | 59.5 | 1 |
| jevfire (zero-training wrapper on a stock 27B-class model) | 55.7 | 2 |
| joshua-diffusion (zero-training wrapper on a stock 27B-class model) | 55.6 | 3 |
| decider-35b-a3b (NVFP4) | 54.3 | 4 |
| decider-2b | 44.0 | 14 |

decider-35b-a3b is fourth of 32 and the highest-scoring trained model on this edition. Per-area scores of the panel:

| area | decider-35b-a3b | Jev |
|---|---|---|
| knowledge (GPQA, GSM8K, CRUXEval, MMLU) | 0.51 | 0.69 |
| language | 0.61 | 0.62 |
| retrieval | 0.34 | 0.37 |
| tools | 0.72 | 0.73 |
| arts | 0.53 | 0.56 |

The gap to Jev is the knowledge area: 0.18 behind there, within 0.03 everywhere else.

## decider-35b-a3b against decider-2b v10

The same supervised recipe on Qwen3.5-35B-A3B-Base (34.7B parameters, 3B active per token), one epoch of the public mixture
(463M tokens) with the routed experts frozen and Muon on the block matrices, 394 minutes on four B300s. No RL stage. Every row
below is scored by both models on identical inputs; intervals are 95% paired bootstrap intervals. `docs/HISTORY.md` has the
training details and the optimizer comparison, `moe/` the scripts.

| on the same rows | decider-2b v10 | decider-35b-a3b v1 | difference |
|---|---|---|---|
| regression set, 67 in-task tasks, accuracy / NLL / ECE | 0.805 / 0.474 / 0.037 | 0.855 / 0.357 / 0.026 | higher accuracy on 93 of 95 tasks |
| regression set, 28 held-out tasks | 0.755 / 0.622 / 0.084 | 0.810 / 0.497 / 0.069 | |
| 847 in-task validation rows, accuracy / NLL | 83.2% / 0.444 | 90.0% / 0.329 | +6.7 (+4.5 to +9.0) |
| OpenJev, 5,252 rows | 63.3% / 0.916 | 68.3% / 0.752 | +5.0 (+3.8 to +6.2) |
| Mind2Web, 1,770 rows | 82.7% / 0.543 | 89.6% / 0.316 | +6.9 (+5.1 to +8.7) |
| TypeSafe workflow decisions, 102 rows | 80.4% / 0.585 | 86.3% / 0.342 | +5.9 (−2.0 to +13.7) |
| Bespoke's public suite, macro / micro | 0.704 / 0.711 | 0.774 / 0.787 | Jev 1.13.0: 0.760 / 0.773 |
| JevBench public items, easy / standard / hard | 1.000 / 0.889 / 0.459 | 1.000 / 0.972 / 0.676 | Jev 1.13.0: 1.000 / 0.986 / 0.730 |
| live MiniWoB++ click tasks, greedy play | 90.9% | 97.2% | +6.2 (+1.7 to +10.8) |
| live MiniWoB++ click tasks, sampled play | 93.2% | 86.4% | −6.8 (−12.5 to −1.7) |
| zero-shot games, win rate, greedy / sampled | 26.5% / 23.7% | 37.2% / 24.1% | +10.7 (+5.6 to +15.8) / +0.4 |

The largest gains are on knowledge and reasoning tasks (MedQA +31 points, MedMCQA +24, TruthfulQA +22, Winogrande +20, MMLU
+19). The browser rows show what v10's RL stage does and this model lacks: its argmax is right more often, but its served
distribution still puts mass on wrong elements, so sampled play is behind v10 and 3 points ahead of v8. Serving cost is 3 to 4
times that of decider-2b per decision (47 ms per request eager, about 520 decisions/s in batches of 64 on one B300).
The NVFP4 build (19.6 GB, ModelOpt) served by vLLM loses 1.0 to 1.5 accuracy points against bf16 in the same engine on the TypeSafe
and validation rows and changes the argmax on 3 to 4% of rows; `moe/vllm_check.py` is the readout through vLLM.

## decider-4b against decider-2b v10 and decider-35b-a3b

Qwen3.5-4B-Base (4.2B, dense), one supervised pass over mixture v2 (742M tokens: the public mixture plus 26 further public decision
datasets and ten programmatic families with verifiable gold), `torch.optim.AdamW` on the bf16 parameters with no master copy, peak LR
1e-5, 577 minutes on two B300s. No RL stage. Temperature 1.05, fitted on the in-task half of the regression set as for the other
models. Every row is scored by all three models on identical inputs; intervals are 95% paired bootstrap intervals (tasks for the
regression set, rows for the fixtures, boards for the games, task-seed pairs for the browser).

| on the same rows | decider-2b v10 | decider-4b v1 | decider-35b-a3b v1 | 4B minus 2B | 4B minus 35B |
|---|---|---|---|---|---|
| regression set, 67 in-task tasks, accuracy / NLL / ECE | 0.805 / 0.474 / 0.037 | 0.834 / 0.404 / 0.027 | 0.855 / 0.357 / 0.026 | +2.8 (+2.0 to +3.7); higher on 87 of 95 tasks | −2.1 (−2.9 to −1.5) |
| regression set, 28 held-out tasks | 0.755 / 0.622 / 0.084 | 0.788 / 0.558 / 0.071 | 0.810 / 0.497 / 0.069 | +3.3 (+2.1 to +4.5) | −2.3 (−3.5 to −1.2) |
| 847 in-task validation rows, accuracy / NLL | 83.2% / 0.444 | 86.0% / 0.417 | 90.0% / 0.329 | +2.7 (+0.7 to +4.7) | −4.0 (−6.1 to −2.1) |
| OpenJev, 5,252 rows | 63.3% / 0.916 | 63.8% / 0.894 | 68.3% / 0.752 | +0.6 (−0.6 to +1.8) | −4.5 (−5.7 to −3.3) |
| Mind2Web, 1,770 rows | 82.7% / 0.543 | 88.3% / 0.367 | 89.6% / 0.316 | +5.7 (+4.0 to +7.3) | −1.2 (−2.8 to +0.2) |
| TypeSafe workflow decisions, 102 rows | 80.4% / 0.585 | 80.4% / 0.609 | 86.3% / 0.342 | 0.0 (−9.8 to +9.8) | −5.9 (−12.7 to +1.0) |
| Bespoke's public suite, macro / micro | 0.704 / 0.711 | 0.757 / 0.765 | 0.774 / 0.787 | | |
| JevBench public items, easy / standard / hard | 1.000 / 0.889 / 0.459 | 1.000 / 0.958 / 0.541 | 1.000 / 0.972 / 0.676 | | |
| live MiniWoB++ click tasks, greedy play (all / 6 held-out) | 90.9% / 91.7% | 91.5% / 75.0% | 97.2% / 97.9% | +0.6 (−4.5 to +5.7) / −16.7 (−27.1 to −6.2) | −5.7 (−9.7 to −2.3) / −22.9 |
| live MiniWoB++ click tasks, sampled play (all / held-out) | 93.2% / 91.7% | 90.9% / 79.2% | 86.4% / 79.2% | −2.3 (−8.0 to +2.8) / −12.5 (−25.0 to 0.0) | +4.5 (−0.6 to +10.2) / 0.0 |
| zero-shot games, win rate, greedy / sampled | 26.5% / 23.7% | 28.6% / 27.7% | 37.2% / 24.1% | +2.1 (−3.0 to +7.3) / +4.0 (+1.1 to +7.1) | −8.5 (−13.7 to −3.0) / +3.5 (+1.2 to +5.9) |
| Decision Index 4,000-request sample: index / calibration error (site definition) / mean confidence against accuracy | 42.3 / 0.093 / 0.658 vs 0.566 | 48.5 / 0.086 / 0.722 vs 0.637 | 50.4 / 0.027 / 0.684 vs 0.690 | | |

The gain over the 2B is on knowledge and reasoning tasks (MedQA +16.5 points, MedMCQA +13.7, Winogrande +12.2, TruthfulQA +12.0,
MMLU +10.9; nine knowledge tasks 0.800 against 0.707), on Mind2Web and on in-task rows; on the two fixtures in neither model's
training data (TypeSafe, OpenJev) the 4B is level with the 2B. The seven tasks where it is lower are the abstention probe (−4.6),
agent trajectories (−1.3), HelpSteer3 preference (−1.1) and four under one point. The browser rows show the missing RL stage: greedy
play matches v10 over all tasks and is 17 points below it on the six tasks v10's RL never rewarded. The ten text games are
zero-shot for the 4B (mixture v2 has no game rows): CliffWalking −13 and Blackjack −0.6 at teacher level, Breakout 18 against the
teacher's 22, BabyAI-GoTo 0.54 above the teacher's 0.34, Pong −21 (not learned), the grid worlds 0 as for every model. The Decision
Index row is a readout of the 4,000-request sample (about 88 cases per benchmark, about 4 points under a full run) through the
public server with the stored temperature, calibration computed with the site's benchmark-weighted definition; the 4B's
confidence exceeds its accuracy by 0.085 on it, about the 2B's gap, where the 35B has none. Speed: 24.7 ms median per decision on
one B300 eager without CUDA graphs over 200 game-state decisions (2B 17.9 ms, 35B 41.4 ms, same method); the CUDA-graph and FP8
paths were not measured on the 4B.

## The 94 public tasks

Large label sets sub-sampled to 10 options; one temperature fitted on in-task data.

| | in-task acc / ECE (69 tasks) | held-out acc / NLL / ECE (24 tasks) |
|---|---|---|
| Qwen3.5-2B-Base, zero-shot | 0.620 / 0.121 | 0.642 / 0.853 / 0.105 |
| decider v8, state-first (default), T=1.30 | 0.811 / 0.037 | 0.741 / 0.655 / 0.088 |
| decider v9, state-first (default), T=1.36 | 0.812 / 0.041 | 0.741 / 0.655 / 0.087 |
| decider v8, schema-first (the cacheable layout), T=1.18 | 0.790 / 0.038 | 0.707 / 0.757 / 0.104 |
| `scripts/train.sh full`, one run from the base model, T=1.03 | 0.809 / 0.030 | 0.739 / 0.620 / 0.079 |
| decider v8, rebuilt set (67 / 28 tasks, see note), T=1.30 | 0.806 / 0.038 | 0.757 / 0.622 / 0.083 |
| decider v10, rebuilt set (67 / 28 tasks, see note), T=1.30 | 0.805 / 0.037 | 0.755 / 0.622 / 0.084 |
| decider-4b v1, rebuilt set, T=1.05 | 0.834 / 0.027 | 0.788 / 0.558 / 0.071 |
| decider-35b-a3b v1, rebuilt set, T=1.08 | 0.855 / 0.026 | 0.810 / 0.497 / 0.069 |

The "rebuilt set" rows were measured on a different machine (B300) after the data pipeline was rebuilt: two datasets no longer
download (TREC-fine, the game states) and the current mixture adds held-out probes, so that set has 67 in-task and 28 held-out
tasks and its numbers are not comparable to the rows above it, only to each other. v10 matches v8 on it; the largest per-task
moves are CommitmentBank −5 points (250 rows) and PAWS +2.

Schema-first trades accuracy for speed, and the cost depends on the workload: on the 69 tasks with a fixed label set
(classification, routing, scales, which is what a cached schema is for) it loses 1.5 points on average (median 0.7, calibration
equal); on the 24 tasks whose options change per example (multiple-choice QA, tool choice) it loses 5, because the options are read
before the question they belong to; on full label sets of 50-219 options and on states of several thousand tokens it loses 5-24.
State-first is therefore the default and the schema cache is opt-in (`Decider.schema`, `DECIDER_SCHEMA_CACHE=1`).

decider-0.8b, the same recipe from Qwen3.5-0.8B-Base: 0.78 in-task / 0.71 held-out on the 94 tasks with the same calibration.
It loses on knowledge tasks, not on the decision format.

## JevBench public items

**JevBench** ([Benchmark Heaven](https://benchmarkheaven.com/jev-models), harness at
[fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench)) ranks Jev-class systems on 534 decisions in four tiers;
231 of the items are public (easy 48, standard 72, hard 111). decider is not on that leaderboard. We ran both versions over the
public items with the request the harness's TypeSafe adapter builds (one question, `state` plus `instructions` and `criteria`,
exact label set) and score argmax accuracy the same way. The other systems' numbers below are their published per-item outcomes
on the same public items; the leaderboard's Intelligence score also covers 303 held-out and imported items, and its total score
adds speed and cost measured from the operator's server, so this table is a partial comparison.

| system (public JevBench items) | easy (48) | standard (72) | hard (111) |
|---|---|---|---|
| GPT-5.6 Luna, low reasoning (verbalized probabilities) | 1.000 | 0.972 | 0.964 |
| Jev 1.13.0 (TypeSafe AI) | 1.000 | 0.986 | 0.730 |
| **decider-35b-a3b v1** (34.7B, 3B active) | 1.000 | 0.972 | 0.676 |
| djev (Maisa, diffusion-gemma) | 1.000 | 0.986 | 0.676 |
| OpenJev (DiffusionGemma 26B-A4B) | 1.000 | 0.972 | 0.640 |
| SemIf (Qwen3.5-4B) | 1.000 | 0.986 | 0.613 |
| **decider-4b v1** (4.2B) | 1.000 | 0.958 | 0.541 |
| open-alternative-jev (Qwen3.5-4B) | 1.000 | 0.833 | 0.568 |
| system-one-open (Gemma 4 E2B) | 1.000 | 0.931 | 0.486 |
| system-one (Qwen3-8B) | 1.000 | 0.889 | 0.486 |
| **decider-2b v10** (1.9B) | 1.000 | 0.889 | 0.459 |
| decider-2b v8 | 1.000 | 0.875 | 0.441 |
| Bespoke Nimble 9B | 1.000 | 0.931 | 0.369 |
| open-jev-deberta-v3-large | 1.000 | 0.431 | 0.378 |

The decider-2b rows were read again on 2026-09-23 in process, bf16, decider-ai 1.2.1. The values first published (standard / hard:
v10 0.847 / 0.459, v8 0.861 / 0.459) came from the FP8 server of 2026-09-19 and do not reproduce item for item.

On the standard tier decider-2b v10 misses answer-adequacy judgments (4 of 12), routing (3 of 12) and one policy item. The hard
tier is long policy texts, multi-hop and temporal-numeric reasoning, which a 2B model without reasoning does not do well: it is at
0.26 (long policy), 0.33 (temporal-numeric) and 0.50 (multi-hop) on those families and at 1.00 on the trap and hard-routing
families. Its top-label ECE on the hard items is 0.31, meaning it is confident where it is wrong there.

## Bespoke's public suite

[Nimble](https://github.com/bespokelabsai/nimble), 2026-09-19: Qwen3.5-9B + LoRA on 2,676 contrastive examples, with 13
human-labelled subsets, 3,880 records in Jev's wire format, on which they measured Nimble and Jev 1.13.0. The subsets rebuild
byte-for-byte from their manifests; decider answers them through `system_one` as shipped (`decider/bench/public_suite.py`).
"trained" marks tasks whose *train* split is in decider's mixture.

| subset (type) | decider-2b v9 | decider-2b v10 | decider-4b | decider-35b-a3b | Nimble-9B | Jev 1.13.0 |
|---|---|---|---|---|---|---|
| vitaminc-dev (choice, contrastive fact verification) | 0.651 | 0.639 | 0.756 | 0.795 | 0.766 | 0.801 |
| massive-en-US (choice, 18 scenarios; trained) | 0.826 | 0.823 | 0.860 | 0.880 | 0.869 | 0.874 |
| massive-de-DE (same utterances in German) | 0.794 | 0.797 | 0.843 | 0.869 | 0.834 | 0.869 |
| boolq (noul; trained) | 0.803 | 0.803 | 0.873 | 0.887 | 0.860 | 0.897 |
| squad2 (noul, answerability) | 0.786 | 0.776 | 0.706 | 0.749 | 0.806 | 0.829 |
| paws (noul, paraphrase; trained) | 0.716 | 0.720 | 0.716 | 0.768 | 0.828 | 0.892 |
| multinli (choice; trained) | 0.843 | 0.856 | 0.933 | 0.910 | 0.853 | 0.829 |
| civil_comments (noul; trained) | 0.843 | 0.840 | 0.880 | 0.907 | 0.703 | 0.810 |
| aegis2 (noul, prompt safety) | 0.720 | 0.728 | 0.820 | 0.808 | 0.812 | 0.804 |
| helpsteer2 (score, 5 levels; trained) | 0.438 | 0.426 | 0.466 | 0.478 | 0.390 | 0.341 |
| summeval-relevance (score) | 0.329 | 0.354 | 0.425 | 0.483 | 0.492 | 0.350 |
| summeval-consistency (score) | 0.646 | 0.660 | 0.833 | 0.757 | 0.757 | 0.812 |
| pubmedqa (choice; trained) | 0.720 | 0.724 | 0.728 | 0.768 | 0.756 | 0.772 |
| **macro / micro** | **0.701 / 0.711** | **0.704 / 0.711** | **0.757 / 0.765** | **0.774 / 0.787** | 0.748 / 0.759 | 0.760 / 0.773 |

Nimble's and Jev's numbers are copied from their report. decider-35b-a3b is above both on the average (0.774 against 0.748 and 0.760) and behind Jev on PAWS, SummEval consistency and SQuAD2. decider-4b is above Nimble-9B and 0.3 points under Jev on the average (0.731 on the six subsets whose train split is not in the mixture), behind Jev on PAWS, VitaminC and SQuAD2, where it is also 7 points under the 2B. A 2B model is 5 points under a 9B and 6 under Jev on the average; it is
ahead on moderation (civil_comments) and on HelpSteer2, and behind most where a claim has to be checked against evidence that
nearly matches it (VitaminC, PAWS, SummEval consistency) and on prompt-safety judgments (Aegis).

## Speed

decider-2b on one GH200, bf16 + torch.compile + CUDA graphs; support tickets are about 230 tokens, chat messages about 12. v10 is
unchanged. On one B300 (decider-ai 1.2.1, 2026-09-23) v10 takes 3.2 ms per ticket request with graphs and runs about 2,700
decisions/s in batches of 32; the model card has the B300 table. decider-35b-a3b runs eager (`use_graphs=False`) at 47 ms per request and about 520 decisions/s in batches of 64 on one
B300; its CUDA-graph and FP8 paths are untested. decider-4b, eager, batch of one, on one B300: 24.7 ms median per decision over 200
game-state decisions of 156 tokens median (2B 17.9 ms, 35B 41.4 ms, same decisions and method); its CUDA-graph and FP8 paths were
not measured.

| in-process, per forward | full forward | schema cache | |
|---|---|---|---|
| 3 questions, tickets: 1 request / 32 requests | 4.0 / 74 ms | 3.4 / 47 ms | 1.2x / 1.6x |
| 10 described questions, tickets | 5.9 / 154 ms | 4.0 / 64 ms | 1.5x / 2.4x |
| 10 described questions, chat messages | 6.0 / 121 ms | 3.9 / 29 ms (11,180 decisions/s) | 1.5x / 4.2x |
| one question with 151 options, chat messages | 8.5 / 217 ms | 3.6 / 11.5 ms | 2.4x / 19x |
| 10 questions scored independently, chat messages | 14.3 / 276 ms | 4.6 / 75 ms | 3.1x / 3.7x |

Independent scoring with the cache reruns the state once per question, so it only pays for short states (tickets: 1.0-1.7x).
For long states the state-first path runs the state once and forks its cache per question (7 questions on 11k tokens: 252 ms
instead of 1464 ms). HTTP, 5 questions per request, tickets, without compile/FP8: `/decide` 193 req/s and `/v1/systemone` packed
with the schema cache 352 req/s at 64 clients (p50 8 ms at one client); independent scoring 70-75 req/s either way.

Apple Silicon (M1 Pro 32 GB, macOS 27.2, float16, PyTorch 2.14.0, Transformers 5.17.0): across the three 2B smoke-test
workloads the median request is 133 ms with the MPS patch and 171 ms without it, each row itself the median of five runs after
two warmups; the held-out MASSIVE Scenario set (1,500 examples, temperature 1.30) scores accuracy 0.7553 / ECE 0.0438 against
the published bf16 row's 0.756 / 0.041. These are nine specific smoke-test workloads, not a representative accuracy suite.
Conditions and raw output: `docs/benchmarks/mps-full-model.md`, `docs/benchmarks/mps-heldout.md`.

## Input shapes

Accuracy; state-first unless noted.

| | v5 | v8 | v8 schema-first |
|---|---|---|---|
| all 64 / 50 / 70 / 219 labels offered at once: HWU64, TREC-fine, DBpedia L2, L3 (held-out) | 0.25 / 0.29 / 0.17 / 0.09 | 0.84 / 0.72 / 0.73 / 0.86 | 0.80 / 0.48 / 0.60 / 0.69 |
| CLINC 151-way / Banking 77-way | 0.11 / 0.19 | 0.88 / 0.87 | |
| options named by opaque ids, only descriptions tell them apart (8 held-out tasks; plain names: 0.77) | 0.73 | 0.78 | 0.75 |
| JSON state, question names one of 4 / 16 / 64 records by path (one record: 0.70) | 0.60 / 0.51 / 0.43 | 0.69 / 0.64 / 0.51 | 0.65 / 0.53 / 0.45 |
| same, 16 / 64 records, array positions written into the state (`render_state` does this) | | 0.68 / 0.62 | 0.61 / 0.60 |
| the record is in an 11k-token / 20-30k-token state | 0.45 / 0.47 | 0.61 (0.68 indexed) / 0.57 | 0.49 |
| QuALITY, whole article (5-8k tokens); clipped to 5000 characters: 0.50 | 0.71 | 0.70 | 0.56 |

## Custom questions and catch-all options

v6 to v8; v7 added teacher-written data for exactly this.

| | v6 | v8 |
|---|---|---|
| hand-written battery: the GENERIC option is right although a catch-all is offered ("support" vs "other") / the catch-all is right | 0.60 / 0.90 | 0.85 / 0.95 |
| teacher-written routing messages, 6 held-out domains: generic / specific / catch-all | 0.50 / 0.95 / 0.82 | 0.94 / 0.97 / 0.90 |
| teacher-written custom questions, held-out domains: noul / choice / score | 0.94 / 0.96 / 0.74 | 0.96 / 0.98 / 0.83 |
| off-topic abstention probe / abstention battery | 0.83 / 7 of 8 | 0.83 / 8 of 8 |

The teacher labels come from Qwen3.5-27B; the hand-written battery (60 choice cases, 49 yes/no) is small. Both are in the repo.

## Terse buckets and applications (v9)

v8 needed the generic option to look like a bucket (`general_support`); v9 adds teacher-written messages over plain option lists
(`support`, `help`, `account`, no descriptions) and labelled shell commands. Held-out terse-bucket messages, generic / specific
/ catch-all: v8 0.59 / 0.96 / 0.93, v9 0.86 / 0.95 / 0.88; hand battery 0.95 / 0.95 / 0.90. The 94-task set is unchanged (0.812
/ 0.741). Three hand-written application checks (`decider/probes/applications.py`), zero-shot:

| | v8 | v9 |
|---|---|---|
| model router, 31 prompts: tier (small / code / large reasoning / a person) and "needs live data" | 0.90 / 0.81 | 0.94 / 0.84 |
| shell command safety, 45 commands: safe / caution / destructive, and "touches things outside the project" | 0.71 / 0.56 | 0.80 / 0.98 |
| browser agent, 16 page states as JSON: which element to act on, which action | 1.00 / 0.88 | 1.00 / 0.88 |

No destructive command was ever called safe; the command misses are caution/safe borderlines (`npm run build`, `mkdir && cp`).

## Form filling, against a specialist

`decider/probes/cua_s1_forms.py`. Cua's CUA-S1-FORMS (2026-09-18) is a 0.7M-parameter byte-level System One model for one task:
for each form element, pick the document value to fill in, or check / click / skip. On its synthetic test split (14,254
decisions, forms disjoint from its training forms) it scores 0.9995; its card puts Jev's hosted API at 0.836. decider v9,
zero-shot, scores 0.41 with their bare strings (it almost never chooses a bare `skip`), 0.67 with a one-sentence question and
`skip (leave this element alone)`, and 0.24 when every rule is spelled out in the question. Entities are rarely confused (wrong
target on 229 of 6,018 fills); the misses are the action conventions, above all re-filling an already filled field.

## Isolated Score levels

Each level is judged in its own row, without its number or its neighbours; the per-level P(fits) are normalised. Adding a level
cannot change another level's fit. Against the usual listwise scoring (all levels in one list):

| | listwise acc / ECE | isolated acc / ECE | mean sum of fits |
|---|---|---|---|
| teacher-written score questions, held-out domains | 0.822 / 0.058 | 0.827 / 0.059 | 0.99 |
| HelpSteer2 (5 attributes, 5 levels) | 0.598 / 0.069 | 0.610 / 0.044 | 1.01 |
| hate-speech intensity scales | 0.563 / 0.058 | 0.552 / 0.032 | 1.04 |
| LIAR2 truthfulness (6 levels) | 0.370 / 0.048 | 0.337 / 0.075 | 1.12 |

Before training for it (v6) the same procedure lost up to 20 points and the fits summed to 1.4-3.5.

## Independence

Packed into one prompt, reversing the question order changes up to 12% of answers (7 multi-question tasks). Scored one row per
question there is nothing to change, at the same accuracy (within 0.7 points of packed on every task).

## Text games (supervised stages)

The four trained text games stay at teacher level (Pong 8, Breakout 22, CliffWalking -13); held-out Freeway, 6 at v4, is 0.
`decider/games/` also has the Super Mario Bros demo (`media/mario_*.gif`).

The v10 RL stage changed the served distributions rather than the win rates. Win rates on the same 234 boards, sampled play,
with 95% intervals over boards (from [docs/CHANGELOG.md](CHANGELOG.md)):

| game | v8 | v10 | difference |
|---|---|---|---|
| bag draws (64 boards x 4) | 35.2% | 41.4% | +6.2 (+0.8 to +11.7) |
| 5x5 slippery grid (64 x 4) | 14.1% | 18.8% | +4.7 (−2.0 to +11.3) |
| tic-tac-toe against minimax with 25% random moves (74 x 4) | 23.6% | 23.0% | −0.7 (−4.7 to +3.0) |
| 4x4 minesweeper, 4 mines (32 x 4) | 2.3% | 0.0% | −2.3 (−4.7 to 0.0) |

## Browser tasks

The v10 browser results are on the 22 click-only MiniWoB++ tasks: small synthetic pages, elements listed as text. Typing,
scrolling and real websites were not tested. On the same rows, sampled play over 22 tasks x 8 seeds: v8 83.0%, v10 93.2%
(+10.2, +5.1 to +15.9); on the 6 tasks never used for reward, 72.9% to 91.7% (+18.8, +6.2 to +31.2). Greedy play is 90.9% for
v10, 91.5% for decider-4b (75.0% on the six held-out tasks, 17 points under v10 there; no RL stage) and 97.2% for decider-35b-a3b. In the browser v10 predicts the outcome of its own click (success, failure, continue) at a
log score of −0.03 against v8's −0.35. Recordings and the per-task figure are in `media/` and
[docs/CHANGELOG.md](CHANGELOG.md).
