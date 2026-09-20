# decider: one-pass typed decisions with calibrated probabilities

[![tests](https://github.com/Mapika/decider/actions/workflows/tests.yml/badge.svg)](https://github.com/Mapika/decider/actions/workflows/tests.yml)
[![weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-Mapika%2Fdecider--2b-yellow)](https://huggingface.co/Mapika/decider-2b)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

![the model playing ten games from text state descriptions, plus Super Mario Bros with the RL checkpoint](media/montage.gif)

*Ten text games and Super Mario Bros, each move one typed decision over the legal actions; see `decider/games/` and
`docs/HISTORY.md`.*

A language model that does not generate text. It reads a **state** and a set of **typed questions** and returns, from one
forward pass, a probability distribution for every question. There is no decoding, no parsing, and no output outside the options
you defined. It is an open reproduction of the "System One" model class (TypeSafe AI's *Jev*), released as a 2B model built on
`Qwen/Qwen3.5-2B-Base` and a 35B mixture-of-experts model built on `Qwen/Qwen3.5-35B-A3B-Base`.

**Contents:** [Models](#models) · [Quick start](#quick-start) · [What it does](#what-it-does) · [How it works](#how-it-works) ·
[Results](#results) · [Train](#train) · [Serve](#serve) · [Repository layout](#repository-layout) · [Changelog](#changelog) ·
[Limitations](#limitations)

## Models

| model | what it is | numbers |
|---|---|---|
| [decider-2b](https://huggingface.co/Mapika/decider-2b) **v10** | the main model: text and JSON states, up to 255 options, 32k tokens; v8 plus 384 steps of calibration-aware RL on live browser tasks and exact games | live browser 93% (v8: 83%); belief 0.22 nats above the exact laws (v8: 0.47); same accuracy as v8 on the regression set |
| [decider-35b-a3b](https://huggingface.co/Mapika/decider-35b-a3b) **v1** | the supervised recipe on Qwen3.5-35B-A3B-Base (3B active parameters, routed experts frozen, Muon); bf16, 65 GB; an NVFP4 build for vLLM is at [decider-35b-a3b-nvfp4](https://huggingface.co/Mapika/decider-35b-a3b-nvfp4) | above decider-2b v10 on 93 of 95 regression tasks (in-task / held-out 0.855 / 0.810 against 0.805 / 0.755); JevBench hard 0.676; Bespoke macro 0.774; no RL stage |
| [decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | the same supervised recipe from Qwen3.5-0.8B-Base, 1.5 GB | 94 tasks in-task / held-out 0.78 / 0.71, same calibration; loses on knowledge tasks, not on the decision format |
| [decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) | decisions from an image plus the same prompt | [try it in the browser](https://huggingface.co/spaces/hugging-apps/decider-2b-vision-demo) (Space built by the Hugging Face team) |

The v8 weights stay available under the Hub tag `v8`. `docs/HISTORY.md` describes every version.

## Quick start

```bash
pip install git+https://github.com/Mapika/decider          # or: git clone ... && pip install -e ".[serve]"
```

```python
from decider.infer import Decider
d = Decider("Mapika/decider-2b")                             # one CUDA GPU, bf16, about 4 GB; downloads the weights on first use
d.system_one(
    {"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
     "refund_policy": "Duplicate charges are eligible for a refund."},
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
     "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
     "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
# {"answers": {"department": {"choice": "billing", "confidence": 0.56, "certainty": 0.37, "probabilities": {"returns": 0.44, "billing": 0.56, "other": 0.00}},
#              "refund_requested": {"noul": 0.99},
#              "frustration": {"score": 0.76, "probabilities": {"0": 0.34, "1": 0.55, "2": 0.10}, "level_fit": {"0": 0.34, "1": 0.55, "2": 0.10}, "fit_mass": 0.99}}}
#                                                             (v10 weights; "returns" also mentions refunds, so the mass is split)

d.decide("My card was charged twice.", [{"question": "Which team?", "options": ["billing", "technical", "sales"]}])
# [{"choice": "billing", "confidence": 0.77, "probs": {"billing": 0.77, "technical": 0.19, "sales": 0.04}}]      the plain form
```

`examples/` has three complete programs (confidence-gated routing, composite scoring, a hierarchical beam over Choice
probabilities); `python examples/routing_with_confidence.py` runs against the released weights. Serving over HTTP in TypeSafe's
wire format is under [Serve](#serve).

## What it does

Both released models take the same requests and return the same answer shape.

| | |
|---|---|
| question types | **Choice** (2-255 options, each optionally with a description or a JSON rubric), **Score** (2-10 described levels, returns the expected level), **Noul** (probability of yes) |
| state | a string, or any JSON value; questions can name a part of it by path (`` `tickets[3].text` ``); up to 32k tokens |
| independence | every question is scored on its own: adding, removing or reordering questions cannot change another answer |
| isolated levels | every Score level is judged alone (it sees neither its number nor its neighbours); the per-level fits are normalised |
| abstention | a catch-all option ("other", "none of the above", ...) is chosen when nothing on offer fits |
| calibration | trained with a proper scoring rule; one temperature fitted on in-task data, checked on held-out tasks; v10 adds RL with a proper-score belief reward |
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

## Results

Unless a version is named, the numbers in this section were measured on the v8 weights on one GH200; v9 is v8 plus the
terse-bucket and command data, with the same numbers on the 94 tasks. "Held-out" means no example of that dataset was trained on.
`docs/HISTORY.md` has the per-stage measurements.

### decider-35b-a3b against decider-2b v10

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
| JevBench public items, easy / standard / hard | 1.000 / 0.847 / 0.459 | 1.000 / 0.972 / 0.676 | Jev 1.13.0: 1.000 / 0.986 / 0.730 |
| live MiniWoB++ click tasks, greedy play | 90.9% | 97.2% | +6.2 (+1.7 to +10.8) |
| live MiniWoB++ click tasks, sampled play | 93.2% | 86.4% | −6.8 (−12.5 to −1.7) |
| zero-shot games, win rate, greedy / sampled | 26.5% / 23.7% | 37.2% / 24.1% | +10.7 (+5.6 to +15.8) / +0.4 |

The largest gains are on knowledge and reasoning tasks (MedQA +31 points, MedMCQA +24, TruthfulQA +22, Winogrande +20, MMLU
+19). The browser rows show what v10's RL stage does and this model lacks: its argmax is right more often, but its served
distribution still puts mass on wrong elements, so sampled play is behind v10 and 3 points ahead of v8. Serving cost is 3 to 4
times that of decider-2b per decision (47 ms per request eager, about 520 decisions/s in batches of 64 on one B300).
The NVFP4 build (19.6 GB, ModelOpt) served by vLLM loses 1.0 to 1.5 accuracy points against bf16 in the same engine on the TypeSafe
and validation rows and changes the argmax on 3 to 4% of rows; `moe/vllm_check.py` is the readout through vLLM.

### The 94 public tasks

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

The two "rebuilt set" rows were measured on a different machine (B300) after the data pipeline was rebuilt: two datasets no longer
download (TREC-fine, the game states) and the current mixture adds held-out probes, so that set has 67 in-task and 28 held-out
tasks and its numbers are not comparable to the rows above it, only to each other. v10 matches v8 on it; the largest per-task
moves are CommitmentBank −5 points (250 rows) and PAWS +2.

Schema-first trades accuracy for speed, and the cost depends on the workload: on the 69 tasks with a fixed label set
(classification, routing, scales, which is what a cached schema is for) it loses 1.5 points on average (median 0.7, calibration
equal); on the 24 tasks whose options change per example (multiple-choice QA, tool choice) it loses 5, because the options are read
before the question they belong to; on full label sets of 50-219 options and on states of several thousand tokens it loses 5-24.
State-first is therefore the default and the schema cache is opt-in (`Decider.schema`, `DECIDER_SCHEMA_CACHE=1`).

### External suites

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
| open-alternative-jev (Qwen3.5-4B) | 1.000 | 0.833 | 0.568 |
| system-one-open (Gemma 4 E2B) | 1.000 | 0.931 | 0.486 |
| system-one (Qwen3-8B) | 1.000 | 0.889 | 0.486 |
| **decider-2b v10** (1.9B) | 1.000 | 0.847 | 0.459 |
| decider-2b v8 | 1.000 | 0.861 | 0.459 |
| Bespoke Nimble 9B | 1.000 | 0.931 | 0.369 |
| open-jev-deberta-v3-large | 1.000 | 0.431 | 0.378 |

On the standard tier decider misses answer-adequacy judgments (7 of 12) and routing (3 of 12). The hard tier is long policy
texts, multi-hop and temporal-numeric reasoning, which a 2B model without reasoning does not do: it is at 0.26 to 0.33 on those
families and at 0.88 to 1.00 on the trap and hard-routing families. Its top-label ECE on the hard items is 0.30, meaning it is
confident where it is wrong there.

**Bespoke's public suite** ([Nimble](https://github.com/bespokelabsai/nimble), 2026-09-19: Qwen3.5-9B + LoRA on 2,676
contrastive examples, with 13 human-labelled subsets, 3,880 records in Jev's wire format, on which they measured Nimble and Jev
1.13.0). The subsets rebuild byte-for-byte from their manifests; decider answers them through `system_one` as shipped
(`decider/bench/public_suite.py`). "trained" marks tasks whose *train* split is in decider's mixture.

| subset (type) | decider-2b v9 | decider-2b v10 | decider-35b-a3b | Nimble-9B | Jev 1.13.0 |
|---|---|---|---|---|---|
| vitaminc-dev (choice, contrastive fact verification) | 0.651 | 0.639 | 0.795 | 0.766 | 0.801 |
| massive-en-US (choice, 18 scenarios; trained) | 0.826 | 0.823 | 0.880 | 0.869 | 0.874 |
| massive-de-DE (same utterances in German) | 0.794 | 0.797 | 0.869 | 0.834 | 0.869 |
| boolq (noul; trained) | 0.803 | 0.803 | 0.887 | 0.860 | 0.897 |
| squad2 (noul, answerability) | 0.786 | 0.776 | 0.749 | 0.806 | 0.829 |
| paws (noul, paraphrase; trained) | 0.716 | 0.720 | 0.768 | 0.828 | 0.892 |
| multinli (choice; trained) | 0.843 | 0.856 | 0.910 | 0.853 | 0.829 |
| civil_comments (noul; trained) | 0.843 | 0.840 | 0.907 | 0.703 | 0.810 |
| aegis2 (noul, prompt safety) | 0.720 | 0.728 | 0.808 | 0.812 | 0.804 |
| helpsteer2 (score, 5 levels; trained) | 0.438 | 0.426 | 0.478 | 0.390 | 0.341 |
| summeval-relevance (score) | 0.329 | 0.354 | 0.483 | 0.492 | 0.350 |
| summeval-consistency (score) | 0.646 | 0.660 | 0.757 | 0.757 | 0.812 |
| pubmedqa (choice; trained) | 0.720 | 0.724 | 0.768 | 0.756 | 0.772 |
| **macro / micro** | **0.701 / 0.711** | **0.704 / 0.711** | **0.774 / 0.787** | 0.748 / 0.759 | 0.760 / 0.773 |

Nimble's and Jev's numbers are copied from their report. decider-35b-a3b is above both on the average (0.774 against 0.748 and 0.760) and behind Jev on PAWS, SummEval consistency and SQuAD2. A 2B model is 5 points under a 9B and 6 under Jev on the average; it is
ahead on moderation (civil_comments) and on HelpSteer2, and behind most where a claim has to be checked against evidence that
nearly matches it (VitaminC, PAWS, SummEval consistency) and on prompt-safety judgments (Aegis).

### Speed

decider-2b on one GH200, bf16 + torch.compile + CUDA graphs; support tickets are about 230 tokens, chat messages about 12. v10 is
unchanged. decider-35b-a3b runs eager (`use_graphs=False`) at 47 ms per request and about 520 decisions/s in batches of 64 on one
B300; its CUDA-graph and FP8 paths are untested.

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

<details>
<summary><b>More results of the supervised stages</b>: input shapes, custom questions, terse buckets, applications, form filling, isolated levels, independence, games</summary>

**Input shapes** (accuracy; state-first unless noted)

| | v5 | v8 | v8 schema-first |
|---|---|---|---|
| all 64 / 50 / 70 / 219 labels offered at once: HWU64, TREC-fine, DBpedia L2, L3 (held-out) | 0.25 / 0.29 / 0.17 / 0.09 | 0.84 / 0.72 / 0.73 / 0.86 | 0.80 / 0.48 / 0.60 / 0.69 |
| CLINC 151-way / Banking 77-way | 0.11 / 0.19 | 0.88 / 0.87 | |
| options named by opaque ids, only descriptions tell them apart (8 held-out tasks; plain names: 0.77) | 0.73 | 0.78 | 0.75 |
| JSON state, question names one of 4 / 16 / 64 records by path (one record: 0.70) | 0.60 / 0.51 / 0.43 | 0.69 / 0.64 / 0.51 | 0.65 / 0.53 / 0.45 |
| same, 16 / 64 records, array positions written into the state (`render_state` does this) | | 0.68 / 0.62 | 0.61 / 0.60 |
| the record is in an 11k-token / 20-30k-token state | 0.45 / 0.47 | 0.61 (0.68 indexed) / 0.57 | 0.49 |
| QuALITY, whole article (5-8k tokens); clipped to 5000 characters: 0.50 | 0.71 | 0.70 | 0.56 |

**Custom questions and catch-all options** (v6 to v8; v7 added teacher-written data for exactly this)

| | v6 | v8 |
|---|---|---|
| hand-written battery: the GENERIC option is right although a catch-all is offered ("support" vs "other") / the catch-all is right | 0.60 / 0.90 | 0.85 / 0.95 |
| teacher-written routing messages, 6 held-out domains: generic / specific / catch-all | 0.50 / 0.95 / 0.82 | 0.94 / 0.97 / 0.90 |
| teacher-written custom questions, held-out domains: noul / choice / score | 0.94 / 0.96 / 0.74 | 0.96 / 0.98 / 0.83 |
| off-topic abstention probe / abstention battery | 0.83 / 7 of 8 | 0.83 / 8 of 8 |

The teacher labels come from Qwen3.5-27B; the hand-written battery (60 choice cases, 49 yes/no) is small. Both are in the repo.

**Terse buckets and applications (v9).** v8 needed the generic option to look like a bucket (`general_support`); v9 adds
teacher-written messages over plain option lists (`support`, `help`, `account`, no descriptions) and labelled shell commands.
Held-out terse-bucket messages, generic / specific / catch-all: v8 0.59 / 0.96 / 0.93, v9 0.86 / 0.95 / 0.88; hand battery 0.95
/ 0.95 / 0.90. The 94-task set is unchanged (0.812 / 0.741). Three hand-written application checks
(`decider/probes/applications.py`), zero-shot:

| | v8 | v9 |
|---|---|---|
| model router, 31 prompts: tier (small / code / large reasoning / a person) and "needs live data" | 0.90 / 0.81 | 0.94 / 0.84 |
| shell command safety, 45 commands: safe / caution / destructive, and "touches things outside the project" | 0.71 / 0.56 | 0.80 / 0.98 |
| browser agent, 16 page states as JSON: which element to act on, which action | 1.00 / 0.88 | 1.00 / 0.88 |

No destructive command was ever called safe; the command misses are caution/safe borderlines (`npm run build`, `mkdir && cp`).

**Form filling, against a specialist (`decider/probes/cua_s1_forms.py`).** Cua's CUA-S1-FORMS (2026-09-18) is a 0.7M-parameter
byte-level System One model for one task: for each form element, pick the document value to fill in, or check / click / skip. On
its synthetic test split (14,254 decisions, forms disjoint from its training forms) it scores 0.9995; its card puts Jev's hosted
API at 0.836. decider v9, zero-shot, scores 0.41 with their bare strings (it almost never chooses a bare `skip`), 0.67 with a
one-sentence question and `skip (leave this element alone)`, and 0.24 when every rule is spelled out in the question. Entities are
rarely confused (wrong target on 229 of 6,018 fills); the misses are the action conventions, above all re-filling an already filled
field.

**Isolated Score levels.** Each level is judged in its own row, without its number or its neighbours; the per-level P(fits) are
normalised. Adding a level cannot change another level's fit. Against the usual listwise scoring (all levels in one list):

| | listwise acc / ECE | isolated acc / ECE | mean sum of fits |
|---|---|---|---|
| teacher-written score questions, held-out domains | 0.822 / 0.058 | 0.827 / 0.059 | 0.99 |
| HelpSteer2 (5 attributes, 5 levels) | 0.598 / 0.069 | 0.610 / 0.044 | 1.01 |
| hate-speech intensity scales | 0.563 / 0.058 | 0.552 / 0.032 | 1.04 |
| LIAR2 truthfulness (6 levels) | 0.370 / 0.048 | 0.337 / 0.075 | 1.12 |

Before training for it (v6) the same procedure lost up to 20 points and the fits summed to 1.4-3.5.

**Independence.** Packed into one prompt, reversing the question order changes up to 12% of answers (7 multi-question tasks).
Scored one row per question there is nothing to change, at the same accuracy (within 0.7 points of packed on every task).

**Text games (supervised stages).** The four trained text games stay at teacher level (Pong 8, Breakout 22, CliffWalking -13);
held-out Freeway, 6 at v4, is 0. `decider/games/` also has the Super Mario Bros demo (`media/mario_*.gif`).

</details>

## Train

```bash
uv venv --python 3.12 .venv312 && uv pip install -p .venv312/bin/python -e ".[serve,train]"
scripts/train.sh full                       # datasets -> data/tasks.pkl -> data/mixture_full.pkl -> one epoch from Qwen3.5-2B-Base -> scripts/evaluate.sh
scripts/train.sh delta runs/some/model      # or: continue an existing decider checkpoint on the new formats + a replay sample
```

`decider/data/mixture.py` lists every component of the supervised mixture with its size. The released weights up to v9 were
produced in stages (`delta` runs on top of each other, see `docs/HISTORY.md`); `full` is the same data as a single run, and it
reproduces them: one epoch (1.47M examples, 455M tokens, 5.3 h on a GH200 plus 45 min of evaluation) gives a model that matches
v9 on the 94-task set (in-task 0.809 vs 0.812, held-out 0.739 vs 0.741 on the shared tasks) and on every probe family within
noise, with a fitted temperature of 1.03 instead of 1.36 (better calibrated before scaling: in-task ECE 0.030 vs 0.056). Held-out
terse-bucket routing came out higher (generic / specific / catch-all 0.91 / 0.94 / 0.92) and held-out Freeway play returned (9
against the teacher's 5); the 16-page browser probe came out lower (0.75 / 0.69).

The RL stage that turns v8 into v10 (`docs/RL.md`) needs a live Chrome with MiniWoB++, the exact game environments and the
training loop of a separate research repository; it is not in this package yet.

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

## Repository layout

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
decider/bench/           engine and schema-cache benchmarks, HTTP load test, Bespoke's public suite
decider/games/           ten text games + Super Mario Bros behind the same interface, imitation and PPO
decider/vision/          the vision-language variant (decisions from pixels)
teacher_data/            the teacher-written data the mixture needs (label descriptions, custom questions, routing messages, situations)
scripts/                 train.sh, evaluate.sh, serve.sh, stage_release.py, upload_hf.py
moe/                     the frozen-expert Muon training, evaluation and NVFP4 quantization scripts of decider-35b-a3b
examples/                routing with confidence gates, composite scoring, hierarchical beam over Choice probabilities
tests/                   unit tests for the request/answer layer and the prompt layouts (no GPU; `python -m pytest tests`)
docs/CHANGELOG.md        what changed in every release, newest first, with the v10 recordings and figures
docs/HISTORY.md          how the released weights were produced (v1 to v10 and the 35B) and what was measured at each stage
docs/RL.md               the calibration-aware RL stage that produced v10: rewards, retention, gates, what it changed
media/                   browser and game recordings, figures
```

## Changelog

| version | date | what changed |
|---|---|---|
| decider-35b-a3b v1, and its NVFP4 build | 2026-09-20 | the supervised recipe on Qwen3.5-35B-A3B-Base, routed experts frozen, Muon; above the 2B on 93 of 95 tasks, no RL stage |
| decider-2b v10 | 2026-09-19 | v8 plus 384 steps of calibration-aware RL on live browser tasks and exact games: browser 83% to 93% sampled, belief 0.47 to 0.22 nats above the exact laws, everything else unchanged |
| decider-2b v9 | | terse buckets and command safety in the data; the Hub weights stayed v8 |
| decider-2b v8 | | isolated Score levels, generic options next to a catch-all, the cacheable prompt layout; kept under the Hub tag `v8` |
| decider-2b v6 to v7 | | the input shapes Jev accepts: described options, 255 options, JSON states, the `/v1/systemone` request |
| decider-2b v4 to v5 | | situation-to-action data, ten games, the proper abstention fix |

[docs/CHANGELOG.md](docs/CHANGELOG.md) has the full v10 entry with the browser and game recordings, the calibration figures and
the same-rows comparison against v8. [docs/HISTORY.md](docs/HISTORY.md) is the long form: how each stage was trained and measured.
[docs/RL.md](docs/RL.md) is the RL recipe.

## Limitations

* decider-2b is a 2B model without reasoning: knowledge-heavy multiple choice (MMLU, MedQA) improves little over the base
  model, judgments that need several steps should be split into several questions, and on JevBench's hard tier (long policies,
  multi-hop, temporal arithmetic) it is at 0.46 with a top-label ECE of 0.30. decider-35b-a3b closes part of that gap (hard tier
  0.68, MMLU +19 points) at 3 to 4 times the cost per decision, without the RL stage, and with a hard-tier ECE of 0.15.
* English only. Calibration is measured on public datasets and teacher-labelled probes, not on your traffic: check it on your own labels.
* The schema cache costs accuracy (see Results); use it for fixed classification-style schemas with short states.
* v10 continues the v8 weights, so the v9 results on terse buckets (generic 0.86) do not apply to it; v8's 0.59 does. A plain
  `support` next to `other` sends an in-scope complaint to `other`. Name or describe the generic option as a bucket.
* Rules written into the question ("fill if empty, otherwise skip; check only if required and unchecked") are not followed at
  this size: on the form-filling probe a one-sentence question scores 0.67 and a paragraph of rules 0.24. State the decision as
  a plain question with described options; a fixed convention has to be in the training data, not in the question.
* Picking one record out of a long JSON array by position is the least accurate input shape (0.51 with 64 records against 0.70
  with one); address records by key, or let `render_state` write the index into the array (0.62).
* TREC-fine with all 50 labels fell from 0.76 (v6) to 0.72 (v8); held-out Freeway play fell to 0 and did not come back with the game data replayed.
* The custom-question data is labelled by a 27B teacher that shares some of the biases it is meant to fix (it agreed with only 72%
  of its own generic-option labels); see `decider/data/mixture.py` for how those labels are filtered.
* The v10 browser results are on the 22 click-only MiniWoB++ tasks: small synthetic pages, elements listed as text. Typing,
  scrolling and real websites were not tested. OpenJev accuracy is 0.8 points lower than v8. The games are mostly lost by both
  versions.
* The vision variant (`decider/vision`) is still on v5 text weights, currently retraining.
* The released weights were produced by staged continuation runs (`docs/HISTORY.md`); `scripts/train.sh full` reproduces the
  supervised stages in one run (see Train) but is not byte-identical to them, and the hand-written probes with 16-60 cases move
  by a few cases either way.

## Citation

```bibtex
@software{marosi2026decider,
  author = {Marosi, Mark},
  title  = {decider: one-pass typed decisions with calibrated probabilities},
  year   = {2026},
  url    = {https://github.com/Mapika/decider}
}
```
