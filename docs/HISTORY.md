# Development history

How the released weights were produced, stage by stage (v1 to v9), with the measurements taken at each stage. The reference
implementation in this repository folds all of it into one data pipeline and one training recipe (see the README); module and
script names mentioned below are the ones used at the time and no longer exist.


![the v4 model playing ten games from text state descriptions, plus Mario with the RL checkpoint](media/montage.gif)

*The v4 model playing every game in `decider/games.py` from text state descriptions (one typed decision per step, ~4 ms each), plus Super Mario Bros with the RL checkpoint. Built by `decider/montage.py`.*

Goal: reproduce the shape of TypeSafe AI's *Jev* (typed decisions with calibrated
probabilities, all fields from one forward pass, no token generation) on a
single GH200, starting from a modern small open base model.

## Design
* **Backbone**: `Qwen/Qwen3.5-2B-Base` (Feb 2026, hybrid linear-attention).
* **Interface**: `Context` + N typed `Question`s, each with an explicit option list.
  Every question gets an answer slot `Answer k: (`; the logits at that slot are
  restricted to the option-letter tokens `A..J` and softmaxed. All N decisions
  come from one forward pass (no decoding).
* **Candidate conditioning**: label sets larger than 10 are sub-sampled per
  example (gold always kept) and shuffled, so the model must read the options.
  From v6, up to 255 options can be offered at once (see "v6" below).
* **Training objective**: cross-entropy (a proper scoring rule) on ~60 public
  decision datasets (intents, routing, moderation, NLI, MCQ, sentiment, ...).
  Optional Brier term. Post-hoc temperature is fitted on in-task data and
  tested on held-out tasks.
* **Evaluation**: 47 in-task + 16 held-out (never trained) tasks. Metrics:
  accuracy, NLL, Brier, ECE, AURC, selective accuracy at 80% coverage.
  The held-out set is the important one: does calibration transfer to tasks
  the model never saw?

## Serving

```bash
DECIDER_MODEL=runs/r3_v2/model .venv312/bin/uvicorn decider.serve:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/decide -H 'content-type: application/json' -d '{"context": "My card was charged twice.",
  "schema": {"Which department?": {"type": "choice", "options": ["billing", "technical", "sales"]},
             "Refund needed?": {"type": "bool"},
             "Urgency?": {"type": "scale", "legend": {"0": "low", "1": "medium", "2": "high"}}}}'
```
Requests arriving within a few milliseconds are scored in one forward pass. `decider.infer.Decider`
uses the same CUDA-graph engine in-process.

Measured on one GH200 (support-ticket contexts, ~230 tokens, 5 typed fields per request):

| | latency p50 | throughput |
|---|---|---|
| in-process `Decider.decide`, eager PyTorch | 49 ms | |
| in-process `Decider.decide`, CUDA-graph engine | 6.6 ms | |
| HTTP, 1 client | 10 ms | 97 req/s |
| HTTP, 4 clients | 25 ms | 150 req/s |
| HTTP, 64 clients | 231 ms | 263 req/s = 1314 decisions/s |

The single-request cost was launch overhead (1742 kernels per forward, GPU busy 5 ms of 44);
CUDA graphs remove it. Batched time was half elementwise kernels, a third matmul and 11% a
cuDNN depthwise-conv fallback; `torch.compile` (with `use_cache=False`, which avoids the
graph breaks) plus a fusable conv fixes the first and last, FP8 (`decider/fp8.py`,
e4m3 weights with per-token activation scales on the Hopper tensor cores) the matmuls.

Engine configurations, same GH200, in-process (`decider/engine.py`):

| engine | single request (228 tok, 3 q) | batch 32 | decisions/s at batch 32 |
|---|---|---|---|
| eager | 49 ms | 117 ms | 837 |
| CUDA graphs | 7.3 ms | 117 ms | 817 |
| + compile + fused conv (default) | 4.0 ms | 70 ms | 1367 |
| + FP8 linears (server default) | 4.1 ms | 58 ms | 1666 |

Accuracy is unchanged across all four (18-task check, 400 examples each: in-task accuracy
0.833 to 0.836, held-out ECE 0.070 to 0.072; see `runs/engcmp`).

HTTP server with the FP8 engine (default), same request mix, closed-loop clients:

| clients | req/s | decisions/s | p50 | p99 |
|---|---|---|---|---|
| 1 | 134 | 669 | 6.8 ms | 7.9 ms |
| 4 | 259 | 1293 | 15.5 ms | 17.5 ms |
| 16 | 334 | 1672 | 47.5 ms | 70.9 ms |
| 64 | 431 | 2152 | 126 ms | 267 ms |

Startup captures 72 (batch, length) shapes, about 8 minutes with compile + FP8; set
`DECIDER_COMPILE=0` for a 30 s start at eager speed.

## v4: situation-to-action data and multiple games

The v3 model had no "read a situation, pick the safe action" data, which is why it ran into the first
goomba zero-shot. v4 continues training from v3 on 45k such examples plus a 2x replay of the general
mixture (`decider/data3.py`, `decider/build_v4.py`; one epoch, 35 min):

* **AgentTraj-L** (AgentGym): ALFWorld, BabyAI, WebShop, ScienceWorld trajectories -> next action among candidates (20k)
* **Mind2Web**: which page element the next operation targets, among candidates (5.9k)
* **synthetic situations** from a local Qwen3.5-27B teacher (`decider/synth_gen.py`): 32 domains, 3-6 options, best action, danger flag (1.5k)
* **teacher-labelled game states** from the four training games below (12k) and Mario (6k)

On the 92-task set (T=1.15 fitted on in-task data, applied by default via `decider_config.json`):
in-task acc 0.813 / ECE 0.033 (69 tasks), held-out acc 0.748 / ECE 0.076 (23 tasks),
i.e. the same as v3 on the shared tasks, plus AgentTraj next-action 92%, Mind2Web 81%, synthetic 76%.

### Ten games behind one interface

`decider/games.py` renders each game's state as text and provides a scripted teacher; `decider/games_eval.py`
plays every game with the model, the teacher, and random. Four games contribute training data
(Pong, Breakout, CliffWalking, MiniGrid Empty); six are never trained on. Three episodes each:

| game | split | random | teacher | zero-shot-r3 | v4-r7 |
|---|---|---|---|---|---|
| pong | train | -20.00 | 8.00 | -21.00 | 8.00 |
| freeway | held-out | 0.00 | 5.00 | 2.00 | 6.00 |
| breakout | train | 1.33 | 22.00 | 7.00 | 22.00 |
| frozenlake | held-out | 0.00 | 1.00 | 0.00 | 0.00 |
| cliffwalking | train | -720.00 | -13.00 | -60.00 | -13.00 |
| blackjack | held-out | -1.00 | -1.00 | -1.00 | -1.00 |
| minigrid_empty | train | 0.00 | 0.96 | 0.00 | 0.00 |
| minigrid_lavagap | held-out | 0.00 | 0.00 | 0.00 | 0.00 |
| minigrid_doorkey | held-out | 0.00 | 0.00 | 0.00 | 0.00 |
| babyai_goto | held-out | 0.20 | 0.25 | 0.00 | 0.00 |

Atari and toy-text games reach teacher level from text alone; Freeway, never trained on, transfers
(and beats its teacher). The grid worlds fail for everyone including the teachers: the egocentric
7x7 view rendered as text is a poor state description and needs a map-based rendering before it
measures the model. Blackjack with three seeded hands is uninformative.

### RL on the gym games

`decider/games_rl.py` runs the same PPO loop over the common game interface (text-state or
pixels-only policy): 16 environment copies per game, batched sampling, per-game per-step
baselines, a success bonus for goal-reaching games, clipped updates with a KL early stop.
Starting from the v4 model, 30 iterations on Pong, Breakout and CliffWalking (about 90 s each),
greedy evaluation every 5; the saved checkpoint is the best one (iteration 25):

| game | SFT start | after RL |
|---|---|---|
| Breakout (train) | 22 | 71 |
| CliffWalking (train) | -13 | -13 (optimal; a mid-run detour to -60 until a success bonus was added) |
| Pong (train) | 8 | 8 |
| Freeway (held-out) | 6 | 7 |
| MiniGrid Empty (held-out here) | 0 | 0 |

Breakout, the game with dense reward, tripled over the scripted teacher it had imitated; the
others held. Without the success bonus the CliffWalking policy learned to avoid the cliff by
never finishing, a standard failure of sparse-goal RL. The first version also produced NaN
gradients from an entropy term over masked options (0 times -inf); masked entropy fixed it.

### v5: the proper abstention fix

v3/v4 had learned the literal option "none of the above" as an abstain signal (offered
verbatim, they abstained even on clear cases). Varying the wording (r9) did not help: the
gold-removed augmentation itself teaches "abstain when the exact label is missing", so a
coarser but correct option ("billing" for a card charged twice) was rejected. v5 (r10) redefines
the augmentation: 75% add an abstain option (twelve wordings) with the answer unchanged; 25%
replace the whole option list with labels from an unrelated task so the abstain option is right
only when nothing fits; abstain-style options are always kept in prompts so their presence
carries no information. Result: a routing battery passes with every wording without the helper
rewrite; a held-out probe with off-topic option lists goes from 0.68 (v4) to 0.83; the 91
shared tasks are unchanged (in-task 0.815, held-out 0.742). `decider_config.json` now carries
`neutralize_none: false` for v5 so the helper stops rewriting the option.

## v6: the input shapes Jev accepts

TypeSafe released Jev on 2026-09-15 with public docs. Compared with them, v5 had the same output contract but a much
narrower input: bare option labels, at most 10 options, a 1536-token context, plain-text state only, and answers that
depended on which other questions shared the prompt. v6 (`runs/r11_v6`, continued from v5 for one epoch, 2.4 h) closes those:

| Jev | v5 | v6 |
|---|---|---|
| options carry a description or a JSON rubric (`criteria`) | bare labels | `name: description` / `name: {"what", "not_for", "examples"}`, trained with opaque names so the description is read |
| up to 255 options per Choice | 10 (letters A-J) | 255: one label token per option (A..Z, then two-letter tokens); 10 or fewer options render exactly as before |
| ~32k-token budget for state + questions | 1536-token context | 32k accepted; trained to 16k, probed to 30k |
| state is a string, object or array; questions point into it by path | text | JSON states, `` `tickets[3].text` `` paths, several records per state |
| every answer independent of the other questions | packed prompt: question k sees questions 1..k-1 | one row per question; the state is run once and its cache forked to every question |
| `confidence` from the shape of the distribution | top probability | `confidence` = top probability (calibrated), plus `certainty` = 1 - normalised entropy |
| HTTP API + Python/JS SDKs, cookbooks | `/decide` | `POST /v1/systemone` with the same wire format: TypeSafe's own SDK works against it (`TYPESAFE_BASE_URL`); `examples/` |

Not copied: their training method (RLCD) is unpublished, and their workflow evals need their API.

**Data** (`decider/build_v6.py`, all derived from the existing mixture, 391k examples, 173M tokens): label descriptions for all
669 fixed labels written by a local Qwen3.5-27B from the label name and five training examples (`decider/describe_labels.py`;
held-out tasks from the name only); described and opaque-named option lists (40k); full native label sets (CLINC 151,
Banking 77, MASSIVE 60, ...) and small label sets padded with labels from unrelated task families (89k); JSON states holding
2-60 records with path questions, up to 14k tokens (32k); multi-question examples asked one question at a time or reordered
(30k); 200k replay of the general mixture at the original 10-option protocol.

**Regression set** (94 tasks, original protocol, T fitted on in-task data: 1.15): in-task acc 0.813 / ECE 0.032 (v5 0.815 / 0.028),
held-out acc 0.736 / NLL 0.664 / ECE 0.084 (v5 0.738 / 0.678 / 0.084), off-topic abstention 0.832 (0.829), abstain probe 0.63 (0.57).
Largest moves: CB 0.89 to 0.80 (56 examples), TREC 0.80 to 0.75. Text games: trained games unchanged; held-out Freeway 6 (v4), 3 (v5), 0 (v6);
BabyAI GoTo 0 to 0.31.

**Full label sets** (every label offered at once; `--max_options 255`; held-out = dataset never trained on):

| task | options | v5 acc | v6 acc | v6 ECE |
|---|---|---|---|---|
| HWU64 (held-out) | 64 | 0.245 | 0.841 | 0.018 |
| TREC fine (held-out) | 50 | 0.292 | 0.758 | 0.052 |
| DBpedia level 2 (held-out labels) | 70 | 0.173 | 0.701 | 0.141 |
| DBpedia level 3 (held-out labels) | 219 | 0.089 | 0.871 | 0.055 |
| CLINC | 151 | 0.110 | 0.879 | 0.058 |
| Banking77 | 77 | 0.186 | 0.862 | 0.020 |
| MASSIVE intent | 60 | 0.291 | 0.862 | 0.027 |
| GoEmotions | 28 | 0.529 | 0.615 | 0.032 |

`examples/hierarchical_beam.py` (TypeSafe's taxonomy-walk pattern on DBpedia 9 -> 70 -> 219): v5 needed the walk (flat 0.13, beam-3 0.77);
v6 answers the flat 219-way question directly (0.905) and the walk is no longer better (0.85).

**Described options** (mean accuracy; names replaced by opaque ids such as `c7` means only the description identifies an option):

| | plain names | names + descriptions | opaque + descriptions | opaque + JSON rubric |
|---|---|---|---|---|
| 8 held-out tasks, v5 | 0.769 | 0.762 | 0.729 | 0.691 |
| 8 held-out tasks, v6 | 0.763 | 0.784 | 0.771 | 0.770 |
| 3 in-task (incl. Banking 77-way), v6 | 0.872 | 0.876 | 0.848 | 0.848 |

Descriptions now help slightly over bare names, and an option is usable from its description alone (ECE with opaque names 0.097 to 0.057).

**JSON state, question names one record by path** (records from held-out tasks; ceiling = one record, 0.72):

| records in the state | 4 | 16 | 64 | ~11k tokens | 20-30k tokens |
|---|---|---|---|---|---|
| v5 | 0.600 | 0.510 | 0.432 | 0.445 | 0.47 |
| v6 | 0.696 | 0.644 | 0.488 | 0.640 | 0.57 |
| v6, array positions written into the state | | 0.660 | 0.574 | 0.665 | |

Records addressed by key stay near the ceiling (0.73 at 16 records); records addressed by array position degrade because the model has
to count, so `decider.systemone.render_state` writes `"_index": i` into arrays of 8 or more elements (not trained on; it helps anyway).
Plain long reading already worked in v5: QuALITY with the whole article (5-8k tokens) 0.71 for both, against 0.51 with the article clipped to 5000 characters.

**Independence** (`decider/independence_probe.py`, 7 multi-question tasks). Packed into one prompt, reversing the question order flips
up to 17% of v5's answers (up to 12% for v6). Scored one row per question there is nothing to flip, and v6 gives up little for it
(accuracy within 1.5 points of packed on every task; v5 lost up to 4). The cost: 5 questions on a 190-token state take 23 ms instead of 10 ms
over HTTP. For long states the state is run once and its cache (attention KV and delta-net states) forked per question
(`Engine.score_shared`): 7 questions on an 11k-token state 252 ms instead of 1464 ms, same answers up to bf16 round-off
(max |dp| 0.002 in fp32).

```python
from decider.infer import Decider
d = Decider("runs/r11_v6/model")
d.system_one({"ticket": {"messages": [{"from": "customer", "text": "I was charged twice for order A-104. Please refund the duplicate."}]},
              "refund_policy": "Duplicate charges are eligible for a refund."},
             {"department": {"type": "choice", "instructions": "Which team should handle this?",
                             "criteria": {"returns": "Exchanges, refunds, wrong or damaged items", "billing": {"what": "Charges, invoices", "not_for": "delivery"}, "other": None}},
              "refund_requested": {"type": "noul", "instructions": "Does `ticket.messages[0].text` request a refund?"},
              "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["calm", "frustrated", "very frustrated"]}})
```
```bash
DECIDER_MODEL=runs/r11_v6/model .venv312/bin/uvicorn decider.serve:app --port 8000
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=local python your_typesafe_sdk_script.py     # typesafe-sdk 0.6.0, unchanged
```

## v7 to v9: generic options, isolated levels, terse buckets

Each stage is one continuation epoch on the previous weights over the new data plus a replay sample of everything before
(`runs/r12_v7`, `r13_v8`, `r15_v9b`; 2-3 h each). The 94-task regression set did not move across them (in-task 0.811-0.813,
held-out 0.736-0.741).

**v7: the generic option next to a catch-all.** v6 sent "the app logs me out every time, fix it" to `other` when `support`
was also offered, at high confidence: every catch-all in its training data had been the answer whenever nothing specific
matched. v7 adds teacher-written data for exactly this case (Qwen3.5-27B writes messages for a domain, labels them over
option lists that contain both a generic bucket and a catch-all, then re-checks its own labels from letter logits; labels it
disagreed with are kept only under the trust rules in `decider/data/mixture.py`). Hand battery, generic right / catch-all
right: 0.60 / 0.90 to 0.85 / 0.95; teacher-written routing, held-out domains, generic: 0.50 to 0.94. Free-form custom
questions (noul / choice / score over teacher-written situations) went 0.94 / 0.96 / 0.74 to 0.96 / 0.98 / 0.83.

**v8: isolated Score levels.** Jev judges every Score level on its own. Done naively on v7 (one yes/no row per level,
"Proposed answer: <level>. Does the proposed answer fit?") the fits summed to 1.4-3.5 and accuracy fell up to 20 points, because
the model had only ever seen levels in a list. v8 trains the isolated form (`augment.isolated`, half of the Score data) and
matches listwise scoring within about a point with fits summing to 0.99-1.12. The same run trained the schema-first layout
50/50 with state-first, which is what makes the schema cache possible; its accuracy cost is in the README.

**v9: terse buckets and shell commands.** v8 needed the generic option to look like a bucket (`general_support`); a plain
`support` next to `other` still lost. v9 adds teacher-written messages over terse option lists (`support`, `help`, `account`,
no descriptions; lists kept only when the bucket name is a real bucket word) and labelled shell commands (safe / caution /
destructive). Held-out terse-bucket messages, generic: 0.59 to 0.86, catch-all 0.93 to 0.88. A first version that up-weighted
the generic option over-reached into the buckets on the catch-all side and was dropped in favour of weighting the catch-all
examples 3x (`r15_v9b`, released as v9). Three zero-shot application probes (model router, command safety, browser agent) are
in `decider/probes/applications.py` and the README.

## v10: calibration-aware RL on live browser tasks and exact games

v10 continues the v8 weights (the ones on the Hub; v9 was described in the README but the Hub weights were v8) for 384 steps of
reinforcement learning. The rewards are outcomes only: whether a live MiniWoB++ click task's own checker reports success, whether
a 4x4 minesweeper board, a slippery 5x5 grid or a bag-draw game is won, and how well the model's stated belief about the next
outcome of its action matches the exact law (a proper log score). A KL limit to v8 on replayed supervised rows (0.01 nats mean,
0.05 max per step, otherwise the step follows only the KL gradient) and a rendering-consistency term hold the original tasks in
place. No gold labels. `docs/RL.md` has the recipe and the gates.

On the same rows and seeds as v8: live browser tasks 83.0% to 93.2% sampled success (six never-rewarded tasks 72.9% to 91.7%),
belief excess over the exact laws 0.47 to 0.22 nats, click-outcome log score −0.35 to −0.03, Mind2Web 81.1% to 82.7%, bag-draw
wins +6 points; TypeSafe rows +2 points within noise, 847 in-task validation rows and Bespoke's suite unchanged, OpenJev
64.1% to 63.3%. Greedy browser play barely moved (90.3% to 90.9%): the gain is in the served distribution. Tic-tac-toe, grid
and minesweeper play did not change. Recordings: `media/v10_browser_*.gif`.

Two earlier attempts at this stage did not produce a checkpoint that passed every gate: at peak learning rate 2e-6 the model
drifted on short replayed rows after step 384, and an equal-weight win gate over the four environments could not be reached
because grid and bag play does not move at this size. v10 comes from the run with the rate halved, the consistency term in every
arm and the win gate weighted toward the browser (0.5 browser, 0.25 minesweeper, 0.125 grid, 0.125 bags); all four arms of that
run produced an eligible checkpoint and the one with the highest weighted sampled win was released.

## Vision: decisions from pixels

Qwen3.5-2B is a vision-language model; `decider/vision.py` uses the full model with the same
lettered prompt and slot logits, so an image (or a game frame) goes in front of the text and all
answers still come from one forward pass. A 256x240 game frame costs 64 visual tokens.
Training (`decider/train_vision.py`): frames from the games and Mario labelled by the scripted
teachers (`decider/frames_data.py`, rare actions oversampled), DAgger frames from the model's own
play (`decider/frames_dagger.py`), multiple-choice image tasks from The Cauldron
(`decider/data_vision.py`: A-OKVQA, AI2D, ScienceQA, IconQA, TQA, Raven, Hateful Memes; Visual7W
and VSR held out), and a text replay. v1 started from the base VLM; v2 from the v4 text weights
transplanted into the VLM (`decider/transplant.py`), which keeps the text skills (abstention probe
84% vs 54% in v1).

Image tasks (v2, 300 items each): A-OKVQA 85%, AI2D 93%, ScienceQA 95%, IconQA 94%, Raven 80%,
Hateful Memes 80%; held-out Visual7W 87% and VSR 75%, ECE 0.03 to 0.05. Frame-level agreement
with the teachers: Pong 96%, Breakout 96%, CliffWalking 100%, MiniGrid Empty 100%, Mario 77%.

Playing from pixels only (no text state; `decider/games_pixels.py`, 3 episodes):

| game | split | teacher (text state) | base VLM zero-shot | vision v1 | vision v2 |
|---|---|---|---|---|---|
| pong | train | 8.00 | -21.00 | -21.00 | 3.00 |
| breakout | train | 22.00 | 0.00 | 0.00 | 1.00 |
| cliffwalking | train | -13.00 | -6000.00 | -60.00 | -13.00 |
| minigrid_empty | train | 0.96 | 0.00 | 0.96 | 0.96 |
| freeway | held-out | 5.00 | 8.00 | 5.00 | 6.00 |
| frozenlake | held-out | 1.00 | nan | 0.00 | 0.00 |
| blackjack | held-out | -1.00 | nan | -0.33 | -1.00 |
| minigrid_lavagap | held-out | 0.00 | nan | 0.00 | 0.00 |
| minigrid_doorkey | held-out | 0.00 | nan | 0.00 | 0.00 |
| babyai_goto | held-out | 0.25 | nan | 0.00 | 0.30 |
| mario 1-1 (px) | train | 2023 | 898 | 315 | 315 |

### RL from pixels

`games_rl.py --vision` runs the same PPO loop with the vision model as the policy (frames only).
From the v2 model, 16 iterations on Breakout and Pong (16 environments each, 300-step rollouts,
about 3.5 minutes per iteration), greedy evaluation every 4:

| iteration | Breakout | Pong | Freeway (held-out) | MiniGrid Empty |
|---|---|---|---|---|
| 0 (v2 supervised) | 1 | 3 | 6 | 0.96 |
| 4 | 19 | -9 | 7 | 0.96 |
| 8 | 11 | 8 | 9 | 0.96 |
| 12 | 16 | 8 | 8 | 0.96 |
| 16 | 26 | -5 | 8 | 0.96 |

Reward fixed what imitation could not: the relaunch after a lost life (the DAgger frames came
from a policy that never launched), and by iteration 16 Breakout from pixels exceeds the
RAM-state teacher (22). Pong moves with the same updates and is not stable across checkpoints.
The published vision model uses the iteration-12 checkpoint (the balanced one).

v3 (vision release, built on the v5 text weights, then 12 iterations of pixel RL; best checkpoint
iteration 4): Breakout 41, Pong 3, CliffWalking -13, MiniGrid Empty 0.96, Freeway 0, BabyAI 0,
Mario 315. It trades v2's Pong (8) and Freeway (8) for Breakout and the corrected abstention.

Frame accuracy does not equal play: v1 never launched the ball in Breakout (a rare action in the
training frames) and lost every Pong point; oversampling and DAgger fixed CliffWalking and got Pong
to +3 and MiniGrid Empty to teacher level from the image (the text rendering of that game never
worked). v2 still launches the ball only once per game (after a lost life it answers "stay"),
because the DAgger frames came from a policy that never launched. Mario from pixels dies at the
first goomba; the frame gives less warning than the RAM-derived text. Pixel RL on Breakout and Pong
is the next step and runs the same `games_rl.py --vision`.

## Demo: Super Mario Bros from typed decisions

`decider/mario.py` drives the NES emulator (`gym-super-mario-bros`) with the model: every 4 frames the
emulator RAM is rendered as a short text state (ground, gaps, pipes, enemies ahead), the model answers
one choice field ("What should Mario do right now?": run right / jump right / long jump / jump up /
step left / wait) plus a bool ("Is Mario in immediate danger?"), and the action is held for the next
frames. Decisions take ~4 ms, far below the 67 ms of 4 frames.

```bash
uv pip install -p .venv312/bin/python gym-super-mario-bros==7.4.0 nes-py "gym==0.23.1" imageio imageio-ffmpeg
.venv312/bin/python -m decider.mario runs/r3_v2/model --episodes 3 --video mario.mp4
.venv312/bin/python -m decider.mario --policy heuristic     # scripted baseline on the same state text
```

Results on World 1-1 (3200 px long, deterministic emulator, 3 episodes each):

| policy | distance |
|---|---|
| button spam (right + jump) | 698 px |
| decider-2b zero-shot | 315 px (runs into the first goomba; danger field 0.07) |
| scripted teacher on the same state text | 2023 px |
| decider-2b fine-tuned on 6k teacher-labelled states (2 epochs, 5 min) | 2023 px |

Zero-shot, the classifier has no game sense: with a goomba one tile ahead it still says
"run right" at 0.85. After the short fine-tune (`decider/mario_data.py` labels states with
the scripted policy plus random-action noise for coverage, mixed with a replay of the general
data so the model keeps its other abilities) it reproduces the teacher exactly, reading only the
text, at 4.4 ms per decision. On seven levels never used for training (1-4, 2-2, 2-3, 3-2, 4-2, 7-1,
8-1) it matches the teacher's distance within a few pixels on six and beats it on 2-2, so it learned
the state-to-action mapping rather than a trajectory; its ceiling is the teacher's rules
(`--level`, `--noop_start` to desync the deterministic emulator). Videos: `media/mario_zeroshot.gif`, `media/mario_finetuned.gif`
(`media/mario_finetuned.mp4`).

![zero-shot](media/mario_zeroshot.gif) ![fine-tuned](media/mario_finetuned.gif)

### RL on top of imitation

The softmax over action options is a policy, so `decider/mario_rl.py` trains it with PPO-clip
directly against the emulator: 48 emulators in parallel, one batched forward per decision, reward =
tiles gained per decision, a death penalty and a flag bonus, per-level per-step baselines, 4
minibatch steps per iteration with a KL early stop. Warm-started from the imitation checkpoint,
40 iterations (~100 s each) on 8 training levels; greedy evaluation on those 8 and on 7 unseen levels:

| | train levels (mean px) | unseen levels (mean px) |
|---|---|---|
| imitation start | 766 | 774 |
| RL, best checkpoint (iter 20) | 1017 | 1080 |
| RL, final (iter 40) | 1072 | 780 |

Individual levels moved a lot (6-1: 502 to 2812 at one checkpoint; 1-1 past the teacher's 2023 to
2226; 3-2: 1125 to 2013), and sampled rollouts finished level 1-1, which the teacher never did.
Plain REINFORCE at a higher learning rate collapsed the policy within 10 iterations and at a lower one
did not move it; the clipped update with per-level baselines was what made it learn.
GIFs: `media/mario_rl_1-1.gif`, `media/mario_rl_6-1.gif`.

## Layout
```
decider/data.py        task registry -> Example(context, [Q(text, options, gold)])
decider/prompt.py      prompt/slot construction, multi-question packing
decider/model.py       DecisionModel: hidden state at slots -> letter logits
decider/train.py       finetune (bucketed shapes, bf16, grad-ckpt)
decider/evaluate.py    per-task metrics, saves probs
decider/report.py      side-by-side comparison + temperature scaling
decider/engine.py      Engine: shape-bucketed CUDA graphs (7x lower single-request latency than eager)
decider/serve.py       micro-batching HTTP server (POST /decide {context, schema})
decider/loadtest.py    closed-loop load test against the server
decider/bench_engine.py  eager vs CUDA graph vs torch.compile on fixed shapes
decider/bench_latency.py  decisions/s and latency of the one-pass interface (eager)
data/tasks.pkl    cached examples (python -m decider.data)
runs/             zs_2b, zs_4b (zero-shot baselines), r1_200k, ...
```
Model weights: https://huggingface.co/Mapika/decider-2b (each release is staged with `scripts/stage_release.py` and uploaded with `scripts/upload_hf.py`).
Setup: `uv venv --python 3.12 .venv312 && uv pip install -p .venv312/bin/python torch transformers peft accelerate datasets pillow "numpy<2" scikit-learn flash-linear-attention fastapi "uvicorn[standard]" httpx`.
