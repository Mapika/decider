# Demo clips

How the clips in the README were made. Every clip is built from frames recorded during the run; nothing is interpolated,
re-ordered or sped up inside a clip. Where a clip shows every n-th recorded frame, the factor against real emulator time is
stated. Everything was produced on 2026-09-22 on one NVIDIA B300 SXM6, bf16, batch of one, no CUDA graphs and no
`torch.compile`; these are the plain eager-path latencies, not the CUDA-graph engine `Decider` uses by default.

## `media/pong_35b.gif`

**What is shown.** Atari Pong, played twice from the same environment seed: on the left decider-35b-a3b as released, on the
right the same weights with the games-RL overlay applied. Both panels are the emulator's own RGB output. The model never sees
the picture; it reads a text state built from the console RAM (ball position, direction, paddle offset) and picks one of
"move paddle up", "move paddle down", "stay". The choice is the argmax of the served distribution. Under each panel: the
cumulative score (own points minus opponent points), the chosen action with its probability, and the measured median model
time.

**Checkpoints.** Left: `Mapika/decider-35b-a3b` v1. Right: the same weights with the trained tensors of the games-RL run
described below swapped in (learning rate 2e-6, seed 1, iteration 20; 613 non-expert tensors, the routed experts stay at
base). The overlay is not released.

**Environment and seed.** `ALE/Pong-v5` through `decider.games.envs.Pong`, `obs_type="ram"`, `frameskip=4`,
`repeat_action_probability=0.0`, environment seed 0, decision cap 1200, greedy actions. One decision per environment step,
that is one decision per four emulator frames.

**Result.** The released weights lost the episode 21-0 and it ended after 764 decisions. The overlay arm was at -3 when it
hit the 1200-decision cap. The clip plays the first 764 decisions of both, so the left panel ends exactly when its episode
ended.

**Latency.** Median 43.0 ms per decision on the left and 42.7 ms on the right, measured inside this run: the timer wraps the
single forward pass that produces the answer distribution, with `torch.cuda.synchronize()` on both sides, batch of one.
Environment stepping, prompt building and frame capture are outside the timer. The MoE experts run through the `grouped_mm`
kernel, which is how the 35B is served.

**Time compression.** Every second recorded decision is shown at 20 frames per second, so the clip runs at 2.7 times emulator
speed.

**Selection.** Seeds 0, 1 and 2 were played for both arms; seed 0 is shown. All three gave the same outcome (released -21,
overlay -3), so the choice of seed does not change what the clip says.

### The games-RL run behind the right panel

Outcome-reward RL of decider-35b-a3b on ten text games (Pong, Breakout, Freeway, CliffWalking, FrozenLake, Blackjack, three
MiniGrid tasks, BabyAI GoTo), 2026-09-20 to 2026-09-21. The reward is the game's own score change plus a bonus when an
episode ends in success; no scripted teacher and no gold labels are used. PPO with clip 0.2, GAE against a value head that
reads the slot hidden state, per-game sample quotas, and a retention gate: after every step the KL to the untouched model is
measured on a fixed pool of rows from the supervised training mixture, and a step that moves the mean above 0.01 nats or any slot above 0.05
nats is undone. Four arms (learning rates 1e-6 and 2e-6, two seeds each); the checkpoint per arm is chosen on a validation
seed set and reported on a fresh test seed set.

Greedy test scores of the arm shown in the clip (2e-6, seed 1, iteration 20), against the untouched model:

| game | released 35B | with overlay | random | rule-based teacher |
|---|---|---|---|---|
| Pong | -21 | -12 | -20.4 | 8 |
| Breakout | 0 | 33 | 0.8 | 22 |
| Freeway | 8 | 8 | 0 | 5 |
| CliffWalking | -555 | -60 | -575 | -13 |
| FrozenLake | 0.00 | 1.00 | 0.00 | 1.00 |
| Blackjack | -0.72 | -0.11 | -1.00 | -0.60 |
| MiniGrid empty | 0.89 | 0.96 | 0.00 | 0.96 |
| MiniGrid LavaGap | 0.00 | 0.09 | 0.00 | 0.19 |
| MiniGrid DoorKey | 0.10 | 0.00 | 0.00 | 0.00 |
| BabyAI GoTo | 0.59 | 0.45 | 0.12 | 0.34 |

Regression accuracy on the 95-task set after the overlay: in-task 0.855 (untouched 0.855), held-out 0.809 (untouched
0.811). The Pong test score is -12 over ten test seeds; the clip's seed 0 ends at -3 because the episode is cut at 1200
decisions. The two 2e-6 seeds disagree on which games they gain (the other seed reaches Breakout 15 and BabyAI 0.83), so no
overlay from this run has been merged or released; CliffWalking never leaves a -60 wall loop in any arm.

## `media/vision_2b.gif`

**What is shown.** `Mapika/decider-2b-vision` deciding on ten held-out images. Each panel has the image, the question, one row
per option with the served probability as a bar, the gold answer marked in green, and the model time measured for that image.

**Data.** The evaluation split of the local build of `HuggingFaceM4/the_cauldron` (`decider/vision/data.py`). Twelve rows
were drawn at random (`random.Random(20260922)`) from each of six subsets: aokvqa, visual7w, vsr, ai2d, scienceqa, iconqa.
72 rows were scored; the first is discarded as CUDA warm-up (2331 ms against a 59 ms median), leaving 71.

**Result.** Over the 71 scored rows the model picked the gold answer 87.3% of the time. The ten panels in the clip are 7
correct and 3 wrong.

**Selection.** Deliberately not the ten best. The rule: the most confident correct example from each of the six subsets, then
the two most confident wrong examples, then examples whose top probability is closest to 60%. Three panels show the model
confidently wrong.

**Latency.** Per-image model time is printed on each panel; the footer gives the median over the run, 59 ms. The timer wraps
`VisionDecisionModel.slot_logits` alone with `torch.cuda.synchronize()` on both sides, batch of one. Image decoding and
prompt building are outside it. Per-image time tracks the number of vision tokens, which is why it ranges from 24 ms to 98 ms.

**Time compression.** Not applicable: each panel is held for 2000 ms.

## `media/montage.gif`

Recorded in 2026-09 on the v8 weights: the ten text games from `decider/games/` and Super Mario Bros from the PPO checkpoint
of `decider/mario_rl.py`, one typed decision per move. See `docs/HISTORY.md`.
