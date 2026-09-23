# Demo clips

How `media/showcase.gif` in the README was made. Every tile is a recorded episode; nothing is re-run for the clip, nothing is
interpolated or re-ordered, and every probability drawn is the served value logged at that decision (the build script checks
at every decision shown that the most probable option is the logged action). The script is `make_marketing.py` in the
research repository; it runs on CPU from the logs.

**Layout.** Five tiles, each looping its own window on a 16 s loop (tile lengths divide the loop, so the GIF has no jump).
Each tile shows the game, the three most probable options of the decision on screen (the chosen one, always the most probable,
in green) and the decision time: the median of the per-decision times logged in that episode, first decision excluded as
warm-up. All five episodes were recorded on one NVIDIA B300 SXM6 in bf16, batch of one, on a machine shared with other jobs,
so the times move by up to about 1.5x between runs.

| tile | model | input | episode | window shown | decision time |
|---|---|---|---|---|---|
| Tetris | decider-2b v10 | text state + 8 shortlisted placements (see below) | NES Tetris, test seed 4: 125 pieces, 35 lines | the 36 pieces in which the line count rises most (+14), every 6th emulator frame | 12 ms (`Decider` with CUDA graphs, whole request) |
| Breakout | decider-2b-vision | the game frame only | ALE seed 0, 618 decisions, score 18 | the 300 decisions in which the score rises most (+12) | 23 ms |
| Pong | decider-35b-a3b + games-RL overlay (not released) | text state built from the console RAM | ALE seed 0, 1200-decision cap; the episode is lost 3-6 | the rally ending in the model's first point (decisions 173-312) | 43 ms (eager) |
| Snake | decider-35b-a3b | text state (10x10 board) | seed 0, 21 food, ends at turn 231 by running into itself | food 7 to 21 (decisions 61-210) | 47 ms (eager) |
| Connect Four | decider-35b-a3b | text state listing immediate threats | seed 2 against an opponent that wins if it can, else blocks, else plays at random; game 3 of 10 | the model's first won game, moves 35-44 and the final board | 47 ms (eager) |

**Pictures.** Tetris, Breakout and Pong are the emulator's own frames, scaled 2x with nearest neighbour; the Tetris frame is
cropped to the well and the LINES, SCORE, NEXT and LEVEL boxes. Snake and Connect Four were recorded as light-coloured
drawings; for the dark page the boards are redrawn from the board in the logged state text, cell for cell (the build checks
every decision). The final Connect Four board, after the winning drop, is not a logged decision and is rebuilt by replaying
the logged moves in the game environment. On-screen Tetris option labels are shortened to piece and columns, plus "clears N"
from the lines-cleared fact the model saw.

## Tetris: shortlist harness

The model does not read the board and choose among all legal placements; with that setup (17 to 34 options per piece) every
model clears 0 lines. The harness follows the one the open jev-tetris project uses for Jev and Laya: every legal placement is
ranked by the standard four-feature heuristic (aggregate height, lines cleared, holes, bumpiness), the top 8 are offered, each
option states its measured consequences (lines cleared, holes, bumpiness, total height, and the board after it lands), and the
instructions say which direction of each fact is better. The model chooses one of the 8. Options are listed in placement
order, not rank order. NES Tetris, 5 test seeds, 300-piece cap, rule fixed before the test seeds were run:

| player | lines per seed | mean lines | mean pieces | median ms |
|---|---|---|---|---|
| decider-2b v10 | 25, 16, 11, 14, 35 | 20.2 | 85 | 12.2 |
| decider-4b v1 | 8, 5, 11, 4, 11 | 7.8 | 54 | 20.5 |
| decider-35b-a3b v1 | 10, 13, 11, 18, 14 | 13.2 | 65 | 87.2 |
| heuristic's first choice | 117, 118, 115, 91, 65 | 101.2 | 274 | |
| random pick from the 8 | 0, 1, 1, 1, 0 | 0.6 | 30 | |
| always the lowest-ranked of the 8 | 0, 0, 0, 0, 0 | 0.0 | 18 | |

Every model beats a random pick from the same shortlist by more than two standard errors, so the choice within the 8 is the
model's and it matters; none approaches the heuristic's own first choice, and every model tops out between 54 and 125
pieces. Latency is one full request (batch of one, about 1,300 prompt tokens), CUDA graphs for the 2B and 4B, eager for the
35B.

## Snake and Connect Four results

| game | player | per seed | mean | rule-based player |
|---|---|---|---|---|
| Snake, food eaten | decider-35b-a3b | 21, 20, 21 | 20.7 | 16.9 (greedy toward the food, avoiding walls and body) |
| Connect Four vs win-or-block, score rate over 10 games | decider-35b-a3b | 0.25, 0.65, 0.40 | 0.43 | 0.51 (the same win-or-block rule) |
| Connect Four vs win-or-block | decider-4b v1 | 0.10, 0.20, 0.10, 0.20, 0.10 | 0.14 | |

Random play scores 0.4 food at Snake and 0.05 at Connect Four. decider-2b does not beat random play at either game.

## The games-RL run behind the Pong tile

Outcome-reward RL of decider-35b-a3b on ten text games (Pong, Breakout, Freeway, CliffWalking, FrozenLake, Blackjack, three
MiniGrid tasks, BabyAI GoTo), 2026-09-20 to 2026-09-21. The reward is the game's own score change plus a bonus when an
episode ends in success; no scripted teacher and no gold labels are used. PPO with clip 0.2, GAE against a value head that
reads the slot hidden state, per-game sample quotas, and a retention gate: after every step the KL to the untouched model is
measured on a fixed pool of rows from the supervised training mixture, and a step that moves the mean above 0.01 nats or any slot above 0.05
nats is undone. Four arms (learning rates 1e-6 and 2e-6, two seeds each); the checkpoint per arm is chosen on a validation
seed set and reported on a fresh test seed set.

Greedy test scores of the arm used in the Pong tile (2e-6, seed 1, iteration 20), against the untouched model:

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
0.811). The Pong test score is -12 over ten test seeds; seed 0, the tile's episode, ends at -3 because it is cut at 1200
decisions. The two 2e-6 seeds disagree on which games they gain (the other seed reaches Breakout 15 and BabyAI 0.83), so no
overlay from this run has been merged or released; CliffWalking never leaves a -60 wall loop in any arm.

## Image questions (no clip)

The earlier vision clips were retired on 2026-09-23; their measurements stand. On 71 held-out rows of the evaluation split of
`HuggingFaceM4/the_cauldron` (twelve random rows from each of aokvqa, visual7w, vsr, ai2d, scienceqa and iconqa, the first
discarded as warm-up), `Mapika/decider-2b-vision` picked the gold answer 87.3% of the time (median 59 ms per image). The
text-only `Mapika/decider-35b-a3b` weights loaded onto the vision-language version of their base, with no image training,
picked it 91.5% of the time (65 of 71; median 85 ms per image). On a larger probe (300 rows per subset) the two are at 0.871
and 0.880 in-task, so the 4-point gap on 71 rows is within the sample noise: the text-trained weights answer image questions
at the level of the purpose-trained vision model.
