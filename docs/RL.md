# The calibration-aware RL stage (v8 to v10)

v10 is the v8 weights continued for 384 optimizer steps with rewards that come only from outcomes. This document states the
recipe, the gates it had to pass, what it changed and what it did not. The training code lives in a separate research
repository (it needs a live Chrome with MiniWoB++, exact game environments and four GPUs for the parallel arms); the readout it
trains is the one in `decider/prompt.py` and `decider/model.py`, vendored unchanged, so a checkpoint loads in this package
without any conversion.

## What is trained

The served distribution itself: one forward pass, letter logits at the answer slot, options in the given order, softmax of the
logits divided by the stored temperature 1.30. Every loss term below is computed on that distribution, not on a separate policy
head, and rollouts score one request per forward so that the PPO ratio at the start of training is exactly 1.

## Environments and rewards

| environment | per iteration | outcome reward | belief question |
|---|---|---|---|
| live MiniWoB++ click tasks, 16 training tasks | 4 tasks x 4 repeats, up to 12 clicks | +1 if the task's own checker reports success, −1 otherwise | what the click will do: success, failure, or the episode continues |
| exact 4x4 minesweeper | 4 boards x 4 repeats | +1 win, −1 loss | for every hidden cell of the visited state: mine or safe, against the exact posterior |
| 5x5 grid with a slippery move | 2 boards x 4 repeats | goal reached | where the sampled move lands, against the transition law |
| draws from bags of known composition | 2 boards x 4 repeats | game won | what the next draw will be, against the bag's composition |

The 22 browser tasks are MiniWoB++'s click tasks; the elements on the page are listed as text and are the options. Six of them
(click-checkboxes-transfer, click-tab-2-hard, click-collapsible-2, click-dialog-2, focus-text-2, navigate-tree) were never
rewarded and only validated. The MiniWoB++ episode timer is disabled after every reset, because the model's forward passes
would otherwise time out the ten-second episodes.

No gold labels enter at any point. The two external fixtures used for reporting (102 TypeSafe workflow rows, the 5,252-row
OpenJev fixture) never enter reward, gate or checkpoint selection.

## Loss terms

1. **Actor**, coefficient 0.1. PPO clipped surrogate (clip 0.2) on the served probability of the sampled action, with the
   terminal return and a leave-one-replicate-out baseline over the 4 repeats of the same board or page.
2. **Belief**, coefficient 0.2. The model is asked, in the same one-pass format, what will happen next (the belief question in
   the table), and its answer is scored with the log score against the exact law. Options are read in two orders and the
   letter logits averaged, so the belief does not depend on option order. This term is a proper scoring rule: it is minimised
   only by stating the true probabilities.
3. **Rendering consistency**, coefficient 1.0. On replayed training rows with at most 10 options and renderings under 1,024
   tokens, KL from the served answer (state-first layout, given option order, detached) to the answer in each of three other
   renderings: reversed order, schema-first layout, schema-first reversed. It keeps the two layouts and the two orders in
   agreement while the weights move.
4. **Retention**, coefficient 0.7. On 8 replayed supervised rows per step, KL(v8 ‖ student) on the served distribution. If the
   mean over the 8 rows exceeds 0.01 nats or any row exceeds 0.05 nats, the step drops terms 1 and 2, zeroes the optimizer's
   first moment and applies only this term. 94 of the 576 steps were handled this way.

Optimizer: AdamW with FP32 master weights over BF16 weights, betas 0.9 / 0.95, weight decay 0.01, peak learning rate 1e-6 with
16 warm-up steps and a cosine schedule over 576 steps, gradient norm clipped at 1.0. Each iteration produces 96 sampled
transitions, trained in 12 minibatches of 8.

## Gates and selection

Checkpoints were saved at steps 192, 384 and 576 and validated on fixed seeds. A checkpoint is eligible only if all six hold,
each against the v8 starting point:

| gate | condition |
|---|---|
| sampled win | weighted win rate (0.5 browser rewarded tasks, 0.25 minesweeper, 0.125 grid, 0.125 bags) at least 0.02 above v8 |
| greedy win | same weighting, at least equal to v8 |
| belief excess | mean log-score gap to the exact law at least 0.02 nats below v8 |
| general NLL | macro NLL over 847 supervised validation rows at most 0.005 above v8 |
| general accuracy | at most 0.002 below v8 |
| rendering consistency | argmax unanimity across the four renderings on 256 rows at most 0.01 below v8 |

Among eligible checkpoints the one with the greatest sampled win is selected. Four arms ran in parallel (two seeds, two KL
budgets); every arm produced an eligible checkpoint. The released v10 is seed 9419, budget 0.01 / 0.05, step 384.

## What it changed

Measured on the same rows and seeds as v8. Intervals are 95%.

| | v8 | v10 | difference |
|---|---|---|---|
| live browser tasks, 22 x 8 seeds, sampled play | 83.0% | 93.2% | +10.2 (+5.1 to +15.9) |
| the 6 never-rewarded tasks | 72.9% | 91.7% | +18.8 (+6.2 to +31.2) |
| greedy play | 90.3% | 90.9% | +0.6 |
| belief excess over the exact laws (nats) | 0.473 | 0.219 | |
| click-outcome log score | −0.349 | −0.034 | |
| bag-draw games, win rate | | | +6.2 (+0.8 to +12.1) |
| Mind2Web element and action choice, 1,770 rows | 81.1% | 82.7% | +1.5 (+0.7 to +2.4) |
| TypeSafe workflow decisions, 102 rows | 78.4% | 80.4% | +2.0 (−2.0 to +5.9) |
| 847 in-task validation rows | 83.6% | 83.2% | −0.4 (−1.3 to +0.6) |
| Bespoke's public suite, macro | 0.706 | 0.704 | |
| OpenJev, 5,252 rows | 64.1% | 63.3% | −0.8 (−1.3 to −0.3) |

Recordings of both versions on the same pages are in `media/v10_browser_*.gif`; the figures `media/v10_vs_v8.png`,
`media/v10_calibration.png` and `media/v10_browser_tasks.png` plot the table above and the per-task browser results.

## What it did not change

Tic-tac-toe, grid and minesweeper win rates (a 2B model without search loses most of these games before and after), general
accuracy on the training tasks, calibration on Bespoke's suite, the architecture, the readout, the temperature and the speed.
The v9 terse-bucket data is not in v10, because the RL stage started from the v8 weights that were on the Hub.

## What was learned on the way

Two earlier attempts on these weights are the reason for three details above. At a peak learning rate of 2e-6 the model
drifted on short replayed rows from step 384 on (KL up to 5.8 nats on single rows, confident wrong answers) and no checkpoint
passed every gate; halving the rate removed it. The rendering-consistency term, first run in one arm only, halved that drift,
so it is in the recipe. An equal-weight win gate over the four environments could not be reached, because a 2B model's grid and
bag play does not move; the gate was re-weighted toward the browser, where the policy does move, before the run that produced
v10, and the equal-weight numbers were recorded but not gated.
