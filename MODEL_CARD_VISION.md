---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
language: [en]
pipeline_tag: image-text-to-text
tags: [decision-model, calibrated, structured-output, vision, one-pass]
---

# decider-2b-vision: typed decisions from an image in one forward pass

The vision variant of [decider-2b](https://huggingface.co/Mapika/decider-2b) (v5 language
weights transplanted into the full Qwen3.5-2B vision-language model), fine-tuned so that an
image (a photo, a diagram, a game frame) plus a text question with lettered options yields a
calibrated probability over the options at a single answer slot. No generation. A 256x240 game
frame costs 64 visual tokens. Text-only questions work too, with decider-2b v5's behaviour,
including its abstention handling.

**Contents:** [The decider family](#the-decider-family) · [Usage](#usage) · [Training](#training) · [Results](#results) · [Limitations](#limitations) · [Changelog](#changelog)

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
import torch
from decider.vision import VisionDecisionModel
from decider.infer import Example, Q
m = VisionDecisionModel("<this repo>", grad_ckpt=False).cuda().eval()
ex = Example("This is a visual question about the image.",
             [Q("What is the person holding?", ["a phone", "a cup", "a book", "nothing"], 0)])
inp = m.prepare([(image, ex)])                  # image: PIL image, numpy array, or PNG bytes; None for text-only
probs = torch.softmax(m.slot_logits(inp), -1)[0, :4]
```

## Training

One epoch (80k examples, 50k with images): game frames from Pong, Breakout, CliffWalking,
MiniGrid and Super Mario Bros labelled by scripted policies (rare actions oversampled, plus
DAgger frames from an earlier model's own play); multiple-choice image tasks from The Cauldron
(A-OKVQA, AI2D, ScienceQA, IconQA, TQA, Raven, Hateful Memes); a replay of the text mixture.
Then PPO from pixels on Breakout and Pong (the softmax over action options is the policy).
Code: https://github.com/Mapika/decider (`decider/vision/`).

## Results

300 items per task.

| task | accuracy | ECE |
|---|---|---|
| Pong frames (agreement with the RAM-state teacher) | 0.96 | 0.02 |
| Breakout frames | 0.96 | 0.02 |
| Visual7W (held out) | 0.89 | 0.03 |
| A-OKVQA / AI2D / ScienceQA / IconQA / Raven / Hateful Memes | 0.85 / 0.93 / 0.95 / 0.94 / 0.80 / 0.80 | 0.02 to 0.07 |

Playing from pixels only (no text state), three episodes each: Breakout 41 (the RAM-state
teacher scores 22), Pong 3 (teacher 8), CliffWalking -13 (optimal), MiniGrid Empty 0.96
(teacher level); held-out Freeway 0, FrozenLake 0, the harder grid worlds 0 (their scripted
teachers also score 0), Mario 1-1 315 px. The previous vision release (v4-based) scored
Breakout 16, Pong 8, Freeway 8, BabyAI-GoTo 0.30; this one trades Pong and Freeway for
Breakout and for the corrected abstention behaviour.

## Limitations

Not a chat model and not a captioner: it answers lettered options at one slot. The text weights inside are decider-2b **v5**,
so the text-only behaviour is that of v5 (its abstention handling, none of the v6 to v10 input shapes, calibration or browser
results); a retrain on the current text weights has not been released. Game play from pixels is measured on the Atari, MiniGrid
and Mario frames it was trained on plus a few held-out games, three episodes each. English only; calibration is measured on the
listed datasets, not on your images.

## Changelog

| version | what changed |
|---|---|
| **current weights** | v5 text weights transplanted into the Qwen3.5-2B vision-language model, one epoch on game frames, The Cauldron multiple-choice tasks and a text replay, then PPO from pixels on Breakout and Pong |
| previous release | v4-based: Breakout 16, Pong 8, Freeway 8, BabyAI-GoTo 0.30 |

Every decider release is listed in [docs/CHANGELOG.md](https://github.com/Mapika/decider/blob/main/docs/CHANGELOG.md) of the
GitHub repository.
