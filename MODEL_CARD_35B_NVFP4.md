---
license: apache-2.0
base_model: Mapika/decider-35b-a3b
language: [en]
pipeline_tag: text-classification
tags: [decision-model, calibrated, structured-output, multi-task, system-one, one-pass, mixture-of-experts, nvfp4, modelopt, vllm]
---

# decider-35b-a3b-nvfp4: the 35B one-pass decision model in NVFP4 for vLLM and TensorRT-LLM

The weights of [Mapika/decider-35b-a3b](https://huggingface.co/Mapika/decider-35b-a3b) v1 quantized to NVFP4 with NVIDIA
ModelOpt 0.46.1: 4-bit floating-point weights and activations with FP8 block scales (block size 16) on the attention
projections, the shared experts and all 256 routed experts of every layer. The delta-net convolutions and input projections,
the routers, the shared-expert gates, the embeddings and the LM head stay in bf16, as ModelOpt's default NVFP4 recipe
leaves them. 19.6 GB against 65 GB for the bf16 checkpoint. The checkpoint is in the Hugging Face layout ModelOpt exports
(`hf_quant_config.json`, `quantization_config` in `config.json`) and is meant for vLLM, TensorRT-LLM and SGLang on Blackwell
GPUs. The bf16 repository is the reference; read its card for what the model is, how it was trained and how it was measured.

The `decider` package included here is the same inference package as in the bf16 repository, kept for the prompt
construction and the `system_one` request shape. `transformers` alone does not load this checkpoint's quantized weights; use
vLLM (or TensorRT-LLM) for the forward pass.

## Usage with vLLM

The decision readout is the same as in the bf16 model: the prompt ends at the answer slot `Answer: (`, the logits of the
option-label tokens at that position are softmaxed at the stored temperature (1.08). With vLLM, restrict sampling to the
label tokens and ask for their processed logits:

```python
from vllm import LLM, SamplingParams
from decider.infer import Example, Q                 # decider/ is included in this repo
from decider.prompt import build, label_table

llm = LLM(model="Mapika/decider-35b-a3b-nvfp4", max_model_len=34816, logprobs_mode="processed_logits", max_logprobs=256)
tok = llm.get_tokenizer(); _, label_ids, _ = label_table(tok)

class NoShuffle:                                      # keep option order as given
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]

options = ["billing", "technical support", "sales"]
item = build(Example("My card was charged twice for the same purchase.", [Q("Which department should handle this?", options, 0)], "infer"),
             tok, NoShuffle(), max_options=255, max_ctx_tokens=32768)
params = SamplingParams(max_tokens=1, logprobs=len(options), allowed_token_ids=label_ids[:len(options)])
out = llm.generate([dict(prompt_token_ids=item["ids"])], params)[0].outputs[0].logprobs[0]
import math
logits = [out[label_ids[j]].logprob for j in range(len(options))]
z = [math.exp((x - max(logits)) / 1.08) for x in logits]; probs = [x / sum(z) for x in z]     # P(billing), P(technical support), P(sales)
```

`moe/vllm_check.py` in the GitHub repository is this loop over whole fixtures, with the comparison against the bf16
predictions reported below. For the HTTP server and the TypeSafe request shape, run `decider.serve` against the bf16
weights, or adapt its `system_one` translation to the loop above.

## Accuracy against the bf16 weights

**Fake quantization in PyTorch** (ModelOpt's simulated NVFP4 on the bf16 model, before export): the same 150-row subset of
every regression task for both models, 95 tasks, state-first layout, temperature 1.

| | in-task acc / NLL / ECE (67 tasks) | held-out acc / NLL / ECE (28 tasks) |
|---|---|---|
| bf16 | 0.861 / 0.353 / 0.053 | 0.818 / 0.492 / 0.089 |
| NVFP4, fake quantization | 0.857 / 0.360 / 0.053 | 0.809 / 0.502 / 0.093 |

Accuracy moves by −0.4 points in-task and −0.9 held-out; NLL by +0.007 and +0.010 nats. Per task the mean change is −0.55
points (lower on 50 tasks, higher on 22, equal on 23); the largest drops are on the abstention probe (−6.0), StrategyQA
(−5.3), Social IQa (−4.7) and LIAR2 (−4.0), all on 150 rows, where a 5-point move is two to three standard errors.

**The exported checkpoint served by vLLM** on the same rows as the bf16 model's fixture predictions:

| vLLM 0.29, one B300, same rows | bf16 weights | NVFP4 weights |
|---|---|---|
| TypeSafe workflow decisions, 102 rows: accuracy / NLL | 0.853 / 0.345 | 0.843 / 0.365 |
| same rows, argmax agreement with the in-process bf16 predictions / mean total variation | 99.0% / 0.010 | 96.1% / 0.051 |
| 847 in-task validation rows: accuracy / NLL | 0.897 / 0.329 | 0.882 / 0.339 |
| same rows, agreement / mean total variation | 99.8% / 0.005 | 96.7% / 0.030 |
| rows per second, TypeSafe packets (3,650 tokens on average) | 36 | 51 |
| rows per second, validation rows (150 tokens on average) | 498 | 274 |

The in-process bf16 model scores 0.863 and 0.900 on these two sets; vLLM's bf16 path is within one row of it on each. The
NVFP4 weights lose 1.0 point on the TypeSafe rows and 1.5 points on the validation rows against bf16 in the same engine, change
the argmax on 3 to 4% of rows, and move the served distribution by 0.03 to 0.05 total variation. The loss is larger than the
fake-quant estimate on the regression subset (0.4 points). The throughput numbers come from single batches of a few seconds
each and are indicative only: NVFP4 is faster on long inputs and slower on short ones in this engine version.

Calibration used 512 training prompts of at most 2,048 tokens drawn at random from the public mixture. No evaluation row was
used for calibration or for any choice made here.

## What is quantized

| kept in bf16 | quantized to NVFP4 |
|---|---|
| `embed_tokens`, `lm_head` | attention `q_proj`, `k_proj`, `v_proj`, `o_proj` (10 full-attention layers) |
| `linear_attn.conv1d`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b` (30 delta-net layers) | delta-net `in_proj_qkv`, `in_proj_z`, `out_proj` |
| `mlp.gate` (router), `shared_expert_gate`, norms | `gate_proj`, `up_proj`, `down_proj` of each of the 256 routed experts and of the shared expert, every layer |

The KV cache is not quantized (`kv_cache_quant_algo: null`). The full module list is in `hf_quant_config.json`;
`quantization_report.json` holds the fake-quant per-task numbers.

## Limitations

Everything in the bf16 card applies: no RL stage, English only, overconfident on the hardest external items. In addition:
the quantization costs 1.0 to 1.5 accuracy points against bf16 on the two fixtures measured through vLLM and changes the
argmax on 3 to 4% of rows, more than the fake-quant estimate; the fake-quant measurement is on 150 rows per task, not the full
set, and its losses concentrate on a few reasoning and abstention tasks; the checkpoint was run only through vLLM 0.29 on a
B300 here, not through TensorRT-LLM or SGLang; and the OpenJev, Mind2Web, browser, game and JevBench numbers of the bf16 card
were not re-measured with these weights.
