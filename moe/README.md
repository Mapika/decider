# decider-35b-a3b training scripts

The scripts that produced [Mapika/decider-35b-a3b](https://huggingface.co/Mapika/decider-35b-a3b) from Qwen3.5-35B-A3B-Base
(`docs/HISTORY.md`, section "decider-35b-a3b"). They are research scripts kept as run, with paths relative to this repository;
`python -m decider.data` and `scripts/train.sh` are the maintained entry points for the dense models.

| file | what it does |
|---|---|
| `prepare_items.py` | tokenizes the public mixture once with the `scripts/train.sh full` settings so every run reads byte-identical items in the same order |
| `optim.py` | `MasterAdamW` and `MasterMuon` (FP32 master weights, bf16 parameters; Muon with Newton-Schulz orthogonalization and RMS matching to AdamW's update size) and the rule that assigns parameters to Muon |
| `model.py` | `MoEDecisionModel`: the public `DecisionModel` loaded with grouped-GEMM experts and non-reentrant gradient checkpointing |
| `train_ab.py` | one data-parallel training arm (`--optimizer adamw|muon`), routed experts frozen, saves the trainable parameters as an overlay at the marks |
| `eval_overlay.py` | base weights plus an overlay, evaluated on the public regression set |
| `compare.py` | training curves and regression metrics at a fitted temperature for every mark |
| `ptq_nvfp4.py` | NVFP4 post-training quantization of the merged checkpoint with NVIDIA ModelOpt, fake-quant accuracy against bf16, HF export |

Run from the repository root with `PYTHONPATH=.`; the training arm is launched with `torchrun --nproc_per_node 4 moe/train_ab.py
--optimizer muon --out runs/moe_optimizer_ab_v1/muon --max_tokens 8192`. Merging an overlay into a standalone checkpoint is
`load_state_dict(overlay, strict=False)` on the base model followed by `save_pretrained`, plus `experts_implementation:
grouped_mm` in `config.json` and a `decider_config.json` with the fitted temperature. On this box NCCL needed
`NCCL_NVLS_ENABLE=0`.
