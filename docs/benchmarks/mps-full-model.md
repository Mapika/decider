# Full-model MPS measurements

Apple M1 Pro, 32 GB; macOS 27.2; PyTorch 2.14.0; Transformers 5.17.0.
All arms use float16 weights on MPS. MLX/Metal was available for the optimized path.

Each row is the median of five synchronized complete request measurements after two
warmups. Model loading is excluded. Text timing includes `Decider.decide()`; vision
timing includes image preparation, `slot_logits()`, softmax, and transfer of
probabilities to CPU. Each arm runs in a separate process, sequentially, with
identical locally cached checkpoints and inputs. These are nine specific smoke-test
workloads, not a representative accuracy suite.

The **reference** arm uses Transformers' pure-PyTorch gated-delta and convolution
implementations. In the evidence environment, `flash-linear-attention` 0.5.2 was
installed, but its `fla.ops` import required unavailable Triton support on macOS, so
Transformers selected its pure-PyTorch fallback and the unpatched MPS path ran. The
FLA/Triton implementation itself was not tested. The **conv-only** arm adds Decider's existing fused causal
convolution. The **optimized** arm adds the MPS gated-delta patch as well. This
separates the attention contribution from the already-existing convolution patch.

| Model | Input | Reference ms | Conv-only ms | Optimized ms | Optimized speedup | Maximum probability difference |
|---|---|---:|---:|---:|---:|---:|
| 0.8B | Billing request | 143.51 | 149.08 | 145.84 | 0.98× | 0.0001263 |
| 0.8B | Login request | 142.26 | 161.34 | 98.51 | 1.44× | 0.0000592 |
| 0.8B | Sales request | 161.91 | 146.99 | 99.56 | 1.63× | 0.0018024 |
| 2B | Billing request | 170.98 | 172.89 | 132.96 | 1.29× | 0.0000422 |
| 2B | Login request | 170.87 | 173.48 | 129.87 | 1.32× | 0.0000159 |
| 2B | Sales request | 171.94 | 171.85 | 135.48 | 1.27× | 0.0000672 |
| 2B Vision | Red image | 386.55 | 406.95 | 277.35 | 1.39× | 0 |
| 2B Vision | Green image | 385.65 | 395.42 | 276.65 | 1.39× | 0.0000015 |
| 2B Vision | Blue image | 388.82 | 393.24 | 277.48 | 1.40× | 0 |

All nine top-ranked answers matched. The largest probability change was 0.0018024
(0.18024 percentage points) on the 0.8B sales request. This exceeds a provisional
0.001 absolute probability-difference check; exact full-model numerical parity is
not claimed. Close decision boundaries may be sensitive to numerical differences.
Vision inputs are synthetic 224×224 solid-color images, not real-world vision data.
The separate [held-out evaluation](mps-heldout.md) covers 1,500 MASSIVE Scenario
examples; no aggregate held-out score or long-context evaluation was run here.

## Reproduce

Install the project and optional `metal` dependencies, and cache the three model
checkpoints. Run without other GPU work:

```sh
for model in decider-0.8b decider-2b decider-2b-vision; do
  for mode in reference conv optimized; do
    HF_HUB_OFFLINE=1 python -m decider.bench.mps "Mapika/$model" "$mode" > "$model-$mode.json"
  done
done
```

See [raw measurements](mps-full-model.json) for all timings, probabilities,
checkpoint revisions, dtype, patch return values, and software versions. The
runnable input definitions are in `decider/bench/mps.py`.
The optional Metal path accelerates part of the PyTorch computation; this is not
an all-MLX model. Full-model speedups include both attention and convolution changes.
