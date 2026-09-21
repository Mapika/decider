# Held-out MPS evaluation

This is the requested `decider.evaluate.py` check, not a smoke-test prompt.
It evaluates all 1,500 examples in the held-out `massive_scenario` test set from
the repository's data loader.

| Model/path | Examples | Temperature | Accuracy | ECE | NLL |
|---|---:|---:|---:|---:|---:|
| Published decider-2b v10 BF16, rebuilt-set reference | 1,500-task row | 1.30 | 0.756 | 0.041 | — |
| This PR, decider-2b FP16 on MPS | 1,500 | 1.30 | 0.7553 | 0.0438 | 0.6980 |

The published per-task table reports accuracy/ECE, not per-task NLL. Its MASSIVE
Scenario row is `0.756 / 0.041`; the aggregate held-out NLL elsewhere in the model
card is not comparable to this single-task NLL.

The MPS result was produced by `python -m decider.evaluate` with `--device mps`,
`--temperature 1.30`, batch size 8, and max context 1536. It used the cached
`Mapika/decider-2b` snapshot `b37f7e1ba3fbc9238004cf531fabbee2619973fd`, PyTorch
2.14.0, Transformers 5.17.0, and float16 weights. The run took 210.7 seconds;
model loading is excluded from the reported task timer.

This is one held-out dataset, not the aggregate 24-task published score. It shows
that the MPS path's accuracy and calibration are close to the published BF16 row
on this set; it does not establish long-context behavior or the aggregate held-out
score. The raw `evaluate.py` output is recorded in
[`mps-heldout.json`](mps-heldout.json).

## Reproduce

The repository does not commit generated dataset caches. Build the same one-task
cache and run the evaluator on an Apple Silicon MPS machine:

```sh
python -m decider.data.core massive_scenario --out /tmp/massive_scenario.pkl
python -m decider.evaluate \
  --model Mapika/decider-2b \
  --data /tmp/massive_scenario.pkl \
  --out /tmp/mps-eval \
  --tasks massive_scenario \
  --device mps \
  --temperature 1.30 \
  --bs 8 \
  --max_ctx 1536
```
