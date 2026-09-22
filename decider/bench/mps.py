"""Offline full-model MPS check.

Usage: python -m decider.bench.mps MODEL reference|conv|optimized

Runs one mode per process without concurrent GPU work. JSON includes per-request
probabilities and synchronized timings; model loading is excluded. ``reference``
is the pure Transformers PyTorch path, ``conv`` adds only Decider's fused
causal convolution, and ``optimized`` adds the MPS attention patch as well.
"""
import inspect
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download

from decider import mps_ops
from decider.engine import patch_conv
from decider.infer import Decider, Example, Q


def configure(mode):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    if mode not in ("reference", "conv", "optimized"):
        raise ValueError("mode must be reference, conv, or optimized")
    # Avoid FLA/Triton in the reference arm: this is the pure Transformers path.
    mq.torch_chunk_gated_delta_rule = inspect.unwrap(mq.torch_chunk_gated_delta_rule)
    mq.causal_conv1d_fn = inspect.unwrap(mq.causal_conv1d_fn)
    if mode == "reference":
        mps_ops.patch_mps = lambda: False
        return False, False
    if mode == "conv":
        mps_ops.patch_mps = lambda: False
        patch_conv()
        return False, True
    return mps_ops.patch_mps(), True


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    name, mode = sys.argv[1:]
    assert torch.backends.mps.is_available(), "MPS required"
    path = snapshot_download(name, local_files_only=True)
    patch_result, conv_result = configure(mode)
    if name.endswith("vision"):
        from PIL import Image
        from decider.vision.model import VisionDecisionModel

        model = VisionDecisionModel(path, grad_ckpt=False).to("mps").eval()
        cases = [(Image.new("RGB", (224, 224), color), Example(
            "Identify the dominant color in the image.",
            [Q("What color is shown?", ["red", "green", "blue"])]))
            for color in ("red", "green", "blue")]

        def run(case):
            logits = model.slot_logits(model.prepare([case]))
            return torch.softmax(logits, -1)[0, :3].cpu().tolist()
    else:
        model = Decider(path, device="mps", use_graphs=False)
        cases = ["My card was charged twice for the same purchase.",
                 "I cannot log in after resetting my password.",
                 "I would like pricing for fifty licenses."]

        def run(case):
            return model.decide(case, [{"question": "Which department should handle this?",
                "options": ["billing", "technical", "sales"]}])[0]["probs_list"]

    results = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            for _ in range(2):
                run(case)
                torch.mps.synchronize()
            times = []
            for _ in range(5):
                torch.mps.synchronize()
                start = time.perf_counter()
                probs = run(case)
                torch.mps.synchronize()
                times.append((time.perf_counter() - start) * 1000)
            assert all(torch.isfinite(torch.tensor(probs)))
            results.append(dict(case=index, probs=probs, times_ms=times,
                                median_ms=statistics.median(times)))
    weights = model.lm if name.endswith("vision") else model.m.lm
    print(json.dumps({
        "model": name,
        "snapshot_revision": Path(path).name,
        "mode": mode,
        "patch_mps": patch_result,
        "patch_conv": conv_result,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": str(next(weights.parameters()).dtype),
        "hardware": platform.machine(),
        "macOS": platform.mac_ver()[0],
        "warmups": 2,
        "measurements_per_case": 5,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
