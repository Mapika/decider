"""Shared fixtures.

`tok`: a real Qwen tokenizer.  DECIDER_TEST_TOKENIZER names a local model folder or Hub id; otherwise Qwen/Qwen3.5-2B-Base
is loaded (from the Hub cache when offline).  Tests that need it skip when none is available.

`cuda` marker: tests that load a real checkpoint on a GPU.  They skip without CUDA.  DECIDER_TEST_MODEL names the checkpoint
(a local folder or Hub id); the default is Mapika/decider-2b at the Hub tag v10.  The tolerances in test_engine_v2_cuda.py were
measured on decider-2b v10.  The chunked shared-prefix fork is bit-identical to the single fork on v10 but not on every
checkpoint: on decider-2b v11 it differs by up to 3.3e-5 in probability with equal suffixes and 4.4e-4 with mixed ones, on
decider-4b v1 by 4.8e-6 (argmax unchanged), so up to three of those tests fail on such a checkpoint without a fault in
the code.
"""
import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a CUDA device and a real checkpoint (DECIDER_TEST_MODEL, default Mapika/decider-2b tag v10)")


@pytest.fixture(scope="session")
def tok():
    transformers = pytest.importorskip("transformers")
    for name in (os.environ.get("DECIDER_TEST_TOKENIZER", ""), "Qwen/Qwen3.5-2B-Base"):
        if not name:
            continue
        try:
            return transformers.AutoTokenizer.from_pretrained(name)
        except Exception:
            continue
    pytest.skip("no tokenizer available (set DECIDER_TEST_TOKENIZER or allow the Hub download)")


@pytest.fixture(scope="session")
def cuda_model():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if os.environ.get("DECIDER_TEST_MODEL"):
        return os.environ["DECIDER_TEST_MODEL"]
    from huggingface_hub import snapshot_download
    return snapshot_download("Mapika/decider-2b", revision="v10")
