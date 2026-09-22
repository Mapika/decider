"""Shared fixtures.

`tok`: a real Qwen tokenizer.  DECIDER_TEST_TOKENIZER names a local model folder or Hub id; otherwise Qwen/Qwen3.5-2B-Base
is loaded (from the Hub cache when offline).  Tests that need it skip when none is available.

`cuda` marker: tests that load a real checkpoint on a GPU.  They skip without CUDA.  DECIDER_TEST_MODEL names the checkpoint
(default Mapika/decider-2b).
"""
import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a CUDA device and a real checkpoint (DECIDER_TEST_MODEL, default Mapika/decider-2b)")


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
    return os.environ.get("DECIDER_TEST_MODEL", "Mapika/decider-2b")
