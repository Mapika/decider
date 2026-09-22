"""Issue #5: the server picks its device like decider.infer.Decider and fails with a message that names the requirement."""
import pytest
import torch

from decider import serve


def _avail(monkeypatch, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize("cuda,mps,want", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")])
def test_auto_follows_decider(monkeypatch, cuda, mps, want):
    _avail(monkeypatch, cuda, mps)
    dev, dtype = serve.resolve_device("auto")
    assert dev == want
    assert dtype == (torch.float16 if want == "mps" else torch.bfloat16)


def test_explicit_cpu_on_a_cuda_machine(monkeypatch):
    _avail(monkeypatch, True, False)
    assert serve.resolve_device("cpu") == ("cpu", torch.bfloat16)


@pytest.mark.parametrize("req,msg", [("cuda", "torch.cuda.is_available"), ("cuda:1", "torch.cuda.is_available"),
                                     ("mps", "torch.backends.mps.is_available"), ("tpu", "expected auto")])
def test_unavailable_device_is_a_clear_error(monkeypatch, req, msg):
    _avail(monkeypatch, False, False)
    with pytest.raises(RuntimeError, match=msg):
        serve.resolve_device(req)


@pytest.mark.parametrize("flag", ["FP8", "COMPILE"])
def test_cuda_only_options_off_cuda(monkeypatch, flag):
    _avail(monkeypatch, False, False)
    monkeypatch.setattr(serve, flag, True)
    with pytest.raises(RuntimeError, match="need CUDA"):
        serve.resolve_device("auto")
