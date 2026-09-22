"""The engine must turn the cuDNN SDPA backend off before any forward: on Blackwell it returns wrong output for the cached
prefix + suffix attention shapes that `Engine.score_shared` and the schema cache use (docs/CHANGELOG.md, 1.0.2)."""
import pytest

torch = pytest.importorskip("torch")

from decider.engine import set_attention_backend_policy


def test_policy_disables_cudnn_sdp():
    if not hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        pytest.skip("this torch build has no cuDNN SDPA switch")
    torch.backends.cuda.enable_cudnn_sdp(True)
    set_attention_backend_policy()
    assert torch.backends.cuda.cudnn_sdp_enabled() is False
    # the other backends stay available
    assert torch.backends.cuda.math_sdp_enabled() is True
