import inspect

import pytest
torch = pytest.importorskip("torch")
F = pytest.importorskip("torch.nn.functional")
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as _dispatch_delta_rule

# Transformers decorates this function to prefer FLA when installed. Unwrap it
# so these tests always compare with the pure-PyTorch reference implementation.
torch_chunk_gated_delta_rule = inspect.unwrap(_dispatch_delta_rule)
from decider.mps_ops import mps_chunk_gated_delta_rule, fast_invert_unitriangular_64, patch_mps


@pytest.mark.parametrize("disable_metal", [False, True])
def test_unitriangular_inverse(monkeypatch, disable_metal):
    """Check both optional Metal and PyTorch fallback inversion."""
    if disable_metal:
        monkeypatch.setattr("decider.mps_ops._metal_invert_kernel", None)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    L = torch.randn(16, 64, 64, device=device).tril(-1) * 0.1
    inv = fast_invert_unitriangular_64(L)
    I = torch.eye(64, device=device).expand_as(L)
    A = I + L
    prod = A @ inv
    err = (prod - I).abs().max().item()
    assert err < 1e-5, f"Inversion error too high: {err}"
    print(f"[PASS] test_unitriangular_inverse max err: {err:.2e}")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_metal_failure_falls_back(monkeypatch):
    import decider.mps_ops as ops

    class BrokenKernel:
        def __call__(self, **kwargs):
            raise RuntimeError("test Metal failure")

    monkeypatch.setattr(ops, "_metal_invert_kernel", BrokenKernel())
    monkeypatch.setattr(ops, "_metal_failure_reported", False)
    L = torch.randn(1, 64, 64, device="mps", dtype=torch.float32).tril(-1) * 0.1
    inv = ops.fast_invert_unitriangular_64(L)
    I = torch.eye(64, device="mps").expand_as(L)
    assert torch.allclose((I + L) @ inv, I, atol=1e-5, rtol=1e-5)
    assert ops._metal_invert_kernel is None


def test_delta_rule_shapes():
    """Verify numerical correctness against reference for B in (1, 2), S in (64, 512, 1024, 1596)."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    shapes = [64, 512, 1024, 1596]
    H, D = 16, 128

    for B in [1, 2]:
        for S in shapes:
            torch.manual_seed(S + B * 1000)
            q = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
            k = torch.randn(B, S, H, D, device=device, dtype=torch.float32) / (D ** 0.5)
            v = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
            g = -torch.rand(B, S, H, device=device, dtype=torch.float32)
            beta = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=torch.float32))

            ref_out, ref_state = torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
            opt_out, opt_state = mps_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)

            max_diff = (ref_out - opt_out).abs().max().item()
            cos_sim = F.cosine_similarity(ref_out.flatten(), opt_out.flatten(), dim=0).item()
            state_diff = (ref_state - opt_state).abs().max().item()

            assert torch.allclose(ref_out, opt_out, atol=1e-3, rtol=1e-3), f"Failed for B={B}, S={S}, max_diff={max_diff}"
            assert torch.allclose(ref_state, opt_state, atol=1e-3, rtol=1e-3), f"State diff failed for B={B}, S={S}: {state_diff}"
            print(f"[PASS] B={B} S={S:4d}: max_diff={max_diff:.2e}, cos_sim={cos_sim:.8f}, state_diff={state_diff:.2e}")


def test_patch_dispatch_and_guards(monkeypatch):
    """Exercise dispatch without requiring MPS hardware or leaving global patches behind."""
    from types import SimpleNamespace
    import transformers
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    import decider.mps_ops as ops
    monkeypatch.setattr(transformers, "__version__", "5.17.0")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert not patch_mps()
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    def reference_delta(query, key=None, value=None, g=None, beta=None, chunk_size=64, **kwargs):
        return "reference delta"

    def reference_conv(hidden_states, *args, **kwargs):
        return "reference conv"

    monkeypatch.setattr(mq, "torch_chunk_gated_delta_rule", reference_delta)
    monkeypatch.setattr(mq, "causal_conv1d_fn", reference_conv)
    monkeypatch.setattr(ops, "mps_chunk_gated_delta_rule", lambda *a, **kw: "MPS delta")
    monkeypatch.setattr(ops, "fused_causal_conv1d_fn", lambda *a, **kw: "MPS conv")
    assert patch_mps()
    patched = mq.torch_chunk_gated_delta_rule
    assert patch_mps() and mq.torch_chunk_gated_delta_rule is patched
    for is_mps in (False, True):
        tensor = SimpleNamespace(is_mps=is_mps)
        prefix = "MPS" if is_mps else "reference"
        assert patched(query=tensor) == prefix + " delta"
        assert mq.causal_conv1d_fn(tensor) == prefix + " conv"
        assert patched(tensor, chunk_size=32) == "reference delta"
    monkeypatch.setattr(transformers, "__version__", "5.16.0")
    with pytest.warns(RuntimeWarning, match="requires Transformers"):
        assert not patch_mps()


def test_l2norm_option():
    """Verify use_qk_l2norm_in_kernel branch."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    B, S, H, D = 1, 128, 16, 128
    torch.manual_seed(123)
    q = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    k = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    v = torch.randn(B, S, H, D, device=device, dtype=torch.float32)
    g = -torch.rand(B, S, H, device=device, dtype=torch.float32)
    beta = torch.sigmoid(torch.randn(B, S, H, device=device, dtype=torch.float32))

    ref_out, _ = torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    opt_out, _ = mps_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    assert torch.allclose(ref_out, opt_out, atol=1e-3, rtol=1e-3)
    print("[PASS] test_l2norm_option")
