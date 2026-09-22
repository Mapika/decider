"""MPS kernels for Qwen3.5 gated-delta attention.

The optimization is measured against Transformers' PyTorch reference fallback for
this kernel; it is not an end-to-end model speedup claim.

The gated-delta and L2-normalization code follows Transformers 5.17's
``modeling_qwen3_5.py`` under the Apache License 2.0. The MPS inversion and
backend dispatch are Decider additions.
"""
import functools
import inspect
import logging
import warnings

import torch, torch.nn.functional as F
from decider.engine import fused_causal_conv1d_fn


# Adapted from transformers 5.17.0's modeling_qwen3_5.py (Apache-2.0).
# The original license applies to the gated-delta and normalization portions;
# the MPS inversion and dispatch below are Decider additions.
# Native Metal Shading Language (MSL) JIT kernel via MLX/Metal
_metal_invert_kernel = None
_metal_failure_reported = False
_compat_warning_reported = False
try:
    import mlx.core as mx
    import mlx.core.fast as fast

    _METAL_INVERT_SRC = """
    uint col = thread_position_in_threadgroup.x;
    uint mat_idx = threadgroup_position_in_grid.x;
    threadgroup float s_inv[4096];

    for (uint r = 0; r < 64; ++r) {
        s_inv[r * 64 + col] = (r == col) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint mat_offset = mat_idx * 4096;
    for (uint r = 1; r < 64; ++r) {
        float sum = 0.0f;
        if (col < r) {
            for (uint k = col; k < r; ++k) {
                sum += L[mat_offset + r * 64 + k] * s_inv[k * 64 + col];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < r) {
            s_inv[r * 64 + col] = -sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint r = 0; r < 64; ++r) {
        out_inv[mat_offset + r * 64 + col] = s_inv[r * 64 + col];
    }
    """
    _metal_invert_kernel = fast.metal_kernel(
        name="invert_unitriangular_64",
        input_names=["L"],
        output_names=["out_inv"],
        source=_METAL_INVERT_SRC,
    )
except (ImportError, OSError, RuntimeError, AttributeError):
    logging.getLogger(__name__).debug("MLX Metal kernel unavailable; using PyTorch MPS fallback", exc_info=True)
    _metal_invert_kernel = None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def fast_invert_unitriangular_64(L: torch.Tensor) -> torch.Tensor:
    """Invert batches of 64x64 unit lower-triangular matrices."""
    global _metal_invert_kernel, _metal_failure_reported
    if _metal_invert_kernel is not None and L.is_mps and not L.requires_grad:
        assert L.dtype == torch.float32, "Metal inversion requires float32 input"
        try:
            orig_shape = L.shape
            L_flat = L.reshape(-1, 64, 64).contiguous()
            num_m = L_flat.shape[0]
            torch.mps.synchronize()
            L_mx = mx.from_dlpack(torch.to_dlpack(L_flat))
            mx.eval(L_mx)
            out_mx = _metal_invert_kernel(
                inputs=[L_mx],
                grid=(num_m * 64, 1, 1),
                threadgroup=(64, 1, 1),
                output_shapes=[(num_m, 64, 64)],
                output_dtypes=[mx.float32],
            )[0]
            mx.eval(out_mx)
            torch.mps.synchronize()
            return torch.from_dlpack(out_mx).to(L.device).reshape(orig_shape)
        except Exception:
            # Optional acceleration must not turn a recoverable backend issue into
            # a failed request; disable the failing kernel for this process.
            _metal_invert_kernel = None
            if not _metal_failure_reported:
                logging.getLogger(__name__).warning("MLX Metal inversion failed; using the PyTorch MPS fallback", exc_info=True)
                _metal_failure_reported = True

    N = 64
    inv = torch.eye(N, device=L.device, dtype=L.dtype).expand_as(L).clone()

    # b = 1: 32 blocks of 2x2. For [[1, 0], [l, 1]], inverse is [[1, 0], [-l, 1]]
    idx_row = torch.arange(1, 64, 2, device=L.device)
    idx_col = torch.arange(0, 64, 2, device=L.device)
    inv[..., idx_row, idx_col] = -L[..., idx_row, idx_col]

    # b = 2: 16 blocks of 4x4
    for j in range(16):
        r1, r2, r3 = j * 4, j * 4 + 2, j * 4 + 4
        inv[..., r2:r3, r1:r2] = -inv[..., r2:r3, r2:r3] @ L[..., r2:r3, r1:r2] @ inv[..., r1:r2, r1:r2]

    # b = 4: 8 blocks of 8x8
    for j in range(8):
        r1, r2, r3 = j * 8, j * 8 + 4, j * 8 + 8
        inv[..., r2:r3, r1:r2] = -inv[..., r2:r3, r2:r3] @ L[..., r2:r3, r1:r2] @ inv[..., r1:r2, r1:r2]

    # b = 8: 4 blocks of 16x16
    for j in range(4):
        r1, r2, r3 = j * 16, j * 16 + 8, j * 16 + 16
        inv[..., r2:r3, r1:r2] = -inv[..., r2:r3, r2:r3] @ L[..., r2:r3, r1:r2] @ inv[..., r1:r2, r1:r2]

    # b = 16: 2 blocks of 32x32
    for j in range(2):
        r1, r2, r3 = j * 32, j * 32 + 16, j * 32 + 32
        inv[..., r2:r3, r1:r2] = -inv[..., r2:r3, r2:r3] @ L[..., r2:r3, r1:r2] @ inv[..., r1:r2, r1:r2]

    # b = 32: 1 block of 64x64
    inv[..., 32:64, 0:32] = -inv[..., 32:64, 32:64] @ L[..., 32:64, 0:32] @ inv[..., 0:32, 0:32]
    return inv


def mps_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Optimized chunk-gated delta rule for Apple Silicon (MPS)."""
    assert chunk_size == 64, f"MPS optimized delta rule requires chunk_size=64, got {chunk_size}"
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    recurrent_state_shape = (batch_size, num_v_heads, k_head_dim, v_head_dim)
    padded_output_shape = (batch_size, num_v_heads, -1, v_head_dim)
    decay = g

    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    scaling = query.shape[-1] ** -0.5
    query = query * scaling

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    decay = F.pad(decay, (0, pad_size))

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strictly_upper_mask, float("-inf")).exp()

    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

    # Fast block divide-and-conquer unitriangular inverse on MPS
    L = ut_system.tril(-1)
    inv = fast_invert_unitriangular_64(L)
    new_values = inv @ v_beta
    k_cumdecay = inv @ decayed_k_beta

    if initial_state is None:
        last_recurrent_state = torch.zeros(recurrent_state_shape, dtype=new_values.dtype, device=new_values.device)
    else:
        last_recurrent_state = initial_state.to(new_values)
    core_attn_out = torch.zeros_like(new_values)

    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    num_chunks = query.shape[2]
    qk = torch.cat([query, k_cumdecay], dim=3)
    kt = key.transpose(-1, -2)

    for i in range(num_chunks):
        qk_state = qk[:, :, i] @ last_recurrent_state
        inter_chunk_attn = qk_state[:, :, :chunk_size]
        v_new = new_values[:, :, i] - qk_state[:, :, chunk_size:]
        core_attn_out[:, :, i] = inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
        last_recurrent_state = last_recurrent_state * chunk_decay[:, :, i] + kt[:, :, i] @ v_new

    last_recurrent_state = None if not output_final_state else last_recurrent_state
    core_attn_out = core_attn_out.reshape(padded_output_shape)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, last_recurrent_state


def patch_mps():
    """Patch Qwen3.5 operations on MPS tensors while preserving other backends."""
    global _compat_warning_reported
    if not torch.backends.mps.is_available():
        return False
    try:
        import transformers
        import transformers.models.qwen3_5.modeling_qwen3_5 as mq
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if not all(callable(getattr(mq, name, None)) for name in ("torch_chunk_gated_delta_rule", "causal_conv1d_fn")):
            raise AttributeError("unsupported Transformers Qwen3.5 implementation")
        params = inspect.signature(mq.torch_chunk_gated_delta_rule).parameters
        required = {"query", "key", "value", "g", "beta", "chunk_size"}
        if version < (5, 17) or not required.issubset(params):
            if not _compat_warning_reported:
                warnings.warn(f"MPS patch requires Transformers >=5.17 with the Qwen3.5 gated-delta signature; got {transformers.__version__}", RuntimeWarning, stacklevel=2)
                _compat_warning_reported = True
            return False
        if getattr(mq.torch_chunk_gated_delta_rule, "_decider_mps_patch", False):
            return True

        original_delta, original_conv = mq.torch_chunk_gated_delta_rule, mq.causal_conv1d_fn
        delta_params = tuple(inspect.signature(original_delta).parameters)
        query_index, chunk_index = delta_params.index("query"), delta_params.index("chunk_size")

        @functools.wraps(original_delta)
        def delta(*args, **kwargs):
            query = args[query_index] if len(args) > query_index else kwargs["query"]
            chunk_size = args[chunk_index] if len(args) > chunk_index else kwargs.get("chunk_size", 64)
            return mps_chunk_gated_delta_rule(*args, **kwargs) if query.is_mps and chunk_size == 64 else original_delta(*args, **kwargs)

        @functools.wraps(original_conv)
        def conv(hidden_states, *args, **kwargs):
            return fused_causal_conv1d_fn(hidden_states, *args, **kwargs) if hidden_states.is_mps else original_conv(hidden_states, *args, **kwargs)

        delta._decider_mps_patch = True
        mq.torch_chunk_gated_delta_rule, mq.causal_conv1d_fn = delta, conv
        return True
    except (ImportError, AttributeError, TypeError, ValueError):
        logging.getLogger(__name__).debug("MPS patch unavailable; using Transformers fallback", exc_info=True)
        return False
