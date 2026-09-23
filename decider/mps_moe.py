"""Two MPS slow paths in transformers' Qwen3.5-MoE reference code, replaced in-process (decider-35b-a3b on Apple Silicon).

Contributed by @nassersala in issue #6 under the project's Apache-2.0 licence; adapted for the package (out-of-range expert
ids are dropped as `histc` drops them, and the replacements are installed from `decider.mps_ops.patch_mps`).

On an M-series Mac (torch 2.14, transformers 5.17) `torch.histc` takes about 45 ms a call on MPS (once per MoE layer, to
count tokens per expert) and `torch.linalg.solve_triangular` about 17 ms (twice per linear-attention layer): most of a
2.8-4 s decision. With both replaced the 35B takes 0.23-0.5 s a decision for typical inputs (reported in issue #6).
The `histc` replacement is an exact count. The block inverse is within 1e-6 of the MPS solver on real systems; on six items
the probabilities moved at most 0.033, less than switching to the exact CPU solver does (0.042), and no argmax changed.

The replacements are visible only inside those two transformers modules (install() gives each its own view of `torch`), and
there they act only on MPS tensors of the matching call shape; everything else goes to torch.
"""
import torch

_histc, _solve = torch.histc, torch.linalg.solve_triangular


def count_ids(x, bins, min=0):
    """histc for integer-valued ids binned one per bin: a count of ids min..min+bins-1; other values are dropped as histc
    drops them (transformers passes expert-parallel sentinels >= num_experts and relies on that)."""
    idx = x.long() - int(min)
    keep = (idx >= 0) & (idx < bins)
    out = torch.zeros(bins, device=x.device, dtype=x.dtype)
    return out.scatter_add_(0, idx.clamp(0, bins - 1), keep.to(x.dtype))


def histc(input, bins=100, min=0, max=0, *, out=None):
    # expert ids 0..n-1 into n bins over [0, n-1]: bin k holds id k exactly, so this is a count (histc: ~45 ms on MPS)
    if input.device.type == "mps" and input.dim() == 1 and out is None and min == 0 and max > 0 and bins == max - min + 1:
        return count_ids(input, bins, min)
    return _histc(input, bins=bins, min=min, max=max) if out is None else _histc(input, bins=bins, min=min, max=max, out=out)


def unit_lower_inverse(A):
    """Inverse of unit lower-triangular A (last dim a power of two) by block doubling:
    [[A11, 0], [A21, A22]]^-1 = [[X11, 0], [-X22 A21 X11, X22]]."""
    n = A.shape[-1]
    inv = torch.ones(*A.shape[:-2], n, 1, 1, device=A.device, dtype=A.dtype)  # n diagonal 1x1 blocks
    b = 1
    while b < n:
        nb = n // (2 * b)
        blocks = A.reshape(*A.shape[:-2], nb, 2 * b, nb, 2 * b).diagonal(dim1=-4, dim2=-2).movedim(-1, -3)
        x11, x22, a21 = inv[..., 0::2, :, :], inv[..., 1::2, :, :], blocks[..., b:, :b]
        new = torch.zeros(*A.shape[:-2], nb, 2 * b, 2 * b, device=A.device, dtype=A.dtype)
        new[..., :b, :b], new[..., b:, b:], new[..., b:, :b] = x11, x22, -(x22 @ a21 @ x11)
        inv, b = new, 2 * b
    return inv[..., 0, :, :]


def solve_unit_lower(A, B):
    """solve_triangular(A, B, upper=False, unitriangular=True): only the strict lower triangle of A is read."""
    n = A.shape[-1]
    return unit_lower_inverse(A.tril(-1) + torch.eye(n, device=A.device, dtype=A.dtype)) @ B


def solve_triangular(A, B, *, upper, left=True, unitriangular=False, out=None):
    # unit lower-triangular: invert by block doubling, all batched matmuls (the MPS solver: ~17 ms per call)
    n = A.shape[-1]
    if A.device.type == "mps" and not upper and left and unitriangular and out is None and n > 0 and n & (n - 1) == 0:
        return solve_unit_lower(A, B)
    return _solve(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)


class _Proxy:
    """A module's view of `torch` (or `torch.linalg`) with some attributes replaced; everything else is the real one."""
    def __init__(self, real, **over):
        self._real, self._over = real, over

    def __getattr__(self, name):
        return self._over[name] if name in self._over else getattr(self._real, name)


_installed = {}


def install():
    """Replace the two operations inside the two transformers modules that call them, and nowhere else: the grouped-MoE
    expert count in `transformers.integrations.moe` and the gated delta rule in `modeling_qwen3_5_moe` see a `torch` whose
    `histc` / `linalg.solve_triangular` are the replacements; every other caller keeps torch's.  -> names patched."""
    import importlib
    targets = {"transformers.integrations.moe": dict(histc=histc),
               "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe": dict(linalg=_Proxy(torch.linalg, solve_triangular=solve_triangular))}
    for name, over in targets.items():
        if name in _installed:
            continue
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if getattr(mod, "torch", None) is not torch:
            continue                                  # not the layout this was written for: leave it alone
        _installed[name] = mod
        mod.torch = _Proxy(torch, **over)
    return sorted(_installed)


def uninstall():
    for name, mod in list(_installed.items()):
        mod.torch = torch
        del _installed[name]
