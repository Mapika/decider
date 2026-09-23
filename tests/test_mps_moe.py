"""decider.mps_moe (issue #6): the histc and triangular-solve replacements, checked on CPU tensors (the wrappers themselves
only act on MPS tensors; everything else must reach torch unchanged)."""
import pytest

torch = pytest.importorskip("torch")
from decider import mps_moe


def test_count_ids_matches_histc_including_dropped_sentinels():
    g = torch.Generator().manual_seed(0)
    for n in (8, 256):
        ids = torch.randint(0, n, (4096,), generator=g)
        ids = torch.cat([ids, torch.full((37,), n), torch.full((5,), n + 3)])       # expert-parallel sentinels >= n
        x = torch.sort(ids).values.float()
        ref = torch.histc(x, bins=n, min=0, max=n - 1)
        assert torch.equal(mps_moe.count_ids(x, n, 0), ref)


def test_unit_lower_solve_matches_torch():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(2, 4, 3, 64, 64, generator=g, dtype=torch.float64) * 0.1           # full matrix: only the strict lower part counts
    B = torch.randn(2, 4, 3, 64, 32, generator=g, dtype=torch.float64)
    ref = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    assert torch.allclose(mps_moe.solve_unit_lower(A, B), ref, atol=1e-10, rtol=1e-10)
    A32, B32 = A.float(), B.float()
    ref32 = torch.linalg.solve_triangular(A32, B32, upper=False, unitriangular=True)
    assert (mps_moe.solve_unit_lower(A32, B32) - ref32).abs().max() < 1e-4


def test_wrappers_pass_non_mps_calls_through():
    x = torch.tensor([0.0, 0.9, 1.6, 3.0]); A = torch.randn(64, 64); B = torch.randn(64, 4)
    assert torch.equal(mps_moe.histc(input=x, bins=4, min=0, max=3), torch.histc(x, bins=4, min=0, max=3))
    assert torch.equal(mps_moe.solve_triangular(A, B, upper=False, unitriangular=True),
                       torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True))
    E = torch.zeros(0, 0); assert mps_moe.solve_triangular(E, torch.zeros(0, 1), upper=False, unitriangular=True).shape == (0, 1)


def test_install_is_scoped_to_the_two_transformers_modules():
    moe = pytest.importorskip("transformers.integrations.moe")
    mq = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    names = mps_moe.install()
    try:
        assert torch.histc is mps_moe._histc and torch.linalg.solve_triangular is mps_moe._solve       # global torch untouched
        assert set(names) == {moe.__name__, mq.__name__}
        assert moe.torch.histc is mps_moe.histc and moe.torch.sort is torch.sort
        assert mq.torch.linalg.solve_triangular is mps_moe.solve_triangular and mq.torch.linalg.norm is torch.linalg.norm
        assert mq.torch.histc is torch.histc
        assert mps_moe.install() == names                                                           # idempotent
    finally:
        mps_moe.uninstall()
    assert moe.torch is torch and mq.torch is torch
