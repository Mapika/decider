"""EngineV2 on CPU with a tiny stand-in backbone: bucket planning, and answers that do not depend on the bucket.

The real Qwen3.5 delta-net layers need CUDA (Triton), so the backbone here is a small causal module with the same
contract: `core(input_ids=..., use_cache=False).last_hidden_state`.  Right padding must not change any earlier
position, which is the property the whole bucketing scheme rests on.
"""
import types
import pytest

torch = pytest.importorskip("torch")

from decider.engine_v2 import EngineV2, T_BUCKETS, B_BUCKETS
from decider.prompt import MAX_OPTIONS

H, V, K = 16, 512, MAX_OPTIONS
PAD = 0


class _CausalCore(torch.nn.Module):
    """h[t] = tanh(mean of embeddings up to t, weighted by position).  Strictly causal, so pad columns on the right
    cannot influence any earlier position."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, H, generator=g))
        self.mix = torch.nn.Linear(H, H, bias=False)
        with torch.no_grad():
            self.mix.weight.copy_(torch.randn(H, H, generator=g) / H ** 0.5)

    def forward(self, input_ids=None, use_cache=False, **kw):
        x = self.emb(input_ids)                                     # [B,T,H]
        w = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        h = torch.tanh(self.mix(torch.cumsum(x, 1) / w))
        return types.SimpleNamespace(last_hidden_state=h, past_key_values=None)


class _Tok:
    pad_token_id = PAD


def _model():
    g = torch.Generator().manual_seed(1)
    core = _CausalCore()
    lm_head = torch.nn.Linear(H, V, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.randn(V, H, generator=g) / H ** 0.5)
    m = types.SimpleNamespace(tok=_Tok(), letters=torch.arange(K),
                              lm=types.SimpleNamespace(model=core, lm_head=lm_head))
    return m


@pytest.fixture(scope="module")
def eng():
    return EngineV2(model=_model(), device="cpu", use_graphs=False)


def _item(ids, slots, nopts):
    return dict(ids=list(ids), slots=list(slots), nopts=list(nopts), golds=[0] * len(slots),
                perms=[list(range(n)) for n in nopts])


def _rand_item(n_tok, n_opt, seed):
    g = torch.Generator().manual_seed(seed)
    ids = (torch.randint(1, V, (n_tok,), generator=g)).tolist()
    return _item(ids, [n_tok - 1], [n_opt])


# ---- bucket planning -----------------------------------------------------
def test_t_bucket_and_pad_len(eng):
    assert eng.t_bucket(1) == 64 and eng.t_bucket(64) == 64 and eng.t_bucket(65) == 128
    assert eng.t_bucket(8192) == 8192 and eng.t_bucket(8193) is None
    assert eng.pad_len(8193) == 9216 and eng.pad_len(200) == 256


def test_graph_shapes_respect_the_token_budget(eng):
    sh = eng.graph_shapes()
    assert sh, "grid is empty"
    for B, T in sh:
        assert B == 1 or B * T <= eng.token_budget
        assert T in T_BUCKETS and B in B_BUCKETS
    for T in T_BUCKETS:                                  # every length bucket is reachable with batch 1
        assert (1, T) in sh
    assert len(set(sh)) == len(sh)
    assert len(sh) == 89                                 # the grid the server captures at start-up (docs/SERVING.md)


def test_b_plan_only_uses_captured_buckets():
    e = EngineV2(model=_model(), device="cpu", use_graphs=False)
    e.graphs = {(B, T): None for B, T in e.graph_shapes()}
    e.seal()
    for T in (64, 1024, 2048, 8192):
        av = e.b_for_t[T]
        for n in (1, 3, 5, 17, 33, 100):
            plan = e.b_plan(n, T)
            assert all(b in av for b in plan), (T, n, plan)
            assert sum(plan) >= n
            assert sum(plan) - n < max(av), (T, n, plan)     # padding never exceeds one full bucket
    assert e.b_plan(1, 64) == [1] and e.b_plan(3, 64) == [4]


def test_sealed_engine_never_captures(eng):
    e = EngineV2(model=_model(), device="cpu", use_graphs=False)
    e.seal()
    assert e.sealed
    before = dict(e.stats)
    e.score_items([_rand_item(100, 3, 5)])
    assert e.stats["graph_captures"] == before["graph_captures"] == 0
    assert e.stats["eager_forwards"] == before["eager_forwards"] + 1


# ---- numerics ------------------------------------------------------------
def test_right_padding_does_not_change_earlier_positions(eng):
    it = _rand_item(70, 4, 11)
    p_small = eng.score_items([it])[0]
    padded = _item(it["ids"] + [PAD] * 500, it["slots"], it["nopts"])
    p_big = eng.score_items([padded])[0]
    assert torch.allclose(p_small, p_big, atol=1e-6)


def test_batching_does_not_change_answers(eng):
    items = [_rand_item(50 + 7 * i, 2 + i % 6, 100 + i) for i in range(11)]
    alone = [eng.score_items([it])[0] for it in items]
    together = eng.score_items(items)
    for a, b in zip(alone, together):
        assert torch.allclose(a, b, atol=1e-5)
        assert int(a.argmax()) == int(b.argmax())


def test_chunking_covers_every_item(eng):
    e = EngineV2(model=_model(), device="cpu", use_graphs=False, b_buckets=[1, 2, 4], token_budget=1 << 20)
    e.graphs = {(B, T): None for B, T in e.graph_shapes()}
    e.seal()
    items = [_rand_item(60, 3, 200 + i) for i in range(9)]
    out = e.score_items(items)
    assert len(out) == len(items)
    ref = eng.score_items(items)
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, atol=1e-6)


def test_probabilities_are_masked_to_the_option_count(eng):
    p = eng.score_items([_rand_item(40, 3, 9)])[0]
    assert p.shape == (1, MAX_OPTIONS)
    assert float(p[0, 3:].sum()) == 0.0
    assert abs(float(p[0].sum()) - 1.0) < 1e-5


def test_temperature_is_applied(eng):
    it = _rand_item(40, 5, 3)
    hot = eng.score_items([it], temperature=5.0)[0]
    cold = eng.score_items([it], temperature=0.5)[0]
    assert float(cold[0].max()) > float(hot[0].max())


def test_long_rows_run_eager_and_are_counted(eng):
    e = EngineV2(model=_model(), device="cpu", use_graphs=False)
    e.seal()
    before = e.stats["unbucketed_requests"]
    e.score_items([_rand_item(9000, 3, 4)])
    assert e.stats["unbucketed_requests"] == before + 1


def test_long_rows_are_chunked_to_the_token_budget():
    e = EngineV2(model=_model(), device="cpu", use_graphs=False, token_budget=20000)
    e.seal()
    items = [_rand_item(9000, 3, 300 + i) for i in range(5)]       # padded to 9216: two rows fit the budget, so 3 forwards
    before = e.stats["eager_forwards"]
    out = e.score_items(items)
    assert len(out) == 5 and e.stats["eager_forwards"] == before + 3
    ref = [e.score_items([it])[0] for it in items]
    for a, b in zip(out, ref):
        assert torch.allclose(a, b, atol=1e-6)


def test_init_applies_the_attention_backend_policy():
    if not hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        pytest.skip("this torch build has no cuDNN SDPA switch")
    torch.backends.cuda.enable_cudnn_sdp(True)
    EngineV2(model=_model(), device="cpu", use_graphs=False)
    assert torch.backends.cuda.cudnn_sdp_enabled() is False


def test_sealed_graph_engine_runs_a_miss_eager_without_capturing():
    """The graph-enabled branch: after seal(), a shape with no graph must call the uncompiled forward and never _capture."""
    e = EngineV2(model=_model(), device="cpu", use_graphs=True)
    e.graphs = {}; e.seal()
    def boom(*a, **k): raise AssertionError("capture or compiled forward called after seal()")
    e._capture = boom; e._fwd_impl = boom
    p = e.score_items([_rand_item(100, 3, 5)])[0]
    assert e.stats["graph_captures"] == 0 and e.stats["replays"] == 0 and e.stats["eager_forwards"] == 1
    assert bool(p.isfinite().all())
