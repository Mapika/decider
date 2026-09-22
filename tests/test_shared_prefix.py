"""decider.shared_prefix cache helpers on CPU with stand-in cache layers.

The real prefix cache needs a GPU checkpoint (tests/test_engine_v2_cuda.py checks the answers); what is checked here is
the part that has to hold for any transformers layout: which tensors are found, what a fork costs, and that forking
leaves the original cache untouched and gives the fork its own storage.
"""
import pytest

torch = pytest.importorskip("torch")

from decider.shared_prefix import (cache_row_bytes, cache_state_tensors, chunk_rows, common_prefix_len, fork_cache,
                                   unknown_state_names)


class _AttnLayer:
    def __init__(self, b=1, seq=8):
        self.keys = torch.zeros(b, 2, seq, 4)
        self.values = torch.ones(b, 2, seq, 4)
        self.is_initialized = True


class _LinearLayer:
    """A linear-attention layer: dicts of conv and recurrent states, and the bookkeeping dicts beside them."""
    def __init__(self, b=1):
        self.keys = torch.tensor([])                      # present but empty, as before the first attention update
        self.values = None
        self.conv_states = {0: torch.zeros(b, 6, 4)}
        self.recurrent_states = {0: torch.ones(b, 2, 3, 3)}
        self.has_previous_state = {0: True}
        self.is_conv_states_initialized = {0: True}


class _Cache:
    def __init__(self, layers):
        self.layers = layers


def _cache(b=1):
    return _Cache([_AttnLayer(b), _LinearLayer(b), _AttnLayer(b, seq=8)])


def test_common_prefix_len():
    assert common_prefix_len([[1, 2, 3, 9], [1, 2, 3, 8]]) == 3
    assert common_prefix_len([[1, 2, 3], [1, 2, 3]]) == 2          # never the whole shortest row
    assert common_prefix_len([[5, 1], [4, 1]]) == 0


def test_state_tensors_cover_both_layouts_and_skip_empties():
    c = _cache()
    found = cache_state_tensors(c)
    shapes = sorted(tuple((t[0][t[1]] if isinstance(t[0], dict) else getattr(t[0], t[1])).shape) for t in found)
    assert shapes == [(1, 2, 3, 3), (1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4), (1, 2, 8, 4), (1, 6, 4)]
    assert len(found) == 6                                          # 4 key/value tensors, 1 conv, 1 recurrent; None and empty skipped


def test_cache_row_bytes_is_per_row():
    one, four = cache_row_bytes(_cache(1)), cache_row_bytes(_cache(4))
    assert one == four and one == (4 * 2 * 8 * 4 + 6 * 4 + 2 * 3 * 3) * 4


def test_chunk_rows_clamps_to_the_budget():
    assert chunk_rows(1000, 32, budget_bytes=10_000) == 10
    assert chunk_rows(1000, 4, budget_bytes=10_000) == 4            # never more than the rows asked for
    assert chunk_rows(10_000, 32, budget_bytes=1000) == 1           # never below one row
    assert chunk_rows(0, 7, budget_bytes=1000) == 7                 # an empty cache does not bound anything


def test_fork_copies_the_rows_without_touching_the_original():
    c = _cache()
    f = fork_cache(c, 3)
    for container, key in cache_state_tensors(f):
        t = container[key] if isinstance(container, dict) else getattr(container, key)
        assert t.shape[0] == 3
    # the fork's tensors are fresh storage: writing into them must not reach the original
    for container, key in cache_state_tensors(f):
        (container[key] if isinstance(container, dict) else getattr(container, key)).fill_(7.0)
    assert float(c.layers[0].values.min()) == 1.0 and float(c.layers[1].recurrent_states[0].min()) == 1.0
    assert c.layers[0].keys.shape[0] == 1
    # the per-state dicts are the fork's own
    f.layers[1].has_previous_state[0] = False
    assert c.layers[1].has_previous_state[0] is True


def test_indexer_keys_are_enumerated_and_forked():
    """transformers 5.17's DynamicIndexedLayer carries `indexer_keys` [batch, seq, dim] beside keys and values.  Missing
    it left the fork at one row there and the suffix forward failed in update_indexer."""
    class _Indexed(_AttnLayer):
        def __init__(self):
            super().__init__()
            self.indexer_keys = torch.zeros(1, 8, 3)
            self.is_indexer_initialized = True
    c = _Cache([_Indexed()])
    assert any(k == "indexer_keys" for _, k in cache_state_tensors(c))
    f = fork_cache(c, 4)
    assert f.layers[0].indexer_keys.shape == (4, 8, 3)
    assert c.layers[0].indexer_keys.shape == (1, 8, 3)


def test_unknown_state_names_finds_tensors_outside_the_enumerated_set():
    assert unknown_state_names(_cache()) == set()
    class _Odd(_AttnLayer):
        def __init__(self):
            super().__init__()
            self.some_new_state = torch.zeros(1, 4)
    assert unknown_state_names(_Cache([_AttnLayer(), _Odd()])) == {"some_new_state"}
    class _OddDict(_AttnLayer):
        def __init__(self):
            super().__init__()
            self.other_states = {0: torch.zeros(1, 4)}
    assert unknown_state_names(_Cache([_OddDict()])) == {"other_states"}


# ---- the scoring loop, against a cache-aware stand-in backbone --------------
"""The fake backbone below keeps a cache in the shape decider.shared_prefix expects and makes the answers depend on the
position: `h[b, t]` is a function of the sum of the ids up to that position, over the cached prefix and the suffix
together.  A fork left at one row makes `torch.cat` fail; a fork that shares storage with the prefix corrupts the
in-place recurrent state and changes the next chunk's answers; a wrong slot offset reads a different position."""
import torch.nn.functional as F

from decider.prompt import MAX_OPTIONS
from decider.shared_prefix import fork_budget_bytes, score_shared

H = 8
PAD = 7


class _FakeLayer:
    def __init__(self):
        self.keys = None
        self.values = None
        self.recurrent_states = {}
        self.is_initialized = True


class _FakeCache:
    def __init__(self, layers):
        self.layers = layers


class _FakeCore(torch.nn.Module):
    """`h` depends on the running sum of the ids, so padding cannot change an earlier slot and a wrong offset shows."""

    def __init__(self, extra_state=False):
        super().__init__()
        self.extra_state = extra_state
        self.calls = []                                  # batch size of every forward, prefix and suffix

    def forward(self, input_ids=None, use_cache=False, past_key_values=None, **kw):
        import types
        x = input_ids.float()                            # [B, T]
        B, T = x.shape
        self.calls.append(B)
        if past_key_values is None:
            layer = _FakeLayer()
            if self.extra_state:
                layer.some_new_state = torch.zeros(B, 2)
            cache = _FakeCache([layer])
            prefix = torch.zeros(B, 0)
            run = torch.zeros(B, 1)
        else:
            cache = past_key_values
            layer = cache.layers[0]
            prefix = layer.keys[:, 0, :, 0]              # a fork left at one row makes the cat below fail
            run = layer.recurrent_states[0]
        full = torch.cat([prefix, x], 1)                 # a fork left at one row fails here
        cum = torch.cumsum(x, 1) + run                   # [B, T]; the prefix enters through the recurrent state
        h = torch.stack([torch.sin(cum * (j + 1) * 0.01) for j in range(H)], -1)
        if use_cache:
            layer.keys = full[:, None, :, None].clone()
            layer.values = layer.keys
            total = run + x.sum(1, keepdim=True)
            if 0 in layer.recurrent_states:
                layer.recurrent_states[0].copy_(total)   # in place, as the real linear-attention layers do: a fork
            else:                                        # sharing storage with the prefix corrupts the next chunk
                layer.recurrent_states[0] = total
        return types.SimpleNamespace(last_hidden_state=h, past_key_values=cache)


class _Tok:
    pad_token_id = PAD


class _Eng:
    def __init__(self, extra_state=False):
        g = torch.Generator().manual_seed(0)
        self.core = _FakeCore(extra_state)
        self.W = torch.randn(MAX_OPTIONS, H, generator=g)
        self.dev = "cpu"
        self.tok = _Tok()
        self.stats = {}


def _items(suffix_lens, prefix=300, n_q=1, seed=0):
    import random
    g = random.Random(seed)
    pre = [g.randrange(1, 500) for _ in range(prefix)]
    out = []
    for i, L in enumerate(suffix_lens):
        suf = [1000 + i] + [g.randrange(1, 500) for _ in range(L - 1)]     # the rows differ at the first suffix token
        ids = pre + suf
        slots = [prefix + (L * (k + 1)) // n_q - 1 for k in range(n_q)]
        nopts = [2 + k % 4 for k in range(n_q)]
        out.append(dict(ids=ids, slots=slots, nopts=nopts, golds=[0] * n_q,
                        perms=[list(range(k)) for k in nopts]))
    return out


def _cat(res):
    return torch.cat(res)


def _same(a, b):
    """Equal up to float32 round-off.  Chunking changes the padded width of a suffix forward, which moves the last bits
    of a float32 reduction; a wrong slot offset or a wrong fork moves the answers by order 0.1 (the test below)."""
    return torch.allclose(a, b, atol=1e-6, rtol=0.0)


def test_the_stand_in_backbone_is_position_sensitive():
    """Without this the equality tests below would pass for the wrong reason."""
    eng = _Eng()
    a = _items([40, 40], prefix=300)
    b = [dict(a[0], slots=[a[0]["slots"][0] - 1]), a[1]]
    assert not torch.allclose(_cat(score_shared(eng, a)), _cat(score_shared(eng, b)))


def test_chunking_matches_the_single_fork_with_unequal_suffixes():
    eng = _Eng()
    items = _items([31, 44, 12, 60, 23])
    ref = _cat(score_shared(eng, items, rows_per_fork=len(items)))
    for m in (1, 2, 3, 4, 6):
        got = _cat(score_shared(eng, items, rows_per_fork=m))
        assert _same(got, ref), m


def test_the_chunk_size_is_the_batch_of_each_suffix_forward():
    eng = _Eng()
    items = _items([31, 44, 12, 60, 23])
    eng.core.calls.clear()
    score_shared(eng, items, rows_per_fork=2)
    assert eng.core.calls == [1, 2, 2, 1]                     # the prefix once, then chunks of 2, 2 and 1


def test_multi_question_rows_keep_their_shape_and_order():
    eng = _Eng()
    items = _items([48, 33, 60], n_q=3)
    ref = score_shared(eng, items, rows_per_fork=len(items))
    assert len(ref) == 3 and all(tuple(p.shape) == (3, MAX_OPTIONS) for p in ref)
    got = score_shared(eng, items, rows_per_fork=1)
    for a, b in zip(got, ref):
        assert _same(a, b)
    swapped = score_shared(eng, [items[2], items[0], items[1]], rows_per_fork=2)   # answers follow the items
    assert _same(swapped[0], ref[2]) and _same(swapped[1], ref[0])


def test_probabilities_are_masked_to_each_question_s_option_count():
    eng = _Eng()
    p = score_shared(eng, _items([40, 52], n_q=3), rows_per_fork=1)[0]
    for k, n in enumerate([2, 3, 4]):
        assert float(p[k, n:].sum()) == 0.0
        assert abs(float(p[k].sum()) - 1.0) < 1e-5


def test_returns_none_when_the_request_does_not_qualify():
    eng = _Eng()
    assert score_shared(eng, _items([40])) is None                       # one row
    assert score_shared(eng, []) is None
    assert score_shared(eng, _items([40, 40], prefix=30)) is None        # prefix below min_prefix
    assert score_shared(eng, _items([40, 40], prefix=30), min_prefix=20) is not None


def test_the_chunk_size_is_clamped_to_one_and_n():
    eng = _Eng()
    items = _items([31, 44, 12])
    ref = _cat(score_shared(eng, items, rows_per_fork=len(items)))
    for m in (99, 0, -3):
        eng.core.calls.clear()
        got = _cat(score_shared(eng, items, rows_per_fork=m))
        assert _same(got, ref), m
        assert max(eng.core.calls) <= len(items) and min(eng.core.calls) >= 1
    eng.core.calls.clear()
    score_shared(eng, items, budget_bytes=0)                             # a budget below one row still forks one row
    assert eng.core.calls == [1, 1, 1, 1]


def test_fork_budget_keeps_the_configured_size_when_the_cuda_query_fails(monkeypatch):
    def raise_it(*a, **k):
        raise RuntimeError("no CUDA device")
    monkeypatch.setattr(torch.cuda, "mem_get_info", raise_it)
    assert fork_budget_bytes("cpu", gb=4) == 4 * (1 << 30)
    monkeypatch.setenv("DECIDER_SHARED_FORK_GB", "2")
    assert fork_budget_bytes("cpu") == 2 * (1 << 30)


def test_an_unknown_cache_layout_is_not_chunked():
    """A tensor outside STATE_NAMES could carry a batch dimension we would not fork: take one fork of every row instead,
    as 1.1.0 did, and count it."""
    eng = _Eng(extra_state=True)
    items = _items([31, 44, 12, 60])
    eng.core.calls.clear()
    got = score_shared(eng, items, rows_per_fork=1)                      # the forced chunk size is overridden
    assert eng.core.calls == [1, 4]
    assert eng.stats["shared_unchunked_layout"] == 1
    plain = _Eng()
    assert _same(_cat(got), _cat(score_shared(plain, items, rows_per_fork=len(items))))
    assert "shared_unchunked_layout" not in plain.stats
