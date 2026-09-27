"""decider.engine_gguf against a fake llama_cpp module (no llama.cpp build, no model file).

The fake returns, for every batch position with its logits flag set, a vector that depends only on (token, position), so a
correct engine gives the same probabilities whatever the grouping of rows into decodes.  Checked: slot gathering and the
letter columns, the option mask, scalar and per-slot temperatures, grouping by n_seq_max and n_batch with the offsets back
into item order, the n_ctx guard, the llama_decode error, and Decider's GGUF path resolution (a .gguf file, gguf_file in a
folder) with decider_config.json read from the right folder and schema() refused.
"""
import ctypes, json, sys, types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

N_VOCAB = 248320


def _row(token, pos):
    v = np.arange(N_VOCAB, dtype=np.float64)
    return (np.sin(token * 0.001 * (v + 1) + pos * 0.37) * 3).astype(np.float32)


class _Batch:
    def __init__(self, n):
        self.token = [0] * n; self.pos = [0] * n; self.n_seq_id = [0] * n; self.logits = [0] * n
        self.seq_id = [[0] for _ in range(n)]; self.n_tokens = 0


def fake_llama(decode_rc=0):
    L = types.ModuleType("llama_cpp")
    L.calls = []
    L.llama_log_callback = lambda f: f
    L.llama_log_set = lambda cb, data: None
    L.llama_backend_init = lambda: None
    L.llama_model_default_params = lambda: types.SimpleNamespace(n_gpu_layers=0)
    L.llama_model_load_from_file = lambda path, mp: object()
    L.llama_context_default_params = lambda: types.SimpleNamespace(n_ctx=0, n_batch=0, n_ubatch=0, n_seq_max=0, kv_unified=False,
                                                                   n_threads=0, n_threads_batch=0)
    ctx_params = {}

    def init_from_model(model, cp):
        ctx_params["n_batch"], ctx_params["n_ctx"] = cp.n_batch, cp.n_ctx
        return object()

    L.llama_init_from_model = init_from_model
    L.llama_model_get_vocab = lambda model: None
    L.llama_vocab_n_tokens = lambda vocab: N_VOCAB
    L.llama_batch_init = lambda n, embd, nseq: _Batch(n)
    L.llama_get_memory = lambda ctx: None
    L.llama_memory_clear = lambda mem, data: None
    L.llama_batch_free = L.llama_free = L.llama_model_free = lambda *a: None
    state = {}

    def decode(ctx, b):
        # llama.cpp asserts n_tokens <= the context's n_batch and aborts the process; the fake fails the test instead
        assert b.n_tokens <= ctx_params["n_batch"], f"llama.cpp would abort: {b.n_tokens} tokens > context n_batch {ctx_params['n_batch']}"
        rows = sorted({b.seq_id[i][0] for i in range(b.n_tokens)})
        L.calls.append(dict(n_tokens=b.n_tokens, rows=len(rows)))
        state["out"] = {i: _row(b.token[i], b.pos[i]) for i in range(b.n_tokens) if b.logits[i]}
        return decode_rc

    def logits_ith(ctx, i):
        arr = state["out"][i]
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    L.llama_decode = decode; L.llama_get_logits_ith = logits_ith
    return L


@pytest.fixture
def engine_factory(tok, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: tok))

    def make(decode_rc=0, **kw):
        L = fake_llama(decode_rc)
        monkeypatch.setitem(sys.modules, "llama_cpp", L)
        import importlib, decider.engine_gguf as eg
        importlib.reload(eg)
        return eg.GGUFEngine("model.gguf", "tokdir", **kw), L
    return make


def _items(tok):
    from decider.infer import Example, Q, _NoShuffle
    from decider.prompt import build, MAX_OPTIONS
    exs = [Example("The card was charged twice.", [Q("Which team?", ["billing", "tech", "sales"]), Q("Urgent?", ["yes", "no"])]),
           Example("App crashes on start.", [Q("Bug?", ["yes", "no"])]),
           Example("x " * 300, [Q("Long?", ["a", "b", "c", "d", "e"])]),
           Example("Short.", [Q("Pick", ["one", "two", "three", "four"])])]
    return [build(e, tok, _NoShuffle(), max_options=MAX_OPTIONS) for e in exs]


def _reference(eng, items, T):
    """softmax over the letter logits of the fake at every slot, computed directly."""
    out = []
    for it, Ts in zip(items, T):
        rows = []
        for k, (s, n) in enumerate(zip(it["slots"], it["nopts"])):
            lg = torch.from_numpy(_row(it["ids"][s], s)[eng.letters][:n]).double()
            t = Ts[k] if isinstance(Ts, list) else Ts
            rows.append(torch.softmax(lg / t, -1))
        out.append(rows)
    return out


@pytest.mark.parametrize("n_seq_max,n_batch", [(1, 8192), (8, 8192), (2, 8192), (8, 400)])
def test_probabilities_match_direct_readout_for_any_grouping(engine_factory, tok, n_seq_max, n_batch):
    eng, L = engine_factory(n_seq_max=n_seq_max, n_batch=n_batch)
    items = _items(tok)
    T = [[1.2, 1.5], 1.0, 2.0, 0.8]                              # per-slot, per-item scalar
    probs = eng.score_items(items, temperature=T)
    ref = _reference(eng, items, T)
    for p, it, r in zip(probs, items, ref):
        assert p.shape[0] == len(it["slots"])
        for k, n in enumerate(it["nopts"]):
            assert torch.allclose(p[k, :n].double(), r[k], atol=1e-6)
            assert float(p[k, n:].abs().sum()) == 0.0             # masked options get exactly zero
    assert all(c["rows"] <= n_seq_max and (c["n_tokens"] <= n_batch or c["rows"] == 1) for c in L.calls)
    if n_seq_max == 1:
        assert len(L.calls) == len(items)


def test_scalar_temperature_and_score_shared(engine_factory, tok):
    eng, _ = engine_factory()
    items = _items(tok)
    a = eng.score_items(items, temperature=1.3); b = eng.score_shared(items, temperature=1.3)
    for x, y, r in zip(a, b, _reference(eng, items, [1.3] * len(items))):
        assert torch.equal(x, y)
        assert torch.allclose(x[0, :len(r[0])].double(), r[0], atol=1e-6)


def test_row_longer_than_the_packing_budget_decodes_alone(engine_factory, tok):
    """A row between n_batch (the packing budget) and n_ctx: one decode of its own, which the context must accept."""
    eng, L = engine_factory(n_ctx=4096, n_batch=128, n_seq_max=4)
    items = _items(tok)
    long_rows = [i for i, it in enumerate(items) if len(it["ids"]) > 128]
    assert long_rows                                               # the 336-token row
    probs = eng.score_items(items, temperature=1.0)
    for p, it, r in zip(probs, items, _reference(eng, items, [1.0] * len(items))):
        assert torch.allclose(p[0, :len(r[0])].double(), r[0], atol=1e-6)
    assert any(c["n_tokens"] > 128 and c["rows"] == 1 for c in L.calls)


def test_row_longer_than_n_ctx_raises(engine_factory, tok):
    eng, _ = engine_factory(n_ctx=64)
    with pytest.raises(ValueError, match="exceeds n_ctx"):
        eng.score_items(_items(tok))


def test_decode_error_raises(engine_factory, tok):
    eng, _ = engine_factory(decode_rc=1)
    with pytest.raises(RuntimeError, match="llama_decode returned 1"):
        eng.score_items(_items(tok)[:1])


def test_decider_resolves_gguf_paths(tmp_path, monkeypatch, tok):
    import decider.engine_gguf as eg
    seen = []

    class FakeEngine:
        def __init__(self, gguf_path, src, **kw):
            seen.append((gguf_path, src, kw)); self.m = types.SimpleNamespace(tok=tok)

        def score_items(self, items, temperature=1.0):
            seen.append(("T", temperature))
            return [torch.full((len(it["slots"]), 255), 0.0).index_fill_(1, torch.tensor([0]), 1.0) for it in items]

    monkeypatch.setattr(eg, "GGUFEngine", FakeEngine)
    (tmp_path / "decider_config.json").write_text(json.dumps({"temperature": 1.7, "version": "4b-test", "layout": "plain"}))
    (tmp_path / "m-Q4_K_M.gguf").write_bytes(b"")
    from decider.infer import Decider
    d = Decider(str(tmp_path / "m-Q4_K_M.gguf"))
    assert seen[0][:2] == (str(tmp_path / "m-Q4_K_M.gguf"), str(tmp_path)) and d.name == "decider-4b-test" and d.T == 1.7
    out = d.decide("ctx", [{"question": "q", "options": ["a", "b"]}])
    assert out[0]["choice"] == "a" and seen[-1] == ("T", 1.7)
    d2 = Decider(str(tmp_path), gguf_file="m-Q4_K_M.gguf", gguf_options=dict(n_ctx=4096))
    assert seen[-1] == (str(tmp_path / "m-Q4_K_M.gguf"), str(tmp_path), {"n_ctx": 4096})
    assert d2.schema_first is False
    with pytest.raises(NotImplementedError):
        d2.schema({"x": {"type": "noul", "instructions": "?"}})
