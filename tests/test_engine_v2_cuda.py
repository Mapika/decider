"""Real-checkpoint regression tests for EngineV2 (marker `cuda`; skipped without a GPU).

The shape class that exposed the cuDNN SDPA fault (docs/CHANGELOG.md 1.0.2): two rows over one state, prefix about 4,000
tokens, suffixes of 200 to 500 tokens.  The shared-state path (prefix once, cache forked, suffixes scored) must give finite
output with the same argmax as the full forward and as the masked eager reference, on a synthetic prompt of those lengths.
The graph path at a captured shape must match the eager forward as well.

    DECIDER_TEST_MODEL=/path/or/hub-id CUDA_VISIBLE_DEVICES=0 python -m pytest -q -m cuda tests/test_engine_v2_cuda.py
"""
import pytest

torch = pytest.importorskip("torch")

from decider.prompt_fast import build_rows

REVIEWS = [
    "The blender crushes ice in seconds and the jar is easy to clean, though the lid leaks a little when it is full.",
    "These running shoes fit true to size and the cushioning held up over three hundred kilometres of pavement.",
    "The novel starts slowly but the last hundred pages are hard to put down; the ending is earned.",
    "This laptop is light and the battery lasts a working day, but the fan is loud under load.",
    "The tent kept us dry through two nights of rain and packs down small enough for a carry-on.",
    "The headphones cancel most aeroplane noise and the ear cups stay comfortable for hours.",
]
FINAL = ("The coffee maker brews a full carafe in six minutes, keeps it hot without scorching, and the reusable filter "
         "means no more paper filters; the water tank is easy to fill from the front.")
OPTIONS = ["coffee maker", "running shoes", "novel", "laptop", "tent", "headphones", "blender"]
FILLER = " Consider the wording, the product features named, and the situation described before answering."


def _prompt(tok, prefix_tokens=4000, suffix_tokens=(223, 240)):
    """Two rows sharing a prefix of exactly `prefix_tokens` tokens, with question blocks of about the given lengths."""
    n = lambda text: len(tok.encode("Context:\n" + text, add_special_tokens=False))
    lines, i = [], 0
    while n("\n".join(lines) + "\nFinal review: " + FINAL) < prefix_tokens - 60:
        lines.append(f"Review {i + 1}: {REVIEWS[i % len(REVIEWS)]}"); i += 1
    state = "\n".join(lines) + "\nFinal review: " + FINAL + "\n" + "(end of the review list) " * 40   # trailing text is what truncation removes
    assert n(state) >= prefix_tokens
    rows = []
    for target in suffix_tokens:
        q = "Which product does the final review describe?"
        while len(tok.encode(q, add_special_tokens=False)) < target - 60:
            q += FILLER
        rows.append([(q, OPTIONS)])
    items, ctx_len = build_rows(tok, state, rows, max_ctx_tokens=prefix_tokens)
    assert ctx_len == prefix_tokens
    lengths = [len(it["ids"]) for it in items]
    assert all(200 <= n - ctx_len <= 500 for n in lengths), lengths            # the shape class under test
    assert FINAL[:30] in tok.decode(items[0]["ids"][:ctx_len])                # the final review is inside the kept state
    return items


@pytest.fixture(scope="module")
def engine(cuda_model):
    from decider.engine_v2 import EngineV2
    e = EngineV2(cuda_model, use_graphs=True)
    e.warmup(shapes=[(2, 6144)]); e.seal()
    return e


@pytest.mark.cuda
def test_engine_applies_the_attention_backend_policy(engine):
    assert torch.backends.cuda.cudnn_sdp_enabled() is False


@pytest.mark.cuda
def test_shared_path_matches_full_forward_on_the_long_prefix_shape(engine):
    from decider.model import collate
    items = _prompt(engine.tok)
    with torch.no_grad():
        b = collate(items, engine.tok.pad_token_id)
        lg = engine.m.slot_logits(*[b[k].to(engine.dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
        ref = torch.softmax(lg, -1).cpu()
        full = torch.cat(engine.score_items(items))                             # graph replay at (2, 6144)
        shared = torch.cat(engine.score_shared(items))                          # prefix once, cache forked, suffixes scored
    assert engine.stats["shared_calls"] == 1 and engine.stats["replays"] >= 1
    for name, p in (("full", full), ("shared", shared)):
        assert bool(p.isfinite().all()), name
        assert (p.argmax(-1) == ref.argmax(-1)).all(), (name, p.argmax(-1).tolist(), ref.argmax(-1).tolist())
    assert float((full - ref).abs().max()) < 0.05
    assert float((shared - full).abs().max()) < 0.05
    assert (ref.argmax(-1) == OPTIONS.index("coffee maker")).all() and float(ref.max(-1).values.min()) > 0.5, ref.max(-1)


@pytest.mark.cuda
def test_cached_continuation_matches_the_single_prefill_at_the_known_bad_splits(engine):
    """The exact configuration of decider/bench/probe_cache_split.py that fails with the cuDNN SDPA backend on: the seed-0
    random sequence of length 4,300, split at 3,840 / 3,968 / 4,096.  With the backend off the relative error at the last
    position is about 0.01; with it on, about 1.0 (or NaN)."""
    g = torch.Generator().manual_seed(0)
    for L in (1200, 2500):                                                     # the probe draws these first; keep its stream
        torch.randint(1000, 200000, (L,), generator=g)
    ids = torch.tensor(torch.randint(1000, 200000, (4300,), generator=g).tolist(), device=engine.dev)[None]
    with torch.no_grad():
        full = engine.core(input_ids=ids, use_cache=False).last_hidden_state[0, -1]
        for P in (3840, 3968, 4096):
            cache = engine.core(input_ids=ids[:, :P], use_cache=True).past_key_values
            h = engine.core(input_ids=ids[:, P:], past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
            assert bool(h.isfinite().all()), P
            rel = float((full - h).abs().max() / full.abs().max())
            assert rel < 0.05, (P, rel)


@pytest.mark.cuda
def test_chunked_shared_fork_is_exact_when_the_suffixes_are_equally_long(engine):
    """The memory-bounded fork (decider.shared_prefix): the prefix cache is copied m rows at a time instead of n at once.
    With every suffix the same length, every chunk pads to the same width as the single fork, and the answers are
    bit-identical at m = 1, 2, 3 -- the batch size of the suffix forward on its own changes nothing."""
    items = _prompt(engine.tok, suffix_tokens=(300, 300, 300, 300))
    full = torch.cat(engine.score_shared(items, rows_per_fork=len(items)))
    assert bool(full.isfinite().all())
    for m in (1, 2, 3):
        got = torch.cat(engine.score_shared(items, rows_per_fork=m))
        assert float((got - full).abs().max()) == 0.0, (m, float((got - full).abs().max()))


@pytest.mark.cuda
@pytest.mark.parametrize("suffixes", [(223, 240, 300, 450),
                                      tuple(223 + 8 * i for i in range(32))])           # 4 rows and 32 rows
def test_chunked_shared_fork_matches_the_single_fork_with_mixed_suffixes(engine, suffixes):
    """With suffixes of different lengths a chunk pads to its own longest suffix, not the request's, so the kernels
    reduce in a different order.  Measured on decider-2b: at most 2.9e-5 of probability over four rows (223 to 450-token
    suffixes) and 4.5e-5 over 32 (223 to 471), argmax unchanged.  A request whose suffixes are further apart moves
    further: 1.5e-2 on a 32-row sample row with 182 to 932-token suffixes, still well inside the server's 0.05 tolerance
    against the eager reference."""
    items = _prompt(engine.tok, suffix_tokens=suffixes)
    full = torch.cat(engine.score_shared(items, rows_per_fork=len(items)))
    for m in (1, 3):
        got = torch.cat(engine.score_shared(items, rows_per_fork=m))
        assert bool(got.isfinite().all()), m
        assert (got.argmax(-1) == full.argmax(-1)).all(), m
        assert float((got - full).abs().max()) <= 1e-4, (len(items), m, float((got - full).abs().max()))


@pytest.mark.cuda
def test_the_fork_budget_bounds_peak_memory(engine):
    """A small budget must lower the peak reserved memory of the same request, and `chunk_rows` must turn a budget into
    the row count the loop uses."""
    from decider import shared_prefix
    items = _prompt(engine.tok, suffix_tokens=(300,) * 8)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    wide = torch.cat(engine.score_shared(items, rows_per_fork=len(items)))
    peak_wide = torch.cuda.max_memory_reserved()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    narrow = torch.cat(engine.score_shared(items, rows_per_fork=1))
    peak_narrow = torch.cuda.max_memory_reserved()
    assert peak_narrow < peak_wide, (peak_narrow, peak_wide)
    assert float((narrow - wide).abs().max()) == 0.0
    pre = torch.tensor(items[0]["ids"][:2000], device=engine.dev)[None]
    cache = engine.core(input_ids=pre, use_cache=True).past_key_values
    per_row = shared_prefix.cache_row_bytes(cache)
    assert per_row > 0
    assert shared_prefix.chunk_rows(per_row, 32, budget_bytes=per_row * 3) == 3
    assert shared_prefix.chunk_rows(per_row, 32, budget_bytes=per_row // 2) == 1
