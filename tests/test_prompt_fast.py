"""prompt_fast.build_rows must produce exactly what prompt.build produces, and its usage count must match systemone.
No torch needed; the `tok` fixture (tests/conftest.py) skips when no tokenizer is available.  DECIDER_TEST_SAMPLE may name a
.jsonl(.gz) of {"state", "questions"} rows to check real requests as well."""
import gzip, json, os, random
import pytest

from decider import systemone as S1
from decider.prompt import build, MAX_OPTIONS
from decider.prompt_fast import build_rows, unique_tokens

SAMPLE = os.environ.get("DECIDER_TEST_SAMPLE", "")


class _NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


class _Q:
    def __init__(self, text, options): self.text, self.options, self.gold = text, options, 0


class _Ex:
    def __init__(self, context, qs): self.context, self.qs, self.task, self.image = context, qs, "test", None


def _reference(tok, ctx, rows, max_ctx_tokens=32768):
    """What prompt.build produces for the same rows (the 1.0.x server called it once per row)."""
    return [build(_Ex(ctx, [_Q(t, list(o)) for t, o in row]), tok, _NoShuffle(),
                  max_options=MAX_OPTIONS, max_ctx_tokens=max_ctx_tokens) for row in rows]


def _cases():
    rng = random.Random(7)
    out = [("short state", "a customer wants a refund", [[("Which queue?", ["billing", "technical", "sales"])]]),
           ("empty-ish state", "", [[("Yes or no?", ["no", "yes"])]]),
           ("unicode state", "árvíztűrő tükörfúrógép 🙂\n\nline two", [[("Which?", ["a", "b"])]]),
           ("json state", json.dumps({"a": [1, 2, 3], "b": "x" * 300}), [[("Which?", ["one", "two", "three"])]])]
    # many rows over one state (the independent layout)
    st = " ".join(rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(500))
    out.append(("many rows", st, [[(f"Question number {i}?", ["no", "yes"])] for i in range(12)]))
    # packed layout: all questions in one row
    out.append(("packed", st, [[(f"Question number {i}?", ["no", "yes"]) for i in range(5)]]))
    # wide rendering (> 10 options uses one label token per option)
    wide = [f"option {i}" for i in range(40)]
    out.append(("wide", "pick one", [[("Which label?", wide)]]))
    out.append(("wide 255", "pick one", [[("Which label?", [f"o{i}" for i in range(255)])]]))
    return out


@pytest.mark.parametrize("name,ctx,rows", [(n, c, r) for n, c, r in _cases()], ids=[n for n, _, _ in _cases()])
def test_build_rows_matches_build(tok, name, ctx, rows):
    items, ctx_len = build_rows(tok, ctx, rows)
    ref = _reference(tok, ctx, rows)
    assert len(items) == len(ref)
    for a, b in zip(items, ref):
        assert a["ids"] == b["ids"], name
        assert a["slots"] == b["slots"]
        assert a["nopts"] == b["nopts"]
    assert all(it["ids"][:ctx_len] == items[0]["ids"][:ctx_len] for it in items)
    assert unique_tokens(items, ctx_len) == S1.unique_tokens(ref)


def test_truncation_matches(tok):
    ctx = "word " * 5000
    rows = [[("Which?", ["a", "b"])]]
    for cap in (16, 64, 1024):
        items, ctx_len = build_rows(tok, ctx, rows, max_ctx_tokens=cap)
        ref = _reference(tok, ctx, rows, max_ctx_tokens=cap)
        assert items[0]["ids"] == ref[0]["ids"]
        assert ctx_len == cap


def test_no_rows_counts_zero_tokens(tok):
    items, ctx_len = build_rows(tok, "some state", [])
    assert items == [] and ctx_len > 0
    assert unique_tokens(items, ctx_len) == 0 == S1.unique_tokens([])


@pytest.mark.skipif(not (SAMPLE and os.path.exists(SAMPLE)), reason="DECIDER_TEST_SAMPLE not set")
def test_build_rows_on_real_suite_rows(tok):
    n = int(os.environ.get("DECIDER_TEST_ROWS", "200"))
    checked = 0
    with (gzip.open if SAMPLE.endswith(".gz") else open)(SAMPLE, "rt") as f:
        for line in f:
            if checked >= n:
                break
            r = json.loads(line)
            try:
                ctx = S1.render_state(r["state"])
                rqs = {k: S1.render_question(v) for k, v in r["questions"].items()}
                flat, _ = S1.plan_rows(rqs, False)
            except ValueError:
                continue
            rows = [[(x["question"], list(x["options"]))] for x in flat]
            items, ctx_len = build_rows(tok, ctx, rows)
            ref = _reference(tok, ctx, rows)
            for a, b in zip(items, ref):
                assert a["ids"] == b["ids"] and a["slots"] == b["slots"] and a["nopts"] == b["nopts"], r["id"]
            assert unique_tokens(items, ctx_len) == S1.unique_tokens(ref), r["id"]
            checked += 1
    assert checked > 0
