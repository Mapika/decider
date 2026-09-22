"""decider.serve.prepare must plan and tokenize exactly the rows the 1.0.x server (decider.serve_v1._prepare_s1) built.

Needs fastapi (to import decider.serve) and a tokenizer; no torch.  The reference is serve_v1's row construction with the
tokenizer passed in (that function closes over the module's engine).
"""
import pytest

pytest.importorskip("fastapi")

from decider import systemone as S1
from decider.prompt import build, MAX_OPTIONS
from decider.serve import prepare


class _NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


class _Q:
    def __init__(self, text, options): self.text, self.options, self.gold = text, options, 0


class _Ex:
    def __init__(self, context, qs): self.context, self.qs, self.task, self.image = context, qs, "test", None


def _prepare_s1(tok, state, questions, independent, isolated, max_state_tokens=32768):
    """decider/serve_v1.py `_prepare_s1`, with the tokenizer as an argument."""
    ctx = S1.render_state(state); rqs = {k: S1.render_question(v) for k, v in questions.items()}
    flat, index = S1.plan_rows(rqs, isolated and independent)
    rows = [[r] for r in flat] if independent else [flat]
    items = [build(_Ex(ctx, [_Q(r["question"], list(r["options"])) for r in row]), tok, _NoShuffle(),
                   max_options=MAX_OPTIONS, max_ctx_tokens=max_state_tokens) for row in rows]
    return (rqs, index), items


QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which queue?", "criteria": ["billing", "technical", "sales"]},
    "flag": {"type": "noul", "instructions": "Does this need a human?"},
    "sev": {"type": "score", "instructions": "How severe?", "criteria": ["none", "low", "medium", "high"]},
    "wide": {"type": "choice", "instructions": "Which label?", "criteria": {f"L{i}": f"label {i}" for i in range(24)}},
}
STATES = ["the checkout is down for all customers",
          {"ticket": {"body": "x" * 900, "tags": ["a", "b"]}, "rows": [{"v": i} for i in range(12)]},
          ""]


@pytest.mark.parametrize("independent,isolated", [(True, False), (True, True), (False, False)])
def test_prepare_matches_serve(tok, independent, isolated):
    for state in STATES:
        (rqs_a, idx_a), ref = _prepare_s1(tok, state, QUESTIONS, independent, isolated)
        rqs_b, idx_b, items, ctx_len = prepare(tok, state, QUESTIONS, independent, isolated)
        assert list(rqs_a) == list(rqs_b) and idx_a == idx_b
        assert len(items) == len(ref)
        for a, b in zip(items, ref):
            assert a["ids"] == b["ids"]
            assert a["slots"] == b["slots"] and a["nopts"] == b["nopts"]
        from decider.prompt_fast import unique_tokens
        assert unique_tokens(items, ctx_len) == S1.unique_tokens(ref)


def test_empty_question_map_has_zero_usage(tok):
    from decider.prompt_fast import unique_tokens
    (rqs_a, idx_a), ref = _prepare_s1(tok, "hello", {}, True, False)
    rqs_b, idx_b, items, ctx_len = prepare(tok, "hello", {}, True, False)
    assert items == ref == [] and rqs_b == {} and idx_b == []
    assert unique_tokens(items, ctx_len) == S1.unique_tokens(ref) == 0


def test_assemble_shape_is_unchanged(tok):
    """The answer dicts come out of systemone.assemble, so every wire field is the one decider.serve emits."""
    import random
    rqs, index, items, _ = prepare(tok, "a state", QUESTIONS, True, True)
    rng = random.Random(0)
    probs = []
    for it in items:
        n = it["nopts"][0]
        v = [rng.random() for _ in range(n)]; s = sum(v)
        probs.append([x / s for x in v] + [0.0] * (MAX_OPTIONS - n))
    ans = S1.assemble(rqs, index, probs)
    assert set(ans) == set(QUESTIONS)
    assert set(ans["queue"]) == {"type", "choice", "confidence", "certainty", "probabilities"}
    assert set(ans["flag"]) == {"type", "noul"}
    assert {"type", "score", "confidence", "certainty", "legend", "probabilities", "level_fit", "fit_mass"} == set(ans["sev"])


def test_bad_questions_raise_value_error(tok):
    for bad in ({"q": {"type": "choice", "instructions": "x", "criteria": ["only one"]}},
                {"q": {"type": "score", "instructions": "x", "criteria": list(range(20))}},
                {"q": {"type": "nope", "instructions": "x"}},
                {"q": {"type": "choice", "criteria": ["a", "b"]}}):
        try:
            prepare(tok, "s", bad, True, False)
            raise AssertionError(f"no ValueError for {bad}")
        except ValueError:
            pass
