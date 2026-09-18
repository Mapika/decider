"""Prompt rendering with the real tokenizer (downloads the Qwen3.5-2B tokenizer; no torch needed)."""
import random, pytest
from decider import prompt

try:
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
except Exception as e:                                                       # offline CI
    TOK = None
    REASON = f"tokenizer unavailable: {e}"

needs_tok = pytest.mark.skipif(TOK is None, reason="tokenizer unavailable")


class Q:
    def __init__(self, text, options, gold=0): self.text, self.options, self.gold = text, options, gold


class Ex:
    def __init__(self, context, qs): self.context, self.qs, self.task, self.image = context, qs, "test", None


def test_abstain_detection():
    assert prompt.is_abstain_option("None of the above")
    assert prompt.is_abstain_option(" other ")
    assert not prompt.is_abstain_option("other services")
    assert not prompt.is_abstain_option("billing")


@needs_tok
def test_label_table_is_255_single_tokens_starting_with_a_to_j():
    names, ids, _ = prompt.label_table(TOK)
    assert len(names) == len(ids) == prompt.MAX_OPTIONS == 255
    assert names[:10] == list("ABCDEFGHIJ") and len(set(ids)) == 255
    assert prompt.letter_ids(TOK) == ids


@needs_tok
def test_state_first_prompt_has_one_slot_per_question_at_the_answer_paren():
    ex = Ex("My card was charged twice.", [Q("Which team?", ["billing", "technical", "sales"]), Q("Urgent?", ["no", "yes"], 1)])
    b = prompt.build(ex, TOK, rng=random.Random(0))
    text = TOK.decode(b["ids"])
    assert text.count("Answer") == 2 and text.startswith("Context:\nMy card was charged twice.")
    for q, perm, gold in zip(ex.qs, b["perms"], b["golds"]):                # options are permuted; perms maps labels back
        assert sorted(perm) == list(range(len(q.options))) and perm[gold] == q.gold
        for j, oi in enumerate(perm):
            assert f"({prompt.LETTERS[j]}) {q.options[oi]}" in text
    for q, slot in zip(ex.qs, b["slots"]):
        assert TOK.decode(b["ids"][: slot + 1]).endswith("(")                 # the slot is the token before the label
    assert list(b["nopts"]) == [3, 2]


@needs_tok
def test_schema_first_prompt_puts_questions_before_the_state():
    ex = Ex("My card was charged twice.", [Q("Which team?", ["billing", "technical"])])
    b = prompt.build(ex, TOK, rng=random.Random(0), layout="schema_first")
    text = TOK.decode(b["ids"])
    assert text.index("Which team?") < text.index("Context") < text.index("charged twice")
    prefix = prompt.schema_prefix_ids(TOK, ex.qs)
    assert b["ids"][: len(prefix)] == list(prefix)                           # the cacheable prefix is a literal prefix of the prompt


@needs_tok
def test_wide_rendering_uses_one_label_token_per_option():
    opts = [f"label_{i}" for i in range(60)]
    ex = Ex("some text", [Q("Which label?", opts, 41)])
    b = prompt.build(ex, TOK, max_options=255)
    names, ids, _ = prompt.label_table(TOK)
    text = TOK.decode(b["ids"])
    assert list(b["nopts"]) == [60] and f"({names[59]}) label_{b['perms'][0][59]}" in text
    narrow = prompt.build(ex, TOK, rng=random.Random(0), max_options=10)
    assert list(narrow["nopts"]) == [10] and "label_41" in TOK.decode(narrow["ids"])   # sub-sampling keeps the gold option
    assert narrow["perms"][0][narrow["golds"][0]] == 41
