"""Prompt construction.  One context, N typed questions, N answer slots.

All N decisions are read from a single forward pass: the logits at each
"Answer k: (" slot are restricted to the option-letter tokens.  No answer
letters are ever inserted, so slot k sees the context and all questions but
no earlier answers (the decisions are conditionally independent given input).
"""
import random

LETTERS = "ABCDEFGHIJ"
MAX_OPTIONS = len(LETTERS)


def build(example, tok, rng=None, max_options=MAX_OPTIONS, max_ctx_tokens=1536):
    """Returns dict(ids=list[int], slots=list[int], golds=list[int], nopts=list[int], perms=list[list[int]])."""
    rng = rng or random
    ctx_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    ids = list(ctx_ids)
    slots, golds, nopts, perms = [], [], [], []
    multi = len(example.qs) > 1
    for k, q in enumerate(example.qs):
        opts = list(range(len(q.options)))
        if len(opts) > max_options:
            others = [i for i in opts if i != q.gold]
            keep = rng.sample(others, max_options - 1) + [q.gold]
            opts = keep
        rng.shuffle(opts)
        lines = [f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"]
        for j, oi in enumerate(opts):
            lines.append(f"\n({LETTERS[j]}) {q.options[oi]}")
        lines.append(f"\nAnswer{' ' + str(k + 1) if multi else ''}: (")
        piece = tok.encode("".join(lines), add_special_tokens=False)
        ids.extend(piece)
        slots.append(len(ids) - 1)          # position of " (" token
        golds.append(opts.index(q.gold) if q.gold in opts else -1)
        nopts.append(len(opts))
        perms.append(opts)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)


def letter_ids(tok):
    out = []
    for L in LETTERS:
        t = tok.encode(L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        out.append(t[0])
    return out


def render(example, tok, **kw):
    b = build(example, tok, **kw)
    return tok.decode(b["ids"])
