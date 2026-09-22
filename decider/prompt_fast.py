"""Row construction for the systemone path that tokenizes the state once per request instead of once per question.

`prompt.build` encodes `"Context:\\n" + state` for every row it is called with, so an independent request with n
questions tokenizes the whole state n times.  The question block is encoded as its own string there and simply appended,
so the ids are exactly `ctx_ids + question_piece`; this module reuses that fact and shares `ctx_ids` across the rows.

`build_rows` returns the same item dicts as
    [prompt.build(Example(ctx, [Q(text, options, 0) for text, options in row]), tok, <no-shuffle rng>,
                  max_options=MAX_OPTIONS, max_ctx_tokens=max_ctx_tokens) for row in rows]
(ids, slots, golds, nopts, perms) and is checked against it in tests/test_prompt_fast.py.
"""
from decider.prompt import LETTERS, NARROW, MAX_OPTIONS, label_table, _enc_opt


def context_ids(tok, context, max_ctx_tokens=32768):
    return tok.encode("Context:\n" + context, add_special_tokens=False)[:max_ctx_tokens]


def question_piece(tok, text, options, k=0, multi=False):
    """Token ids of one `\\n\\nQuestion...: <text>\\nOptions:...\\nAnswer...: (` block, independent of the state."""
    head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {text}\nOptions:"
    tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
    if len(options) <= NARROW:
        return tok.encode(head + "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(options)) + tail,
                          add_special_tokens=False)
    _, lab_ids, open_ids = label_table(tok)
    piece = tok.encode(head, add_special_tokens=False)
    for j, o in enumerate(options):
        piece += open_ids + [lab_ids[j]] + _enc_opt(tok, o)
    return piece + tok.encode(tail, add_special_tokens=False)


def build_rows(tok, context, rows, max_ctx_tokens=32768):
    """rows: list of rows, each a list of (question text, options).  -> (items, len(ctx_ids)).

    Option order is kept as given (no shuffling, no subsetting): systemone.render_question already caps a choice at
    MAX_OPTIONS options, so prompt.build's sampling branch is unreachable here."""
    ctx = context_ids(tok, context, max_ctx_tokens)
    items = []
    for row in rows:
        multi = len(row) > 1
        ids = list(ctx); slots = []; nopts = []
        for k, (text, options) in enumerate(row):
            if not 2 <= len(options) <= MAX_OPTIONS:
                raise ValueError(f"2..{MAX_OPTIONS} options required")
            ids.extend(question_piece(tok, text, options, k, multi))
            slots.append(len(ids) - 1); nopts.append(len(options))
        items.append(dict(ids=ids, slots=slots, golds=[0] * len(row), nopts=nopts,
                          perms=[list(range(len(o))) for _, o in row]))
    return items, len(ctx)


def unique_tokens(items, ctx_len):
    """systemone.unique_tokens(items) without re-walking the shared context (all rows start with the same ctx_len ids).
    No rows (a request with no questions) counts 0, as systemone.unique_tokens does."""
    if not items:
        return 0
    sufs = [it["ids"][ctx_len:] for it in items]
    if len(sufs) < 2:
        return ctx_len + sum(len(s) for s in sufs)
    lcp = 0; short = min(len(s) for s in sufs); s0 = sufs[0]
    while lcp < short and all(s[lcp] == s0[lcp] for s in sufs): lcp += 1
    return ctx_len + lcp + sum(len(s) - lcp for s in sufs)
