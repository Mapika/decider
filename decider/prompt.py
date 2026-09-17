"""Prompt construction.  One context, N typed questions, N answer slots.

All N decisions are read from a single forward pass: the logits at each
"Answer k: (" slot are restricted to the option-letter tokens.  No answer
letters are ever inserted, so slot k sees the context and all questions but
no earlier answers (the decisions are conditionally independent given input).
"""
import random

LETTERS = "ABCDEFGHIJ"
NARROW = len(LETTERS)      # <= NARROW options: the original "(A) .. (J)" rendering, tokenized as a string (unchanged since v1)
MAX_OPTIONS = 255          # width of the label head. > NARROW options: "wide" rendering, one label token per option:
                           # A..Z then the first 229 two-letter upper-case strings that are single tokens (AA, AB, ...)
ABSTAIN_PREFIXES = ("none of the above", "none of these", "not listed", "no suitable", "does not apply", "cannot tell")
ABSTAIN_EXACT = ("other", "unsure", "something else", "neither of these", "other / not covered")


def is_abstain_option(o):
    o = o.strip().lower()
    return o.startswith(ABSTAIN_PREFIXES) or o in ABSTAIN_EXACT


_LABELS = {}
_OPT_CACHE = {}


def _enc_opt(tok, text):
    """Token ids of ") <option text>" (cached: fixed label sets repeat the same strings millions of times)."""
    key = (id(tok), text)
    v = _OPT_CACHE.get(key)
    if v is None:
        v = tok.encode(f") {text}", add_special_tokens=False)
        if len(_OPT_CACHE) < 2_000_000:
            _OPT_CACHE[key] = v
    return v


def label_table(tok):
    """(label strings, label token ids), MAX_OPTIONS entries; the first NARROW are A..J so narrow questions are unchanged."""
    key = id(tok)
    if key not in _LABELS:
        import string
        U = string.ascii_uppercase
        names = list(U) + [a + b for a in U for b in U]
        out = []
        for n in names:
            t = tok.encode(n, add_special_tokens=False)
            if len(t) == 1:
                out.append((n, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        assert len(out) == MAX_OPTIONS and len({i for _, i in out}) == MAX_OPTIONS
        _LABELS[key] = ([n for n, _ in out], [i for _, i in out],
                        tok.encode("\n(", add_special_tokens=False))
    return _LABELS[key]


def _select(q, rng, max_options):
    opts = list(range(len(q.options)))
    if len(opts) > max_options:
        # always keep the gold and any abstain-style option (its mere presence must not carry information)
        forced = {q.gold} | {i for i, o in enumerate(q.options) if is_abstain_option(o)}
        others = [i for i in opts if i not in forced]
        opts = rng.sample(others, max_options - len(forced)) + list(forced)
    rng.shuffle(opts)
    return opts


def _options_ids(tok, q, opts):
    if len(opts) <= NARROW:
        return tok.encode("".join(f"\n({LETTERS[j]}) {q.options[oi]}" for j, oi in enumerate(opts)), add_special_tokens=False)
    _, lab_ids, open_ids = label_table(tok); out = []
    for j, oi in enumerate(opts):
        out += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])
    return out


def build_schema_first(example, tok, rng=None, max_options=NARROW, max_ctx_tokens=1536):
    """Schema-first layout: all question/option blocks, then the context, then one answer slot per question.

        Question 1: ...\nOptions:\n(A) ...          <- prefix: depends only on the questions, so its cache (attention KV and
        \n\nQuestion 2: ...                            delta-net states) is computed once per schema and reused for every state
        \n\nContext:\n<state>\n\nAnswer 1: (\nAnswer 2: (

    The three parts are tokenized separately, so `ids[:prefix_len]` is identical for every state."""
    rng = rng or random
    perms = [_select(q, rng, max_options) for q in example.qs]
    pre = schema_prefix_ids(tok, example.qs, perms)
    suf, slots = schema_suffix_ids(tok, example.context, len(example.qs), max_ctx_tokens)
    return dict(ids=pre + suf, slots=[len(pre) + s for s in slots], golds=[p.index(q.gold) if q.gold in p else -1 for p, q in zip(perms, example.qs)],
                nopts=[len(p) for p in perms], perms=perms, prefix_len=len(pre))


def schema_prefix_ids(tok, qs, perms=None):
    """Token ids of the question/option blocks (the cacheable part of the schema-first layout)."""
    multi = len(qs) > 1; pre = []
    for k, q in enumerate(qs):
        opts = perms[k] if perms is not None else list(range(len(q.options)))
        pre += tok.encode(f"{chr(10) * 2 if k else ''}Question{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False) + _options_ids(tok, q, opts)
    return pre


def schema_suffix_ids(tok, context, n_q, max_ctx_tokens=1536):
    """Token ids after the schema prefix: the context and one answer slot per question. Returns (ids, slot positions in ids)."""
    ids = tok.encode("\n\nContext:\n", add_special_tokens=False) + tok.encode(context, add_special_tokens=False)[:max_ctx_tokens]; slots = []
    for k in range(n_q):
        ids += tok.encode(f"{chr(10) * 2 if k == 0 else chr(10)}Answer{' ' + str(k + 1) if n_q > 1 else ''}: (", add_special_tokens=False); slots.append(len(ids) - 1)
    return ids, slots


def build(example, tok, rng=None, max_options=NARROW, max_ctx_tokens=1536, layout="state_first"):
    """Returns dict(ids=list[int], slots=list[int], golds=list[int], nopts=list[int], perms=list[list[int]])."""
    if layout == "schema_first":
        return build_schema_first(example, tok, rng, max_options, max_ctx_tokens)
    rng = rng or random
    ctx_ids = tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    ids = list(ctx_ids)
    slots, golds, nopts, perms = [], [], [], []
    multi = len(example.qs) > 1
    for k, q in enumerate(example.qs):
        opts = list(range(len(q.options)))
        if len(opts) > max_options:
            # always keep the gold and any abstain-style option (its mere presence must not carry information)
            forced = {q.gold} | {i for i, o in enumerate(q.options) if is_abstain_option(o)}
            others = [i for i in opts if i not in forced]
            keep = rng.sample(others, max_options - len(forced)) + list(forced)
            opts = keep
        rng.shuffle(opts)
        head = f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:"
        tail = f"\nAnswer{' ' + str(k + 1) if multi else ''}: ("
        if len(opts) <= NARROW:
            lines = [head] + [f"\n({LETTERS[j]}) {q.options[oi]}" for j, oi in enumerate(opts)] + [tail]
            piece = tok.encode("".join(lines), add_special_tokens=False)
        else:                                   # wide: "\n(" + <label token> + ") text", built from ids so every label is one token
            _, lab_ids, open_ids = label_table(tok)
            piece = tok.encode(head, add_special_tokens=False)
            for j, oi in enumerate(opts):
                piece += open_ids + [lab_ids[j]] + _enc_opt(tok, q.options[oi])
            piece += tok.encode(tail, add_special_tokens=False)
        ids.extend(piece)
        slots.append(len(ids) - 1)          # position of " (" token
        golds.append(opts.index(q.gold) if q.gold in opts else -1)
        nopts.append(len(opts))
        perms.append(opts)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)


def letter_ids(tok):
    ids = label_table(tok)[1]
    for j, L in enumerate(LETTERS):
        assert tok.encode(L, add_special_tokens=False) == [ids[j]], L
    return ids


def render(example, tok, **kw):
    b = build(example, tok, **kw)
    return tok.decode(b["ids"])
