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


def build_schema_first(example, tok, rng=None, max_options=NARROW, max_ctx_tokens=1536, chat=None):
    """Schema-first layout: all question/option blocks, then the context, then one answer slot per question.

        Question 1: ...\nOptions:\n(A) ...          <- prefix: depends only on the questions, so its cache (attention KV and
        \n\nQuestion 2: ...                            delta-net states) is computed once per schema and reused for every state
        \n\nContext:\n<state>\n\nAnswer 1: (\nAnswer 2: (

    The three parts are tokenized separately, so `ids[:prefix_len]` is identical for every state.
    chat: a ChatTemplate for a chat-layout model (see `build_chat`); None renders the plain layout above."""
    rng = rng or random
    perms = [_select(q, rng, max_options) for q in example.qs]
    pre = schema_prefix_ids(tok, example.qs, perms, chat=chat)
    suf, slots = schema_suffix_ids(tok, example.context, len(example.qs), max_ctx_tokens, chat=chat)
    return dict(ids=pre + suf, slots=[len(pre) + s for s in slots], golds=[p.index(q.gold) if q.gold in p else -1 for p, q in zip(perms, example.qs)],
                nopts=[len(p) for p in perms], perms=perms, prefix_len=len(pre))


def schema_prefix_ids(tok, qs, perms=None, chat=None):
    """Token ids of the question/option blocks (the cacheable part of the schema-first layout).
    chat: a ChatTemplate; the prefix then starts with the template head (the user turn opens before the first question)."""
    multi = len(qs) > 1; pre = [] if chat is None else list(chat.head)
    for k, q in enumerate(qs):
        opts = perms[k] if perms is not None else list(range(len(q.options)))
        pre += tok.encode(f"{chr(10) * 2 if k else ''}Question{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False) + _options_ids(tok, q, opts)
    return pre


def schema_suffix_ids(tok, context, n_q, max_ctx_tokens=1536, chat=None):
    """Token ids after the schema prefix: the context and one answer slot per question. Returns (ids, slot positions in ids).
    chat: a ChatTemplate; the context then ends the user turn, and the template tail and the answer pieces follow."""
    ids = tok.encode("\n\nContext:\n", add_special_tokens=False) + tok.encode(context, add_special_tokens=False)[:max_ctx_tokens]; slots = []
    if chat is not None:
        ids += chat.tail
        for k in range(n_q):
            ids += chat.answer_ids(tok, k, n_q > 1); slots.append(len(ids) - 1)
        return ids, slots
    for k in range(n_q):
        ids += tok.encode(f"{chr(10) * 2 if k == 0 else chr(10)}Answer{' ' + str(k + 1) if n_q > 1 else ''}: (", add_special_tokens=False); slots.append(len(ids) - 1)
    return ids, slots


def build(example, tok, rng=None, max_options=NARROW, max_ctx_tokens=1536, layout="state_first", chat=None):
    """Returns dict(ids=list[int], slots=list[int], golds=list[int], nopts=list[int], perms=list[list[int]]).
    chat: a ChatTemplate (chat_template(tok)) for a model trained in the chat layout (decider_config.json "layout": "chat");
    None, the default, is the plain layout every earlier model uses, unchanged."""
    if layout == "schema_first":
        return build_schema_first(example, tok, rng, max_options, max_ctx_tokens, chat=chat)
    if chat is not None:
        return build_chat(example, tok, chat, rng, max_options, max_ctx_tokens)
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


# ---- chat layout (1.2.0) ----------------------------------------------------------------------------------------------
# A model whose decider_config.json has "layout": "chat" (or "chat_template": true) was trained with every prompt wrapped
# in its tokenizer's chat template: one user turn holding the plain content, thinking switched off, and the answer pieces in
# the assistant turn.  For Qwen3.5, state-first:
#
#     <|im_start|>user\n  Context:\n<state>  \n\nQuestion: <q>\nOptions:\n(A) ..\n(B) ..
#     <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n  Answer: (
#
# With several questions in one row all question blocks come first and the answer pieces ("Answer 1: (", "\nAnswer 2: (",
# ...) follow the template tail.  Schema-first: the user turn holds the question blocks and then "\n\nContext:\n<state>".
# Every part is tokenized on its own (head, context, each question header, each option block, tail, each answer piece);
# the letter is read at the final " (" token as in the plain layout.
LAYOUTS = ("plain", "chat")


def resolve_layout(cfg):
    """The prompt layout a decider_config.json names: "plain" (no "layout" key and no "chat_template": true, which is every
    released model) or "chat".  Raises ValueError for any other value, so a model trained in a layout
    this version does not know is refused instead of being read in the wrong one."""
    cfg = cfg or {}
    layout = cfg.get("layout")
    if layout is None:
        layout = "chat" if cfg.get("chat_template") is True else "plain"
    if layout not in LAYOUTS:
        raise ValueError(f"decider_config.json names the prompt layout {layout!r}; this version of decider-ai knows "
                         f"{', '.join(repr(x) for x in LAYOUTS)}. Upgrade decider-ai, or check the model's config.")
    if layout == "plain" and cfg.get("chat_template") is True:
        raise ValueError('decider_config.json says "layout": "plain" and "chat_template": true; these contradict each other')
    return layout


def with_layout(cfg, layout=None):
    """cfg with its prompt layout replaced by `layout` ("plain" or "chat"; None keeps cfg as it is).  This is how a server reads
    a stock checkpoint, which has no decider_config.json, in the chat layout (DECIDER_LAYOUT=chat).  Raises ValueError for any
    other value."""
    cfg = dict(cfg or {})
    if layout is None or layout == "":
        return cfg
    if layout not in LAYOUTS:
        raise ValueError(f"layout {layout!r}: expected one of {', '.join(repr(x) for x in LAYOUTS)}")
    cfg.pop("chat_template", None)
    cfg["layout"] = layout
    return cfg


class ChatTemplate:
    """Token ids of the tokenizer's chat template around one user turn, and the answer pieces of the assistant turn.

    head/tail come from tok.apply_chat_template([user turn], add_generation_prompt=True) with thinking disabled
    (enable_thinking=False where the template knows the switch; a thinking block the template leaves open is closed here),
    with no system prompt.  This is the computation of the chat-layout research server the chat models were trained against."""
    SENTINEL = "@@DECIDER_USER_CONTENT@@"

    def __init__(self, tok, answer="Answer"):
        if not getattr(tok, "chat_template", None):
            raise ValueError('the model is configured for the chat layout ("layout": "chat") but its tokenizer has no chat template')
        msgs = [{"role": "user", "content": self.SENTINEL}]
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        head, tail = s.split(self.SENTINEL)
        for opened, closed in (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>")):
            if tail.endswith(opened):
                tail += closed
        self.head_text, self.tail_text = head, tail
        self.head = tok.encode(head, add_special_tokens=False)
        self.tail = tok.encode(tail, add_special_tokens=False)
        self.answer = answer
        names, ids, _ = label_table(tok)
        pre = tok.encode(f"{answer}: (", add_special_tokens=False)
        for n, i in list(zip(names, ids))[:NARROW]:
            if tok.encode(f"{answer}: ({n}", add_special_tokens=False) != pre + [i]:
                raise ValueError(f"chat layout: the label {n!r} does not tokenize as one token after {answer!r}: (")
        if tok.encode(tail + f"{answer}: (", add_special_tokens=False) != self.tail + pre:
            raise ValueError("chat layout: the chat template tail merges with the answer prefix")

    def answer_ids(self, tok, k, multi):
        """Token ids of answer piece k: "Answer: (" (or "Answer k: (" with several questions), later pieces after a newline."""
        return tok.encode(f"{'' if k == 0 else chr(10)}{self.answer}{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)


_CHAT = {}


def chat_template(tok):
    """The ChatTemplate of a tokenizer, cached per tokenizer object."""
    key = id(tok)
    if key not in _CHAT:
        _CHAT[key] = (tok, ChatTemplate(tok))           # keep tok alive so its id is not reused
    return _CHAT[key][1]


def chat_for(tok, cfg):
    """The ChatTemplate to pass as `chat=` for a model with this decider_config.json, or None for a plain-layout model."""
    return chat_template(tok) if resolve_layout(cfg) == "chat" else None


def load_decider_config(path):
    """decider_config.json of a model folder or a Hub repository id; {} when the model has none (a raw base model)."""
    import json, os
    if os.path.isdir(path):
        f = os.path.join(path, "decider_config.json")
        return json.load(open(f)) if os.path.exists(f) else {}
    try:
        from huggingface_hub import hf_hub_download
        return json.load(open(hf_hub_download(path, "decider_config.json")))
    except Exception:
        return {}


def chat_for_model(path, tok):
    """chat_for(tok, the model's decider_config.json): what every script that builds prompts for `path` passes as `chat=`."""
    return chat_for(tok, load_decider_config(path))


def build_chat(example, tok, chat, rng=None, max_options=NARROW, max_ctx_tokens=1536):
    """State-first chat layout (see the comment above LAYOUTS).  Same return fields as `build`.  max_ctx_tokens caps the
    tokens of "Context:\\n" + state, as in the plain layout; the template head and tail are not counted."""
    rng = rng or random
    qs = example.qs; multi = len(qs) > 1
    ids = list(chat.head) + tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    golds, nopts, perms = [], [], []
    for k, q in enumerate(qs):
        opts = _select(q, rng, max_options)
        ids += tok.encode(f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False) + _options_ids(tok, q, opts)
        golds.append(opts.index(q.gold) if q.gold in opts else -1); nopts.append(len(opts)); perms.append(opts)
    ids += chat.tail
    slots = []
    for k in range(len(qs)):
        ids += chat.answer_ids(tok, k, multi); slots.append(len(ids) - 1)
    return dict(ids=ids, slots=slots, golds=golds, nopts=nopts, perms=perms)


def letter_ids(tok):
    ids = label_table(tok)[1]
    for j, L in enumerate(LETTERS):
        assert tok.encode(L, add_special_tokens=False) == [ids[j]], L
    return ids


def render(example, tok, **kw):
    b = build(example, tok, **kw)
    return tok.decode(b["ids"])
