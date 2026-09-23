"""Prompt cases shared by tests/test_layout.py: every place the package turns a request into token ids, over a fixed set of
requests.  `plain_cases` renders them for a plain-layout model (every released checkpoint up to decider-2b v10, 4B, 35B, 0.8B,
vision); `chat_cases` renders the same requests for a chat-layout model (decider-2b v11).

`decider_items` stands in for decider.infer.Decider's prompt building without loading a model: it calls the Decider's own
`_decide_items` / `_system_one_items` on an instance that has only a tokenizer and the config flags."""
import hashlib, json, random
from types import SimpleNamespace


class NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


class Q:
    def __init__(self, text, options, gold=0): self.text, self.options, self.gold = text, options, gold


class Ex:
    def __init__(self, context, qs): self.context, self.qs, self.task, self.image = context, qs, "test", None


def _long(n, seed=3):
    rng = random.Random(seed)
    return " ".join(rng.choice(["alpha", "beta", "gamma", "delta", "ticket", "refund", "{", "}", "é"]) for _ in range(n))


STATES = ["a customer wants a refund",
          "",
          "árvíztűrő tükörfúrógép 🙂\n\nline two",
          json.dumps({"ticket": {"body": "x" * 900, "tags": ["a", "b"]}, "rows": [{"v": i} for i in range(12)]}),
          _long(3000)]

EXAMPLES = [
    ("one question", STATES[0], [Q("Which queue?", ["billing", "technical", "sales"])]),
    ("three questions", STATES[0], [Q("Which queue?", ["billing", "technical", "sales"]), Q("Urgent?", ["no", "yes"], 1),
                                    Q("Sentiment?", ["angry", "neutral", "happy", "none of the above"], 3)]),
    ("empty state", STATES[1], [Q("Yes or no?", ["no", "yes"])]),
    ("unicode", STATES[2], [Q("Which?", ["a", "b"])]),
    ("json state", STATES[3], [Q("Which?", ["one", "two", "three"]), Q("Flag?", ["no", "yes"])]),
    ("long state", STATES[4], [Q("Which topic?", ["alpha", "beta", "gamma", "delta"])]),
    ("wide 40", STATES[0], [Q("Which label?", [f"option {i}" for i in range(40)], 17)]),
    ("wide 255", STATES[0], [Q("Which label?", [f"o{i}" for i in range(255)], 200)]),
    ("wide and narrow", STATES[3], [Q("Which label?", [f"label {i}" for i in range(24)], 5), Q("Urgent?", ["no", "yes"])]),
    ("ten options", STATES[0], [Q("Which digit?", [str(i) for i in range(10)], 9)]),
    ("eleven options", STATES[0], [Q("Which digit?", [str(i) for i in range(11)], 10)]),
]

QUESTIONS = {
    "queue": {"type": "choice", "instructions": "Which queue?", "criteria": ["billing", "technical", "sales"]},
    "flag": {"type": "noul", "instructions": "Does this need a human?"},
    "sev": {"type": "score", "instructions": "How severe?", "criteria": ["none", "low", "medium", "high"]},
    "wide": {"type": "choice", "instructions": "Which label?", "criteria": {f"L{i}": f"label {i}" for i in range(24)}},
    "abstain": {"type": "choice", "instructions": "Which team?", "criteria": ["sales", "support", "None of the above"]},
}
S1_STATES = [STATES[0], {"ticket": {"body": "x" * 900, "tags": ["a", "b"]}, "rows": [{"v": i} for i in range(12)]}, "", STATES[4]]
SCHEMA = {"Which team?": {"type": "choice", "options": ["billing", "technical", "none of the above"]},
          "Refund?": {"type": "bool"},
          "Severity?": {"type": "scale", "legend": {"0": "none", "1": "low", "2": "high"}},
          "Wide?": {"type": "choice", "options": [f"w{i}" for i in range(30)]}}


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _item(it):
    keep = ("ids", "slots", "golds", "nopts", "perms", "prefix_len")
    return {k: (list(it[k]) if k == "ids" else it[k]) for k in keep if k in it}


def prompt_cases(tok, chat=None):
    """decider.prompt / decider.prompt_fast over EXAMPLES and STATES."""
    from decider import prompt as P
    from decider.prompt_fast import build_rows
    kw = {} if chat is None else {"chat": chat}
    out = {}
    for name, ctx, qs in EXAMPLES:
        ex = Ex(ctx, qs)
        for layout in ("state_first", "schema_first"):
            for cap in (1536, 64, 32768):
                out[f"build/{layout}/{name}/cap{cap}/keep"] = _item(P.build(ex, tok, NoShuffle(), max_options=P.MAX_OPTIONS, max_ctx_tokens=cap, layout=layout, **kw))
            out[f"build/{layout}/{name}/rng0"] = _item(P.build(ex, tok, random.Random(0), layout=layout, **kw))
        out[f"schema_prefix/{name}"] = P.schema_prefix_ids(tok, qs, **kw)
    for i, st in enumerate(STATES):
        for n_q in (1, 3):
            for cap in (64, 1536):
                out[f"schema_suffix/{i}/{n_q}/{cap}"] = P.schema_suffix_ids(tok, st, n_q, cap, **kw)
        for rows_name, rows in (("independent", [[("Which queue?", ["billing", "technical", "sales"])], [("Urgent?", ["no", "yes"])],
                                                  [("Which label?", [f"option {j}" for j in range(40)])]]),
                                ("packed", [[("Which queue?", ["billing", "technical", "sales"]), ("Urgent?", ["no", "yes"]),
                                             ("Which label?", [f"option {j}" for j in range(40)])]]),
                                ("many", [[(f"Question number {j}?", ["no", "yes"])] for j in range(12)])):
            for cap in (32768, 64):
                items, ctx_len = build_rows(tok, st, rows, max_ctx_tokens=cap, **kw)
                out[f"build_rows/{i}/{rows_name}/{cap}"] = [[_item(it) for it in items], ctx_len]
    return out


def serve_cases(tok, chat=None, neutralize=True):
    """decider.serve.prepare and decider.serve._prepare_decide (needs fastapi)."""
    from decider import serve
    kw = {} if chat is None else {"chat": chat}
    out = {}
    for i, st in enumerate(S1_STATES):
        for independent, isolated in ((True, False), (True, True), (False, False), (False, True)):
            for cap in (32768, 64):
                rqs, index, items, ctx_len = serve.prepare(tok, st, QUESTIONS, independent, isolated, cap, **kw)
                out[f"prepare/{i}/{independent}/{isolated}/{cap}"] = [index, [_item(it) for it in items], ctx_len]
    saved = (serve.eng, serve.NEUTRALIZE_NONE, getattr(serve, "CHAT", None))
    try:
        serve.eng = SimpleNamespace(tok=tok); serve.NEUTRALIZE_NONE = neutralize
        if chat is not None:
            serve.CHAT = chat
        for i, st in enumerate(STATES):
            qs, it = serve._prepare_decide(st, SCHEMA)
            out[f"decide/{i}"] = [[q["options"] for q in qs], _item(it)]
    finally:
        serve.eng, serve.NEUTRALIZE_NONE = saved[0], saved[1]
        if hasattr(serve, "CHAT"):
            serve.CHAT = saved[2]
    return out


def decider_items(tok, chat=None, neutralize=True, isolated_levels=True):
    """decider.infer.Decider's prompt building (decide_batch, system_one), on an instance without a model."""
    from decider.infer import Decider
    d = object.__new__(Decider)
    d.m = SimpleNamespace(tok=tok); d.eng = None; d.neutralize_none = neutralize; d.isolated_levels = isolated_levels
    d.schema_first = False; d.chat = chat
    out = {}
    reqs = [(st, [{"question": "Which queue?", "options": ["billing", "technical", "none of the above"]},
                  {"question": "Urgent?", "options": ["no", "yes"]},
                  {"question": "Which label?", "options": [f"option {j}" for j in range(40)]}]) for st in STATES]
    for cap in (1536, 64):
        r2, items = d._decide_items(reqs, cap)
        out[f"decide_batch/{cap}"] = [[[q["options"] for q in qs] for _, qs in r2], [_item(it) for it in items]]
    for i, st in enumerate(S1_STATES):
        for independent in (True, False):
            for layout in ("state_first", "schema_first"):
                for isolated in (None, False):
                    rqs, index, items = d._system_one_items(st, QUESTIONS, independent, 32768, layout, isolated)
                    out[f"system_one/{i}/{independent}/{layout}/{isolated}"] = [index, [_item(it) for it in items]]
    return out
