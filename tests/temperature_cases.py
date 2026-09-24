"""Probability cases over a stand-in model, for tests/test_temperature.py.

`cases(cfg)` runs every CPU-runnable scoring path of the package (Decider.decide_batch / decide_json / system_one on the
engine and on the eager model, decider.serve's row preparation followed by its engine calls, decider.shared_prefix) over a
tiny deterministic backbone and a byte tokenizer, and returns {case name: probability rows}.  It runs unchanged on the
1.3.0 package, which is how tests/data/temperature_1_3_0_pins.json was produced (see that file's "generated_with"): the
1.4.0 package given a config without a by-type map must return the same numbers.
"""
import types

import torch

from decider.engine_v2 import EngineV2
from decider.model import DecisionModel
from decider.prompt import MAX_OPTIONS

H, V, K = 16, 512, MAX_OPTIONS


class ByteTok:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class _Core(torch.nn.Module):
    """Strictly causal: h[t] from the position-weighted mean of the embeddings up to t (as tests/test_engine_v2.py)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(V, H, generator=g))
            self.mix.weight.copy_(torch.randn(H, H, generator=g) / H ** 0.5)

    def forward(self, input_ids=None, use_cache=False, attention_mask=None, **kw):
        x = self.emb(input_ids)
        w = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        h = torch.tanh(self.mix(torch.cumsum(x, 1) / w)) * 4.0              # * 4: confident enough that T matters
        return types.SimpleNamespace(last_hidden_state=h, past_key_values=None)


class StandInModel:
    """Enough of DecisionModel for EngineV2 and for Decider's eager path (DecisionModel.slot_logits itself)."""
    slot_logits = DecisionModel.slot_logits

    def __init__(self):
        g = torch.Generator().manual_seed(1)
        head = torch.nn.Linear(H, V, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.randn(V, H, generator=g) / H ** 0.5)
        self.tok = ByteTok(); self.letters = torch.arange(K)
        self.lm = types.SimpleNamespace(model=_Core(), lm_head=head)


def engine():
    return EngineV2(model=StandInModel(), device="cpu", use_graphs=False)


def decider(cfg, eng=None, eager=False):
    """A decider.infer.Decider over the stand-in, with the temperatures resolved from `cfg` as Decider.__init__ does."""
    from decider.infer import Decider
    d = object.__new__(Decider)
    eng = eng or engine()
    d.m = eng.m; d.eng = None if eager else eng
    d.chat = None; d.neutralize_none = True; d.isolated_levels = bool(cfg.get("isolated_levels", True)); d.schema_first = False
    d.dev = "cpu"; d.abstain_below = 0.0; d.name = "decider-test"; d._se = None; d._schemas = {}
    try:
        from decider import temperature as TT                                  # 1.4.0
        (d.T, d.T_by_type), (d.T_schema, d.T_schema_by_type) = TT.from_config(cfg)
    except ImportError:                                                         # 1.3.0
        d.T = float(cfg.get("temperature", 1.0)); d.T_schema = float(cfg.get("temperature_schema_first", d.T))
    return d


STATE = "My card was charged twice and checkout is down."
QUESTIONS = {"queue": {"type": "choice", "instructions": "Which queue?", "criteria": ["billing", "technical", "sales"]},
             "flag": {"type": "noul", "instructions": "Needs a human?"},
             "sev": {"type": "score", "instructions": "How severe?", "criteria": ["none", "low", "high"]},
             "who": {"type": "choice", "instructions": "Who?", "criteria": ["user", "admin"]}}
SCHEMA = {"Which team?": {"type": "choice", "options": ["billing", "technical", "none"]},
          "Urgent?": {"type": "bool"},
          "Mood?": {"type": "scale", "legend": ["calm", "annoyed", "angry"]}}
REQUESTS = [(STATE, [{"question": "Which queue?", "options": ["billing", "technical", "sales"]},
                     {"question": "Urgent?", "options": ["no", "yes"]}]),
            ("hey turn the lights off", [{"question": "Intent?", "options": ["lights", "alarm", "music", "none of the above"]}])]


def _rows(x):
    """Probability rows as nested float lists, each row without the zero padding up to MAX_OPTIONS."""
    x = x.tolist() if isinstance(x, torch.Tensor) else x
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [_rows(r) for r in x]
    if isinstance(x, list) and len(x) == MAX_OPTIONS:
        n = len(x)
        while n and x[n - 1] == 0.0:
            n -= 1
        return x[:n]
    return x


def cases(cfg):
    out = {}
    eng = engine()
    for eager in (False, True):
        d = decider(cfg, eng, eager)
        tag = "eager" if eager else "engine"
        out[f"decide_batch/{tag}"] = [[a["probs_list"] for a in r] for r in d.decide_batch(REQUESTS)]
        out[f"decide_json/{tag}"] = d.decide_json(STATE, SCHEMA)
        for independent in (True, False):
            for iso in (True, False):
                out[f"system_one/{tag}/{independent}/{iso}"] = d.system_one(STATE, QUESTIONS, independent=independent, isolated=iso)["answers"]
    from decider import serve
    names = [n for n in ("eng", "CHAT", "LAYOUT", "MODEL_NAME", "TEMP", "TEMP_SCHEMA", "TEMP_BY_TYPE", "TEMP_SCHEMA_BY_TYPE", "RELEASE_DATE",
                         "ISOLATED", "NEUTRALIZE_NONE", "SCHEMA_FIRST") if hasattr(serve, n)]
    saved = {n: getattr(serve, n) for n in names}
    try:
        serve.eng = eng; serve.NEUTRALIZE_NONE = True; serve.CHAT = None
        serve.apply_config(dict(cfg))                                          # the server's own config reading
        for independent in (True, False):
            for iso in (True, False):
                rqs, index, items, _ = serve.prepare(eng.tok, STATE, QUESTIONS, independent, iso)
                out[f"serve/items/{independent}/{iso}"] = [_rows(p) for p in serve._score_items(items)]
                out[f"serve/shared/{independent}/{iso}"] = [_rows(p) for p in serve._score_shared(items)]
        qs, it = serve._prepare_decide(STATE, SCHEMA)
        out["serve/decide"] = [_rows(p) for p in serve._score_items([it])]
    finally:
        for n, v in saved.items():
            setattr(serve, n, v)
    return out
