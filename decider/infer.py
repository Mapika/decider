"""Usable inference API: typed decisions with probabilities, all from one forward pass.

    from decider.infer import Decider
    d = Decider("runs/r2_full/model")
    out = d.decide("My card was charged twice for the same purchase.",
                   [{"question": "Which department should handle this?", "options": ["billing", "technical", "sales"]},
                    {"question": "How urgent is this?", "options": ["low", "medium", "high"]}])
    # -> [{'choice': 'billing', 'confidence': 0.97, 'probs': {...}}, {...}]
"""
import torch
from decider.model import DecisionModel, collate
from decider.prompt import build, MAX_OPTIONS
from dataclasses import dataclass


@dataclass
class Q:
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"; image: bytes = None


NEUTRAL_NONE = "not listed here"


def neutralize_options(options):
    """The training augmentation used the literal 'none of the above', and the model learned that exact string as an
    abstain signal (it abstains even on clear cases when the string is offered). Any option that reads like it is
    rewritten to a neutral phrasing for the model and mapped back in the output."""
    out, back = [], {}
    for o in options:
        key = o.strip().lower()
        if key.startswith("none of the above") or key in ("none of the above", "none", "n/a", "none of these"):
            out.append(NEUTRAL_NONE); back[NEUTRAL_NONE] = o
        else:
            out.append(o)
    return out, back


class CompiledSchema:
    def __init__(self, d, rqs, h, index): self.d, self.rqs, self.h, self.index = d, rqs, h, index

    def batch(self, states, max_state_tokens=32768):
        from decider.systemone import render_state, assemble
        probs = self.d._se.score(self.h, [render_state(s) for s in states], temperature=self.d.T_schema, max_ctx_tokens=max_state_tokens)
        return [{"model": self.d.name, "answers": assemble(self.rqs, self.index, [p.tolist() for p in pr])} for pr in probs]

    def __call__(self, state, max_state_tokens=32768):
        return self.batch([state], max_state_tokens)[0]


class Decider:
    """use_graphs=True (default on CUDA) routes scoring through decider.engine.Engine: shape-bucketed
    CUDA graphs, ~7x lower single-request latency than eager. Set False for CPU or debugging."""
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, temperature=None, abstain_below=0.0, use_graphs=None):
        import json, os
        cfg = {}
        try:                                          # model folder may carry decider_config.json (temperature, flags)
            from huggingface_hub import hf_hub_download
            cfg_path = os.path.join(path, "decider_config.json") if os.path.isdir(path) else hf_hub_download(path, "decider_config.json")
            cfg = json.load(open(cfg_path))
        except Exception:
            pass
        if temperature is None:
            temperature = float(cfg.get("temperature", 1.0))
        self.neutralize_none = bool(cfg.get("neutralize_none", True))   # v4 and earlier learned the literal string as an abstain signal
        if use_graphs is None:
            use_graphs = str(device).startswith("cuda")
        if use_graphs:
            from decider.engine import Engine
            self.eng = Engine(path, device=device, dtype=dtype); self.m = self.eng.m
        else:
            self.eng = None; self.m = DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        self.dev = device; self.T = temperature; self.abstain_below = abstain_below
        self.name = "decider-" + str(cfg.get("version", "dev"))
        self.schema_first = bool(cfg.get("schema_first", False)) and self.eng is not None      # default layout. Questions-first (the cacheable one) costs accuracy
        self.T_schema = float(cfg.get("temperature_schema_first", temperature))                # (about 1.5 points on fixed label sets, more elsewhere): opt in with schema()
        self.isolated_levels = bool(cfg.get("isolated_levels", False))      # Score levels judged one per row (v8+)
        self._se = None; self._schemas = {}

    @torch.no_grad()
    def decide_batch(self, requests, max_ctx_tokens=1536):
        """requests: list of (context:str, questions:list[dict(question, options)]). One forward pass for everything."""
        exs, meta = [], []
        if self.neutralize_none:
            requests = [(context, [dict(q, options=neutralize_options(q["options"])[0], _back=neutralize_options(q["options"])[1]) for q in qs]) for context, qs in requests]
        for context, qs in requests:
            for q in qs:
                assert 2 <= len(q["options"]) <= MAX_OPTIONS, f"2..{MAX_OPTIONS} options required"
            exs.append(Example(context, [Q(q["question"], list(q["options"]), 0) for q in qs], "infer"))
        class _NoShuffle:                      # keep option order as given
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        items = [build(e, self.m.tok, _NoShuffle(), max_options=MAX_OPTIONS, max_ctx_tokens=max_ctx_tokens) for e in exs]
        if self.eng is not None:
            probs = torch.cat(self.eng.score_items(items, temperature=self.T))
        else:
            b = collate(items, self.m.tok.pad_token_id)
            logits = self.m.slot_logits(b["input_ids"].to(self.dev), b["attention_mask"].to(self.dev), b["slot_idx"].to(self.dev),
                                        b["slot_batch"].to(self.dev), b["nopts"].to(self.dev))
            probs = torch.softmax(logits / self.T, -1).cpu()
        out, k = [], 0
        for context, qs in requests:
            res = []
            for q in qs:
                p = probs[k, :len(q["options"])].tolist(); k += 1
                j = max(range(len(p)), key=p.__getitem__); back = q.get("_back", {})
                names = [back.get(o, o) for o in q["options"]]
                res.append(dict(choice=names[j] if p[j] >= self.abstain_below else None, confidence=p[j],
                                probs={o: pi for o, pi in zip(names, p)}, probs_list=p))
            out.append(res)
        return out

    def decide(self, context, questions, **kw):
        return self.decide_batch([(context, questions)], **kw)[0]

    # ---- Jev-shaped interface (decider.systemone): state + {id: Choice | Score | Noul with criteria}
    # ---- schema cache (v7+): the questions are run once, requests only run the state (decider.schema_engine)
    def schema(self, questions, independent=True, isolated=None, compile=False):
        """Compile a fixed set of Jev-shaped questions: schema(state) -> answers; schema.batch([state, ...]) -> [answers]."""
        import json
        from decider.schema_engine import SchemaEngine
        from decider.systemone import render_question
        isolated = self.isolated_levels if isolated is None else isolated
        key = (json.dumps(questions, sort_keys=True, ensure_ascii=False), independent, isolated)
        if key not in self._schemas:
            if self._se is None: self._se = SchemaEngine(self.eng)
            if len(self._schemas) >= 64:                                  # drop the oldest schema and its graphs
                old = next(iter(self._schemas)); hid = self._schemas.pop(old)[1].id
                for k in [k for k in self._se.graphs if k[0] == hid]: del self._se.graphs[k]
            from decider.systemone import plan_rows
            rqs = {k: render_question(v) for k, v in questions.items()}
            rows, index = plan_rows(rqs, isolated and independent)
            h = self._se.prepare(rows, independent=independent, compile=compile)      # compile=True: ~25 s per (batch, length) shape, 1.6x faster after
            self._schemas[key] = (rqs, h, index)
        return CompiledSchema(self, *self._schemas[key])

    def system_one(self, state, questions, independent=True, max_state_tokens=32768, max_fwd_tokens=65536, layout=None, isolated=None):
        layout = layout or ("schema_first" if self.schema_first else "state_first")
        isolated = (self.isolated_levels if isolated is None else isolated) and independent
        if layout == "schema_first" and self.eng is not None:
            return self.schema(questions, independent, isolated)(state, max_state_tokens)
        """independent=True scores every question in its own row (state + that question only), so adding, removing or
        reordering questions cannot change any other answer; the state is run once and its cache forked to every
        question (Engine.score_shared).  independent=False packs all questions behind one copy of the state in one row
        (later questions can then see earlier question texts)."""
        from decider.systemone import render_state, render_question, unique_tokens, plan_rows, assemble
        ctx = render_state(state); rqs = {k: render_question(v) for k, v in questions.items()}
        opts = (lambda r: neutralize_options(r["options"])[0]) if self.neutralize_none else (lambda r: list(r["options"]))
        flat, index = plan_rows(rqs, isolated)
        rows = [[r] for r in flat] if independent else [flat]
        class _Keep:
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        items = [build(Example(ctx, [Q(r["question"], opts(r), 0) for r in row]), self.m.tok, _Keep(), max_options=MAX_OPTIONS,
                       max_ctx_tokens=max_state_tokens, layout=layout) for row in rows]
        with torch.no_grad():
            if self.eng is not None and len(items) > 1 and layout == "state_first":
                probs = self.eng.score_shared(items, temperature=self.T)
            else:
                probs = []; per = max(1, max_fwd_tokens // max(len(it["ids"]) for it in items))
                for i in range(0, len(items), per):
                    if self.eng is not None:
                        probs += self.eng.score_items(items[i:i + per], temperature=self.T)
                    else:
                        bt = collate(items[i:i + per], self.m.tok.pad_token_id)
                        lg = self.m.slot_logits(*[bt[k].to(self.dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
                        pr = torch.softmax(lg / self.T, -1).cpu(); c = 0
                        for it in items[i:i + per]:
                            probs.append(pr[c:c + len(it["slots"])]); c += len(it["slots"])
        flatp = [p.tolist() for ps in probs for p in ps]
        return {"model": self.name, "answers": assemble(rqs, index, flatp),
                "usage": {"input_tokens": unique_tokens(items), "output_tokens": 0}}

    # ---- typed schema interface: {question: {"type": "bool"} | {"type": "choice", "options": [...]}
    #                                         | {"type": "scale", "legend": {"0": "none", "1": "low", ...}}}
    @staticmethod
    def _schema_to_questions(schema):
        qs = []
        for qtext, spec in schema.items():
            t = spec.get("type", "choice")
            if t == "bool":
                qs.append(dict(question=qtext, options=["no", "yes"]))
            elif t == "choice":
                qs.append(dict(question=qtext, options=list(spec["options"])))
            elif t == "scale":
                leg = spec["legend"]
                keys = sorted(leg, key=lambda k: float(k)) if isinstance(leg, dict) else list(range(len(leg)))
                labels = [f"{k}: {leg[k]}" if isinstance(leg, dict) else f"{i}: {leg[i]}" for i, k in enumerate(keys)]
                qs.append(dict(question=qtext, options=labels, _keys=keys, _legend=leg))
            else:
                raise ValueError(f"unknown field type {t}")
        return qs

    def decide_json_batch(self, requests, **kw):
        """requests: list of (context, schema). Returns one dict per context keyed by question."""
        qss = [self._schema_to_questions(schema) for _, schema in requests]
        raw = self.decide_batch([(ctx, qs) for (ctx, _), qs in zip(requests, qss)], **kw)
        out = []
        for (ctx, schema), qs, res in zip(requests, qss, raw):
            o = {}
            for (qtext, spec), q, r in zip(schema.items(), qs, res):
                t = spec.get("type", "choice")
                if t == "bool":
                    o[qtext] = {"noul": round(r["probs"]["yes"], 4), "type": "noul"}
                elif t == "choice":
                    o[qtext] = {"choice": r["choice"], "confidence": round(r["confidence"], 4), "type": "choice",
                                "probabilities": {k: round(v, 4) for k, v in r["probs"].items()}}
                else:
                    p = [r["probs"][lab] for lab in q["options"]]
                    keys = q["_keys"]; n = len(p)
                    score = sum(float(k) * pi for k, pi in zip(keys, p))          # expected level on the legend scale
                    j = max(range(n), key=p.__getitem__)
                    o[qtext] = {"score": round(score, 2), "confidence": round(p[j], 4), "type": "scale", "legend": q["_legend"],
                                "probabilities": {str(keys[i]): round(pi, 4) for i, pi in enumerate(p)}}
            out.append(o)
        return out

    def decide_json(self, context, schema, **kw):
        return self.decide_json_batch([(context, schema)], **kw)[0]


if __name__ == "__main__":
    import sys, json, time
    d = Decider(sys.argv[1] if len(sys.argv) > 1 else "runs/r1_200k/model")
    demo = [
        ("My card was charged twice for the same purchase and I want the extra charge refunded.",
         [{"question": "Which department should handle this?", "options": ["billing", "technical support", "sales"]},
          {"question": "What is the customer's sentiment?", "options": ["angry", "neutral", "happy"]},
          {"question": "Does this need a refund action?", "options": ["no", "yes"]}]),
        ("hey can u turn the lights off in the kitchen",
         [{"question": "What is the intent?", "options": ["smart home control", "set alarm", "play music", "none of the above"]},
          {"question": "Is this request toxic?", "options": ["no", "yes"]}]),
        ("The quarterly report shows revenue fell 12% while costs rose sharply.",
         [{"question": "What is the financial sentiment?", "options": ["bearish", "neutral", "bullish"]}]),
    ]
    t = time.time(); res = d.decide_batch(demo); dt = time.time() - t
    for (ctx, qs), r in zip(demo, res):
        print("\n>>", ctx)
        for q, a in zip(qs, r):
            print(f"   {q['question']:45s} -> {a['choice']!s:22s} p={a['confidence']:.2f}  " + " ".join(f"{o}:{p:.2f}" for o, p in a['probs'].items()))
    print(f"\n{sum(len(q) for _, q in demo)} decisions in {dt*1000:.0f} ms (one forward pass)")
    schema = {
        "Revenue currently impacted?": {"type": "bool"},
        "What business impact?": {"type": "choice", "options": ["none", "degraded", "outage"]},
        "Integration issue present?": {"type": "bool"},
        "Account health status?": {"type": "choice", "options": ["healthy", "watch", "at risk"]},
        "Which incident scope?": {"type": "choice", "options": ["single_account", "multi_account", "platform_wide"]},
        "Security concern present?": {"type": "bool"},
        "Duplicate charge reported?": {"type": "bool"},
        "Churn likelihood level?": {"type": "scale", "legend": {"0": "none", "1": "low", "2": "medium", "3": "high"}},
        "Human attention needed?": {"type": "bool"},
        "Immediate feature request?": {"type": "bool"},
    }
    ctx = ("Hi, since this morning our Stripe webhook integration stopped firing and our checkout is down for all customers. "
           "We are losing orders every minute and our partner launch is on Thursday. Also I think we got billed twice last week. "
           "If this is not fixed today we will have to look at other providers.")
    t = time.time(); js = d.decide_json(ctx, schema); dt = time.time() - t
    print(f"\n>> {ctx[:80]}...\n" + json.dumps(js, indent=1)[:3000]); print(f"{len(schema)} typed fields in {dt*1000:.0f} ms (one forward pass)")
