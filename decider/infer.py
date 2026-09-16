"""Usable inference API: typed decisions with probabilities, all from one forward pass.

    from decider.infer import Decider
    d = Decider("runs/r2_full/model")
    out = d.decide("My card was charged twice for the same purchase.",
                   [{"question": "Which department should handle this?", "options": ["billing", "technical", "sales"]},
                    {"question": "How urgent is this?", "options": ["low", "medium", "high"]}])
    # -> [{'choice': 'billing', 'confidence': 0.97, 'probs': {...}}, {...}]
"""
import torch
from .model import DecisionModel, collate
from .prompt import build, MAX_OPTIONS
from dataclasses import dataclass


@dataclass
class Q:
    text: str; options: list; gold: int = 0


@dataclass
class Example:
    context: str; qs: list; task: str = "infer"


class Decider:
    """use_graphs=True (default on CUDA) routes scoring through decider.engine.Engine: shape-bucketed
    CUDA graphs, ~7x lower single-request latency than eager. Set False for CPU or debugging."""
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, temperature=1.0, abstain_below=0.0, use_graphs=None):
        if use_graphs is None:
            use_graphs = str(device).startswith("cuda")
        if use_graphs:
            from .engine import Engine
            self.eng = Engine(path, device=device, dtype=dtype); self.m = self.eng.m
        else:
            self.eng = None; self.m = DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        self.dev = device; self.T = temperature; self.abstain_below = abstain_below

    @torch.no_grad()
    def decide_batch(self, requests, max_ctx_tokens=1536):
        """requests: list of (context:str, questions:list[dict(question, options)]). One forward pass for everything."""
        exs, meta = [], []
        for context, qs in requests:
            for q in qs:
                assert 2 <= len(q["options"]) <= MAX_OPTIONS, f"2..{MAX_OPTIONS} options required"
            exs.append(Example(context, [Q(q["question"], list(q["options"]), 0) for q in qs], "infer"))
        class _NoShuffle:                      # keep option order as given
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        items = [build(e, self.m.tok, _NoShuffle(), max_ctx_tokens=max_ctx_tokens) for e in exs]
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
                j = max(range(len(p)), key=p.__getitem__)
                res.append(dict(choice=q["options"][j] if p[j] >= self.abstain_below else None, confidence=p[j],
                                probs={o: pi for o, pi in zip(q["options"], p)}))
            out.append(res)
        return out

    def decide(self, context, questions, **kw):
        return self.decide_batch([(context, questions)], **kw)[0]

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
