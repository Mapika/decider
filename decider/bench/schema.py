"""Latency / throughput: full forward (state-first rows through Engine) vs schema cache (SchemaEngine).
   python -m decider.bench.schema runs/r11_v6/model [compile]"""
import sys, time, numpy as np, torch
from decider.engine import Engine
from decider.schema_engine import SchemaEngine
from decider.prompt import build
from decider import data as D
from decider import systemone as S1


class K:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def timeit(fn, n=30, warm=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time() - t) * 1000)
    return float(np.median(ts))


def main():
    comp = "compile" in sys.argv[2:]
    eng = Engine(sys.argv[1], compile=comp, conv_patch=comp); se = SchemaEngine(eng)
    _, evals = D.load_cache("data/tasks_v4.pkl")
    tickets = [e.context for e in evals["support_tickets"][:64]]; chats = [e.context for e in evals["clinc_oos"][:64]]
    jev = {"impact": {"type": "choice", "instructions": "What business impact?", "criteria": {"none": "no effect on the business", "degraded": "slower or partially failing", "outage": "a core flow is down"}},
           "team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts", "other": None}},
           "revenue": {"type": "noul", "instructions": "Is revenue currently impacted?"}, "security": {"type": "noul", "instructions": "Is there a security concern?"},
           "dup": {"type": "noul", "instructions": "Is a duplicate charge reported?"}, "human": {"type": "noul", "instructions": "Does this need a human?"},
           "churn": {"type": "score", "instructions": "How likely is the customer to churn?", "criteria": ["no sign of leaving", "mild dissatisfaction", "explicitly considering alternatives", "has decided to leave"]},
           "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["can wait", "this week", "today"]},
           "sentiment": {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {"calm": None, "frustrated": None, "angry": None}},
           "lang": {"type": "choice", "instructions": "Which language is the message written in?", "criteria": ["English", "German", "Spanish", "French"]}}
    rq = [S1.render_question(v) for v in jev.values()]; jq = [D.Q(r["question"], r["options"], 0) for r in rq]
    big = [D.Q("What is the intent of this user message?", evals["clinc_oos"][0].qs[0].options, 0)]
    for name, qs, ctxs in [("3 questions, tickets (~230 tok)", evals["support_tickets"][0].qs, tickets), ("10 Jev-style questions, tickets", jq, tickets),
                           ("10 Jev-style questions, short chat (~12 tok)", jq, chats), ("1 question x 151 options, short chat", big, chats)]:
        h = se.prepare([dict(question=q.text, options=q.options) for q in qs])
        packed = lambda c: build(D.Example(c, qs, "x"), eng.tok, K(), max_options=255)
        rows = lambda c: [build(D.Example(c, [q], "x"), eng.tok, K(), max_options=255) for q in qs]
        n_full = len(packed(ctxs[0])["ids"]); print(f"\n== {name}: schema prefix {h.tpmax} tok, full prompt {n_full} tok")
        for B in (1, 32):
            cs = ctxs[:B]
            t_packed = timeit(lambda: eng.score_items([packed(c) for c in cs]))
            t_cache = timeit(lambda: se.score(h, cs))
            line = f"   batch {B:2d}: packed full forward {t_packed:7.1f} ms | schema cache {t_cache:7.1f} ms ({t_packed / t_cache:4.1f}x) -> {B * len(qs) / t_cache * 1000:7.0f} decisions/s"
            if B == 1 and len(qs) > 1:
                t_rows = timeit(lambda: eng.score_items(rows(cs[0]))); line += f" | independent rows {t_rows:6.1f} ms"
            print(line, flush=True)
    print(se.stats)


if __name__ == "__main__":
    main()
