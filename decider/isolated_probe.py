"""Isolated level scoring vs listwise, on every ordinal-scale question of the eval sets (and the teacher-written score questions).
   listwise : the usual prompt, all levels listed with their numbers, one softmax
   isolated : one row per level (state + question + that level only, number stripped) -> P(fits); normalised over levels
   python -m decider.isolated_probe runs/r12_v7/model [--n 300] [--layout schema_first]"""
import argparse, json, re, random, numpy as np, torch
from .model import DecisionModel, collate
from .prompt import build
from .metrics import summarize
from .systemone import isolated_rows, combine_isolated
from . import data as D
from . import data2, data3  # noqa


class K:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


@torch.no_grad()
def score(m, exs, layout, bs=48):
    items = [build(e, m.tok, K(), max_options=255, max_ctx_tokens=4096, layout=layout) for e in exs]; order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"])); out = [None] * len(items)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; b = collate([items[j] for j in idx], m.tok.pad_token_id)
        p = torch.softmax(m.slot_logits(*[b[k].cuda() for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")]), -1).cpu().numpy(); c = 0
        for j in idx:
            n = len(items[j]["slots"]); out[j] = np.nan_to_num(p[c:c + n]); c += n
    return out


def is_scale(q): return len(q.options) >= 3 and all(re.match(r"^-?\d+:", o) for o in q.options)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--layout", default="state_first"); ap.add_argument("--out", default="")
    a = ap.parse_args(); m = DecisionModel(a.model, grad_ckpt=False).cuda().eval()
    _, evals = D.load_cache("data/tasks_v4.pkl"); sets = {}
    for t, exs in evals.items():
        qs = [(e.context, q) for e in exs[:a.n] for q in e.qs if is_scale(q) and q.gold >= 0]
        if len(qs) >= 100: sets[t] = qs[:a.n * 2]
    try:
        _, pr = D.load_cache("data/probes_v7.pkl"); sets["custom_score (teacher, held-out domains)"] = [(e.context, D.Q(e.qs[0].text, e.qs[0].options, e.qs[0].gold)) for e in pr["custom_score"][:a.n * 2]]
    except Exception: pass
    res = {}
    for t, qs in sets.items():
        lw = score(m, [D.Example(c, [q], t) for c, q in qs], a.layout)
        rows = [D.Example(c, [D.Q(text, opts, 0)], t) for c, q in qs for text, opts in isolated_rows(q.text, q.options)]
        py = score(m, rows, a.layout); k = 0; P_iso, mass = [], []
        for c, q in qs:
            n = len(q.options); p, tot = combine_isolated([float(py[k + j][0][1]) for j in range(n)]); k += n; P_iso.append(p); mass.append(tot)
        G = np.array([q.gold for _, q in qs]); NO = np.array([len(q.options) for _, q in qs]); K_ = NO.max()
        pad = lambda P: np.array([list(p[:n]) + [0.0] * (K_ - n) for p, n in zip(P, NO)])
        A, B_ = summarize(pad([x[0] for x in lw]), G, NO), summarize(pad(P_iso), G, NO)
        mae = lambda P: float(np.mean([abs(sum(j * pj for j, pj in enumerate(p[:n])) - g) for p, n, g in zip(P, NO, G)]))
        res[t] = dict(n=len(qs), listwise=dict(acc=A["acc"], nll=A["nll"], ece=A["ece"], mae=mae([x[0] for x in lw])), isolated=dict(acc=B_["acc"], nll=B_["nll"], ece=B_["ece"], mae=mae(P_iso), mean_mass=float(np.mean(mass))))
        r = res[t]; print(f"[iso] {t[:40]:40s} n={len(qs):4d} | listwise acc {r['listwise']['acc']:.3f} nll {r['listwise']['nll']:.3f} ece {r['listwise']['ece']:.3f} mae {r['listwise']['mae']:.3f} | "
                          f"isolated acc {r['isolated']['acc']:.3f} nll {r['isolated']['nll']:.3f} ece {r['isolated']['ece']:.3f} mae {r['isolated']['mae']:.3f} mass {r['isolated']['mean_mass']:.2f}", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
