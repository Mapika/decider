"""Do the answers depend on which other questions are asked?  Multi-question eval tasks, three ways:
   packed       all questions behind one copy of the context (one row; question k can see question texts 1..k-1)
   reversed     packed, questions in reverse order
   independent  one row per question (context + that question only): independent by construction
Reports accuracy for each, and how far the packed answers move when the company changes.
   python -m decider.independence_probe runs/r10_v5/model [--n 300]"""
import argparse, json, random, numpy as np, torch
from .model import DecisionModel, collate
from .prompt import build
from . import data as D
from . import data2, data3  # noqa


class _Keep:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


@torch.no_grad()
def probs(m, exs, bs=16):
    items = [build(e, m.tok, _Keep()) for e in exs]; order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"])); out = [None] * len(items)
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]; b = collate([items[j] for j in idx], m.tok.pad_token_id)
        p = torch.softmax(m.slot_logits(*[b[k].cuda() for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")]), -1).cpu().numpy()
        c = 0
        for j in idx:
            n = len(items[j]["slots"]); out[j] = np.nan_to_num(p[c:c + n]); c += n
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--data", default="data/tasks_v4.pkl"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    _, evals = D.load_cache(a.data); m = DecisionModel(a.model, grad_ckpt=False).cuda().eval(); res = {}
    for t, exs in evals.items():
        exs = [e for e in exs if len(e.qs) > 1 and all(len(q.options) <= 10 and q.gold >= 0 for q in e.qs)][:a.n]
        if len(exs) < 50: continue
        nq = len(exs[0].qs)
        packed = probs(m, exs)
        rev = probs(m, [D.Example(e.context, e.qs[::-1], e.task) for e in exs]); rev = [r[::-1] for r in rev]
        ind = probs(m, [D.Example(e.context, [q], e.task) for e in exs for q in e.qs]); ind = [np.stack([ind[i * nq + k][0] for k in range(nq)]) for i in range(len(exs))]
        G = np.array([[q.gold for q in e.qs] for e in exs])
        acc = lambda P: float(np.mean([[P[i][k].argmax() == G[i, k] for k in range(nq)] for i in range(len(exs))]))
        d = lambda A, B_: np.array([[np.abs(A[i][k] - B_[i][k]).max() for k in range(nq)] for i in range(len(exs))])
        flip = lambda A, B_: float(np.mean([[A[i][k].argmax() != B_[i][k].argmax() for k in range(nq)] for i in range(len(exs))]))
        res[t] = dict(n=len(exs), nq=nq, acc_packed=acc(packed), acc_reversed=acc(rev), acc_independent=acc(ind),
                      dp_order_mean=float(d(packed, rev).mean()), dp_order_p95=float(np.percentile(d(packed, rev), 95)), flips_order=flip(packed, rev),
                      dp_pack_vs_ind_mean=float(d(packed, ind).mean()), flips_pack_vs_ind=flip(packed, ind))
        r = res[t]; print(f"[indep] {t:20s} nq={nq} acc packed {r['acc_packed']:.3f} reversed {r['acc_reversed']:.3f} independent {r['acc_independent']:.3f} | "
                          f"reorder: mean max|dp| {r['dp_order_mean']:.3f} p95 {r['dp_order_p95']:.3f} flips {r['flips_order']:.3f} | packed vs independent flips {r['flips_pack_vs_ind']:.3f}", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
