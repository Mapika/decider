"""Compare runs: python -m decider.report runs/zs_2b runs/zs_4b runs/r1_200k/final ..."""
import json, pickle, sys, numpy as np
from .metrics import summarize


def load(run):
    r = json.load(open(f"{run}/eval.json"))
    return r["results"]


def fit_temperature(dump, tasks):
    """Fit a single scalar temperature on the given tasks' logits (from probs) by NLL grid search."""
    P, G = [], []
    for t in tasks:
        d = dump[t]; ok = d["golds"] >= 0
        P.append(d["probs"][ok]); G.append(d["golds"][ok])
    P = np.concatenate(P); G = np.concatenate(G)
    L = np.log(np.clip(P, 1e-30, 1)); L[P <= 0] = -np.inf
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(5), 120)):
        z = L / T; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(G)), G], 1e-12, 1)).mean()
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def apply_temperature(probs, T):
    L = np.log(np.clip(probs, 1e-30, 1)); L[probs <= 0] = -np.inf
    z = L / T; z = z - z.max(1, keepdims=True); q = np.exp(z); return q / q.sum(1, keepdims=True)


def main(runs):
    cols = ["acc", "nll", "brier", "ece", "aurc", "acc_at_80"]
    allres = {r: load(r) for r in runs}
    tasks = sorted(set().union(*[set(v) for v in allres.values()]))
    held = {t: allres[runs[0]][t]["heldout"] if t in allres[runs[0]] else False for t in tasks}
    print(f"{'task':22s} " + " ".join(f"{r.split('/')[-2] if r.endswith('final') else r.split('/')[-1]:>28s}" for r in runs))
    print(f"{'':22s} " + " ".join(f"{'acc':>6s}{'nll':>6s}{'ece':>6s}{'a@80':>6s}    " for _ in runs))
    for t in tasks:
        line = f"{t:20s}{'H ' if held[t] else '  '}"
        for r in runs:
            v = allres[r].get(t)
            line += f" {v['acc']:6.3f}{v['nll']:6.3f}{v['ece']:6.3f}{v['acc_at_80']:6.3f}    " if v else " " * 28
        print(line)
    print()
    for r in runs:
        agg = json.load(open(f"{r}/eval.json"))["agg"]
        print(f"{r:30s} IN  " + " ".join(f"{k}={agg['in_task'][k]:.3f}" for k in cols))
        print(f"{'':30s} OUT " + " ".join(f"{k}={agg['heldout'][k]:.3f}" for k in cols))
        # temperature scaling: fit on in-task eval sets, report held-out
        try:
            dump = pickle.load(open(f"{r}/preds.pkl", "rb"))
            res = allres[r]
            T = fit_temperature(dump, [t for t in res if not res[t]["heldout"]])
            outs = []
            for t in res:
                if res[t]["heldout"]:
                    d = dump[t]; ok = d["golds"] >= 0
                    outs.append(summarize(apply_temperature(d["probs"][ok], T), d["golds"][ok], d["nopts"][ok]))
            m = {k: float(np.mean([o[k] for o in outs])) for k in cols}
            print(f"{'':30s} OUT(T={T:.2f}) " + " ".join(f"{k}={m[k]:.3f}" for k in cols))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
