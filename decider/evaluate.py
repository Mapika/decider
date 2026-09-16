"""Evaluate a DecisionModel (fine-tuned or raw base) on all eval sets. Saves per-question probs."""
import argparse, json, os, pickle, random, time
import numpy as np, torch
from .model import DecisionModel, collate
from .prompt import build
from .metrics import summarize
from . import data as D
from . import data2  # noqa: F401  (registers v2 tasks)


@torch.no_grad()
def run_eval(model, evals, bs=32, max_ctx=1536, temperature=1.0, log=print):
    model.eval()
    dev = next(model.parameters()).device
    results, dump = {}, {}
    for tname, exs in evals.items():
        if not exs:
            continue
        rng = random.Random(1234)
        items = [dict(build(e, model.tok, rng, max_ctx_tokens=max_ctx), task=tname, ex_id=i) for i, e in enumerate(exs)]
        items.sort(key=lambda it: len(it["ids"]))
        P, G, NO, QI = [], [], [], []
        t0 = time.time()
        for i in range(0, len(items), bs):
            b = collate(items[i:i + bs], model.tok.pad_token_id)
            logits = model.slot_logits(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["slot_idx"].to(dev), b["slot_batch"].to(dev), b["nopts"].to(dev))
            p = torch.softmax(logits / temperature, -1).cpu().numpy()
            P.append(np.nan_to_num(p)); G.append(b["golds"].numpy()); NO.append(b["nopts"].numpy()); QI.extend(b["qidx"])
        P = np.concatenate(P); G = np.concatenate(G); NO = np.concatenate(NO); QI = np.asarray(QI)
        ok = G >= 0
        s = summarize(P[ok], G[ok], NO[ok]); s["sec"] = round(time.time() - t0, 1); s["heldout"] = D.TASKS[tname]["heldout"]
        # per-question breakdown for multi-question tasks
        if QI.max() > 0:
            s["per_q"] = [summarize(P[ok & (QI == k)], G[ok & (QI == k)], NO[ok & (QI == k)])["acc"] for k in range(QI.max() + 1)]
        results[tname] = s
        dump[tname] = dict(probs=P, golds=G, nopts=NO, qidx=QI)
        log(f"[eval] {tname:20s} n={s['n']:5d} acc={s['acc']:.3f} (chance {s['chance']:.2f}) nll={s['nll']:.3f} brier={s['brier']:.3f} ece={s['ece']:.3f} aurc={s['aurc']:.3f} acc@80={s['acc_at_80']:.3f} {'HELDOUT' if s['heldout'] else ''}")
    return results, dump


def aggregate(results):
    def agg(keys):
        rs = [results[k] for k in keys if k in results]
        if not rs:
            return {}
        return {m: float(np.mean([r[m] for r in rs])) for m in ["acc", "nll", "brier", "ece", "aurc", "acc_at_80", "chance"]}
    tr = [k for k, v in results.items() if not v["heldout"]]; ho = [k for k, v in results.items() if v["heldout"]]
    return dict(in_task=agg(tr), heldout=agg(ho), n_in=len(tr), n_heldout=len(ho))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", default="data/tasks.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    _, evals = pickle.load(open(a.data, "rb"))
    if a.tasks:
        evals = {k: v for k, v in evals.items() if k in a.tasks.split(",")}
    if a.limit:
        evals = {k: v[:a.limit] for k, v in evals.items()}
    m = DecisionModel(a.model, grad_ckpt=False).cuda()
    os.makedirs(a.out, exist_ok=True)
    res, dump = run_eval(m, evals, bs=a.bs, temperature=a.temperature)
    agg = aggregate(res)
    print("[agg]", json.dumps(agg, indent=1))
    json.dump(dict(results=res, agg=agg, model=a.model), open(f"{a.out}/eval.json", "w"), indent=1)
    pickle.dump(dump, open(f"{a.out}/preds.pkl", "wb"))
