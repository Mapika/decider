"""decider on Bespoke's public benchmark suite (github.com/bespokelabsai/nimble, docs/PUBLIC_BENCHMARKS.md): 13 human-labelled
subsets, 3,880 records, in Jev's wire format, on which Bespoke-Nimble-9B and Jev 1.13.0 were both measured.  The records are rebuilt
byte-for-byte from the manifests committed in that repository (their converters + the upstream files); each is one state and one
question (choice with described options, noul with true/false criteria, or a 5-level score), scored here with `Decider.system_one`
exactly as a user would call it.

    python -m decider.bench.public_suite runs/r15_v9b/model /path/to/nimble/data/public --out suite.json [--isolated 0|1]"""
import argparse, json, math, os, time, collections

SUBSETS = ["vitaminc-dev", "massive-en-US", "massive-de-DE", "boolq", "squad2", "paws", "multinli", "civil_comments", "aegis2", "helpsteer2",
           "summeval-relevance", "summeval-consistency", "pubmedqa"]
TRAINED = {"boolq", "paws", "civil_comments", "helpsteer2", "pubmedqa", "multinli", "massive-en-US"}   # decider's mixture holds a train split of these


def ece(rows, bins=10):
    b = [[0, 0.0, 0.0] for _ in range(bins)]
    for conf, ok in rows:
        i = min(bins - 1, int(conf * bins)); b[i][0] += 1; b[i][1] += conf; b[i][2] += ok
    n = sum(x[0] for x in b)
    return sum(x[0] / n * abs(x[1] / x[0] - x[2] / x[0]) for x in b if x[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("root"); ap.add_argument("--out", default="public_suite.json"); ap.add_argument("--isolated", type=int, default=None)
    ap.add_argument("--subsets", default=None)
    a = ap.parse_args()
    from decider.infer import Decider
    d = Decider(a.model); results = {}
    for name in (a.subsets.split(",") if a.subsets else SUBSETS):
        recs = [json.loads(l) for l in open(os.path.join(a.root, name, "all.jsonl"), encoding="utf-8")]
        t = time.time(); rows = []; brier = 0.0; mae = 0.0; conf_ok = []; kinds = collections.Counter()
        for r in recs:
            q = r["input"]["questions"]["decision"]; tgt = r["reference"]["target"]
            ans = d.system_one(r["input"]["state"], {"decision": q}, isolated=None if a.isolated is None else bool(a.isolated))["answers"]["decision"]
            if q["type"] == "choice":
                probs = ans["probabilities"]; pred = ans["choice"]; ok = pred == tgt; ptrue = probs.get(tgt, 0.0)
                brier += sum((p - (k == tgt)) ** 2 for k, p in probs.items()); conf_ok.append((ans["confidence"], ok))
            elif q["type"] == "noul":
                p1 = ans["noul"]; pred = p1 >= 0.5; ok = pred == tgt; ptrue = p1 if tgt else 1 - p1
                brier += 2 * (p1 - float(tgt)) ** 2; conf_ok.append((max(p1, 1 - p1), ok))
            else:
                probs = ans["probabilities"]; pred = int(max(probs, key=probs.get)); ok = pred == tgt; ptrue = probs[str(tgt)]
                brier += sum((p - (int(k) == tgt)) ** 2 for k, p in probs.items()); conf_ok.append((ans["confidence"], ok))
                mae += abs(ans["score"] - tgt)
            kinds[q["type"]] += 1; rows.append((r["id"], ok, ptrue))
        n = len(rows); acc = sum(ok for _, ok, _ in rows) / n
        res = dict(n=n, type=next(iter(kinds)), acc=round(acc, 4), ece=round(ece(conf_ok), 4), brier=round(brier / n, 4),
                   nll=round(-sum(math.log(max(p, 1e-15)) for _, _, p in rows) / n, 4), trained=name in TRAINED, sec=round(time.time() - t, 1))
        if kinds.get("score"): res["score_mae"] = round(mae / n, 4)
        results[name] = res; print(f"[suite] {name:22s} n={n:4d} acc={acc:.3f} ece={res['ece']:.3f} brier={res['brier']:.3f}" + (f" mae={res['score_mae']:.3f}" if "score_mae" in res else "") + (" (trained task)" if name in TRAINED else "") + f" {res['sec']}s", flush=True)
    accs = [r["acc"] for r in results.values()]; ns = [r["n"] for r in results.values()]
    results["_macro"] = round(sum(accs) / len(accs), 4); results["_micro"] = round(sum(a_ * n_ for a_, n_ in zip(accs, ns)) / sum(ns), 4)
    held = [r["acc"] for k, r in results.items() if not k.startswith("_") and not r["trained"]]
    results["_macro_untrained"] = round(sum(held) / len(held), 4) if held else None
    print("[suite] macro", results["_macro"], "micro", results["_micro"], "macro over subsets decider never trained on", results["_macro_untrained"])
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
