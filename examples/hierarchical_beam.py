"""Hierarchical classification with a beam over Choice probabilities (DBpedia: 9 -> 70 -> 219 classes), against asking all
219 classes at once.  At each level the options are the children of a kept node and every option shows its own children,
so the model sees what lives under a branch before committing to it.
   python examples/hierarchical_beam.py [model] [--n 300] [--beam 3]"""
import argparse, collections, json, random, re, sys
sys.path.insert(0, ".")
from datasets import load_dataset
from decider.infer import Decider

ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?", default="Mapika/decider-2b"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--beam", type=int, default=3)
a = ap.parse_args()
nice = lambda s: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()
tr = load_dataset("DeveloperOats/DBPedia_Classes", split="train"); te = load_dataset("DeveloperOats/DBPedia_Classes", split="test").shuffle(seed=0).select(range(a.n))
tree = collections.defaultdict(lambda: collections.defaultdict(set))
for l1, l2, l3 in zip(tr["l1"], tr["l2"], tr["l3"]): tree[nice(l1)][nice(l2)].add(nice(l3))
leaves = sorted({l3 for l1 in tree for l2 in tree[l1] for l3 in tree[l1][l2]})
d = Decider(a.model); Q = "Which class of entity does this encyclopedia text describe?"


def ask(text, options):
    """options: {name: children or None} -> {name: probability}"""
    opts = [n if not c else f"{n}: {json.dumps(sorted(c)[:12])}" for n, c in options.items()]
    if len(opts) == 1: return {next(iter(options)): 1.0}
    r = d.decide(text, [{"question": Q, "options": opts}], max_ctx_tokens=2048)[0]
    return dict(zip(options, r["probs"].values()))


flat = beam = 0
for i, r in enumerate(te):
    text, gold = r["text"][:1500], nice(r["l3"])
    p = ask(text, {l: None for l in leaves}); flat += max(p, key=p.get) == gold
    paths = [((l1,), pr) for l1, pr in ask(text, {l1: [l2 for l2 in tree[l1]] for l1 in tree}).items()]
    for depth in (1, 2):
        paths = sorted(paths, key=lambda x: -x[1])[:a.beam]; nxt = []
        for path, pr in paths:
            node = tree[path[0]] if depth == 1 else tree[path[0]][path[1]]
            kids = {k: (sorted(node[k]) if depth == 1 else None) for k in node}
            nxt += [(path + (k,), pr * pk) for k, pk in ask(text, kids).items()]
        paths = nxt
    beam += max(paths, key=lambda x: x[1])[0][2] == gold
    if (i + 1) % 50 == 0: print(f"[{i+1}] flat {len(leaves)}-way {flat/(i+1):.3f} | beam-{a.beam} walk {beam/(i+1):.3f}", flush=True)
print(f"flat {len(leaves)}-way accuracy {flat/a.n:.3f}; hierarchical beam-{a.beam} accuracy {beam/a.n:.3f} (n={a.n})")
