"""The vision mixture: balanced game/Mario frames + DAgger frames + Cauldron + a text replay drawn from the current text mixture.
   python -m decider.vision.mixture [--replay data/mixture_full.pkl] [--out data/vision_mix.pkl]"""
import argparse, pickle, random
from collections import Counter
from decider import data as D


def _cli():
    ap = argparse.ArgumentParser(); ap.add_argument("--replay", default="data/mixture_full.pkl"); ap.add_argument("--n_replay", type=int, default=30000); ap.add_argument("--out", default="data/vision_mix.pkl")
    a = ap.parse_args(); rng = random.Random(0)
    ftr, fev = D.load_cache("data/frames.pkl"); dtr, _ = D.load_cache("data/frames_dagger.pkl"); ctr, cev = D.load_cache("data/cauldron.pkl")
    tx, txev = D.load_cache(a.replay); rng.shuffle(tx)
    replay = [e for e in tx[:a.n_replay * 2] if len(e.context) <= 6000][:a.n_replay]        # the vision trainer packs by example, not by token
    # balance rare actions in the original frames
    bal = []
    for task in {e.task for e in ftr}:
        sub = [e for e in ftr if e.task == task]; c = Counter(e.qs[0].options[e.qs[0].gold] for e in sub); n = len(sub)
        for e in sub:
            share = c[e.qs[0].options[e.qs[0].gold]] / n; bal += [e] * (1 if share >= 0.15 else min(6, int(round(0.15 / max(share, 1e-3)))))
    train = bal + dtr + ctr + replay; rng.shuffle(train)
    evals = {k: v for k, v in fev.items() if len(v) >= 4}; evals.update(cev)
    for k in ["clinc_oos", "abstain_probe", "agenttraj", "support_tickets", "banking77", "custom_choice" if "custom_choice" in txev else "boolq"]:
        if k in txev: evals[k] = txev[k][:300]
    print(f"[vision] train {len(train)} (frames {len(bal)} incl. balancing, dagger {len(dtr)}, cauldron {len(ctr)}, text replay {len(replay)}); eval tasks {len(evals)}")
    pickle.dump((train, evals), open(a.out, "wb"))


if __name__ == "__main__":
    _cli()
