"""vision_mix2.pkl = balanced game/Mario frames + DAgger frames + Cauldron + text replay drawn from the v5 delta (has corrective none data)."""
import pickle, random
from collections import Counter
from . import data as D
from . import data2, data3  # noqa
rng = random.Random(0)
ftr, fev = D.load_cache("data/frames.pkl"); dtr, _ = D.load_cache("data/frames_dagger.pkl"); ctr, cev = D.load_cache("data/cauldron.pkl")
v5, v5ev = D.load_cache("data/tasks_v4.pkl"); rng.shuffle(v5); replay = v5[:30000]   # clean replay (no literal-none corrective set)
# balance rare actions in the original frames the same way
bal = []
for task in {e.task for e in ftr}:
    sub = [e for e in ftr if e.task == task]; c = Counter(e.qs[0].options[e.qs[0].gold] for e in sub); n = len(sub)
    for e in sub:
        share = c[e.qs[0].options[e.qs[0].gold]] / n; bal += [e] * (1 if share >= 0.15 else min(6, int(round(0.15 / max(share, 1e-3)))))
train = bal + dtr + ctr + replay; rng.shuffle(train)
evals = {k: v for k, v in fev.items() if len(v) >= 4}; evals.update(cev)
for k in ["clinc_oos", "abstain_probe", "agenttraj", "support_tickets"]: evals[k] = v5ev[k][:300]
print(f"[vision2] train {len(train)} (frames {len(bal)} incl. balancing, dagger {len(dtr)}, cauldron {len(ctr)}, text replay {len(replay)}); eval tasks {len(evals)}")
pickle.dump((train, evals), open("data/vision_mix2.pkl", "wb"))
