"""vision_mix.pkl = frames (games + Mario, pixels only) + Cauldron image choices + a text-only replay of the general mixture."""
import pickle, random
from . import data as D
from . import data2, data3  # noqa
rng = random.Random(0)
ftr, fev = D.load_cache("data/frames.pkl"); ctr, cev = D.load_cache("data/cauldron.pkl")
gtr, gev = D.load_cache("data/tasks_v4.pkl"); rng.shuffle(gtr); replay = gtr[:25000]
train = ftr + ctr + replay; rng.shuffle(train)
evals = {k: v for k, v in fev.items() if len(v) >= 4}; evals.update(cev)
for k in ["clinc_oos", "abstain_probe", "agenttraj", "support_tickets"]: evals[k] = gev[k][:300]
n_img = sum(getattr(e, "image", None) is not None for e in train)
print(f"[vision] train {len(train)} ({n_img} with image: frames {len(ftr)}, cauldron {len(ctr)}; text replay {len(replay)}); eval tasks {len(evals)}")
pickle.dump((train, evals), open("data/vision_mix.pkl", "wb"))
