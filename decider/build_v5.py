"""v5 delta = v4 delta + corrective none-of-the-above replay (80% none offered but gold present, 20% gold removed)."""
import pickle, random
from . import data as D
from . import data2, data3  # noqa
rng = random.Random(1)
delta, evals = D.load_cache("data/tasks_v4_delta.pkl")
gen, gev = D.load_cache("data/tasks_v4.pkl")
pool = [e for e in gen if any(len(q.options) >= 3 and not any("none of the above" in o for o in q.options) for q in e.qs)]
rng.shuffle(pool); extra = []
for e in pool[:25000]:
    qs = []
    for q in e.qs:
        if len(q.options) >= 3 and not any("none of the above" in o for o in q.options):
            if rng.random() < 0.8: qs.append(D.Q(q.text, list(q.options) + ["none of the above"], q.gold))
            else:
                opts = [o for i, o in enumerate(q.options) if i != q.gold] + ["none of the above"]; qs.append(D.Q(q.text, opts, len(opts) - 1))
        else: qs.append(q)
    extra.append(D.Example(e.context, qs, e.task))
train = delta + extra; rng.shuffle(train)
print(f"[v5] delta {len(delta)} + corrective {len(extra)} = {len(train)}")
pickle.dump((train, evals), open("data/tasks_v5_delta.pkl", "wb"))
