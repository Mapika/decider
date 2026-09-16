"""tasks_v4.pkl = v2 mixture + situation->action tasks (agenttraj, mind2web, games, synth) + Mario teacher states."""
import pickle, sys
from . import data as D
from . import data2, data3  # noqa
train, evals = D.load_cache("data/tasks_v2.pkl")
n0 = len(train)
for n in data3.NEW_TASKS:
    try:
        tr, ev = D.load_task(n)
    except Exception as e:
        print(f"[v4] FAILED {n}: {e}", flush=True); continue
    if not (tr or ev):
        print(f"[v4] EMPTY {n}", flush=True); continue
    train.extend(tr); evals[n] = ev
    print(f"[v4] {n:12s} train={len(tr):6d} eval={len(ev):5d}", flush=True)
try:
    mtr, mev = D.load_cache("data/mario.pkl")
    mtr = [e for e in mtr if e.task == "mario"]; train.extend(mtr); evals["mario"] = mev["mario"]
    print(f"[v4] mario        train={len(mtr):6d} eval={len(mev['mario']):5d}")
except Exception as e:
    print("[v4] mario skipped:", e)
print(f"[v4] total train {len(train)} (+{len(train)-n0})  eval tasks {len(evals)}")
pickle.dump((train, evals), open("data/tasks_v4.pkl", "wb"))
# delta set for continued training from r3: the new situation->action data + a 2x replay of general data
import random
new = train[n0:]; rng = random.Random(0); gen = train[:n0]; rng.shuffle(gen); replay = gen[:2 * len(new)]
delta_evals = {k: v for k, v in evals.items() if k in data3.NEW_TASKS + ["mario", "clinc_oos", "support_tickets", "abstain_probe", "trec", "sciq", "reward_bench", "hermes_tools", "helpsteer2", "paws"]}
pickle.dump((new + replay, delta_evals), open("data/tasks_v4_delta.pkl", "wb"))
print(f"[v4] delta set: {len(new)} new + {len(replay)} replay = {len(new)+len(replay)}; eval tasks {len(delta_evals)}")
