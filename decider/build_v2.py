"""Build data/tasks_v2.pkl = v1 cache + data2 tasks + abstention probe."""
import pickle, random, sys
from . import data as D
from . import data2 as D2
train, evals = D.load_cache("data/tasks.pkl")
names = sys.argv[1:] or D2.NEW_TASKS
for n in names:
    if n == "abstain_probe":
        continue
    try:
        tr, ev = D.load_task(n)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[v2] FAILED {n}: {e}", flush=True); continue
    if not (tr or ev):
        print(f"[v2] EMPTY {n}", flush=True); continue
    if not D.TASKS[n]["heldout"]:
        train.extend(tr)
    evals[n] = ev
    ex = (tr or ev)[0]
    print(f"[v2] {n:20s} train={len(tr):6d} eval={len(ev):5d} heldout={D.TASKS[n]['heldout']} nq={len(ex.qs)} nopt={len(ex.qs[0].options)} gold0={ex.qs[0].options[ex.qs[0].gold]!r}", flush=True)
# abstention probe: held-out classification tasks, half of the examples with the gold option removed
rng = random.Random(7); probe = []
for t in ["trec", "bbc_news", "massive_scenario", "student_questions", "dolly_category", "fin_sentiment"]:
    for e in evals[t][:250]:
        q = e.qs[0]
        if len(q.options) < 3:
            continue
        if rng.random() < 0.5:
            opts = [o for i, o in enumerate(q.options) if i != q.gold] + ["none of the above"]
            probe.append(D.Example(e.context, [D.Q(q.text, opts, len(opts) - 1)], "abstain_probe"))
        else:
            probe.append(D.Example(e.context, [D.Q(q.text, list(q.options) + ["none of the above"], q.gold)], "abstain_probe"))
evals["abstain_probe"] = probe
print(f"[v2] abstain_probe eval={len(probe)}")
print(f"[v2] total train {len(train)}  eval tasks {len(evals)}  heldout {sum(D.TASKS[k]['heldout'] for k in evals)}")
pickle.dump((train, evals), open("data/tasks_v2.pkl", "wb"))
