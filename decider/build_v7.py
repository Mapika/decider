"""v7 delta set: fixes for the v6 weaknesses, on top of v6.
  custom     teacher-written states with free-form noul / choice / score questions in the API's rendering (decider.synth_custom),
             including option lists where a GENERIC option is right although a catch-all is offered; 6 domains held out as a probe
  indexed    JSON states whose long arrays carry "_index" (what systemone.render_state sends), questions by array position
  situation  the v4 situation-to-action data (games, Mario, AgentTraj, Mind2Web, synthetic situations): held-out Freeway had slipped
  replay     v6 formats + the general mixture
Train with --max_options 255 --schema_first_prob 0.5: every example is rendered state-first or schema-first at random.
   python -m decider.build_v7  -> data/tasks_v7_delta.pkl, data/probes_v7.pkl"""
import collections, json, pickle, random
from . import data as D
from . import data2, data3, data4  # noqa
from . import systemone as S1
from .build_v6 import Builder
from .synth_custom import to_example, DOMAINS

HELD_DOMAINS = set(DOMAINS[-6:])


def main():
    rng = random.Random(7)
    recs = [json.loads(l) for l in open("data/synth_custom.jsonl")]
    tr_recs = [r for r in recs if r["domain"] not in HELD_DOMAINS]; ho_recs = [r for r in recs if r["domain"] in HELD_DOMAINS]
    new = []; n_generic = 0
    for r in tr_recs:
        ex = to_example(r, D, S1, "custom+v7"); qs = list(zip(ex.qs, r["questions"]))
        sub = rng.sample(qs, rng.randint(min(3, len(qs)), len(qs)))
        new.append(D.Example(ex.context, [q for q, _ in sub], "custom+v7"))                       # packed subset, random order
        special = [q for q, m in qs if m["type"] == "choice" and any(k for k in m["criteria"] if k == m["answer"]) and r["recipe"] in ("generic", "catchall")
                   and (("general" in m["answer"].lower() or "misc" in m["answer"].lower() or "service" in m["answer"].lower() or "support" in m["answer"].lower()) or m["answer"] == r["catchall"])]
        n_generic += len(special)
        singles = special * 3 + rng.sample([q for q, _ in qs], min(4, len(qs)))                  # generic / catch-all cases are the point: weight them
        for q in singles:
            new.append(D.Example(ex.context, [q], "custom+v7"))                                   # asked alone
        sub2 = rng.sample(qs, rng.randint(min(2, len(qs)), len(qs)))
        new.append(D.Example(ex.context, [q for q, _ in sub2], "custom+v7"))                      # a second packed subset (options are reshuffled at render time)
    # targeted routing set: short messages over terse option lists with a generic bucket and a catch-all
    routes = [json.loads(l) for l in open("data/synth_routing.jsonl")]
    for r in routes:
        c = r["questions"][0]["criteria"]; r["questions"][0]["criteria"] = {k: (None if v in ("null", "", None) else v) for k, v in c.items()}
    def trusted(r):
        """The zero-shot checker shares the bias being fixed (it sends generic cases to the catch-all), so a generic label written at
        generation time is kept unless the checker chose a SPECIFIC option (real ambiguity). Everything else needs agreement."""
        q = r["questions"][0]
        return q["teacher_ok"] or (r["group"] == "generic" and q["teacher_pred"] == r["catchall"])
    routes = [r for r in routes if trusted(r)]
    for r in routes:
        if r["domain"] not in HELD_DOMAINS:
            new.append(to_example(r, D, S1, "routing+v7"))
    n_custom = len(new)
    train, evals = D.load_cache("data/tasks_v4.pkl"); desc = json.load(open("data/label_desc.json"))
    B = Builder(train, evals, desc, seed=7)
    single = [e for e in train if len(e.qs) == 1 and e.qs[0].gold >= 0 and e.task not in ("games", "mario")]
    short = [e for e in single if len(e.context) <= 1200]; longer = [e for e in single if 300 <= len(e.context) <= 6000]
    def indexed(e):
        return D.Example(json.dumps(S1.annotate_indices(json.loads(e.context)), ensure_ascii=False), e.qs, e.task)
    for i in range(20000):
        r = rng.random()
        if r < 0.55: e = B.json_state(short, rng.randint(8, 40), rng.randint(1, 4), 12000, hetero=rng.random() < 0.4)
        elif r < 0.8: e = B.json_state(longer, rng.randint(8, 60), rng.randint(1, 4), 40000, hetero=rng.random() < 0.4)
        else: e = B.json_state(short, rng.randint(2, 7), rng.randint(1, 3), 6000, hetero=rng.random() < 0.4)
        new.append(indexed(e))
    n_idx = len(new) - n_custom
    sit = [e for e in train if e.task in ("games", "mario", "synth", "agenttraj", "mind2web")]; new += rng.sample(sit, min(30000, len(sit))); n_sit = len(new) - n_custom - n_idx
    v6, _ = D.load_cache("data/tasks_v6_delta.pkl")
    v6new = [e for e in v6 if e.task.endswith("+v6")]; v6gen = [e for e in v6 if not e.task.endswith("+v6")]
    rep = rng.sample(v6new, 80000) + rng.sample(v6gen, 120000)
    out = new + rep; rng.shuffle(out)
    print(f"[v7] custom+routing {n_custom} ({len(tr_recs)} states, {n_generic} generic/catch-all singles, {len(routes)} routing messages) + indexed json {n_idx} + situation {n_sit} + replay {len(rep)} = {len(out)}")
    keep = ["clinc_oos", "support_tickets", "abstain_probe", "offtopic_probe", "trec", "sciq", "hermes_tools", "helpsteer2", "agenttraj", "games", "massive_scenario"]
    pickle.dump((out, {k: v for k, v in evals.items() if k in keep}), open("data/tasks_v7_delta.pkl", "wb"))
    # probe: held-out domains, one question per row, grouped by type / recipe
    probes = collections.defaultdict(list)
    for r in ho_recs:
        ex = to_example(r, D, S1, "custom")
        for q, m in zip(ex.qs, r["questions"]):
            key = f"custom_{m['type']}" + (f"_{r['recipe']}" if m["type"] == "choice" and r["recipe"] in ("generic", "catchall") and (m["answer"] == r["catchall"] or r["recipe"] == "generic" and r["catchall"] in m["criteria"]) else "")
            probes[key].append(D.Example(ex.context, [q], key))
    for r in routes:
        if r["domain"] in HELD_DOMAINS:
            probes[f"routing_{r['group']}"].append(D.Example(*[getattr(to_example(r, D, S1, "routing"), a) for a in ("context", "qs")], f"routing_{r['group']}"))
    pickle.dump(({}, dict(probes)), open("data/probes_v7.pkl", "wb")); print("[v7] probes:", {k: len(v) for k, v in probes.items()})


if __name__ == "__main__":
    main()
