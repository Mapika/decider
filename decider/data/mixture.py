"""The training mixture and the probes, in one place.

    python -m decider.data.mixture [--base data/tasks.pkl] [--out data/mixture.pkl] [--probes data/probes.pkl] [--mode full|delta]

`--base` is the converted-task cache from `python -m decider.data.core` (plain examples, ~1M).  On top of it (counts in MIX):

  general     every training task at the original protocol (label sets > 10 sub-sampled to 10)
  wide        full native label sets (CLINC 151, Banking 77, ...) and small label sets padded with unrelated labels, up to 255 options
  described   options with descriptions / JSON rubrics, often under opaque names          (teacher_data/label_descriptions.json)
  json        JSON states with several records, questions by path; half of them with "_index" written into long arrays
  single      multi-question examples asked one question at a time or as a reordered subset
  custom      teacher-written states with free-form noul / choice / score questions        (teacher_data/custom_questions.jsonl)
  routing     short messages over terse option lists with a generic bucket and a catch-all  (teacher_data/routing_messages.jsonl,
              and routing_terse.jsonl: the bucket carries a plain name such as `support`, no descriptions)
  commands    shell commands labelled safe / caution / destructive and "touches things outside the project"  (teacher_data/commands.jsonl)
  isolated    one yes/no row per Score level (and per option of some Choice questions)
  rules       rule-conditioned decisions over JSON records, every example with a state twin and a rule twin whose label flips
              (decider.data.rules: programmatic, exact labels; three domains and two rule families held out for the probes)

mode=full  is the single-run recipe: train Qwen3.5-2B-Base on it for one epoch (scripts/train.sh).
mode=delta keeps only a replay sample of `general`; it is what a continuation from an existing decider checkpoint uses
           (the released weights were produced this way, in stages; see docs/HISTORY.md).
Abstain options are added at train time (decider.data.augment.none_augment), and every example is rendered state-first or
schema-first at random by the trainer, so neither appears here.
Six teacher domains are held out of training and form the custom_* / routing_* probes."""
import argparse, collections, json, pickle, random
from decider import data as D
from decider import systemone as S1
from decider.data.augment import Builder, is_scale, narrow, indexed, isolated
from decider.data.teacher_questions import to_example, DOMAINS
from decider.data import rules as R

MIX = dict(wide_per_task=8000, padded=25000, described=40000, json=32000, json_indexed=20000, single=30000, isolated_scale=26000, isolated_choice=9000,
           isolated_routing=1200, rules=90000, replay=200000)
HELD_DOMAINS = set(DOMAINS[-6:])
SCALE_TASKS = ["helpsteer2", "helpsteer3_pref", "hate_speech_scales", "liar2", "prosocial_safety", "stsb"]
TEACHER = "teacher_data"


def npad(rng):          # distractors to add: log-uniform, plus a heavy tail of very long lists
    return rng.randint(100, 250) if rng.random() < 0.25 else int(2 ** rng.uniform(0, 7.9))


def load_teacher():
    recs = [json.loads(l) for l in open(f"{TEACHER}/custom_questions.jsonl")]; routes = [json.loads(l) for l in open(f"{TEACHER}/routing_messages.jsonl")]
    for r in routes:
        q = r["questions"][0]; q["criteria"] = {k: (None if v in ("null", "", None) else v) for k, v in q["criteria"].items()}
    # The zero-shot checker shares the bias being fixed (it sends generic cases to the catch-all), so a generic label written at
    # generation time is kept unless the checker chose a SPECIFIC option (real ambiguity). Everything else needs agreement.
    routes = [r for r in routes if r["questions"][0]["teacher_ok"] or (r["group"] == "generic" and r["questions"][0]["teacher_pred"] == r["catchall"])]
    import re, os
    BUCKET = re.compile(r"(support|help|question|account|service|assist|feedback|inquir|enquir|contact|issue|request|info|general|misc|other|else|query|queries|concern|complaint|problem|advice|guidance|topics?|customer|care|desk|admin|office|reception)", re.I)
    if os.path.exists(f"{TEACHER}/routing_terse.jsonl"):
        terse = [json.loads(l) for l in open(f"{TEACHER}/routing_terse.jsonl")]
        for r in terse: r["questions"][0]["criteria"] = {k: (None if v in ("null", "", None) else v) for k, v in r["questions"][0]["criteria"].items()}
        gname = {}                                                                 # the bucket the teacher named, per option list
        for r in terse:
            if r["group"] == "generic": gname[r["questions"][0]["instructions"] + json.dumps(list(r["questions"][0]["criteria"]))] = r["questions"][0]["answer"]
        keep = lambda r: BUCKET.search(gname.get(r["questions"][0]["instructions"] + json.dumps(list(r["questions"][0]["criteria"])), "") or "")   # the teacher sometimes calls a specific option the bucket
        routes += [r for r in terse if keep(r) and (r["questions"][0]["teacher_ok"] or (r["group"] == "generic" and r["questions"][0]["teacher_pred"] == r["catchall"]))]
    commands = []
    if os.path.exists(f"{TEACHER}/commands.jsonl"):
        order = ["safe", "caution", "destructive"]
        for r in (json.loads(l) for l in open(f"{TEACHER}/commands.jsonl")):
            q = r["questions"][0]; pred = q.get("teacher_pred")
            if q["teacher_ok"] or (pred in order and abs(order.index(pred) - order.index(q["answer"])) == 1):      # the "outside" flag keeps its written label: the checker is weak there
                commands.append(r)
    return recs, routes, commands


def formats(train, evals, B, rng):
    """wide / padded / described / json / single, from the plain examples."""
    by_task = collections.defaultdict(list)
    for e in train: by_task[e.task].append(e)
    fixed = sorted({k[0] for k, (o, sp) in B.sets.items() if sp == "train" and not is_scale(o)})
    big = [t for t in fixed if any(len(o) > 10 for (tt, _), (o, _) in B.sets.items() if tt == t)]; small = [t for t in fixed if t not in big]
    modes = ["named", "opaque", "named_json", "opaque_json"]; out = collections.defaultdict(list)
    for t in big:
        for e in rng.sample(by_task[t], min(MIX["wide_per_task"], len(by_task[t]))):
            n = max(len(q.options) for q in e.qs); r = rng.random()
            out["wide"].append(B.choice(e, wide=n if r < 0.35 else rng.randint(11, n) if r < 0.7 else None, pad=npad(rng) if rng.random() < 0.2 else 0,
                                        mode=rng.choice(modes) if rng.random() < 0.25 else "plain"))
    pool3 = [e for t in small for e in by_task[t] if all(len(q.options) >= 3 for q in e.qs)]
    for e in rng.sample(pool3, MIX["padded"]):
        out["padded"].append(B.choice(e, pad=npad(rng), mode=rng.choice(modes) if rng.random() < 0.3 else "plain"))
    for t in fixed:
        for e in rng.sample(by_task[t], min(MIX["described"] // len(fixed), len(by_task[t]))):
            out["described"].append(B.choice(e, mode=rng.choice(["named", "named", "opaque", "opaque", "named_json", "opaque_json"])))
    single = [e for e in train if len(e.qs) == 1 and e.qs[0].gold >= 0 and e.task not in ("games", "mario")]
    short = [e for e in single if len(e.context) <= 1200]; longer = [e for e in single if 300 <= len(e.context) <= 6000]; het = lambda: rng.random() < 0.4
    for _ in range(MIX["json"]):
        r = rng.random()
        if r < 0.10: out["json"].append(B.json_wrap(rng.choice(single if rng.random() < 0.5 else short)))
        elif r < 0.75: out["json"].append(B.json_state(short, rng.randint(2, 10), rng.randint(1, 4), 6000, hetero=het()))
        elif r < 0.93: out["json"].append(B.json_state(longer, rng.randint(4, 24), rng.randint(1, 4), 24000, hetero=het()))
        else: out["json"].append(B.json_state(longer, rng.randint(12, 60), rng.randint(1, 5), 48000, hetero=het()))
    for _ in range(MIX["json_indexed"]):
        r = rng.random()
        e = B.json_state(short, rng.randint(8, 40), rng.randint(1, 4), 12000, hetero=het()) if r < 0.55 else \
            B.json_state(longer, rng.randint(8, 60), rng.randint(1, 4), 40000, hetero=het()) if r < 0.8 else B.json_state(short, rng.randint(2, 7), rng.randint(1, 3), 6000, hetero=het())
        out["json_indexed"].append(indexed(e))
    for e in rng.sample([e for e in train if len(e.qs) > 1], MIX["single"]):      # independence: a question scored alone must be in-distribution
        qs = [rng.choice(e.qs)] if rng.random() < 0.6 else rng.sample(e.qs, rng.randint(1, len(e.qs)))
        x = narrow(D.Example(e.context, qs, e.task), rng); out["single"].append(D.Example(x.context, x.qs, e.task + "+fmt"))
    return out


def teacher_sets(recs, routes, rng, commands=()):
    out = collections.defaultdict(list)
    for r in recs:
        if r["domain"] in HELD_DOMAINS: continue
        ex = to_example(r, D, S1, "custom+fmt"); qs = list(zip(ex.qs, r["questions"]))
        special = [q for q, m in qs if m["type"] == "choice" and r["recipe"] in ("generic", "catchall") and
                   (any(w in m["answer"].lower() for w in ("general", "misc", "service", "support")) or m["answer"] == r["catchall"])]
        for _ in range(2):                                                         # two packed subsets (options are reshuffled at render time)
            out["custom"].append(D.Example(ex.context, [q for q, _ in rng.sample(qs, rng.randint(min(2, len(qs)), len(qs)))], "custom+fmt"))
        for q in special * 3 + rng.sample([q for q, _ in qs], min(4, len(qs))):    # asked alone; generic / catch-all cases weighted
            out["custom"].append(D.Example(ex.context, [q], "custom+fmt"))
        for q, m in zip(ex.qs, r["questions"]):
            if m["type"] == "score": out["isolated"] += isolated(ex, q, "custom+iso") * 2
            elif m["type"] == "choice" and rng.random() < 0.5: out["isolated"] += isolated(ex, q, "custom+iso")
    for r in commands:
        ex = to_example(r, D, S1, "commands+fmt"); out["commands"].append(ex)                    # both questions packed
        out["commands"].append(D.Example(ex.context, [rng.choice(ex.qs)], "commands+fmt"))       # and one alone
    tr_routes = [r for r in routes if r["domain"] not in HELD_DOMAINS]
    out["routing"] = [to_example(r, D, S1, "routing+fmt") for r in tr_routes]
    for r in rng.sample([r for r in tr_routes if r["questions"][0]["teacher_ok"]], min(MIX["isolated_routing"], len(tr_routes))):
        ex = to_example(r, D, S1, "routing"); out["isolated"] += isolated(ex, ex.qs[0], "routing+iso")
    return out


def isolated_sets(train, rng):
    scale = lambda q: len(q.options) >= 3 and is_scale(q.options)
    pool = [(e, q) for e in train if e.task in SCALE_TASKS for q in e.qs if scale(q) and q.gold >= 0]; out = []
    for e, q in rng.sample(pool, MIX["isolated_scale"]): out += isolated(e, q, e.task + "+iso")
    small = [(e, q) for e in rng.sample(train, 120000) for q in e.qs if 3 <= len(q.options) <= 8 and not scale(q) and q.gold >= 0 and len(e.context) < 1500 and e.task not in ("games", "mario")]
    for e, q in rng.sample(small, MIX["isolated_choice"]): out += isolated(e, q, e.task + "+iso")
    return out


def abstention_probes(evals, rng):
    """abstain_probe: held-out tasks + "none of the above"; in half, the gold option is removed.  offtopic_probe: an abstain option in
    varied wordings; in half, the whole option list comes from another task (abstaining is right)."""
    from decider.data.augment import ABSTAIN_WORDINGS
    src = ["trec", "bbc_news", "massive_scenario", "student_questions", "dolly_category", "fin_sentiment"]; a, o = [], []
    for t in src:
        for e in evals[t][:270]:
            q = e.qs[0]
            if len(q.options) < 3: continue
            if rng.random() < 0.5: opts = [x for i, x in enumerate(q.options) if i != q.gold] + ["none of the above"]; a.append(D.Example(e.context, [D.Q(q.text, opts, len(opts) - 1)], "abstain_probe"))
            else: a.append(D.Example(e.context, [D.Q(q.text, list(q.options) + ["none of the above"], q.gold)], "abstain_probe"))
            w = rng.choice(ABSTAIN_WORDINGS)
            if rng.random() < 0.5:
                other = evals[rng.choice([s for s in src if s != t])][0].qs[0].options; opts = rng.sample(other, min(len(other), len(q.options))) + [w]
                o.append(D.Example(e.context, [D.Q(q.text, opts, len(opts) - 1)], "offtopic_probe"))
            else: o.append(D.Example(e.context, [D.Q(q.text, list(q.options) + [w], q.gold)], "offtopic_probe"))
    return a, o


def probes(train, evals, desc, recs, routes):
    """Evaluation-only sets for the input shapes (built from eval splits and held-out teacher domains)."""
    P = Builder(train, evals, desc, seed=66); prng = random.Random(66); out = {}
    for n in D.tasks_heldout.NEW_TASKS:
        try: out[n] = D.load_task(n)[1]
        except Exception as ex: print("[mixture] probe task failed", n, ex)
    for t in ["bbc_news", "trec", "student_questions", "dolly_category", "fin_sentiment", "massive_scenario", "tweet_irony", "cr_reviews", "ag_news", "emotion", "banking77"]:
        exs = [e for e in evals[t] if (t, e.qs[0].text) in P.sets][:400]
        for mode in ["plain", "named", "opaque", "opaque_json"]:
            out[f"{mode}:{t}"] = []
            for e in exs:
                x = P.choice(e, wide=255, mode=mode, cap_desc=False); x.task = f"{mode}:{t}"; out[f"{mode}:{t}"].append(x)
    held = [e for t in ["bbc_news", "trec", "student_questions", "dolly_category", "fin_sentiment", "cr_reviews", "tweet_irony", "sciq", "social_iqa", "fin_phrasebank"] for e in evals[t] if len(e.qs) == 1]
    P.by.clear(); short = [e for e in held if len(e.context) <= 1500]
    for k in (1, 4, 16, 64):
        out[f"json_k{k}"] = [P.json_state(short, k, 1, 10**6, hetero=prng.random() < 0.5) for _ in range(500)]
    longp = [e for t in ["bbc_news", "quality", "pubmedqa"] for e in evals[t] if len(e.context) > 1200]
    out["json_long"] = [P.json_state(longp + held, 40, 1, 90000, hetero=True) for _ in range(200)]
    out["json_xl"] = [P.json_state(longp + held, 70, 1, 118000, hetero=True) for _ in range(100)]
    for k in ("json_k16", "json_k64", "json_long"): out[k + "_idx"] = [indexed(e) for e in out[k]]
    for r in recs:
        if r["domain"] not in HELD_DOMAINS: continue
        ex = to_example(r, D, S1, "custom")
        for q, m in zip(ex.qs, r["questions"]):
            special = m["type"] == "choice" and r["recipe"] in ("generic", "catchall") and (m["answer"] == r["catchall"] or r["recipe"] == "generic" and r["catchall"] in m["criteria"])
            out.setdefault(f"custom_{m['type']}" + (f"_{r['recipe']}" if special else ""), []).append(D.Example(ex.context, [q], "custom"))
    for r in routes:
        if r["domain"] in HELD_DOMAINS:
            ex = to_example(r, D, S1, "routing"); k = f"routing_{'terse_' if r['recipe'] == 'routing_terse' else ''}{r['group']}"; out.setdefault(k, []).append(D.Example(ex.context, ex.qs, k))
    out.update(R.probe_sets())
    for k, v in out.items():
        for e in v: e.task = k
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="data/tasks.pkl"); ap.add_argument("--out", default="data/mixture.pkl"); ap.add_argument("--probes", default="data/probes.pkl")
    ap.add_argument("--mode", default="full", choices=["full", "delta"]); ap.add_argument("--seed", type=int, default=6)
    a = ap.parse_args(); rng = random.Random(a.seed)
    train, evals = D.load_cache(a.base); desc = json.load(open(f"{TEACHER}/label_descriptions.json")); recs, routes, commands = load_teacher()
    if "abstain_probe" not in evals or "offtopic_probe" not in evals or not evals["offtopic_probe"]:
        evals["abstain_probe"], evals["offtopic_probe"] = abstention_probes(evals, random.Random(7))
    B = Builder(train, evals, desc, seed=a.seed); parts = formats(train, evals, B, rng)
    parts.update(teacher_sets(recs, routes, rng, commands)); parts["isolated"] += isolated_sets(train, rng)
    parts["rules"] = R.build(MIX["rules"], seed=a.seed + 11)
    general = [narrow(e, rng) for e in train]
    if a.mode == "delta": rng.shuffle(general); general = general[:MIX["replay"]]
    parts["general"] = general; out = [e for v in parts.values() for e in v]; rng.shuffle(out)
    print("[mixture]", {k: len(v) for k, v in parts.items()}, "total", len(out), flush=True)
    pickle.dump((out, evals), open(a.out, "wb"))
    pr = probes(train, evals, desc, recs, routes); pickle.dump(({}, pr), open(a.probes, "wb")); print("[mixture] probes:", {k: len(v) for k, v in pr.items()})


if __name__ == "__main__":
    main()
