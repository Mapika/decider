"""v6 delta set: the input shapes TypeSafe's Jev accepts and v5 was never trained on, built from existing data.

  described  options carry a description or a JSON rubric ({what, not_for, examples}); names are often opaque
             ("c7", "route_03") so the description has to be read                      (data/label_desc.json)
  wide       more than 10 options, up to 255: full native label sets (clinc 151, banking 77, massive 60, ...)
             and small label sets padded with labels from unrelated task families
  json       the state is a JSON document holding several records; each question names the part it is about
             with a path (`tickets[3].text`); from 2 records up to about 14k tokens
  single     multi-question examples asked one question at a time or as a reordered subset (independent scoring)
  replay     the general v4 mixture, so nothing else moves

Every new-format example gets task "<task>+v6" (the train-time abstain augmentation builds its label pool from
plain tasks only).  Large label sets are sub-sampled here, so train with --max_options 255.
   python -m decider.build_v6            -> data/tasks_v6_delta.pkl, data/probes_v6.pkl"""
import collections, json, pickle, random, re, string
from . import data as D
from . import data2, data3, data4  # noqa
from .describe_labels import fixed_label_sets
from .prompt import is_abstain_option

FAMILIES = dict(
    affect=["sst2", "sst5", "imdb", "yelp", "amazon_stars", "tweet_sentiment", "cr_reviews", "fin_sentiment", "fin_phrasebank",
            "emotion", "go_emotions", "tweet_emotion"],          # a sad text is also "very negative": never pad one with the other
    topic=["ag_news", "bbc_news", "yahoo_topics", "newsgroups", "dbpedia", "dbpedia_l2", "dbpedia_l3", "student_questions", "dolly_category", "trec", "trec_fine", "bias_in_bios"],
    intent=["clinc_oos", "banking77", "massive_intent", "massive_scenario", "bitext_support", "support_tickets", "hwu64"],
    toxicity=["civil_comments", "toxic_chat", "tweet_hate", "tweet_offensive", "hate_offensive", "hate_speech_scales", "prosocial_safety", "insincere_questions"],
    nli=["mnli", "snli", "cb", "rte", "fever", "liar2"])
FAM = {t: f for f, ts in FAMILIES.items() for t in ts}
KEYSETS = [("what", "not_for", "examples"), ("description", "excludes", "examples"), ("covers", "does_not_cover", "e.g."), ("definition", "not", "samples")]
LETTERS = string.ascii_lowercase


def is_scale(opts):
    return all(re.match(r"^-?\d+:", o) for o in opts)


def opaque_names(n, rng):
    s = rng.randrange(6)
    if s == 0: names = [f"c{i + 1}" for i in range(n)]
    elif s == 1: names = [f"option_{i + 1}" for i in range(n)]
    elif s == 2: names = [f"route_{i + 1:02d}" for i in range(n)]
    elif s == 3: names = [f"L{x}" for x in rng.sample(range(1000, 9999), n)]
    elif s == 4: names = [f"cat-{x}" for x in rng.sample(range(100, 999), n)]
    else:
        names = set()
        while len(names) < n: names.add("".join(rng.choice(LETTERS) for _ in range(3)) + str(rng.randrange(10)))
        names = sorted(names)
    rng.shuffle(names); return names


class Builder:
    def __init__(self, train, evals, desc, seed=0):
        self.rng = random.Random(seed); self.desc = desc
        self.sets = fixed_label_sets(train, evals)                       # (task, qtext) -> (options, split)
        self.by = collections.defaultdict(list)                          # short training inputs per label, for "examples"
        for e in train:
            for q in e.qs:
                k = (e.task, q.text)
                if k in self.sets and q.gold >= 0 and len(e.context) <= 160 and len(self.by[k + (q.options[q.gold],)]) < 60:
                    self.by[k + (q.options[q.gold],)].append(e.context)
        # distractor sources: fixed label sets with >= 4 plain (non-scale, non-abstain) labels, grouped by family
        self._cands = {}
        self.sources = [(k, [o for o in opts if not is_abstain_option(o)]) for k, (opts, sp) in self.sets.items()
                        if sp == "train" and len(opts) >= 4 and not is_scale(opts)]

    # ---- option-list transforms; an option is (name, source key) until it is rendered to a string
    def subsample(self, opts, gold, cap):
        if len(opts) <= cap: return opts, gold
        forced = {gold} | {i for i, (o, _) in enumerate(opts) if is_abstain_option(o)}
        keep = sorted(self.rng.sample([i for i in range(len(opts)) if i not in forced], max(0, cap - len(forced))) + list(forced))
        return [opts[i] for i in keep], keep.index(gold)

    def pad(self, opts, gold, task, m):
        """Add up to m distractor labels from label sets of other task families (fewer if not enough distinct ones exist)."""
        fam = FAM.get(task, task)
        if fam not in self._cands:
            self._cands[fam] = [(o, k) for k, labs in self.sources if FAM.get(k[0], k[0]) != fam for o in labs]
        have = {o.lower() for o, _ in opts}; extra = []
        for o, k in self.rng.sample(self._cands[fam], min(len(self._cands[fam]), 2 * m + 20)):
            if len(extra) >= m: break
            if o.lower() not in have:
                have.add(o.lower()); extra.append((o, k))
        return opts + extra, gold

    def render(self, opts, mode, ctx=None):
        """mode: plain | named | opaque | named_json | opaque_json"""
        if mode == "plain": return [o for o, _ in opts]
        rng = self.rng; names = opaque_names(len(opts), rng) if mode.startswith("opaque") else [o for o, _ in opts]
        keys = rng.choice(KEYSETS); style = rng.randrange(3); out = []
        for (o, k), name in zip(opts, names):
            d = self.desc.get(k[0], {}).get(k[1], {}).get(o)
            if d is None:                                               # no description: an opaque name would be unanswerable
                out.append(o if not mode.startswith("opaque") else f"{name}: {o}"); continue
            if not mode.startswith("opaque") and rng.random() < 0.12:
                out.append(name); continue                              # some options left undescribed
            if mode.endswith("json"):
                obj = {keys[0]: d["what"]}
                if rng.random() < 0.6: obj[keys[1]] = d["not_for"]
                exs = [x for x in self.by.get(k + (o,), []) if x != ctx]
                if exs and rng.random() < 0.5: obj[keys[2]] = rng.sample(exs, min(len(exs), rng.randint(1, 3)))
                if len(obj) == 1 and rng.random() < 0.3: obj = obj[keys[0]]
                out.append(f"{name}: {json.dumps(obj, ensure_ascii=False)}")
            else:
                s = d["what"] if style == 0 else f"{d['what']} Not for: {d['not_for']}" if style == 1 else (d["what"] if rng.random() < 0.5 else f"{d['what']} (not: {d['not_for']})")
                out.append(f"{name}: {s}")
        return out

    def choice(self, e, wide=None, pad=0, mode="plain", cap_desc=True):
        """Rebuild every fixed-label question of e.  wide: cap on native options (None = 10); pad: distractors to add."""
        qs = []
        for q in e.qs:
            k = (e.task, q.text)
            if k not in self.sets or q.gold < 0 or is_scale(q.options):
                qs.append(q); continue
            opts, gold = self.subsample([(o, k) for o in q.options], q.gold, wide or 10)
            m = mode
            if m != "plain" and cap_desc:                              # training only: keep described lists to a few thousand tokens
                cap = 24 if m.endswith("json") else 48
                opts, gold = self.subsample(opts, gold, cap); pad_q = min(pad, cap - len(opts))
            else:
                pad_q = pad
            if pad_q > 0 and len(q.options) >= 3:
                opts, gold = self.pad(opts, gold, e.task, min(pad_q, 255 - len(opts)))
            qs.append(D.Q(q.text, self.render(opts, m, e.context), gold))
        return D.Example(e.context, qs, e.task + "+v6")

    # ---- JSON states with path references
    def _noise(self, i):
        rng = self.rng; f = {}
        if rng.random() < 0.6: f["id"] = rng.choice([i + 1, f"r{i + 1}", f"{rng.randrange(16**6):06x}"])
        if rng.random() < 0.3: f["ts"] = f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00Z"
        if rng.random() < 0.25: f["source"] = rng.choice(["email", "web", "chat", "api", "import", "mobile"])
        if rng.random() < 0.2: f["lang"] = "en"
        return f

    def json_state(self, pool, k, n_q, budget_chars, hetero=False):
        """pool: list of single-question examples. Returns an Example with a JSON context and n_q path questions."""
        rng = self.rng
        if hetero:
            recs = rng.sample(pool, k)
        else:
            t = rng.choice(pool).task; same = [e for e in rng.sample(pool, min(len(pool), 4000)) if e.task == t]
            recs = (same * (k // max(1, len(same)) + 1))[:k] if len(same) < k else rng.sample(same, k)
        out, used = [], 0
        for e in recs:
            if used + len(e.context) > budget_chars and len(out) >= 2: break
            out.append(e); used += len(e.context)
        recs = out; k = len(recs)
        cont = rng.choice(["records", "items", "messages", "documents", "entries", "rows", "tickets", "posts", "inputs", "samples"])
        tkey = rng.choice(["text", "body", "content", "message", "value", "description"])
        shape = rng.choice(["list", "list", "list", "dict", "array", "nested"])
        if shape == "array":
            state = [e.context for e in recs]; paths = [f"[{i}]" for i in range(k)]
        elif shape == "dict":
            ids = [f"{rng.choice(LETTERS)}{i + 1}" for i in range(k)]
            state = {cont: {i_: dict(self._noise(j), **{tkey: e.context}) for j, (i_, e) in enumerate(zip(ids, recs))}}
            paths = [f"{cont}.{i_}.{tkey}" for i_ in ids]
        else:
            lst = [dict(self._noise(j), **{tkey: e.context}) for j, e in enumerate(recs)]
            paths = [f"{cont}[{j}].{tkey}" for j in range(k)]
            if shape == "nested":
                top = rng.choice(["case", "batch", "session", "request", "payload"])
                state = {top: {"id": f"{rng.randrange(10**5):05d}", cont: lst}, "note": rng.choice(["imported", "pending review", "auto-collected", "n/a"])}
                paths = [f"{top}.{p}" for p in paths]
            else:
                state = {cont: lst}
        qs = []
        for j in rng.sample(range(k), min(n_q, k)):
            q = recs[j].qs[0]; p = f"`{paths[j]}`"
            text = rng.choice(["{q} (about {p})", "For {p}: {q}", "{q} Judge only {p}.", "Considering {p}: {q}", "{p} - {q}"]).format(q=q.text, p=p)
            opts, gold = self.subsample([(o, None) for o in q.options], q.gold, 10)
            qs.append(D.Q(text, [o for o, _ in opts], gold))
        ctx = json.dumps(state, ensure_ascii=False, indent=rng.choice([None, None, 1, 2]))
        return D.Example(ctx, qs, "json_state+v6")

    def json_wrap(self, e):
        rng = self.rng; key = rng.choice(["message", "text", "input", "body", "content", "document"])
        state = dict(self._noise(0), **{key: e.context})
        if rng.random() < 0.5: state = {rng.choice(["request", "event", "item", "ticket"]): state}
        return D.Example(json.dumps(state, ensure_ascii=False, indent=rng.choice([None, 1, 2])), e.qs, e.task + "+v6")


def main():
    rng = random.Random(6); import time; t0 = time.time()
    train, evals = D.load_cache("data/tasks_v4.pkl")
    desc = json.load(open("data/label_desc.json"))
    B = Builder(train, evals, desc, seed=6)
    by_task = collections.defaultdict(list)
    for e in train: by_task[e.task].append(e)
    fixed_tasks = sorted({k[0] for k, (o, sp) in B.sets.items() if sp == "train" and not is_scale(o)})
    big = [t for t in fixed_tasks if any(len(o) > 10 for (tt, _), (o, _) in B.sets.items() if tt == t)]
    small = [t for t in fixed_tasks if t not in big]
    print("[v6] big-label tasks", big); print("[v6] small-label tasks", len(small))
    new = []
    npad = lambda: rng.randint(100, 250) if rng.random() < 0.25 else int(2 ** rng.uniform(0, 7.9))   # log-uniform, plus a heavy tail of very long lists
    # wide: native large label sets
    for t in big:
        for e in rng.sample(by_task[t], min(8000, len(by_task[t]))):
            n = max(len(q.options) for q in e.qs); r = rng.random()
            wide = n if r < 0.35 else rng.randint(11, n) if r < 0.7 else None
            mode = rng.choice(["named", "opaque", "named_json", "opaque_json"]) if rng.random() < 0.25 else "plain"
            pad = npad() if rng.random() < 0.2 else 0
            new.append(B.choice(e, wide=wide, pad=pad, mode=mode))
    n_wide = len(new); print(f"[v6] wide done {time.time()-t0:.0f}s", flush=True)
    # padded: small label sets + unrelated distractors
    pool3 = [e for t in small for e in by_task[t] if all(len(q.options) >= 3 for q in e.qs)]
    for e in rng.sample(pool3, 25000):
        mode = rng.choice(["named", "opaque", "named_json", "opaque_json"]) if rng.random() < 0.3 else "plain"
        new.append(B.choice(e, pad=npad(), mode=mode))
    n_pad = len(new) - n_wide
    # described, no padding (binary tasks included)
    per = 40000 // len(fixed_tasks)
    for t in fixed_tasks:
        for e in rng.sample(by_task[t], min(per, len(by_task[t]))):
            new.append(B.choice(e, mode=rng.choice(["named", "named", "opaque", "opaque", "named_json", "opaque_json"])))
    n_desc = len(new) - n_wide - n_pad; print(f"[v6] described done {time.time()-t0:.0f}s", flush=True)
    # json states
    single = [e for e in train if len(e.qs) == 1 and e.qs[0].gold >= 0 and e.task not in ("games", "mario")]
    short = [e for e in single if len(e.context) <= 1200]; longer = [e for e in single if 300 <= len(e.context) <= 6000]
    for i in range(32000):
        r = rng.random()
        if r < 0.10: new.append(B.json_wrap(rng.choice(single if rng.random() < 0.5 else short)))
        elif r < 0.75: new.append(B.json_state(short, rng.randint(2, 10), rng.randint(1, 4), 6000, hetero=rng.random() < 0.4))
        elif r < 0.93: new.append(B.json_state(longer, rng.randint(4, 24), rng.randint(1, 4), 24000, hetero=rng.random() < 0.4))
        else: new.append(B.json_state(longer, rng.randint(12, 60), rng.randint(1, 5), 48000, hetero=rng.random() < 0.4))
    n_json = len(new) - n_wide - n_pad - n_desc
    def narrow(e):                                  # replay keeps the original protocol: large label sets sub-sampled to 10
        if all(len(q.options) <= 10 for q in e.qs): return e
        qs = []
        for q in e.qs:
            opts, gold = B.subsample([(o, None) for o in q.options], q.gold, 10); qs.append(D.Q(q.text, [o for o, _ in opts], gold))
        return D.Example(e.context, qs, e.task)
    # independence: multi-question examples asked one question at a time, or as a random subset in random order, so a
    # question scored alone (one row per question at inference) is in-distribution and packed answers depend less on company
    multi = [e for e in train if len(e.qs) > 1]
    for e in rng.sample(multi, 30000):
        qs = [rng.choice(e.qs)] if rng.random() < 0.6 else rng.sample(e.qs, rng.randint(1, len(e.qs)))
        x = narrow(D.Example(e.context, qs, e.task)); new.append(D.Example(x.context, x.qs, e.task + "+v6"))
    n_ind = len(new) - n_wide - n_pad - n_desc - n_json
    gen = list(train); rng.shuffle(gen); replay = [narrow(e) for e in gen[:200000]]
    out = new + replay; rng.shuffle(out)
    print(f"[v6] wide {n_wide} + padded {n_pad} + described {n_desc} + json {n_json} + single/reordered {n_ind} + replay {len(replay)} = {len(out)}")
    keep = ["clinc_oos", "banking77", "support_tickets", "abstain_probe", "offtopic_probe", "trec", "sciq", "reward_bench", "hermes_tools", "helpsteer2", "paws", "agenttraj", "massive_scenario"]
    pickle.dump((out, {k: v for k, v in evals.items() if k in keep}), open("data/tasks_v6_delta.pkl", "wb"))

    # ---- probes (evaluation only; built from eval splits, so nothing here is trained on)
    P = Builder(train, evals, desc, seed=66); prng = random.Random(66); probes = {}
    for n in data4.NEW_TASKS:
        try: probes[n] = D.load_task(n)[1]
        except Exception as ex: print("[v6] probe task failed", n, ex)
    for t in ["bbc_news", "trec", "student_questions", "dolly_category", "fin_sentiment", "massive_scenario", "tweet_irony", "cr_reviews", "ag_news", "emotion", "banking77"]:
        exs = [e for e in evals[t] if (t, e.qs[0].text) in P.sets][:400]
        for mode in ["plain", "named", "opaque", "opaque_json"]:
            out_ = []
            for e in exs:
                x = P.choice(e, wide=255, mode=mode, cap_desc=False); x.task = f"{mode}:{t}"; out_.append(x)
            probes[f"{mode}:{t}"] = out_
    held = [e for t in ["bbc_news", "trec", "student_questions", "dolly_category", "fin_sentiment", "cr_reviews", "tweet_irony", "sciq", "social_iqa", "fin_phrasebank"] for e in evals[t] if len(e.qs) == 1]
    P.by.clear()
    for k, budget in [(1, 10**6), (4, 10**6), (16, 10**6), (64, 10**6)]:
        probes[f"json_k{k}"] = [P.json_state([e for e in held if len(e.context) <= 1500], k, 1, budget, hetero=prng.random() < 0.5) for _ in range(500)]
    longp = [e for t in ["bbc_news", "quality", "pubmedqa"] for e in evals[t] if len(e.context) > 1200]
    probes["json_long"] = [P.json_state(longp + held, 40, 1, 90000, hetero=True) for _ in range(200)]
    for v in probes.values():
        for e in v:
            if e.task.endswith("+v6"): e.task = e.task[:-3]
    pickle.dump(({}, probes), open("data/probes_v6.pkl", "wb"))
    print("[v6] probes:", {k: len(v) for k, v in probes.items()})


if __name__ == "__main__":
    main()
