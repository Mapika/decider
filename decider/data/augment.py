"""Data augmentations: everything that turns a plain (context, question, options, gold) example into the input shapes the
model has to handle.  All of it is derived from the existing examples; nothing here needs a model.

  none_augment        abstain options: added with the answer unchanged (75%), or the whole option list replaced by labels of an
                      unrelated task so that abstaining is right (25%)                                   [train time, per epoch]
  Builder.choice      described options (name: description / JSON rubric, often opaque names), full label sets up to 255
                      options, small label sets padded with labels from unrelated task families
  Builder.json_state  a JSON state holding several records; questions name one by path (`tickets[3].text`)
  indexed             "_index" written into long arrays (what systemone.render_state sends)
  narrow              the original protocol: large label sets sub-sampled to 10
  isolated            one yes/no row per level / option (systemone.ISOLATED): isolated level scoring
"""
import collections, json, pickle, random, re, string
from decider import data as D
from decider.prompt import is_abstain_option

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
        self.sets = fixed_label_sets(train, evals, skip=SKIP_TASKS)                       # (task, qtext) -> (options, split)
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
        return D.Example(e.context, qs, e.task + "+fmt")

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
        return D.Example(ctx, qs, "json_state+fmt")

    def json_wrap(self, e):
        rng = self.rng; key = rng.choice(["message", "text", "input", "body", "content", "document"])
        state = dict(self._noise(0), **{key: e.context})
        if rng.random() < 0.5: state = {rng.choice(["request", "event", "item", "ticket"]): state}
        return D.Example(json.dumps(state, ensure_ascii=False, indent=rng.choice([None, 1, 2])), e.qs, e.task + "+fmt")


SKIP_TASKS = {"mario", "synth", "games", "hh_rlhf", "shp", "ultrafeedback_pref", "reward_bench", "arena_pref", "helpsteer3_pref"}     # no describable label set


def fixed_label_sets(train, evals, min_n=50, skip=()):
    """(task, question text) -> (options, "train" | "eval") for every question whose option list never changes."""
    sets = collections.defaultdict(collections.Counter); split = {}
    for e in train:
        for q in e.qs:
            sets[(e.task, q.text)][tuple(q.options)] += 1; split[(e.task, q.text)] = "train"
    for t, v in evals.items():
        for e in v:
            for q in e.qs:
                if (t, q.text) not in split or split[(t, q.text)] == "eval":
                    sets[(t, q.text)][tuple(q.options)] += 1; split[(t, q.text)] = "eval"
    out = {}
    for k, c in sets.items():
        top, n = c.most_common(1)[0]
        if n / sum(c.values()) > 0.98 and sum(c.values()) >= min_n and k[0] not in skip:
            out[k] = (list(top), split[k])
    return out


def narrow(e, rng, cap=10):
    """The original protocol: label sets larger than `cap` are sub-sampled (gold and abstain options always kept)."""
    if all(len(q.options) <= cap for q in e.qs): return e
    qs = []
    for q in e.qs:
        if len(q.options) <= cap or q.gold < 0:
            qs.append(q); continue
        forced = {q.gold} | {i for i, o in enumerate(q.options) if is_abstain_option(o)}
        keep = sorted(rng.sample([i for i in range(len(q.options)) if i not in forced], max(0, cap - len(forced))) + list(forced))
        qs.append(D.Q(q.text, [q.options[i] for i in keep], keep.index(q.gold)))
    return D.Example(e.context, qs, e.task)


def indexed(e):
    """Re-render a JSON-state example the way the API sends it: long arrays carry their element index."""
    from decider.systemone import annotate_indices
    return D.Example(json.dumps(annotate_indices(json.loads(e.context)), ensure_ascii=False), e.qs, e.task)


def isolated(e, q, tag):
    """One yes/no example per option of q: the option alone, without its number or its neighbours; "yes" only for the gold one."""
    from decider.systemone import isolated_rows
    return [D.Example(e.context, [D.Q(text, opts, 1 if j == q.gold else 0)], tag) for j, (text, opts) in enumerate(isolated_rows(q.text, q.options))]


# ---- abstention (applied at train time, so every epoch sees different option lists)
NONE_OPT = "none of the above"
ABSTAIN_WORDINGS = ["none of the above", "none of these", "not listed here", "other", "something else", "unsure",
                    "does not apply", "no suitable option", "neither of these", "cannot tell from the text",
                    "none of the above (out of scope)", "other / not covered"]
NONE_GOLD_RATE = 0.25




def none_augment(e, rng, p, pool=None):
    """With prob p, for a question with >=3 options and no abstain-style option:
       75%: add an abstain option (random wording), gold unchanged  -> "an abstain option present does not mean abstain"
       25%: replace ALL options by labels drawn from other tasks + an abstain option, gold = abstain
            -> abstain when nothing on offer fits, not when the exact label is merely missing."""
    if p <= 0 or pool is None:
        return e
    qs = []
    for q in e.qs:
        if len(q.options) >= 3 and rng.random() < p and not any(is_abstain_option(o) for o in q.options):
            w = rng.choice(ABSTAIN_WORDINGS)
            if rng.random() < NONE_GOLD_RATE:
                others = [t for t in pool if t != e.task.split('+')[0]]
                src = pool[rng.choice(others)]
                k = min(len(q.options), len(src)); opts = rng.sample(src, k) + [w]
                qs.append(D.Q(q.text, opts, len(opts) - 1))
            else:
                qs.append(D.Q(q.text, list(q.options) + [w], q.gold))
        else:
            qs.append(q)
    return D.Example(e.context, qs, e.task)


def build_label_pool(train):
    """task -> sorted distinct option strings (only tasks with a fixed label set of 3..60 options)."""
    pool = {}
    for e in train:
        if "+" in e.task:                      # v6 re-renderings (descriptions, padding, JSON states): not a clean label set
            continue
        for q in e.qs:
            if 3 <= len(q.options) <= 60:
                pool.setdefault(e.task, set()).update(q.options)
    return {t: sorted(v) for t, v in pool.items() if 3 <= len(v) <= 200}
