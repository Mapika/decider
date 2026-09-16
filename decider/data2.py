"""Mixture v2: ordinal scales, pairwise preference judging, tool selection, more QA/NLI, abstention probe."""
import json, random, re
from collections import defaultdict
from .data import task, Example, Q, _ld, _sub, _cls, _mcq, _split_pair, _both, TRAIN_CAP, EVAL_CAP, SEED, TASKS


def _scale(legend):
    """legend: dict level->label (levels sortable as float). Returns option strings 'k: label' in level order."""
    keys = sorted(legend, key=lambda k: float(k))
    return keys, [f"{k}: {legend[k]}" for k in keys]


L5 = {0: "not at all", 1: "slightly", 2: "somewhat", 3: "very", 4: "extremely"}
NONE = "no function call needed"


def _clip(s, n):
    s = s if isinstance(s, str) else str(s)
    return s if len(s) <= n else s[:n] + " ..."


# ------------------------------------------------------------ ordinal scales
@task("helpsteer2")
def _hs2():
    tr, ev = _split_pair("nvidia/HelpSteer2", None, "train", "validation")
    legends = {
        "helpfulness": {0: "not helpful", 1: "slightly helpful", 2: "partially helpful", 3: "mostly helpful", 4: "extremely helpful"},
        "correctness": {0: "completely incorrect", 1: "mostly incorrect", 2: "partially correct", 3: "mostly correct", 4: "completely correct"},
        "coherence": {0: "incoherent", 1: "mostly incoherent", 2: "partially coherent", 3: "mostly coherent", 4: "perfectly coherent"},
        "complexity": {0: "basic, anyone could write it", 1: "simple", 2: "intermediate", 3: "advanced", 4: "expert-level"},
        "verbosity": {0: "far too short", 1: "somewhat short", 2: "adequate length", 3: "somewhat long", 4: "far too long"},
    }
    qtext = {"helpfulness": "How helpful is the response to the prompt?", "correctness": "How correct and accurate is the response?",
             "coherence": "How coherent and clear is the response?", "complexity": "What level of expertise does the response's content require?",
             "verbosity": "How verbose is the response relative to what the prompt asks for?"}
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            ctx = f"Prompt:\n{_clip(r['prompt'], 2500)}\n\nResponse:\n{_clip(r['response'], 2500)}"
            qs = []
            for a in legends:
                keys, opts = _scale(legends[a]); v = int(r[a])
                if v < 0 or v > 4:
                    continue
                qs.append(Q(qtext[a], opts, keys.index(v)))
            out.append(Example(ctx, qs, t))
        return out
    return _both(conv, tr, ev, "helpsteer2")


@task("helpsteer3_pref")
def _hs3():
    tr, ev = _split_pair("nvidia/HelpSteer3", "preference", "train", "validation")
    legend = {-3: "response 1 is much better", -2: "response 1 is better", -1: "response 1 is slightly better", 0: "both are about the same",
              1: "response 2 is slightly better", 2: "response 2 is better", 3: "response 2 is much better"}
    keys, opts = _scale(legend)
    def conv(ds, t, cap):
        ds = ds.filter(lambda r: str(r["language"]).lower().startswith("en"))
        out = []
        for r in _sub(ds, cap):
            conv_txt = "\n".join(f"{m['role']}: {_clip(m['content'], 1200)}" for m in r["context"][-4:])
            ctx = f"Conversation:\n{_clip(conv_txt, 3000)}\n\nResponse 1:\n{_clip(r['response1'], 1800)}\n\nResponse 2:\n{_clip(r['response2'], 1800)}"
            out.append(Example(ctx, [Q("Which response is the better final reply, and by how much?", opts, keys.index(int(r["overall_preference"])))], t))
        return out
    return _both(conv, tr, ev, "helpsteer3_pref")


@task("stsb")
def _stsb():
    tr, ev = _split_pair("SetFit/stsb", None, "train", "test")
    legend = {0: "completely unrelated", 1: "on the same topic but different meaning", 2: "share some details", 3: "roughly equivalent, important details differ",
              4: "mostly equivalent, minor details differ", 5: "completely equivalent in meaning"}
    keys, opts = _scale(legend)
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Text 1: {r['text1']}\nText 2: {r['text2']}", lambda r: int(round(float(r["label"]))),
                                "How similar in meaning are the two texts?", opts, t, cap)
    return _both(f, tr, ev, "stsb")


@task("hate_speech_scales")
def _mhs():
    ds = _ld("ucberkeley-dlab/measuring-hate-speech", None, "train")
    cols = ["hatespeech", "insult", "dehumanize", "violence", "sentiment"]
    agg = defaultdict(lambda: defaultdict(list)); text = {}
    for r in ds:
        cid = r["comment_id"]; text[cid] = r["text"]
        for c in cols:
            if r[c] is not None:
                agg[cid][c].append(float(r[c]))
    hs_leg = {0: "not hate speech", 1: "unclear or borderline", 2: "hate speech"}
    sent_leg = {0: "strongly positive", 1: "positive", 2: "neutral", 3: "negative", 4: "strongly negative"}
    ids = sorted(agg); random.Random(SEED).shuffle(ids)
    def conv(idl, t, cap):
        out = []
        for cid in idl[:cap]:
            a = agg[cid]
            if not all(a[c] for c in cols):
                continue
            m = {c: int(round(sum(a[c]) / len(a[c]))) for c in cols}
            k, o = _scale(hs_leg); qs = [Q("Is this comment hate speech?", o, k.index(min(2, max(0, m["hatespeech"]))))]
            for c, qt in [("insult", "How insulting is this comment?"), ("dehumanize", "How dehumanizing is this comment?"), ("violence", "How much does this comment incite violence?")]:
                k, o = _scale(L5); qs.append(Q(qt, o, k.index(min(4, max(0, m[c])))))
            k, o = _scale(sent_leg); qs.append(Q("What is the sentiment of this comment?", o, k.index(min(4, max(0, m["sentiment"])))))
            out.append(Example(_clip(text[cid], 2000), qs, t))
        return out
    return conv(ids[1500:], "hate_speech_scales", TRAIN_CAP), conv(ids[:1500], "hate_speech_scales", EVAL_CAP)


@task("liar2")
def _liar():
    tr, ev = _split_pair("chengxuphd/liar2", None, "train", "test")
    legend = {0: "pants on fire (absurdly false)", 1: "false", 2: "barely true", 3: "half true", 4: "mostly true", 5: "true"}
    keys, opts = _scale(legend)
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Statement: {r['statement']}\nSpeaker: {r['speaker']} ({_clip(r['speaker_description'] or '', 300)})\nSubject: {r['subject']}\nDate: {r['date']}",
                                lambda r: int(r["label"]), "How truthful is this political statement, as rated by fact-checkers?", opts, t, cap)
    return _both(f, tr, ev, "liar2")


@task("prosocial_safety")
def _pro():
    tr, ev = _split_pair("allenai/prosocial-dialog", None, "train", "validation")
    legend = {0: "casual, no safety concern", 1: "possibly needs caution", 2: "probably needs caution", 3: "needs caution", 4: "needs intervention"}
    lab = {"__casual__": 0, "__possibly_needs_caution__": 1, "__probably_needs_caution__": 2, "__needs_caution__": 3, "__needs_intervention__": 4}
    keys, opts = _scale(legend)
    f = lambda ds, t, cap: _cls(ds, lambda r: r["context"], lambda r: lab.get(r["safety_label"]), "How much does this utterance call for a cautious or corrective response?", opts, t, cap)
    return _both(f, tr, ev, "prosocial_safety")


# ------------------------------------------------------------ pairwise preference judging
def _pair_ex(ctx, r1, r2, better, t, rng, tie=False, extra=None):
    """better: 0 -> r1, 1 -> r2, 2 -> tie. Randomise order so position carries no signal."""
    flip = rng.random() < 0.5
    a, b = (r2, r1) if flip else (r1, r2)
    g = better if better == 2 or not flip else 1 - better
    opts = ["response A", "response B"] + (["about equally good"] if tie else [])
    qs = [Q("Which response is the better answer to the prompt?", opts, g)]
    return Example(f"{ctx}\n\nResponse A:\n{a}\n\nResponse B:\n{b}", qs + (extra or []), t)


@task("ultrafeedback_pref")
def _uf():
    tr, ev = _split_pair("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs", "test_prefs")
    keys, opts = _scale({i: f"{i}/10" for i in range(1, 11)})
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        ds = ds.filter(lambda r: r["score_chosen"] > r["score_rejected"])
        for r in _sub(ds, cap):
            c = r["chosen"][-1]["content"]; j = r["rejected"][-1]["content"]
            ctx = f"Prompt:\n{_clip(r['prompt'], 2000)}"
            ex = _pair_ex(ctx, _clip(c, 1800), _clip(j, 1800), 0, t, rng)
            out.append(ex)
        return out
    return _both(conv, tr, ev, "ultrafeedback_pref")


@task("shp")
def _shp():
    tr, ev = _split_pair("stanfordnlp/shp", None, "train", "test")
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        ds = ds.filter(lambda r: r["score_ratio"] >= 2.0)
        for r in _sub(ds, cap):
            ctx = f"Forum post ({r['domain'].split('_')[0]}):\n{_clip(r['history'], 2000)}"
            better = 0 if int(r["labels"]) == 1 else 1     # labels==1 -> A preferred
            out.append(_pair_ex(ctx, _clip(r["human_ref_A"], 1500), _clip(r["human_ref_B"], 1500), better, t, rng))
        return out
    return _both(conv, tr, ev, "shp")


@task("hh_rlhf")
def _hh():
    tr, ev = _split_pair("Anthropic/hh-rlhf", None, "train", "test")
    def split_last(s):
        i = s.rfind("\n\nAssistant:")
        return s[:i].strip(), s[i + len("\n\nAssistant:"):].strip()
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(ds, cap):
            p1, c = split_last(r["chosen"]); p2, j = split_last(r["rejected"])
            if p1 != p2 or not c or not j or c == j:
                continue
            out.append(_pair_ex(f"Conversation:\n{_clip(p1, 2500)}", _clip(c, 1500), _clip(j, 1500), 0, t, rng))
        return out
    return conv(tr, "hh_rlhf", 12000), conv(ev, "hh_rlhf", EVAL_CAP)


@task("arena_pref", heldout=True)
def _arena():
    ds = _ld("lmarena-ai/arena-human-preference-55k", None, "train")
    def conv(d, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(d, cap * 3):
            try:
                ps = json.loads(r["prompt"]); ra = json.loads(r["response_a"]); rb = json.loads(r["response_b"])
            except Exception:
                continue
            if len(ps) != 1 or not ra[0] or not rb[0]:
                continue
            better = 0 if r["winner_model_a"] else (1 if r["winner_model_b"] else 2)
            out.append(_pair_ex(f"Prompt:\n{_clip(ps[0], 2000)}", _clip(ra[0], 1800), _clip(rb[0], 1800), better, t, rng, tie=True))
            if len(out) >= cap:
                break
        return out
    return [], conv(ds, "arena_pref", EVAL_CAP)


@task("reward_bench", heldout=True)
def _rb():
    ds = _ld("allenai/reward-bench", "default", "filtered")
    def conv(d, t, cap):
        rng = random.Random(SEED)
        return [_pair_ex(f"Prompt:\n{_clip(r['prompt'], 2000)}", _clip(r["chosen"], 1800), _clip(r["rejected"], 1800), 0, t, rng) for r in _sub(d, cap)]
    return [], conv(ds, "reward_bench", EVAL_CAP)


# ------------------------------------------------------------ tool / function selection
def _tool_ex(user_msg, funcs, gold_name, pool, rng, t, k=6):
    """funcs: dict name->description available in this context; gold_name None -> no call."""
    names = list(funcs)
    while len(names) < k - 1 and pool:
        n = rng.choice(pool)
        if n not in names:
            names.append(n)
    rng.shuffle(names)
    desc = "\n".join(f"- {n}: {funcs.get(n) or pool_desc.get(n, '')}" for n in names)
    ctx = f"Available functions:\n{_clip(desc, 3000)}\n\nUser request:\n{_clip(user_msg, 1500)}"
    opts = names + [NONE]
    g = opts.index(gold_name) if gold_name in opts else len(opts) - 1
    return Example(ctx, [Q("Which function should be called to handle this request?", opts, g)], t)


pool_desc = {}


@task("glaive_tools")
def _glaive():
    ds = _ld("glaiveai/glaive-function-calling-v2", None, "train").shuffle(seed=SEED)
    rx_obj = re.compile(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"', re.S)
    def parse(r):
        funcs = dict(rx_obj.findall(r["system"]))
        chat = r["chat"]
        m = re.search(r"USER:\s*(.*?)\n\n\nASSISTANT:\s*(.*?)(?:<\|endoftext\|>|\n\n\n)", chat, re.S)
        if not m or not funcs:
            return None
        user, asst = m.group(1).strip(), m.group(2).strip()
        if asst.startswith("<functioncall>"):
            mm = re.search(r'"name":\s*"([^"]+)"', asst)
            gold = mm.group(1) if mm else None
            if gold not in funcs:
                return None
        else:
            gold = None
        return user, funcs, gold
    parsed = [p for p in (parse(r) for r in ds.select(range(60000))) if p]
    for _, f, _ in parsed:
        pool_desc.update(f)
    pool = list(pool_desc)
    def conv(items, t, cap):
        rng = random.Random(SEED); out = []
        calls = [p for p in items if p[2]]; nones = [p for p in items if not p[2]]
        sel = calls[: int(cap * 0.75)] + nones[: cap - int(cap * 0.75)]
        rng.shuffle(sel)
        return [_tool_ex(u, f, g, pool, rng, t) for u, f, g in sel]
    return conv(parsed[2000:], "glaive_tools", TRAIN_CAP), conv(parsed[:2000], "glaive_tools", EVAL_CAP)


@task("toolace")
def _toolace():
    ds = _ld("Team-ACE/ToolACE", None, "train").shuffle(seed=SEED)
    def parse(r):
        i = r["system"].find("you can invoke:")
        if i < 0:
            return None
        try:
            fl, _ = json.JSONDecoder().raw_decode(r["system"][i + len("you can invoke:"):].lstrip())
        except Exception:
            return None
        funcs = {f["name"]: f.get("description", "") for f in fl if "name" in f}
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            return None
        a = conv[1]["value"].strip()
        if a.startswith("["):
            mm = re.match(r"\[\s*([^(\]]+?)\s*\(", a)
            gold = mm.group(1).strip() if mm else None
            if gold not in funcs:
                return None
        else:
            gold = None
        return conv[0]["value"], funcs, gold
    parsed = [p for p in (parse(r) for r in ds) if p]
    for _, f, _ in parsed:
        pool_desc.update(f)
    pool = list(pool_desc)
    def conv(items, t, cap):
        rng = random.Random(SEED)
        return [_tool_ex(u, f, g, pool, rng, t, k=8) for u, f, g in items[:cap]]
    return conv(parsed[1000:], "toolace", TRAIN_CAP), conv(parsed[:1000], "toolace", EVAL_CAP)


@task("hermes_tools", heldout=True)
def _hermes():
    ds = _ld("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", "train")
    def parse(r):
        try:
            tl = json.loads(r["tools"])
        except Exception:
            return None
        funcs = {t["function"]["name"]: t["function"].get("description", "") for t in tl if "function" in t}
        hum = next((c["value"] for c in r["conversations"] if c["from"] == "human"), None)
        gpt = next((c["value"] for c in r["conversations"] if c["from"] == "gpt"), "")
        mm = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", gpt, re.S)
        gold = None
        if mm:
            try:
                gold = json.loads(mm.group(1)).get("name")
            except Exception:
                m2 = re.search(r"'name':\s*'([^']+)'", mm.group(1)); gold = m2.group(1) if m2 else None
        if not hum or not funcs or (gold and gold not in funcs):
            return None
        return hum, funcs, gold
    parsed = [p for p in (parse(r) for r in ds) if p]
    pool = list({n for _, f, _ in parsed for n in f}); pool_desc.update({n: d for _, f, _ in parsed for n, d in f.items()})
    rng = random.Random(SEED)
    return [], [_tool_ex(u, f, g, pool, rng, "hermes_tools", k=8) for u, f, g in parsed[:EVAL_CAP]]


# ------------------------------------------------------------ more QA / NLI / relevance
@task("copa")
def _copa():
    tr, ev = _split_pair("aps/super_glue", "copa", "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["premise"], lambda r: [r["choice1"], r["choice2"]], lambda r: int(r["label"]),
                                "What was the CAUSE of this?", t, cap)
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            q = "What was the cause of this?" if r["question"] == "cause" else "What happened as a result?"
            out.append(Example(r["premise"], [Q(q, [r["choice1"], r["choice2"]], int(r["label"]))], t))
        return out
    return _both(conv, tr, ev, "copa")


@task("wic")
def _wic():
    tr, ev = _split_pair("aps/super_glue", "wic", "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Word: {r['word']}\nSentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}", lambda r: int(r["label"]),
                                "Is the word used with the same meaning in both sentences?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "wic")


@task("multirc")
def _multirc():
    tr, ev = _split_pair("aps/super_glue", "multirc", "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Paragraph: {_clip(r['paragraph'], 2500)}\nQuestion: {r['question']}\nCandidate answer: {r['answer']}", lambda r: int(r["label"]),
                                "Is the candidate answer correct according to the paragraph?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "multirc")


@task("cb", heldout=True)
def _cb():
    tr, ev = _split_pair("aps/super_glue", "cb", "train", "validation")
    from .data import NLI
    cb2nli = {0: 0, 1: 2, 2: 1}      # super_glue cb: 0 entailment, 1 contradiction, 2 neutral -> NLI list order
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Text 1: {r['premise']}\nText 2: {r['hypothesis']}", lambda r: cb2nli.get(int(r["label"]), -1), "What is the logical relation between text 1 and text 2?", NLI, t, cap)
    return _both(f, tr, ev, "cb")


@task("fever")
def _fever():
    tr, ev = _split_pair("copenlu/fever_gold_evidence", None, "train", "validation")
    names = ["supported by the evidence", "refuted by the evidence", "not enough information"]
    lab = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    def ctx(r):
        ev_txt = "\n".join(f"[{e[0]}] {e[2]}" for e in r["evidence"][:4] if len(e) >= 3)
        return f"Claim: {r['claim']}\nEvidence:\n{_clip(ev_txt, 2500)}"
    f = lambda ds, t, cap: _cls(ds, ctx, lambda r: lab.get(r["label"]), "Is the claim supported or refuted by the evidence?", names, t, cap)
    return _both(f, tr, ev, "fever")


@task("wiki_qa")
def _wqa():
    tr, ev = _split_pair("microsoft/wiki_qa", None, "train", "test")
    def conv(ds, t, cap):
        pos = ds.filter(lambda r: r["label"] == 1); neg = ds.filter(lambda r: r["label"] == 0).shuffle(seed=SEED)
        n = min(len(pos), cap // 3)
        from datasets import concatenate_datasets
        d = concatenate_datasets([pos.select(range(n)), neg.select(range(min(len(neg), 2 * n)))]).shuffle(seed=SEED)
        return _cls(d, lambda r: f"Question: {r['question']}\nCandidate sentence (from '{r['document_title']}'): {r['answer']}", lambda r: int(r["label"]),
                    "Does the candidate sentence answer the question?", ["no", "yes"], t, cap)
    return _both(conv, tr, ev, "wiki_qa")


@task("msmarco_rel")
def _msm():
    tr, ev = _split_pair("microsoft/ms_marco", "v1.1", "train", "validation")
    def conv(ds, t, cap):
        rng = random.Random(SEED); out = []
        for r in _sub(ds, cap):
            ps = r["passages"]; sel = [i for i, s in enumerate(ps["is_selected"]) if s == 1]
            if not sel:
                continue
            i = sel[0] if rng.random() < 0.5 else rng.choice([j for j in range(len(ps["passage_text"])) if j not in sel] or sel)
            out.append(Example(f"Query: {r['query']}\nPassage: {_clip(ps['passage_text'][i], 1500)}",
                               [Q("Does the passage answer the query?", ["no", "yes"], int(i in sel))], t))
        return out
    return _both(conv, tr, ev, "msmarco_rel")


@task("medmcqa")
def _medmcqa():
    tr, ev = _split_pair("openlifescienceai/medmcqa", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: int(r["cop"]), "Which option is correct?", t, cap)
    return _both(f, tr, ev, "medmcqa")


@task("quality", heldout=True)
def _quality():
    tr, ev = _split_pair("emozilla/quality", None, "train", "validation")
    def conv(ds, t, cap):
        base = min(int(r["answer"]) for r in ds)
        return _mcq(ds, lambda r: f"Article: {_clip(r['article'], 5000)}\nQuestion: {r['question']}", lambda r: list(r["options"]), lambda r: int(r["answer"]) - base,
                    "Which option correctly answers the question about the article?", t, cap)
    return _both(conv, tr, ev, "quality")


@task("xstory_cloze", heldout=True)
def _xsc():
    ds = _ld("juletxara/xstory_cloze", "en", "eval")
    return [], _mcq(ds, lambda r: " ".join(r[f"input_sentence_{i}"] for i in range(1, 5)), lambda r: [r["sentence_quiz1"], r["sentence_quiz2"]],
                    lambda r: int(r["answer_right_ending"]) - 1, "Which sentence is the right ending of the story?", "xstory_cloze", EVAL_CAP)


@task("abstain_probe", heldout=True)
def _abstain():
    return [], []       # filled by build_v2 from held-out eval sets


NEW_TASKS = [n for n in TASKS if TASKS[n]["loader"].__module__ == __name__]
