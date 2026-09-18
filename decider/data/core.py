"""Task registry: many public decision datasets -> one unified format.

Example = one context + one or more typed questions, each with a fixed option
list and a gold index.  Large label sets are sub-sampled to MAX_OPTIONS at
formatting time (gold always kept), so the model must condition on the
candidate list rather than memorise a fixed head.
"""
import random, re
from dataclasses import dataclass, field
from datasets import load_dataset

MAX_OPTIONS = 10
SEED = 0
TRAIN_CAP = 20000     # per task
EVAL_CAP = 1500       # per task


@dataclass
class Q:
    text: str
    options: list          # canonical option strings
    gold: int              # index into options (or -1 for unknown)


@dataclass
class Example:
    context: str
    qs: list               # list[Q]
    task: str
    image: bytes = None    # optional PNG bytes (vision tasks); use getattr(e, "image", None) for old caches


TASKS = {}     # name -> dict(loader=fn, heldout=bool)


def task(name, heldout=False):
    def deco(fn):
        TASKS[name] = dict(loader=fn, heldout=heldout)
        return fn
    return deco


def _nice(label):
    return str(label).replace("_", " ").strip()


def _ld(name, cfg=None, split=None, rev=None):
    kw = dict(split=split)
    if rev:
        kw["revision"] = rev
    return load_dataset(name, cfg, **kw) if cfg else load_dataset(name, **kw)


def _sub(ds, n, seed=SEED):
    if n is None or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _cls(ds, text_fn, label_fn, question, names, task, cap, seed=SEED):
    """Single-label classification -> one question per example."""
    ds = _sub(ds, cap, seed)
    out = []
    for r in ds:
        y = label_fn(r)
        if y is None or y < 0 or y >= len(names):
            continue
        ctx = text_fn(r)
        if not ctx or not ctx.strip():
            continue
        out.append(Example(ctx.strip(), [Q(question, list(names), int(y))], task))
    return out


def _names(ds, col="label"):
    f = ds.features[col]
    return [_nice(n) for n in f.names]


def _split_pair(name, cfg, train_split, eval_split, rev=None):
    tr = _ld(name, cfg, train_split, rev)
    ev = _ld(name, cfg, eval_split, rev)
    return tr, ev


def _both(loader_fn, tr, ev, task):
    return loader_fn(tr, task, TRAIN_CAP), loader_fn(ev, task, EVAL_CAP)


# ---------------------------------------------------------------- intents
@task("clinc_oos")
def _clinc():
    tr, ev = _split_pair("clinc/clinc_oos", "plus", "train", "test")
    names = _names(tr, "intent")
    oos = names.index("oos")
    names[oos] = "none of the above (out of scope)"
    q = "What is the intent of this user message?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["intent"], q, names, t, cap)
    return _both(f, tr, ev, "clinc_oos")


@task("banking77")
def _banking():
    tr, ev = _split_pair("mteb/banking77", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    q = "Which banking-support intent does this customer message express?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, [_nice(n) for n in names], t, cap)
    return _both(f, tr, ev, "banking77")


@task("massive_intent")
def _massive():
    tr, ev = _split_pair("SetFit/amazon_massive_intent_en-US", None, "train", "test")
    names = sorted(set(tr["label_text"]))
    idx = {n: i for i, n in enumerate(names)}
    names_n = [_nice(n) for n in names]
    q = "What is the intent of this voice-assistant utterance?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, names_n, t, cap)
    return _both(f, tr, ev, "massive_intent")


@task("massive_scenario", heldout=True)
def _massive_sc():
    tr, ev = _split_pair("SetFit/amazon_massive_scenario_en-US", None, "train", "test")
    names = sorted(set(tr["label_text"]))
    idx = {n: i for i, n in enumerate(names)}
    q = "Which scenario (domain) does this voice-assistant utterance belong to?"
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_text"]], q, [_nice(n) for n in names], t, cap)
    return _both(f, tr, ev, "massive_scenario")


@task("bitext_support")
def _bitext():
    ds = _ld("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "train")
    ds = ds.shuffle(seed=SEED)
    cats = sorted(set(ds["category"])); intents = sorted(set(ds["intent"]))
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap):
            out.append(Example(r["instruction"].strip(), [
                Q("Which support category does this request fall under?", [_nice(c) for c in cats], cats.index(r["category"])),
                Q("What is the specific intent of this request?", [_nice(c) for c in intents], intents.index(r["intent"])),
            ], t))
        return out
    n = len(ds); ev = ds.select(range(0, 3000)); tr = ds.select(range(3000, n))
    return conv(tr, "bitext_support", TRAIN_CAP), conv(ev, "bitext_support", EVAL_CAP)


@task("support_tickets")
def _tickets():
    ds = _ld("Tobi-Bueck/customer-support-tickets", None, "train")
    ds = ds.filter(lambda r: r["language"] == "en" and r["body"] and r["type"] and r["queue"] and r["priority"]).shuffle(seed=SEED)
    types = sorted(set(ds["type"])); queues = sorted(set(ds["queue"])); pris = sorted(set(ds["priority"]))
    def conv(d, t, cap):
        out = []
        for r in _sub(d, cap):
            ctx = f"Subject: {r['subject'] or ''}\n{r['body']}"[:2500]
            out.append(Example(ctx, [
                Q("What type of ticket is this?", types, types.index(r["type"])),
                Q("Which queue should this ticket be routed to?", queues, queues.index(r["queue"])),
                Q("What priority should this ticket get?", pris, pris.index(r["priority"])),
            ], t))
        return out
    n = len(ds); ev = ds.select(range(0, 3000)); tr = ds.select(range(3000, n))
    return conv(tr, "support_tickets", TRAIN_CAP), conv(ev, "support_tickets", EVAL_CAP)


# ---------------------------------------------------------------- topic / news
@task("ag_news")
def _ag():
    tr, ev = _split_pair("fancyzhx/ag_news", None, "train", "test")
    names = _names(tr)
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the topic of this news article?", names, t, cap)
    return _both(f, tr, ev, "ag_news")


@task("dbpedia")
def _dbp():
    tr, ev = _split_pair("fancyzhx/dbpedia_14", None, "train", "test")
    names = _names(tr)
    f = lambda ds, t, cap: _cls(ds, lambda r: f"{r['title']}: {r['content']}", lambda r: r["label"], "What kind of entity does this encyclopedia text describe?", names, t, cap)
    return _both(f, tr, ev, "dbpedia")


@task("yahoo_topics")
def _yahoo():
    tr, ev = _split_pair("community-datasets/yahoo_answers_topics", None, "train", "test")
    names = _names(tr, "topic")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"{r['question_title']}\n{r['question_content']}"[:1500], lambda r: r["topic"], "Which topic category does this question belong to?", names, t, cap)
    return _both(f, tr, ev, "yahoo_topics")


@task("newsgroups")
def _ng():
    tr, ev = _split_pair("SetFit/20_newsgroups", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2000], lambda r: idx[r["label_text"]], "Which newsgroup was this post made in?", names, t, cap)
    return _both(f, tr, ev, "newsgroups")


@task("bbc_news", heldout=True)
def _bbc():
    tr, ev = _split_pair("SetFit/bbc-news", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2000], lambda r: idx[r["label_text"]], "What section of the news site does this article belong to?", names, t, cap)
    return _both(f, tr, ev, "bbc_news")


@task("trec", heldout=True)
def _trec():
    tr, ev = _split_pair("SetFit/TREC-QC", None, "train", "test")
    names = sorted(set(tr["label_coarse_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: idx[r["label_coarse_text"]], "What type of answer is this question asking for?", names, t, cap)
    return _both(f, tr, ev, "trec")


@task("student_questions", heldout=True)
def _stuq():
    tr, ev = _split_pair("SetFit/student-question-categories", None, "train", "test")
    names = sorted(set(tr["label_text"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:1500], lambda r: idx[r["label_text"]], "Which subject is this student question about?", names, t, cap)
    return _both(f, tr, ev, "student_questions")


@task("dolly_category", heldout=True)
def _dolly():
    ds = _ld("argilla/databricks-dolly-15k-curated-en", None, "train").shuffle(seed=SEED)
    names = sorted(set(ds["category"])); idx = {n: i for i, n in enumerate(names)}
    f = lambda d, t, cap: _cls(d, lambda r: (r["original-instruction"] + ("\n" + r["original-context"] if r["original-context"] else ""))[:1500], lambda r: idx[r["category"]], "What category of instruction is this?", [_nice(n) for n in names], t, cap)
    return f(ds.select(range(2000, len(ds))), "dolly_category", TRAIN_CAP), f(ds.select(range(2000)), "dolly_category", EVAL_CAP)


# ---------------------------------------------------------------- sentiment / emotion
@task("imdb")
def _imdb():
    tr, ev = _split_pair("stanfordnlp/imdb", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2500], lambda r: r["label"], "What is the sentiment of this movie review?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "imdb")


@task("sst2")
def _sst2():
    tr, ev = _split_pair("stanfordnlp/sst2", None, "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["sentence"], lambda r: r["label"], "What is the sentiment of this sentence?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "sst2")


@task("sst5")
def _sst5():
    tr, ev = _split_pair("SetFit/sst5", None, "train", "test")
    names = ["very negative", "negative", "neutral", "positive", "very positive"]
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the sentiment of this sentence?", names, t, cap)
    return _both(f, tr, ev, "sst5")


@task("yelp")
def _yelp():
    tr, ev = _split_pair("Yelp/yelp_review_full", None, "train", "test")
    names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2000], lambda r: r["label"], "How many stars did this reviewer most likely give?", names, t, cap)
    return _both(f, tr, ev, "yelp")


@task("amazon_stars")
def _amz():
    tr, ev = _split_pair("SetFit/amazon_reviews_multi_en", None, "train", "test")
    names = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2000], lambda r: r["label"], "How many stars did this reviewer most likely give?", names, t, cap)
    return _both(f, tr, ev, "amazon_stars")


@task("emotion")
def _emo():
    tr, ev = _split_pair("dair-ai/emotion", "split", "train", "test")
    names = _names(tr)
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Which emotion does this text express?", names, t, cap)
    return _both(f, tr, ev, "emotion")


@task("go_emotions")
def _goemo():
    tr, ev = _split_pair("google-research-datasets/go_emotions", "simplified", "train", "test")
    names = _names(tr, "labels") if hasattr(tr.features["labels"], "names") else [_nice(n) for n in tr.features["labels"].feature.names]
    tr = tr.filter(lambda r: len(r["labels"]) == 1); ev = ev.filter(lambda r: len(r["labels"]) == 1)
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["labels"][0], "Which emotion best describes this comment?", names, t, cap)
    return _both(f, tr, ev, "go_emotions")


@task("tweet_sentiment")
def _tws():
    tr, ev = _split_pair("cardiffnlp/tweet_eval", "sentiment", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the sentiment of this tweet?", ["negative", "neutral", "positive"], t, cap)
    return _both(f, tr, ev, "tweet_sentiment")


@task("tweet_emotion")
def _twe():
    tr, ev = _split_pair("cardiffnlp/tweet_eval", "emotion", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Which emotion does this tweet express?", ["anger", "joy", "optimism", "sadness"], t, cap)
    return _both(f, tr, ev, "tweet_emotion")


@task("tweet_irony", heldout=True)
def _twi():
    tr, ev = _split_pair("cardiffnlp/tweet_eval", "irony", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this tweet ironic?", ["not ironic", "ironic"], t, cap)
    return _both(f, tr, ev, "tweet_irony")


@task("fin_sentiment", heldout=True)
def _fin():
    tr, ev = _split_pair("zeroshot/twitter-financial-news-sentiment", None, "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "What is the financial sentiment of this headline?", ["bearish", "bullish", "neutral"], t, cap)
    return _both(f, tr, ev, "fin_sentiment")


@task("cr_reviews", heldout=True)
def _cr():
    tr, ev = _split_pair("SetFit/SentEval-CR", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this product-review sentence positive or negative?", ["negative", "positive"], t, cap)
    return _both(f, tr, ev, "cr_reviews")


@task("counterfactual")
def _cf():
    tr, ev = _split_pair("SetFit/amazon_counterfactual_en", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Does this review sentence contain a counterfactual statement?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "counterfactual")


@task("subjectivity")
def _subj():
    tr, ev = _split_pair("SetFit/subj", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this sentence objective or subjective?", ["objective", "subjective"], t, cap)
    return _both(f, tr, ev, "subjectivity")


# ---------------------------------------------------------------- safety / moderation
@task("tweet_offensive")
def _two():
    tr, ev = _split_pair("cardiffnlp/tweet_eval", "offensive", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this tweet offensive?", ["not offensive", "offensive"], t, cap)
    return _both(f, tr, ev, "tweet_offensive")


@task("tweet_hate")
def _twh():
    tr, ev = _split_pair("cardiffnlp/tweet_eval", "hate", "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Does this tweet contain hate speech?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "tweet_hate")


@task("hate_offensive")
def _hso():
    tr, ev = _split_pair("SetFit/hate_speech_offensive", None, "train", "test")
    names = ["hate speech", "offensive language", "neither"]
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "How should this tweet be classified?", names, t, cap)
    return _both(f, tr, ev, "hate_offensive")


@task("civil_comments")
def _civil():
    attrs = ["toxicity", "obscene", "threat", "insult", "identity_attack"]
    def conv(ds, t, cap):
        ds = ds.shuffle(seed=SEED)
        pos = ds.filter(lambda r: r["toxicity"] >= 0.5).select(range(min(cap // 2, 100000)))
        neg = ds.filter(lambda r: r["toxicity"] < 0.5).select(range(min(cap // 2, 100000)))
        out = []
        for r in list(pos) + list(neg):
            qs = [Q(f"Is this comment {a.replace('_', ' ')}?" if a != "toxicity" else "Is this comment toxic?", ["no", "yes"], int(r[a] >= 0.5)) for a in attrs]
            out.append(Example(r["text"][:2000], qs, t))
        random.Random(SEED).shuffle(out)
        return out
    tr = _ld("google/civil_comments", None, "train[:400000]"); ev = _ld("google/civil_comments", None, "test[:60000]")
    return conv(tr, "civil_comments", 12000), conv(ev, "civil_comments", EVAL_CAP)


@task("toxic_chat")
def _tc():
    tr, ev = _split_pair("lmsys/toxic-chat", "toxicchat0124", "train", "test")
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            out.append(Example(r["user_input"][:2000], [
                Q("Is this user request toxic?", ["no", "yes"], int(r["toxicity"])),
                Q("Is this user request a jailbreak attempt?", ["no", "yes"], int(r["jailbreaking"])),
            ], t))
        return out
    return conv(tr, "toxic_chat", TRAIN_CAP), conv(ev, "toxic_chat", EVAL_CAP)


@task("sms_spam")
def _sms():
    ds = _ld("ucirvine/sms_spam", None, "train").shuffle(seed=SEED)
    f = lambda d, t, cap: _cls(d, lambda r: r["sms"], lambda r: r["label"], "Is this SMS message spam?", ["ham (legitimate)", "spam"], t, cap)
    return f(ds.select(range(1000, len(ds))), "sms_spam", TRAIN_CAP), f(ds.select(range(1000)), "sms_spam", EVAL_CAP)


@task("enron_spam")
def _enron():
    tr, ev = _split_pair("SetFit/enron_spam", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"][:2000], lambda r: r["label"], "Is this email spam?", ["ham (legitimate)", "spam"], t, cap)
    return _both(f, tr, ev, "enron_spam")


@task("insincere_questions")
def _insq():
    tr, ev = _split_pair("SetFit/insincere-questions", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Is this forum question insincere (rhetorical, provocative, or not seeking a real answer)?", ["sincere", "insincere"], t, cap)
    return _both(f, tr, ev, "insincere_questions")


@task("ade", heldout=True)
def _ade():
    tr, ev = _split_pair("SetFit/ade_corpus_v2_classification", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["text"], lambda r: r["label"], "Does this sentence report an adverse drug effect?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "ade")


# ---------------------------------------------------------------- NLI / pairs
def _pair(ds, a, b, q, names, t, cap, lab="label", fmt=None):
    fmt = fmt or (lambda r: f"Text 1: {r[a]}\nText 2: {r[b]}")
    return _cls(ds, fmt, lambda r: r[lab], q, names, t, cap)


NLI = ["entailment (text 2 follows from text 1)", "neutral", "contradiction"]


@task("snli")
def _snli():
    tr, ev = _split_pair("stanfordnlp/snli", None, "train", "test")
    f = lambda ds, t, cap: _pair(ds, "premise", "hypothesis", "What is the logical relation between text 1 and text 2?", NLI, t, cap)
    return _both(f, tr, ev, "snli")


@task("mnli")
def _mnli():
    tr, ev = _split_pair("nyu-mll/multi_nli", None, "train", "validation_matched")
    f = lambda ds, t, cap: _pair(ds, "premise", "hypothesis", "What is the logical relation between text 1 and text 2?", NLI, t, cap)
    return _both(f, tr, ev, "mnli")


@task("rte")
def _rte():
    tr, ev = _split_pair("nyu-mll/glue", "rte", "train", "validation")
    f = lambda ds, t, cap: _pair(ds, "sentence1", "sentence2", "Does text 1 entail text 2?", ["yes, entailment", "no, not entailment"], t, cap)
    return _both(f, tr, ev, "rte")


@task("qnli")
def _qnli():
    tr, ev = _split_pair("nyu-mll/glue", "qnli", "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Question: {r['question']}\nSentence: {r['sentence']}", lambda r: r["label"], "Does the sentence contain the answer to the question?", ["yes", "no"], t, cap)
    return _both(f, tr, ev, "qnli")


@task("qqp")
def _qqp():
    tr, ev = _split_pair("nyu-mll/glue", "qqp", "train", "validation")
    f = lambda ds, t, cap: _pair(ds, "question1", "question2", "Are these two questions asking the same thing?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "qqp")


@task("mrpc")
def _mrpc():
    tr, ev = _split_pair("nyu-mll/glue", "mrpc", "train", "validation")
    f = lambda ds, t, cap: _pair(ds, "sentence1", "sentence2", "Are these two sentences paraphrases of each other?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "mrpc")


@task("paws", heldout=True)
def _paws():
    tr, ev = _split_pair("google-research-datasets/paws", "labeled_final", "train", "test")
    f = lambda ds, t, cap: _pair(ds, "sentence1", "sentence2", "Do these two sentences have the same meaning?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "paws")


@task("cola")
def _cola():
    tr, ev = _split_pair("nyu-mll/glue", "cola", "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["sentence"], lambda r: r["label"], "Is this sentence grammatically acceptable?", ["unacceptable", "acceptable"], t, cap)
    return _both(f, tr, ev, "cola")


# ---------------------------------------------------------------- reading / QA / MCQ
@task("boolq")
def _boolq():
    tr, ev = _split_pair("google/boolq", None, "train", "validation")
    f = lambda ds, t, cap: _cls(ds, lambda r: f"Passage: {r['passage'][:2000]}\nQuestion: {r['question']}", lambda r: int(r["answer"]), "Based on the passage, what is the answer?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "boolq")


@task("strategyqa", heldout=True)
def _sqa():
    tr, ev = _split_pair("ChilleD/StrategyQA", None, "train", "test")
    f = lambda ds, t, cap: _cls(ds, lambda r: r["question"], lambda r: int(r["answer"]), "What is the answer to this question?", ["no", "yes"], t, cap)
    return _both(f, tr, ev, "strategyqa")


@task("pubmedqa", heldout=True)
def _pmq():
    ds = _ld("qiaojin/PubMedQA", "pqa_labeled", "train").shuffle(seed=SEED)
    names = ["yes", "no", "maybe"]
    f = lambda d, t, cap: _cls(d, lambda r: ("Abstract: " + " ".join(r["context"]["contexts"]))[:2500] + f"\nQuestion: {r['question']}", lambda r: names.index(r["final_decision"]), "Based on the abstract, what is the answer?", names, t, cap)
    return f(ds.select(range(500, len(ds))), "pubmedqa", TRAIN_CAP), f(ds.select(range(500)), "pubmedqa", EVAL_CAP)


def _mcq(ds, ctx_fn, opts_fn, gold_fn, q, t, cap):
    out = []
    for r in _sub(ds, cap):
        opts = [o.strip() for o in opts_fn(r)]
        g = gold_fn(r)
        if g is None or g < 0 or g >= len(opts) or len(opts) < 2:
            continue
        ctx = ctx_fn(r)
        if not ctx.strip():
            continue
        out.append(Example(ctx.strip(), [Q(q, opts, g)], t))
    return out


def _key_idx(r, key="answerKey"):
    labels = r["choices"]["label"]; k = r[key]
    return labels.index(k) if k in labels else None


@task("arc")
def _arc():
    tr = _ld("allenai/ai2_arc", "ARC-Challenge", "train"); tr2 = _ld("allenai/ai2_arc", "ARC-Easy", "train")
    ev = _ld("allenai/ai2_arc", "ARC-Challenge", "test")
    from datasets import concatenate_datasets
    tr = concatenate_datasets([tr, tr2])
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "arc")


@task("commonsense_qa")
def _csqa():
    tr, ev = _split_pair("tau/commonsense_qa", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "commonsense_qa")


@task("qasc")
def _qasc():
    tr, ev = _split_pair("allenai/qasc", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "qasc")


@task("openbookqa")
def _obqa():
    tr, ev = _split_pair("allenai/openbookqa", "main", "train", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question_stem"], lambda r: r["choices"]["text"], _key_idx, "Which option correctly completes or answers the question?", t, cap)
    return _both(f, tr, ev, "openbookqa")


@task("sciq", heldout=True)
def _sciq():
    tr, ev = _split_pair("allenai/sciq", None, "train", "validation")
    def opts(r):
        o = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        rng = random.Random(hash(r["question"]) & 0xffff); perm = list(range(4)); rng.shuffle(perm)
        r["_perm"] = perm
        return [o[i] for i in perm]
    f = lambda ds, t, cap: _mcq(ds, lambda r: (f"Support: {r['support'][:1500]}\n" if r["support"] else "") + f"Question: {r['question']}", opts, lambda r: r["_perm"].index(0), "Which option correctly answers the question?", t, cap)
    return _both(f, tr, ev, "sciq")


@task("hellaswag")
def _hs():
    tr, ev = _split_pair("Rowan/hellaswag", None, "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["ctx"], lambda r: r["endings"], lambda r: int(r["label"]) if r["label"] != "" else None, "Which ending is the most plausible continuation?", t, cap)
    return _both(f, tr, ev, "hellaswag")


@task("piqa")
def _piqa():
    tr, ev = _split_pair("ybisk/piqa", None, "train", "validation", rev="refs/convert/parquet")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["goal"], lambda r: [r["sol1"], r["sol2"]], lambda r: r["label"], "Which solution is the correct way to achieve the goal?", t, cap)
    return _both(f, tr, ev, "piqa")


@task("social_iqa", heldout=True)
def _siqa():
    tr, ev = _split_pair("allenai/social_i_qa", None, "train", "validation", rev="refs/convert/parquet")
    f = lambda ds, t, cap: _mcq(ds, lambda r: f"{r['context']}\nQuestion: {r['question']}", lambda r: [r["answerA"], r["answerB"], r["answerC"]], lambda r: int(r["label"]) - 1, "Which answer is most appropriate?", t, cap)
    return _both(f, tr, ev, "social_iqa")


@task("winogrande")
def _wg():
    tr, ev = _split_pair("allenai/winogrande", "winogrande_xl", "train", "validation")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["sentence"], lambda r: [r["option1"], r["option2"]], lambda r: int(r["answer"]) - 1 if r["answer"] in ("1", "2") else None, "Which option correctly fills the blank (_)?", t, cap)
    return _both(f, tr, ev, "winogrande")


@task("race")
def _race():
    tr, ev = _split_pair("ehovy/race", "all", "train", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: f"Article: {r['article'][:2500]}\nQuestion: {r['question']}", lambda r: r["options"], lambda r: "ABCD".index(r["answer"]), "Which option correctly answers the question about the article?", t, cap)
    return _both(f, tr, ev, "race")


@task("mmlu")
def _mmlu():
    tr = _ld("cais/mmlu", "all", "auxiliary_train"); ev = _ld("cais/mmlu", "all", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"][:2000], lambda r: r["choices"], lambda r: int(r["answer"]), "Which option is correct?", t, cap)
    return _both(f, tr, ev, "mmlu")


@task("medqa")
def _medqa():
    tr, ev = _split_pair("GBaker/MedQA-USMLE-4-options", None, "train", "test")
    f = lambda ds, t, cap: _mcq(ds, lambda r: r["question"][:2500], lambda r: [r["options"][k] for k in "ABCD"], lambda r: "ABCD".index(r["answer_idx"]), "Which option is the correct answer?", t, cap)
    return _both(f, tr, ev, "medqa")


@task("truthfulqa", heldout=True)
def _tqa():
    ds = _ld("truthfulqa/truthful_qa", "multiple_choice", "validation")
    def opts(r):
        return r["mc1_targets"]["choices"]
    f = lambda d, t, cap: _mcq(d, lambda r: r["question"], opts, lambda r: r["mc1_targets"]["labels"].index(1), "Which answer is true?", t, cap)
    return [], f(ds, "truthfulqa", EVAL_CAP)


@task("fin_phrasebank", heldout=True)
def _fpb():
    ds = _ld("AdaptLLM/finance-tasks", "FPB", "test")
    f = lambda d, t, cap: _mcq(d, lambda r: r["input"][:2000], lambda r: r["options"], lambda r: int(r["gold_index"]), "What is the sentiment of this financial news sentence?", t, cap)
    return [], f(ds, "fin_phrasebank", EVAL_CAP)


# ---------------------------------------------------------------- multi-attribute
@task("bias_in_bios")
def _bib():
    tr, ev = _split_pair("LabHC/bias_in_bios", None, "train", "test")
    profs = ["accountant","architect","attorney","chiropractor","comedian","composer","dentist","dietitian","dj","filmmaker","interior designer","journalist","model","nurse","painter","paralegal","pastor","personal trainer","photographer","physician","poet","professor","psychologist","rapper","software engineer","surgeon","teacher","yoga teacher"]
    def conv(ds, t, cap):
        out = []
        for r in _sub(ds, cap):
            out.append(Example(r["hard_text"][:2000], [
                Q("What is this person's profession?", profs, int(r["profession"])),
                Q("Which pronoun set is used for this person?", ["he/him", "she/her"], int(r["gender"])),
            ], t))
        return out
    return conv(tr, "bias_in_bios", TRAIN_CAP), conv(ev, "bias_in_bios", EVAL_CAP)


# ---------------------------------------------------------------- driver
def load_cache(path="data/tasks.pkl"):
    import pickle, sys, __main__
    __main__.Example = Example; __main__.Q = Q      # tolerate caches pickled from `python -m decider.data`
    sys.modules.setdefault("s1", sys.modules["decider"]); sys.modules.setdefault("s1.data", sys.modules[__name__])   # caches from the old package name
    return pickle.load(open(path, "rb"))


def load_task(name):
    return TASKS[name]["loader"]()


def load_all(names=None, verbose=True):
    names = names or list(TASKS)
    train, evals = [], {}
    for n in names:
        try:
            tr, ev = load_task(n)
        except Exception as e:
            print(f"[data] FAILED {n}: {e}")
            continue
        held = TASKS[n]["heldout"]
        if not held:
            train.extend(tr)
        evals[n] = ev
        if verbose:
            ex = (tr or ev or [None])[0]
            print(f"[data] {n:22s} train={len(tr):6d} eval={len(ev):5d} heldout={held}" + (f" nq={len(ex.qs)} nopt={len(ex.qs[0].options)}" if ex else " (built later by decider.data.mixture)"))
    return train, evals


if __name__ == "__main__":
    import argparse, pickle, os
    from decider import data as D                        # registers every task module; pickles under the package name, not __main__
    ap = argparse.ArgumentParser(description="Download and convert the registered tasks into one cache."); ap.add_argument("tasks", nargs="*"); ap.add_argument("--out", default="data/tasks.pkl")
    a = ap.parse_args(); train, evals = D.load_all(a.tasks or None)
    print("total train", len(train), "eval tasks", len(evals), "eval examples", sum(len(v) for v in evals.values()))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump((train, evals), f)
