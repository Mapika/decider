"""v6 evaluation-only tasks: large label sets offered in full (up to 219 options) and long inputs.
All held out: no example of these datasets is trained on (dbpedia_l2/l3 share their source, Wikipedia
abstracts, with the in-task 14-class dbpedia, but the label sets are different and much finer)."""
import ast, re
from .data import task, Example, Q, _ld, _sub, _cls, EVAL_CAP

NEW_TASKS = ["hwu64", "trec_fine", "dbpedia_l2", "dbpedia_l3", "quality_full"]


def _split_camel(s):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower()


def _str_labels(ds, col, text_col, question, name, nice=lambda s: s, clip=2000):
    names = sorted(set(ds[col])); idx = {n: i for i, n in enumerate(names)}
    return [], _cls(ds, lambda r: r[text_col][:clip], lambda r: idx[r[col]], question, [nice(n) for n in names], name, EVAL_CAP)


@task("hwu64", heldout=True)
def _hwu():
    return _str_labels(_ld("FastFit/hwu_64", split="test"), "label", "text", "What is the intent of this request to a home assistant?", "hwu64")


@task("dbpedia_l2", heldout=True)
def _db2():
    return _str_labels(_ld("DeveloperOats/DBPedia_Classes", split="test"), "l2", "text", "Which class of entity does this encyclopedia text describe?", "dbpedia_l2", _split_camel, 1500)


@task("dbpedia_l3", heldout=True)
def _db3():
    return _str_labels(_ld("DeveloperOats/DBPedia_Classes", split="test"), "l3", "text", "Which class of entity does this encyclopedia text describe?", "dbpedia_l3", _split_camel, 1500)


TREC_FINE = {"ABBR:abb": "an abbreviation", "ABBR:exp": "the expansion of an abbreviation", "ENTY:animal": "an animal", "ENTY:body": "an organ or part of the body",
             "ENTY:color": "a color", "ENTY:cremat": "a creative work (book, film, song, invention)", "ENTY:currency": "a currency name", "ENTY:dismed": "a disease or medicine",
             "ENTY:event": "an event", "ENTY:food": "a food or drink", "ENTY:instru": "a musical instrument", "ENTY:lang": "a language", "ENTY:letter": "a letter of the alphabet",
             "ENTY:other": "some other kind of entity", "ENTY:plant": "a plant", "ENTY:product": "a product", "ENTY:religion": "a religion", "ENTY:sport": "a sport",
             "ENTY:substance": "an element or substance", "ENTY:symbol": "a symbol or sign", "ENTY:techmeth": "a technique or method", "ENTY:termeq": "an equivalent term",
             "ENTY:veh": "a vehicle", "ENTY:word": "a word with a special property", "DESC:def": "the definition of something", "DESC:desc": "a description of something",
             "DESC:manner": "the manner of an action (how something is done)", "DESC:reason": "a reason (why)", "HUM:gr": "a group or organization", "HUM:ind": "an individual person",
             "HUM:title": "the title or role of a person", "HUM:desc": "a description of a person", "LOC:city": "a city", "LOC:country": "a country", "LOC:mount": "a mountain",
             "LOC:other": "some other location", "LOC:state": "a state or province", "NUM:code": "a postcode or other code", "NUM:count": "a count of something", "NUM:date": "a date",
             "NUM:dist": "a distance or linear measure", "NUM:money": "a price or amount of money", "NUM:ord": "a rank or order", "NUM:other": "some other number",
             "NUM:period": "a duration or period of time", "NUM:perc": "a percentage or fraction", "NUM:speed": "a speed", "NUM:temp": "a temperature",
             "NUM:volsize": "a size, area or volume", "NUM:weight": "a weight"}


@task("trec_fine", heldout=True)
def _trecf():
    ds = _ld("CogComp/trec", split="test"); raw = ds.features["fine_label"].names
    names = [TREC_FINE[n] for n in raw]
    return [], _cls(ds, lambda r: r["text"], lambda r: int(r["fine_label"]), "What kind of answer is this question asking for?", names, "trec_fine", EVAL_CAP)


@task("quality_full", heldout=True)
def _qfull():
    """QuALITY with the whole article (about 5-8k tokens); the `quality` task clips the article to 5000 characters."""
    ds = _sub(_ld("emozilla/quality", split="validation"), 600); out = []
    base = min(int(r["answer"]) for r in ds)
    for r in ds:
        opts = r["options"] if isinstance(r["options"], list) else ast.literal_eval(r["options"])
        g = int(r["answer"]) - base
        if 0 <= g < len(opts):
            out.append(Example(f"Article: {r['article']}\nQuestion: {r['question']}", [Q("Which option correctly answers the question about the article?", [o.strip() for o in opts], g)], "quality_full"))
    return [], out
