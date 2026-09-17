"""Jev-shaped requests on top of the decider prompt format (same wire format as TypeSafe's POST /v1/systemone).

    state      str | dict | list            JSON state is serialised compactly; questions may name a part by path (`ticket.messages[0].text`)
    questions  {id: {"type": "choice", "instructions": ..., "criteria": {name: description | {...} | [...] | None}}      up to 255 options
                     {"type": "score",  "instructions": ..., "criteria": [level 0 description, level 1 description, ...]}  2..10 levels
                     {"type": "noul",   "instructions": ..., "criteria": {"true": ..., "false": ...} (optional)}}
    ids are never shown to the model.  `instructions` and every description may be a string or any JSON value.
"""
import json, math

MAX_CHOICE, MAX_LEVELS = 255, 10


def _txt(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


ANNOTATE_MIN = 8


def annotate_indices(x, min_len=ANNOTATE_MIN):
    """Write each element's position into long arrays ({"_index": i, ...}).  A path such as `records[47].text` otherwise makes
    the model count 47 elements; with the index written down it is a lookup (json_k64 probe: 0.49 -> 0.57 accuracy)."""
    if isinstance(x, list):
        if len(x) >= min_len:
            return [({"_index": i, **annotate_indices(v, min_len)} if isinstance(v, dict) else {"_index": i, "value": annotate_indices(v, min_len)}) for i, v in enumerate(x)]
        return [annotate_indices(v, min_len) for v in x]
    if isinstance(x, dict):
        return {k: annotate_indices(v, min_len) for k, v in x.items()}
    return x


def render_state(state, index_arrays=True):
    if isinstance(state, str):
        return state
    return json.dumps(annotate_indices(state) if index_arrays else state, ensure_ascii=False)


def render_question(spec):
    """-> dict(question=str, options=[str], type=..., names=[...])  (names: what the answer reports for each option)"""
    t = spec.get("type", "choice"); ins = _txt(spec.get("instructions", spec.get("question", ""))); crit = spec.get("criteria", spec.get("options"))
    if not ins:
        raise ValueError("question without instructions")
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or not 2 <= len(crit) <= MAX_CHOICE:
            raise ValueError(f"choice criteria: a map of 2..{MAX_CHOICE} options")
        names = list(crit); opts = [n if crit[n] in (None, "") else f"{n}: {_txt(crit[n])}" for n in names]
    elif t == "score":
        if isinstance(crit, dict):                                   # legend form {"0": "...", "1": "..."}
            crit = [crit[k] for k in sorted(crit, key=float)]
        if not isinstance(crit, (list, tuple)) or not 2 <= len(crit) <= MAX_LEVELS:
            raise ValueError(f"score criteria: an ordered list of 2..{MAX_LEVELS} level descriptions")
        names = list(range(len(crit))); opts = [f"{i}: {_txt(c)}" for i, c in enumerate(crit)]
    elif t in ("noul", "bool"):
        names = [False, True]; c = crit or {}
        f, tr = c.get("false", c.get(False)), c.get("true", c.get(True))
        opts = ["no" if f in (None, "") else f"no: {_txt(f)}", "yes" if tr in (None, "") else f"yes: {_txt(tr)}"]
    else:
        raise ValueError(f"unknown question type {t!r}")
    return dict(question=ins, options=opts, type="noul" if t == "bool" else t, names=names, legend=[_txt(c) for c in crit] if t == "score" else None,
                isolated=bool(spec.get("isolated", True)))


# ---- isolated levels: every Score level is judged in its own row, without its number or its neighbours
ISOLATED = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?"
_NUM = None


def strip_level_number(text):
    """"2: somewhat" -> "somewhat" (dataset legends carry the number; an isolated level must not)."""
    import re
    return re.sub(r"^\s*-?\d+\s*:\s*", "", text)


def isolated_rows(question, levels):
    """-> one yes/no question per level: [(question text, ["no", "yes"])]."""
    return [(ISOLATED.format(q=question, level=strip_level_number(l)), ["no", "yes"]) for l in levels]


def combine_isolated(p_yes):
    """Per-level P(fits), each computed without reference to any other level -> a distribution over levels.
    Also returns the unnormalised mass: near 1 when exactly one level fits, low when none does, high when several do."""
    tot = sum(p_yes) or 1e-9
    return [x / tot for x in p_yes], tot


def plan_rows(rqs, isolated=True):
    """One scoring row per question; a Score question with isolated levels becomes one yes/no row per level.
    -> (rows [{"question", "options"}], index [(id, "iso" | "list", first row, n rows)])"""
    rows, index = [], []
    for k, r in rqs.items():
        if isolated and r["type"] == "score" and r.get("isolated", True):
            rws = isolated_rows(r["question"], r["legend"]); index.append((k, "iso", len(rows), len(rws))); rows += [dict(question=t, options=o) for t, o in rws]
        else:
            index.append((k, "list", len(rows), 1)); rows.append(dict(question=r["question"], options=r["options"]))
    return rows, index


def assemble(rqs, index, probs):
    """probs: one probability list per row (plan_rows order) -> {id: answer}."""
    out = {}
    for k, kind, s, n in index:
        if kind == "iso":
            fit = [float(probs[s + j][1]) for j in range(n)]; p, mass = combine_isolated(fit); a = format_answer(rqs[k], p)
            a["level_fit"] = {str(j): round(x, 4) for j, x in enumerate(fit)}; a["fit_mass"] = round(mass, 4); out[k] = a
        else:
            out[k] = format_answer(rqs[k], probs[s])
    return out


def certainty(p):
    """1 - normalised entropy: 1 when all mass is on one option, 0 when the distribution is flat."""
    h = -sum(x * math.log(x) for x in p if x > 0)
    return max(0.0, 1.0 - h / math.log(len(p))) if len(p) > 1 else 1.0


def format_answer(rq, p, nd=4):
    """rq: render_question output; p: probabilities in option order."""
    p = [float(x) for x in p[:len(rq["options"])]]; s = sum(p) or 1.0; p = [x / s for x in p]
    j = max(range(len(p)), key=p.__getitem__)
    if rq["type"] == "noul":
        return {"type": "noul", "noul": round(p[1], nd)}
    if rq["type"] == "choice":
        return {"type": "choice", "choice": rq["names"][j], "confidence": round(p[j], nd), "certainty": round(certainty(p), nd),
                "probabilities": {n: round(x, nd) for n, x in zip(rq["names"], p)}}
    return {"type": "score", "score": round(sum(i * x for i, x in enumerate(p)), 2), "confidence": round(p[j], nd), "certainty": round(certainty(p), nd),
            "legend": {str(i): d for i, d in enumerate(rq["legend"])}, "probabilities": {str(i): round(x, nd) for i, x in enumerate(p)}}


def unique_tokens(items):
    """Input tokens of a request whose rows share a prefix (the state): the prefix counts once."""
    ids = [it["ids"] for it in items]
    if len(ids) < 2:
        return sum(len(x) for x in ids)
    lcp = 0; short = min(len(x) for x in ids)
    while lcp < short and all(x[lcp] == ids[0][lcp] for x in ids): lcp += 1
    return lcp + sum(len(x) - lcp for x in ids)
