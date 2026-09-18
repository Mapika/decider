"""Rule-conditioned decisions: the answer follows from a rule applied to the state, never from matching the state to an option.

Every example is generated three times over the same options:
  the base            a record and a rule; the gold outcome is computed, not written
  its state twin      one attribute changed just enough to flip the outcome (the number crosses the threshold, the flag flips,
                      the field is emptied or filled); everything else, including the rule and the options, is identical
  its rule twin       the same record, the rule inverted (comparator or outcomes swapped); the answer flips again
A model that matches state text to option text gets nothing from these: the twins share almost all their text and disagree on
the label.  Decoys make matching actively wrong: a fraction of wrong options are named after a value that appears in the state.

Families (question forms): threshold, presence, boolean, membership, contains, count, compare (two fields), compound (and / or /
unless), precedence (ordered rules, first match wins), pointer (which option holds the value of field X; decoys hold values of other
fields), select (which of several records satisfies the rule), form (a UI element with document entities: fill / check / click / skip,
rendered the way accessibility snapshots are).  Rules are asked as choice, noul ("does this record satisfy ...") or score
(thresholds as levels).  Ten domains; the last three and two families are held out for the probes.

    python -m decider.data.rules --n 60000 --out data/rules.pkl          (writes train examples and probe sets)
"""
import argparse, json, pickle, random, string
from decider.data.core import Example, Q
from decider.systemone import render_state

# ---------------------------------------------------------------- domains: field name -> (kind, generator)
FIRST = ["Lucas", "Priya", "Mateo", "Aisha", "Noah", "Hana", "Ibrahim", "Sofia", "Elena", "Kenji", "Amara", "Diego", "Mei", "Tomás", "Zara", "Omar", "Ines", "Ravi"]
LAST = ["Patel", "Novak", "Okafor", "Silva", "Chen", "Haddad", "Fischer", "Moreau", "Kowalski", "Tanaka", "Reyes", "Byrne", "Nakamura", "Costa", "Ali", "Jensen"]
CITIES = ["Berlin", "Austin", "Lisbon", "Nairobi", "Osaka", "Toronto", "Madrid", "Pune", "Oslo", "Bogotá", "Cairo", "Seoul", "Denver", "Lyon", "Perth"]
COUNTRIES = ["Germany", "United States", "Portugal", "Kenya", "Japan", "Canada", "Spain", "India", "Norway", "Colombia", "Egypt", "Korea", "France", "Brazil"]


def person(r): return f"{r.choice(FIRST)} {r.choice(LAST)}"
def email(r): return f"{r.choice(FIRST).lower()}.{r.choice(LAST).lower()}@example.{r.choice(['com', 'org', 'net', 'invalid'])}"
def words(r, pool, lo=0, hi=3): return r.sample(pool, r.randint(lo, min(hi, len(pool))))
def code(r, pre, n=6): return pre + "-" + "".join(r.choices(string.ascii_uppercase + string.digits, k=n))


# kind: num (int range), bool, enum (list), str (generator, may be empty), list (pool)
DOMAINS = {
    "account":   dict(name=("str", person), email=("str", email), age=("num", (13, 80)), status=("enum", ["active", "suspended", "closed", "pending"]),
                      balance=("num", (0, 5000)), verified=("bool", None), plan=("enum", ["free", "starter", "pro", "enterprise"]), country=("enum", COUNTRIES),
                      tags=("list", ["vip", "trial", "late-payer", "partner", "beta", "flagged"]), referrer=("str", person)),
    "order":     dict(id=("str", lambda r: code(r, "ORD")), total=("num", (5, 2000)), items=("list", ["lamp", "kettle", "charger", "book", "monitor", "chair", "cable", "headset"]),
                      status=("enum", ["placed", "paid", "shipped", "delivered", "returned"]), gift=("bool", None), destination=("enum", COUNTRIES), days_open=("num", (0, 60)),
                      priority=("enum", ["low", "normal", "high"]), coupon=("str", lambda r: code(r, "SAVE", 4)), customer=("str", person)),
    "ticket":    dict(subject=("str", lambda r: r.choice(["Login fails", "Refund request", "Broken link", "Invoice wrong", "Feature idea", "App crashes", "Password reset"])),
                      priority=("enum", ["low", "medium", "high", "urgent"]), category=("enum", ["billing", "technical", "account", "sales"]), assignee=("str", person),
                      age_hours=("num", (0, 200)), tier=("enum", ["basic", "silver", "gold"]), resolved=("bool", None), replies=("num", (0, 12)), reporter=("str", person)),
    "device":    dict(model=("enum", ["X200", "Nova 3", "Atlas", "Kite Mini", "Orbit"]), battery=("num", (0, 100)), firmware=("num", (1, 9)), online=("bool", None),
                      owner=("str", person), location=("enum", CITIES), alerts=("list", ["overheat", "low-storage", "update-due", "tamper", "offline-24h"]), uptime_days=("num", (0, 400))),
    "applicant": dict(name=("str", person), age=("num", (18, 65)), years_experience=("num", (0, 30)), degree=("enum", ["none", "bachelor", "master", "phd"]),
                      skills=("list", ["python", "sql", "excel", "react", "go", "figma", "rust"]), salary=("num", (30, 250)), remote=("bool", None), visa_required=("bool", None),
                      city=("enum", CITIES), referred_by=("str", person)),
    "file":      dict(path=("str", lambda r: f"/{r.choice(['home', 'srv', 'var', 'opt'])}/{r.choice(['data', 'logs', 'build', 'notes'])}/{code(r, 'f', 4).lower()}.{r.choice(['csv', 'log', 'png', 'py', 'pdf'])}"),
                      size_kb=("num", (1, 90000)), modified_days_ago=("num", (0, 900)), owner=("str", person), readonly=("bool", None), shared=("bool", None),
                      tags=("list", ["archive", "temp", "secret", "backup", "draft"]), versions=("num", (1, 40))),
    "email":     dict(sender=("str", email), subject=("str", lambda r: r.choice(["Invoice attached", "Meeting moved", "Your order", "Weekly report", "Password", "Re: contract", "Lunch?"])),
                      attachment=("bool", None), spam_score=("num", (0, 100)), unread=("bool", None), folder=("enum", ["inbox", "archive", "spam", "drafts"]),
                      domain=("enum", ["example.com", "mail.invalid", "corp.example", "news.example"]), recipients=("num", (1, 40)), labels=("list", ["work", "family", "finance", "travel"])),
    # held out of training (probes):
    "shipment":  dict(carrier=("enum", ["Posta", "SwiftShip", "Globex", "Kuriar"]), weight_kg=("num", (1, 300)), fragile=("bool", None), express=("bool", None),
                      distance_km=("num", (5, 9000)), insured=("bool", None), origin=("enum", CITIES), contents=("list", ["glass", "books", "electronics", "clothes", "food"]), stops=("num", (0, 6)), driver=("str", person)),
    "event":     dict(title=("str", lambda r: r.choice(["Board meeting", "Team offsite", "Launch party", "Retro", "Customer visit", "Training day"])), attendees=("num", (1, 400)),
                      days_away=("num", (0, 120)), location=("enum", CITIES), cost=("num", (0, 20000)), recurring=("bool", None), confirmed=("bool", None), organizer=("str", person),
                      rooms=("list", ["Aurora", "Basalt", "Cedar", "Delta"])),
    "sensor":    dict(id=("str", lambda r: code(r, "S", 4)), temperature=("num", (-20, 90)), humidity=("num", (0, 100)), battery=("num", (0, 100)), calibrated=("bool", None),
                      zone=("enum", ["north", "south", "east", "west", "roof"]), faults=("list", ["drift", "noise", "dropout", "clock"]), readings=("num", (0, 5000)), technician=("str", person)),
}
HELD_DOMAINS = ["shipment", "event", "sensor"]
FAMILIES = ["threshold", "presence", "boolean", "membership", "contains", "count", "compare", "compound", "precedence", "pointer", "select", "form"]
HELD_FAMILIES = ["compare", "count"]                       # never trained in any domain; the probe asks whether rule reading transfers across forms
OUTCOMES = [("approve", "reject"), ("escalate", "handle locally"), ("archive", "keep"), ("notify", "stay silent"), ("expedite", "standard"), ("flag for review", "clear"),
            ("assign to Dana", "assign to Rui"), ("tier A", "tier B"), ("green", "red"), ("include", "exclude"), ("yes", "no"), ("retry", "give up"), ("north desk", "south desk")]


def sample_record(dom, r, empty_p=0.15):
    rec = {}
    for k, (kind, g) in DOMAINS[dom].items():
        if kind == "num": rec[k] = r.randint(*g)
        elif kind == "bool": rec[k] = r.random() < 0.5
        elif kind == "enum": rec[k] = r.choice(g)
        elif kind == "str": rec[k] = "" if r.random() < empty_p else g(r)
        elif kind == "list": rec[k] = words(r, g)
    return rec


def fields(dom, kind): return [k for k, (kd, _) in DOMAINS[dom].items() if kd == kind]


# ---------------------------------------------------------------- conditions: (text, test(rec), flip(rec) -> modified copy)
def cond_threshold(dom, r, rec):
    k = r.choice(fields(dom, "num")); lo, hi = DOMAINS[dom][k][1]; t = r.randint(lo + 1, hi - 1)
    op = r.choice(["at least", "more than", "below", "at most"])
    test = {"at least": lambda v: v >= t, "more than": lambda v: v > t, "below": lambda v: v < t, "at most": lambda v: v <= t}[op]
    txt = r.choice([f"`{k}` is {op} {t}", f"the {k.replace('_', ' ')} is {op} {t}", f"{k} {'>=' if op == 'at least' else '>' if op == 'more than' else '<' if op == 'below' else '<='} {t}"])
    def flip(x):
        y = dict(x); y[k] = (t if op == "at least" else t + 1 if op == "more than" else t - 1 if op == "below" else t) if not test(x[k]) else \
                         (t - 1 if op == "at least" else t if op == "more than" else t if op == "below" else t + 1)
        return y
    return txt, lambda x: test(x[k]), flip, k


def cond_presence(dom, r, rec):
    k = r.choice(fields(dom, "str")); neg = r.random() < 0.5
    txt = r.choice([f"`{k}` is empty", f"there is no {k.replace('_', ' ')}", f"the {k.replace('_', ' ')} field is blank"]) if neg else \
          r.choice([f"`{k}` is filled in", f"a {k.replace('_', ' ')} is given", f"the {k.replace('_', ' ')} field is not empty"])
    test = (lambda x: x[k] == "") if neg else (lambda x: x[k] != "")
    def flip(x):
        y = dict(x); y[k] = "" if x[k] else DOMAINS[dom][k][1](r); return y
    return txt, test, flip, k


def cond_boolean(dom, r, rec):
    k = r.choice(fields(dom, "bool")); want = r.random() < 0.5
    txt = r.choice([f"`{k}` is {str(want).lower()}", f"the record is {'' if want else 'not '}{k.replace('_', ' ')}", f"{k.replace('_', ' ')} = {'yes' if want else 'no'}"])
    def flip(x):
        y = dict(x); y[k] = not x[k]; return y
    return txt, lambda x: x[k] == want, flip, k


def cond_membership(dom, r, rec):
    k = r.choice(fields(dom, "enum")); pool = DOMAINS[dom][k][1]; vals = r.sample(pool, r.choice([1, 1, 2])); neg = r.random() < 0.3
    txt = (f"`{k}` is {'not ' if neg else ''}" + (f"{vals[0]!r}" if len(vals) == 1 else f"one of {vals!r}")) if r.random() < 0.6 else \
          f"the {k.replace('_', ' ')} is {'anything but ' if neg else ''}{' or '.join(map(str, vals))}"
    test = (lambda x: (x[k] in vals) != neg)
    def flip(x):
        y = dict(x); y[k] = r.choice([v for v in pool if v not in vals]) if x[k] in vals else r.choice(vals); return y
    return txt, test, flip, k


def cond_contains(dom, r, rec):
    k = r.choice(fields(dom, "list")); pool = DOMAINS[dom][k][1]; v = r.choice(pool); neg = r.random() < 0.3
    txt = r.choice([f"`{k}` {'does not contain' if neg else 'contains'} {v!r}", f"{v!r} is {'not ' if neg else ''}among the {k}", f"the {k} {'exclude' if neg else 'include'} {v}"])
    test = lambda x: (v in x[k]) != neg
    def flip(x):
        y = dict(x); y[k] = [z for z in x[k] if z != v] if v in x[k] else x[k] + [v]; return y
    return txt, test, flip, k


def cond_count(dom, r, rec):
    k = r.choice(fields(dom, "list")); t = r.randint(1, 3); op = r.choice(["at least", "fewer than"])
    txt = r.choice([f"`{k}` has {op} {t} {'entry' if t == 1 else 'entries'}", f"there are {op} {t} {k}"])
    test = (lambda x: len(x[k]) >= t) if op == "at least" else (lambda x: len(x[k]) < t)
    pool = DOMAINS[dom][k][1]
    def flip(x):
        y = dict(x); n = t if (op == "at least") != test(x) else t - 1
        y[k] = (x[k] + [z for z in pool if z not in x[k]])[:n] if n >= 0 else []; return y
    return txt, test, flip, k


def cond_compare(dom, r, rec):
    ks = fields(dom, "num")
    if len(ks) < 2: return cond_threshold(dom, r, rec)
    a, b = r.sample(ks, 2); op = r.choice(["greater than", "less than"])
    txt = r.choice([f"`{a}` is {op} `{b}`", f"the {a.replace('_', ' ')} exceeds the {b.replace('_', ' ')}" if op == "greater than" else f"the {a.replace('_', ' ')} is smaller than the {b.replace('_', ' ')}"])
    test = (lambda x: x[a] > x[b]) if op == "greater than" else (lambda x: x[a] < x[b])
    def flip(x):
        y = dict(x); y[a] = x[b] + (1 if not test(x) and op == "greater than" or test(x) and op == "less than" else -1); return y
    return txt, test, flip, a


COND = dict(threshold=cond_threshold, presence=cond_presence, boolean=cond_boolean, membership=cond_membership, contains=cond_contains, count=cond_count, compare=cond_compare)


def cond_compound(dom, r, rec):
    fa, fb = r.sample([f for f in COND if f not in HELD_FAMILIES or dom in HELD_DOMAINS], 2)
    (ta, a, fla, ka), (tb, b, flb, kb) = COND[fa](dom, r, rec), COND[fb](dom, r, rec)
    if ka == kb: return cond_compound(dom, r, rec)
    kind = r.choice(["and", "or", "unless"])
    txt = {"and": f"{ta} and {tb}", "or": f"{ta} or {tb}", "unless": f"{ta}, unless {tb}"}[kind]
    test = {"and": lambda x: a(x) and b(x), "or": lambda x: a(x) or b(x), "unless": lambda x: a(x) and not b(x)}[kind]
    def flip(x):                                   # flip one clause so that the whole condition flips; try both
        for f in (fla, flb):
            y = f(x)
            if test(y) != test(x): return y
        return flb(fla(x))
    return txt, test, flip, ka


# ---------------------------------------------------------------- rendering
def render_record(rec, r):
    mode = r.random()
    if mode < 0.55: return render_state(rec)
    if mode < 0.8: return "\n".join(f"{k}: {json.dumps(v) if isinstance(v, (bool, list)) else v}" for k, v in rec.items())
    return "\n".join(f"{k.replace('_', ' ').capitalize()}: {', '.join(v) if isinstance(v, list) else ('yes' if v is True else 'no' if v is False else v if v != '' else '(empty)')}" for k, v in rec.items())


def decoy_name(rec, r):
    vals = [v for v in rec.values() if isinstance(v, str) and v] + [x for v in rec.values() if isinstance(v, list) for x in v]
    return r.choice(vals) if vals else None


def rule_question(txt, yes, no, r):
    return r.choice([f"If {txt}, choose {yes}; otherwise choose {no}.", f"Rule: {yes} when {txt}, else {no}.", f"Choose {yes} only if {txt}. In every other case choose {no}.",
                     f"Decide for this record. {yes} if {txt}; {no} if not.", f"Apply the policy: {no} by default, {yes} when {txt}."])


def make_rule(dom, r, rec, fam):
    txt, test, flip, key = (cond_compound if fam == "compound" else COND[fam])(dom, r, rec)
    yes, no = r.choice(OUTCOMES)
    if r.random() < 0.3:                            # decoy: the wrong outcome carries a value that appears in the state
        d = decoy_name(rec, r)
        if d: yes, no = (yes, f"send to {d}") if r.random() < 0.5 else (f"send to {d}", no)
    form = r.choice(["choice", "choice", "noul"])
    if form == "noul":
        q = r.choice([f"Does this record satisfy: {txt}?", f"Is it true that {txt}?", f"Check the condition \"{txt}\". Does it hold?"]); opts = ["no", "yes"]
        gold = lambda x: int(test(x))
    else:
        q = rule_question(txt, yes, no, r); opts = [yes, no] if r.random() < 0.5 else [no, yes]
        gold = lambda x: opts.index(yes if test(x) else no)
    inverted = (lambda x: 1 - gold(x)) if form == "noul" else (lambda x: opts.index(no if test(x) else yes))
    inv_q = (r.choice([f"Does this record fail the condition: {txt}?", f"Is it false that {txt}?"]) if form == "noul" else rule_question(txt, no, yes, r))
    return q, opts, gold, flip, inv_q, inverted


def make_precedence(dom, r, rec):
    conds = []
    fams = [f for f in COND if f not in HELD_FAMILIES or dom in HELD_DOMAINS]
    while len(conds) < 3:
        c = COND[r.choice(fams)](dom, r, rec)
        if all(c[3] != o[3] for o in conds): conds.append(c)
    names = r.sample(["expedite", "escalate", "archive", "flag", "notify", "assign to Dana", "hold", "approve"], 3) + ["do nothing"]
    lines = [f"{i + 1}. if {t}: {n}" for i, (t, _, _, _) in enumerate(conds) for n in [names[i]]]
    q = r.choice(["Apply the first rule that matches, in this order:\n", "Rules, in priority order (the first match decides):\n", "Go through the rules in order and stop at the first that applies:\n"]) + \
        "\n".join(lines) + f"\n4. otherwise: {names[3]}"
    def outcome(x):
        for i, (_, test, _, _) in enumerate(conds):
            if test(x): return names[i]
        return names[3]
    opts = list(names); r.shuffle(opts)
    gold = lambda x: opts.index(outcome(x))
    def flip(x):
        for _, _, fl, _ in conds:
            y = fl(x)
            if outcome(y) != outcome(x): return y
        return x
    # rule twin: reverse the priority order
    rlines = [f"{i + 1}. if {t}: {n}" for i, (t, n) in enumerate(zip([c[0] for c in conds][::-1], names[:3][::-1]))]
    inv_q = q.split("\n")[0] + "\n" + "\n".join(rlines) + f"\n4. otherwise: {names[3]}"
    def inv_outcome(x):
        for (_, test, _, _), n in zip(conds[::-1], names[:3][::-1]):
            if test(x): return n
        return names[3]
    return q, opts, gold, flip, inv_q, lambda x: opts.index(inv_outcome(x))


def make_pointer(dom, r, rec):
    """Which option holds the value of field X?  Options are "label: value" for several fields (decoys are other fields' values,
    including ones that share text with the state); a twin swaps the target field."""
    ks = [k for k in rec if isinstance(rec[k], (str, int)) and rec[k] != ""]
    if len(ks) < 3: return None
    r.shuffle(ks); tgt, alt = ks[0], ks[1]
    opts = [f"{k.replace('_', ' ')}: {rec[k]}" for k in ks[:min(len(ks), r.randint(3, 6))]] + ["none of these"]
    q = r.choice([f"Which entry gives the record's `{tgt}`?", f"Pick the option that states the {tgt.replace('_', ' ')} of this record.", f"Which of these is the value of `{tgt}`?"])
    gold = lambda x: opts.index(f"{tgt.replace('_', ' ')}: {x[tgt]}") if f"{tgt.replace('_', ' ')}: {x[tgt]}" in opts else len(opts) - 1
    def flip(x):                                   # the target's value changes -> none of the listed entries holds it
        y = dict(x); y[tgt] = (x[tgt] + 1) if isinstance(x[tgt], int) else (DOMAINS[dom][tgt][1](r) if DOMAINS[dom][tgt][0] == "str" else r.choice([v for v in DOMAINS[dom][tgt][1] if v != x[tgt]])); return y
    inv_q = q.replace(f"`{tgt}`", f"`{alt}`").replace(tgt.replace("_", " "), alt.replace("_", " "))
    inv = lambda x: opts.index(f"{alt.replace('_', ' ')}: {x[alt]}") if f"{alt.replace('_', ' ')}: {x[alt]}" in opts else len(opts) - 1
    return q, opts, gold, flip, inv_q, inv


def make_select(dom, r, fam):
    """Several records; which one satisfies the rule?  Exactly one does (or none, with a "none" option)."""
    recs = [sample_record(dom, r) for _ in range(r.randint(3, 6))]
    txt, test, flip, key = (cond_compound if fam == "compound" else COND[fam])(dom, r, recs[0])
    for i, x in enumerate(recs):                   # force exactly one satisfying record (index j)
        pass
    j = r.randrange(len(recs)); fixed = []
    for i, x in enumerate(recs):
        want = i == j
        y = x
        for _ in range(4):
            if test(y) == want: break
            y = flip(y)
        if test(y) != want: return None
        fixed.append(y)
    recs = fixed; names = [f"record {i + 1}" if r.random() < 0.5 else (x.get("name") or x.get("id") or x.get("title") or x.get("customer") or f"record {i + 1}") for i, x in enumerate(recs)]
    if len(set(names)) < len(names): names = [f"record {i + 1}" for i in range(len(recs))]
    state = {"records": [dict(x, name=n) if "name" not in x else x for x, n in zip(recs, names)]} if r.random() < 0.7 else {n: x for n, x in zip(names, recs)}
    opts = names + ["none of them"]
    q = r.choice([f"Which record satisfies: {txt}?", f"Exactly one of these records has {txt}. Which?", f"Find the record where {txt}."])
    return state, q, opts, j


FORM_FIELDS = [("Full name", "name", person), ("Email address", "email", email), ("Phone number", "phone", lambda r: f"+1 202 555 {r.randint(100, 999):04d}"),
               ("City", "city", lambda r: r.choice(CITIES)), ("Date of birth", "dob", lambda r: f"{r.randint(1950, 2006)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}"),
               ("Employer", "employer", lambda r: "Example " + r.choice(["Trading", "Labs", "Clinic", "Logistics"])), ("Policy number", "policy", lambda r: code(r, "POL")),
               ("Emergency contact", "ice", person), ("Account number", "account", lambda r: code(r, "ACCT", 8)), ("Postal code", "zip", lambda r: str(r.randint(10000, 99999))),
               ("Passport number", "passport", lambda r: code(r, "P", 7)), ("Job title", "title", lambda r: r.choice(["Nurse", "Engineer", "Teacher", "Driver", "Analyst"]))]
ENTITY_LABELS = {"name": ["Name", "Patient", "Applicant"], "email": ["E-mail", "Email", "Contact email"], "phone": ["Telephone", "Mobile", "Phone"], "city": ["Town", "City", "Location"],
                 "dob": ["Born", "DOB", "Birth date"], "employer": ["Employer", "Company"], "policy": ["Policy no", "Policy"], "ice": ["ICE", "Emergency contact"],
                 "account": ["Account", "Account number"], "zip": ["ZIP", "Postcode"], "passport": ["Passport", "Passport no"], "title": ["Role", "Job title", "Occupation"]}
CHROME = [("Button", "Minimize"), ("Button", "Maximize"), ("Button", "Close"), ("TitleBar", None), ("ScrollBar", "Vertical"), ("MenuItem", "File"), ("MenuItem", "Help"),
          ("Edit", "Address bar"), ("Pane", "Sidebar"), ("Button", "Reload"), ("Tab", "Form")]


def make_form(r):
    """One UI element of a form plus document entities; rule: fill an empty field whose entity exists, check a required unchecked box,
    click the submit button, otherwise skip.  Rendered as an accessibility snapshot (three lines) or as JSON."""
    n = r.randint(3, 9); flds = r.sample(FORM_FIELDS, n); ents = []
    for lab, key, g in flds:
        if r.random() < 0.85: ents.append((r.choice(ENTITY_LABELS[key]), g(r), key))
    for _ in range(r.randint(0, 3)):                  # distractor entities
        lab, key, g = r.choice(FORM_FIELDS); ents.append((r.choice(ENTITY_LABELS[key]) + " (old)" if r.random() < 0.3 else r.choice(["Reference", "Office", "Status", "Notes"]), g(r), None))
    r.shuffle(ents)
    title = r.choice(["Northwind Clinic - New Patient Registration", "Example Library - Membership Sign-up", "Acme Rentals - Booking Form", "City Pool - Season Pass", "Example Bank - Account Opening"])
    kind = r.choice(["field", "field", "field", "checkbox", "button", "chrome"])
    if kind == "field":
        lab, key, g = r.choice(flds); ent = next((e for e in ents if e[2] == key), None)
        state = r.choice(["", "", "filled", "stale"]) if ent else r.choice(["", "filled"])
        value = "" if state == "" else ("stale value" if state == "stale" else (ent[1] if ent else g(r)))
        el = dict(role="Edit", label=lab, value=value)
        gold_action = ("fill", ent) if ent and state != "filled" else ("skip", None)
        def flip(_):
            e2 = dict(el); e2["value"] = (ent[1] if ent else g(r)) if not value or value == "stale value" else ""; return e2
    elif kind == "checkbox":
        lab = r.choice(["I agree to the terms", "Subscribe to the newsletter", "I confirm the information is correct", "Send me offers", "I accept the privacy policy"])
        required = lab in ("I agree to the terms", "I confirm the information is correct", "I accept the privacy policy"); checked = r.random() < 0.4
        el = dict(role="CheckBox", label=lab, checked=checked); gold_action = ("check", None) if required and not checked else ("skip", None)
        def flip(_):
            e2 = dict(el); e2["checked"] = not checked; return e2
    elif kind == "button":
        lab = r.choice(["Submit", "Submit Form", "Cancel", "Reset", "Save draft", "Back"]); el = dict(role="Button", label=lab)
        gold_action = ("click", None) if lab.startswith("Submit") else ("skip", None)
        def flip(_):
            e2 = dict(el); e2["label"] = "Cancel" if lab.startswith("Submit") else "Submit"; return e2
    else:
        role, lab = r.choice(CHROME); el = dict(role=role, label=lab or title); gold_action = ("skip", None); flip = None
    opts = [f"fill {l}: {v}" for l, v, _ in ents] + ["check", "click", "skip"]
    def gold_of(e):
        if e["role"] == "Edit":
            ent2 = next((x for x in ents if x[2] == next((k for l2, k, _ in flds if l2 == e["label"]), None)), None)
            return opts.index(f"fill {ent2[0]}: {ent2[1]}") if ent2 and (e["value"] == "" or e["value"] == "stale value") else opts.index("skip")
        if e["role"] == "CheckBox":
            req = e["label"] in ("I agree to the terms", "I confirm the information is correct", "I accept the privacy policy"); return opts.index("check") if req and not e["checked"] else opts.index("skip")
        if e["role"] == "Button": return opts.index("click") if e["label"].startswith("Submit") else opts.index("skip")
        return opts.index("skip")
    def render(e):
        if r.random() < 0.6:
            st = ("checked" if e.get("checked") else "unchecked") if e["role"] == "CheckBox" else f'value="{e.get("value", "")}"'
            return f"TASK fill the form from the document, then submit\nFORM {title}\nELEMENT {e['role']} \"{e['label']}\" {st}"
        return render_state({"task": "fill the form from the document, then submit", "form": title, "element": e})
    q = r.choice(["What should be done with this element?", "What should the form-filling agent do with this element?",
                  "Fill an empty field with its document value (also when the value is stale), check a required box that is unchecked, click the submit button; skip anything else. What applies here?"])
    ex = [Example(render(el), [Q(q, opts, gold_of(el))], "rules_form")]
    if flip is not None:
        e2 = flip(None); ex.append(Example(render(e2), [Q(q, opts, gold_of(e2))], "rules_form"))
    return ex


# ---------------------------------------------------------------- assembly
def build(n, seed=0, domains=None, families=None, tag="rules"):
    r = random.Random(seed); out = []; domains = domains or [d for d in DOMAINS if d not in HELD_DOMAINS]; families = families or [f for f in FAMILIES if f not in HELD_FAMILIES]
    tries = 0
    while len(out) < n and tries < n * 4:
        tries += 1; dom = r.choice(domains); fam = r.choice(families)
        try:
            if fam == "form":
                out += make_form(r); continue
            if fam == "select":
                res = make_select(dom, r, r.choice([f for f in COND if f in families] or ["threshold"]))
                if res is None: continue
                state, q, opts, j = res; out.append(Example(render_state(state), [Q(q, opts, j)], f"{tag}_select")); continue
            rec = sample_record(dom, r)
            made = make_precedence(dom, r, rec) if fam == "precedence" else make_pointer(dom, r, rec) if fam == "pointer" else make_rule(dom, r, rec, fam)
            if made is None: continue
            q, opts, gold, flip, inv_q, inv = made
            twin = flip(rec)
            base_ctx = render_record(rec, r)
            g0 = gold(rec); out.append(Example(base_ctx, [Q(q, opts, g0)], f"{tag}_{fam}"))
            if twin != rec and gold(twin) != g0:                          # state twin: same rule and options, one attribute moved, label flips
                out.append(Example(render_record(twin, r), [Q(q, opts, gold(twin))], f"{tag}_{fam}"))
            if r.random() < 0.7 and inv(rec) != g0:                        # rule twin: same state, rule inverted, label flips
                out.append(Example(base_ctx, [Q(inv_q, opts, inv(rec))], f"{tag}_{fam}"))
        except (ValueError, IndexError, KeyError, RecursionError):
            continue
    return out[:n]


def probe_sets(seed=1):
    """Held-out probes: unseen domains with trained families, unseen families in trained domains, and both."""
    trained_d = [d for d in DOMAINS if d not in HELD_DOMAINS]; trained_f = [f for f in FAMILIES if f not in HELD_FAMILIES and f != "form"]
    return {"rules_held_domain": build(1500, seed, HELD_DOMAINS, trained_f, "rules_hd"),
            "rules_held_family": build(1500, seed + 1, trained_d, HELD_FAMILIES, "rules_hf"),
            "rules_held_both": build(1000, seed + 2, HELD_DOMAINS, HELD_FAMILIES, "rules_hb"),
            "rules_form": build(1000, seed + 3, trained_d, ["form"], "rules_form")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60000); ap.add_argument("--out", default="data/rules.pkl"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); train = build(a.n, a.seed); pr = probe_sets()
    import collections
    print("[rules] train", len(train), collections.Counter(e.task for e in train)); print("[rules] probes", {k: len(v) for k, v in pr.items()})
    for e in train[:6]: print("---", e.task, "\n", e.context[:300], "\n Q:", e.qs[0].text[:200], "\n opts:", e.qs[0].options[:6], "gold", e.qs[0].gold)
    pickle.dump((train, pr), open(a.out, "wb"))


if __name__ == "__main__":
    main()
