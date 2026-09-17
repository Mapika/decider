"""Teacher-written custom questions (Qwen3.5-27B, HF generate): one generation = one state + 8 typed questions with answers,
in the Jev request shape (noul / choice with criteria / score with levels).  Targets what the public datasets do not cover:
free-form yes/no questions about arbitrary properties, user-named options (snake_case ids, descriptions), and option lists
with a GENERIC option next to a catch-all ("support" vs "other"), where the generic one is the right answer.
   python -m decider.synth_custom gen data/synth_custom_raw.jsonl --n 7000
   python -m decider.synth_custom verify data/synth_custom_raw.jsonl data/synth_custom.jsonl     (teacher re-answers every question
        on its own, from letter logits; only questions where it agrees with the answer written at generation time are kept)"""
import argparse, json, random, re, time, torch

DOMAINS = ["retail banking app", "online shop customer service", "IT helpdesk of a mid-size company", "HR department inbox", "a medical practice's front desk", "smart-home voice assistant",
           "airline customer support", "B2B SaaS support desk", "restaurant bookings and enquiries", "school administration office", "insurance claims intake", "property management (tenants and repairs)",
           "logistics and parcel delivery", "hotel reception", "telecom provider support", "e-commerce marketplace seller support", "legal intake at a law firm", "recruiting and job applications",
           "content moderation for a community forum", "security operations (alerts and incidents)", "DevOps on-call (deploys, alerts, logs)", "code review and pull requests", "sales lead qualification",
           "accounts payable (invoices, purchase orders)", "procurement requests", "utility company (electricity, water) support", "car dealership and service garage", "pharmacy counter",
           "university admissions", "library help desk", "event ticketing", "food delivery app", "ride-hailing support", "fitness studio membership", "municipal citizen services", "tax advisory office",
           "veterinary clinic", "travel agency", "warehouse operations", "manufacturing quality control", "fleet and vehicle telematics", "energy grid monitoring", "clinical trial coordination",
           "newsroom tip line", "non-profit donor relations", "game studio player support", "app store review triage", "email triage for an executive assistant", "research lab equipment booking",
           "an autonomous agent's tool-call trace", "a web-browsing agent's page observations", "a voice call transcript at a call centre", "a farm management system", "a construction site daily log"]
STATE_KINDS = ["a short chat message (one or two sentences, informal, maybe typos)", "a short chat message (one or two sentences, informal, maybe typos)",
               "a one-line request typed into a search or command box", "two or three chat messages from the same person, sent in a row", "an email with a subject line", "a support ticket with subject and body", "a JSON object with 4-8 named fields (some nested)",
               "a JSON object holding a short conversation as an array of turns plus some metadata", "a log excerpt of 4-8 lines", "a form submission rendered as 'Field: value' lines",
               "a product or service review", "a paragraph from a report or policy document", "sensor or metric readings with units and limits", "a voicemail transcript", "a JSON array of 3-6 small records"]
NAME_STYLES = ["snake_case identifiers (e.g. billing_issue)", "short natural phrases (e.g. billing issue)", "Title Case labels", "kebab-case or dotted ids (e.g. billing.issue)"]
CATCHALLS = ["other", "none of the above", "unrelated", "something else", "not_applicable", "out_of_scope", "none of these", "other / not covered", "misc"]
CHOICE_RECIPES = [
    ("generic", "One of the choice questions MUST offer 2-3 specific options, ONE GENERIC option (such as general_support, general enquiry, customer_service, misc_request, general question) "
                "and the catch-all option \"{ca}\". Write the state so that it is clearly within the scope of the question but matches none of the specific options, so the GENERIC option is the correct answer (not the catch-all)."),
    ("generic", "One of the choice questions MUST offer 2-3 specific options, ONE GENERIC option (such as general_support, general enquiry, customer_service, misc_request, general question) "
                "and the catch-all option \"{ca}\". Write the state so that it is clearly within the scope of the question but matches none of the specific options, so the GENERIC option is the correct answer (not the catch-all)."),
    ("catchall", "One of the choice questions MUST offer 3-4 options plus the catch-all option \"{ca}\", and must ask about a scope the state has nothing to do with, so that the catch-all is the correct answer."),
    ("specific_with_catchall", "One of the choice questions MUST include the catch-all option \"{ca}\" although one of the specific options clearly fits and is the correct answer."),
    ("plain", "No catch-all options in this item."), ("plain", "No catch-all options in this item.")]
SYS = "You write evaluation data for a decision model that answers typed questions about a state. Output strict JSON only, no markdown."
PROMPT = """Domain: {domain}.
Write ONE realistic state: {kind}. Make it specific (names, numbers, dates) and natural, at most 90 words; {tone}.
Then write exactly 8 questions about it, each answerable by a careful reader in a second:
- 3 of type "noul": a yes/no question or a statement to judge. Ask about varied properties: intent, who is speaking or affected, implied facts, presence of a detail, a numeric comparison, scope ("is this something the X team would handle"), tone, risk. At least one noul answer must be true and at least one false. One of them may add "criteria": {{"true": "...", "false": "..."}}.
- 3 of type "choice": "criteria" is a map from option name to a description or null, 3-7 options, names written as {names}. Describe the options in {ndesc} of the choice questions. Exactly one option is correct. Keep every description under 12 words.
- 2 of type "score": "criteria" is an ordered array of 3-5 level descriptions (situations, not numbers), each under 12 words, lowest first; "answer" is the index of the best-fitting level.
{recipe}
{pathrule}Do not leak answers in the wording. Return JSON: {{"state": <string or JSON value>, "questions": [{{"type": "...", "instructions": "...", "criteria": ..., "answer": <true/false | option name | level index>}}, ...]}}"""
TONES = ["the writer is calm", "the writer is annoyed", "the content is terse and factual", "the content is rambling and buries the point", "it mixes two topics", "it contains an irrelevant detail",
         "it is ambiguous on one point but clear on the rest", "it is polite and formal", "it is urgent", "it is routine and unremarkable"]


def gen(a):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed); done = bad = 0; t0 = time.time()
    with open(a.path, "a") as f:
        while done < a.n:
            metas, prompts = [], []
            for _ in range(a.bs):
                d, kind = rng.choice(DOMAINS), rng.choice(STATE_KINDS); rname, rtext = rng.choice(CHOICE_RECIPES); ca = rng.choice(CATCHALLS)
                path = "Because the state is JSON, at least two questions must name the field they are about with a path in backticks, such as `order.items[1].status`.\n" if "JSON" in kind else ""
                p = PROMPT.format(domain=d, kind=kind, tone=rng.choice(TONES), names=rng.choice(NAME_STYLES), ndesc=rng.choice(["none", "one", "two", "all"]), recipe=rtext.format(ca=ca), pathrule=path)
                metas.append(dict(domain=d, kind=kind, recipe=rname, catchall=ca))
                prompts.append(tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            for meta, t in zip(metas, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                try:
                    j = json.loads(re.search(r"\{.*\}", t, re.S).group(0)); qs = [q for q in j["questions"] if check(q)]
                    assert j["state"] and len(qs) >= 5
                    f.write(json.dumps(dict(meta, state=j["state"], questions=qs), ensure_ascii=False) + "\n"); done += 1
                except Exception:
                    bad += 1
            f.flush(); print(f"[synth] {done}/{a.n} ok, {bad} rejected, {(time.time()-t0)/60:.1f} min", flush=True)


ROUTE_PROMPT = """Domain: {domain}.
Design ONE routing question for short messages arriving there. Give "instructions" (the question) and "criteria": an option map with {k} specific options, ONE GENERIC option
(the bucket for messages that are in scope but match none of the specific options, e.g. general_support, general enquiry, customer_service, other_request, misc_question) and the
catch-all option "{ca}" (for messages that have nothing to do with this place). Option names as {names}. {desc}
Then write 9 short, realistic messages (4-25 words, {tone}), each with the correct option:
- 4 that are clearly in scope but fit none of the specific options, so the GENERIC option is correct (questions about opening hours, policies, how-to, complaints, odd requests ...);
- 3 that fit one of the specific options;
- 2 that are entirely out of scope (wrong number, spam, a request for a different kind of business, small talk), so "{ca}" is correct.
Return JSON: {{"instructions": "...", "criteria": {{...}}, "generic": "<name of the generic option>", "messages": [{{"text": "...", "answer": "<option name>"}}, ...]}}"""


def gen_routing(a):
    """Targeted set for the generic-vs-catch-all weakness: short messages, terse option lists."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed + 1); done = bad = 0; t0 = time.time()
    with open(a.path, "a") as f:
        while done < a.n:
            metas, prompts = [], []
            for _ in range(a.bs):
                d = rng.choice(DOMAINS); ca = rng.choice(CATCHALLS)
                desc = rng.choice(["Give no descriptions (every value null).", "Give no descriptions (every value null).", "Describe every option in under 10 words.", "Describe only the generic and the catch-all option, under 10 words each."])
                p = ROUTE_PROMPT.format(domain=d, k=rng.choice([2, 2, 3, 3, 4, 5]), ca=ca, names=rng.choice(NAME_STYLES), desc=desc, tone=rng.choice(["informal, some with typos", "polite", "terse", "mixed tone"]))
                metas.append(dict(domain=d, kind="short message", recipe="routing", catchall=ca))
                prompts.append(tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            for meta, t in zip(metas, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                try:
                    j = json.loads(re.search(r"\{.*\}", t, re.S).group(0)); crit = j["criteria"]
                    assert isinstance(crit, dict) and 3 <= len(crit) <= 9 and meta["catchall"] in crit and j["generic"] in crit and j["generic"] != meta["catchall"]
                    for msg in j["messages"]:
                        if isinstance(msg.get("text"), str) and msg.get("answer") in crit and 3 <= len(msg["text"]) <= 300:
                            grp = "generic" if msg["answer"] == j["generic"] else "catchall" if msg["answer"] == meta["catchall"] else "specific"
                            f.write(json.dumps(dict(meta, group=grp, state=msg["text"], questions=[dict(type="choice", instructions=j["instructions"], criteria=crit, answer=msg["answer"])]), ensure_ascii=False) + "\n")
                    done += 1
                except Exception:
                    bad += 1
            f.flush(); print(f"[route] {done}/{a.n} ok, {bad} rejected, {(time.time()-t0)/60:.1f} min", flush=True)


def check(q):
    t, c, ans = q.get("type"), q.get("criteria"), q.get("answer")
    if not isinstance(q.get("instructions"), str) or len(q["instructions"]) < 8: return False
    if t == "noul": return isinstance(ans, bool) and (c is None or isinstance(c, dict))
    if t == "choice": return isinstance(c, dict) and 2 <= len(c) <= 10 and ans in c
    if t == "score": return isinstance(c, list) and 2 <= len(c) <= 10 and isinstance(ans, int) and not isinstance(ans, bool) and 0 <= ans < len(c)
    return False


def to_example(rec, D, S1, task="custom"):
    """One packed Example per generated state, in the prompt form the API renders."""
    qs = []
    for q in rec["questions"]:
        r = S1.render_question(q); gold = r["names"].index(q["answer"]) if q["type"] != "score" else q["answer"]
        qs.append(D.Q(r["question"], r["options"], gold))
    return D.Example(S1.render_state(rec["state"]), qs, task)


def verify(a):
    """Teacher answers every question alone (one row per question, letter logits, zero-shot) and we keep agreements."""
    from .model import DecisionModel, collate
    from .prompt import build
    from . import data as D, systemone as S1
    recs = [json.loads(l) for l in open(a.path)]
    m = DecisionModel(a.model, grad_ckpt=False).cuda().eval(); rng = random.Random(0); kept = tot = 0; agree = {}
    class K:
        def shuffle(self, x): pass
        def sample(self, xs, k): return xs[:k]
    rows = []
    for i, rec in enumerate(recs):
        ex = to_example(rec, D, S1)
        for k, q in enumerate(ex.qs):
            rows.append((i, k, build(D.Example(ex.context, [q], "c"), m.tok, K(), max_options=255, max_ctx_tokens=4096)))
    rows.sort(key=lambda r: len(r[2]["ids"])); t0 = time.time(); ok = {}
    with torch.no_grad():
        for s in range(0, len(rows), a.vbs):
            chunk = rows[s:s + a.vbs]; b = collate([r[2] for r in chunk], m.tok.pad_token_id)
            p = torch.softmax(m.slot_logits(*[b[x].cuda() for x in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")]), -1).cpu()
            for (i, k, it), pr in zip(chunk, p):
                q = recs[i]["questions"][k]; g = it["golds"][0]; pred = int(pr.argmax())
                good = abs(pred - g) <= 1 if q["type"] == "score" else pred == g
                ok[(i, k)] = (bool(good), float(pr[g]), pred); key = (q["type"], recs[i].get("group") or (recs[i]["recipe"] if q["type"] == "choice" else "-")); agree.setdefault(key, []).append(good)
            if (s // a.vbs) % 50 == 0: print(f"[verify] {s}/{len(rows)} {(time.time()-t0)/60:.1f} min", flush=True)
    with open(a.out, "w") as f:
        for i, rec in enumerate(recs):
            names = lambda q: list(q["criteria"]) if q["type"] == "choice" else None
            qs = [dict(q, teacher_p=round(ok[(i, k)][1], 4), teacher_ok=ok[(i, k)][0], teacher_pred=(names(q)[ok[(i, k)][2]] if q["type"] == "choice" and ok[(i, k)][2] < len(names(q)) else ok[(i, k)][2]))
                  for k, q in enumerate(rec["questions"]) if ok[(i, k)][0] or a.keep_all]; tot += len(rec["questions"]); kept += len(qs)
            if len(qs) >= min(2, len(rec["questions"])): f.write(json.dumps(dict(rec, questions=qs), ensure_ascii=False) + "\n")
    print(f"[verify] kept {kept}/{tot} questions; agreement by type:", {str(k): round(sum(v) / len(v), 3) for k, v in sorted(agree.items())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "verify", "gen_routing"]); ap.add_argument("path"); ap.add_argument("out", nargs="?")
    ap.add_argument("--n", type=int, default=7000); ap.add_argument("--model", default="Qwen/Qwen3.5-27B"); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=1100); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--vbs", type=int, default=32); ap.add_argument("--keep_all", action="store_true", help="write disagreements too, with teacher_ok / teacher_pred")
    a = ap.parse_args(); dict(gen=gen, verify=verify, gen_routing=gen_routing)[a.cmd](a)
