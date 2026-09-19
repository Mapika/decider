"""Teacher-written contrastive pairs: two states that differ in one fact, one policy, two different answers.

The rule-conditioned generator (decider.data.rules) fixed rule reading over JSON records; it did not move natural-language evidence
checking (VitaminC, PAWS, SummEval).  This is the natural-language version of the same idea, after Bespoke's curation recipe
(github.com/bespokelabsai/nimble): the teacher writes a policy, a question, a base state whose answer needs two pieces of evidence
read together, and a changed state where at most eight words of one fact differ and the answer flips.  Then, in separate calls, the
teacher re-answers every state from letter logits alone (no reasoning, no label in sight); a pair is kept only when both answers
match the written labels and differ from each other.  Optionally the focus sentence is blanked and the teacher's confidence in the
old answer must drop (the "removal check": the focus fact is what decides).

    python -m decider.data.teacher_contrastive gen data/contrastive_raw.jsonl --n 6000 [--fp8 --mem 0.36]
    python -m decider.data.teacher_contrastive verify data/contrastive_raw.jsonl teacher_data/contrastive_pairs.jsonl [--fp8 --mem 0.36]

Families target where Jev-style decisions were weakest on the public suite: claim verification, answerability, paraphrase, entailment,
policy and eligibility rules, guardrails, moderation, rubric rating, indirection (exact arithmetic stays with decider.data.rules).  Domains are shared with
teacher_questions (the last six are held out of training and become the `contrastive_*` probes)."""
import argparse, json, random, re, time, torch
from decider.data.teacher_questions import DOMAINS, STATE_KINDS, check

FAMILIES = [
    ("verification", "choice", "State = an evidence passage (2-4 sentences) and a claim. Question: how the evidence bears on the claim, criteria SUPPORTS / REFUTES / NOT ENOUGH INFO (each described). The changed state edits a number, name, date or negation in the evidence so that the label changes (e.g. SUPPORTS -> REFUTES, or REFUTES -> NOT ENOUGH INFO by removing the deciding detail)."),
    ("answerability", "noul", "State = a passage (3-5 sentences) and a question about it. Question: can the question be answered from the passage alone? The changed state removes or replaces the one detail the answer depends on, so the label flips."),
    ("paraphrase", "noul", "State = two sentences. Question: do they say the same thing? The changed state swaps a role, a number, a direction or a qualifier in one sentence so that the label flips; the sentences must share almost all their words in both versions."),
    ("entailment", "choice", "State = a premise (2-3 sentences) and a hypothesis. Question: entailment / contradiction / neutral, described. The changed state edits one fact in the premise so the label changes."),
    ("policy", "noul", "State = a short policy (1-2 rules with a condition) and a record or message it applies to, as two or three sentences. Question: does the policy permit / require / apply. Both the rule sentence and the fact sentence are needed to answer. The changed state alters the fact (who did it, when, how much, which account) so the label flips."),
    ("eligibility", "choice", "State = eligibility rules with one or two simple thresholds (age, amount, membership tier) and an applicant or case described in prose with the relevant values; no date arithmetic. Question: which outcome applies, 3-4 described options. The changed state moves one value across a threshold or changes the tier."),
    ("guardrail", "noul", "State = a user message to an assistant, plus one sentence of context about who the user is or what was agreed earlier. Question: does this message violate the given policy (name the policy in the instructions). The changed state edits the message or the context so the label flips (e.g. the request becomes about the user's own data, or an adult becomes a minor)."),
    ("moderation", "score", "State = a comment or post plus one line of context (where it was posted, who it replies to). Question: severity on 4 described levels. The changed state edits a few words so the level changes by at least two."),
    ("rubric", "score", "State = a task and a response to it (short). Question: rate the response on 4-5 described levels (correctness, completeness or faithfulness). The changed state edits the response so that one requirement is now met or now broken and the level changes by at least two."),
    ("indirection", "choice", "State = 3-4 sentences that mention two or three people or items with references such as 'the former', 'the second one', 'her manager', 'the earlier order'. Question resolves the reference (who or which), 3-4 options named after the entities. The changed state edits one relation so a different entity is the answer."),
    ("routing_condition", "choice", "State = a request plus one sentence of context (customer tier, region, time, status). Question: which handler / queue, 3-5 options whose descriptions carry conditions (e.g. 'gold customers with any billing issue'). The changed state edits the context so a different handler applies."),
]
SYS = "You write evaluation data for a decision model that answers typed questions about a state. Output strict JSON only, no markdown."
PROMPT = """Domain: {domain}.
Task family: {family}. {spec}
Write ONE item with a base state and a changed state, {kind_hint}; {tone}. The base state's answer must need TWO pieces of evidence in the state read together (a rule and a fact, or two facts); no single sentence gives the answer away. The changed state is identical to the base state except for at most eight words in ONE place, which change ONE fact (the focus fact) and therefore the answer. Everything else, including the question and its criteria, stays the same. Do not hint at the answer in the wording, do not use the words of the option names in the state.
Return JSON:
{{"question": {{"type": "{qtype}", "instructions": "...", "criteria": {crit}}},
 "base": {{"state": <string or JSON value>, "answer": <option name | true/false | level index>, "focus": "<the sentence or field that carries the focus fact, quoted from the state>"}},
 "changed": {{"state": <the same state with the edit>, "answer": <different from base>, "edit": "<old words> -> <new words>"}}}}
The answer field is {ans}."""
CRIT = {"choice": '{"<option name>": "<description>", ...} (3-5 options)', "noul": '{"true": "<when the answer is yes>", "false": "<when it is no>"}', "score": '["<level 0, lowest>", "<level 1>", ...] (4-5 levels)'}
ANS = {"choice": "the option name, exactly as in criteria", "noul": "a JSON boolean, true or false", "score": "the level index as a JSON integer"}
TONES = ["plain and factual", "conversational", "formal", "terse, like notes", "with an irrelevant detail added", "with the two pieces of evidence far apart", "with a distracting near-miss detail that does not change the answer"]


def load_teacher(a, decision=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if a.mem: torch.cuda.set_per_process_memory_fraction(a.mem)
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if decision:
        from decider.model import DecisionModel
        m = DecisionModel(a.model, grad_ckpt=False) if not a.fp8 else _fp8_decision(a.model)
        return tok, m.cuda().eval()
    if a.fp8:
        from decider.fp8 import convert_to_fp8
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu"); n = convert_to_fp8(m); print(f"[teacher] {n} linears -> fp8", flush=True)
        return tok, m.cuda().eval()
    return tok, AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()


def _fp8_decision(name):
    from decider.model import DecisionModel
    from decider.fp8 import convert_to_fp8
    m = DecisionModel(name, grad_ckpt=False); convert_to_fp8(m.lm); return m


def render_state(s): return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def coerce(q, member):
    """Teacher output arrives with strings where JSON booleans / integers were asked for."""
    a = member.get("answer"); t = q.get("type")
    if t == "noul" and isinstance(a, str) and a.strip().lower() in ("true", "false", "yes", "no"): member["answer"] = a.strip().lower() in ("true", "yes")
    if t == "score" and isinstance(a, str) and a.strip().isdigit(): member["answer"] = int(a.strip())
    if t == "score" and isinstance(q.get("criteria"), dict):                                   # legend map -> ordered list
        try: q["criteria"] = [q["criteria"][k] for k in sorted(q["criteria"], key=float)]
        except Exception: pass


def valid(j):
    q = j.get("question"); b, c = j.get("base"), j.get("changed")
    if not (isinstance(q, dict) and isinstance(b, dict) and isinstance(c, dict)): return False
    coerce(q, b); coerce(q, c)
    if not (b.get("state") and c.get("state")) or b["state"] == c["state"]: return False
    qa = dict(q, answer=b.get("answer")); qb = dict(q, answer=c.get("answer"))
    if not (check(qa) and check(qb)) or b["answer"] == c["answer"]: return False
    wa, wb = render_state(b["state"]).split(), render_state(c["state"]).split()
    if abs(len(wa) - len(wb)) > 12: return False                                   # "the same state with an edit"
    same = sum(x == y for x, y in zip(wa, wb)); return same >= 0.6 * min(len(wa), len(wb))


def gen(a):
    tok, m = load_teacher(a); rng = random.Random(a.seed); done = bad = 0; t0 = time.time()
    with open(a.path, "a") as f:
        while done < a.n:
            metas, prompts = [], []
            for _ in range(a.bs):
                d = rng.choice(DOMAINS); fam, qtype, spec = rng.choice(FAMILIES); kind = rng.choice(STATE_KINDS)
                kind_hint = "each state written as " + kind if rng.random() < 0.4 else "each state as plain prose or a small JSON object"
                p = PROMPT.format(domain=d, family=fam, spec=spec, kind_hint=kind_hint, tone=rng.choice(TONES), qtype=qtype, crit=CRIT[qtype], ans=ANS[qtype])
                metas.append(dict(domain=d, family=fam, qtype=qtype))
                prompts.append(tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            for meta, t in zip(metas, tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                try:
                    j = json.loads(re.search(r"\{.*\}", t, re.S).group(0)); assert valid(j) and j["question"]["type"] == meta["qtype"]
                    f.write(json.dumps(dict(meta, **j), ensure_ascii=False) + "\n"); done += 1
                except Exception:
                    bad += 1
            f.flush(); print(f"[contrastive] {done}/{a.n} ok, {bad} rejected, {(time.time()-t0)/60:.1f} min", flush=True)


def members(rec):
    """-> [(state, question-with-answer)] for base and changed."""
    return [(rec["base"]["state"], dict(rec["question"], answer=rec["base"]["answer"])), (rec["changed"]["state"], dict(rec["question"], answer=rec["changed"]["answer"]))]


def to_examples(rec, D, S1, task="contrastive"):
    out = []
    for st, q in members(rec):
        r = S1.render_question(q); gold = r["names"].index(q["answer"]) if q["type"] != "score" else q["answer"]
        out.append(D.Example(S1.render_state(st), [D.Q(r["question"], r["options"], gold)], task))
    return out


def blank_focus(rec):
    """The base state with the focus sentence removed (for the removal check); None if the focus text is not found."""
    st = render_state(rec["base"]["state"]); foc = str(rec["base"].get("focus", "")).strip().strip('"')
    if len(foc) < 8 or foc not in st: return None
    return st.replace(foc, "[...]", 1)


def verify(a):
    from decider.model import collate
    from decider.prompt import build
    from decider import data as D
    from decider import systemone as S1
    recs = [json.loads(l) for l in open(a.path)]; tok, m = load_teacher(a, decision=True)
    class K:
        def shuffle(self, x): pass
        def sample(self, xs, k): return xs[:k]
    rows = []                                                       # (rec idx, member idx | 2 = blanked base, built prompt)
    for i, rec in enumerate(recs):
        exs = to_examples(rec, D, S1)
        for k, ex in enumerate(exs): rows.append((i, k, build(ex, m.tok, K(), max_options=255, max_ctx_tokens=4096)))
        bl = blank_focus(rec)
        if bl: rows.append((i, 2, build(D.Example(bl, exs[0].qs, "c"), m.tok, K(), max_options=255, max_ctx_tokens=4096)))
    rows.sort(key=lambda r: len(r[2]["ids"])); t0 = time.time(); res = {}
    with torch.no_grad():
        for s in range(0, len(rows), a.vbs):
            chunk = rows[s:s + a.vbs]; b = collate([r[2] for r in chunk], m.tok.pad_token_id)
            p = torch.softmax(m.slot_logits(*[b[x].cuda() for x in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")]), -1).cpu()
            for (i, k, it), pr in zip(chunk, p): res[(i, k)] = (int(pr.argmax()), float(pr[it["golds"][0]]), pr[:it["nopts"][0]].tolist())
            if (s // a.vbs) % 50 == 0: print(f"[verify] {s}/{len(rows)} {(time.time()-t0)/60:.1f} min", flush=True)
    kept = 0; stats = {}
    with open(a.out, "w") as f:
        for i, rec in enumerate(recs):
            (pa, ca, _), (pb, cb, _) = res[(i, 0)], res[(i, 1)]
            exs = to_examples(rec, D, S1); ga, gb = exs[0].qs[0].gold, exs[1].qs[0].gold
            tol = 1 if rec["question"]["type"] == "score" else 0
            ok = abs(pa - ga) <= tol and abs(pb - gb) <= tol and pa != pb and min(ca, cb) >= a.min_p
            removal = None
            if (i, 2) in res:
                removal = res[(i, 2)][1] <= ca - a.drop                     # confidence in the base answer must drop once the focus is blanked
                ok = ok and (removal or not a.strict)
            stats.setdefault(rec["family"], []).append(ok)
            if ok or a.keep_all:
                f.write(json.dumps(dict(rec, teacher_ok=ok, teacher_p=[round(ca, 3), round(cb, 3)], removal_ok=removal), ensure_ascii=False) + "\n"); kept += ok
    print(f"[verify] kept {kept}/{len(recs)} pairs; by family:", {k: f"{sum(v)}/{len(v)}" for k, v in sorted(stats.items())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "verify"]); ap.add_argument("path"); ap.add_argument("out", nargs="?")
    ap.add_argument("--n", type=int, default=6000); ap.add_argument("--model", default="Qwen/Qwen3.5-27B"); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max_new", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--vbs", type=int, default=16); ap.add_argument("--fp8", action="store_true"); ap.add_argument("--mem", type=float, default=0.0, help="cap on this process's share of GPU memory")
    ap.add_argument("--min_p", type=float, default=0.5); ap.add_argument("--drop", type=float, default=0.15); ap.add_argument("--strict", action="store_true", help="require the removal check when the focus text was found"); ap.add_argument("--keep_all", action="store_true")
    a = ap.parse_args(); dict(gen=gen, verify=verify)[a.cmd](a)
