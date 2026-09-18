"""decider, zero-shot, on the form-filling task of Cua's CUA-S1-FORMS specialist (github.com/trycua/cua, libs/cua-s1, MIT).

Their generator writes one row per form element: a 3-line context (task / form title / element), the document entities as
"fill <label>: <value>" options plus check / click / skip, and the gold option.  Generate their test split first:

    git clone --depth 1 https://github.com/trycua/cua /tmp/cua
    PYTHONPATH=/tmp/cua/libs/cua-s1/python/src python -m cua_s1.synth --output /tmp/cua_s1_data --episodes 6000
    python -m decider.probes.cua_s1_forms runs/r15_v9b/model /tmp/cua_s1_data/test.jsonl [--prompt bare|short|rules]

"bare" is their format verbatim (no question, no descriptions); "short" adds a one-sentence question and "skip (leave this
element alone)"; "rules" spells out every rule in the question.  Results (v9, 14,254 decisions, forms disjoint from their training
set): bare 0.41, short 0.67, rules 0.24.  Their 0.7M-parameter specialist: 0.9995 in-distribution; Jev's hosted API, per their card: 0.836."""
import argparse, collections, json, sys, time

PROMPTS = {
    "bare": ("What should be done with this element?", {}),
    "short": ("What should the form-filling agent do with this element? Fill it with the matching document value if it is an empty form field; "
              "otherwise check, click (submit) or skip.", {"skip": "skip (leave this element alone)"}),
    "rules": ("The agent fills this form from the document and then submits it. What should it do with the element above? "
              "Fill an empty (or stale) form field with the document entity that belongs to it; check a required checkbox that is unchecked; "
              "click the submit button; skip everything else: browser chrome (tabs, address bar, menus), fields that are already filled with the right value, "
              "checkboxes that are already checked or optional, buttons that do not submit, and fields with no matching document entity.",
              {"check": "check: tick this checkbox (it is required and unchecked)", "click": "click: press this button (it submits the form)",
               "skip": "skip: leave this element alone (chrome, already filled correctly, optional or already checked, not the submit button, no matching entity)"}),
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("rows"); ap.add_argument("--prompt", default="short", choices=PROMPTS); ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    from decider.infer import Decider
    q, desc = PROMPTS[a.prompt]
    rows = [json.loads(l) for l in open(a.rows)]
    d = Decider(a.model); t = time.time(); preds = []
    for i in range(0, len(rows), a.bs):
        chunk = rows[i:i + a.bs]
        preds += [o[0] for o in d.decide_batch([(r["context"], [{"question": q, "options": [desc.get(o, o) for o in r["options"]]}]) for r in chunk])]
    gold = [desc.get(r["options"][r["label"]], r["options"][r["label"]]) for r in rows]
    by = collections.defaultdict(lambda: [0, 0]); skip = collections.defaultdict(lambda: [0, 0]); n80 = []; wrong_action = wrong_target = 0
    for r, g, p in zip(rows, gold, preds):
        ok = g == p["choice"]; act = r["meta"]["action"]; by[act][0] += ok; by[act][1] += 1
        if p["confidence"] >= 0.8: n80.append(ok)
        if not ok:
            if p["choice"].split()[0].rstrip(":") == act: wrong_target += 1
            else: wrong_action += 1
        if act == "skip":
            c = r["context"].split("ELEMENT ")[1]; role = c.split()[0]
            kind = role if role != "Edit" else ("Edit filled" if 'value=""' not in c else "Edit empty"); skip[kind][0] += ok; skip[kind][1] += 1
    acc = sum(g == p["choice"] for g, p in zip(gold, preds)) / len(rows)
    print(f"{a.model} [{a.prompt}]: acc {acc:.4f} on {len(rows)} decisions ({time.time() - t:.0f}s); " + ", ".join(f"{k} {v[0] / v[1]:.3f} (n={v[1]})" for k, v in sorted(by.items())))
    print(f"  coverage at conf>=0.8 {len(n80) / len(rows):.3f}, selective acc {sum(n80) / max(1, len(n80)):.4f}; errors: wrong action {wrong_action}, wrong target {wrong_target}")
    print("  skip rows by kind: " + ", ".join(f"{k} {v[0] / v[1]:.2f} (n={v[1]})" for k, v in sorted(skip.items(), key=lambda x: -x[1][1])))


if __name__ == "__main__":
    main()
