"""Label descriptions for every fixed label set, written by a local teacher (Qwen3.5-27B, HF generate).
Used by build_v6 to train "described options" (Jev-style criteria: option name -> description / JSON rubric).
In-task labels are described from the label name + 5 training examples; held-out tasks from the name and the
question only (no eval text is shown to the teacher).
   python -m decider.data.teacher_labels teacher_data/label_descriptions.json"""
import argparse, collections, json, random, re, time, torch
from decider import data as D
from decider.data.augment import fixed_label_sets, SKIP_TASKS

SYS = "You write short, precise category descriptions for a classification rubric. Output strict JSON only."
PROMPT = ("Classification question: {q}\nAll categories: {cats}\n\nDescribe the category \"{label}\".{ex}\n"
          "Return JSON with two keys: \"what\" (one sentence, at most 25 words, saying which inputs belong in this category, concrete, "
          "without just repeating the category name) and \"not_for\" (at most 15 words: similar inputs that belong to OTHER categories instead).")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--data", default="data/tasks_v4.pkl"); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max_new", type=int, default=110)
    a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    train, evals = D.load_cache(a.data); rng = random.Random(0)
    sets = fixed_label_sets(train, evals, skip=SKIP_TASKS)
    by = collections.defaultdict(list)
    for e in train:
        for q in e.qs:
            if (e.task, q.text) in sets and q.gold >= 0 and len(by[(e.task, q.text, q.options[q.gold])]) < 40:
                by[(e.task, q.text, q.options[q.gold])].append(e.context)
    jobs = []
    for (t, qt), (opts, sp) in sorted(sets.items()):
        for lab in opts:
            exs = by.get((t, qt, lab), []) if sp == "train" else []
            exs = rng.sample(exs, min(5, len(exs)))
            ex = ("\nExample inputs in this category:\n" + "\n".join("- " + re.sub(r"\s+", " ", x)[:280] for x in exs)) if exs else ""
            cats = ", ".join(opts) if len(opts) <= 40 else ", ".join(rng.sample(opts, 40)) + ", ..."
            jobs.append((t, qt, lab, PROMPT.format(q=qt, cats=cats, label=lab, ex=ex)))
    print(f"[desc] {len(sets)} label sets, {len(jobs)} labels", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    out = {}; t0 = time.time(); bad = []
    def run(batch, temp):
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for *_, p in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=a.max_new, do_sample=temp > 0, temperature=temp or None, top_p=0.9 if temp else None, pad_token_id=tok.pad_token_id)
        return tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    jobs.sort(key=lambda j: len(j[3]))
    for attempt, temp in enumerate([0.0, 0.7]):
        todo = jobs if attempt == 0 else bad; bad = []
        for i in range(0, len(todo), a.bs):
            batch = todo[i:i + a.bs]
            for (t, qt, lab, p), txt in zip(batch, run(batch, temp)):
                try:
                    j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0)); w, nf = str(j["what"]).strip(), str(j["not_for"]).strip()
                    assert 10 <= len(w) <= 300 and len(nf) <= 200
                    out.setdefault(t, {}).setdefault(qt, {})[lab] = dict(what=w, not_for=nf)
                except Exception:
                    bad.append((t, qt, lab, p))
            print(f"[desc] pass {attempt} {min(i + a.bs, len(todo))}/{len(todo)} bad {len(bad)} {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(f"[desc] done, {sum(len(v2) for v in out.values() for v2 in v.values())} labels, {len(bad)} failed -> {a.out}")


if __name__ == "__main__":
    main()
