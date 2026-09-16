"""Fill RESULTS_TABLE in MODEL_CARD.md from eval.json files. usage: python -m decider.card_table zs_2b=runs/zs_2b zs_4b=runs/zs_4b final=runs/r2_full/final"""
import json, sys
rows = [a.split("=", 1) for a in sys.argv[1:]]
names = {"zs_2b": "Qwen3.5-2B-Base, zero-shot", "zs_4b": "Qwen3.5-4B-Base, zero-shot", "r1": "this model (200k-example run)", "final": "**this model**"}
hdr = "| Model | Split | Acc | NLL | Brier | ECE | AURC | Acc@80% |\n|---|---|---|---|---|---|---|---|\n"
out = hdr
for k, p in rows:
    agg = json.load(open(f"{p}/eval.json"))["agg"]
    for split, lab in [("in_task", f"in-task ({agg['n_in']})"), ("heldout", f"held-out ({agg['n_heldout']})")]:
        a = agg[split]
        out += f"| {names.get(k, k)} | {lab} | {a['acc']:.3f} | {a['nll']:.3f} | {a['brier']:.3f} | {a['ece']:.3f} | {a['aurc']:.3f} | {a['acc_at_80']:.3f} |\n"
# per-task held-out table
res = {k: json.load(open(f"{p}/eval.json"))["results"] for k, p in rows}
last = rows[-1][0]
ho = sorted(t for t, v in res[last].items() if v["heldout"])
ht = "Per-task accuracy / ECE on the held-out datasets:\n\n| Task | " + " | ".join(names.get(k, k).replace("**", "") for k, _ in rows) + " |\n|---|" + "---|" * len(rows) + "\n"
for t in ho:
    ht += f"| {t} | " + " | ".join(f"{res[k][t]['acc']:.3f} / {res[k][t]['ece']:.3f}" if t in res[k] else "" for k, _ in rows) + " |\n"
s = open("MODEL_CARD.md").read()
s = s.replace("HELDOUT_TABLE", ht) if "HELDOUT_TABLE" in s else __import__("re").sub(r"Per-task accuracy / ECE.*?\n\n", ht + "\n", s, flags=__import__("re").S)
if "RESULTS_TABLE" in s:
    s = s.replace("RESULTS_TABLE", out)
else:
    import re; s = re.sub(r"\| Model \| Split.*?\n\n", out + "\n", s, flags=re.S)
open("MODEL_CARD.md", "w").write(s); print(out)
