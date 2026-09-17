"""v8 delta set: isolated level scoring.  Every level of a Score question becomes its own yes/no row
(state + question + that one level, number stripped; systemone.ISOLATED), gold "yes" only for the labelled level.
Each row is trained with ordinary cross-entropy, so P(fits) is a calibrated marginal that never depends on the other
levels; at inference the per-level probabilities are normalised over the levels (systemone.combine_isolated).
Sources: the ordinal-scale datasets, the teacher-written score questions, and (so Choice can be scored the same way)
small label sets and the teacher's choice / routing questions.  Plus a replay of the v7 mix.
   python -m decider.build_v8 -> data/tasks_v8_delta.pkl"""
import json, pickle, random, re
from . import data as D
from . import data2, data3, data4  # noqa
from . import systemone as S1
from .build_v7 import HELD_DOMAINS
from .synth_custom import to_example

SCALE_TASKS = ["helpsteer2", "helpsteer3_pref", "hate_speech_scales", "liar2", "prosocial_safety", "stsb"]


def rows(e, q, tag):
    return [D.Example(e.context, [D.Q(text, opts, 1 if j == q.gold else 0)], tag) for j, (text, opts) in enumerate(S1.isolated_rows(q.text, q.options))]


def main():
    rng = random.Random(8); train, evals = D.load_cache("data/tasks_v4.pkl"); new = []
    is_scale = lambda q: len(q.options) >= 3 and all(re.match(r"^-?\d+:", o) for o in q.options)
    pool = [(e, q) for e in train if e.task in SCALE_TASKS for q in e.qs if is_scale(q) and q.gold >= 0]
    for e, q in rng.sample(pool, 26000): new += rows(e, q, e.task + "+iso")
    n_scale = len(new)
    recs = [json.loads(l) for l in open("data/synth_custom.jsonl")]; recs = [r for r in recs if r["domain"] not in HELD_DOMAINS]
    for r in recs:
        ex = to_example(r, D, S1, "custom")
        for q, m in zip(ex.qs, r["questions"]):
            if m["type"] == "score":
                for _ in range(2): new += rows(ex, q, "custom+iso")
            elif m["type"] == "choice" and rng.random() < 0.5:
                new += rows(ex, q, "custom+iso")
    routes = [json.loads(l) for l in open("data/synth_routing.jsonl")]
    for r in rng.sample([r for r in routes if r["domain"] not in HELD_DOMAINS and r["questions"][0].get("teacher_ok")], 1200):
        r["questions"][0]["criteria"] = {k: (None if v in ("null", "", None) else v) for k, v in r["questions"][0]["criteria"].items()}
        ex = to_example(r, D, S1, "routing"); new += rows(ex, ex.qs[0], "routing+iso")
    n_custom = len(new) - n_scale
    small = [(e, q) for e in rng.sample(train, 120000) for q in e.qs if 3 <= len(q.options) <= 8 and not is_scale(q) and q.gold >= 0 and len(e.context) < 1500 and e.task not in ("games", "mario")]
    for e, q in rng.sample(small, 9000): new += rows(e, q, e.task + "+iso")
    n_choice = len(new) - n_scale - n_custom
    v7, _ = D.load_cache("data/tasks_v7_delta.pkl"); replay = rng.sample(v7, 110000)
    out = new + replay; rng.shuffle(out)
    print(f"[v8] isolated rows: scales {n_scale} + teacher score/choice/routing {n_custom} + small label sets {n_choice}; replay {len(replay)}; total {len(out)}; yes-rate {sum(e.qs[0].gold for e in new)/len(new):.3f}")
    keep = ["helpsteer2", "hate_speech_scales", "clinc_oos", "support_tickets", "offtopic_probe", "trec", "sciq", "agenttraj"]
    pickle.dump((out, {k: v for k, v in evals.items() if k in keep}), open("data/tasks_v8_delta.pkl", "wb"))


if __name__ == "__main__":
    main()
