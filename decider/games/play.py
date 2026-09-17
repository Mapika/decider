"""Play every game with: the model (zero-shot typed decision), the scripted teacher, random.
   python -m decider.games.play [model] [--games a,b] [--episodes 5]"""
import argparse, random, json, numpy as np
from decider.games import envs as G


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?", default="runs/r3_v2/model"); ap.add_argument("--games", default=""); ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--out", default=""); ap.add_argument("--no_model", action="store_true"); ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()
    names = a.games.split(",") if a.games else list(G.GAMES)
    dec = None
    if not a.no_model:
        from decider.infer import Decider
        dec = Decider(a.model, use_graphs=not a.eager)
    res = {}
    for n in names:
        g = G.GAMES[n](); rng = random.Random(0)
        def model_policy(txt, opts, gm):
            r = dec.decide(f"{gm.intro}\n\nSituation: {txt}", [{"question": "What should you do right now?", "options": list(opts)}])[0]
            return r["choice"]
        t = float(np.mean([G.play(g, lambda txt, opts, gm: gm.teacher(), s) for s in range(a.episodes)]))
        r = float(np.mean([G.play(g, lambda txt, opts, gm: rng.choice(opts), s) for s in range(a.episodes)]))
        m = float(np.mean([G.play(g, model_policy, s) for s in range(a.episodes)])) if dec else float("nan")
        res[n] = dict(train=g.train, model=m, teacher=t, random=r)
        print(f"{n:18s} {'train' if g.train else 'HELD-OUT':9s} model {m:8.2f}   teacher {t:8.2f}   random {r:8.2f}", flush=True)
        g.close()
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
