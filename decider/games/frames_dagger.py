"""DAgger for the pixel policy: play the train games with a vision model (sampling), label every frame with the
teacher, oversample rare teacher actions.  python -m decider.games.frames_dagger runs/v1_vision/model data/frames_dagger.pkl"""
import os, io, pickle, random, sys, numpy as np, torch


def _cli():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from collections import Counter
    from PIL import Image
    from decider.games import envs as G
    G.RENDER = True
    from decider.vision.model import VisionDecisionModel
    from decider.infer import Example, Q
    from decider.games.frames_data import VIS_INTRO, png

    model_path, out_path = sys.argv[1], sys.argv[2]; episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    m = VisionDecisionModel(model_path, grad_ckpt=False).cuda().eval(); rng = random.Random(3); data = []
    for name, cls in G.GAMES.items():
        if not cls.train: continue
        g = cls(); n0 = len(data)
        for ep in range(episodes):
            g.reset(seed=5000 + ep); done = False; k = 0; opt = g.options[0]; temp = [1.0, 1.5, 0.7][ep % 3]
            while not done and k < 400:
                if k % g.decide_every == 0:
                    f = g.frame(); lab = g.teacher()
                    ex = Example(f"{cls.intro} {VIS_INTRO}", [Q("What should you do right now?", list(g.options), g.options.index(lab))], f"frames_{name}", image=png(f))
                    data.append(ex)
                    with torch.no_grad(): lg = m.slot_logits(m.prepare([(f, ex)]))
                    p = torch.softmax(lg[0, :len(g.options)] / temp, -1).cpu(); opt = g.options[int(torch.multinomial(p, 1))]
                _, done = g.step(opt); k += 1
        g.close(); c = Counter(e.qs[0].options[e.qs[0].gold] for e in data[n0:]); print(f"[dagger] {name}: {len(data)-n0} frames {c.most_common()}", flush=True)
    # oversample rare teacher actions to at least 15% share per game
    out = []
    for name in {e.task for e in data}:
        sub = [e for e in data if e.task == name]; c = Counter(e.qs[0].options[e.qs[0].gold] for e in sub); n = len(sub)
        for e in sub:
            share = c[e.qs[0].options[e.qs[0].gold]] / n; rep = 1 if share >= 0.15 else min(6, int(round(0.15 / max(share, 1e-3))))
            out += [e] * rep
    rng.shuffle(out); print(f"[dagger] total {len(data)} -> balanced {len(out)}")
    pickle.dump((out, {}), open(out_path, "wb"))


if __name__ == "__main__":
    _cli()
