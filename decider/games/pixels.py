"""Play the games from pixels only: frame + intro -> typed decision (no text state).  python -m decider.games.pixels [model] [--games a,b] [--episodes 3] [--out f.json]"""
import argparse, os, json, random, numpy as np, torch
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from decider.games import envs as G
G.RENDER = True
from decider.vision.model import VisionDecisionModel
from decider.infer import Example, Q
from decider.games.frames_data import VIS_INTRO


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model", nargs="?", default="Qwen/Qwen3.5-2B-Base"); ap.add_argument("--games", default=""); ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default=""); ap.add_argument("--mario", action="store_true")
    a = ap.parse_args()
    m = VisionDecisionModel(a.model, grad_ckpt=False).cuda().eval()
    names = a.games.split(",") if a.games else list(G.GAMES)
    res = {}
    for n in names:
        g = G.GAMES[n](); scores = []
        for s in range(a.episodes):
            g.reset(s); done = False; k = 0; opt = g.options[0]
            while not done:
                if k % g.decide_every == 0:
                    f = g.frame()
                    ex = Example(f"{g.intro} {VIS_INTRO}", [Q("What should you do right now?", list(g.options), 0)])
                    with torch.no_grad():
                        lg = m.slot_logits(m.prepare([(f, ex)]))
                    opt = g.options[int(lg[0].argmax())]
                _, done = g.step(opt); k += 1
            scores.append(g.score())
        res[n] = float(np.mean(scores)); print(f"{n:18s} {'train' if g.train else 'HELD-OUT':9s} pixels-only model {res[n]:8.2f}", flush=True); g.close()
    if a.mario:
        import gym_super_mario_bros
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from decider.games.mario import ACTIONS, RELEASE, INTRO
        OPTS = list(ACTIONS); env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT); obs = env.reset()
        done = False; steps = 0; hold = 0; rel = 0; name = "run right"; info = {"x_pos": 40}
        while not done and steps < 3000:
            if steps % 4 == 0 and steps >= rel:
                ex = Example(f"{INTRO} {VIS_INTRO}", [Q("What should Mario do right now?", OPTS, 0)])
                with torch.no_grad(): lg = m.slot_logits(m.prepare([(obs, ex)]))
                name = OPTS[int(lg[0].argmax())]
                if "jump" in name: hold = steps + 16; rel = hold + 4
            act = ACTIONS[name] if steps < hold or name not in RELEASE else RELEASE[name]
            obs, _, done, info = env.step(act); steps += 1
        res["mario_1-1"] = int(info["x_pos"]); print(f"mario 1-1 pixels-only: x={info['x_pos']}", flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
