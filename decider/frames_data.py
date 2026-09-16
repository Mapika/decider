"""Pixels-only decision data: game frames (and Mario frames) with teacher labels. python -m decider.frames_data data/frames.pkl"""
import io, os, pickle, random, sys, numpy as np
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from PIL import Image
from . import games as G
G.RENDER = True
from .data import Example, Q

def png(arr):
    b = io.BytesIO(); Image.fromarray(np.asarray(arr)).convert("RGB").save(b, format="PNG"); return b.getvalue()

VIS_INTRO = "The image shows the current game screen."

def game_frames(name, cls, episodes, rng, keep=0.5):
    out = []; g = cls()
    for ep in range(episodes):
        g.reset(seed=2000 + ep); done = False; k = 0; eps = [0.0, 0.1, 0.25][ep % 3]; opt = g.options[0]
        while not done and k < 400:
            if k % g.decide_every == 0:
                lab = g.teacher(); f = g.frame()
                if f is not None and rng.random() < keep:
                    out.append(Example(f"{cls.intro} {VIS_INTRO}", [Q("What should you do right now?", list(g.options), g.options.index(lab))], f"frames_{name}", image=png(f)))
                opt = rng.choice(g.options) if rng.random() < eps else lab
            _, done = g.step(opt); k += 1
    g.close(); return out

def mario_frames(episodes, rng, keep=0.5):
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from .mario import heuristic, ACTIONS, RELEASE, INTRO
    from .mario_data import danger
    OPTS = list(ACTIONS); out = []
    for ep in range(episodes):
        lv = ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1", "5-1", "6-1"][ep % 8]
        env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{lv}-v0"), SIMPLE_MOVEMENT); obs = env.reset(); ram = env.unwrapped.ram
        done = False; steps = 0; hold = 0; rel = 0; name = "run right"; eps = [0.0, 0.1, 0.2][ep % 3]; stall = 0; last_x = None
        while not done and steps < 2000:
            if steps % 4 == 0 and steps >= rel:
                lab = heuristic(ram, {})
                if rng.random() < keep:
                    out.append(Example(f"{INTRO} {VIS_INTRO}", [Q("What should Mario do right now?", OPTS, OPTS.index(lab)), Q("Is Mario in immediate danger?", ["no", "yes"], danger(ram))], "frames_mario", image=png(obs)))
                name = rng.choice(OPTS) if rng.random() < eps else lab
                if "jump" in name: hold = steps + 16; rel = hold + 4
            act = ACTIONS[name] if steps < hold or name not in RELEASE else RELEASE[name]
            obs, _, done, info = env.step(act); ram = env.unwrapped.ram; steps += 1
            if info["x_pos"] == last_x: stall += 1
            else: stall = 0; last_x = info["x_pos"]
            if stall > 300: break
        env.close()
    return out

if __name__ == "__main__":
    rng = random.Random(0); train, evals = [], {}
    for name, cls in G.GAMES.items():
        ex = game_frames(name, cls, 24 if cls.train else 6, rng)
        rng.shuffle(ex); n_ev = min(300, len(ex) // 5)
        if cls.train: train += ex[n_ev:]
        evals[f"frames_{name}"] = ex[:n_ev]
        print(f"[frames] {name:18s} train={'y' if cls.train else 'n'} {len(ex)} states", flush=True)
    mx = mario_frames(24, rng); rng.shuffle(mx); train += mx[300:]; evals["frames_mario"] = mx[:300]
    print(f"[frames] mario {len(mx)} states; total train {len(train)}, eval tasks {len(evals)}")
    pickle.dump((train, evals), open(sys.argv[1] if len(sys.argv) > 1 else "data/frames.pkl", "wb"))
