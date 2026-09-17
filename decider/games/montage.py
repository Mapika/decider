"""GIF montage: the model playing every game (+ Mario) side by side.  python -m decider.games.montage runs/r7_v4/model runs/rl_mario4/model media/montage.gif"""
import os, sys, random, numpy as np
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from PIL import Image, ImageDraw
from decider.games import envs as G
G.RENDER = True
from decider.infer import Decider
model_path, mario_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
TILE = (224, 168); FPS = 10; SECONDS = 24; N = FPS * SECONDS
dec = Decider(model_path)

def fit(img):
    im = Image.fromarray(np.asarray(img)).convert("RGB"); im.thumbnail((TILE[0], TILE[1] - 16))
    canvas = Image.new("RGB", TILE, (18, 18, 18)); canvas.paste(im, ((TILE[0] - im.width) // 2, 16 + (TILE[1] - 16 - im.height) // 2)); return canvas

def label(im, text):
    d = ImageDraw.Draw(im); d.rectangle([0, 0, TILE[0], 16], fill=(35, 35, 35)); d.text((4, 2), text, fill=(240, 240, 240)); return im

tiles = {}
for name, cls in G.GAMES.items():
    g = cls(); frames = []; g.reset(0); done = False; k = 0; opt = g.options[0]
    # step budget so the clip is ~N frames: every step for slow games, subsample fast ones
    every = 1 if name.startswith(("minigrid", "babyai", "frozenlake", "cliff", "blackjack")) else 2
    while not done and len(frames) < N * every:
        if k % g.decide_every == 0:
            r = dec.decide(f"{g.intro}\n\nSituation: {g.text()}", [{"question": "What should you do right now?", "options": list(g.options)}])[0]; opt = r["choice"]
        f = g.frame()
        if f is not None and k % every == 0: frames.append(fit(f))
        _, done = g.step(opt); k += 1
    f = g.frame()
    if f is not None: frames.append(fit(f))
    score = g.score(); g.close()
    tiles[name] = [label(im.copy(), f"{name}  score {score:.0f}" if abs(score) >= 10 else f"{name}  score {score:.2f}") for im in frames]
    print(f"{name}: {len(frames)} frames, score {score}", flush=True)

# Mario with the RL checkpoint
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from decider.games.mario import describe, ACTIONS, RELEASE, INTRO
mdec = Decider(mario_path)
env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT); obs = env.reset(); ram = env.unwrapped.ram; info = {"x_pos": 40}
frames = []; steps = 0; hold = 0; rel = 0; name = "run right"; done = False
schema = {"What should Mario do right now?": {"type": "choice", "options": list(ACTIONS)}}
while not done and steps < N * 3:
    if steps % 4 == 0 and steps >= rel:
        name = mdec.decide_json(INTRO + "\n\nSituation: " + describe(ram, info), schema)["What should Mario do right now?"]["choice"]
        if "jump" in name: hold = steps + 16; rel = hold + 4
    act = ACTIONS[name] if steps < hold or name not in RELEASE else RELEASE[name]
    obs, _, done, info = env.step(act); ram = env.unwrapped.ram; steps += 1
    if steps % 3 == 0: frames.append(fit(obs))
tiles["mario (RL)"] = [label(im.copy(), f"mario (RL)  x={info['x_pos']}") for im in frames]; print("mario:", len(frames), "frames, x", info["x_pos"])

names = list(tiles); cols = 4; rows = (len(names) + cols - 1) // cols
W, H = cols * TILE[0], rows * TILE[1]
out_frames = []
for t in range(N):
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    for i, n in enumerate(names):
        fr = tiles[n]; im = fr[min(t, len(fr) - 1)]
        canvas.paste(im, ((i % cols) * TILE[0], (i // cols) * TILE[1]))
    out_frames.append(canvas)
out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
print("wrote", out, W, "x", H, len(out_frames), "frames", os.path.getsize(out) // 1024, "KB")
