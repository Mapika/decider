"""Super Mario Bros controlled by typed decisions: emulator RAM -> short text state -> one choice field.
   python -m decider.games.mario [model] [episodes] [--video out.mp4] [--policy model|heuristic|spam]"""
import sys, time, argparse, numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

ACTIONS = {                      # option text -> SIMPLE_MOVEMENT index
    "run right": 3, "jump right": 2, "run and jump right (long jump)": 4, "jump straight up": 5, "step left": 6, "wait": 0,
}
RELEASE = {"jump right": 3, "run and jump right (long jump)": 3, "jump straight up": 0}   # same motion with A released
ENEMY = {0x00: "green koopa", 0x01: "red koopa", 0x02: "buzzy beetle", 0x03: "hammer bro", 0x04: "goomba", 0x05: "blooper",
         0x06: "goomba", 0x07: "goomba", 0x0D: "piranha plant", 0x11: "bullet bill", 0x12: "spiny", 0x2E: "power-up"}
INTRO = ("You control Mario in Super Mario Bros. The goal is to travel right as far as possible without dying. "
         "Mario dies if he falls into a gap or touches an enemy from the side; he can jump over gaps, pipes, blocks and enemies, "
         "and a long jump covers wide gaps. A tall pipe (3 or more blocks) needs a running start: step left a few tiles, then run and "
         "long-jump at it. Jumping straight up or waiting only helps when something must pass first.")


def tile(ram, x, y):
    """Solid tile at world pixel (x, y)? Level layout lives at 0x0500 as two 16x13 screens."""
    if y < 32 or y >= 240: return 0
    page = (x // 256) % 2; sx = (x % 256) // 16; sy = (y - 32) // 16
    return int(ram[0x0500 + page * 208 + sy * 16 + sx])


def describe(ram, info):
    mx = int(ram[0x6D]) * 256 + int(ram[0x86]); my = int(ram[0x03B8])          # ram y = sprite top - 16 (standing: 176, ground surface at 208)
    vx = int(np.int8(ram[0x57]))
    feet = my + 32                                                             # first row below Mario's feet
    # ground / gaps / obstacles in the next 10 tiles
    cols = []
    for d in range(1, 11):
        x = mx + d * 16
        ground = any(tile(ram, x, y) for y in range(feet, 240, 16))
        obst_h = 0
        for h in range(1, 6):                      # solid blocks contiguous from the ground up (pipes, stairs, walls)
            if tile(ram, x, feet - h * 16):
                obst_h = h
            else:
                break
        overhead = any(tile(ram, x, feet - h * 16) for h in range(3, 6)) and obst_h < 3
        cols.append((d, ground, obst_h, overhead))
    gaps = [d for d, g, _, _ in cols if not g]
    walls = [(d, h) for d, g, h, _ in cols if h]
    over = [d for d, _, _, o in cols if o]
    parts = []
    parts.append(f"Mario is {'standing' if abs(vx) < 2 else 'running right' if vx > 0 else 'moving left'}"
                 f"{' in the air' if not any(tile(ram, mx + dx, feet) for dx in (2, 14)) else ' on the ground'}.")
    if gaps:
        run = []; s = gaps[0]; p = s
        for d in gaps[1:]:
            if d == p + 1: p = d
            else: run.append((s, p)); s = p = d
        run.append((s, p))
        parts.append("Gaps ahead: " + ", ".join(f"{a} tile{'s' if a > 1 else ''} away, {b - a + 1} wide" for a, b in run) + ".")
    else:
        parts.append("Solid ground for the next 10 tiles.")
    if walls:
        d, h = walls[0]; parts.append(f"Obstacle {d} tile{'s' if d > 1 else ''} ahead, {h} block{'s' if h > 1 else ''} high" + (" (pipe or wall)" if h >= 2 else " (step)") + ".")
    if over:
        parts.append(f"Floating blocks overhead {over[0]} tile{'s' if over[0] > 1 else ''} ahead (Mario can run under them).")
    ens = []
    for i in range(5):
        if ram[0x0F + i]:
            ex = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]); ey = int(ram[0xCF + i]); dx = (ex - mx) // 16
            if -3 <= dx <= 12:
                name = ENEMY.get(int(ram[0x16 + i]), "enemy")
                pos = f"{abs(dx)} tile{'s' if abs(dx) != 1 else ''} {'ahead' if dx > 0 else 'behind' if dx < 0 else 'right here'}"
                lvl = "above Mario" if ey < my - 8 else "below Mario" if ey > my + 24 else "at Mario's level"
                ens.append(f"{name} {pos}, {lvl}")
    parts.append(("Enemies: " + "; ".join(ens) + ".") if ens else "No enemies nearby.")
    return " ".join(parts)


def heuristic(ram, info):
    """Rule-based teacher on the same RAM features (also the baseline)."""
    mx = int(ram[0x6D]) * 256 + int(ram[0x86]); my = int(ram[0x03B8]); feet = my + 32; vx = int(np.int8(ram[0x57]))
    on_ground = any(tile(ram, mx + dx, feet) for dx in (2, 14))
    if not on_ground:
        return "run right"                                   # inputs mid-air do nothing useful; keep momentum
    for d in (1, 2, 3):
        x = mx + d * 16
        if not any(tile(ram, x, y) for y in range(feet, 240, 16)): return "run and jump right (long jump)"
        if tile(ram, x, feet - 16):
            tall = tile(ram, x, feet - 32) and tile(ram, x, feet - 48)
            if tall and vx < 24: return "step left"          # need a running start
            return "run and jump right (long jump)" if tile(ram, x, feet - 32) else "jump right"
    for d in (4, 5):
        x = mx + d * 16
        if all(tile(ram, x, feet - h * 16) for h in (1, 2, 3)) and vx >= 24: return "run and jump right (long jump)"   # tall pipe/wall, contiguous from ground
    for i in range(5):
        if ram[0x0F + i]:
            ex = int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]); ey = int(ram[0xCF + i]); dx = (ex - mx) // 16
            if ey > my + 24:                                  # enemy below: Mario is on a ledge and about to drop onto it
                if 1 <= dx <= 5: return "run and jump right (long jump)"
                continue
            if ey < my - 8: continue                         # enemy above (on blocks): run under it
            if 2 <= dx <= 4: return "run and jump right (long jump)"
            if 0 <= dx <= 1: return "jump right"
    return "run right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="runs/r3_v2/model")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--policy", default="model", choices=["model", "heuristic", "spam"])
    ap.add_argument("--video", default="")
    ap.add_argument("--every", type=int, default=4, help="frames per decision")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--trace", type=int, default=-1, help="print every decision once x_pos exceeds this")
    ap.add_argument("--level", default="1-1", help="world-stage, e.g. 1-1, 2-2, 8-1")
    ap.add_argument("--noop_start", type=int, default=0, help="random no-op frames at episode start (desyncs the deterministic emulator)")
    a = ap.parse_args()
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{a.level}-v0"), SIMPLE_MOVEMENT)
    import random as _r; _rng = _r.Random(0)
    dec = None
    if a.policy == "model":
        from decider.infer import Decider
        dec = Decider(a.model)
    schema = {"What should Mario do right now?": {"type": "choice", "options": list(ACTIONS)},
              "Is Mario in immediate danger?": {"type": "bool"}}
    frames = []; results = []
    for ep in range(a.episodes):
        obs = env.reset(); ram = env.unwrapped.ram; info = {"x_pos": 40}
        for _ in range(_rng.randint(0, a.noop_start)):
            obs, _, _, info = env.step(0)
        ram = env.unwrapped.ram
        done = False; steps = 0; t_dec = []; last_text = ""; last_out = None; action_name = "run right"; hold_until = 0; release_until = 0
        while not done and steps < a.max_steps:
            if steps % a.every == 0 and steps >= release_until:
                text = describe(ram, info)
                if a.policy == "model":
                    t = time.time(); out = dec.decide_json(INTRO + "\n\nSituation: " + text, schema); t_dec.append(time.time() - t)
                    action_name = out["What should Mario do right now?"]["choice"]; last_out = out
                elif a.policy == "heuristic":
                    action_name = heuristic(ram, info)
                else:
                    action_name = "run and jump right (long jump)" if (steps // 8) % 2 == 0 else "run right"
                last_text = text
                if a.trace >= 0 and info["x_pos"] > a.trace:
                    print(f"  [{steps}] x={info['x_pos']} y={int(ram[0x03B8])} vx={int(np.int8(ram[0x57]))} -> {action_name:32s} | {text[:120]}", flush=True)
                if "jump" in action_name:
                    hold_until = steps + 16                # a full jump needs the button held ~16 frames
                    release_until = hold_until + 4         # ... and released before the next jump registers
            act = ACTIONS[action_name] if steps < hold_until or action_name not in RELEASE else RELEASE[action_name]
            obs, r, done, info = env.step(act)
            ram = env.unwrapped.ram; steps += 1
            if a.video and steps % 2 == 0:
                frames.append(obs.copy())
            if steps % 200 == 0 and a.policy == "model":
                print(f"  step {steps} x={info['x_pos']} | {last_text[:110]} -> {action_name} ({last_out['What should Mario do right now?']['confidence']:.2f}) danger={last_out['Is Mario in immediate danger?']['noul']:.2f}", flush=True)
        results.append(info["x_pos"])
        print(f"episode {ep}: x_pos={info['x_pos']} steps={steps} flag={info.get('flag_get')} life={info.get('life')}" +
              (f"  decision p50 {np.median(t_dec)*1000:.1f} ms" if t_dec else ""), flush=True)
    print(f"policy={a.policy} level={a.level}: mean distance {np.mean(results):.0f} px over {a.episodes} episodes")
    if a.video and frames:
        import imageio
        imageio.mimwrite(a.video, frames, fps=30, quality=7); print("video:", a.video, len(frames), "frames")


if __name__ == "__main__":
    main()
