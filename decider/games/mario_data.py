"""Label game states with the scripted policy (plus random-action noise for coverage) -> decider Examples.
   python -m decider.games.mario_data data/mario.pkl [episodes_per_level]"""
import sys, random, pickle, numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from decider.games.mario import describe, heuristic, tile, ACTIONS, RELEASE, INTRO
from decider.data import Example, Q, load_cache

LEVELS = ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1", "5-1", "6-1"]
OPTS = list(ACTIONS)


def danger(ram):
    mx = int(ram[0x6D]) * 256 + int(ram[0x86]); my = int(ram[0x03B8]); feet = my + 32
    for d in (1, 2, 3):
        if not any(tile(ram, mx + d * 16, y) for y in range(feet, 240, 16)): return 1
    for i in range(5):
        if ram[0x0F + i]:
            dx = (int(ram[0x6E + i]) * 256 + int(ram[0x87 + i]) - mx) // 16
            if -1 <= dx <= 3: return 1
    return 0


def run_level(level, episodes, eps, rng, out):
    env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"), SIMPLE_MOVEMENT)
    for ep in range(episodes):
        env.reset(); ram = env.unwrapped.ram; info = {"x_pos": 40}; done = False; steps = 0; hold_until = 0; release_until = 0; act_name = "run right"; last_x = 0; stall = 0
        while not done and steps < 2500:
            if steps % 4 == 0 and steps >= release_until:
                text = describe(ram, info); label = heuristic(ram, info)
                out.append(Example(INTRO + "\n\nSituation: " + text, [Q("What should Mario do right now?", OPTS, OPTS.index(label)),
                                                                     Q("Is Mario in immediate danger?", ["no", "yes"], danger(ram))], "mario"))
                act_name = rng.choice(OPTS) if rng.random() < eps else label
                if "jump" in act_name: hold_until = steps + 16; release_until = hold_until + 4
            act = ACTIONS[act_name] if steps < hold_until or act_name not in RELEASE else RELEASE[act_name]
            obs, r, done, info = env.step(act); ram = env.unwrapped.ram; steps += 1
            if info["x_pos"] == last_x: stall += 1
            else: stall = 0; last_x = info["x_pos"]
            if stall > 300: break
        print(f"  {level} ep{ep} eps={eps:.2f}: x={info['x_pos']} states so far {len(out)}", flush=True)
    env.close()


if __name__ == "__main__":
    path = sys.argv[1]; per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = random.Random(0); train, held = [], []
    for lv in LEVELS:
        for ep in range(per):
            run_level(lv, 1, [0.0, 0.1, 0.2, 0.3, 0.15][ep % 5], rng, held if (lv == "1-1" and ep == per - 1) else train)
    from collections import Counter
    print("train", len(train), "held", len(held), "action dist", Counter(e.qs[0].options[e.qs[0].gold] for e in train).most_common(), "danger rate", np.mean([e.qs[1].gold for e in train]))
    # oversample the rare, decisive actions (jumps) x4 and cap the run-up 'step left' states
    jumps = [e for e in train if "jump" in e.qs[0].options[e.qs[0].gold]]
    lefts = [e for e in train if e.qs[0].options[e.qs[0].gold] == "step left"]; rng.shuffle(lefts)
    rest = [e for e in train if e not in jumps and e not in lefts]
    train = rest + jumps * 4 + lefts[:len(jumps)]
    print("after balancing:", len(train), Counter(e.qs[0].options[e.qs[0].gold] for e in train).most_common())
    # replay of general data so the fine-tune does not forget the general decider
    gen_train, gen_evals = load_cache("data/tasks_v2.pkl")
    rng.shuffle(gen_train); replay = gen_train[:2 * len(train)]
    evals = {"mario": held[:1500], "clinc_oos": gen_evals["clinc_oos"][:300], "support_tickets": gen_evals["support_tickets"][:300], "abstain_probe": gen_evals["abstain_probe"][:300]}
    pickle.dump((train + replay, evals), open(path, "wb"))
    print("saved", path, "train", len(train) + len(replay), "(mario", len(train), "+ replay", len(replay), ")")
