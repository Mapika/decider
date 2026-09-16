"""Several games behind one interface: text state, option list, scripted teacher, step.
Atari (RAM -> text), gymnasium toy-text, MiniGrid/BabyAI.  Used for zero-shot evaluation
(model vs teacher vs random) and for teacher-labelled training data on the train games."""
import random, numpy as np
import gymnasium as gym

GAMES = {}
RENDER = False          # set True to create envs with render_mode="rgb_array" (frames via .frame())


def game(name, train=True):
    def deco(cls):
        cls.name = name; cls.train = train; GAMES[name] = cls; return cls
    return deco


class Game:
    def frame(self):
        try: return self.env.render()
        except Exception: return None
    """Subclass contract: reset(seed) -> None; text() -> str; options: list[str]; teacher() -> str;
    step(option) -> (reward, done); score() -> float (higher is better); intro: str; decide_every: int."""
    decide_every = 1
    def close(self): pass


# ------------------------------------------------------------------ Atari from RAM
class Atari(Game):
    env_id = None; hold = 1
    def __init__(self):
        import ale_py; gym.register_envs(ale_py)
        self.env = gym.make(self.env_id, obs_type="ram", frameskip=4, repeat_action_probability=0.0, **({"render_mode": "rgb_array"} if RENDER else {}))
    def reset(self, seed=0):
        self.ram, _ = self.env.reset(seed=seed); self.total = 0.0; self.t = 0; return None
    def _step(self, a):
        self.ram, r, term, trunc, _ = self.env.step(a); self.total += r; self.t += 1
        return r, term or trunc or self.t >= self.max_t
    def score(self): return self.total
    def close(self): self.env.close()


@game("pong")
class Pong(Atari):
    env_id = "ALE/Pong-v5"; max_t = 1500
    intro = "You play Pong (Atari) and control the right paddle. Move the paddle so the ball hits it; the ball bounces off paddles and walls. Missing the ball loses a point."
    options = ["move paddle up", "move paddle down", "stay"]
    ACT = {"move paddle up": 2, "move paddle down": 3, "stay": 0}
    def _s(self):
        r = self.ram; return dict(ball_x=int(r[49]), ball_y=int(r[54]), me_y=int(r[51]), opp_y=int(r[50]))
    def text(self):
        s = self._s(); d = s["ball_y"] - (s["me_y"] + 8)          # paddle centre
        vy = getattr(self, "_vy", 0); vx = getattr(self, "_vx", 0)
        return (f"Ball at x={s['ball_x']}, y={s['ball_y']}, moving {'down' if vy > 0 else 'up' if vy < 0 else 'level'} and {'toward you' if vx > 0 else 'away from you' if vx < 0 else 'sideways'}. "
                f"Your paddle centre is at y={s['me_y'] + 8}, {abs(d)} pixels {'below' if d < 0 else 'above'} the ball. Opponent paddle at y={s['opp_y'] + 8}.")
    def teacher(self):
        s = self._s(); d = s["ball_y"] - (s["me_y"] + 8)
        return "stay" if abs(d) < 6 else ("move paddle down" if d > 0 else "move paddle up")
    def step(self, opt):
        s0 = self._s(); r, done = self._step(self.ACT[opt]); s1 = self._s()
        self._vy = s1["ball_y"] - s0["ball_y"]; self._vx = s1["ball_x"] - s0["ball_x"]; return r, done


@game("freeway", train=False)
class Freeway(Atari):
    env_id = "ALE/Freeway-v5"; max_t = 700
    intro = "You play Freeway (Atari): guide the chicken up across ten lanes of traffic to the top of the screen. Cars move horizontally in each lane; being hit pushes the chicken back."
    options = ["move up", "wait", "move down"]
    ACT = {"move up": 1, "wait": 0, "move down": 2}
    def _lanes(self):
        r = self.ram; y = int(r[14]); cars = [int(r[108 + i]) for i in range(10)]
        return y, cars
    def text(self):
        y, cars = self._lanes(); lane = max(0, min(9, (y - 6) // 16 if y > 6 else -1))
        nxt = cars[lane + 1] if lane + 1 < 10 else None
        near = [i for i, cx in enumerate(cars) if abs(cx - 44) < 14]
        return (f"Chicken height {y} (0 = bottom, 175 = top), in lane {lane if lane >= 0 else 'start'}. Cars are at horizontal positions {cars} (chicken column is 44). "
                f"Lanes with a car about to cross the chicken's column: {near if near else 'none'}.")
    def teacher(self):
        y, cars = self._lanes(); lane = (y - 6) // 16 if y > 6 else -1
        for l in (lane + 1, lane + 2):
            if 0 <= l < 10 and abs(cars[l] - 44) < 18: return "wait"
        return "move up"
    def step(self, opt): return self._step(self.ACT[opt])


@game("breakout")
class Breakout(Atari):
    env_id = "ALE/Breakout-v5"; max_t = 1500
    intro = "You play Breakout (Atari): move the paddle at the bottom so the ball bounces up into the bricks. Missing the ball loses a life."
    options = ["move paddle left", "move paddle right", "stay", "launch ball"]
    ACT = {"move paddle left": 3, "move paddle right": 2, "stay": 0, "launch ball": 1}
    def _s(self):
        r = self.ram; return dict(ball_x=int(r[99]), ball_y=int(r[101]), paddle_x=int(r[72]), lives=int(r[57]))
    def text(self):
        s = self._s(); d = s["ball_x"] - s["paddle_x"]; vy = getattr(self, "_vy", 0)
        inplay = s["ball_y"] > 0
        return (f"{'Ball at x=' + str(s['ball_x']) + ', y=' + str(s['ball_y']) + ', moving ' + ('down' if vy > 0 else 'up') if inplay else 'No ball in play (press launch)'}. "
                f"Paddle at x={s['paddle_x']}, {abs(d)} pixels {'left' if d > 0 else 'right'} of the ball. Lives left: {s['lives']}.")
    def teacher(self):
        s = self._s()
        if s["ball_y"] == 0: return "launch ball"
        d = s["ball_x"] - s["paddle_x"]
        return "stay" if abs(d) < 5 else ("move paddle right" if d > 0 else "move paddle left")
    def step(self, opt):
        s0 = self._s(); r, done = self._step(self.ACT[opt]); s1 = self._s(); self._vy = s1["ball_y"] - s0["ball_y"]; return r, done


# ------------------------------------------------------------------ toy text
@game("frozenlake", train=False)
class FrozenLake(Game):
    intro = "You are on a frozen lake grid (4x4). S = start, F = frozen (safe), H = hole (falling in ends the game), G = goal. The ice is not slippery here. Reach G."
    options = ["move left", "move down", "move right", "move up"]
    def __init__(self): self.env = gym.make("FrozenLake-v1", is_slippery=False, **({"render_mode": "rgb_array"} if RENDER else {})); self.max_t = 40
    def reset(self, seed=0):
        self.s, _ = self.env.reset(seed=seed); self.desc = ["".join(ch.decode() for ch in row) for row in self.env.unwrapped.desc]; self.total = 0; self.t = 0
    def text(self):
        r, c = divmod(int(self.s), 4)
        rows = "\n".join("".join(("[" + ch + "]") if (i == r and j == c) else " " + ch + " " for j, ch in enumerate(row)) for i, row in enumerate(self.desc))
        return f"Grid (your position in brackets):\n{rows}\nYou are at row {r}, column {c}."
    def teacher(self):
        # BFS to goal avoiding holes
        from collections import deque
        r0, c0 = divmod(int(self.s), 4); q = deque([((r0, c0), None)]); seen = {(r0, c0)}
        moves = [(0, -1, "move left"), (1, 0, "move down"), (0, 1, "move right"), (-1, 0, "move up")]
        while q:
            (r, c), first = q.popleft()
            if self.desc[r][c] == "G": return first or "move down"
            for dr, dc, name in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4 and (nr, nc) not in seen and self.desc[nr][nc] != "H":
                    seen.add((nr, nc)); q.append(((nr, nc), first or name))
        return "move down"
    def step(self, opt):
        self.s, r, term, trunc, _ = self.env.step(self.options.index(opt)); self.total += r; self.t += 1
        return r, term or trunc or self.t >= self.max_t
    def score(self): return self.total


@game("cliffwalking")
class CliffWalking(Game):
    intro = "You walk on a 4x12 grid from the bottom-left start to the bottom-right goal. The bottom row between them is a cliff: stepping on it sends you back to the start with a big penalty. Every step costs 1."
    options = ["move up", "move right", "move down", "move left"]
    def __init__(self): self.env = gym.make("CliffWalking-v1", **({"render_mode": "rgb_array"} if RENDER else {})); self.max_t = 60
    def reset(self, seed=0): self.s, _ = self.env.reset(seed=seed); self.total = 0; self.t = 0
    def text(self):
        r, c = divmod(int(self.s), 12)
        return f"You are at row {r} (0 = top, 3 = bottom), column {c} (0 = left, 11 = right). The goal is at row 3, column 11. The cliff occupies row 3, columns 1 to 10."
    def teacher(self):
        r, c = divmod(int(self.s), 12)
        if r == 3 and c == 0: return "move up"
        if c < 11: return "move right"
        return "move down"
    def step(self, opt):
        self.s, r, term, trunc, _ = self.env.step(self.options.index(opt)); self.total += r; self.t += 1
        return r, term or trunc or self.t >= self.max_t
    def score(self): return self.total


@game("blackjack", train=False)
class Blackjack(Game):
    intro = "You play blackjack against a dealer. Get as close to 21 as possible without going over. An ace can count as 11 (usable) or 1. The dealer hits until reaching 17."
    options = ["hit (take another card)", "stick (stop)"]
    def __init__(self): self.env = gym.make("Blackjack-v1", **({"render_mode": "rgb_array"} if RENDER else {})); self.max_t = 10
    def reset(self, seed=0): self.s, _ = self.env.reset(seed=seed); self.total = 0; self.t = 0
    def text(self):
        p, d, ace = self.s
        return f"Your hand totals {p}{' with a usable ace' if ace else ''}. The dealer shows a {'ace' if d == 1 else d}."
    def teacher(self):
        p, d, ace = self.s
        if ace: return "hit (take another card)" if p < 18 else "stick (stop)"
        if p >= 17: return "stick (stop)"
        if 13 <= p <= 16: return "stick (stop)" if d <= 6 else "hit (take another card)"
        if p == 12: return "stick (stop)" if 4 <= d <= 6 else "hit (take another card)"
        return "hit (take another card)"
    def step(self, opt):
        self.s, r, term, trunc, _ = self.env.step(self.options.index(opt)); self.total += r; self.t += 1
        return r, term or trunc or self.t >= self.max_t
    def score(self): return self.total


# ------------------------------------------------------------------ MiniGrid / BabyAI
OBJ = {1: "empty", 2: "wall", 3: "floor", 4: "door", 5: "key", 6: "ball", 7: "box", 8: "goal", 9: "lava"}
COL = {0: "red", 1: "green", 2: "blue", 3: "purple", 4: "yellow", 5: "grey"}
DIRS = ["east", "south", "west", "north"]


class MiniGridGame(Game):
    env_id = None; max_t = 60
    options = ["turn left", "turn right", "move forward", "pick up", "drop", "toggle (open door / use)"]
    def __init__(self):
        import minigrid  # noqa: registers the envs
        self.env = gym.make(self.env_id, **({"render_mode": "rgb_array"} if RENDER else {})); self.decide_every = 1
    def reset(self, seed=0):
        self.obs, _ = self.env.reset(seed=seed); self.total = 0; self.t = 0; self.done_flag = False
    def text(self):
        img = self.obs["image"]; d = self.obs["direction"]
        # egocentric 7x7 view: agent at (3, 6) bottom-centre facing 'up' in the view
        items = []
        for i in range(7):
            for j in range(7):
                o, c, st = img[i, j]
                if o in (2, 4, 5, 6, 7, 8, 9):
                    ahead = 6 - j; side = i - 3
                    pos = (f"{ahead} ahead" if ahead else "here") + (f", {abs(side)} {'right' if side > 0 else 'left'}" if side else "")
                    name = OBJ[o] + ("" if o in (2, 8, 9) else f" ({COL.get(c, '?')})")
                    if o == 4: name += " [" + ("open" if st == 0 else "closed" if st == 1 else "locked") + "]"
                    items.append(f"{name} {pos}")
        carrying = getattr(self.env.unwrapped, "carrying", None)
        return (f"Mission: {self.obs['mission']}. You face {DIRS[d]}. Carrying: {carrying.type + ' (' + carrying.color + ')' if carrying else 'nothing'}. "
                f"In view (cells ahead / to the side): " + ("; ".join(items) if items else "nothing notable") + ".")
    def _front(self):
        return self.obs["image"][3, 5]
    def teacher(self):
        # greedy: head toward the mission target visible in view; open doors; else explore forward/turn
        img = self.obs["image"]; mission = self.obs["mission"]
        want = [o for o, n in OBJ.items() if n in mission and o in (5, 6, 7, 8, 4)]
        if "goal" in mission or "green goal" in mission: want = [8] + want
        target = None
        for i in range(7):
            for j in range(7):
                o, c, st = img[i, j]
                if o in want and (o == 8 or COL.get(c, "") in mission or o == 4):
                    if target is None or (6 - j) + abs(i - 3) < (6 - target[1]) + abs(target[0] - 3): target = (i, j)
        fo, fc, fs = self._front()
        if fo == 4 and fs != 0: return "toggle (open door / use)"
        if fo == 9 or fo == 2 or (fo == 4 and fs != 0):
            return "turn left" if img[2, 5][0] not in (2, 9) else "turn right"
        if target is not None:
            i, j = target
            if j == 5 and i == 3 and img[i, j][0] in (5, 6, 7) and "pick" in mission: return "pick up"
            if i == 3: return "move forward"
            return "turn right" if i > 3 else "turn left"
        return "move forward" if fo not in (2, 9) else "turn left"
    def step(self, opt):
        self.obs, r, term, trunc, _ = self.env.step(self.options.index(opt)); self.total += r; self.t += 1
        return r, term or trunc or self.t >= self.max_t
    def score(self): return self.total


@game("minigrid_empty")
class MGEmpty(MiniGridGame): env_id = "MiniGrid-Empty-8x8-v0"; intro = "You control an agent in a grid world seen from its own viewpoint (7x7 cells ahead and to the sides). Reach the green goal square. 'move forward' moves one cell; turning changes facing."
@game("minigrid_lavagap", train=False)
class MGLava(MiniGridGame): env_id = "MiniGrid-LavaGapS7-v0"; intro = "You control an agent in a grid world seen from its own viewpoint. A wall of lava with one gap separates you from the green goal; touching lava ends the episode. Reach the goal."
@game("minigrid_doorkey", train=False)
class MGDoorKey(MiniGridGame): env_id = "MiniGrid-DoorKey-8x8-v0"; intro = "You control an agent in a grid world seen from its own viewpoint. Pick up the key, use it to open the locked door (toggle while facing it), then reach the green goal."
@game("babyai_goto", train=False)
class BabyGoTo(MiniGridGame): env_id = "BabyAI-GoToObj-v0"; intro = "You control an agent in a grid world seen from its own viewpoint. Complete the mission by moving next to the named object."


# ------------------------------------------------------------------ runner
def play(g, policy, seed, rng=None):
    """policy(text, options, game) -> option string. Returns score."""
    g.reset(seed); done = False; k = 0; opt = g.options[0]
    while not done:
        if k % g.decide_every == 0: opt = policy(g.text(), g.options, g)
        _, done = g.step(opt); k += 1
    return g.score()


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(GAMES)
    for n in names:
        g = GAMES[n]()
        rng = random.Random(0)
        t = np.mean([play(g, lambda txt, opts, gm: gm.teacher(), s) for s in range(5)])
        r = np.mean([play(g, lambda txt, opts, gm: rng.choice(opts), s) for s in range(5)])
        g.reset(0); print(f"{n:18s} train={g.train} teacher {t:8.2f}  random {r:8.2f}   | {g.text()[:150]}")
        g.close()
