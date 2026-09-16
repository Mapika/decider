"""Policy-gradient RL for Mario on the decision model: the softmax over action options IS the policy.
Rollouts: N emulators in parallel, one batched forward per decision step (every 4 frames, jumps held 16).
Update: REINFORCE with a moving baseline + entropy bonus on the answer-slot log-probs.
   python -m decider.mario_rl --init runs/r6_mario/model --out runs/rl_mario --iters 30"""
import argparse, json, math, os, random, time, numpy as np, torch, torch.nn.functional as F
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from .mario import describe, ACTIONS, RELEASE, INTRO
from .model import DecisionModel, collate
from .prompt import build
from .infer import Example, Q

OPTS = list(ACTIONS); NA = len(OPTS)
TRAIN_LEVELS = ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1", "5-1", "6-1"]
TEST_LEVELS = ["1-4", "2-2", "2-3", "3-2", "4-2", "7-1", "8-1"]


class NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


def make_env(level):
    return JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{level}-v0"), SIMPLE_MOVEMENT)


def prompt_item(tok, text):
    ex = Example(INTRO + "\n\nSituation: " + text, [Q("What should Mario do right now?", OPTS, 0)])
    return build(ex, tok, NoShuffle(), max_ctx_tokens=1024)


@torch.no_grad()
def policy_probs(model, items, temperature=1.0):
    b = collate(items, model.tok.pad_token_id)
    lg = model.slot_logits(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["slot_idx"].cuda(), b["slot_batch"].cuda(), b["nopts"].cuda())
    return torch.softmax(lg[:, :NA] / temperature, -1).cpu()


def rollout(model, levels, n_envs, rng, max_frames=1500, greedy=False, noop_max=30):
    """Returns per-env trajectories: list of (items, actions, rewards) and final x."""
    envs, states = [], []
    for i in range(n_envs):
        lv = levels[i % len(levels)]; e = make_env(lv); e.reset()
        for _ in range(rng.randint(0, noop_max)): e.step(0)
        states.append(dict(env=e, level=lv, done=False, frames=0, hold=0, release=0, act=0, x=int(e.unwrapped.ram[0x6D]) * 256 + int(e.unwrapped.ram[0x86]),
                           items=[], actions=[], rewards=[], last_x=None, stall=0, flag=False))
    model.eval()
    while any(not s["done"] for s in states):
        # decide for envs that are due
        due = [s for s in states if not s["done"] and s["frames"] % 4 == 0 and s["frames"] >= s["release"]]
        if due:
            items = [prompt_item(model.tok, describe(s["env"].unwrapped.ram, {})) for s in due]
            probs = policy_probs(model, items)
            for s, it, p in zip(due, items, probs):
                a = int(p.argmax()) if greedy else int(torch.multinomial(p, 1))
                s["items"].append(it); s["actions"].append(a); s["act"] = a
                s["rewards"].append(0.0)
                if "jump" in OPTS[a]:
                    s["hold"] = s["frames"] + 16; s["release"] = s["hold"] + 4
        for s in states:
            if s["done"]: continue
            name = OPTS[s["act"]]
            act = ACTIONS[name] if s["frames"] < s["hold"] or name not in RELEASE else RELEASE[name]
            _, _, done, info = s["env"].step(act); s["frames"] += 1
            x = info["x_pos"]; dx = x - s["x"]; s["x"] = x
            if s["rewards"]: s["rewards"][-1] += dx / 16.0           # tiles gained since the current decision
            if x == s["last_x"]: s["stall"] += 1
            else: s["stall"] = 0; s["last_x"] = x
            if info.get("flag_get"): s["flag"] = True; s["rewards"][-1] += 30.0; done = True
            if done and not s["flag"] and info.get("life", 2) < 2 or (done and not s["flag"]):
                s["rewards"][-1] -= 8.0                                 # died
            if s["frames"] >= max_frames or s["stall"] > 240: done = True
            if done: s["done"] = True; s["env"].close()
    return [dict(level=s["level"], items=s["items"], actions=s["actions"], rewards=s["rewards"], x=s["x"], flag=s["flag"]) for s in states]


def returns(rews, gamma):
    G, out = 0.0, []
    for r in reversed(rews):
        G = r + gamma * G; out.append(G)
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="runs/r6_mario/model")
    ap.add_argument("--out", default="runs/rl_mario")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--eval_every", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/rl.log", "a")
    def log(*s):
        m = " ".join(str(x) for x in s); print(m, flush=True); logf.write(m + "\n"); logf.flush()
    rng = random.Random(0); torch.manual_seed(0)
    model = DecisionModel(a.init).cuda(); tok = model.tok
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95))
    baseline = None; best = -1
    for it in range(1, a.iters + 1):
        t0 = time.time()
        trajs = rollout(model, TRAIN_LEVELS, a.envs, rng)
        t_roll = time.time() - t0
        # advantages
        items, acts, advs = [], [], []
        allG = []
        for tr in trajs:
            G = returns(tr["rewards"], a.gamma); allG.extend(G)
            for it_, ac, g in zip(tr["items"], tr["actions"], G):
                items.append(it_); acts.append(ac); advs.append(g)
        meanG = float(np.mean(allG)); stdG = float(np.std(allG)) + 1e-6
        baseline = meanG if baseline is None else 0.8 * baseline + 0.2 * meanG
        advs = [(g - baseline) / stdG for g in advs]
        # update: REINFORCE with entropy bonus, one pass over the rollout
        model.train(); order = list(range(len(items))); rng.shuffle(order)
        tot_loss, nb, cur, cur_tok = 0.0, 0, [], 0
        def flush(idx):
            nonlocal tot_loss, nb
            b = collate([items[i] for i in idx], tok.pad_token_id)
            lg = model.slot_logits(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["slot_idx"].cuda(), b["slot_batch"].cuda(), b["nopts"].cuda())[:, :NA]
            logp = F.log_softmax(lg, -1); A = torch.tensor([advs[i] for i in idx], device="cuda"); act = torch.tensor([acts[i] for i in idx], device="cuda")
            pg = -(A * logp.gather(1, act[:, None]).squeeze(1)).mean()
            ent = -(logp.exp() * logp).sum(1).mean()
            loss = (pg - a.entropy * ent) * len(idx) / len(items)
            loss.backward(); tot_loss += pg.item() * len(idx); nb += len(idx)
        for i in order:
            L = len(items[i]["ids"])
            if cur and (max(cur_tok, L) * (len(cur) + 1) > a.max_tokens):
                flush(cur); cur, cur_tok = [], 0
            cur.append(i); cur_tok = max(cur_tok, L)
        if cur: flush(cur)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        xs = [tr["x"] for tr in trajs]; flags = sum(tr["flag"] for tr in trajs)
        per_level = {lv: int(np.mean([tr["x"] for tr in trajs if tr["level"] == lv])) for lv in TRAIN_LEVELS}
        log(f"[rl] iter {it}: mean x {np.mean(xs):.0f} (max {max(xs)}) flags {flags}/{len(trajs)} decisions {len(items)} return {meanG:.2f} pg_loss {tot_loss/max(1,nb):.3f} gn {gn:.2f} rollout {t_roll:.0f}s total {time.time()-t0:.0f}s | {per_level}")
        if it % a.eval_every == 0 or it == a.iters:
            ev = rollout(model, TRAIN_LEVELS + TEST_LEVELS, len(TRAIN_LEVELS) + len(TEST_LEVELS), random.Random(1), greedy=True, noop_max=0)
            res = {tr["level"]: int(tr["x"]) for tr in ev}; tr_mean = np.mean([res[l] for l in TRAIN_LEVELS]); te_mean = np.mean([res[l] for l in TEST_LEVELS])
            log(f"[eval] iter {it} greedy: train levels mean {tr_mean:.0f}  unseen levels mean {te_mean:.0f}  {res}")
            json.dump(dict(iter=it, res=res), open(f"{a.out}/eval_{it}.json", "w"))
            if te_mean + tr_mean > best:
                best = te_mean + tr_mean; model.lm.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model"); log(f"[save] best so far -> {a.out}/model")


if __name__ == "__main__":
    main()
