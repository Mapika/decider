"""PPO over the common game interface (text-state model or pixels-only vision model).
   python -m decider.games_rl --init runs/r7_v4/model --games pong,breakout,cliffwalking --out runs/rl_games
   python -m decider.games_rl --vision --init runs/v1_vision/model --games pong,breakout,cliffwalking --out runs/rl_games_vis"""
import argparse, json, os, random, time, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from . import games as G
from .infer import Example, Q
from .prompt import build

REWARD_SCALE = {"pong": 1.0, "breakout": 1.0, "cliffwalking": 10.0, "freeway": 1.0, "minigrid_empty": 1.0}


class NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


class Policy:
    """Uniform API over the text DecisionModel and the VisionDecisionModel."""
    def __init__(self, path, vision):
        self.vision = vision
        if vision:
            from .vision import VisionDecisionModel
            from .frames_data import VIS_INTRO
            self.m = VisionDecisionModel(path).cuda(); self.vis_intro = VIS_INTRO
        else:
            from .model import DecisionModel, collate
            self.m = DecisionModel(path).cuda(); self.collate = collate
        self.tok = self.m.tok
    def example(self, g):
        ctx = f"{g.intro} {self.vis_intro}" if self.vision else f"{g.intro}\n\nSituation: {g.text()}"
        return Example(ctx, [Q("What should you do right now?", list(g.options), 0)]), (g.frame() if self.vision else None)
    def make_item(self, ex, frame):
        if self.vision: return (frame, ex)
        return build(ex, self.tok, NoShuffle(), max_ctx_tokens=1024)
    def logits(self, items):
        if self.vision:
            inp = self.m.prepare(items); return self.m.slot_logits(inp)
        b = self.collate(items, self.tok.pad_token_id)
        return self.m.slot_logits(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["slot_idx"].cuda(), b["slot_batch"].cuda(), b["nopts"].cuda())
    def parameters(self): return self.m.parameters()
    def save(self, path):
        self.m.lm.save_pretrained(path); (self.m.proc if self.vision else self.tok).save_pretrained(path)


def rollout(pol, game_names, n_per_game, rng, max_t, greedy=False, seed0=0):
    envs = []
    for gi, name in enumerate(game_names):
        for j in range(n_per_game):
            g = G.GAMES[name](); g.reset(seed=seed0 + rng.randint(0, 10**6) if not greedy else seed0 + j)
            envs.append(dict(g=g, name=name, done=False, k=0, opt=g.options[0], items=[], acts=[], rews=[], logps=[], nopt=len(g.options), last=g.score()))
    pol.m.eval()
    while any(not e["done"] for e in envs):
        due = [e for e in envs if not e["done"] and e["k"] % e["g"].decide_every == 0]
        if due:
            items = []
            for e in due:
                ex, fr = pol.example(e["g"]); items.append(pol.make_item(ex, fr))
            with torch.no_grad():
                lg = pol.logits(items)
            for e, it, row in zip(due, items, lg):
                p = torch.softmax(row[:e["nopt"]], -1).cpu()
                a = int(p.argmax()) if greedy else int(torch.multinomial(p, 1))
                e["items"].append(it); e["acts"].append(a); e["logps"].append(float(torch.log(p[a] + 1e-12))); e["rews"].append(0.0); e["opt"] = e["g"].options[a]
        for e in envs:
            if e["done"]: continue
            _, done = e["g"].step(e["opt"]); e["k"] += 1
            sc = e["g"].score(); e["rews"][-1] += (sc - e["last"]) / REWARD_SCALE.get(e["name"], 1.0); e["last"] = sc
            if done or e["k"] >= max_t: e["done"] = True
    out = [dict(name=e["name"], items=e["items"], acts=e["acts"], rews=e["rews"], logps=e["logps"], score=e["g"].score(), nopt=e["nopt"]) for e in envs]
    for e in envs: e["g"].close()
    return out


def returns(rews, gamma):
    Gt, out = 0.0, []
    for r in reversed(rews): Gt = r + gamma * Gt; out.append(Gt)
    return out[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--out", required=True); ap.add_argument("--vision", action="store_true")
    ap.add_argument("--games", default="pong,breakout,cliffwalking"); ap.add_argument("--eval_games", default="pong,breakout,cliffwalking,freeway,minigrid_empty")
    ap.add_argument("--iters", type=int, default=30); ap.add_argument("--envs_per_game", type=int, default=16); ap.add_argument("--max_t", type=int, default=300)
    ap.add_argument("--lr", type=float, default=4e-6); ap.add_argument("--gamma", type=float, default=0.97); ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--ppo_steps", type=int, default=4); ap.add_argument("--clip", type=float, default=0.2); ap.add_argument("--kl_stop", type=float, default=0.05)
    ap.add_argument("--update_samples", type=int, default=6000); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--eval_every", type=int, default=5); ap.add_argument("--eval_episodes", type=int, default=2)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/rl.log", "a")
    def log(*s):
        m = " ".join(str(x) for x in s); print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    rng = random.Random(0); torch.manual_seed(0)
    G.RENDER = a.vision
    pol = Policy(a.init, a.vision); opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, betas=(0.9, 0.95))
    games = a.games.split(","); eval_games = a.eval_games.split(","); best = 0.0
    def evaluate(tag):
        res = {}
        for gname in eval_games:
            max_t = G.GAMES[gname].max_t if hasattr(G.GAMES[gname], "max_t") else 400
            trs = rollout(pol, [gname], a.eval_episodes, rng, max_t=max_t if isinstance(max_t, int) else 400, greedy=True)
            res[gname] = float(np.mean([t["score"] for t in trs]))
        log(f"[eval] {tag} greedy: " + " ".join(f"{k}={v:.2f}" for k, v in res.items())); return res
    base_res = evaluate("start")
    for it in range(1, a.iters + 1):
        t0 = time.time(); trajs = rollout(pol, games, a.envs_per_game, rng, a.max_t); t_roll = time.time() - t0
        # per-game per-step baselines
        items, acts, advs, oldlp, nopts = [], [], [], [], []
        for tr in trajs: tr["G"] = returns(tr["rews"], a.gamma)
        for gname in games:
            grp = [tr for tr in trajs if tr["name"] == gname]; T = max(len(tr["G"]) for tr in grp)
            base = [float(np.mean([tr["G"][t] for tr in grp if t < len(tr["G"])])) for t in range(T)]
            A = [tr["G"][t] - base[t] for tr in grp for t in range(len(tr["G"]))]; sd = float(np.std(A)) + 1e-6
            for tr in grp:
                for t in range(len(tr["G"])):
                    items.append(tr["items"][t]); acts.append(tr["acts"][t]); advs.append((tr["G"][t] - base[t]) / sd); oldlp.append(tr["logps"][t]); nopts.append(tr["nopt"])
        order = list(range(len(items))); rng.shuffle(order); order = order[:a.update_samples]
        pol.m.train(); chunks = [order[i::a.ppo_steps] for i in range(a.ppo_steps)]; kl_last = 0.0; pg_tot = 0.0; nb = 0; gn = 0.0
        for chunk in chunks:
            kls = []
            for i in range(0, len(chunk), a.batch):
                idx = chunk[i:i + a.batch]
                lg = pol.logits([items[j] for j in idx])
                logp = F.log_softmax(lg, -1); act = torch.tensor([acts[j] for j in idx], device="cuda"); A = torch.tensor([advs[j] for j in idx], device="cuda")
                lp = logp.gather(1, act[:, None]).squeeze(1); olp = torch.tensor([oldlp[j] for j in idx], device="cuda"); ratio = torch.exp(lp - olp)
                pg = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                valid = torch.isfinite(logp); lps = logp.masked_fill(~valid, 0.0)
                ent = -(lps.exp().masked_fill(~valid, 0.0) * lps).sum(1).mean()
                ((pg - a.entropy * ent) * len(idx) / len(chunk)).backward(); pg_tot += pg.item() * len(idx); nb += len(idx); kls.append(float((olp - lp).mean()))
            gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True); kl_last = float(np.mean(kls)) if kls else 0.0
            if kl_last > a.kl_stop: break
        per_game = {g: round(float(np.mean([tr["score"] for tr in trajs if tr["name"] == g])), 2) for g in games}
        log(f"[rl] iter {it}: sampled {per_game} decisions {len(items)} (used {len(order)}) pg {pg_tot/max(1,nb):.3f} kl {kl_last:.4f} gn {gn:.2f} rollout {t_roll:.0f}s total {time.time()-t0:.0f}s")
        if it % a.eval_every == 0 or it == a.iters:
            res = evaluate(f"iter {it}"); score = sum((res[g] - base_res[g]) / (abs(base_res[g]) + 1.0) for g in eval_games)
            if score > best:
                best = score; pol.save(f"{a.out}/model"); json.dump(res, open(f"{a.out}/best_eval.json", "w")); log(f"[save] best so far (relative gain {score:.2f}) -> {a.out}/model")


if __name__ == "__main__":
    main()
