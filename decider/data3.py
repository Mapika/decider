"""Mixture v4 additions: situation -> action.
  agenttraj : AgentGym/AgentTraj-L (ALFWorld, BabyAI, WebShop, SciWorld, ...) -> next action among candidates
  mind2web  : osunlp/Mind2Web -> which page element to act on, among candidates
  synth     : data/synth.jsonl from the 27B teacher (situation, options, best action, danger)
  games     : teacher-labelled states from the train games in decider.games (+ Mario states)"""
import json, os, random, re
from collections import defaultdict
from .data import task, Example, Q, _ld, SEED, TASKS, TRAIN_CAP, EVAL_CAP


def _clip(s, n): return s if len(s) <= n else s[:n] + " ..."


@task("agenttraj")
def _agenttraj():
    from datasets import load_dataset
    ds = load_dataset("AgentGym/AgentTraj-L", split="train")
    rng = random.Random(SEED); out = []
    rx = re.compile(r"Action:\s*(.+?)\s*$", re.S)
    pool = defaultdict(list)                                   # env prefix -> actions (distractor pool)
    trajs = []
    for r in ds:
        env = r["item_id"].split("_")[0]; conv = r["conversations"]
        steps = []
        for i, m in enumerate(conv):
            if m["from"] == "gpt":
                mm = rx.search(m["value"])
                if mm:
                    act = mm.group(1).strip().split("\n")[0][:120]
                    steps.append((i, act)); pool[env].append(act)
        if len(steps) >= 2:
            trajs.append((env, conv, steps))
    rng.shuffle(trajs)
    for env, conv, steps in trajs:
        task_txt = _clip(conv[2]["value"] if len(conv) > 2 and conv[2]["from"] == "human" else conv[0]["value"], 1500)
        own = [a for _, a in steps]
        for k, (i, act) in enumerate(steps):
            if rng.random() > 0.35: continue                    # subsample steps
            hist = [m for m in conv[:i] if m["from"] == "human"][-2:]
            obs = "\n".join(_clip(m["value"], 700) for m in hist[-1:])
            prev = [a for _, a in steps[:k]][-3:]
            ctx = f"Environment: {env}.\nTask:\n{task_txt}\n\nRecent actions: {', '.join(prev) if prev else 'none'}\nLatest observation:\n{obs}"
            distract = list(dict.fromkeys(a for a in own if a != act))
            rng.shuffle(distract); cands = distract[:3]
            others = list(dict.fromkeys(a for a in pool[env] if a != act and a not in cands))
            while len(cands) < 4 and others:
                cands.append(others.pop(rng.randrange(len(others))))
            opts = [act] + cands; rng.shuffle(opts)
            out.append(Example(ctx, [Q("Which action should the agent take next?", opts, opts.index(act))], "agenttraj"))
        if len(out) >= TRAIN_CAP + EVAL_CAP:
            break
    return out[EVAL_CAP:], out[:EVAL_CAP]


def _elem_desc(c):
    try:
        at = json.loads(c["attributes"])
    except Exception:
        at = {}
    bits = [c.get("tag", "")]
    for k in ("aria_label", "aria-label", "title", "alt", "placeholder", "name", "value", "type", "role", "id", "class"):
        if at.get(k): bits.append(f"{k}={str(at[k])[:40]}")
    return " ".join(bits)[:120]


@task("mind2web")
def _m2w():
    from datasets import load_dataset
    ds = load_dataset("osunlp/Mind2Web", split="train")
    rng = random.Random(SEED); out = []
    for r in ds:
        prev = []
        for a, rep in zip(r["actions"], r["action_reprs"]):
            if a["pos_candidates"]:
                pos = _elem_desc(a["pos_candidates"][0]); negs = [_elem_desc(c) for c in a["neg_candidates"]]
                negs = [n for n in negs if n != pos]; rng.shuffle(negs); opts = [pos] + negs[:5]; rng.shuffle(opts)
                op = a["operation"]["op"]; val = a["operation"].get("value") or ""
                ctx = f"Website: {r['website']} ({r['domain']}).\nTask: {r['confirmed_task']}\nActions so far: {'; '.join(prev[-4:]) if prev else 'none'}\nNext operation: {op}{' ' + repr(val) if val else ''}"
                out.append(Example(ctx, [Q("Which page element should the next operation target?", opts, opts.index(pos))], "mind2web"))
            prev.append(rep)
        if len(out) >= TRAIN_CAP + EVAL_CAP:
            break
    rng.shuffle(out)
    return out[EVAL_CAP:], out[:EVAL_CAP]


@task("synth")
def _synth():
    path = "data/synth.jsonl"
    if not os.path.exists(path):
        return [], []
    rng = random.Random(SEED); out = []
    for line in open(path):
        try:
            j = json.loads(line)
        except Exception:
            continue
        qs = [Q(j["question"], list(j["options"]), int(j["answer"]))]
        if "danger" in j: qs.append(Q("Is the agent in immediate danger?", ["no", "yes"], int(bool(j["danger"]))))
        out.append(Example(f"Setting: {j.get('domain', '')}.\n{j['situation']}", qs, "synth"))
    rng.shuffle(out); n_ev = min(500, len(out) // 10)
    return out[n_ev:], out[:n_ev]


@task("games")
def _games():
    """Teacher-labelled states from the train games (random-action noise for coverage)."""
    from . import games as G
    rng = random.Random(SEED); out = []
    for name, cls in G.GAMES.items():
        if not cls.train:
            continue
        g = cls(); n0 = len(out)
        for ep in range(30):
            g.reset(seed=1000 + ep); done = False; k = 0; eps = [0.0, 0.1, 0.25][ep % 3]; opt = g.options[0]
            while not done and k < 400:
                if k % g.decide_every == 0:
                    txt = g.text(); lab = g.teacher()
                    if rng.random() < 0.5:
                        out.append(Example(f"{cls.intro}\n\nSituation: {txt}", [Q("What should you do right now?", list(g.options), g.options.index(lab))], "games"))
                    opt = rng.choice(g.options) if rng.random() < eps else lab
                _, done = g.step(opt); k += 1
        g.close(); print(f"[games] {name}: {len(out) - n0} states", flush=True)
    rng.shuffle(out); n_ev = min(500, len(out) // 10)
    return out[n_ev:], out[:n_ev]


@task("offtopic_probe", heldout=True)
def _offtopic():
    return [], []       # built into data/tasks_v4.pkl: held-out tasks with an abstain option; half have off-topic option lists (abstain correct)


NEW_TASKS = ["agenttraj", "mind2web", "synth", "games"]
