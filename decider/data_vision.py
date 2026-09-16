"""Image decision tasks from HuggingFaceM4/the_cauldron (multiple-choice and yes/no subsets) -> Examples with PNG bytes.
   python -m decider.data_vision data/cauldron.pkl [cap_per_subset]"""
import io, pickle, random, re, sys
from datasets import load_dataset
from .data import Example, Q

SUBSETS = {  # name: (held_out, question_template)
    "aokvqa": (False, None), "ai2d": (False, None), "scienceqa": (False, None), "iconqa": (False, None), "tqa": (False, None), "raven": (False, None),
    "nlvr2": (False, None), "hateful_memes": (False, None), "visual7w": (True, None), "vsr": (True, None),
}
LETTER_RX = re.compile(r"^([A-J])\.\s*(.+)$")


def png(im):
    b = io.BytesIO(); im.convert("RGB").save(b, format="PNG"); return b.getvalue()


def parse(sub, user, assistant):
    """Return (context, question, options, gold) or None."""
    u = user.strip(); a = assistant.strip()
    if sub in ("vsr", "nlvr2", "hateful_memes"):
        q = u.split("\n")[0].strip(); g = 1 if a.lower().startswith("yes") else 0 if a.lower().startswith("no") else None
        if g is None: return None
        return "", q, ["no", "yes"], g
    if sub == "aokvqa":
        m = re.search(r"Options:\s*(.+)$", u, re.S)
        if not m: return None
        opts = [o.strip().rstrip(".") for o in m.group(1).split(",") if o.strip()]
        q = u.split("\n")[0].strip(); ans = a.rstrip(".").strip().lower()
        golds = [i for i, o in enumerate(opts) if o.lower() == ans]
        if len(opts) < 2 or not golds: return None
        return "", q, opts, golds[0]
    if sub == "raven":
        # options are in the image; the answer is a letter A-H
        m = re.match(r"^([A-H])", a)
        if not m: return None
        opts = list("ABCDEFGH"); return "", u.split("\n")[0].strip(), [f"figure {L}" for L in opts], opts.index(m.group(1))
    # letter-choice subsets: "Question: ...\nChoices:\nA. x\nB. y\n...\nAnswer with the letter." / "Answer: B"
    lines = [l.strip() for l in u.split("\n")]
    opts, letters = [], []
    for l in lines:
        m = LETTER_RX.match(l)
        if m: letters.append(m.group(1)); opts.append(m.group(2).strip().rstrip("."))
    m = re.search(r"Answer:\s*([A-J])", a)
    if len(opts) < 2 or not m or m.group(1) not in letters: return None
    ctx_lines = [l for l in lines if not LETTER_RX.match(l) and l not in ("Choices:", "Answer with the letter.") and l]
    q = next((l[len("Question:"):].strip() for l in ctx_lines if l.startswith("Question:")), ctx_lines[-1] if ctx_lines else "Which option is correct?")
    ctx = "\n".join(l for l in ctx_lines if not l.startswith("Question:"))
    return ctx[:2000], q, opts, letters.index(m.group(1))


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/cauldron.pkl"; cap = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
    rng = random.Random(0); train, evals = [], {}
    for sub, (held, _) in SUBSETS.items():
        ds = load_dataset("HuggingFaceM4/the_cauldron", sub, split="train", streaming=True)
        exs = []; bad = 0
        for r in ds:
            if len(r["images"]) != 1 or not r["texts"]: bad += 1; continue
            t = r["texts"][0]; p = parse(sub, t["user"], t["assistant"])
            if p is None: bad += 1; continue
            ctx, q, opts, g = p
            im = r["images"][0]
            if max(im.size) > 768: im = im.copy(); im.thumbnail((768, 768))
            exs.append(Example(("This is a visual question about the image." + ("\n" + ctx if ctx else "")), [Q(q, opts, g)], f"vis_{sub}", image=png(im)))
            if len(exs) >= (cap if not held else 1500): break
        rng.shuffle(exs); n_ev = min(500, len(exs) // 5)
        if not held: train += exs[n_ev:]
        evals[f"vis_{sub}"] = exs[:n_ev]
        print(f"[cauldron] {sub:14s} held={held} kept {len(exs)} (rejected {bad}) train {0 if held else len(exs)-n_ev} eval {n_ev}", flush=True)
    print(f"[cauldron] total train {len(train)}, eval tasks {len(evals)}")
    pickle.dump((train, evals), open(out_path, "wb"))
