"""Stage 1: proper-scoring-rule fine-tune (CE, optionally + Brier) on the multi-task decision mixture."""
import argparse, json, math, os, pickle, random, time
import numpy as np, torch, torch.nn.functional as F
from .model import DecisionModel, collate
from .prompt import build
from .evaluate import run_eval, aggregate
from . import data as D
from . import data2  # noqa: F401  (registers v2 tasks)
from . import data3  # noqa: F401  (registers v4 tasks)


NONE_OPT = "none of the above"
ABSTAIN_WORDINGS = ["none of the above", "none of these", "not listed here", "other", "something else", "unsure",
                    "does not apply", "no suitable option", "neither of these", "cannot tell from the text",
                    "none of the above (out of scope)", "other / not covered"]
NONE_GOLD_RATE = 0.25


from .prompt import is_abstain_option


_LABEL_POOL = None


def _label_pool(rng):
    """Option strings from many tasks, used to build clearly off-topic option lists."""
    global _LABEL_POOL
    if _LABEL_POOL is None:
        _LABEL_POOL = {}
    return _LABEL_POOL


def none_augment(e, rng, p, pool=None):
    """With prob p, for a question with >=3 options and no abstain-style option:
       75%: add an abstain option (random wording), gold unchanged  -> "an abstain option present does not mean abstain"
       25%: replace ALL options by labels drawn from other tasks + an abstain option, gold = abstain
            -> abstain when nothing on offer fits, not when the exact label is merely missing."""
    if p <= 0 or pool is None:
        return e
    qs = []
    for q in e.qs:
        if len(q.options) >= 3 and rng.random() < p and not any(is_abstain_option(o) for o in q.options):
            w = rng.choice(ABSTAIN_WORDINGS)
            if rng.random() < NONE_GOLD_RATE:
                others = [t for t in pool if t != e.task]
                src = pool[rng.choice(others)]
                k = min(len(q.options), len(src)); opts = rng.sample(src, k) + [w]
                qs.append(D.Q(q.text, opts, len(opts) - 1))
            else:
                qs.append(D.Q(q.text, list(q.options) + [w], q.gold))
        else:
            qs.append(q)
    return D.Example(e.context, qs, e.task)


def build_label_pool(train):
    """task -> sorted distinct option strings (only tasks with a fixed label set of 3..60 options)."""
    pool = {}
    for e in train:
        for q in e.qs:
            if 3 <= len(q.options) <= 60:
                pool.setdefault(e.task, set()).update(q.options)
    return {t: sorted(v) for t, v in pool.items() if 3 <= len(v) <= 200}


def make_items(train, tok, rng, max_ctx, none_prob=0.0):
    items = []; pool = build_label_pool(train) if none_prob > 0 else None
    for i, e in enumerate(train):
        it = build(none_augment(e, rng, none_prob, pool), tok, rng, max_ctx_tokens=max_ctx); it["task"] = e.task; it["ex_id"] = i
        items.append(it)
    return items


def batches_by_tokens(items, max_tokens, rng, bucket=64):
    """Deterministic shape buckets: T is padded to a multiple of `bucket`, and every
    batch in a T-bucket has exactly B = max_tokens // T rows (one ragged batch per bucket).
    Keeps the number of distinct (B, T) shapes ~= 24 so kernels compile once."""
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(items):
        T = ((len(it["ids"]) + bucket - 1) // bucket) * bucket
        groups[T].append(i)
    out = []
    for T, idx in groups.items():
        rng.shuffle(idx)
        B = max(1, max_tokens // T)
        for c in range(0, len(idx), B):
            out.append(idx[c:c + B])
    rng.shuffle(out)
    return out


def loss_fn(logits, golds, nopts, brier_w=0.0, label_smooth=0.0):
    ok = golds >= 0
    logits, golds, nopts = logits[ok], golds[ok], nopts[ok]
    ce = F.cross_entropy(logits, golds, label_smoothing=label_smooth)
    loss = ce
    if brier_w > 0:
        p = torch.softmax(logits, -1); p = torch.nan_to_num(p)
        onehot = F.one_hot(golds, p.shape[1]).float()
        loss = loss + brier_w * ((p - onehot) ** 2).sum(1).mean()
    return loss, ce.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--data", default="data/tasks.pkl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=16384, help="tokens per micro-batch (B*T)")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_ctx", type=int, default=1536)
    ap.add_argument("--brier_w", type=float, default=0.0)
    ap.add_argument("--label_smooth", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_limit", type=int, default=300)
    ap.add_argument("--train_cap", type=int, default=0, help="cap total train examples (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample_every_epoch", action="store_true")
    ap.add_argument("--none_prob", type=float, default=0.0, help="prob. of none-of-the-above augmentation per question")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/train.log", "a")
    def log(*s):
        msg = " ".join(str(x) for x in s); print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a)))
    torch.manual_seed(a.seed); rng = random.Random(a.seed)

    train, evals = D.load_cache(a.data)
    if a.train_cap:
        rng.shuffle(train); train = train[:a.train_cap]
    evals_small = {k: v[:a.eval_limit] for k, v in evals.items()}
    model = DecisionModel(a.model).cuda()
    tok = model.tok
    log(f"[data] train examples {len(train)}; tokenizing...")
    t0 = time.time(); items = make_items(train, tok, rng, a.max_ctx, a.none_prob)
    ntok = sum(len(it["ids"]) for it in items); nq = sum(len(it["slots"]) for it in items)
    log(f"[data] {len(items)} items, {nq} questions, {ntok/1e6:.1f}M tokens, tokenized in {time.time()-t0:.0f}s")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(batches_by_tokens(items, a.max_tokens, random.Random(0))) / a.accum)
    total = int(steps_per_epoch * a.epochs)
    log(f"[sched] {steps_per_epoch} optimizer steps/epoch, {total} total")
    def lr_at(s):
        if s < a.warmup:
            return a.lr * s / a.warmup
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))

    step, micro, ep = 0, 0, 0
    hist = []
    model.train()
    t0 = time.time(); ce_acc, n_acc = 0.0, 0; t_last = t0; tok_acc = 0
    while step < total:
        if ep > 0 and a.resample_every_epoch:
            items = make_items(train, tok, rng, a.max_ctx, a.none_prob)
        for bidx in batches_by_tokens(items, a.max_tokens, rng):
            if step >= total:
                break
            b = collate([items[i] for i in bidx], tok.pad_token_id)
            b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
            logits = model(b)
            loss, ce = loss_fn(logits, b["golds"], b["nopts"], a.brier_w, a.label_smooth)
            (loss / a.accum).backward()
            ce_acc += ce.item(); n_acc += 1; micro += 1; tok_acc += b["input_ids"].numel()
            if micro % a.accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    el = time.time() - t0; now = time.time(); tps = tok_acc / (now - t_last)
                    log(f"[train] step {step}/{total} ep {ep} ce {ce_acc/n_acc:.4f} gn {gn:.2f} lr {lr_at(step):.2e} {el/60:.1f}min eta {(total-step)*(now-t_last)/20/60:.0f}min {tps:.0f}tok/s mem {torch.cuda.max_memory_allocated()/1e9:.0f}GB")
                    ce_acc, n_acc = 0.0, 0; t_last = now; tok_acc = 0
                if step == total:                       # save before the final eval so an eval crash cannot lose the run
                    model.lm.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model"); log("[save]", f"{a.out}/model")
                if step % a.eval_every == 0 or step == total:
                    res, _ = run_eval(model, evals_small, log=log); agg = aggregate(res)
                    log(f"[eval-agg] step {step} " + json.dumps(agg))
                    hist.append(dict(step=step, agg=agg, results=res)); json.dump(hist, open(f"{a.out}/hist.json", "w"), indent=1)
                    model.train()
        ep += 1
    model.lm.save_pretrained(f"{a.out}/model"); tok.save_pretrained(f"{a.out}/model")
    log("[done] saved", f"{a.out}/model")


if __name__ == "__main__":
    main()
