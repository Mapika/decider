"""Fine-tune the vision decision model (image + lettered prompt -> slot logits). Mixed image and text-only examples.
   python -m decider.train_vision --data data/vision_mix.pkl --out runs/v1_vision"""
import argparse, json, math, os, pickle, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn.functional as F
from .vision import VisionDecisionModel
from .metrics import summarize
from . import data as D
from . import data2, data3  # noqa


def batches(examples, bs_img, bs_txt, rng):
    img = [i for i, e in enumerate(examples) if getattr(e, "image", None) is not None]; txt = [i for i, e in enumerate(examples) if getattr(e, "image", None) is None]
    rng.shuffle(img); rng.shuffle(txt)
    out = [img[i:i + bs_img] for i in range(0, len(img), bs_img)] + [txt[i:i + bs_txt] for i in range(0, len(txt), bs_txt)]
    rng.shuffle(out); return out


@torch.no_grad()
def evaluate(model, evals, bs=16, limit=300, log=print):
    model.eval(); res = {}
    for t, exs in evals.items():
        exs = exs[:limit]; P, G, NO = [], [], []
        if len(exs) < 4: continue
        for i in range(0, len(exs), bs):
            chunk = exs[i:i + bs]
            inp = model.prepare([(getattr(e, "image", None), e) for e in chunk])
            p = torch.softmax(model.slot_logits(inp), -1).cpu().numpy(); P.append(np.nan_to_num(p)); G.append(inp["golds"].numpy()); NO.append(inp["nopts"].numpy())
        P, G, NO = np.concatenate(P), np.concatenate(G), np.concatenate(NO); ok = G >= 0
        s = summarize(P[ok], G[ok], NO[ok]); res[t] = s
        log(f"[eval] {t:22s} n={s['n']:4d} acc={s['acc']:.3f} (chance {s['chance']:.2f}) nll={s['nll']:.3f} ece={s['ece']:.3f} acc@80={s['acc_at_80']:.3f}")
    model.train(); return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base"); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--bs_img", type=int, default=16); ap.add_argument("--bs_txt", type=int, default=48); ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=100000); ap.add_argument("--eval_limit", type=int, default=300); ap.add_argument("--freeze_vision", action="store_true")
    ap.add_argument("--save_every", type=int, default=500)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/train.log", "a")
    def log(*s):
        m = " ".join(str(x) for x in s); print(m, flush=True); logf.write(m + "\n"); logf.flush()
    log("[args]", json.dumps(vars(a))); rng = random.Random(0); torch.manual_seed(0)
    train, evals = D.load_cache(a.data)
    model = VisionDecisionModel(a.model).cuda()
    if a.freeze_vision:
        for n, p in model.lm.named_parameters():
            if "visual" in n or "vision" in n: p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95))
    n_img = sum(getattr(e, "image", None) is not None for e in train)
    steps_per_epoch = math.ceil(len(batches(train, a.bs_img, a.bs_txt, random.Random(0))) / a.accum); total = int(steps_per_epoch * a.epochs)
    log(f"[data] {len(train)} examples ({n_img} with image), {len(evals)} eval tasks; {total} optimizer steps")
    def lr_at(s): return a.lr * s / a.warmup if s < a.warmup else a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - a.warmup) / max(1, total - a.warmup))))
    step = micro = 0; t0 = time.time(); ce_acc = n_acc = 0; model.train()
    while step < total:
        for idx in batches(train, a.bs_img, a.bs_txt, rng):
            if step >= total: break
            chunk = [train[i] for i in idx]
            inp = model.prepare([(getattr(e, "image", None), e) for e in chunk])
            lg = model.slot_logits(inp); golds = inp["golds"].cuda(); ok = golds >= 0
            loss = F.cross_entropy(lg[ok], golds[ok]); (loss / a.accum).backward(); ce_acc += loss.item(); n_acc += 1; micro += 1
            if micro % a.accum == 0:
                for g in opt.param_groups: g["lr"] = lr_at(step)
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    el = time.time() - t0; log(f"[train] step {step}/{total} ce {ce_acc/n_acc:.4f} gn {gn:.2f} lr {lr_at(step):.2e} {el/60:.1f}min eta {(total-step)*el/step/60:.0f}min mem {torch.cuda.max_memory_allocated()/1e9:.0f}GB"); ce_acc = n_acc = 0
                if step == total or step % a.save_every == 0:
                    model.lm.save_pretrained(f"{a.out}/model"); model.proc.save_pretrained(f"{a.out}/model"); log("[save]", f"{a.out}/model", f"(step {step})")
                if step % a.eval_every == 0 or step == total:
                    res = evaluate(model, evals, limit=a.eval_limit, log=log); json.dump(res, open(f"{a.out}/eval_{step}.json", "w"), indent=1)
    log("[done]")


if __name__ == "__main__":
    main()
