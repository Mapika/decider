"""Latency / throughput of one-pass typed decisions (Jev-style claims check)."""
import argparse, pickle, random, time, torch, numpy as np
from .model import DecisionModel, collate
from .prompt import build
from . import data as D

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="Qwen/Qwen3.5-2B-Base"); ap.add_argument("--task", default="support_tickets")
a = ap.parse_args()
_, evals = D.load_cache()
m = DecisionModel(a.model, grad_ckpt=False).cuda().eval()
rng = random.Random(0)
items = [dict(build(e, m.tok, rng), task=a.task) for e in evals[a.task][:512]]
dev = "cuda"
def run(bs, n=20):
    ts = []
    for i in range(n):
        b = collate(items[(i * bs) % len(items):(i * bs) % len(items) + bs], m.tok.pad_token_id)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            m.slot_logits(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["slot_idx"].to(dev), b["slot_batch"].to(dev), b["nopts"].to(dev))
        torch.cuda.synchronize(); ts.append(time.time() - t)
    ts = ts[2:]; toks = np.mean([len(it["ids"]) for it in items]); nq = len(items[0]["slots"])
    print(f"bs={bs:4d}: p50 {np.median(ts)*1000:6.1f} ms  p90 {np.percentile(ts,90)*1000:6.1f} ms  -> {bs/np.median(ts):7.0f} contexts/s, {bs*nq/np.median(ts):7.0f} decisions/s, {bs*toks/np.median(ts):9.0f} tok/s (avg {toks:.0f} tok/ctx, {nq} q/ctx)")
for bs in [1, 8, 32, 128]:
    run(bs)
