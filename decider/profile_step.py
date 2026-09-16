"""Time real training batches by shape, and profile a few steps."""
import pickle, random, time, torch, sys
from collections import defaultdict
from .model import DecisionModel, collate
from .train import make_items, batches_by_tokens, loss_fn
from . import data as D
train, _ = D.load_cache()
rng = random.Random(0); rng.shuffle(train); train = train[:6000]
m = DecisionModel("Qwen/Qwen3.5-2B-Base").cuda(); m.train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-6)
items = make_items(train, m.tok, rng, 1536)
bs = batches_by_tokens(items, 16384, rng)
def step(b):
    b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    logits = m(b); loss, _ = loss_fn(logits, b["golds"], b["nopts"]); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
stats = defaultdict(list)
t_all = time.time()
for i, bidx in enumerate(bs[:60]):
    b = collate([items[j] for j in bidx], m.tok.pad_token_id)
    shape = tuple(b["input_ids"].shape)
    torch.cuda.synchronize(); t = time.time(); step(b); torch.cuda.synchronize(); dt = time.time() - t
    stats[shape].append(dt)
    print(f"batch {i} shape {shape} tokens {shape[0]*shape[1]} {dt*1000:.0f} ms -> {shape[0]*shape[1]/dt:.0f} tok/s", flush=True)
print("total", time.time() - t_all)
print("--- per shape (first vs later):")
for s, v in sorted(stats.items()):
    print(s, "n", len(v), "first %.0f ms" % (v[0]*1000), "rest %.0f ms" % (1000*sum(v[1:])/max(1,len(v)-1)))
# profiler on 3 warm steps of a repeated shape
from torch.profiler import profile, ProfilerActivity
b = collate([items[j] for j in bs[0]], m.tok.pad_token_id); step(b)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(3): step(b)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
