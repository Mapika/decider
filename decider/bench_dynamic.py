"""Compile once with dynamic shapes vs per-shape static compile: warmup cost and steady-state speed."""
import sys, time, torch, torch.nn.functional as F
from .engine import Engine, patch_conv
patch_conv()
from .model import DecisionModel
m = DecisionModel("runs/r3_v2/model", grad_ckpt=False).cuda().eval()
core, W = m.lm.model, m.lm.lm_head.weight[m.letters].detach().clone()
def fwd(ids): return F.linear(core(input_ids=ids, use_cache=False).last_hidden_state, W).float()
shapes = [(1, 128), (1, 256), (1, 512), (8, 256), (8, 512), (32, 256), (32, 512), (2, 384), (4, 640), (16, 192)]
ins = {s: torch.randint(0, 1000, s, device="cuda") for s in shapes}
import torch._dynamo; torch._dynamo.config.cache_size_limit = 128
for label, fn in [("static per-shape", torch.compile(fwd, dynamic=False)), ("dynamic=True", torch.compile(fwd, dynamic=True))]:
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        for s in shapes: fn(ins[s])
    torch.cuda.synchronize(); print(f"{label:18s} first pass over {len(shapes)} shapes: {time.time()-t:.1f}s", flush=True)
    with torch.no_grad():
        for s in [(1, 128), (8, 256), (32, 512)]:
            for _ in range(3): fn(ins[s])
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10): fn(ins[s])
            torch.cuda.synchronize(); dt = (time.time() - t) / 10
            print(f"   {label:18s} B={s[0]:2d} T={s[1]:3d}: {dt*1000:6.2f} ms  {s[0]*s[1]/dt:7.0f} tok/s", flush=True)
