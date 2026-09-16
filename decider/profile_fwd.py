"""Where does batched inference time go? Kernel time by category at B=32, T=256 (eager, no mask)."""
import sys, time, torch, collections
from .model import DecisionModel
from torch.profiler import profile, ProfilerActivity
m = DecisionModel(sys.argv[1] if len(sys.argv) > 1 else "runs/r3_v2/model", grad_ckpt=False).cuda().eval()
B, T = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "32,256").split(","))
x = torch.randint(0, 1000, (B, T), device="cuda")
with torch.no_grad():
    for _ in range(3): m.lm.model(input_ids=x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5): m.lm.model(input_ids=x)
        torch.cuda.synchronize()
cat = collections.Counter(); n = collections.Counter()
def cls(name):
    s = name.lower()
    if "nvjet" in s or "gemm" in s or "cutlass" in s or "sm90_xmma" in s or "matmul" in s: return "matmul"
    if "chunk" in s or "delta" in s or "wy_" in s or "fwd_kernel" in s or "recompute" in s or "solve_tril" in s or "cumsum" in s: return "fla (gated delta rule)"
    if "conv" in s: return "conv1d"
    if "flash" in s or "fmha" in s or "attention" in s or "sdpa" in s: return "attention"
    if "elementwise" in s or "vectorized" in s or "reduce" in s or "norm" in s or "copy" in s or "fill" in s or "cat" in s or "index" in s or "softmax" in s or "silu" in s or "sigmoid" in s or "where" in s: return "elementwise/reduce"
    return "other"
tot = 0
for e in prof.events():
    if e.device_type.name != "CUDA": continue
    c = cls(e.name); cat[c] += e.self_device_time_total / 5; n[c] += 1 / 5; tot += e.self_device_time_total / 5
print(f"B={B} T={T}: GPU time {tot/1000:.1f} ms per forward, {B*T/(tot/1e6):.0f} tok/s (kernel-time bound)")
for c, v in cat.most_common():
    print(f"  {c:26s} {v/1000:7.2f} ms  {100*v/tot:5.1f}%  ({n[c]:.0f} kernels)")
others = collections.Counter()
for e in prof.events():
    if e.device_type.name == "CUDA" and cls(e.name) == "other": others[e.name[:70]] += e.self_device_time_total / 5
print("  top 'other':", [(k, round(v/1000, 2)) for k, v in others.most_common(6)])
