"""Try inference optimisations on fixed shape buckets: eager vs CUDA graph vs torch.compile."""
import sys, time, torch, numpy as np
from decider.model import DecisionModel
torch.backends.cuda.matmul.allow_tf32 = True
name = sys.argv[1] if len(sys.argv) > 1 else "runs/r3_v2/model"
m = DecisionModel(name, grad_ckpt=False).cuda().eval()
core, head, letters = m.lm.model, m.lm.lm_head, m.letters
K = 10

def fwd(ids, am):
    h = core(input_ids=ids, attention_mask=am).last_hidden_state
    return torch.nn.functional.linear(h[:, -1], head.weight[letters])

def bench(fn, ids, am, n=20, label=""):
    for _ in range(3): fn(ids, am)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn(ids, am)
    torch.cuda.synchronize(); dt = (time.time() - t) / n
    B, T = ids.shape
    print(f"{label:28s} B={B:3d} T={T:4d}: {dt*1000:7.2f} ms  {B*T/dt:8.0f} tok/s  {B/dt:7.0f} ctx/s", flush=True)
    return dt

shapes = [(1, 128), (1, 256), (8, 256), (32, 256), (32, 512)]
ins = {s: (torch.randint(0, 1000, s, device="cuda"), torch.ones(s, dtype=torch.long, device="cuda")) for s in shapes}

print("== eager")
with torch.no_grad():
    for s in shapes: bench(fwd, *ins[s], label="eager")

print("== kernel count (eager, B=1 T=128)")
from torch.profiler import profile, ProfilerActivity
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    fwd(*ins[(1, 128)]); torch.cuda.synchronize()
evs = [e for e in prof.events() if e.device_type.name == "CUDA"]
print(f"   {len(evs)} CUDA kernels per forward; gpu busy {sum(e.self_device_time_total for e in evs)/1000:.1f} ms")

print("== CUDA graph (manual capture per shape)")
def make_graph(shape):
    ids, am = ins[shape]
    s_ids = ids.clone(); s_am = am.clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.no_grad():
        for _ in range(3): fwd(s_ids, s_am)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        out = fwd(s_ids, s_am)
    def run(ids, am):
        s_ids.copy_(ids); s_am.copy_(am); g.replay(); return out
    return run
try:
    for s in shapes:
        run = make_graph(s)
        with torch.no_grad():
            ref = fwd(*ins[s]); got = run(*ins[s])
        assert torch.allclose(ref.float(), got.float(), atol=2e-2, rtol=2e-2), (ref[0], got[0])
        bench(run, *ins[s], label="cuda-graph")
except Exception as e:
    print("   cuda graph failed:", repr(e)[:300])

print("== torch.compile (default, dynamic=False)")
try:
    cfwd = torch.compile(fwd, dynamic=False)
    with torch.no_grad():
        for s in shapes: bench(cfwd, *ins[s], label="compile")
except Exception as e:
    print("   compile failed:", repr(e)[:300])

print("== torch.compile mode=reduce-overhead")
try:
    cfwd2 = torch.compile(fwd, mode="reduce-overhead", dynamic=False)
    with torch.no_grad():
        for s in shapes: bench(cfwd2, *ins[s], label="compile+cudagraph")
except Exception as e:
    print("   compile reduce-overhead failed:", repr(e)[:300])
