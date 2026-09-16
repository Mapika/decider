import sys, time, torch, torch.nn.functional as F
from .model import DecisionModel
m = DecisionModel(sys.argv[1] if len(sys.argv) > 1 else "runs/r3_v2/model", grad_ckpt=False).cuda().eval()
core, W = m.lm.model, m.lm.lm_head.weight[m.letters].detach().clone()
def fwd(ids):
    return F.linear(core(input_ids=ids, use_cache=False).last_hidden_state, W).float()
shapes = [(1, 128), (8, 256), (32, 256), (32, 512)]
ins = {s: torch.randint(0, 1000, s, device="cuda") for s in shapes}
def bench(fn, ids, label, n=15):
    with torch.no_grad():
        for _ in range(3): fn(ids)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n): fn(ids)
        torch.cuda.synchronize(); dt = (time.time() - t) / n
    B, T = ids.shape; print(f"{label:34s} B={B:3d} T={T:4d}: {dt*1000:7.2f} ms  {B*T/dt:8.0f} tok/s", flush=True); return dt
for s in shapes: bench(fwd, ins[s], "eager use_cache=False")
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
mode = sys.argv[2] if len(sys.argv) > 2 else "default"
cfwd = torch.compile(fwd, dynamic=False, mode=None if mode == "default" else mode)
with torch.no_grad():
    ref = fwd(ins[(8, 256)]); got = cfwd(ins[(8, 256)])
print("compile max abs diff vs eager:", (ref - got).abs().max().item())
for s in shapes: bench(cfwd, ins[s], f"compile[{mode}]")
# manual CUDA graph around the compiled function
def make_graph(shape):
    s_ids = ins[shape].clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st), torch.no_grad():
        for _ in range(3): cfwd(s_ids)
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): out = cfwd(s_ids)
    def run(ids): s_ids.copy_(ids); g.replay(); return out
    return run
try:
    for s in shapes:
        run = make_graph(s)
        with torch.no_grad(): d = (fwd(ins[s]) - run(ins[s])).abs().max().item()
        bench(run, ins[s], f"compile[{mode}]+cudagraph (diff {d:.3f})")
except Exception as e:
    print("graph capture of compiled fn failed:", repr(e)[:300])
