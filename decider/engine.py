"""Low-latency inference engine: shape-bucketed CUDA graphs over the one-pass decision model.

Right padding + causal layers => pad positions never influence earlier slots, so no attention
mask is needed and every (B, T) bucket can be captured once and replayed.  The graph outputs
option-letter logits for all positions [B, T, K]; slots are gathered outside.
"""
import time, torch, torch.nn.functional as F
from .model import DecisionModel, collate
from .prompt import build, MAX_OPTIONS

T_BUCKETS = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048]
B_BUCKETS = [1, 2, 4, 8, 16, 32, 64]


def _bucket(x, buckets):
    for b in buckets:
        if x <= b:
            return b
    return None


def fused_causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    """Depthwise causal conv (kernel k) as k shifted multiply-adds: fuses under torch.compile,
    unlike the cuDNN grouped conv fallback (which was ~11% of batched GPU time)."""
    B, C, T = hidden_states.shape; k = weight.shape[-1]
    x = F.pad(hidden_states.to(weight.dtype), (k - 1, 0))
    out = x[:, :, k - 1:k - 1 + T] * weight[:, k - 1][None, :, None]
    for j in range(k - 1):
        out = out + x[:, :, j:j + T] * weight[:, j][None, :, None]
    if bias is not None:
        out = out + bias[None, :, None]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        from transformers.activations import ACT2FN
        out = ACT2FN[activation](out)
    return out.to(hidden_states.dtype)


def patch_conv():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    mq.causal_conv1d_fn = fused_causal_conv1d_fn


class Engine:
    """compile: torch.compile the forward (needs use_cache=False; ~1.4x batched, fuses elementwise work).
    fp8: e4m3 weights + per-token activation scaling on the big linears (Hopper tensor cores).
    conv_patch: fusable depthwise causal conv instead of the cuDNN fallback."""
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, use_graphs=True, max_ctx_tokens=1536,
                 compile=True, fp8=False, conv_patch=True):
        if conv_patch:
            patch_conv()
        self.m = DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        self.tok = self.m.tok; self.dev = device; self.use_graphs = use_graphs; self.max_ctx = max_ctx_tokens
        self.core, self.W = self.m.lm.model, self.m.lm.lm_head.weight[self.m.letters].detach().clone()
        self.cfg = dict(compile=compile, fp8=fp8, conv_patch=conv_patch, graphs=use_graphs)
        if fp8:
            from .fp8 import convert_to_fp8
            self.cfg["fp8_layers"] = convert_to_fp8(self.core)
        if compile:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = 128
            self._fwd_impl = torch.compile(self._fwd_eager, dynamic=False)
        else:
            self._fwd_impl = self._fwd_eager
        self.graphs = {}                       # (B, T) -> (static_ids, static_out, graph)
        self.pool = torch.cuda.graph_pool_handle() if use_graphs else None
        self.stats = dict(graph_captures=0, forwards=0)

    def _fwd_eager(self, ids):
        h = self.core(input_ids=ids, use_cache=False).last_hidden_state
        return F.linear(h, self.W).float()                          # [B, T, K]

    @torch.no_grad()
    def _fwd(self, ids):
        return self._fwd_impl(ids)

    def _capture(self, B, T):
        s_ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): self._fwd(s_ids)                     # warm-up: compile / triton autotune
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            s_out = self._fwd(s_ids)
        self.stats["graph_captures"] += 1
        return s_ids, s_out, g

    @torch.no_grad()
    def logits_all(self, ids):
        """ids: [B, T] long on device (already right-padded to a bucket). Returns [B, T, K] float."""
        B, T = ids.shape; self.stats["forwards"] += 1
        if not self.use_graphs:
            return self._fwd(ids)
        key = (B, T)
        if key not in self.graphs:
            self.graphs[key] = self._capture(B, T)
        s_ids, s_out, g = self.graphs[key]
        s_ids.copy_(ids); g.replay()
        return s_out

    @torch.no_grad()
    def score_items(self, items, temperature=1.0):
        """items: list of dicts from prompt.build. Returns list of [n_q, MAX_OPTIONS] prob tensors (cpu)."""
        Tmax = max(len(it["ids"]) for it in items)
        T = _bucket(Tmax, T_BUCKETS) or Tmax; B = _bucket(len(items), B_BUCKETS) or len(items)
        ids = torch.full((B, T), self.tok.pad_token_id, dtype=torch.long)
        for b, it in enumerate(items):
            ids[b, :len(it["ids"])] = torch.tensor(it["ids"])
        out = self.logits_all(ids.to(self.dev, non_blocking=True))
        res = []
        ar = torch.arange(MAX_OPTIONS, device=self.dev)
        for b, it in enumerate(items):
            sl = torch.tensor(it["slots"], device=self.dev)
            lg = out[b, sl]                                            # [n_q, K]
            nop = torch.tensor(it["nopts"], device=self.dev)
            lg = lg.masked_fill(ar[None, :] >= nop[:, None], float("-inf"))
            res.append(torch.softmax(lg / temperature, -1).cpu())
        return res

    def warmup(self, shapes=((1, 128), (1, 256), (1, 384), (1, 512), (8, 256), (8, 512), (32, 256), (32, 512))):
        t = time.time()
        for B, T in shapes:
            self.logits_all(torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev))
        torch.cuda.synchronize(); return time.time() - t


if __name__ == "__main__":
    import sys, random, numpy as np
    from . import data as D
    from .infer import Decider
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/r3_v2/model"
    cfg = dict(compile="nocompile" not in sys.argv[2:], fp8="fp8" in sys.argv[2:], conv_patch="noconv" not in sys.argv[2:])
    _, evals = D.load_cache("data/tasks.pkl")
    eng = Engine(path, **cfg); print("engine cfg", eng.cfg)
    rng = random.Random(0)
    exs = evals["support_tickets"][:64] + evals["clinc_oos"][:64] + evals["race"][:32]
    items = [build(e, eng.tok, rng, max_ctx_tokens=1536) for e in exs]
    # correctness vs eager masked forward (DecisionModel.slot_logits)
    ref = []
    with torch.no_grad():
        for i in range(0, len(items), 16):
            b = collate(items[i:i + 16], eng.tok.pad_token_id)
            lg = eng.m.slot_logits(b["input_ids"].cuda(), b["attention_mask"].cuda(), b["slot_idx"].cuda(), b["slot_batch"].cuda(), b["nopts"].cuda())
            ref.append(torch.softmax(lg, -1).cpu())
    ref = torch.cat(ref)
    got = torch.cat(eng.score_items(items))
    print(f"max |p_graph - p_eager| = {(ref - got).abs().max():.4f} over {len(ref)} questions; argmax agreement {(ref.argmax(1) == got.argmax(1)).float().mean():.4f}")
    print(f"warmup capture of 8 buckets: {eng.warmup():.1f}s; captures so far {eng.stats['graph_captures']}")
    # latency: single real requests
    for name, pool in [("support_tickets", exs[:64]), ("clinc_oos", exs[64:128]), ("race", exs[128:])]:
        its = [build(e, eng.tok, rng) for e in pool]
        ts = []
        for it in its[:40]:
            torch.cuda.synchronize(); t = time.time(); eng.score_items([it]); torch.cuda.synchronize(); ts.append(time.time() - t)
        ts = np.array(ts[5:]) * 1000
        print(f"single request {name:16s}: p50 {np.median(ts):5.1f} ms  p90 {np.percentile(ts, 90):5.1f} ms  (avg {np.mean([len(i['ids']) for i in its]):.0f} tok, {len(its[0]['slots'])} q)")
        for bs in (8, 32):
            ts = []
            for i in range(0, min(len(its), bs * 6), bs):
                chunk = its[i:i + bs]
                if len(chunk) < bs: break
                torch.cuda.synchronize(); t = time.time(); eng.score_items(chunk); torch.cuda.synchronize(); ts.append(time.time() - t)
            ts = np.array(ts[1:]) * 1000
            print(f"   batch {bs:2d}: p50 {np.median(ts):6.1f} ms -> {bs/np.median(ts)*1000:6.0f} ctx/s, {bs*len(its[0]['slots'])/np.median(ts)*1000:6.0f} decisions/s")
    print("stats", eng.stats, "graphs", len(eng.graphs), f"mem {torch.cuda.memory_reserved()/1e9:.1f} GB")
