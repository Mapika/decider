"""Low-latency inference engine: shape-bucketed CUDA graphs over the one-pass decision model.

Right padding + causal layers => pad positions never influence earlier slots, so no attention
mask is needed and every (B, T) bucket can be captured once and replayed.  The graph outputs
option-letter logits for all positions [B, T, K]; slots are gathered outside.
"""
import time, torch, torch._dynamo, torch.nn.functional as F
from decider.model import DecisionModel, collate
from decider.prompt import build, MAX_OPTIONS
from decider.temperature import scaled_softmax, slot_temperatures

T_BUCKETS = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048]
B_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
GRAPH_MAX_T = 2048          # longer inputs (up to the 32k request budget) run eagerly: compute dominates there, and one graph
LONG_STEP = 1024            # per (B, T) shape would cost a compile + capture for every new length


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


def read_slots(out, rows, slots, nopts, temperature, n_per_item):
    """One gather + one softmax + one device-to-host copy for the whole batch (was: three small kernels and a sync per item).
    out [B, T, K] logits; rows/slots/nopts: flat python lists, one entry per question; n_per_item: questions per item.
    temperature: a number for every question, or a flat list with one temperature per question (decider.temperature)."""
    dev = out.device; idx = torch.tensor([rows, slots, nopts], dtype=torch.long).to(dev, non_blocking=True)
    lg = out[idx[0], idx[1]]                                                                  # [N, K]
    lg = lg.masked_fill(torch.arange(lg.shape[1], device=dev)[None, :] >= idx[2][:, None], float("-inf"))
    p = scaled_softmax(lg, temperature).cpu()
    return list(torch.split(p, n_per_item))


def fill_ids(items_ids, B, T, pad):
    import numpy as np
    a = np.full((B, T), pad, dtype=np.int64)
    for b, x in enumerate(items_ids): a[b, :len(x)] = x
    return torch.from_numpy(a)


def set_attention_backend_policy():
    """Turn off the cuDNN scaled-dot-product-attention backend. On Blackwell with torch 2.14 / CUDA 13 it returns wrong,
    finite output for masked rectangular attention, which is what the shared-state path (`Engine.score_shared`) and the schema
    cache run when a suffix is scored against a cached prefix; the math and memory-efficient backends are correct. Measured on
    a Decision Index row: the cached path answered a wrong option at p=0.93 where the full forward and the corrected cached path
    both give option_4 at p=0.95 (decider2/SERVING_V2_REVIEW.md in the research notes). Must run before torch.compile and CUDA
    graph capture: captured graphs keep the backend they were captured with."""
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


class Engine:
    """compile: torch.compile the forward (needs use_cache=False; ~1.4x batched, fuses elementwise work).
    fp8: e4m3 weights + per-token activation scaling on the big linears (Hopper tensor cores).
    conv_patch: fusable depthwise causal conv instead of the cuDNN fallback."""
    def __init__(self, path, device="cuda", dtype=torch.bfloat16, use_graphs=True, max_ctx_tokens=1536,
                 compile=True, fp8=False, conv_patch=True):
        set_attention_backend_policy()
        if conv_patch:
            if str(device).startswith("mps"):
                from decider.mps_ops import patch_mps
                patch_mps()
            else:
                patch_conv()
        self.m = DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        use_graphs = use_graphs and torch.device(device).type == "cuda"
        self.tok = self.m.tok; self.dev = device; self.use_graphs = use_graphs; self.max_ctx = max_ctx_tokens
        self.core, self.W = self.m.lm.model, self.m.lm.lm_head.weight[self.m.letters].detach().clone()
        self.cfg = dict(compile=compile, fp8=fp8, conv_patch=conv_patch, graphs=use_graphs)
        if fp8:
            from decider.fp8 import convert_to_fp8
            self.cfg["fp8_layers"] = convert_to_fp8(self.core)
        if compile:
            torch._dynamo.config.cache_size_limit = 128
            self._fwd_impl = torch.compile(self._fwd_eager, dynamic=False)
        else:
            self._fwd_impl = self._fwd_eager
        self.graphs = {}                       # (B, T) -> (static_ids, static_out, graph)
        self.pool = torch.cuda.graph_pool_handle() if (use_graphs and str(device).startswith("cuda")) else None
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
        if T > GRAPH_MAX_T:
            self.stats["long_forwards"] = self.stats.get("long_forwards", 0) + 1
            return self._fwd_eager(ids)
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
        """items: list of dicts from prompt.build. Returns list of [n_q, MAX_OPTIONS] prob tensors (cpu).
        temperature: a number, or one entry per item (a number or one number per slot; decider.temperature.for_items)."""
        Tmax = max(len(it["ids"]) for it in items)
        T = _bucket(Tmax, T_BUCKETS) or -(-Tmax // LONG_STEP) * LONG_STEP
        B = (_bucket(len(items), B_BUCKETS) or len(items)) if T <= GRAPH_MAX_T else len(items)
        ids = fill_ids([it["ids"] for it in items], B, T, self.tok.pad_token_id)
        out = self.logits_all(ids.to(self.dev, non_blocking=True))
        return read_slots(out, [b for b, it in enumerate(items) for _ in it["slots"]], [s for it in items for s in it["slots"]],
                          [n for it in items for n in it["nopts"]], slot_temperatures(temperature, items), [len(it["slots"]) for it in items])

    @torch.no_grad()
    def score_shared(self, items, temperature=1.0, min_prefix=192):
        """Rows that start with the same tokens (one state, one question per row): run the shared prefix once, fork its
        cache (attention KV + delta-net conv/recurrent states), and run only the question suffixes.
        Same answers as score_items up to kernel round-off; cost ~ state + sum(questions) instead of n * state.
        The fork is made in chunks that fit `DECIDER_SHARED_FORK_GB`, so the peak memory does not grow with the question
        count; the implementation is decider.shared_prefix, shared with EngineV2."""
        from decider import shared_prefix                      # imported here: decider.shared_prefix imports this module
        out = shared_prefix.score_shared(self, items, temperature, min_prefix)
        if out is None:
            return self.score_items(items, temperature)
        self.stats["shared_prefix_calls"] = self.stats.get("shared_prefix_calls", 0) + 1
        return out

    def warmup(self, shapes=((1, 128), (1, 256), (1, 384), (1, 512), (8, 256), (8, 512), (32, 256), (32, 512))):
        t = time.time()
        for B, T in shapes:
            self.logits_all(torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev))
        torch.cuda.synchronize(); return time.time() - t


if __name__ == "__main__":
    import sys, random, numpy as np
    from decider import data as D
    from decider.infer import Decider
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/r3_v2/model"
    cfg = dict(compile="nocompile" not in sys.argv[2:], fp8="fp8" in sys.argv[2:], conv_patch="noconv" not in sys.argv[2:])
    _, evals = D.load_cache("data/tasks.pkl")
    eng = Engine(path, **cfg); print("engine cfg", eng.cfg)
    rng = random.Random(0)
    exs = evals["support_tickets"][:64] + evals["clinc_oos"][:64] + evals["race"][:32]
    from decider.prompt import chat_for_model
    chat = chat_for_model(path, eng.tok)
    items = [build(e, eng.tok, rng, max_ctx_tokens=1536, chat=chat) for e in exs]
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
        its = [build(e, eng.tok, rng, chat=chat) for e in pool]
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
