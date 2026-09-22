"""Serving engine: every CUDA graph is keyed on (batch bucket, padded length bucket) only.

Difference from decider.engine.Engine (the library engine behind decider.infer.Decider):
  * the full bucket grid is captured at start-up by `warmup()`, then `seal()` forbids any further capture, so no request
    can ever pay a graph capture or a torch.compile;
  * the length ladder reaches 8192 instead of 2048, so long rows replay a graph instead of running eager;
  * a batch larger than the widest captured batch bucket for its length is split into captured chunks rather than
    capturing a new shape; rows longer than the last bucket run eager in chunks of at most `token_budget` padded tokens;
  * torch.compile and FP8 are off by default.

Numerics are the same as Engine: right padding + causal layers means padded positions never influence earlier slots, so
the answer for a row does not depend on which (B, T) bucket it was padded into, up to kernel reduction order.

The attention backend policy of decider.engine (cuDNN SDPA off) is applied in __init__, before compile and capture.  It
is what makes `score_shared` (a suffix scored against a cached prefix) return the same answers as the full forward; see
docs/CHANGELOG.md 1.0.2 and tests/test_engine_v2_cuda.py.
"""
import time, torch, torch.nn.functional as F
from decider import shared_prefix
from decider.engine import read_slots, fill_ids, patch_conv, set_attention_backend_policy
from decider.model import DecisionModel

T_BUCKETS = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192]
B_BUCKETS = [1, 2, 4, 8, 16, 32]
TOKEN_BUDGET = 32768        # capture (B, T) only when B * T fits this; B = 1 is always captured
LONG_STEP = 1024            # rows longer than the last bucket: pad to a multiple of this and run eager


class EngineV2:
    """compile / fp8 default to False.  `model` lets a test inject a stand-in DecisionModel instead of loading one."""

    def __init__(self, path=None, device="cuda", dtype=torch.bfloat16, use_graphs=None, compile=False, fp8=False,
                 conv_patch=None, t_buckets=None, b_buckets=None, token_budget=TOKEN_BUDGET, max_ctx_tokens=32768,
                 model=None):
        set_attention_backend_policy()              # before compile and capture: captured graphs keep their backend
        if conv_patch is None:
            conv_patch = bool(compile)          # the unrolled depthwise conv only pays off inside a compiled region
        if conv_patch:
            patch_conv()
        self.m = model if model is not None else DecisionModel(path, dtype=dtype, grad_ckpt=False).to(device).eval()
        self.tok = self.m.tok; self.dev = device; self.max_ctx = max_ctx_tokens
        self.core = self.m.lm.model
        self.W = self.m.lm.lm_head.weight[self.m.letters].detach().clone()
        self.t_buckets = sorted(t_buckets or T_BUCKETS)
        self.b_buckets = sorted(b_buckets or B_BUCKETS)
        self.token_budget = int(token_budget)
        self.use_graphs = (str(device).startswith("cuda") if use_graphs is None else bool(use_graphs))
        self.cfg = dict(compile=compile, fp8=fp8, conv_patch=conv_patch, graphs=self.use_graphs,
                        t_buckets=self.t_buckets, b_buckets=self.b_buckets, token_budget=self.token_budget)
        if fp8:
            from decider.fp8 import convert_to_fp8
            self.cfg["fp8_layers"] = convert_to_fp8(self.core)
        if compile:
            from torch import _dynamo                    # not `import torch._dynamo`: that rebinds `torch` locally
            n = len(self.graph_shapes())
            # one specialisation per captured shape, times the frames inside the model.  Without the accumulated
            # limits dynamo stops compiling part way through warm-up and silently leaves the rest eager, which with
            # conv_patch on is slower than not compiling at all.
            _dynamo.config.cache_size_limit = max(256, n + 16)
            _dynamo.config.accumulated_cache_size_limit = max(1 << 16, 64 * n)
            for k in ("accumulated_recompile_limit", "recompile_limit"):
                if hasattr(_dynamo.config, k):
                    setattr(_dynamo.config, k, max(1 << 16, 64 * n))
            self._fwd_impl = torch.compile(self._fwd_eager, dynamic=False)
        else:
            self._fwd_impl = self._fwd_eager
        self.graphs = {}                 # (B, T) -> (static ids, static out, graph)
        self.b_for_t = {}                # T -> sorted captured batch buckets
        self.pool = torch.cuda.graph_pool_handle() if (self.use_graphs and str(device).startswith("cuda")) else None
        self.sealed = False
        self.stats = dict(graph_captures=0, forwards=0, replays=0, eager_forwards=0, eager_rows=0,
                          shared_calls=0, unbucketed_requests=0)

    # ---- bucket arithmetic -------------------------------------------------
    def graph_shapes(self):
        """The full grid captured at start-up: every (B, T) whose padded token count fits the budget, plus all of B = 1."""
        return [(B, T) for T in self.t_buckets for B in self.b_buckets if B == 1 or B * T <= self.token_budget]

    def t_bucket(self, n):
        """Padded length bucket for a row of n tokens, or None when it is longer than the last bucket."""
        for t in self.t_buckets:
            if n <= t:
                return t
        return None

    def pad_len(self, n):
        return self.t_bucket(n) or -(-n // LONG_STEP) * LONG_STEP

    def max_rows(self, T):
        """Rows per forward at padded length T: the widest captured batch bucket, or, when nothing is captured at T, as
        many rows as fit the token budget (at least 1)."""
        av = self.b_for_t.get(T)
        return av[-1] if av else max(1, self.token_budget // max(T, 1))

    def b_plan(self, n, T):
        """Batch sizes covering n rows at length T, each one a captured bucket.  One chunk when a bucket fits."""
        av = self.b_for_t.get(T)
        if not av:
            return [n]
        fit = next((b for b in av if b >= n), None)
        if fit is not None:
            return [fit]
        out = []; big = av[-1]
        while n > big:
            out.append(big); n -= big
        out.append(next(b for b in av if b >= n))
        return out

    # ---- forward -----------------------------------------------------------
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
            for _ in range(3): self._fwd(s_ids)
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            s_out = self._fwd(s_ids)
        self.stats["graph_captures"] += 1
        return s_ids, s_out, g

    @torch.no_grad()
    def logits_all(self, ids):
        """ids: [B, T] long on device, already right-padded to a captured shape where one exists."""
        B, T = ids.shape; self.stats["forwards"] += 1
        if self.use_graphs:
            g = self.graphs.get((B, T))
            if g is None and not self.sealed:            # only reachable before seal(); a sealed engine never captures
                g = self.graphs[(B, T)] = self._capture(B, T)
                self.b_for_t.setdefault(T, [])
                if B not in self.b_for_t[T]: self.b_for_t[T] = sorted(self.b_for_t[T] + [B])
            if g is not None:
                s_ids, s_out, gr = g
                s_ids.copy_(ids); gr.replay(); self.stats["replays"] += 1
                return s_out
        self.stats["eager_forwards"] += 1; self.stats["eager_rows"] += B
        return self._fwd_eager(ids)                      # never the compiled callable: a new shape must not compile

    # ---- scoring -----------------------------------------------------------
    @torch.no_grad()
    def score_items(self, items, temperature=1.0):
        """items: dicts from prompt.build / build_rows.  -> one [n_q, MAX_OPTIONS] cpu probability tensor per item."""
        if not items:
            return []
        Tmax = max(len(it["ids"]) for it in items)
        T = self.t_bucket(Tmax)
        if T is None:
            self.stats["unbucketed_requests"] += 1
            T = -(-Tmax // LONG_STEP) * LONG_STEP
            per = self.max_rows(T); plan = [per] * (len(items) // per) + ([len(items) % per] if len(items) % per else [])
        else:
            plan = self.b_plan(len(items), T)
        out = []; i = 0
        for B in plan:
            chunk = items[i:i + B]; i += len(chunk)
            ids = fill_ids([it["ids"] for it in chunk], B, T, self.tok.pad_token_id)
            lg = self.logits_all(ids.to(self.dev, non_blocking=True))
            out += read_slots(lg, [b for b, it in enumerate(chunk) for _ in it["slots"]],
                              [s for it in chunk for s in it["slots"]], [n for it in chunk for n in it["nopts"]],
                              temperature, [len(it["slots"]) for it in chunk])
        return out

    @torch.no_grad()
    def score_shared(self, items, temperature=1.0, min_prefix=192, budget_bytes=None, rows_per_fork=None):
        """Rows that start with the same tokens (one state, one question per row): run the shared prefix once, fork its
        cache in chunks that fit a byte budget, run only the question suffixes.  The algorithm of Engine.score_shared,
        shared with it in decider.shared_prefix: it is per request, never per schema, so it adds no state that outlives
        the request and no shape that depends on the question set.  The prefix and suffix forwards are eager
        (request-specific shapes).  Correct only with the cuDNN SDPA backend off, which __init__ arranges."""
        out = shared_prefix.score_shared(self, items, temperature, min_prefix, budget_bytes, rows_per_fork)
        if out is None:
            return self.score_items(items, temperature)
        self.stats["shared_calls"] += 1
        return out

    # ---- start-up ----------------------------------------------------------
    def warmup(self, shapes=None, log=None):
        """Capture the whole grid.  Call seal() afterwards: from then on an unknown shape runs eager, never captures."""
        t = time.time(); shapes = list(shapes if shapes is not None else self.graph_shapes())
        for j, (B, T) in enumerate(shapes):
            self.logits_all(torch.full((B, T), self.tok.pad_token_id, dtype=torch.long, device=self.dev))
            if log and (j + 1) % 10 == 0:
                log(f"[engine_v2] {j + 1}/{len(shapes)} graphs, {time.time() - t:.0f}s")
        if self.use_graphs:
            torch.cuda.synchronize()
        return time.time() - t

    def seal(self):
        self.sealed = True
        self.b_for_t = {}
        for B, T in self.graphs:
            self.b_for_t.setdefault(T, []).append(B)
        for T in self.b_for_t:
            self.b_for_t[T].sort()
        return self

    def describe(self):
        return dict(self.cfg, graphs=len(self.graphs), sealed=self.sealed,
                    grid={str(T): self.b_for_t.get(T, []) for T in self.t_buckets})
