"""Schema cache: compute a question schema once, then score states against it.

In production the questions are fixed and only the state changes.  With the schema-first prompt layout
(prompt.build_schema_first) the question/option blocks are a prefix that does not depend on the state, so their
cache - attention K/V for the 6 full-attention layers, conv + recurrent state for the 18 delta-net layers - is computed
once (`prepare`).  A request then runs only "Context: <state>" plus one answer slot per question, as a CUDA graph per
(batch, length) bucket.  The prefix cache is read-only during a request (nothing is written back), so one copy serves
every batch and every graph.

    se = SchemaEngine(engine); h = se.prepare([{"question": ..., "options": [...]}, ...])
    probs = se.score(h, ["state 1", "state 2", ...])          # list of [n_questions, MAX_OPTIONS] tensors
"""
import time, types, torch, torch.nn.functional as F
from decider.prompt import schema_prefix_ids, schema_suffix_ids, MAX_OPTIONS
from decider.engine import read_slots, fill_ids

TS_BUCKETS = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
B_BUCKETS = [1, 2, 4, 8, 16, 32, 64]


class _Q:
    def __init__(self, text, options): self.text, self.options = text, options


class PrefixCache:
    """Duck-typed transformers Cache over fixed, read-only prefixes, for one suffix forward pass.
    A handle holds P prefixes (P = 1: all questions packed in one prefix; P = n_questions: one prefix per question, so every
    question is scored independently).  A batch of R states has R * P rows; row r * P + p continues prefix p."""
    def __init__(self, h, R):
        rep = (lambda t: t.expand(R, *t.shape[1:])) if h.P == 1 else (lambda t: t.repeat(R, *([1] * (t.dim() - 1))))
        self.tp = h.tpmax; self.k = {i: rep(k) for i, k in h.k.items()}; self.v = {i: rep(v) for i, v in h.v.items()}
        self.conv = {i: rep(c).contiguous() for i, c in h.conv.items()}
        self.layers = {i: types.SimpleNamespace(record_past=False, recurrent_states={0: rep(r).contiguous()}) for i, r in h.rec.items()}

    def has_previous_state(self, layer_idx=None, state_idx=None): return True
    def get_seq_length(self, *a, **k): return self.tp
    def update(self, key, value, layer_idx, *a, **k): return torch.cat([self.k[layer_idx], key], 2), torch.cat([self.v[layer_idx], value], 2)
    def update_conv_state(self, x, layer_idx, **k): return torch.cat([self.conv[layer_idx].to(x.dtype), x], -1)
    def update_recurrent_state(self, s, layer_idx, **k): return s


class SchemaEngine:
    def __init__(self, engine, use_graphs=True):
        self.e = engine; self.core = engine.core; self.W = engine.W; self.tok = engine.tok; self.dev = engine.dev
        self.use_graphs = use_graphs and engine.use_graphs; self.graphs = {}; self.stats = dict(prepared=0, captures=0, replays=0, eager=0)
        self.compile = bool(engine.cfg.get("compile")); self._compiled = {}

    @torch.no_grad()
    def prepare(self, questions, independent=False, compile=False):
        """questions: [{"question": str, "options": [str]}] in the order answers are wanted.  Runs the prefix(es) once.
        independent=False: one prefix holding every question (cheapest: a request costs state + n slots).
        independent=True:  one prefix per question, one row per question (a request costs n * (state + 1 slot); no question
                           can influence another)."""
        qs = [_Q(q["question"], list(q["options"])) for q in questions]
        groups = [[q] for q in qs] if independent else [qs]; pres = [schema_prefix_ids(self.tok, g) for g in groups]
        h = types.SimpleNamespace(P=len(groups), nq=len(qs), slots_per_row=1 if independent else len(qs), nopts=[len(q.options) for q in qs], tps=[len(p) for p in pres],
                                  tpmax=max(len(p) for p in pres), k={}, v={}, conv={}, rec={}, id=self.stats["prepared"],
                                  compile=bool(compile and self.compile))
        parts = []
        for pre in pres:
            out = self.core(input_ids=torch.tensor(pre, device=self.dev)[None], use_cache=True).past_key_values; d = dict(k={}, v={}, conv={}, rec={})
            for i, layer in enumerate(out.layers):
                if getattr(layer, "recurrent_states", None) is not None and layer.recurrent_states.get(0) is not None:
                    d["conv"][i] = layer.conv_states[0]; d["rec"][i] = layer.recurrent_states[0]
                else:                                                   # right-pad every prefix's K/V to the longest; the mask hides the padding
                    pad = (0, 0, 0, h.tpmax - len(pre)); d["k"][i] = F.pad(layer.keys, pad); d["v"][i] = F.pad(layer.values, pad)
            parts.append(d)
        for name in ("k", "v", "conv", "rec"):
            getattr(h, name).update({i: torch.cat([d[name][i] for d in parts], 0).clone() for i in parts[0][name]})
        self.stats["prepared"] += 1
        return h

    def _fwd(self, ids, cache, mask, pos):
        hs = self.core(input_ids=ids, past_key_values=cache, attention_mask={"full_attention": mask, "linear_attention": None}, position_ids=pos, use_cache=True).last_hidden_state
        return F.linear(hs, self.W).float()

    def _static(self, h, R, Ts):
        """R request slots -> R * P rows.  Mask: a row sees its own prefix (not the padding up to tpmax) and the causal suffix."""
        ar = torch.arange(Ts, device=self.dev); tps = torch.tensor(h.tps, device=self.dev).repeat(R)                                # [R*P]
        pre = (torch.arange(h.tpmax, device=self.dev)[None, :] < tps[:, None])[:, None, None, :].expand(-1, 1, Ts, -1)             # [B,1,Ts,tpmax]
        mask = torch.cat([pre, (ar[:, None] >= ar[None, :])[None, None].expand(len(tps), 1, -1, -1)], 3).contiguous()
        return PrefixCache(h, R), mask, (tps[:, None] + ar[None, :]).contiguous()

    def _capture(self, h, R, Ts):
        B = R * h.P
        ids = torch.full((B, Ts), self.tok.pad_token_id, dtype=torch.long, device=self.dev); cache, mask, pos = self._static(h, R, Ts)
        fwd = self._fwd
        if h.compile:                                       # one compiled function per graph (20-30 s each: only for preloaded schemas): the cache tensors are constants of that graph
            fwd = torch.compile(lambda i: self._fwd(i, cache, mask, pos), dynamic=False)
            call = lambda: fwd(ids)
        else:
            call = lambda: fwd(ids, cache, mask, pos)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3): call()
        torch.cuda.current_stream().wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.e.pool):
            out = call()
        self.stats["captures"] += 1
        return ids, out, g, (cache, mask, pos)

    def warmup(self, h, batch_sizes=(1, 8, 32), state_tokens=(64, 128, 256)):
        """Capture (and, for a compiled schema, compile) the graphs for these request-batch sizes and suffix lengths ahead of traffic."""
        t = time.time()
        for R in batch_sizes:
            for Ts in state_tokens:
                Ts = next((x for x in TS_BUCKETS if x >= Ts), TS_BUCKETS[-1])
                if (h.id, R, Ts) not in self.graphs: self.graphs[(h.id, R, Ts)] = self._capture(h, R, Ts)
        torch.cuda.synchronize(); return time.time() - t

    def tokenize(self, h, context, max_ctx_tokens=1536):
        """CPU part of a request (do it outside any GPU lock): -> (suffix ids, slot positions)."""
        return schema_suffix_ids(self.tok, context, h.slots_per_row, max_ctx_tokens)

    @staticmethod
    def bucket(n_tokens):
        return next((t for t in TS_BUCKETS if t >= n_tokens), -(-n_tokens // 256) * 256)

    def score(self, h, contexts, temperature=1.0, max_ctx_tokens=1536):
        """-> one [n_questions, MAX_OPTIONS] probability tensor per context."""
        return self.score_rows(h, [self.tokenize(h, c, max_ctx_tokens) for c in contexts], temperature)

    @torch.no_grad()
    def score_rows(self, h, rows, temperature=1.0):
        """rows: [(suffix ids, slots)] from tokenize()."""
        Tmax = max(len(r[0]) for r in rows); Ts = next((t for t in TS_BUCKETS if t >= Tmax), None); n = len(rows)
        R = next((b for b in B_BUCKETS if b >= n), n) if Ts else n; Ts = Ts or -(-Tmax // 256) * 256
        ids = fill_ids([x for x, _ in rows for _ in range(h.P)], R * h.P, Ts, self.tok.pad_token_id).to(self.dev, non_blocking=True)
        if self.use_graphs and Ts <= TS_BUCKETS[-1]:
            key = (h.id, R, Ts)
            if key not in self.graphs: self.graphs[key] = self._capture(h, R, Ts)
            s_ids, s_out, g, _ = self.graphs[key]; s_ids.copy_(ids); g.replay(); out = s_out; self.stats["replays"] += 1
        else:
            out = self._fwd(ids, *self._static(h, R, Ts)); self.stats["eager"] += 1
        if h.P == 1:                                           # packed: n slots in one row per request
            rws = [r for r in range(n) for _ in range(h.nq)]; sls = [x for _, sl in rows for x in sl]
        else:                                                  # independent: one slot in each of the request's P rows
            rws = [r * h.P + p for r in range(n) for p in range(h.P)]; sls = [sl[0] for _, sl in rows for _ in range(h.P)]
        return read_slots(out, rws, sls, h.nopts * n, temperature, [h.nq] * n)
