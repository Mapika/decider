"""GGUF engine: the one-pass decision readout on llama.cpp (llama-cpp-python), for quantized checkpoints on CPU, Apple Silicon
and CUDA without the torch forward.

The prompt rows are built exactly as for the torch engine (decider.prompt with the checkpoint's HF tokenizer), the rows are
decoded by llama.cpp with logits requested only at the answer slots, and the option-letter columns of those logits are the
readout: the slot state projected on the LM-head letter rows, as in decider.model.DecisionModel.  Temperatures and everything
after the probabilities are shared with the torch engine.

One row per llama_decode by default (n_seq_max=1), the memory cleared between decodes: repeatable to the bit and the path the
GGUF measurements were made on.  n_seq_max > 1 packs rows into one decode as separate sequences: faster, but on llama.cpp
(September 2026, Qwen3.5 hybrid layers) a row's probabilities then move with its neighbours (up to 0.02 in BF16 and 0.16 in
Q4_K_M on decider-4b v2.1), so it is off by default.  score_shared has no prefix cache here and scores every row in full (same answers, more compute).

    pip install "decider-ai[gguf]"      # llama-cpp-python; build it with CMAKE_ARGS="-DGGML_CUDA=on" or "-DGGML_METAL=on"
    from decider.infer import Decider
    d = Decider("Mapika/decider-4b-GGUF", gguf_file="decider-4b-v2.1-Q4_K_M.gguf")
"""
import ctypes
import os
from types import SimpleNamespace

import numpy as np
import torch

from decider.prompt import MAX_OPTIONS, letter_ids
from decider.temperature import scaled_softmax, slot_temperatures


def _llama():
    try:
        import llama_cpp
    except ImportError as e:
        raise ImportError('GGUF checkpoints need llama-cpp-python: pip install "decider-ai[gguf]"') from e
    return llama_cpp


_QUIET = None


def _quiet(L):
    """Silence llama.cpp's stderr log (process-wide; the callback object is kept alive at module level)."""
    global _QUIET
    if _QUIET is None:
        _QUIET = L.llama_log_callback(lambda level, text, data: None)
        L.llama_log_set(_QUIET, ctypes.c_void_p(0))


class GGUFEngine:
    """gguf_path: the .gguf file.  tokenizer_dir: a folder (or Hub repo) with the HF tokenizer the GGUF was converted from.
    n_ctx: the longest row in tokens.  system_one cuts the state to 32,768 tokens and adds the question after it, so the default
    40,960 (the model length decider.serve_vllm uses) leaves 8k tokens for the question and its options.
    n_seq_max: rows per decode.  n_gpu_layers: -1 offloads every layer (CUDA / Metal builds); 0 runs on the CPU."""

    def __init__(self, gguf_path, tokenizer_dir, n_ctx=40960, n_seq_max=1, n_batch=8192, n_gpu_layers=-1, n_threads=None,
                 verbose=False):
        from transformers import AutoTokenizer
        L = self.L = _llama()
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.letters = np.asarray(letter_ids(self.tok))
        self.m = SimpleNamespace(tok=self.tok)                      # Decider reads the tokenizer from eng.m
        if not verbose:
            _quiet(L)
        L.llama_backend_init()
        mp = L.llama_model_default_params(); mp.n_gpu_layers = n_gpu_layers
        self.model = L.llama_model_load_from_file(os.fsencode(gguf_path), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {gguf_path}")
        cp = L.llama_context_default_params()
        # llama.cpp aborts (a native assert, not an error code) when one llama_decode gets more tokens than the context's n_batch,
        # so the context takes a full n_ctx row; n_batch below is only the token budget for packing rows (n_seq_max > 1).
        cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = min(2048, n_ctx); cp.n_seq_max = n_seq_max
        cp.kv_unified = True                                        # n_ctx is shared by the rows of a decode, not split
        if n_threads:
            cp.n_threads = cp.n_threads_batch = n_threads
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context")
        self.n_ctx, self.n_batch, self.n_seq_max = n_ctx, min(n_batch, n_ctx), n_seq_max
        self.n_vocab = L.llama_vocab_n_tokens(L.llama_model_get_vocab(self.model))
        self.batch = L.llama_batch_init(n_ctx, 0, n_seq_max)
        self.dev = "gguf"; self.cfg = dict(backend="llama.cpp", gguf=str(gguf_path), n_ctx=n_ctx, n_seq_max=n_seq_max)
        self.stats = dict(forwards=0)

    def __del__(self):
        L = getattr(self, "L", None)
        if L is None:
            return
        try:
            if getattr(self, "batch", None) is not None: L.llama_batch_free(self.batch)
            if getattr(self, "ctx", None): L.llama_free(self.ctx)
            if getattr(self, "model", None): L.llama_model_free(self.model)
        except Exception:
            pass

    def _decode(self, rows):
        """rows: list of (ids, slots).  One llama_decode, row r as sequence r.  -> [sum(len(slots)), K] letter logits."""
        L, b = self.L, self.batch
        L.llama_memory_clear(L.llama_get_memory(self.ctx), True)
        n, where = 0, []
        for r, (ids, slots) in enumerate(rows):
            want = set(slots)
            for i, t in enumerate(ids):
                b.token[n] = t; b.pos[n] = i; b.n_seq_id[n] = 1; b.seq_id[n][0] = r; b.logits[n] = i in want
                n += 1
            where.extend(n - len(ids) + s for s in slots)
        b.n_tokens = n
        rc = L.llama_decode(self.ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc}")
        self.stats["forwards"] += 1
        out = np.empty((len(where), len(self.letters)), dtype=np.float32)
        for k, i in enumerate(where):
            p = ctypes.cast(L.llama_get_logits_ith(self.ctx, i), ctypes.POINTER(ctypes.c_float))
            out[k] = np.ctypeslib.as_array(p, shape=(self.n_vocab,))[self.letters]
        return out

    def _groups(self, items):
        """Consecutive rows per decode: at most n_seq_max rows and n_batch tokens together; a row longer than n_batch goes alone
        (up to n_ctx, which the context's own n_batch covers)."""
        group, tokens = [], 0
        for i, it in enumerate(items):
            n = len(it["ids"])
            if n > self.n_ctx:
                raise ValueError(f"prompt row of {n} tokens exceeds n_ctx {self.n_ctx}")
            if group and (len(group) == self.n_seq_max or tokens + n > self.n_batch):
                yield group; group, tokens = [], 0
            group.append(i); tokens += n
        if group:
            yield group

    @torch.no_grad()
    def score_items(self, items, temperature=1.0):
        """items: list of dicts from prompt.build.  Returns list of [n_q, MAX_OPTIONS] prob tensors (cpu), as Engine.score_items."""
        lg = np.empty((sum(len(it["slots"]) for it in items), len(self.letters)), dtype=np.float32)
        offsets = np.cumsum([0] + [len(it["slots"]) for it in items])
        for g in self._groups(items):
            out = self._decode([(items[i]["ids"], items[i]["slots"]) for i in g])
            k = 0
            for i in g:
                m = len(items[i]["slots"]); lg[offsets[i]:offsets[i] + m] = out[k:k + m]; k += m
        lg = torch.from_numpy(lg)
        if lg.shape[1] < MAX_OPTIONS:
            lg = torch.cat([lg, torch.full((lg.shape[0], MAX_OPTIONS - lg.shape[1]), float("-inf"))], 1)
        nopts = torch.tensor([n for it in items for n in it["nopts"]])
        lg = lg.masked_fill(torch.arange(MAX_OPTIONS)[None, :] >= nopts[:, None], float("-inf"))
        p = scaled_softmax(lg, slot_temperatures(temperature, items))
        return list(torch.split(p, [len(it["slots"]) for it in items]))

    def score_shared(self, items, temperature=1.0, min_prefix=192):
        return self.score_items(items, temperature)

    def warmup(self, *a, **k):
        return 0.0
