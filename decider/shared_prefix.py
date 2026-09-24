"""Shared-prefix scoring with a bounded memory fork, used by both engines.

Rows that start with the same tokens (one state, one question per row) run the shared prefix once and then only the
question suffixes, against a copy of the prefix cache.  Until 1.1.0 the copy was made with
`cache.reorder_cache(zeros(n))`, which expands the prefix to all n rows at once: a 31k-token state with 32 questions
costs n times the prefix cache, which is 133 GB on a 31B model and is wasteful on the 2B.  Here the prefix cache is
forked in chunks of m rows, m chosen so that one fork fits a byte budget, and the rows are scored chunk by chunk.

The budget is `DECIDER_SHARED_FORK_GB` (8 GB), capped at half of the memory currently free on the device (device-free
plus the caching allocator's reserved-but-unused blocks); `m = clamp(budget // prefix_bytes, 1, n)`.  Chunking changes
which rows share a forward and how far each chunk's suffixes are padded, so answers can move by the usual bf16
reduction-order amount; they do not depend on it mathematically (right padding, causal layers).

Cache layout.  transformers 5.17 keeps one layer object per layer in `cache.layers`.  An attention layer carries
`keys` and `values` tensors `[batch, heads, seq, head_dim]`; a sparse-attention layer adds `indexer_keys`
`[batch, seq, dim]`; a linear-attention layer carries `conv_states` and `recurrent_states` dicts of tensors whose first
dimension is the batch.  Qwen3.5 gives a `DynamicCache` whose 24 layers are a mix of the two objects.  `cache_state_tensors`
enumerates whatever of `STATE_NAMES` is present rather than assuming a layout, and `fork_cache` builds a new cache
object from them without touching the original (the linear-attention layers update their states with `copy_()`, so a
fork that shared storage with the prefix would corrupt it for the next chunk).

A layout we do not know is not chunked.  If a layer holds any other tensor attribute, or a dict of tensors, outside
`STATE_NAMES`, we cannot tell whether it carries a batch dimension, so `score_shared` takes one fork of all n rows --
the 1.1.0 behaviour, correct but unbounded -- and counts it in `engine.stats["shared_unchunked_layout"]`.

The prefix and suffix forwards are eager, at request-specific shapes, and are correct only with the cuDNN SDPA backend
off (`decider.engine.set_attention_backend_policy`, applied in both engines' `__init__`).
"""
import copy
import os

import torch
import torch.nn.functional as F

from decider.engine import fill_ids, read_slots
from decider.temperature import item_slice, slot_temperatures

DEFAULT_FORK_GB = 8.0
STATE_NAMES = ("keys", "values", "indexer_keys", "conv_states", "recurrent_states")


def common_prefix_len(ids):
    """Length of the longest common prefix of the rows, capped one token below the shortest row."""
    lcp = 0
    short = min(len(x) for x in ids) - 1
    while lcp < short and all(x[lcp] == ids[0][lcp] for x in ids):
        lcp += 1
    return lcp


def _get(container, key):
    return container[key] if isinstance(container, dict) else getattr(container, key)


def _set(container, key, value):
    if isinstance(container, dict):
        container[key] = value
    else:
        setattr(container, key, value)


def cache_state_tensors(cache):
    """Every tensor in the cache whose first dimension is the batch, as (container, key) pairs.

    Covers `STATE_NAMES`: the attention layers' `keys`/`values`, a sparse-attention layer's `indexer_keys` and the
    linear-attention layers' `conv_states`/`recurrent_states`, whether those are dicts of tensors (transformers 5.17) or
    single tensors, and ignores anything not present."""
    out = []
    for layer in getattr(cache, "layers", None) or []:
        for attr in STATE_NAMES:
            v = getattr(layer, attr, None)
            if isinstance(v, torch.Tensor):
                if v.numel():
                    out.append((layer, attr))
            elif isinstance(v, dict):
                for k, t in v.items():
                    if isinstance(t, torch.Tensor) and t.numel():
                        out.append((v, k))
    return out


def cache_row_bytes(cache):
    """Bytes one row of this cache holds, summed over the layers and over `STATE_NAMES`."""
    return sum(_get(c, k).nbytes // max(_get(c, k).shape[0], 1) for c, k in cache_state_tensors(cache))


def unknown_state_names(cache):
    """Attribute names the cache's layers hold that carry tensors and are not in `STATE_NAMES`.

    A tensor outside the enumerated set may or may not have a batch dimension, and `fork_cache` would leave it at one
    row.  When this is not empty the caller must not chunk."""
    out = set()
    for layer in getattr(cache, "layers", None) or []:
        for name, v in vars(layer).items():
            if name in STATE_NAMES:
                continue
            if isinstance(v, torch.Tensor) or (isinstance(v, dict) and any(isinstance(t, torch.Tensor) for t in v.values())):
                out.add(name)
    return out


def fork_budget_bytes(device=None, gb=None):
    """DECIDER_SHARED_FORK_GB, capped at half of the memory free on the device right now.

    The cap applies only when the CUDA memory queries succeed: on CPU, on MPS and when the driver does not know the
    device string, the configured budget is kept as it is."""
    gb = float(os.environ.get("DECIDER_SHARED_FORK_GB", DEFAULT_FORK_GB)) if gb is None else float(gb)
    budget = int(gb * (1 << 30))
    try:
        free, _ = torch.cuda.mem_get_info(device)
        free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        budget = min(budget, free // 2)
    except Exception:                       # no CUDA device, or a device string the driver does not know
        pass
    return max(int(budget), 1)


def chunk_rows(prefix_bytes, n, budget_bytes=None, device=None):
    """Rows per fork: as many copies of the prefix cache as the budget holds, at least 1 and at most n.  One row is the
    minimum even when a single copy is over the budget: the budget bounds the fork, it cannot make it free."""
    if budget_bytes is None:
        budget_bytes = fork_budget_bytes(device)
    if prefix_bytes <= 0:
        return max(1, int(n))
    return max(1, min(int(n), int(budget_bytes // prefix_bytes)))


def fork_cache(cache, m, row=0):
    """A new cache holding `m` copies of `cache`'s row `row`.  The original is not read from again and not modified."""
    fork = copy.copy(cache)
    layers = getattr(cache, "layers", None)
    if layers is not None:
        new = []
        for layer in layers:
            nl = copy.copy(layer)
            for k, v in list(vars(nl).items()):          # the per-state dicts are mutated by the forward: give the fork its own
                if isinstance(v, dict):
                    setattr(nl, k, dict(v))
            new.append(nl)
        fork.layers = new
    idx = {}
    for container, key in cache_state_tensors(fork):
        t = _get(container, key)
        i = idx.get(t.device)
        if i is None:
            i = idx[t.device] = torch.full((m,), row, dtype=torch.long, device=t.device)
        _set(container, key, t.index_select(0, i))       # a fresh contiguous tensor: the fork never shares storage
    return fork


def _count(engine, key):
    stats = getattr(engine, "stats", None)
    if isinstance(stats, dict):
        stats[key] = stats.get(key, 0) + 1


@torch.no_grad()
def score_shared(engine, items, temperature=1.0, min_prefix=192, budget_bytes=None, rows_per_fork=None):
    """Score `items` through the shared prefix.  -> one probability tensor per item, in item order, or None when the
    request does not qualify (fewer than two rows, or a common prefix below `min_prefix`) and the caller should use
    `score_items`.

    `temperature`: a number, or one entry per item (decider.temperature.slot_temperatures).
    `rows_per_fork` forces the chunk size; it exists for the tests that compare chunked against unchunked answers."""
    ids = [it["ids"] for it in items]
    n = len(ids)
    if n < 2:
        return None
    lcp = common_prefix_len(ids)
    if lcp < min_prefix:
        return None
    slot_temperatures(temperature, items)                  # a length mismatch fails before any forward
    core, W, dev, pad = engine.core, engine.W, engine.dev, engine.tok.pad_token_id
    pre = torch.tensor(ids[0][:lcp], device=dev)[None]
    cache = core(input_ids=pre, use_cache=True).past_key_values
    unknown = unknown_state_names(cache)
    if unknown:                                  # a state we cannot fork row by row: one fork of everything, as in 1.1.0
        _count(engine, "shared_unchunked_layout")
        m = n
    elif rows_per_fork:
        m = int(rows_per_fork)
    else:
        m = chunk_rows(cache_row_bytes(cache), n, budget_bytes, dev)
    m = max(1, min(m, n))
    out = []
    for i in range(0, n, m):
        part = items[i:i + m]
        b = len(part)
        fork = fork_cache(cache, b)
        Ts = max(len(it["ids"]) for it in part) - lcp
        suf = fill_ids([it["ids"][lcp:] for it in part], b, Ts, pad)
        h = core(input_ids=suf.to(dev), past_key_values=fork, use_cache=True).last_hidden_state
        rows = [j for j, it in enumerate(part) for _ in it["slots"]]
        sl = [s - lcp for it in part for s in it["slots"]]
        idx = torch.tensor([rows, sl], device=dev)
        out += read_slots(F.linear(h[idx[0], idx[1]], W).float()[:, None, :], list(range(len(rows))), [0] * len(rows),
                          [k for it in part for k in it["nopts"]],
                          slot_temperatures(item_slice(temperature, i, i + b), part), [len(it["slots"]) for it in part])
        del fork, h, suf                                  # drop this chunk's fork before the next one is built
    return out
