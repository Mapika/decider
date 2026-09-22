"""Batch planning for the server's queue: which queued rows share one forward.

Pure python, no torch, no engine: `plan_batches` takes the row lengths and two callables that describe the engine's
bucket grid, and returns the groups to run.  `decider.serve.batcher` calls it once per collection.

Cost model.  A forward of B rows padded to length T costs `overhead + B * T` token-units: `overhead` is the fixed cost of
one forward (launch, replay, the slot gather and the device-to-host copy) expressed as the number of padded tokens that
take the same time, and `B * T` is the padded token work.  Two rows of different lengths are therefore worth merging
into the longer row's bucket exactly when the padding they add is cheaper than a second forward's overhead.
`DEFAULT_MERGE_OVERHEAD_TOKENS` is measured on decider-2b (docs/SERVING.md, section on the batching policy); the server
reads `DECIDER_MERGE_OVERHEAD_TOKENS` over it.

The partition is exact, not greedy.  Rows are sorted by padded length descending (stable, so rows of the same bucket keep
their arrival order) and split into consecutive groups; each group runs at the bucket of its longest member, so a group
starting at position k costs `overhead + g * T[k]`.  A dynamic program over the sorted sequence takes the cheapest split,
subject to `g <= min(max_batch, max_rows(T))`, with ties going to the larger group.  The per-bucket grouping the server
did before is one of the partitions the program may choose (rows of equal length are adjacent in the sorted order), so
the planned cost is never above it, and with `overhead = 0` the plan is exactly that grouping.
"""

DEFAULT_MERGE_OVERHEAD_TOKENS = 512


def group_cap(T, max_rows, max_batch):
    """Rows one forward may take at padded length T: the engine's widest captured batch bucket, and the server's cap."""
    return max(1, min(int(max_batch), int(max_rows(T))))


def batch_cost(size, T, overhead=DEFAULT_MERGE_OVERHEAD_TOKENS):
    """Cost of one forward of `size` rows padded to `T`, in token-units."""
    return overhead + size * T


def plan_cost(groups, overhead=DEFAULT_MERGE_OVERHEAD_TOKENS):
    """Cost of a plan: the sum over its forwards."""
    return sum(batch_cost(len(idx), T, overhead) for T, idx in groups)


def per_bucket_groups(lengths, pad_len, max_rows, max_batch):
    """The grouping of 1.1.0: rows of the same padded length, chunked to the cap.  The baseline `plan_batches` improves on."""
    return _exact_bucket(list(range(len(lengths))), [pad_len(n) for n in lengths], max_rows, max_batch)


def _exact_bucket(idx, pads, max_rows, max_batch):
    buckets = {}
    for i in idx:
        buckets.setdefault(pads[i], []).append(i)
    out = []
    for T, members in buckets.items():
        cap = group_cap(T, max_rows, max_batch)
        for k in range(0, len(members), cap):
            out.append((T, members[k:k + cap]))
    return out


def plan_batches(lengths, pad_len, max_rows, max_batch, overhead=DEFAULT_MERGE_OVERHEAD_TOKENS, mergeable=None):
    """Partition queued rows into forwards.

    lengths:    token count of every queued row, in arrival order.
    pad_len(n): the engine's padded length for a row of n tokens (`EngineV2.pad_len`).
    max_rows(T):rows the engine runs in one forward at padded length T (`EngineV2.max_rows`).
    max_batch:  the server's own cap (`DECIDER_MAX_BATCH`).
    overhead:   fixed cost of a forward in token-units (see the module docstring).
    mergeable(n): False for a row that must not be padded into another row's bucket.  Rows above the last captured
                  length bucket run eager at a request-specific shape, so they are grouped by exact length as before.

    -> [(padded length, [row indices]), ...].  Every row appears in exactly one group; a group's padded length is at
    least every member's own padded length; a group holds at most `min(max_batch, max_rows(T))` rows.
    """
    n = len(lengths)
    if n == 0:
        return []
    pads = [pad_len(x) for x in lengths]
    merge = [True] * n if mergeable is None else [bool(mergeable(x)) for x in lengths]
    out = _exact_bucket([i for i in range(n) if not merge[i]], pads, max_rows, max_batch)

    order = sorted((i for i in range(n) if merge[i]), key=lambda i: -pads[i])    # stable: equal buckets keep arrival order
    m = len(order)
    dp = [0] * (m + 1)                       # dp[k]: cheapest cost of covering order[k:]
    take = [0] * (m + 1)
    for k in range(m - 1, -1, -1):
        T = pads[order[k]]
        cap = min(group_cap(T, max_rows, max_batch), m - k)
        best, best_g = None, 1
        for g in range(1, cap + 1):
            c = overhead + g * T + dp[k + g]
            if best is None or c <= best:        # ties go to the larger group: same modelled cost, one fewer launch
                best, best_g = c, g
        dp[k], take[k] = best, best_g
    k = 0
    while k < m:
        g = take[k]
        out.append((pads[order[k]], sorted(order[k:k + g])))                     # inside a group, arrival order
        k += g
    return out
