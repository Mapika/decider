"""decider.batching.plan_batches: the server's cross-request batch partition (pure python, no torch).

The properties every plan must have, checked over random row sets against the engine's real bucket ladder: every row is
covered exactly once, a group's padded length is at least every member's own, a group is within the cap, and the plan
never costs more than the per-bucket grouping it replaces.
"""
import random

from decider.batching import (DEFAULT_MERGE_OVERHEAD_TOKENS, per_bucket_groups, plan_batches, plan_cost)

T_BUCKETS = [64, 128, 192, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536, 2048, 3072, 4096, 6144, 8192]
B_BUCKETS = [1, 2, 4, 8, 16, 32]
TOKEN_BUDGET = 32768
LONG_STEP = 1024
MAX_BATCH = 32


def t_bucket(n):
    for t in T_BUCKETS:
        if n <= t:
            return t
    return None


def pad_len(n):
    return t_bucket(n) or -(-n // LONG_STEP) * LONG_STEP


def max_rows(T):
    av = [B for B in B_BUCKETS if B == 1 or B * T <= TOKEN_BUDGET] if t_bucket(T) == T else []
    return av[-1] if av else max(1, TOKEN_BUDGET // max(T, 1))


def mergeable(n):
    return t_bucket(n) is not None


def check(lengths, groups, overhead=DEFAULT_MERGE_OVERHEAD_TOKENS):
    covered = sorted(i for _, idx in groups for i in idx)
    assert covered == list(range(len(lengths))), (covered, lengths)
    for T, idx in groups:
        assert idx, "empty group"
        assert T >= max(pad_len(lengths[i]) for i in idx), (T, [lengths[i] for i in idx])
        assert len(idx) <= max(1, min(MAX_BATCH, max_rows(T))), (T, len(idx))
        assert len(set(idx)) == len(idx)
    base = per_bucket_groups(lengths, pad_len, max_rows, MAX_BATCH)
    assert plan_cost(groups, overhead) <= plan_cost(base, overhead)


def plan(lengths, overhead=DEFAULT_MERGE_OVERHEAD_TOKENS):
    return plan_batches(lengths, pad_len, max_rows, MAX_BATCH, overhead, mergeable)


def test_empty_and_single():
    assert plan([]) == []
    g = plan([100])
    assert g == [(128, [0])]


def test_close_lengths_merge_into_one_forward():
    """Four rows in neighbouring buckets: one forward at 256 costs 512 + 1024, four forwards cost 2048 + 832."""
    lengths = [200, 130, 70, 250]
    g = plan(lengths)
    check(lengths, g)
    assert len(g) == 1 and g[0][0] == 256 and g[0][1] == [0, 1, 2, 3]


def test_a_long_row_is_not_merged_with_short_ones():
    lengths = [8000, 60, 60, 60, 60]
    g = plan(lengths)
    check(lengths, g)
    by_T = {T: idx for T, idx in g}
    assert by_T[8192] == [0]
    assert sorted(i for T, idx in g if T != 8192 for i in idx) == [1, 2, 3, 4]


def test_unbucketed_rows_keep_their_own_forward():
    lengths = [12000, 12000, 100]
    g = plan(lengths)
    check(lengths, g)
    assert (12288, [0, 1]) in g                                # same padded length, still grouped as before
    assert (128, [2]) in g                                     # the short row is not padded to 12288


def test_group_size_is_capped_by_the_engine_grid():
    lengths = [4000] * 10                                      # max_rows(4096) = 8 (8 * 4096 = 32768)
    g = plan(lengths)
    check(lengths, g)
    assert sorted(len(idx) for _, idx in g) == [2, 8]


def test_max_batch_is_respected():
    lengths = [60] * 40
    g = plan(lengths)
    check(lengths, g)
    assert all(len(idx) <= MAX_BATCH for _, idx in g)


def test_arrival_order_inside_a_group():
    lengths = [250, 60, 130]
    g = plan(lengths)
    assert g == [(256, [0, 1, 2])]


def test_a_huge_overhead_merges_everything_bucketed():
    lengths = [60, 300, 900, 2000]
    g = plan(lengths, overhead=10 ** 9)
    check(lengths, g, overhead=10 ** 9)
    assert len(g) == 1 and g[0][0] == 2048


def test_a_zero_overhead_never_pays_padding():
    lengths = [60, 300, 900, 2000]
    g = plan(lengths, overhead=0)
    check(lengths, g, overhead=0)
    assert sorted(T for T, _ in g) == [64, 320, 1024, 2048]


def test_random_row_sets():
    rng = random.Random(0)
    for trial in range(400):
        n = rng.randint(1, 40)
        lengths = []
        for _ in range(n):
            r = rng.random()
            lengths.append(rng.randint(1, 300) if r < 0.6 else rng.randint(300, 9000) if r < 0.95 else rng.randint(9000, 40000))
        over = rng.choice([0, 128, 512, 1024, 4096])
        g = plan(lengths, overhead=over)
        check(lengths, g, overhead=over)


def test_plan_is_at_least_as_cheap_as_the_baseline_on_mixed_concurrency():
    """The case the change is for: eight requests of different lengths arriving together."""
    lengths = [190, 200, 210, 1100, 1150, 60, 64, 900]
    g = plan(lengths)
    check(lengths, g)
    base = per_bucket_groups(lengths, pad_len, max_rows, MAX_BATCH)
    assert len(g) < len(base)
    assert plan_cost(g) < plan_cost(base)
