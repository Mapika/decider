"""The rule-conditioned data: labels are computed, twins flip them, every gold index is valid."""
import collections
from decider.data import rules as R


def test_build_is_deterministic_and_well_formed():
    a, b = R.build(400, seed=3), R.build(400, seed=3)
    assert [(e.context, e.qs[0].text, e.qs[0].gold) for e in a] == [(e.context, e.qs[0].text, e.qs[0].gold) for e in b]
    for e in a:
        q = e.qs[0]
        assert 2 <= len(q.options) <= 255 and 0 <= q.gold < len(q.options) and len(set(q.options)) == len(q.options)
    assert len({e.task for e in a}) >= 8


def test_twins_share_text_and_disagree_on_the_label():
    ex = R.build(1500, seed=5)
    by_q = collections.defaultdict(list)
    for e in ex:
        if e.task.endswith(("threshold", "boolean", "presence", "membership", "contains")):
            by_q[(e.qs[0].text, tuple(e.qs[0].options))].append(e)
    pairs = [v for v in by_q.values() if len(v) >= 2]
    assert len(pairs) > 100
    flips = sum(len({e.qs[0].gold for e in v}) > 1 for v in pairs)
    assert flips / len(pairs) > 0.9                              # state twins: same rule and options, the label flips
    golds = collections.Counter(e.qs[0].gold for e in ex if e.task.endswith("boolean"))
    assert min(golds.values()) / sum(golds.values()) > 0.4      # balanced


def test_held_out_probes_do_not_leak_into_training():
    train = R.build(600, seed=1); pr = R.probe_sets(seed=9)
    train_ctx = {e.context for e in train}
    assert not any(e.context in train_ctx for k, v in pr.items() if k != "rules_form" for e in v)   # rules_form is in-distribution by design
    for e in pr["rules_held_domain"]:
        assert any(k in e.context.lower() for k in ("carrier", "weight", "attendees", "temperature", "humidity", "organizer", "zone", "driver", "records"))
