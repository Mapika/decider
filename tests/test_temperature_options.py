"""temperature_by_options (1.8.0): T(n) = max(min, a + b ln n) per question, from decider_config.json."""
import math
import pytest
from decider import temperature as TT


def test_from_config_reads_the_rule_and_for_items_gives_one_T_per_slot():
    (T, m), (Ts, ms) = TT.from_config({"temperature": 3.0, "temperature_by_options": {"a": 10.124, "b": -1.633}})
    assert T == 3.0 and m == {"by_options": (10.124, -1.633, 0.05)} and ms == {}          # schema cache keeps the scalar
    items = [{"slots": [5, 9], "nopts": [2, 10]}, {"slots": [3], "nopts": [151]}]
    got = TT.for_items(T, m, items)
    assert got[0] == [pytest.approx(10.124 - 1.633 * math.log(2)), pytest.approx(10.124 - 1.633 * math.log(10))]
    assert got[1] == [pytest.approx(max(0.05, 10.124 - 1.633 * math.log(151)))]
    assert TT.slot_temperatures(got, items) == got[0] + got[1]


def test_floor_one_option_and_reporting():
    spec = TT.by_options({"a": 1.0, "b": -1.0, "min": 0.2}, "x")
    assert TT.T_of_options(spec, 1) == TT.T_of_options(spec, 2) and TT.T_of_options(spec, 50) == 0.2
    assert "ln n_options" in TT.effective(1.0, {"by_options": spec})["noul"]


def test_override_switches_the_rule_off_and_bad_specs_are_refused():
    (T, m), _ = TT.from_config({"temperature_by_options": {"a": 2.0, "b": 0.0}}, temperature=1.5)
    assert (T, m) == (1.5, {})
    for bad in ({"a": 1.0}, {"a": 1.0, "b": float("nan")}, {"a": 1, "b": 1, "min": 0}, {"a": 1, "b": 1, "c": 2}, [1, 2]):
        with pytest.raises(ValueError):
            TT.from_config({"temperature_by_options": bad})
    with pytest.raises(ValueError):
        TT.from_config({"temperature_by_options": {"a": 1, "b": 0}, "temperature_by_type": {"noul": 1.0}})
