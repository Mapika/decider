"""decider.serve_vllm second reading (decider_config.json "second_reading"), without vLLM and without a GPU.

* the config block parses to (below, a, b, min); absent, empty or below 0 means one reading; the env override wins.
* second_pass re-reads only the rows below the threshold, with their options reversed, averages the two letter log-softmaxes
  (the reversed one mapped back to the given order) and takes the softmax at the second reading's T(n); other rows keep the
  first reading exactly.
"""
import asyncio, math

import pytest


def _V():
    pytest.importorskip("fastapi")
    import decider.serve_vllm as V
    return V


def test_second_reading_config():
    V = _V()
    assert V.second_reading(None) is None
    assert V.second_reading({}) is None
    spec = {"below": 0.7, "temperature_by_options": {"a": 1.678, "b": 0.067, "min": 0.05}}
    assert V.second_reading(spec) == (0.7, 1.678, 0.067, 0.05)
    assert V.second_reading(spec, "0") is None
    assert V.second_reading(spec, "0.9")[0] == 0.9
    with pytest.raises(ValueError):
        V.second_reading({"below": 0.7})


def test_second_pass_rereads_only_unsure_rows(monkeypatch):
    V = _V()
    T1 = 2.0; a, b, lo = 1.0, 0.0, 0.05
    first_lp = [[0.0, -3.0, -4.0], [0.0, -0.1]]                # row 0 sure, row 1 unsure after the first reading
    probs = [V.softmax_T(x, T1) for x in first_lp]
    assert max(probs[0]) >= 0.7 > max(probs[1])
    rev_lp = {1: [-0.05, 0.0]}                               # reversed order: [option 1, option 0]
    items = [{"ids": [1, 2, 3], "nopts": [3], "slots": [2]}, {"ids": [1, 2, 4], "nopts": [2], "slots": [2]}]
    rev_items = [{"ids": [9, 9, 9], "nopts": [3], "slots": [2]}, {"ids": [8, 8, 8], "nopts": [2], "slots": [2]}]
    asked = []

    async def fake_logprobs(ids, n):
        i = 0 if ids == [9, 9, 9] else 1; asked.append(i); return rev_lp[i]
    monkeypatch.setattr(V, "SECOND", (0.7, a, b, lo))
    monkeypatch.setattr(V, "TEMP", T1); monkeypatch.setattr(V, "TEMP_BY_TYPE", {})
    monkeypatch.setattr(V, "_prepare_rev", lambda s, q: rev_items)
    monkeypatch.setattr(V, "_logprobs", fake_logprobs)
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setattr(V, "cpu", ThreadPoolExecutor(1))
    out = asyncio.run(V.second_pass({}, {}, items, probs))
    assert asked == [1]
    assert out[0] == probs[0]
    lsm = V._lsm
    want = V.softmax_T([(x + y) / 2 for x, y in zip(lsm(first_lp[1]), lsm(rev_lp[1][::-1]))], max(lo, a + b * math.log(2)))
    assert max(abs(x - y) for x, y in zip(out[1], want)) < 1e-9
