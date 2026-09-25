"""decider.serve_vllm and the DECIDER_LAYOUT override, without vLLM and without a GPU.

* prompt.with_layout: a stock checkpoint (no decider_config.json) is read in the chat layout with DECIDER_LAYOUT=chat, the
  override wins over the config in both directions, and an unknown value is refused.
* serve_vllm.softmax_T over log-softmax values equals softmax(logits / T) over the same letters (the log-softmax differs from
  the logit by one constant per position), which is why reading vLLM's logprobs keeps the readout of decider.serve.
* rows with more than 128 options are split into slices of at most 128 letter ids and joined in label order.
* score_rows: one slot at the end of every row, the first row alone when the rows share a long prefix, per-type temperatures.
"""
import asyncio, math, random

import pytest

from decider import prompt as P


def test_with_layout_override():
    assert P.resolve_layout(P.with_layout({}, "chat")) == "chat"
    assert P.resolve_layout(P.with_layout({}, None)) == "plain"
    assert P.resolve_layout(P.with_layout({"layout": "chat"}, "plain")) == "plain"
    assert P.resolve_layout(P.with_layout({"chat_template": True}, "plain")) == "plain"      # no contradiction left behind
    assert P.resolve_layout(P.with_layout({"layout": "chat"}, "")) == "chat"
    with pytest.raises(ValueError):
        P.with_layout({}, "schema")


def test_serve_layout_env(monkeypatch):
    pytest.importorskip("fastapi")
    import decider.serve as S
    monkeypatch.setenv("DECIDER_LAYOUT", "chat")
    S.apply_config({})
    assert S.LAYOUT == "chat"
    monkeypatch.setenv("DECIDER_LAYOUT", "bogus")
    with pytest.raises(ValueError):
        S.apply_config({})
    monkeypatch.delenv("DECIDER_LAYOUT")
    S.apply_config({})
    assert S.LAYOUT == "plain"


def _V():
    pytest.importorskip("fastapi")
    import decider.serve_vllm as V
    return V


def test_softmax_of_logprobs_equals_softmax_of_logits():
    V = _V()
    torch = pytest.importorskip("torch")
    rng = random.Random(0)
    for n in (2, 5, 151):
        logits = torch.tensor([rng.gauss(0, 4) for _ in range(1000)], dtype=torch.float64)
        letters = list(range(n))
        lp = torch.log_softmax(logits, -1)[letters].tolist()
        for T in (1.0, 1.943, 0.5):
            want = torch.softmax(logits[letters] / T, -1).tolist()
            got = V.softmax_T(lp, T)
            assert max(abs(a - b) for a, b in zip(want, got)) < 1e-12


class _Out:
    def __init__(self, lp): self.logprobs = [lp]


class _Final:
    def __init__(self, lp): self.outputs = [_Out(lp)]


class _LP:
    def __init__(self, x): self.logprob = x


class FakeEngine:
    """generate(prompt, sampling params) -> one output with the log-softmax of fixed per-row logits at the requested ids."""
    def __init__(self, logits_of):
        self.logits_of = logits_of; self.calls = []

    async def generate(self, prompt, sp, request_id):
        ids = prompt["prompt_token_ids"]
        self.calls.append((len(ids), list(sp.logprob_token_ids)))
        lg = self.logits_of(ids)
        m = max(lg.values()); z = m + math.log(sum(math.exp(v - m) for v in lg.values()))
        yield _Final({t: _LP(lg[t] - z) for t in sp.logprob_token_ids})


def _setup(V, monkeypatch, n_letters=255):
    vllm = pytest.importorskip("vllm")               # SamplingParams / TokensPrompt only; no GPU
    letters = list(range(1000, 1000 + n_letters))
    monkeypatch.setattr(V, "LETTER_IDS", letters)
    rng = random.Random(1)
    table = {t: rng.gauss(0, 3) for t in letters}
    table.update({t: rng.gauss(0, 3) for t in range(50)})
    eng = FakeEngine(lambda ids: {t: table[t] + 0.01 * len(ids) for t in table})
    monkeypatch.setattr(V, "engine", eng)
    return letters, table, eng


def test_wide_rows_are_split_and_joined(monkeypatch):
    V = _V()
    letters, table, eng = _setup(V, monkeypatch)
    lp = asyncio.run(V._logprobs([1, 2, 3], 200))
    assert [len(c[1]) for c in eng.calls] == [128, 72]
    assert eng.calls[0][1] + eng.calls[1][1] == letters[:200]
    T = 1.943
    want = [math.exp(table[t] / T) for t in letters[:200]]; s = sum(want)
    got = V.softmax_T(lp, T)
    assert max(abs(a / s - b) for a, b in zip(want, got)) < 1e-12


class GatedEngine(FakeEngine):
    """FakeEngine whose generate() waits on an event per prompt length, and records start order."""
    def __init__(self, logits_of):
        super().__init__(logits_of); self.started = []; self.gates = {}

    async def generate(self, prompt, sp, request_id):
        n = len(prompt["prompt_token_ids"]); self.started.append(n)
        gate = self.gates.setdefault(n, asyncio.Event())
        try:
            await gate.wait()
        except asyncio.CancelledError:
            self.cancelled = getattr(self, "cancelled", 0) + 1; raise
        async for out in FakeEngine.generate(self, prompt, sp, request_id):
            yield out


def test_score_rows_prefill_first_waits_and_temperatures(monkeypatch):
    V = _V()
    letters, table, _ = _setup(V, monkeypatch)
    lg = lambda ids: {t: table[t] + 0.01 * len(ids) for t in table}
    eng = GatedEngine(lg); monkeypatch.setattr(V, "engine", eng)
    monkeypatch.setattr(V, "TEMP", 2.0); monkeypatch.setattr(V, "TEMP_BY_TYPE", {"noul": 1.0})
    monkeypatch.setattr(V, "PREFILL_FIRST", 4); monkeypatch.setattr(V, "BLOCK", 8)
    base = list(range(10))
    items = [dict(ids=base + list(range(20, 21 + k)), slots=[10 + k], nopts=[2 + k], types=["noul" if k == 0 else "choice"]) for k in range(3)]

    async def go():
        t = asyncio.ensure_future(V.score_rows(items))
        for _ in range(5): await asyncio.sleep(0)
        assert eng.started == [11]                       # only the first row runs until it finishes
        eng.gates[11].set()
        for _ in range(5): await asyncio.sleep(0)
        assert sorted(eng.started) == [11, 12, 13]       # then the others, together
        eng.gates[12].set(); eng.gates[13].set()
        return await t
    out = asyncio.run(go())
    for k, (o,) in enumerate(out):
        T = 1.0 if k == 0 else 2.0                       # noul row at its type temperature, choice rows at the default
        x = [lg(items[k]["ids"])[t] / T for t in letters[:2 + k]]; m = max(x); e = [math.exp(v - m) for v in x]
        assert max(abs(a - b / sum(e)) for a, b in zip(o, e)) < 1e-12


def test_score_rows_short_prefix_runs_together_and_cancels(monkeypatch):
    V = _V()
    letters, table, _ = _setup(V, monkeypatch)
    eng = GatedEngine(lambda ids: {t: table[t] for t in table}); monkeypatch.setattr(V, "engine", eng)
    monkeypatch.setattr(V, "PREFILL_FIRST", 512); monkeypatch.setattr(V, "BLOCK", 784)
    items = [dict(ids=[1, 2, 3 + k] + [0] * k, slots=[2 + k], nopts=[2], types=["choice"]) for k in range(3)]

    async def go():
        t = asyncio.ensure_future(V.score_rows(items))
        for _ in range(5): await asyncio.sleep(0)
        assert sorted(eng.started) == [3, 4, 5]          # no shared block: all rows at once
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert eng.cancelled == 3                        # every row's generate saw the cancellation
    asyncio.run(go())


def test_check_size_limits(monkeypatch):
    V = _V()
    from fastapi import HTTPException
    monkeypatch.setattr(V, "MAX_ROWS", 3); monkeypatch.setattr(V, "MAX_ROW_TOKENS", 100)
    monkeypatch.setattr(V, "MAX_MODEL_LEN", 90); monkeypatch.setattr(V, "MAX_REQUEST_TOKENS", 150)
    V.check_size([10, 20])
    for lens, marker in (([1, 1, 1, 1], "too many questions"), ([101], "too many tokens"), ([95], "maximum model length"),
                         ([80, 80], "too many tokens")):
        with pytest.raises(HTTPException) as e:
            V.check_size(lens)
        assert e.value.status_code == 413 and marker in e.value.detail


def test_score_rows_order_and_prefill_first(monkeypatch):
    V = _V()
    _setup(V, monkeypatch)
    monkeypatch.setattr(V, "TEMP", 2.0); monkeypatch.setattr(V, "TEMP_BY_TYPE", {"noul": 1.0})
    monkeypatch.setattr(V, "PREFILL_FIRST", 4); monkeypatch.setattr(V, "BLOCK", 8)
    base = list(range(10))
    items = [dict(ids=base + [20 + k], slots=[10], nopts=[2 + k], types=["choice" if k else "noul"]) for k in range(3)]
    V.stats["prefill_first"] = 0
    out = asyncio.run(V.score_rows(items))
    assert [len(o[0]) for o in out] == [2, 3, 4]
    assert V.stats["prefill_first"] == 1
    assert all(abs(sum(o[0]) - 1) < 1e-12 for o in out)
    bad = [dict(ids=[1, 2, 3], slots=[1], nopts=[2], types=["choice"])]
    with pytest.raises(ValueError):
        asyncio.run(V.score_rows(bad))


class _FakeRequest:
    """ASGI request stand-in: receive() blocks until disconnect() is called."""
    def __init__(self): self.ev = asyncio.Event()

    def disconnect(self): self.ev.set()

    async def receive(self):
        await self.ev.wait()
        return {"type": "http.disconnect"}


def _route_setup(V, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    letters, table, _ = _setup(V, monkeypatch)
    eng = GatedEngine(lambda ids: {t: table[t] for t in table}); monkeypatch.setattr(V, "engine", eng)
    monkeypatch.setattr(V, "cpu", ThreadPoolExecutor(1)); monkeypatch.setattr(V, "outstanding", 0)
    monkeypatch.setattr(V, "PREFILL_FIRST", 512); monkeypatch.setattr(V, "BLOCK", 784)
    items = [dict(ids=[1, 2, 3 + k] + [0] * k, slots=[2 + k], nopts=[2], types=["choice"]) for k in range(3)]
    rqs = {f"q{k}": {"type": "choice", "names": ["a", "b"], "options": ["a", "b"]} for k in range(3)}
    index = [(f"q{k}", "list", k, 1) for k in range(3)]
    monkeypatch.setattr(V, "_prepare_s1", lambda *a: (rqs, index, items, 3, 2))
    return eng, V.S1Req(state="s", questions={f"q{k}": {} for k in range(3)})


def test_route_client_disconnect_cancels_rows_then_releases(monkeypatch):
    V = _V()
    from fastapi import HTTPException
    eng, req = _route_setup(V, monkeypatch)
    fr = _FakeRequest()

    async def go():
        t = asyncio.ensure_future(V.systemone(req, fr))
        for _ in range(10): await asyncio.sleep(0.01)
        assert V.outstanding == 3 and sorted(eng.started) == [3, 4, 5]
        fr.disconnect()
        with pytest.raises(HTTPException) as e:
            await t
        assert e.value.status_code == 499
        assert eng.cancelled == 3 and V.outstanding == 0
    asyncio.run(go())


def test_route_handler_cancelled_cancels_rows_then_releases(monkeypatch):
    V = _V()
    eng, req = _route_setup(V, monkeypatch)

    async def go():
        t = asyncio.ensure_future(V.systemone(req, _FakeRequest()))
        for _ in range(10): await asyncio.sleep(0.01)
        assert V.outstanding == 3
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert eng.cancelled == 3 and V.outstanding == 0
    asyncio.run(go())


def test_route_answers(monkeypatch):
    V = _V()
    eng, req = _route_setup(V, monkeypatch)
    monkeypatch.setattr(V, "TEMP", 1.0); monkeypatch.setattr(V, "TEMP_BY_TYPE", {})

    async def go():
        t = asyncio.ensure_future(V.systemone(req, _FakeRequest()))
        for _ in range(10): await asyncio.sleep(0.01)
        for g in list(eng.gates.values()): g.set()
        return await t
    out = asyncio.run(go())
    assert set(out["answers"]) == {"q0", "q1", "q2"} and V.outstanding == 0


class SlowCancelEngine(GatedEngine):
    """GatedEngine whose rows take until `stop` is set to finish their cancellation (a slow vLLM abort)."""
    def __init__(self, logits_of):
        super().__init__(logits_of); self.stop = asyncio.Event()

    async def generate(self, prompt, sp, request_id):
        n = len(prompt["prompt_token_ids"]); self.started.append(n)
        try:
            await self.gates.setdefault(n, asyncio.Event()).wait()
        except asyncio.CancelledError:
            self.cancelled = getattr(self, "cancelled", 0) + 1
            await self.stop.wait()
            raise
        async for out in FakeEngine.generate(self, prompt, sp, request_id):
            yield out


def test_route_cancelled_again_during_cleanup_still_releases_after_rows_stop(monkeypatch):
    V = _V()
    _, req = _route_setup(V, monkeypatch)
    letters, table, _ = _setup(V, monkeypatch)
    eng = SlowCancelEngine(lambda ids: {t: table[t] for t in table}); monkeypatch.setattr(V, "engine", eng)

    async def go():
        t = asyncio.ensure_future(V.systemone(req, _FakeRequest()))
        for _ in range(10): await asyncio.sleep(0.01)
        assert V.outstanding == 3 and sorted(eng.started) == [3, 4, 5]
        t.cancel()                                        # first cancellation: cleanup starts, rows are slow to stop
        for _ in range(10): await asyncio.sleep(0.01)
        assert not t.done() and eng.cancelled == 3 and V.outstanding == 3
        t.cancel()                                        # cancelled again during the cleanup
        for _ in range(10): await asyncio.sleep(0.01)
        assert not t.done() and V.outstanding == 3        # still waiting for the rows; admission still held
        eng.stop.set()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert V.outstanding == 0
    asyncio.run(go())


def test_warmup_fits_the_model_length(monkeypatch):
    V = _V()
    _, _, eng = _setup(V, monkeypatch)
    monkeypatch.setattr(V, "MAX_MODEL_LEN", 1000); monkeypatch.setattr(V, "PREFILL_FIRST", 512); monkeypatch.setattr(V, "BLOCK", 784)

    def prep(state, questions, independent):         # width probes: 10 tokens per option; shared state: one token per "x "
        if len(questions) == 1:
            n = len(questions["q"]["criteria"])
            return None, None, [dict(ids=list(range(10 * n)), slots=[10 * n - 1], nopts=[n], types=["choice"])], 0, 0
        k = state.count("x")
        return None, None, [dict(ids=list(range(k)) + [5000 + i], slots=[k], nopts=[2], types=["choice"]) for i in range(8)], 0, 0
    monkeypatch.setattr(V, "_prepare_s1", prep)
    asyncio.run(V._warmup())
    lens = [c[0] for c in eng.calls]
    assert max(lens) < 1000
    assert sorted({x for x in lens if x != 751}) == [20, 30, 50, 90, 170, 330, 650]    # 129 and 255 options do not fit
    assert lens.count(751) == 8                                                       # the state shortened to 750 tokens
