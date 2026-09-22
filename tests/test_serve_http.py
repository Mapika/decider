"""decider.serve over ASGI with a stand-in engine: wire format, limits, overload, /decide, and byte equality with the 1.0.x
server (decider.serve_v1) on the same requests.

Needs fastapi and httpx (the torch-cpu CI job).  The stand-in engine scores deterministically from the row's token ids, and
the stand-in tokenizer is byte-level, so no model, tokenizer download or torch is needed here; the serve_v1 comparison
needs torch (its module imports decider.engine) and skips without it.
"""
import asyncio, json
import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from decider import serve
from decider.prompt import MAX_OPTIONS


class ByteTok:
    """Byte-level tokenizer: enough for narrow (<= 10 options) prompts, which never need the label table."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))


class _Row:
    """One probability row: slicing and .tolist(), which is all the servers use."""
    def __init__(self, v): self.v = v
    def tolist(self): return list(self.v)
    def __getitem__(self, i): return _Row(self.v[i]) if isinstance(i, slice) else self.v[i]


class _P:
    """[n_questions, MAX_OPTIONS] tensor stand-in for one item: iterates over rows, .tolist() gives the nested list."""
    def __init__(self, rows): self.rows = rows
    def tolist(self): return [list(r) for r in self.rows]
    def __iter__(self): return (_Row(r) for r in self.rows)
    def __len__(self): return len(self.rows)


def _probs(ids, n):
    """A deterministic distribution over n options from the row's ids (so both servers get the same numbers)."""
    raw = [((sum(ids) * 31 + j * 7919 + len(ids)) % 97) + 1 for j in range(n)]
    s = sum(raw)
    return [x / s for x in raw] + [0.0] * (MAX_OPTIONS - n)


class FakeEngine:
    def __init__(self, delay=0.0):
        self.tok = ByteTok(); self.sealed = True; self.graphs = {}; self.max_ctx = 1536; self.delay = delay
        self.stats = dict(graph_captures=0, forwards=0, replays=0, eager_forwards=0, eager_rows=0, shared_calls=0, unbucketed_requests=0)
        self.calls = []

    def pad_len(self, n): return -(-n // 64) * 64
    def max_rows(self, T): return 32

    def _score(self, items):
        import time
        if self.delay: time.sleep(self.delay)
        self.stats["forwards"] += 1
        return [_P([_probs(it["ids"], n) for n in it["nopts"]]) for it in items]

    def score_items(self, items, temperature=1.0):
        self.calls.append(("items", len(items)))
        return self._score(items)

    def score_shared(self, items, temperature=1.0, min_prefix=192):
        self.calls.append(("shared", len(items))); self.stats["shared_calls"] += 1
        return self._score(items)


QUESTIONS = {"queue": {"type": "choice", "instructions": "Which queue?", "criteria": ["billing", "technical", "sales"]},
             "flag": {"type": "noul", "instructions": "Does this need a human?"},
             "sev": {"type": "score", "instructions": "How severe?", "criteria": ["none", "low", "medium", "high"]}}


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t")


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def served(monkeypatch):
    """serve with a stand-in engine and its workers started inside the test's event loop."""
    eng = FakeEngine()
    monkeypatch.setattr(serve, "eng", eng)
    monkeypatch.setattr(serve, "MODEL_NAME", "decider-test")
    monkeypatch.setattr(serve, "TEMP", 1.0)
    monkeypatch.setattr(serve, "ISOLATED", False)
    monkeypatch.setattr(serve, "SCHEMA_FIRST", False)
    monkeypatch.setattr(serve, "outstanding", 0)
    monkeypatch.setattr(serve, "gpu", None)
    monkeypatch.setattr(serve, "cpu", None)
    monkeypatch.setattr(serve, "batcher_task", None)

    async def go(fn):
        serve.start_workers()
        try:
            async with _client(serve.app) as cl:
                return await fn(cl)
        finally:
            serve._stop()
    return eng, lambda fn: _run(go(fn))


def test_systemone_wire_format(served):
    eng, run = served

    async def fn(cl):
        return await cl.post("/v1/systemone", json={"state": "the checkout is down", "questions": QUESTIONS})
    r = run(fn)
    assert r.status_code == 200
    body = json.loads(r.content)
    assert list(body) == ["model", "answers", "usage"] and body["model"] == "decider-test"
    assert list(body["usage"]) == ["input_tokens", "output_tokens"] and body["usage"]["output_tokens"] == 0
    assert set(body["answers"]) == set(QUESTIONS)
    assert set(body["answers"]["queue"]) == {"type", "choice", "confidence", "certainty", "probabilities"}
    assert set(body["answers"]["flag"]) == {"type", "noul"}
    assert body["usage"]["input_tokens"] > 0
    assert eng.calls and all(k == "items" for k, _ in eng.calls)


def test_empty_question_map(served):
    eng, run = served

    async def fn(cl):
        return await cl.post("/v1/systemone", json={"state": "hello", "questions": {}})
    r = run(fn)
    assert r.status_code == 200
    assert r.json() == {"model": "decider-test", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}


def test_invalid_question_is_422_with_detail(served):
    eng, run = served

    async def fn(cl):
        return await cl.post("/v1/systemone", json={"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["one"]}}})
    r = run(fn)
    assert r.status_code == 422 and r.json() == {"detail": "choice criteria: a map of 2..255 options"}


def test_shared_path_is_used_for_long_multi_question_states(served, monkeypatch):
    eng, run = served
    monkeypatch.setattr(serve, "SHARED", True); monkeypatch.setattr(serve, "SHARED_MIN_TOKENS", 100)

    async def fn(cl):
        return await cl.post("/v1/systemone", json={"state": "x" * 400, "questions": QUESTIONS})
    r = run(fn)
    assert r.status_code == 200 and eng.calls == [("shared", 3)]


def test_limits_return_413_before_scoring(served, monkeypatch):
    eng, run = served
    long_state = "w" * 5000
    monkeypatch.setattr(serve, "MAX_ROW_TOKENS", 1000)

    async def fn(cl):
        return await cl.post("/v1/systemone", json={"state": long_state, "questions": QUESTIONS})
    r = run(fn)
    assert r.status_code == 413 and "too many tokens" in r.json()["detail"] and "DECIDER_MAX_ROW_TOKENS" in r.json()["detail"]
    assert eng.calls == [] and serve.stats["rejected_too_large"] >= 1

    monkeypatch.setattr(serve, "MAX_ROW_TOKENS", 10 ** 6); monkeypatch.setattr(serve, "MAX_ROWS", 2)
    r = run(fn)
    assert r.status_code == 413 and "too many questions" in r.json()["detail"]

    monkeypatch.setattr(serve, "MAX_ROWS", 1024); monkeypatch.setattr(serve, "MAX_REQUEST_TOKENS", 6000)
    r = run(fn)
    assert r.status_code == 413 and "DECIDER_MAX_REQUEST_TOKENS" in r.json()["detail"]
    assert eng.calls == [] and serve.outstanding == 0


def test_overload_returns_503_and_recovers(served, monkeypatch):
    eng, run = served
    monkeypatch.setattr(serve, "MAX_QUEUE_ROWS", 4)
    eng.delay = 0.2

    async def fn(cl):
        body = {"state": "s", "questions": QUESTIONS}                       # 3 rows each: two in flight exceed 4
        rs = await asyncio.gather(*[cl.post("/v1/systemone", json=body) for _ in range(3)])
        later = await cl.post("/v1/systemone", json=body)
        return [r.status_code for r in rs], later.status_code
    codes, later = run(fn)
    assert sorted(codes) == [200, 503, 503], codes
    assert later == 200 and serve.outstanding == 0 and serve.stats["rejected_overloaded"] >= 2


def test_decide_route(served):
    eng, run = served
    schema = {"Which team?": {"type": "choice", "options": ["billing", "technical", "none of the above"]},
              "Urgent?": {"type": "bool"}, "Mood": {"type": "scale", "legend": ["calm", "angry"]}}

    async def fn(cl):
        return await cl.post("/decide", json={"context": "my card was charged twice", "schema": schema})
    r = run(fn)
    assert r.status_code == 200
    body = r.json()
    assert list(body) == list(schema)
    assert set(body["Which team?"]) == {"choice", "confidence", "type", "probabilities"} and body["Which team?"]["type"] == "choice"
    assert set(body["Which team?"]["probabilities"]) == set(schema["Which team?"]["options"])     # neutralised option mapped back
    assert body["Urgent?"]["type"] == "noul" and body["Mood"]["type"] == "scale" and body["Mood"]["legend"] == ["calm", "angry"]


def test_health_and_stats(served):
    eng, run = served

    async def fn(cl):
        return (await cl.get("/health")).json(), (await cl.get("/stats")).json(), (await cl.get("/v1/models")).json()
    h, s, m = run(fn)
    assert h["ok"] is True and "model" in h
    for k in ("requests", "batches", "decisions", "rows", "shared_prefix_requests", "errors", "rejected_too_large",
              "rejected_overloaded", "outstanding_rows", "engine", "graphs", "sealed", "limits"):
        assert k in s
    assert set(s["limits"]) == {"max_rows", "max_row_tokens", "max_request_tokens", "max_queue_rows"}
    assert list(m["models"][0]) == ["name", "description", "release_date"]


def test_bytes_match_the_1_0_server(served, monkeypatch):
    """Same requests, same stand-in scoring: the 1.1.0 and 1.0.x servers must return identical bytes."""
    pytest.importorskip("torch")
    from decider import serve_v1
    eng, run = served
    monkeypatch.setattr(serve_v1, "eng", eng); monkeypatch.setattr(serve_v1, "MODEL_NAME", "decider-test")
    monkeypatch.setattr(serve_v1, "TEMP", 1.0); monkeypatch.setattr(serve_v1, "ISOLATED", False)
    monkeypatch.setattr(serve_v1, "SCHEMA_FIRST", False); monkeypatch.setattr(serve_v1, "SHARED_MIN_TOKENS", 100)
    monkeypatch.setattr(serve, "SHARED", True); monkeypatch.setattr(serve, "SHARED_MIN_TOKENS", 100)
    bodies = [{"state": "the checkout is down", "questions": QUESTIONS},
              {"state": "x" * 400, "questions": QUESTIONS},                                    # shared path in both
              {"state": {"ticket": {"body": "y" * 300}, "rows": [{"v": i} for i in range(12)]}, "questions": QUESTIONS, "independent": False},
              {"state": "hello", "questions": {}},
              {"state": "s", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["one"]}}},
              {"questions": QUESTIONS}]
    decide = {"context": "my card was charged twice",
              "schema": {"Which team?": {"type": "choice", "options": ["billing", "technical", "none"]}, "Urgent?": {"type": "bool"}}}

    async def fn(cl):
        serve_v1.queue = asyncio.Queue(); t = asyncio.get_running_loop().create_task(serve_v1.batcher())
        try:
            async with _client(serve_v1.app) as old:
                out = []
                for b in bodies:
                    a, o = await cl.post("/v1/systemone", json=b), await old.post("/v1/systemone", json=b)
                    out.append(((a.status_code, a.content), (o.status_code, o.content)))
                a, o = await cl.post("/decide", json=decide), await old.post("/decide", json=decide)
                out.append(((a.status_code, a.content), (o.status_code, o.content)))
                return out
        finally:
            t.cancel()
    pairs = run(fn)
    for new, old in pairs:
        assert new == old, (new, old)
    assert pairs[0][0][0] == 200 and pairs[3][0][0] == 200 and pairs[4][0][0] == 422 and pairs[5][0][0] == 422 and pairs[-1][0][0] == 200


class FakeSchemaEngine:
    """SchemaEngine stand-in: real prefix/suffix tokenisation, deterministic scoring, counts of GPU-side work."""
    def __init__(self, tok):
        self.tok = tok; self.graphs = {}; self.stats = dict(prepared=0, captures=0, replays=0, eager=0)

    def prepare(self, questions, independent=False, compile=False):
        import types
        from decider.prompt import schema_prefix_ids
        qs = [types.SimpleNamespace(text=q["question"], options=list(q["options"])) for q in questions]
        groups = [[q] for q in qs] if independent else [qs]; pres = [schema_prefix_ids(self.tok, g) for g in groups]
        h = types.SimpleNamespace(P=len(groups), nq=len(qs), slots_per_row=1 if independent else len(qs), nopts=[len(q.options) for q in qs],
                                  tps=[len(p) for p in pres], tpmax=max(len(p) for p in pres), id=self.stats["prepared"])
        self.stats["prepared"] += 1; self.stats["captures"] += 1          # 1.0.x captures a graph the first time a schema is scored
        return h

    def tokenize(self, h, context, max_ctx_tokens):
        from decider.prompt import schema_suffix_ids
        return schema_suffix_ids(self.tok, context, h.slots_per_row, max_ctx_tokens)

    @staticmethod
    def bucket(n): return -(-n // 32) * 32

    def score_rows(self, h, rows, temperature=1.0):
        self.stats["replays"] += 1
        return [_P([_probs(r[0], n) for n in h.nopts]) for r in rows]


def test_schema_cache_requests_are_bounded_before_any_gpu_preparation(served, monkeypatch):
    """The second review's counterexample: under a 128-token bound a 222-token request got 413 in state-first and 200 in
    schema mode, and a rejected three-question request still prepared its prefixes on the GPU.  Both must now be rejected
    before `se.prepare` runs, and the schema path must otherwise work (usage with cached_tokens, one preparation per schema)."""
    eng, run = served
    se = FakeSchemaEngine(eng.tok)
    monkeypatch.setattr(serve, "SCHEMA_FIRST", True); monkeypatch.setattr(serve, "se", se)
    monkeypatch.setattr(serve, "schemas", {}); monkeypatch.setattr(serve, "seen", {}); monkeypatch.setenv("DECIDER_SCHEMA_MIN_SEEN", "1")
    monkeypatch.setattr(serve, "MAX_ROWS", 2); monkeypatch.setattr(serve, "MAX_ROW_TOKENS", 128); monkeypatch.setattr(serve, "MAX_REQUEST_TOKENS", 128)
    big = {"type": "choice", "instructions": "word " * 200, "criteria": ["a", "b"]}
    small = {"type": "choice", "instructions": "which?", "criteria": ["a", "b"]}

    async def fn(cl):
        out = {}
        out["state_first"] = await cl.post("/v1/systemone", json={"state": "hello", "questions": {"q": big}, "layout": "state_first"})
        out["schema_big"] = await cl.post("/v1/systemone", json={"state": "hello", "questions": {"q": big}})
        out["prepared_after_big"] = se.stats["prepared"]
        out["schema_many"] = await cl.post("/v1/systemone", json={"state": "hello", "questions": {str(i): small for i in range(3)}})
        out["prepared_after_many"] = se.stats["prepared"]
        out["schema_empty"] = await cl.post("/v1/systemone", json={"state": "hello", "questions": {}})
        out["ok1"] = await cl.post("/v1/systemone", json={"state": "hello", "questions": {"q": small}})
        out["ok2"] = await cl.post("/v1/systemone", json={"state": "hello again", "questions": {"q": small}})
        out["stats"] = (await cl.get("/stats")).json()
        return out
    o = run(fn)
    assert o["state_first"].status_code == 413 and "DECIDER_MAX_ROW_TOKENS" in o["state_first"].json()["detail"]
    assert o["schema_big"].status_code == 413 and "too many tokens" in o["schema_big"].json()["detail"]
    assert o["prepared_after_big"] == 0                                    # rejected before any prefix preparation
    assert o["schema_many"].status_code == 413 and "too many questions" in o["schema_many"].json()["detail"]
    assert o["prepared_after_many"] == 0
    assert o["schema_empty"].status_code == 200 and o["schema_empty"].json()["answers"] == {} and o["schema_empty"].json()["usage"]["input_tokens"] == 0
    for k in ("ok1", "ok2"):
        assert o[k].status_code == 200, o[k].text
        body = o[k].json()
        assert list(body["usage"]) == ["input_tokens", "cached_tokens", "output_tokens"] and body["usage"]["cached_tokens"] > 0
        assert set(body["answers"]) == {"q"}
    assert se.stats["prepared"] == 1 and se.stats["replays"] == 2          # one preparation per schema, one replay per request
    assert o["stats"]["schema_cache"] == dict(prepared=1, captures=1, replays=2, eager=0, schemas=1)
    assert serve.outstanding == 0 and o["stats"]["rejected_too_large"] >= 3


# ---- batching policy ------------------------------------------------------
def _row(n, nopts=3):
    return dict(ids=list(range(n)), slots=[n - 1], nopts=[nopts], golds=[0], perms=[list(range(nopts))])


def _queue_rows(loop, spec):
    """spec: [(row length, request id)].  -> the futures, in the order the rows were queued."""
    futs = []
    for n, rid in spec:
        f = loop.create_future(); futs.append(f)
        serve.queue.put_nowait((f, _row(n), rid))
    return futs


def test_queued_rows_of_different_lengths_share_one_forward(served):
    """The cross-request merge: rows waiting at the same moment are partitioned by decider.batching.plan_batches, so
    three rows in three different 64-token buckets run as one forward at the longest bucket instead of three."""
    eng, run = served

    async def fn(cl):
        return await asyncio.gather(*_queue_rows(asyncio.get_running_loop(), [(200, 1), (60, 2), (130, 3)]))
    out = run(fn)
    assert eng.calls == [("items", 3)], eng.calls
    for (n, _), p in zip([(200, 1), (60, 2), (130, 3)], out):
        assert p.tolist() == [_probs(_row(n)["ids"], 3)]                  # every row got its own answer, not a neighbour's
    assert len({tuple(p.tolist()[0]) for p in out}) == 3


def test_split_collections_return_each_row_its_own_answer(served, monkeypatch):
    """With merging off the three rows run as three forwards; the results must still come back on the right futures."""
    eng, run = served
    monkeypatch.setattr(serve, "MERGE_OVERHEAD_TOKENS", 0)
    spec = [(200, 1), (60, 2), (130, 3)]

    async def fn(cl):
        return await asyncio.gather(*_queue_rows(asyncio.get_running_loop(), spec))
    out = run(fn)
    assert eng.calls == [("items", 1), ("items", 1), ("items", 1)], eng.calls
    for (n, _), p in zip(spec, out):
        assert p.tolist() == [_probs(_row(n)["ids"], 3)]


def test_a_cancelled_row_does_not_disturb_the_others(served):
    """A request that goes away before its batch runs: the batcher must skip its future, serve the rest and stay alive."""
    eng, run = served

    async def fn(cl):
        loop = asyncio.get_running_loop()
        futs = _queue_rows(loop, [(200, 1), (60, 2), (130, 3)])
        futs[1].cancel()
        done = await asyncio.gather(futs[0], futs[2])
        again = await asyncio.gather(*_queue_rows(loop, [(80, 4)]))       # the batcher is still running
        return done, again, serve.batcher_task.done()
    (a, c), again, dead = run(fn)
    assert not dead
    assert a.tolist() == [_probs(_row(200)["ids"], 3)] and c.tolist() == [_probs(_row(130)["ids"], 3)]
    assert again[0].tolist() == [_probs(_row(80)["ids"], 3)]


def test_a_failing_forward_fails_every_future_in_its_group(served):
    """score_items raising must reach every row of that group as an exception, and must not kill the batcher."""
    eng, run = served
    good_score = eng.score_items

    def boom(items, temperature=1.0):
        eng.calls.append(("items", len(items)))
        raise RuntimeError("forward failed")

    async def fn(cl):
        loop = asyncio.get_running_loop()
        eng.score_items = boom                                            # restored below, not via monkeypatch: the
        try:                                                              # `served` fixture shares that monkeypatch
            bad = await asyncio.gather(*_queue_rows(loop, [(200, 1), (60, 2)]), return_exceptions=True)
        finally:
            eng.score_items = good_score
        good = await asyncio.gather(*_queue_rows(loop, [(80, 3)]))
        return bad, good, serve.batcher_task.done()
    bad, good, dead = run(fn)
    assert len(bad) == 2 and all(isinstance(x, RuntimeError) and str(x) == "forward failed" for x in bad)
    assert not dead and good[0].tolist() == [_probs(_row(80)["ids"], 3)]


def test_merging_is_off_when_the_overhead_is_zero(served, monkeypatch):
    """With DECIDER_MERGE_OVERHEAD_TOKENS = 0 a forward is free, so no row is ever padded into a longer row's bucket."""
    eng, run = served
    monkeypatch.setattr(serve, "MERGE_OVERHEAD_TOKENS", 0)

    async def fn(cl):
        return await asyncio.gather(*_queue_rows(asyncio.get_running_loop(), [(200, 1), (60, 2), (130, 3)]))
    run(fn)
    assert eng.calls == [("items", 1), ("items", 1), ("items", 1)], eng.calls


# ---- the adaptive collection window ---------------------------------------
class _Fut:
    """Future stand-in for adaptive_ms: only `done()` is read."""
    def __init__(self, done): self._done = done
    def done(self): return self._done


def _batch(spec):
    return [(_Fut(done), None, rid) for rid, done in spec]


def test_adaptive_window_follows_the_request_mix(monkeypatch):
    """multi -> single -> multi: the window is asked for only after a collection that held more than one live request."""
    monkeypatch.setattr(serve, "ADAPTIVE_WAIT_MS", 2.0)
    assert serve.adaptive_ms(_batch([(1, False), (2, False)])) == 2.0        # two requests: wait next time
    assert serve.adaptive_ms(_batch([(1, False), (1, False)])) == 0.0        # one request, several rows: do not wait
    assert serve.adaptive_ms(_batch([(1, False)])) == 0.0
    assert serve.adaptive_ms(_batch([(3, False), (4, False)])) == 2.0        # back to two requests


def test_a_cancelled_row_is_not_evidence_of_concurrency(monkeypatch):
    monkeypatch.setattr(serve, "ADAPTIVE_WAIT_MS", 2.0)
    assert serve.adaptive_ms(_batch([(1, False), (2, True)])) == 0.0         # the second request's row is already done
    assert serve.adaptive_ms(_batch([(1, True), (2, True)])) == 0.0


def test_the_adaptive_window_is_off_when_the_variable_is_zero(monkeypatch):
    monkeypatch.setattr(serve, "ADAPTIVE_WAIT_MS", 0.0)
    assert serve.adaptive_ms(_batch([(1, False), (2, False)])) == 0.0


def test_collect_waits_only_for_the_window_it_was_given():
    """_collect takes what is queued; a positive adaptive window holds the collection open for a row that is still
    coming, and a zero window does not."""
    async def go(adaptive):
        q = asyncio.Queue(); q.put_nowait("a")
        async def later():
            await asyncio.sleep(0.01)
            q.put_nowait("b")
        t = asyncio.get_running_loop().create_task(later())
        got = await serve._collect(q, 0.0, adaptive)
        await t
        return got
    assert _run(go(50.0)) == ["a", "b"]
    assert _run(go(0.0)) == ["a"]


def test_a_quiet_queue_drops_the_adaptive_window():
    """The window is for overlapping requests.  When the first row of a collection was slow to arrive, the queue was
    idle, and the collection must go as soon as it is empty however large the window was."""
    async def go(idle_reset_ms):
        q = asyncio.Queue()
        async def feed():
            await asyncio.sleep(0.02); q.put_nowait("a")
            await asyncio.sleep(0.01); q.put_nowait("b")
        t = asyncio.get_running_loop().create_task(feed())
        got = await serve._collect(q, 0.0, 50.0, idle_reset_ms)
        await t
        return got
    assert _run(go(0.0)) == ["a"]                                            # idle queue: window dropped
    assert _run(go(1000.0)) == ["a", "b"]                                    # window kept
