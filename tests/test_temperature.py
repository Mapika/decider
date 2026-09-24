"""Per-answer-type temperatures (1.4.0, decider.temperature).

* decider_config.json validation: every temperature finite and > 0, only "choice", "noul" and "score" as map keys.
* resolution: the by-type value, else "temperature"; the schema-cache precedence; an explicit override switches the map off.
* application on every path that runs on CPU: read_slots, EngineV2.score_items (chunked too), decider.shared_prefix,
  SchemaEngine.score_rows, Decider (engine and eager; decide, decide_json, system_one with and without isolated levels,
  the schema cache handle), decider.serve (rows of several requests and types in one forward, the shared path, the schema
  cache, /health), decider.serve_v1.
* a config without a map: the engines receive the scalar itself, and the probabilities equal the ones 1.3.0 returned
  (tests/data/temperature_1_3_0_pins.json, produced by running tests/temperature_cases.py on origin/main 5f91c01).
"""
import json, math
from pathlib import Path

import pytest

from decider import temperature as TT

PINS = Path(__file__).parent / "data" / "temperature_1_3_0_pins.json"
MAP = {"choice": 1.48, "noul": 2.22, "score": 1.38}


# ---- config validation and resolution (no torch) ------------------------------------------------------------------
def test_no_map_resolves_to_the_1_3_0_numbers():
    for cfg in ({}, {"temperature": 1.3}, {"temperature": 1.3, "temperature_schema_first": 1.18}, {"temperature": "1.2"}):
        (T, m), (Ts, ms) = TT.from_config(cfg)
        assert T == float(cfg.get("temperature", 1.0)) and m == {}
        assert Ts == float(cfg.get("temperature_schema_first", T)) and ms == {}


def test_map_with_fallback():
    (T, m), (Ts, ms) = TT.from_config({"temperature": 1.3, "temperature_by_type": {"noul": 2.0}})
    assert T == 1.3 and m == {"noul": 2.0}
    assert TT.effective(T, m) == {"choice": 1.3, "noul": 2.0, "score": 1.3}
    assert TT.for_types(T, m, ["choice", "noul", "score", None]) == [1.3, 2.0, 1.3, 1.3]
    assert (Ts, ms) == (1.3, {"noul": 2.0})                     # no schema-first value: the state-first temperatures


def test_schema_first_precedence():
    cfg = {"temperature": 1.3, "temperature_by_type": {"noul": 2.0, "score": 1.5}, "temperature_schema_first": 1.1,
           "temperature_schema_first_by_type": {"choice": 0.9}}
    (_, _), (Ts, ms) = TT.from_config(cfg)
    assert TT.effective(Ts, ms) == {"choice": 0.9, "noul": 1.1, "score": 1.1}   # schema map, then schema scalar
    cfg.pop("temperature_schema_first")
    (_, _), (Ts, ms) = TT.from_config(cfg)
    assert TT.effective(Ts, ms) == {"choice": 0.9, "noul": 2.0, "score": 1.5}   # schema map, then the state-first map


def test_explicit_override_switches_the_map_off():
    cfg = {"temperature": 1.3, "temperature_by_type": MAP}
    (T, m), (Ts, ms) = TT.from_config(cfg, temperature=1.0)
    assert (T, m) == (1.0, {}) and (Ts, ms) == (1.0, {})
    (T, m), _ = TT.from_config(cfg, temperature="0.8")                            # DECIDER_TEMPERATURE is a string
    assert (T, m) == (0.8, {})
    (T, m), _ = TT.from_config(cfg, temperature=1.0, temperature_by_type={"score": 3.0})
    assert (T, m) == (1.0, {"score": 3.0})
    (T, m), _ = TT.from_config(cfg, temperature_by_type={"score": 3.0})
    assert (T, m) == (1.3, {"score": 3.0})


@pytest.mark.parametrize("m,needle", [
    ({"bool": 2.0}, "unknown key 'bool'"), ({"scale": 2.0}, "unknown key 'scale'"), ({"Choice": 1.0}, "unknown key"),
    ({"noul": 0}, "finite number > 0"), ({"noul": -1.0}, "finite number > 0"), ({"noul": float("nan")}, "finite number > 0"),
    ({"noul": float("inf")}, "finite number > 0"), ({"noul": "2.0"}, "finite number > 0"), ({"noul": True}, "finite number > 0"),
    ({"noul": None}, "finite number > 0"), ([1.0, 2.0], "must be a map"), ("1.5", "must be a map")])
def test_invalid_maps_are_rejected(m, needle):
    with pytest.raises(ValueError, match="temperature_by_type") as e:
        TT.from_config({"temperature": 1.3, "temperature_by_type": m})
    assert needle in str(e.value)
    if "unknown key" in needle:
        assert '"choice", "noul" and "score"' in str(e.value) and '"bool" field is "noul"' in str(e.value)
    with pytest.raises(ValueError, match="temperature_schema_first_by_type"):
        TT.from_config({"temperature_schema_first_by_type": m})


@pytest.mark.parametrize("key", ["temperature", "temperature_schema_first"])
@pytest.mark.parametrize("v", [0, -1.0, float("nan"), float("inf"), "abc", True, None])
def test_invalid_scalars_are_rejected(key, v):
    with pytest.raises(ValueError, match=key):
        TT.from_config({key: v})


def test_invalid_override_is_rejected():
    with pytest.raises(ValueError):
        TT.from_config({}, temperature=0.0)
    with pytest.raises(ValueError, match="unknown key"):
        TT.from_config({}, temperature_by_type={"bool": 1.0})


def test_no_map_passes_the_scalar_itself():
    items = [dict(slots=[3, 7], types=["choice", "noul"])]
    T = 1.3
    assert TT.for_items(T, {}, items) is T and TT.for_types(T, {}, ["noul"]) is T
    assert TT.slot_temperatures(T, items) is T and TT.item_slice(T, 0, 1) is T


def test_for_items_and_slot_temperatures():
    items = [dict(slots=[3, 7], types=["choice", "noul"]), dict(slots=[5]), dict(slots=[1, 2, 4], types=["score"] * 3)]
    per = TT.for_items(1.3, MAP, items)
    assert per == [[1.48, 2.22], [1.3], [1.38] * 3]                    # an item without "types" gets "temperature"
    assert TT.slot_temperatures(per, items) == [1.48, 2.22, 1.3, 1.38, 1.38, 1.38]
    assert TT.slot_temperatures([2.0, [1.0], 3.0], items) == [2.0, 2.0, 1.0, 3.0, 3.0, 3.0]
    with pytest.raises(ValueError):
        TT.slot_temperatures([1.0], items)
    with pytest.raises(ValueError):
        TT.slot_temperatures([[1.0], [1.0], [1.0]], items)


def test_row_types_follow_plan_rows():
    from decider import systemone as S1
    qs = {"a": {"type": "choice", "instructions": "x", "criteria": ["p", "q"]}, "b": {"type": "bool", "instructions": "y"},
          "c": {"type": "score", "instructions": "z", "criteria": ["lo", "mid", "hi"]}, "d": {"type": "noul", "instructions": "w"}}
    rqs = {k: S1.render_question(v) for k, v in qs.items()}
    rows, index = S1.plan_rows(rqs, True)
    assert S1.row_types(rqs, index) == ["choice", "noul", "score", "score", "score", "noul"] and len(rows) == 6
    rows, index = S1.plan_rows(rqs, False)
    assert S1.row_types(rqs, index) == ["choice", "noul", "score", "noul"]


# ---- engines ------------------------------------------------------------------------------------------------------
torch_required = pytest.mark.skipif(__import__("importlib").util.find_spec("torch") is None, reason="needs torch")


def _softmax(z, T, n):
    import torch
    return torch.softmax(torch.as_tensor(z[:n], dtype=torch.float32) / T, -1)


@torch_required
def test_scaled_softmax_scalar_is_the_1_3_0_expression_and_list_is_per_row():
    import torch
    lg = torch.randn(4, 6, generator=torch.Generator().manual_seed(0)) * 3
    for T in (1.3, 0.7, 2):
        assert torch.equal(TT.scaled_softmax(lg, T), torch.softmax(lg / T, -1))
    Ts = [1.0, 2.0, 0.5, 1.3]
    got = TT.scaled_softmax(lg, Ts)
    for i, T in enumerate(Ts):
        assert torch.allclose(got[i], torch.softmax(lg[i] / T, -1), atol=1e-7)
    with pytest.raises(ValueError):
        TT.scaled_softmax(lg, [1.0, 2.0])


@torch_required
def test_read_slots_per_slot():
    import torch
    from decider.engine import read_slots
    out = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(1)) * 3
    rows, slots, nopts = [0, 0, 1], [1, 4, 2], [3, 2, 8]
    got = read_slots(out, rows, slots, nopts, [1.0, 2.0, 0.5], [2, 1])
    ref = [_softmax(out[r, s], T, n) for r, s, n, T in zip(rows, slots, nopts, [1.0, 2.0, 0.5])]
    flat = torch.cat(got)
    for i, (p, n) in enumerate(zip(ref, nopts)):
        assert torch.allclose(flat[i, :n], p, atol=1e-6) and float(flat[i, n:].sum()) == 0.0


@pytest.fixture(scope="module")
def tc():
    pytest.importorskip("torch")
    import temperature_cases
    return temperature_cases


def _items(tc, spec, seed=0):
    """spec: [(length, [answer type per slot])] -> items with option counts 3, 2, 4, ..."""
    import random
    g = random.Random(seed); out = []
    for L, types in spec:
        ids = [g.randrange(1, 256) for _ in range(L)]
        n = len(types)
        out.append(dict(ids=ids, slots=[L * (k + 1) // n - 1 for k in range(n)], nopts=[3 + (k % 3) for k in range(n)],
                        golds=[0] * n, perms=[list(range(3 + (k % 3))) for k in range(n)], types=list(types)))
    return out


def _per_slot_reference(score, items, T, m):
    """Score every item alone at the scalar temperature of each of its slots: the per-type answer must equal it."""
    import torch
    ref = []
    for it in items:
        rows = []
        for j, t in enumerate(TT.item_types(it)):
            rows.append(score([it], m.get(t, T))[0][j])
        ref.append(torch.stack(rows))
    return ref


def _assert_same(got, ref):
    import torch
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-6), (a, b)


@torch_required
def test_engine_v2_applies_each_slot_its_type(tc):
    eng = tc.engine()
    items = _items(tc, [(90, ["choice", "noul"]), (40, ["score"]), (130, ["noul", "score", "choice"]), (60, [None])])
    T = 1.3
    got = eng.score_items(items, temperature=TT.for_items(T, MAP, items))
    _assert_same(got, _per_slot_reference(eng.score_items, items, T, MAP))
    plain = eng.score_items(items, temperature=T)                         # and it differs from one temperature for all
    assert not torch_allclose(got[0][1], plain[0][1])
    with pytest.raises(ValueError):
        eng.score_items(items, temperature=[1.0])


def torch_allclose(a, b):
    import torch
    return torch.allclose(a, b, atol=1e-4)


@torch_required
def test_engine_v2_chunks_keep_the_item_temperatures(tc):
    from decider.engine_v2 import EngineV2
    e = EngineV2(model=tc.StandInModel(), device="cpu", use_graphs=False, token_budget=20000)
    e.seal()
    items = _items(tc, [(9000, ["noul"]), (9000, ["choice"]), (9000, ["score", "noul"]), (9000, ["choice"]), (9000, ["noul"])])
    before = e.stats["eager_forwards"]
    got = e.score_items(items, temperature=TT.for_items(1.3, MAP, items))
    assert e.stats["eager_forwards"] - before == 3                         # chunks of 2, 2 and 1 rows
    _assert_same(got, _per_slot_reference(e.score_items, items, 1.3, MAP))


@torch_required
def test_engine_v1_applies_each_slot_its_type(tc):
    from decider.engine import Engine
    e = object.__new__(Engine)
    e.m = tc.StandInModel(); e.tok = e.m.tok; e.dev = "cpu"; e.use_graphs = False; e.core = e.m.lm.model
    e.W = e.m.lm.lm_head.weight[e.m.letters].detach().clone(); e._fwd_impl = e._fwd_eager; e.stats = dict(graph_captures=0, forwards=0)
    items = _items(tc, [(90, ["choice", "noul"]), (40, ["score"])])
    got = e.score_items(items, temperature=TT.for_items(1.3, MAP, items))
    _assert_same(got, _per_slot_reference(e.score_items, items, 1.3, MAP))


@torch_required
def test_shared_prefix_applies_each_slot_its_type_across_chunks():
    pytest.importorskip("torch")
    import test_shared_prefix as SP
    from decider.shared_prefix import score_shared
    eng = SP._Eng()
    items = SP._items([31, 44, 12, 60, 23], n_q=2)
    for it, ts in zip(items, [["choice", "noul"], ["score", "score"], ["noul", "choice"], ["choice", "choice"], ["score", "noul"]]):
        it["types"] = ts
    T = 1.3
    ref = [torch_stack([score_shared(eng, items, temperature=MAP[t], rows_per_fork=len(items))[i][j] for j, t in enumerate(it["types"])])
           for i, it in enumerate(items)]
    for m in (1, 2, 5):
        _assert_same(score_shared(eng, items, temperature=TT.for_items(T, MAP, items), rows_per_fork=m), ref)


def torch_stack(rows):
    import torch
    return torch.stack(rows)


@torch_required
def test_schema_engine_score_rows_per_row_temperature():
    import types, torch
    from decider.schema_engine import SchemaEngine
    se = object.__new__(SchemaEngine)
    se.tok = types.SimpleNamespace(pad_token_id=0); se.dev = "cpu"; se.use_graphs = False; se.stats = dict(eager=0, replays=0)
    for P, spr in ((3, 1), (1, 3)):                                          # independent and packed schemas
        h = types.SimpleNamespace(P=P, nq=3, nopts=[3, 2, 4], slots_per_row=spr)
        lg = torch.randn(2 * P, 32, 8, generator=torch.Generator().manual_seed(P)) * 3
        se._static = lambda h, R, Ts: (None, None, None)
        se._fwd = lambda ids, *a, lg=lg: lg[: ids.shape[0], : ids.shape[1]]
        rows = [([1] * 20, [19] if P == 3 else [5, 11, 19]), ([2] * 25, [24] if P == 3 else [7, 15, 24])]
        Ts = [1.48, 2.22, 1.38]
        got = se.score_rows(h, rows, temperature=Ts)
        for r, (_, sl) in enumerate(rows):
            for q in range(3):
                row, slot = (r * P + q, sl[0]) if P == 3 else (r, sl[q])
                assert torch.allclose(got[r][q, :h.nopts[q]], _softmax(lg[row, slot], Ts[q], h.nopts[q]), atol=1e-6)
        scalar = se.score_rows(h, rows, temperature=1.3)
        assert torch.allclose(scalar[0][1, :2], _softmax(lg[1 if P == 3 else 0, rows[0][1][0] if P == 3 else rows[0][1][1]], 1.3, 2), atol=1e-6)
        with pytest.raises(ValueError):
            se.score_rows(h, rows, temperature=[1.0, 2.0])


# ---- Decider ------------------------------------------------------------------------------------------------------
class _Recorder:
    """Wraps an engine and records the temperature argument of every call."""
    def __init__(self, eng):
        self.eng = eng; self.m = eng.m; self.tok = eng.tok; self.temps = []

    def score_items(self, items, temperature=1.0):
        self.temps.append(temperature); return self.eng.score_items(items, temperature)

    def score_shared(self, items, temperature=1.0, **kw):
        self.temps.append(temperature); return self.eng.score_shared(items, temperature, **kw)


@torch_required
@pytest.mark.parametrize("eager", [False, True])
def test_decider_decide_json_uses_the_field_types(tc, eager):
    """bool -> noul, choice -> choice, scale -> score: each field equals the answer read at that type's temperature alone."""
    cfg = {"temperature": 1.3, "temperature_by_type": MAP}
    d = tc.decider(cfg, eager=eager)
    got = d.decide_json(tc.STATE, tc.SCHEMA)
    for field, t in (("Which team?", "choice"), ("Urgent?", "noul"), ("Mood?", "score")):
        ref = tc.decider({"temperature": MAP[t]}, eager=eager).decide_json(tc.STATE, tc.SCHEMA)
        assert got[field] == ref[field], (field, got[field], ref[field])
    plain = tc.decider({"temperature": 1.3}, eager=eager).decide_json(tc.STATE, tc.SCHEMA)
    assert got["Urgent?"] != plain["Urgent?"]


@torch_required
def test_decider_plain_questions_are_choice(tc):
    d = tc.decider({"temperature": 1.3, "temperature_by_type": {"choice": 0.7, "noul": 5.0}})
    ref = tc.decider({"temperature": 0.7})
    for a, b in zip(d.decide_batch(tc.REQUESTS), ref.decide_batch(tc.REQUESTS)):
        _close([x["probs_list"] for x in a], [x["probs_list"] for x in b])


@torch_required
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("independent", [True, False])
@pytest.mark.parametrize("iso", [True, False])
def test_decider_system_one_uses_the_question_types(tc, eager, independent, iso):
    """Each answer equals the answer read with its type's temperature for every question; an isolated Score's level rows
    use the "score" temperature."""
    cfg = {"temperature": 1.3, "temperature_by_type": MAP}
    got = tc.decider(cfg, eager=eager).system_one(tc.STATE, tc.QUESTIONS, independent=independent, isolated=iso)["answers"]
    for k, spec in tc.QUESTIONS.items():
        ref = tc.decider({"temperature": MAP[spec["type"]]}, eager=eager).system_one(tc.STATE, tc.QUESTIONS, independent=independent, isolated=iso)["answers"]
        assert got[k] == ref[k], (k, got[k], ref[k])
    if iso and independent:
        assert "level_fit" in got["sev"]


@torch_required
def test_decider_passes_the_scalar_without_a_map(tc):
    d = tc.decider({"temperature": 1.3})
    rec = _Recorder(d.eng); d.eng = rec
    d.decide_batch(tc.REQUESTS); d.decide_json(tc.STATE, tc.SCHEMA)
    d.system_one(tc.STATE, tc.QUESTIONS); d.system_one(tc.STATE, tc.QUESTIONS, independent=False)
    assert rec.temps and all(t is d.T for t in rec.temps) and d.T == 1.3


@torch_required
def test_compiled_schema_passes_one_temperature_per_schema_row(tc):
    import types
    d = tc.decider({"temperature": 1.3, "temperature_by_type": {"noul": 2.0}, "temperature_schema_first": 1.1,
                    "temperature_schema_first_by_type": {"score": 0.9}})
    seen = []

    class _SE:
        def score(self, h, contexts, temperature=1.0, max_ctx_tokens=0):
            seen.append(temperature)
            import torch
            return [torch.full((h.nq, 8), 0.1) for _ in contexts]
    from decider.infer import CompiledSchema
    from decider import systemone as S1
    d._se = _SE()
    rqs = {k: S1.render_question(v) for k, v in tc.QUESTIONS.items()}
    rows, index = S1.plan_rows(rqs, True)
    CompiledSchema(d, rqs, types.SimpleNamespace(nq=len(rows)), index).batch([tc.STATE])
    assert seen == [[1.1, 1.1, 0.9, 0.9, 0.9, 1.1]]                     # choice, noul, 3 isolated score levels, choice
    d2 = tc.decider({"temperature": 1.3, "temperature_schema_first": 1.1})
    d2._se = _SE(); seen.clear()
    CompiledSchema(d2, rqs, types.SimpleNamespace(nq=len(rows)), index).batch([tc.STATE])
    assert seen == [1.1] and seen[0] is d2.T_schema


def test_decider_rejects_an_invalid_map_before_loading(tmp_path):
    pytest.importorskip("torch")
    from decider.infer import Decider
    (tmp_path / "decider_config.json").write_text(json.dumps({"temperature": 1.3, "temperature_by_type": {"bool": 2.0}}))
    with pytest.raises(ValueError, match="unknown key 'bool'"):
        Decider(str(tmp_path), device="cpu")                             # the folder has no weights: the config fails first
    (tmp_path / "decider_config.json").write_text(json.dumps({"temperature": 1.3, "temperature_by_type": {"noul": float("nan")}}))
    with pytest.raises(ValueError, match="finite number > 0"):
        Decider(str(tmp_path), device="cpu")


# ---- unchanged behaviour without a map: the 1.3.0 probabilities ------------------------------------------------------
def _close(a, b, path=""):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for k in a:
            _close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-6), (path, a, b)
    else:
        assert a == b, (path, a, b)


@torch_required
def test_without_a_map_the_probabilities_are_those_of_1_3_0(tc, monkeypatch):
    """Every case of tests/temperature_cases.py against the numbers 1.3.0 returned.  On the machine that produced the pins the
    two are bitwise equal; the 1e-6 tolerance only absorbs float round-off of other CPUs.  That no map means the very same
    arithmetic is pinned by test_decider_passes_the_scalar_without_a_map and test_serve_passes_the_scalar_without_a_map."""
    pytest.importorskip("fastapi")
    monkeypatch.delenv("DECIDER_TEMPERATURE", raising=False)
    pins = json.load(open(PINS))
    assert pins["generated_with"].startswith("decider-ai 1.3.0")
    for name, cfg in pins["configs"].items():
        got = json.loads(json.dumps(tc.cases(cfg)))
        _close(got, pins["cases"][name], name)


@torch_required
def test_a_map_changes_the_answers_against_1_3_0(tc, monkeypatch):
    """The comparison above would pass for the wrong reason if the cases did not depend on the temperature."""
    pytest.importorskip("fastapi")
    monkeypatch.delenv("DECIDER_TEMPERATURE", raising=False)
    pins = json.load(open(PINS))
    got = json.loads(json.dumps(tc.cases(dict(pins["configs"]["t1.3_iso"], temperature_by_type=MAP))))
    with pytest.raises(AssertionError):
        _close(got, pins["cases"]["t1.3_iso"])


# ---- decider.serve -----------------------------------------------------------------------------------------------
@pytest.fixture
def served(monkeypatch):
    pytest.importorskip("torch"); pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    import asyncio, httpx
    import temperature_cases as tc
    from decider import serve
    rec = _Recorder(tc.engine())
    rec.sealed = True; rec.graphs = {}; rec.stats = {}; rec.pad_len = lambda n: -(-n // 64) * 64; rec.max_rows = lambda T: 32
    for k, v in dict(eng=rec, MODEL_NAME="decider-test", TEMP=1.3, TEMP_BY_TYPE=dict(MAP), TEMP_SCHEMA=1.3, TEMP_SCHEMA_BY_TYPE=dict(MAP),
                     ISOLATED=True, SCHEMA_FIRST=False, NEUTRALIZE_NONE=True, CHAT=None, outstanding=0, gpu=None, cpu=None,
                     batcher_task=None, SHARED=True, SHARED_MIN_TOKENS=10 ** 9).items():
        monkeypatch.setattr(serve, k, v)

    def run(fn):
        async def go():
            serve.start_workers()
            try:
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serve.app), base_url="http://t") as cl:
                    return await fn(cl)
            finally:
                serve._stop()
        return asyncio.run(go())
    return serve, rec, run, tc


def test_serve_mixed_batch_gets_each_answer_its_temperature(served):
    """A /decide request and a /v1/systemone request in flight together: their rows share forwards, and every answer equals
    the one a server with that type's temperature for everything returns."""
    import asyncio
    serve, rec, run, tc = served
    body = {"state": tc.STATE, "questions": tc.QUESTIONS}
    dec = {"context": tc.STATE, "schema": tc.SCHEMA}

    async def fn(cl):
        a, b = await asyncio.gather(cl.post("/v1/systemone", json=body), cl.post("/decide", json=dec))
        return a.json(), b.json()
    s1, de = run(fn)
    assert any(isinstance(t, list) for t in rec.temps)
    for t in TT.TYPES:
        serve.TEMP_BY_TYPE = {}; serve.TEMP = MAP[t]
        ref_s1, ref_de = run(fn)
        for k, spec in tc.QUESTIONS.items():
            if spec["type"] == t:
                assert s1["answers"][k] == ref_s1["answers"][k], k
        for f, spec in tc.SCHEMA.items():
            if TT.FIELD_TYPES[spec["type"]] == t:
                assert de[f] == ref_de[f], f


def test_serve_shared_path_gets_each_answer_its_temperature(served):
    serve, rec, run, tc = served
    items = serve.prepare(rec.tok, tc.STATE, tc.QUESTIONS, True, True)[2]
    got = serve._score_shared(items)
    assert isinstance(rec.temps[-1], list) and rec.temps[-1] == [[MAP[t] for t in it["types"]] for it in items]
    serve.TEMP_BY_TYPE = {}
    for it, p in zip(items, got):
        serve.TEMP = MAP[it["types"][0]]
        assert _torch_close(serve._score_shared([it])[0], p)


def _torch_close(a, b):
    import torch
    return torch.allclose(a, b, atol=1e-6)


def test_serve_passes_the_scalar_without_a_map(served):
    serve, rec, run, tc = served
    serve.TEMP_BY_TYPE = {}; serve.TEMP = 1.3

    async def fn(cl):
        await cl.post("/v1/systemone", json={"state": tc.STATE, "questions": tc.QUESTIONS})
        await cl.post("/decide", json={"context": tc.STATE, "schema": tc.SCHEMA})
    run(fn)
    items = serve.prepare(rec.tok, tc.STATE, tc.QUESTIONS, True, True)[2]
    serve._score_shared(items)
    assert rec.temps and all(t is serve.TEMP for t in rec.temps)


def test_serve_schema_cache_gets_one_temperature_per_schema_row(served, monkeypatch):
    serve, rec, run, tc = served
    import test_serve_http as SH
    seen = []

    class _SE(SH.FakeSchemaEngine):
        def score_rows(self, h, rows, temperature=1.0):
            seen.append(temperature); return super().score_rows(h, rows, temperature)
    monkeypatch.setattr(serve, "SCHEMA_FIRST", True); monkeypatch.setattr(serve, "se", _SE(rec.tok))
    monkeypatch.setattr(serve, "schemas", {}); monkeypatch.setattr(serve, "seen", {}); monkeypatch.setenv("DECIDER_SCHEMA_MIN_SEEN", "1")
    monkeypatch.setattr(serve, "TEMP_SCHEMA", 1.1); monkeypatch.setattr(serve, "TEMP_SCHEMA_BY_TYPE", {"noul": 2.0})

    async def fn(cl):
        r = await cl.post("/v1/systemone", json={"state": tc.STATE, "questions": tc.QUESTIONS})
        return r.status_code, (await cl.get("/health")).json()
    code, health = run(fn)
    assert code == 200 and seen == [[1.1, 2.0, 1.1, 1.1, 1.1, 1.1]]        # choice, noul, 3 isolated score levels, choice
    assert health["temperature_schema_first_by_type"] == {"choice": 1.1, "noul": 2.0, "score": 1.1}
    serve.TEMP_SCHEMA_BY_TYPE = {}; seen.clear()
    assert run(fn)[0] == 200 and seen == [1.1] and seen[0] is serve.TEMP_SCHEMA


def test_health_reports_the_temperatures(served):
    serve, rec, run, tc = served

    async def fn(cl):
        return (await cl.get("/health")).json()
    h = run(fn)
    assert h["temperature"] == 1.3 and h["temperature_by_type"] == MAP and "temperature_schema_first_by_type" not in h
    serve.TEMP_BY_TYPE = {"noul": 2.0}
    assert run(fn)["temperature_by_type"] == {"choice": 1.3, "noul": 2.0, "score": 1.3}


def test_serve_apply_config(monkeypatch):
    pytest.importorskip("fastapi")
    from decider import serve
    for k in ("TEMP", "TEMP_SCHEMA", "TEMP_BY_TYPE", "TEMP_SCHEMA_BY_TYPE", "LAYOUT", "MODEL_NAME", "RELEASE_DATE", "ISOLATED",
              "NEUTRALIZE_NONE", "SCHEMA_FIRST"):
        monkeypatch.setattr(serve, k, getattr(serve, k))
    monkeypatch.delenv("DECIDER_TEMPERATURE", raising=False)
    serve.apply_config({"temperature": 1.3, "temperature_by_type": {"noul": 2.0}})
    assert (serve.TEMP, serve.TEMP_BY_TYPE, serve.TEMP_SCHEMA, serve.TEMP_SCHEMA_BY_TYPE) == (1.3, {"noul": 2.0}, 1.3, {"noul": 2.0})
    serve.apply_config({"temperature": 1.3})
    assert (serve.TEMP, serve.TEMP_BY_TYPE) == (1.3, {})
    monkeypatch.setenv("DECIDER_TEMPERATURE", "1.0")                  # the override is the one state-first temperature
    serve.apply_config({"temperature": 1.3, "temperature_by_type": {"noul": 2.0}})
    assert (serve.TEMP, serve.TEMP_BY_TYPE) == (1.0, {})
    monkeypatch.delenv("DECIDER_TEMPERATURE")
    with pytest.raises(ValueError, match="unknown key 'scale'"):
        serve.apply_config({"temperature": 1.3, "temperature_by_type": {"scale": 2.0}})
    with pytest.raises(ValueError, match="finite number > 0"):
        serve.apply_config({"temperature": 1.3, "temperature_by_type": {"score": -1}})


def test_serve_v1_applies_the_item_types(monkeypatch):
    pytest.importorskip("torch"); pytest.importorskip("fastapi")
    from decider import serve_v1
    import temperature_cases as tc
    monkeypatch.setattr(serve_v1, "eng", tc.engine()); monkeypatch.setattr(serve_v1, "TEMP", 1.3)
    monkeypatch.setattr(serve_v1, "TEMP_BY_TYPE", dict(MAP)); monkeypatch.setattr(serve_v1, "ISOLATED", True)
    seen = []
    qs, it = serve_v1._prepare(tc.STATE, tc.SCHEMA)
    (rqs, index), items = serve_v1._prepare_s1(tc.STATE, tc.QUESTIONS, True)
    serve_v1._locked(lambda items, temperature: seen.append(temperature), [it] + items)
    assert seen == [[[1.48, 2.22, 1.38]] + [[MAP[t]] for t in ["choice", "noul", "score", "score", "score", "choice"]]]
    monkeypatch.setattr(serve_v1, "TEMP_BY_TYPE", {}); seen.clear()
    serve_v1._locked(lambda items, temperature: seen.append(temperature), items)
    assert seen == [1.3] and seen[0] is serve_v1.TEMP
    import asyncio
    monkeypatch.setattr(serve_v1, "TEMP_BY_TYPE", {"noul": 2.0}); monkeypatch.setattr(serve_v1, "SCHEMA_FIRST", True)
    monkeypatch.setattr(serve_v1, "TEMP_SCHEMA", 1.1); monkeypatch.setattr(serve_v1, "TEMP_SCHEMA_BY_TYPE", {"score": 0.9})
    h = asyncio.run(serve_v1.health())
    assert h["temperature"] == 1.3 and h["temperature_by_type"] == {"choice": 1.3, "noul": 2.0, "score": 1.3}
    assert h["temperature_schema_first_by_type"] == {"choice": 1.1, "noul": 1.1, "score": 0.9}


# ---- decider.calibrate ----------------------------------------------------------------------------------------------
def test_calibrate_recovers_the_temperatures():
    np = pytest.importorskip("numpy")
    from decider.calibrate import fit_by_type, nll
    rng = np.random.default_rng(0)

    def listwise(T, K, n, t):
        out = []
        for _ in range(n):
            z = rng.normal(0, 3, K); p = np.exp(z / T); p /= p.sum()
            out.append({"type": t, "logits": z.tolist(), "gold": int(rng.choice(K, p=p))})
        return out

    def isolated(T, L, n):
        out = []
        for _ in range(n):
            z = rng.normal(0, 3, (L, 2)); e = np.exp(z / T); py = e[:, 1] / e.sum(1); p = py / py.sum()
            q = np.exp(z) / np.exp(z).sum(1, keepdims=True)                # stored at T = 1, as collect() stores them
            out.append({"type": "score", "level_probs": q.tolist(), "gold": int(rng.choice(L, p=p))})
        return out
    recs = listwise(1.5, 4, 3000, "choice") + listwise(2.2, 2, 3000, "noul") + isolated(1.4, 5, 3000)
    out = fit_by_type(recs, min_rows=100)
    got = out["temperature_by_type"]
    assert set(got) == {"choice", "noul", "score"} and out["rows"] == {"choice": 3000, "noul": 3000, "score": 3000}
    assert abs(got["choice"] - 1.5) < 0.15 and abs(got["noul"] - 2.2) < 0.2 and abs(got["score"] - 1.4) < 0.15
    for t in got:
        assert out["nll"][t]["fitted"] <= out["nll"][t]["T=1"]
    TT.by_type(got, "fitted map")                                       # a valid config map
    assert "score" not in fit_by_type(recs[:6000] + recs[6000:6010], min_rows=100)["temperature_by_type"]
    r = {"type": "choice", "probs": [0.5, 0.25, 0.25], "gold": 1}
    assert abs(nll(r, 1.0) - math.log(4)) < 1e-9
    with pytest.raises(ValueError):
        fit_by_type([{"type": "bool", "logits": [0, 1], "gold": 1}])


def test_calibrate_isolated_levels_do_not_underflow():
    """Confident level rows: at a small T every P(yes) underflows in linear space; the objective must stay the true one."""
    pytest.importorskip("numpy")
    from decider.calibrate import fit_by_type, nll
    r = {"type": "score", "level_logits": [[40, 0], [39, 0]], "gold": 0}
    assert abs(nll(r, 0.05) - 20.0) < 1e-6 and nll(r, 20.0) < 0.72
    assert fit_by_type([r], min_rows=1)["temperature_by_type"]["score"] > 1.0
    one_fits = {"type": "score", "level_probs": [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0]], "gold": 1}
    assert abs(nll(one_fits, 1.0)) < 1e-12 and nll(dict(one_fits, gold=0), 1.0) == 690.0     # a zero-probability gold is capped


@pytest.mark.parametrize("rec", [
    {"type": "choice", "logits": [0, 1, 2], "gold": -1}, {"type": "choice", "logits": [0, 1, 2], "gold": 3},
    {"type": "choice", "logits": [0, 1, 2], "gold": 0.9}, {"type": "choice", "logits": [0, 1, 2], "gold": True},
    {"type": "choice", "probs": [-0.1, 1.1], "gold": 0}, {"type": "choice", "probs": [0.0, 0.0], "gold": 0},
    {"type": "choice", "logits": [0, float("nan")], "gold": 0}, {"type": "choice", "logits": [1.0], "gold": 0},
    {"type": "choice", "logits": [[0, 1], [1, 0]], "gold": 0}, {"type": "score", "level_logits": [[1, 2, 3, 4]], "gold": 0},
    {"type": "score", "level_logits": [[1, 2]], "gold": 0}, {"type": "noul", "level_logits": [[1, 2], [2, 1]], "gold": 0},
    {"type": "choice", "logits": [0, 1], "probs": [0.5, 0.5], "gold": 0}, {"type": "choice", "gold": 0},
    {"type": "choice", "logits": [float("-inf"), float("-inf")], "gold": 0},
    {"type": "score", "level_logits": [[float("-inf"), float("-inf")], [0, 0]], "gold": 1},
    {"type": "noul", "logits": [0, 1, 2], "gold": 1},
    {"type": "score", "level_probs": [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], "gold": 1}])
def test_calibrate_rejects_malformed_records(rec):
    pytest.importorskip("numpy")
    from decider.calibrate import fit_by_type
    with pytest.raises(ValueError, match="record 0"):
        fit_by_type([rec], min_rows=1)


@torch_required
def test_calibrate_collect_reads_temperature_one(tc):
    from decider.calibrate import collect
    d = tc.decider({"temperature": 1.3, "temperature_by_type": MAP})
    golds = {"queue": "billing", "flag": True, "sev": 2}
    recs = collect(d, [(tc.STATE, tc.QUESTIONS, golds)])
    assert [r["type"] for r in recs] == ["choice", "noul", "score"] and [r["gold"] for r in recs] == [0, 1, 2]
    assert len(recs[2]["level_probs"]) == 3                               # the model reads Score with isolated levels
    ref = tc.decider({"temperature": 1.0}).system_one(tc.STATE, tc.QUESTIONS)["answers"]
    assert round(recs[1]["probs"][1], 4) == ref["flag"]["noul"]
    for bad in ({"queue": "refunds"}, {"flag": "yes"}, {"sev": 3}, {"sev": 1.5}):
        with pytest.raises(ValueError, match="gold of"):
            collect(d, [(tc.STATE, tc.QUESTIONS, bad)])
