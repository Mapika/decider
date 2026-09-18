"""The request/answer layer (no model, no torch): Jev-shaped questions -> prompt rows -> typed answers."""
import json, math, pytest
from decider import systemone as s1


def test_choice_with_descriptions_json_rubrics_and_null():
    rq = s1.render_question({"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "charges", "returns": {"what": "refunds", "not_for": "delivery"}, "other": None}})
    assert rq["type"] == "choice" and rq["names"] == ["billing", "returns", "other"]
    assert rq["options"][0] == "billing: charges"
    assert rq["options"][1] == 'returns: {"what": "refunds", "not_for": "delivery"}'
    assert rq["options"][2] == "other"                       # a null description shows the bare name


def test_choice_accepts_a_plain_list_and_rejects_bad_sizes():
    assert s1.render_question({"type": "choice", "instructions": "q", "criteria": ["a", "b"]})["names"] == ["a", "b"]
    with pytest.raises(ValueError):
        s1.render_question({"type": "choice", "instructions": "q", "criteria": {"only": None}})
    with pytest.raises(ValueError):
        s1.render_question({"type": "choice", "instructions": "q", "criteria": {str(i): None for i in range(256)}})
    with pytest.raises(ValueError):
        s1.render_question({"type": "choice", "criteria": {"a": None, "b": None}})     # no instructions


def test_score_levels_list_or_legend_map():
    a = s1.render_question({"type": "score", "instructions": "How bad?", "criteria": ["fine", "bad", "awful"]})
    b = s1.render_question({"type": "score", "instructions": "How bad?", "criteria": {"2": "awful", "0": "fine", "1": "bad"}})
    assert a["options"] == b["options"] == ["0: fine", "1: bad", "2: awful"]
    assert a["legend"] == ["fine", "bad", "awful"] and a["names"] == [0, 1, 2]


def test_noul_with_and_without_criteria():
    plain = s1.render_question({"type": "noul", "instructions": "Refund asked?"})
    desc = s1.render_question({"type": "noul", "instructions": "Refund asked?", "criteria": {"true": "money back", "false": "anything else"}})
    assert plain["options"] == ["no", "yes"] and plain["names"] == [False, True]
    assert desc["options"] == ["no: anything else", "yes: money back"]


def test_render_state_serialises_json_and_indexes_long_arrays():
    assert s1.render_state("plain text") == "plain text"
    short = json.loads(s1.render_state({"items": [{"a": 1}, {"a": 2}]}))
    assert short == {"items": [{"a": 1}, {"a": 2}]}                         # arrays under ANNOTATE_MIN are untouched
    long = json.loads(s1.render_state({"items": [{"a": i} for i in range(10)] + [], "n": 3}))
    assert long["items"][7] == {"_index": 7, "a": 7}
    scalars = json.loads(s1.render_state(list(range(9))))
    assert scalars[4] == {"_index": 4, "value": 4}
    assert json.loads(s1.render_state({"items": list(range(9))}, index_arrays=False)) == {"items": list(range(9))}


def test_plan_rows_isolates_score_levels_and_assembles_answers():
    rqs = {"team": s1.render_question({"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}}),
           "mood": s1.render_question({"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}),
           "refund": s1.render_question({"type": "noul", "instructions": "Refund?"})}
    rows, index = s1.plan_rows(rqs, isolated=True)
    assert [(k, kind, n) for k, kind, _, n in index] == [("team", "list", 1), ("mood", "iso", 3), ("refund", "list", 1)]
    assert len(rows) == 5
    assert rows[1]["question"] == "How angry?\nProposed answer: calm\nDoes the proposed answer fit?"
    assert rows[1]["options"] == ["no", "yes"]                               # the level sees neither its number nor its neighbours
    probs = [[0.2, 0.8], [0.9, 0.1], [0.5, 0.5], [0.6, 0.4], [0.05, 0.95]]
    out = s1.assemble(rqs, index, probs)
    assert out["team"]["choice"] == "tech" and out["team"]["confidence"] == 0.8
    assert out["team"]["probabilities"] == {"billing": 0.2, "tech": 0.8}
    assert out["refund"] == {"type": "noul", "noul": 0.95}
    mood = out["mood"]
    assert mood["level_fit"] == {"0": 0.1, "1": 0.5, "2": 0.4} and mood["fit_mass"] == 1.0
    assert mood["probabilities"] == {"0": 0.1, "1": 0.5, "2": 0.4} and mood["score"] == 1.3
    assert mood["legend"] == {"0": "calm", "1": "annoyed", "2": "furious"}


def test_plan_rows_listwise_when_isolation_is_off():
    rqs = {"mood": s1.render_question({"type": "score", "instructions": "How angry?", "criteria": ["calm", "furious"], "isolated": False})}
    rows, index = s1.plan_rows(rqs, isolated=True)
    assert index == [("mood", "list", 0, 1)] and rows[0]["options"] == ["0: calm", "1: furious"]


def test_certainty_and_combine_isolated():
    assert s1.certainty([1.0, 0.0, 0.0]) == 1.0
    assert abs(s1.certainty([1 / 3] * 3)) < 1e-9
    p, mass = s1.combine_isolated([0.2, 0.6, 0.2])
    assert abs(sum(p) - 1) < 1e-9 and abs(mass - 1.0) < 1e-9 and p[1] == 0.6


def test_strip_level_number():
    assert s1.strip_level_number("2: somewhat") == "somewhat"
    assert s1.strip_level_number("-1 : negative") == "negative"
    assert s1.strip_level_number("no number") == "no number"


def test_unique_tokens_counts_a_shared_prefix_once():
    items = [{"ids": [1, 2, 3, 4]}, {"ids": [1, 2, 9]}, {"ids": [1, 2, 3, 7, 8]}]
    assert s1.unique_tokens(items) == 2 + 2 + 1 + 3
    assert s1.unique_tokens([{"ids": [1, 2]}]) == 2
