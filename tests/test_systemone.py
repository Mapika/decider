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


@pytest.mark.parametrize("question_type", ["noul", "bool"])
@pytest.mark.parametrize("criteria", [["bad"], [], "bad", False])
def test_noul_criteria_must_be_a_map(question_type, criteria):
    with pytest.raises(ValueError, match="noul criteria: a map of optional true/false descriptions"):
        s1.render_question({"type": question_type, "instructions": "Refund asked?", "criteria": criteria})


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
    assert out["team"]["choice"] == "tech" and out["team"]["confidence"] == 0.6 and out["team"]["x_p_max"] == 0.8
    assert out["team"]["probabilities"] == {"billing": 0.2, "tech": 0.8}
    assert out["refund"] == {"type": "noul", "noul": 0.95}
    mood = out["mood"]
    assert mood["level_fit"] == {"0": 0.1, "1": 0.5, "2": 0.4} and mood["fit_mass"] == 1.0
    assert mood["probabilities"] == {"0": 0.1, "1": 0.5, "2": 0.4} and mood["score"] == 1.3
    assert mood["confidence"] == 0.25 and mood["x_p_max"] == 0.5        # 1 - (0.1 + 0.4) / (2/3)
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


# ---- confidence (1.3.0): TypeSafe's definitions; x_p_max keeps the top probability that `confidence` was before
REPORT = {"Red": 0.195, "Green": 0.1994, "Blue": 0.2252, "Yellow": 0.2286, "Purple": 0.1518}      # issue #15


def _choice(names):
    return s1.render_question({"type": "choice", "instructions": "Which colour?", "criteria": list(names)})


def _score(n):
    return s1.render_question({"type": "score", "instructions": "How much?", "criteria": [f"level {i}" for i in range(n)]})


def test_choice_confidence_on_the_conformance_report_example():
    a = s1.format_answer(_choice(REPORT), list(REPORT.values()))
    assert a["choice"] == "Yellow" and a["x_p_max"] == 0.2286
    assert a["confidence"] == round((5 * 0.2286 - 1) / 4, 4) == 0.0358          # the report's reference value, 0.036


@pytest.mark.parametrize("p,expected", [([0.5, 0.5], 0.0), ([0.82, 0.18], 0.64), ([0.2, 0.8], 0.6), ([1.0, 0.0], 1.0), ([0.0, 1.0], 1.0)])
def test_choice_confidence_with_two_options(p, expected):
    a = s1.format_answer(_choice(["a", "b"]), p)
    assert a["confidence"] == pytest.approx(expected, abs=1e-9) and a["x_p_max"] == max(p)


def test_choice_confidence_formula_and_range():
    assert s1.choice_confidence([1 / 3] * 3) == pytest.approx(0.0, abs=1e-12)
    assert s1.choice_confidence([0.88, 0.12]) == pytest.approx(0.76)            # the spec's response example
    assert s1.choice_confidence([0.8, 0.1, 0.1]) == pytest.approx((3 * 0.8 - 1) / 2)
    assert s1.choice_confidence([1.0]) == 1.0
    assert s1.choice_confidence([0.0, 0.0]) == 0.0
    for p in ([0.3, 0.3, 0.4], [0.999, 0.001], [1 / 255] * 255):
        assert 0.0 <= s1.choice_confidence(p) <= 1.0


@pytest.mark.parametrize("p,expected", [
    ([0.0, 0.95, 0.05], 0.925),                    # TypeSafe's documented score example
    ([0.01, 0.02, 0.07, 0.3, 0.6], 0.55),          # system-one-adapter-python's tests
    ([0.2] * 5, 0.0),
    ([0.3, 0.7], 0.4),                             # two levels: equal to the choice formula
    ([0.5, 0.0, 0.5], 0.0),                        # mass at both ends: floored at 0
    ([1.0], 1.0),
    ([0.0, 0.0, 0.0], 0.0)])                       # zero total counts as uniform, as in the adapter
def test_score_confidence(p, expected):
    assert s1.score_confidence(p) == pytest.approx(expected, abs=1e-9)


def test_score_answer_reports_both_values():
    a = s1.format_answer(_score(3), [0.0, 0.95, 0.05])
    assert a["confidence"] == 0.925 and a["x_p_max"] == 0.95 and a["score"] == 1.05
    b = s1.format_answer(_score(5), [0.01, 0.02, 0.07, 0.3, 0.6])
    assert b["confidence"] == 0.55 and b["x_p_max"] == 0.6


def test_noul_answer_has_no_confidence():
    a = s1.format_answer(s1.render_question({"type": "noul", "instructions": "Refund?"}), [0.3, 0.7])
    assert a == {"type": "noul", "noul": 0.7}


# ---- noul without instructions (1.3.0)
def test_noul_without_instructions_uses_the_criteria():
    for spec in ({"type": "noul", "criteria": {"true": "money back", "false": "anything else"}},
                 {"type": "noul", "instructions": None, "criteria": {"true": "money back", "false": "anything else"}},
                 {"type": "bool", "instructions": "", "criteria": {"true": "money back", "false": "anything else"}}):
        rq = s1.render_question(spec)
        assert rq["question"] == s1.NOUL_WITHOUT_INSTRUCTIONS == "Which answer fits the context?"
        assert rq["options"] == ["no: anything else", "yes: money back"] and rq["type"] == "noul"
    one = s1.render_question({"type": "noul", "criteria": {"true": {"what": "a refund request"}}})
    assert one["options"] == ["no", 'yes: {"what": "a refund request"}']
    assert s1.render_question({"type": "noul", "criteria": {"false": "no refund"}})["options"] == ["no: no refund", "yes"]


@pytest.mark.parametrize("criteria", [None, {}, {"true": None, "false": ""}, {"true": None, True: "refund requested"}])
def test_noul_without_instructions_or_descriptions_is_rejected(criteria):
    spec = {"type": "noul"} if criteria is None else {"type": "noul", "criteria": criteria}
    with pytest.raises(ValueError, match="noul question without instructions: criteria must describe true or false"):
        s1.render_question(spec)


def test_noul_without_instructions_and_bad_criteria_keeps_the_criteria_message():
    with pytest.raises(ValueError, match="noul criteria: a map of optional true/false descriptions"):
        s1.render_question({"type": "noul", "criteria": ["bad"]})


def test_questions_with_instructions_render_as_before():
    """Rendering of every question that gives instructions is unchanged by 1.3.0 (these are the 1.2.2 outputs)."""
    assert s1.render_question({"type": "noul", "instructions": "Refund asked?", "criteria": {"true": "money back"}}) == dict(
        question="Refund asked?", options=["no", "yes: money back"], type="noul", names=[False, True], legend=None, isolated=True)
    assert s1.render_question({"type": "noul", "instructions": {"ask": "refund?"}}) == dict(
        question='{"ask": "refund?"}', options=["no", "yes"], type="noul", names=[False, True], legend=None, isolated=True)
    assert s1.render_question({"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "other": None}})["question"] == "Which team?"
    with pytest.raises(ValueError, match="question without instructions"):
        s1.render_question({"type": "score", "criteria": ["a", "b"]})
