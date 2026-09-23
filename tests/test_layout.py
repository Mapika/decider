"""Prompt layouts (1.2.0).

* Plain layout: every prompt the package builds for a plain-layout model (no "layout" key in decider_config.json: decider-2b
  v10, 4B, 35B, 0.8B, vision) has the token ids 1.1.3 produced (1.1.4 did not change prompt code).  tests/data/plain_layout_pins.json holds sha256 digests of
  the items from 1.1.3 (origin/main 572d752) over the cases in tests/layout_cases.py; the scoring engines are unchanged, so
  equal ids give equal probabilities.
* Chat layout (decider-2b v11): the package's rendering equals, token for token, the reference renderers the model was
  trained and served with (the research server's `build_chat` for state-first rows, the training builder's schema-first
  wrapping), copied below as they are.
* decider_config.json layout resolution, and the refusal of an unknown layout.

No torch needed except for the Decider refusal test.  The plain pins need the Qwen/Qwen3.5-2B-Base tokenizer; the chat tests
need a tokenizer with the Qwen3.5 chat template (DECIDER_TEST_CHAT_TOKENIZER, default Qwen/Qwen3.5-2B, else the Base one,
which ships the same template).  Both skip when unavailable.
"""
import json, os
from pathlib import Path

import pytest

import layout_cases as LC
from decider import prompt as P
from decider.prompt_fast import build_rows

PINS = Path(__file__).parent / "data" / "plain_layout_pins.json"
QWEN_HEAD = "<|im_start|>user\n"
QWEN_TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _load_tok(names):
    transformers = pytest.importorskip("transformers")
    for name in names:
        if not name:
            continue
        try:
            return transformers.AutoTokenizer.from_pretrained(name)
        except Exception:
            continue
    pytest.skip(f"no tokenizer available among {names}")


@pytest.fixture(scope="module")
def base_tok():
    return _load_tok(["Qwen/Qwen3.5-2B-Base"])


@pytest.fixture(scope="module")
def chat_tok():
    t = _load_tok([os.environ.get("DECIDER_TEST_CHAT_TOKENIZER", ""), "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base"])
    if not getattr(t, "chat_template", None):
        pytest.skip("tokenizer has no chat template")
    return t


# ---- plain layout: pinned to 1.1.3 ---------------------------------------------------------------------------------
def _check_pins(prefix, cases):
    pins = json.load(open(PINS))["pins"]
    bad = [k for k, v in cases.items() if pins.get(prefix + k) != LC.digest(v)]
    assert not bad, f"{len(bad)} plain-layout prompts differ from 1.1.3, e.g. {bad[:5]}"
    return len(cases)


def test_plain_prompt_functions_match_1_1_3(base_tok):
    assert _check_pins("prompt:", LC.prompt_cases(base_tok)) == 149


def test_plain_decider_prompts_match_1_1_3(base_tok):
    pytest.importorskip("torch")
    n = 0
    for neu in (True, False):
        for iso in (True, False):
            n += _check_pins(f"decider:neutralize={neu}:iso={iso}:", LC.decider_items(base_tok, neutralize=neu, isolated_levels=iso))
    assert n == 4 * 34


def test_plain_serve_prompts_match_1_1_3(base_tok):
    pytest.importorskip("fastapi")
    for neu in (True, False):
        _check_pins(f"serve:neutralize={neu}:", LC.serve_cases(base_tok, neutralize=neu))


def test_every_pin_is_checked(base_tok):
    """The pin file and the cases stay in step: no pin is left without a case."""
    pins = json.load(open(PINS))["pins"]
    names = {"prompt:" + k for k in LC.prompt_cases(base_tok)}
    assert len(pins) == 359 and names <= set(pins)


# ---- layout resolution ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg,layout", [({}, "plain"), (None, "plain"), ({"temperature": 1.3, "schema_first_trained": True}, "plain"),
                                        ({"layout": "plain"}, "plain"), ({"layout": "chat"}, "chat"), ({"chat_template": True}, "chat"),
                                        ({"layout": "chat", "chat_template": True}, "chat"), ({"chat_template": False}, "plain")])
def test_resolve_layout(cfg, layout):
    assert P.resolve_layout(cfg) == layout


@pytest.mark.parametrize("cfg", [{"layout": "xml_v2"}, {"layout": "Chat"}, {"layout": ""}, {"layout": "plain", "chat_template": True}])
def test_unknown_or_contradictory_layout_is_refused(cfg):
    with pytest.raises(ValueError) as e:
        P.resolve_layout(cfg)
    if cfg.get("layout") not in ("plain",):
        assert repr(cfg["layout"]) in str(e.value)


def test_decider_refuses_unknown_layout_before_loading(tmp_path):
    pytest.importorskip("torch")
    from decider.infer import Decider
    (tmp_path / "decider_config.json").write_text(json.dumps({"layout": "xml_v2", "temperature": 1.0}))
    with pytest.raises(ValueError, match="xml_v2"):
        Decider(str(tmp_path), device="cpu")              # no weights in the folder: the layout check comes first


def test_server_refuses_unknown_layout():
    pytest.importorskip("fastapi")
    from decider import serve
    with pytest.raises(ValueError, match="xml_v2"):
        serve.apply_config({"layout": "xml_v2"})
    serve.apply_config({})
    assert serve.LAYOUT == "plain"


def test_chat_template_without_template_is_refused(base_tok):
    class NoTemplate:
        chat_template = None
    with pytest.raises(ValueError, match="no chat template"):
        P.ChatTemplate(NoTemplate())


# ---- chat layout: equal to the reference renderers ----------------------------------------------------------------------
class RefChatTemplate:
    """decider2/serve_chat.py ChatTemplate with system=None, instruction=None, answer="Answer" (the served v11 configuration)."""
    SENTINEL = "@@DECIDER_USER_CONTENT@@"

    def __init__(self, tok, system=None, instruction=None, answer="Answer"):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": self.SENTINEL}]
        kw = {"enable_thinking": False} if "enable_thinking" in tok.chat_template else {}
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        head, tail = s.split(self.SENTINEL)
        for opened, closed in (("<think>\n", "</think>\n\n"), ("<think>", "</think>\n\n"), ("<|channel>thought\n", "<channel|>")):
            if tail.endswith(opened):
                tail += closed
        self.head_text, self.tail_text = head, tail
        self.head = tok.encode(head, add_special_tokens=False); self.tail = tok.encode(tail, add_special_tokens=False)
        self.instruction = instruction; self.answer = answer

    def answer_ids(self, tok, k, multi):
        return tok.encode(f"{'' if k == 0 else chr(10)}{self.answer}{' ' + str(k + 1) if multi else ''}: (", add_special_tokens=False)


def ref_build_chat(example, tok, tmpl, max_ctx_tokens=32768):
    """decider2/serve_chat.py build_chat (state-first; options in the order given)."""
    qs = example.qs; multi = len(qs) > 1
    ids = list(tmpl.head) + tok.encode("Context:\n" + example.context, add_special_tokens=False)[:max_ctx_tokens]
    nopts, perms = [], []
    for k, q in enumerate(qs):
        opts = list(range(len(q.options)))
        ids += tok.encode(f"\n\nQuestion{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False) + P._options_ids(tok, q, opts)
        nopts.append(len(opts)); perms.append(opts)
    if tmpl.instruction:
        ids += tok.encode("\n\n" + tmpl.instruction, add_special_tokens=False)
    ids += tmpl.tail
    slots = []
    for k in range(len(qs)):
        ids += tmpl.answer_ids(tok, k, multi); slots.append(len(ids) - 1)
    return dict(ids=ids, slots=slots, nopts=nopts, perms=perms)


def ref_build_arch_chat_schema_first(example, tok, tmpl, max_ctx_tokens, perms):
    """decider2/v11/chat_layout.py build_arch_chat(layout="schema_first", option_perms=perms): ids, slots, prefix_len."""
    qs = example.qs; multi = len(qs) > 1
    ctx = tok.encode(example.context, add_special_tokens=False)[:max_ctx_tokens]
    ids = list(tmpl.head)
    for k, q in enumerate(qs):
        sep = "\n\n" if k else ""
        ids += tok.encode(f"{sep}Question{' ' + str(k + 1) if multi else ''}: {q.text}\nOptions:", add_special_tokens=False)
        ids += P._options_ids(tok, q, perms[k])
    prefix_len = len(ids)
    ids += tok.encode("\n\nContext:\n", add_special_tokens=False) + ctx
    ids += tmpl.tail
    slots = []
    for k in range(len(qs)):
        ids += tmpl.answer_ids(tok, k, multi); slots.append(len(ids) - 1)
    return dict(ids=ids, slots=slots, prefix_len=prefix_len)


def test_chat_template_is_the_served_one(chat_tok):
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok)
    assert (ct.head_text, ct.tail_text) == (QWEN_HEAD, QWEN_TAIL) == (ref.head_text, ref.tail_text)
    assert ct.head == ref.head and ct.tail == ref.tail and len(ct.head) == 3 and len(ct.tail) == 9
    assert P.chat_template(chat_tok) is ct


def test_state_first_chat_equals_reference(chat_tok):
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok); n = 0
    for name, ctx, qs in LC.EXAMPLES:
        ex = LC.Ex(ctx, qs)
        for cap in (32768, 1536, 64, 5):
            a = P.build(ex, chat_tok, LC.NoShuffle(), max_options=P.MAX_OPTIONS, max_ctx_tokens=cap, chat=ct)
            b = ref_build_chat(ex, chat_tok, ref, cap)
            assert a["ids"] == b["ids"] and a["slots"] == b["slots"] and a["nopts"] == b["nopts"] and a["perms"] == b["perms"], (name, cap)
            assert all(a["ids"][s] == a["ids"][a["slots"][0]] for s in a["slots"])            # every slot is the " (" token
            n += 1
    assert n == 4 * len(LC.EXAMPLES)


def test_state_first_chat_text(chat_tok):
    ct = P.chat_template(chat_tok)
    ex = LC.Ex("state", [LC.Q("Which?", ["a", "b"]), LC.Q("Urgent?", ["no", "yes"])])
    txt = chat_tok.decode(P.build(ex, chat_tok, LC.NoShuffle(), chat=ct)["ids"])
    assert txt == (QWEN_HEAD + "Context:\nstate\n\nQuestion 1: Which?\nOptions:\n(A) a\n(B) b\n\nQuestion 2: Urgent?\nOptions:\n(A) no\n(B) yes"
                   + QWEN_TAIL + "Answer 1: (\nAnswer 2: (")


def test_schema_first_chat_equals_reference(chat_tok):
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok)
    for name, ctx, qs in LC.EXAMPLES:
        ex = LC.Ex(ctx, qs)
        for cap in (32768, 64):
            a = P.build(ex, chat_tok, LC.NoShuffle(), max_options=P.MAX_OPTIONS, max_ctx_tokens=cap, layout="schema_first", chat=ct)
            b = ref_build_arch_chat_schema_first(ex, chat_tok, ref, cap, a["perms"])
            assert a["ids"] == b["ids"] and a["slots"] == b["slots"] and a["prefix_len"] == b["prefix_len"], (name, cap)
            pre = P.schema_prefix_ids(chat_tok, qs, a["perms"], chat=ct)
            suf, slots = P.schema_suffix_ids(chat_tok, ctx, len(qs), cap, chat=ct)
            assert pre + suf == a["ids"] and [len(pre) + s for s in slots] == a["slots"]


def test_build_rows_chat_equals_build_chat(chat_tok):
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok)
    for st in LC.STATES:
        for rows in ([[("Which queue?", ["billing", "technical", "sales"])], [("Which label?", [f"o{j}" for j in range(40)])]],
                     [[("Which queue?", ["billing", "technical", "sales"]), ("Urgent?", ["no", "yes"]), ("Which label?", [f"o{j}" for j in range(40)])]],
                     [[(f"Question number {j}?", ["no", "yes"])] for j in range(12)]):
            for cap in (32768, 64):
                items, ctx_len = build_rows(chat_tok, st, rows, max_ctx_tokens=cap, chat=ct)
                for it, row in zip(items, rows):
                    b = ref_build_chat(LC.Ex(st, [LC.Q(t, list(o)) for t, o in row]), chat_tok, ref, cap)
                    assert it["ids"] == b["ids"] and it["slots"] == b["slots"] and it["nopts"] == b["nopts"]
                    assert it["ids"][:ctx_len] == items[0]["ids"][:ctx_len]


def test_serve_prepare_chat_equals_reference(chat_tok):
    pytest.importorskip("fastapi")
    from decider import serve, systemone as S1
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok)
    for st in LC.S1_STATES:
        for independent, isolated in ((True, False), (True, True), (False, False)):
            rqs, index, items, ctx_len = serve.prepare(chat_tok, st, LC.QUESTIONS, independent, isolated, 32768, chat=ct)
            flat, _ = S1.plan_rows(rqs, isolated and independent)
            rows = [[r] for r in flat] if independent else [flat]
            assert len(rows) == len(items)
            for it, row in zip(items, rows):
                b = ref_build_chat(LC.Ex(S1.render_state(st), [LC.Q(r["question"], list(r["options"])) for r in row]), chat_tok, ref)
                assert it["ids"] == b["ids"] and it["slots"] == b["slots"]
            assert S1.unique_tokens(items) == __import__("decider.prompt_fast", fromlist=["x"]).unique_tokens(items, ctx_len)


def test_serve_decide_and_decider_chat_equal_reference(chat_tok):
    pytest.importorskip("fastapi"); pytest.importorskip("torch")
    ct = P.chat_template(chat_tok); ref = RefChatTemplate(chat_tok)
    got = LC.serve_cases(chat_tok, chat=ct, neutralize=False)
    for i, st in enumerate(LC.STATES):
        opts, it = got[f"decide/{i}"]
        b = ref_build_chat(LC.Ex(st, [LC.Q(q, o) for q, o in zip(LC.SCHEMA, opts)]), chat_tok, ref, 1536)
        assert it["ids"] == b["ids"] and it["slots"] == b["slots"]
    d = LC.decider_items(chat_tok, chat=ct, neutralize=False)
    for key, (opts_rows, items) in [(k, v) for k, v in d.items() if k.startswith("decide_batch/")]:
        cap = int(key.split("/")[1])
        for st, opts, it in zip(LC.STATES, opts_rows, items):
            qtexts = ["Which queue?", "Urgent?", "Which label?"]
            b = ref_build_chat(LC.Ex(st, [LC.Q(q, o) for q, o in zip(qtexts, opts)]), chat_tok, ref, cap)
            assert it["ids"] == b["ids"] and it["slots"] == b["slots"]
    # every chat item starts with the template head and has the tail right before the first answer piece
    for k, v in d.items():
        items = v[1]
        for it in items:
            assert it["ids"][:3] == ct.head
            first = it["slots"][0] - len(ct.answer_ids(chat_tok, 0, len(it["slots"]) > 1)) + 1
            assert it["ids"][first - 9:first] == ct.tail, k


def test_plain_and_chat_differ_only_by_the_wrapping(chat_tok):
    """For one single-question row the chat ids are head + plain content with the question header and option block tokenized
    apart + tail + "Answer: (": decoded, the user turn is the plain prompt's text up to its answer piece."""
    ct = P.chat_template(chat_tok)
    ex = LC.Ex("My card was charged twice.", [LC.Q("Which team?", ["billing", "technical", "sales"])])
    plain = chat_tok.decode(P.build(ex, chat_tok, LC.NoShuffle())["ids"])
    chat = chat_tok.decode(P.build(ex, chat_tok, LC.NoShuffle(), chat=ct)["ids"])
    assert plain.endswith("\nAnswer: (")
    assert chat == QWEN_HEAD + plain[:-len("\nAnswer: (")] + QWEN_TAIL + "Answer: ("


def test_scripts_resolve_the_layout_from_a_folder_and_refuse_nothing_for_plain(tmp_path):
    """Codex review of 1.2.0: evaluation, probes and benchmarks must build chat prompts for a chat model."""
    import json
    from decider.prompt import load_decider_config, chat_for_model
    d = tmp_path / "m"; d.mkdir()
    assert load_decider_config(str(d)) == {}
    (d / "decider_config.json").write_text(json.dumps({"layout": "chat"}))
    assert load_decider_config(str(d)) == {"layout": "chat"}
    (d / "decider_config.json").write_text(json.dumps({"temperature": 1.3}))

    class Tok:                       # never touched for a plain-layout config
        pass
    assert chat_for_model(str(d), Tok()) is None


def test_run_eval_passes_chat_to_build(monkeypatch):
    pytest.importorskip("torch")
    from decider import evaluate as E
    seen = []

    def fake_build(e, tok, rng, **kw):
        seen.append(kw.get("chat")); raise StopIteration
    monkeypatch.setattr(E, "build", fake_build)

    class M:
        tok = None
        def eval(self): pass
        def parameters(self):
            import torch; yield torch.zeros(1)
    sentinel = object()
    try:
        E.run_eval(M(), {"t": [object()]}, chat=sentinel, log=lambda *a: None)
    except StopIteration:
        pass
    assert seen == [sentinel]
