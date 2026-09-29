"""Final-logit softcapping (Gemma) on the letter logits: DecisionModel.cap and softcap_value, without loading a model."""
import types
import pytest
torch = pytest.importorskip("torch")
from decider.model import DecisionModel, softcap_value  # noqa: E402


def test_softcap_value_reads_config_or_text_config():
    assert softcap_value(types.SimpleNamespace(final_logit_softcapping=30.0)) == 30.0
    assert softcap_value(types.SimpleNamespace(text_config=types.SimpleNamespace(final_logit_softcapping=30.0))) == 30.0
    assert softcap_value(types.SimpleNamespace()) is None                                         # Qwen: no field
    assert softcap_value(types.SimpleNamespace(final_logit_softcapping=None, text_config=None)) is None


def test_cap_matches_tanh_and_is_identity_without_softcap():
    m = object.__new__(DecisionModel); x = torch.tensor([[-80.0, -5.0, 0.0, 5.0, 80.0, float("-inf")]])
    m.softcap = None
    assert torch.equal(m.cap(x), x)
    m.softcap = 30.0
    y = m.cap(x)
    assert torch.allclose(y[0, :5], 30.0 * torch.tanh(x[0, :5] / 30.0))
    assert y.abs().max() <= 30.0          # every path masks options past nopts to -inf after cap (read_slots, slot_logits)
