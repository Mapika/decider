"""Answer temperatures from decider_config.json (1.4.0).

Every answer is softmax(logits / T) over its option letters.  decider_config.json sets T:

    "temperature": 1.3                                   one value for every answer (the only form before 1.4.0)
    "temperature_by_type": {"choice": 1.48, "noul": 2.22, "score": 1.38}
                                                         optional; one value per answer type, a missing type uses "temperature"
    "temperature_schema_first": 1.18                     optional; the schema cache (questions-first layout), as before
    "temperature_schema_first_by_type": {...}            optional; per answer type on the schema cache

The answer types are the /v1/systemone question types.  A /decide field maps onto them: "choice" -> "choice", "bool" -> "noul",
"scale" -> "score".  Plain `Decider.decide` questions (a question and its options, no type) are "choice".  A Score question
read with isolated levels (one yes/no row per level) uses the "score" temperature on every one of its level rows: the rows form
one Score answer, and a temperature fitted on Score answers is fitted through that same readout (decider.calibrate).

On the state-first layout the temperature of an answer of type t is temperature_by_type[t], else temperature.  On the schema
cache it is temperature_schema_first_by_type[t], else temperature_schema_first, else (no schema-first value at all) the
state-first temperature of t.  An explicit override (Decider(temperature=...), DECIDER_TEMPERATURE) replaces the state-first
"temperature" and switches temperature_by_type off, so the override is the one temperature of every state-first answer, as it
was before 1.4.0.

A config without a by-type map gives every path the single number it gave in 1.3.0, and that number reaches the engines as the
same Python float, so the probabilities are bit-identical to 1.3.0.
"""
import math

TYPES = ("choice", "noul", "score")
FIELD_TYPES = {"choice": "choice", "bool": "noul", "scale": "score"}        # /decide field type -> answer type
_KEYS_TEXT = ('the keys are "choice", "noul" and "score" (a /decide "bool" field is "noul", a "scale" field is "score"; '
              'a /v1/systemone "bool" question is "noul")')


def positive(value, where):
    """A temperature: a number (or a numeric string, as float() reads it, which 1.3.0 accepted for "temperature") that is finite
    and > 0.  Raises ValueError naming `where`."""
    if isinstance(value, bool):
        raise ValueError(f"{where} must be a finite number > 0, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a finite number > 0, got {value!r}") from None
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{where} must be a finite number > 0, got {value!r}")
    return v


def by_type(m, where):
    """Validate a {answer type: temperature} map.  None -> {}.  Unknown keys, non-numbers and values that are not finite and > 0
    raise ValueError."""
    if m is None:
        return {}
    if not isinstance(m, dict):
        raise ValueError(f"{where} must be a map {{answer type: temperature}}, got {type(m).__name__}; " + _KEYS_TEXT)
    out = {}
    for k, v in m.items():
        if k not in TYPES:
            raise ValueError(f"{where} has the unknown key {k!r}; " + _KEYS_TEXT)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}[{k!r}] must be a finite number > 0, got {v!r}")
        out[k] = positive(v, f"{where}[{k!r}]")
    return out


def from_config(cfg, temperature=None, temperature_by_type=None):
    """-> ((T, by_type) for the state-first layout, (T, by_type) for the schema cache).

    temperature / temperature_by_type: explicit overrides (Decider arguments, DECIDER_TEMPERATURE).  An explicit temperature
    without an explicit map switches the config's map off (see the module docstring)."""
    cfg = cfg or {}
    where = "decider_config.json"
    if temperature is not None:
        T = positive(temperature, "temperature")
        m = by_type(temperature_by_type, "temperature_by_type") if temperature_by_type is not None else {}
    else:
        T = positive(cfg.get("temperature", 1.0), f'{where} "temperature"')
        m = (by_type(temperature_by_type, "temperature_by_type") if temperature_by_type is not None
             else by_type(cfg.get("temperature_by_type"), f'{where} "temperature_by_type"'))
    ms = by_type(cfg.get("temperature_schema_first_by_type"), f'{where} "temperature_schema_first_by_type"')
    if "temperature_schema_first" in cfg:
        schema = (positive(cfg["temperature_schema_first"], f'{where} "temperature_schema_first"'), ms)
    else:
        schema = (T, {**m, **ms})
    return (T, m), schema


def effective(T, m):
    """{answer type: the temperature it gets} for reporting (/health, the ready line)."""
    return {t: m.get(t, T) for t in TYPES}


def for_types(T, m, types):
    """One temperature per slot, or the scalar T itself when there is no map (the 1.3.0 call)."""
    if not m:
        return T
    return [m.get(t, T) for t in types]


def item_types(it):
    """The answer type of every slot of a prompt item ("types", set where the item is built); None for an item without it."""
    ts = it.get("types")
    return list(ts) if ts is not None else [None] * len(it["slots"])


def for_items(T, m, items):
    """The `temperature` argument of Engine.score_items / score_shared: the scalar T when there is no map (the 1.3.0 call),
    else one list of per-slot temperatures per item."""
    if not m:
        return T
    return [[m.get(t, T) for t in item_types(it)] for it in items]


def slot_temperatures(temperature, items):
    """Engine side.  temperature: a scalar (returned as it is), or one entry per item, each a scalar or one value per slot of
    that item.  -> the scalar, or a flat list with one temperature per slot in item order."""
    if not isinstance(temperature, (list, tuple)):
        return temperature
    if len(temperature) != len(items):
        raise ValueError(f"temperature: {len(temperature)} entries for {len(items)} items")
    flat = []
    for t, it in zip(temperature, items):
        n = len(it["slots"])
        if isinstance(t, (list, tuple)):
            if len(t) != n:
                raise ValueError(f"temperature: {len(t)} values for an item with {n} slots")
            flat += list(t)
        else:
            flat += [t] * n
    return flat


def item_slice(temperature, lo, hi):
    """The per-item temperature entries of items[lo:hi] (a scalar is shared by every item)."""
    return temperature[lo:hi] if isinstance(temperature, (list, tuple)) else temperature


def scaled_softmax(lg, temperature):
    """softmax(lg / T) over the last axis.  A scalar T is the 1.3.0 expression unchanged; a list gives one T per row of lg."""
    import torch
    if isinstance(temperature, (list, tuple)):
        if len(temperature) != lg.shape[0]:
            raise ValueError(f"temperature: {len(temperature)} values for {lg.shape[0]} slots")
        t = torch.tensor(temperature, dtype=lg.dtype).to(lg.device, non_blocking=True)[:, None]
        return torch.softmax(lg / t, -1)
    return torch.softmax(lg / temperature, -1)
