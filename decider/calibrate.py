"""Fit decider_config.json "temperature_by_type" by NLL, one temperature per answer type (decider.temperature).

    python -m decider.calibrate records.jsonl [--min-rows 50]
    -> {"temperature_by_type": {"choice": 1.48, "noul": 2.22, "score": 1.38}, "temperature": 1.6, "rows": {...}, "nll": {...}}

A record is one answer with its gold, read at temperature 1:
    {"type": "choice" | "noul" | "score", "logits": [one value per option], "gold": option index}
    {"type": "score", "level_logits": [[no, yes], one pair per level], "gold": level index}      Score with isolated levels
"probs" / "level_probs" (probabilities at temperature 1) may stand in for "logits" / "level_logits": log p is the logit up to a
constant, which the softmax ignores.  A noul gold is 1 for true, 0 for false.  Every record is checked first (check_record):
a malformed one (gold out of range or not an integer, wrong shape, NaN, a row without a finite logit, an isolated Score whose
levels all have P(yes) = 0) raises ValueError naming its position.

An isolated-levels Score answer is fitted through the readout the server uses: every level row is softmax(row / T), and the
answer is P(yes) of each level divided by their sum (decider.systemone.combine_isolated).  So the "score" temperature is fitted on
Score answers whichever readout the model serves them with; fit it on records collected with the same isolated_levels setting.

`collect(decider, examples)` produces records from a decider.infer.Decider and labelled /v1/systemone-shaped examples.
"temperature" in the output is one temperature fitted on all records together, for comparison; a type with fewer than
--min-rows records is left out of the map (it then uses "temperature" of the config).
"""
import json, math, sys
import numpy as np

from decider.temperature import TYPES

GRID = np.exp(np.linspace(math.log(0.05), math.log(20.0), 801))       # 0.05 .. 20, about 0.75 % apart


def _log(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, np.log(np.clip(x, 1e-300, None)), -np.inf)


def _logits(rec, key):
    if key in rec:
        return np.asarray(rec[key], dtype=np.float64)
    alt = {"logits": "probs", "level_logits": "level_probs"}[key]
    return _log(rec[alt])


def _lse(z, axis=-1):
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):                              # an all -inf row gives -inf, handled by the callers
        return (m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))).squeeze(axis)


def nll(rec, T):
    """Negative log-likelihood of one record's gold at temperature T."""
    return float(_Batch([rec]).nll(T))


def _isolated(rec):
    return "level_logits" in rec or "level_probs" in rec


def check_record(rec, i=None):
    """Raise ValueError unless `rec` is a well-formed record (module docstring): an integral gold within the options or levels,
    a 1-D row of at least two options or an [levels, 2] array of at least two levels, logits that are not NaN or +inf,
    probabilities in [0, 1] with positive mass, and isolated levels only on a Score record."""
    where = f"record {i}" if i is not None else "record"
    if not isinstance(rec, dict):
        raise ValueError(f"{where}: not a JSON object")
    if rec.get("type") not in TYPES:
        raise ValueError(f"{where}: type {rec.get('type')!r}, expected one of {', '.join(TYPES)}")
    iso = _isolated(rec)
    keys = ("level_logits", "level_probs") if iso else ("logits", "probs")
    if sum(k in rec for k in keys) != 1 or (iso and ("logits" in rec or "probs" in rec)):
        raise ValueError(f"{where}: give exactly one of logits, probs, level_logits, level_probs")
    if iso and rec["type"] != "score":
        raise ValueError(f"{where}: isolated levels (level_logits / level_probs) belong to a score record, not {rec['type']!r}")
    key = keys[0] if keys[0] in rec else keys[1]
    try:
        a = np.asarray(rec[key], dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {key} is not a numeric array") from None
    if iso and (a.ndim != 2 or a.shape[1] != 2 or a.shape[0] < 2):
        raise ValueError(f"{where}: {key} must be one [no, yes] pair per level, at least two levels; got shape {a.shape}")
    if not iso and (a.ndim != 1 or a.shape[0] < 2):
        raise ValueError(f"{where}: {key} must be one value per option, at least two options; got shape {a.shape}")
    if key.endswith("probs"):
        if not np.all(np.isfinite(a)) or np.any(a < 0) or np.any(a > 1) or np.any(a.sum(-1) <= 0):
            raise ValueError(f"{where}: {key} must be probabilities in [0, 1] with positive mass per row")
    elif np.any(np.isnan(a)) or np.any(a == np.inf) or not np.all(np.any(np.isfinite(a), axis=-1)):
        raise ValueError(f"{where}: {key} contains NaN or +inf, or a row without a finite value")
    if rec["type"] == "noul" and a.shape[0] != 2:
        raise ValueError(f"{where}: a noul record has exactly two options (false, true); got {a.shape[0]}")
    if iso and not np.any(a[:, 1] > 0 if key == "level_probs" else np.isfinite(a[:, 1])):
        raise ValueError(f"{where}: no level has a yes probability above 0, so the served Score answer has no distribution")
    g = rec.get("gold")
    if isinstance(g, bool) or not isinstance(g, (int, np.integer)) or not 0 <= g < a.shape[0]:
        raise ValueError(f"{where}: gold must be an integer index in 0..{a.shape[0] - 1}, got {g!r}")
    return rec


class _Batch:
    """Records packed into padded arrays, so one temperature is scored over all of them at once."""
    def __init__(self, records):
        records = [check_record(r, i) for i, r in enumerate(records)]
        lists = [r for r in records if not _isolated(r)]
        isos = [r for r in records if _isolated(r)]
        self.n = len(lists) + len(isos)
        self.L = self.Lg = self.I = self.Im = self.Ig = None
        if lists:
            zs = [_logits(r, "logits") for r in lists]
            K = max(len(z) for z in zs)
            self.L = np.full((len(zs), K), -np.inf)
            for i, z in enumerate(zs):
                self.L[i, :len(z)] = z
            self.Lg = np.array([int(r["gold"]) for r in lists])
        if isos:
            zs = [_logits(r, "level_logits").reshape(-1, 2) for r in isos]
            n = max(len(z) for z in zs)
            self.I = np.zeros((len(zs), n, 2)); self.Im = np.zeros((len(zs), n), dtype=bool)
            for i, z in enumerate(zs):
                self.I[i, :len(z)] = z; self.Im[i, :len(z)] = True
            self.Ig = np.array([int(r["gold"]) for r in isos])

    def nll(self, T):
        """Sum of the records' NLL at temperature T."""
        tot = 0.0
        if self.L is not None:
            z = self.L / T
            zg = z[np.arange(len(z)), self.Lg]
            tot += float(np.sum(np.where(np.isfinite(zg), _lse(z) - zg, 690.0)))
        if self.I is not None:                          # in log space: P(yes) of very confident rows underflows otherwise
            z = self.I / T
            with np.errstate(invalid="ignore"):
                lp = np.where(self.Im, z[..., 1] - _lse(z), -np.inf)          # log P(yes) per level, -inf on padding
            norm = _lse(lp)                                                  # log of the summed P(yes)
            lg = lp[np.arange(len(z)), self.Ig]
            with np.errstate(invalid="ignore"):
                per = np.where(np.isfinite(lg), norm - lg, 690.0)               # check_record guarantees a finite norm
            tot += float(np.sum(per))
        return tot


def mean_nll(records, T):
    b = records if isinstance(records, _Batch) else _Batch(records)
    return b.nll(T) / b.n


def fit(records, grid=GRID):
    """The temperature with the lowest mean NLL on the grid, refined by a golden-section search between its neighbours."""
    records = records if isinstance(records, _Batch) else _Batch(list(records))
    if not records.n:
        raise ValueError("no records to fit")
    vals = [mean_nll(records, T) for T in grid]
    i = int(np.argmin(vals))
    lo, hi = math.log(grid[max(i - 1, 0)]), math.log(grid[min(i + 1, len(grid) - 1)])
    r = (math.sqrt(5) - 1) / 2
    a, b = hi - r * (hi - lo), lo + r * (hi - lo)
    fa, fb = mean_nll(records, math.exp(a)), mean_nll(records, math.exp(b))
    for _ in range(40):
        if fa < fb:
            hi, b, fb = b, a, fa; a = hi - r * (hi - lo); fa = mean_nll(records, math.exp(a))
        else:
            lo, a, fa = a, b, fb; b = lo + r * (hi - lo); fb = mean_nll(records, math.exp(b))
    T = math.exp((lo + hi) / 2)
    return T if mean_nll(records, T) <= vals[i] else float(grid[i])


def fit_by_type(records, min_rows=50):
    """-> {"temperature_by_type": {type: T}, "temperature": pooled T, "rows": {type: n}, "nll": {type: {"T=1": .., "fitted": ..}}}"""
    records = [check_record(r, i) for i, r in enumerate(records)]
    groups = {t: [] for t in TYPES}
    for r in records:
        groups[r["type"]].append(r)
    out = {"temperature_by_type": {}, "temperature": round(fit(records), 3) if records else None,
           "rows": {t: len(g) for t, g in groups.items()}, "nll": {}}
    for t, g in groups.items():
        if len(g) < max(1, min_rows):
            continue
        b = _Batch(g); T = fit(b)
        out["temperature_by_type"][t] = round(T, 3)
        out["nll"][t] = {"T=1": round(mean_nll(b, 1.0), 4), "fitted": round(mean_nll(b, T), 4)}
    return out


def collect(decider, examples, independent=True):
    """Records for fit_by_type from a decider.infer.Decider, read at temperature 1 on the uncached state-first path.
    examples: iterable of (state, questions, golds) with questions as /v1/systemone takes them and golds {question id: gold},
    the gold being a Choice option name, a Noul true/false, or a Score level index.  Questions without a gold are skipped.
    Score questions are read with the Decider's isolated_levels setting, as system_one serves them."""
    recs = []
    for state, questions, golds in examples:
        rqs, index, items = decider._system_one_items(state, questions, independent, layout="state_first")
        rows = decider._system_one_probs(items, "state_first", temperature=1.0)
        for k, kind, s, n in index:
            if k not in golds:
                continue
            rq, g = rqs[k], golds[k]
            if rq["type"] == "noul" and not (isinstance(g, bool) or g in (0, 1)):
                raise ValueError(f"gold of noul question {k!r} must be true or false, got {g!r}")
            if rq["type"] == "score" and (isinstance(g, bool) or not isinstance(g, int) or not 0 <= g < len(rq["names"])):
                raise ValueError(f"gold of score question {k!r} must be a level index in 0..{len(rq['names']) - 1}, got {g!r}")
            key = bool(g) if rq["type"] == "noul" else g
            if key not in rq["names"]:
                raise ValueError(f"gold of choice question {k!r} is not one of its options: {g!r}")
            gold = rq["names"].index(key)
            if kind == "iso":
                recs.append({"type": rq["type"], "level_probs": [rows[s + j][:2] for j in range(n)], "gold": gold})
            else:
                recs.append({"type": rq["type"], "probs": rows[s][:len(rq["options"])], "gold": gold})
    return recs


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", help="JSON lines, one record per line (see the module docstring)")
    ap.add_argument("--min-rows", type=int, default=50, help="fit a type only when it has at least this many records")
    a = ap.parse_args(argv)
    with open(a.records) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(fit_by_type(recs, a.min_rows), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
