"""Replay {"id", "state", "questions"} rows against a running /v1/systemone server and compare two runs.

    python -m decider.bench.replay_systemone run  --url http://127.0.0.1:8000 --rows 1000 --conc 1 \
        --data sample.jsonl.gz --out runs/a_c1.jsonl
    python -m decider.bench.replay_systemone compare runs/a_c1.jsonl runs/c_c1.jsonl

`run` posts what decision_index/engines/http.py posts ({"model", "state", "questions"}), in file order, with `conc`
requests in flight, records per-request wall time and writes every response.  A 400/413/422 carrying one of the Decision
Index capacity markers counts as unsupported, any other non-200 as an error; neither is counted in the latency percentiles.

`compare` exits 1 when the two files do not cover the same request ids, when a status, an answer key set or an argmax
differs, when a probability is missing or not finite, or when any answer has an option whose probability differs by more
than --tol.  `prob_diffs_over_tol` counts answers, not options.
"""
import argparse, asyncio, gzip, json, math, os, statistics, sys, time

CAPACITY_MARKERS = ("options per choice", "a choice needs at least two options", "a score takes 2 to 10 levels",
                    "the canvas holds", "maximum context length", "maximum model length",
                    "longer than the maximum model length", "context window", "too many tokens")


def load_rows(path, n):
    op = gzip.open if str(path).endswith(".gz") else open
    out = []
    with op(path, "rt") as f:
        for line in f:
            out.append(json.loads(line))
            if n and len(out) >= n:
                break
    return out


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs); k = min(len(xs) - 1, max(0, int(math.ceil(q / 100 * len(xs))) - 1))
    return xs[k]


async def _run(a):
    import httpx
    rows = load_rows(a.data, a.rows)
    results = [None] * len(rows)
    nxt = 0; lock = asyncio.Lock()

    async def worker(cl):
        nonlocal nxt
        while True:
            async with lock:
                i = nxt; nxt += 1
            if i >= len(rows):
                return
            r = rows[i]
            body = {"model": a.model, "state": r["state"], "questions": r["questions"]}
            if a.independent is not None:
                body["independent"] = a.independent
            t = time.perf_counter()
            try:
                resp = await cl.post(a.url.rstrip("/") + "/v1/systemone", json=body, timeout=a.timeout)
                ms = (time.perf_counter() - t) * 1000
                if resp.status_code == 200:
                    results[i] = dict(id=r.get("id", i), ms=ms, status=200, response=resp.json())
                else:
                    txt = resp.text[:400]
                    kind = "unsupported" if resp.status_code in (400, 413, 422) and any(s in txt for s in CAPACITY_MARKERS) else "error"
                    results[i] = dict(id=r.get("id", i), ms=ms, status=resp.status_code, kind=kind, detail=txt)
            except Exception as e:
                results[i] = dict(id=r.get("id", i), ms=(time.perf_counter() - t) * 1000, status=0, kind="error",
                                  detail=f"{type(e).__name__}: {e}"[:400])

    limits = httpx.Limits(max_connections=max(a.conc, 8), max_keepalive_connections=max(a.conc, 8))
    async with httpx.AsyncClient(limits=limits) as cl:
        for _ in range(600):                                    # wait for the server to finish warming up
            try:
                if (await cl.get(a.url.rstrip("/") + "/health", timeout=10)).json().get("ok"):
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(cl) for _ in range(a.conc)])
        wall = time.perf_counter() - t0
    ok = [x for x in results if x and x["status"] == 200]
    uns = [x for x in results if x and x.get("kind") == "unsupported"]
    err = [x for x in results if x and x.get("kind") == "error"]
    lat = [x["ms"] for x in ok]
    summary = dict(label=a.label, url=a.url, rows=len(rows), conc=a.conc, ok=len(ok), unsupported=len(uns), errors=len(err),
                   wall_s=round(wall, 2), throughput_rps=round(len(ok) / wall, 2) if wall else 0,
                   median_ms=round(statistics.median(lat), 1) if lat else None,
                   mean_ms=round(statistics.fmean(lat), 1) if lat else None,
                   p95_ms=round(pct(lat, 95), 1), p99_ms=round(pct(lat, 99), 1),
                   max_ms=round(max(lat), 1) if lat else None)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(json.dumps({"_summary": summary}) + "\n")
            for x in results:
                f.write(json.dumps(x) + "\n")
    print(json.dumps(summary, indent=1))
    if err[:3]:
        print("first errors:", json.dumps([dict(id=x["id"], status=x["status"], detail=x["detail"][:200]) for x in err[:3]], indent=1))
    return summary


# ---- agreement -----------------------------------------------------------
def _answer_pick(a):
    """The reported decision of one answer: the argmax label for a choice, the level for a score, the side for a noul."""
    t = a.get("type")
    if t == "choice":
        return ("choice", a.get("choice"))
    if t == "score":
        p = a.get("probabilities") or {}
        return ("score", max(p, key=lambda k: p[k]) if p else None)
    return ("noul", (a.get("noul", 0) >= 0.5))


def _probs(a):
    p = a.get("probabilities")
    if p is not None:
        return dict(p)
    if a.get("type") == "noul":
        return {"true": a.get("noul", 0.0), "false": 1 - a.get("noul", 0.0)}
    return {}


def compare(pa, pb, tol=1e-3, show=8):
    def load(p):
        out = {}
        with open(p) as f:
            for line in f:
                x = json.loads(line)
                if "_summary" in x: continue
                out[x["id"]] = x
        return out
    A, B = load(pa), load(pb)
    ids = [i for i in A if i in B]
    missing = [i for i in A if i not in B] + [i for i in B if i not in A]
    both_ok = [i for i in ids if A[i]["status"] == 200 and B[i]["status"] == 200]
    status_diff = [i for i in ids if A[i]["status"] != B[i]["status"]]
    n_ans = 0; pick_diff = []; worst = 0.0; worst_at = None; over_tol = []; nonfinite = []
    key_diff = []
    for i in both_ok:
        aa, bb = A[i]["response"]["answers"], B[i]["response"]["answers"]
        if set(aa) != set(bb):
            key_diff.append(i); continue
        for k in aa:
            n_ans += 1
            if _answer_pick(aa[k]) != _answer_pick(bb[k]):
                pick_diff.append((i, k, _answer_pick(aa[k]), _answer_pick(bb[k])))
            pa_, pb_ = _probs(aa[k]), _probs(bb[k])
            if set(pa_) != set(pb_) or not all(math.isfinite(float(v)) for v in list(pa_.values()) + list(pb_.values())):
                nonfinite.append((i, k)); continue
            d_ans = max(abs(float(pa_[o]) - float(pb_[o])) for o in pa_) if pa_ else 0.0
            if d_ans > worst:
                worst, worst_at = d_ans, (i, k, max(pa_, key=lambda o: abs(float(pa_[o]) - float(pb_[o]))))
            if d_ans > tol: over_tol.append((i, k, d_ans))
    usage_diff = [i for i in both_ok if A[i]["response"].get("usage") != B[i]["response"].get("usage")]
    out = dict(a=pa, b=pb, requests_a=len(A), requests_b=len(B), missing_ids=len(missing), compared_requests=len(both_ok),
               compared_answers=n_ans, status_mismatches=len(status_diff), answer_key_mismatches=len(key_diff),
               nonfinite_or_missing_probs=len(nonfinite), argmax_mismatches=len(pick_diff), prob_diffs_over_tol=len(over_tol),
               tol=tol, max_abs_prob_diff=round(worst, 6), max_at=worst_at, usage_mismatches=len(usage_diff))
    out["ok"] = not (missing or status_diff or key_diff or nonfinite or pick_diff or over_tol)
    print(json.dumps(out, indent=1))
    for x in pick_diff[:show]: print("  argmax:", x)
    for x in over_tol[:show]: print("  prob  :", x)
    for i in status_diff[:show]: print("  status:", i, A[i]["status"], B[i]["status"])
    for i in missing[:show]: print("  missing:", i)
    for x in nonfinite[:show]: print("  nonfinite/missing probs:", x)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="replay_systemone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8000")
    r.add_argument("--data", required=True, help=".jsonl or .jsonl.gz of {id, state, questions} rows")
    r.add_argument("--rows", type=int, default=1000)
    r.add_argument("--conc", type=int, default=1)
    r.add_argument("--model", default="default")
    r.add_argument("--independent", type=lambda s: s == "1", default=None)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--out", default=None)
    r.add_argument("--label", default="")
    c = sub.add_parser("compare")
    c.add_argument("a"); c.add_argument("b")
    c.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return asyncio.run(_run(a))
    else:
        out = compare(a.a, a.b, a.tol)
        sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
