"""Closed-loop load test: N concurrent clients hitting POST /decide.  python -m decider.bench.loadtest [url] [conc,...] [systemone] [packed] [state_first]
systemone: POST /v1/systemone with the same five questions in the Jev shape (independent per-question scoring unless `packed`;
`state_first` forces the uncached layout on a schema-first model)."""
import asyncio, json, random, sys, time, httpx, numpy as np
from decider import data as D


def _cli():
    URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4,16,64").split(",")]
    _, evals = D.load_cache("data/tasks.pkl")
    SCHEMA = {"Revenue currently impacted?": {"type": "bool"}, "What business impact?": {"type": "choice", "options": ["none", "degraded", "outage"]},
              "Which queue?": {"type": "choice", "options": ["billing", "technical", "sales", "hr", "none of the above"]},
              "Priority level?": {"type": "scale", "legend": {"0": "low", "1": "medium", "2": "high"}}, "Human attention needed?": {"type": "bool"}}
    ctxs = [e.context for e in evals["support_tickets"][:400]]
    S1 = "systemone" in sys.argv; PACKED = "packed" in sys.argv; LAYOUT = "state_first" if "state_first" in sys.argv else None
    QUESTIONS = {"revenue": {"type": "noul", "instructions": "Revenue currently impacted?"}, "impact": {"type": "choice", "instructions": "What business impact?", "criteria": ["none", "degraded", "outage"]},
                 "queue": {"type": "choice", "instructions": "Which queue?", "criteria": ["billing", "technical", "sales", "hr", "none of the above"]},
                 "priority": {"type": "score", "instructions": "Priority level?", "criteria": ["low", "medium", "high"]}, "human": {"type": "noul", "instructions": "Human attention needed?"}}


    async def client(cl, n, lat, dur):
        t_end = time.monotonic() + dur
        while time.monotonic() < t_end:
            t = time.monotonic()
            if S1: r = await cl.post(URL + "/v1/systemone", json={"state": random.choice(ctxs), "questions": QUESTIONS, "independent": not PACKED, "layout": LAYOUT}, timeout=60)
            else: r = await cl.post(URL + "/decide", json={"context": random.choice(ctxs), "schema": SCHEMA}, timeout=60)
            r.raise_for_status(); lat.append(time.monotonic() - t)


    async def main():
        async with httpx.AsyncClient() as cl:
            while True:
                try:
                    if (await cl.get(URL + "/health", timeout=5)).json().get("ok"): break
                except Exception: pass
                await asyncio.sleep(1)
            for conc in CONCS:
                lat = []; dur = 15
                t0 = time.monotonic()
                await asyncio.gather(*[client(cl, i, lat, dur) for i in range(conc)])
                el = time.monotonic() - t0; lat = np.array(lat) * 1000
                print(f"conc={conc:3d}: {len(lat)/el:7.1f} req/s  {len(lat)*len(SCHEMA)/el:7.0f} decisions/s   latency p50 {np.median(lat):6.1f} ms  p90 {np.percentile(lat,90):6.1f}  p99 {np.percentile(lat,99):6.1f}", flush=True)
            print(json.dumps((await cl.get(URL + "/stats")).json())[:300])

    asyncio.run(main())


if __name__ == "__main__":
    _cli()
