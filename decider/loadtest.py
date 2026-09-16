"""Closed-loop load test: N concurrent clients hitting POST /decide.  python -m decider.loadtest [url] [conc,...]"""
import asyncio, json, random, sys, time, httpx, numpy as np
from . import data as D
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
CONCS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,4,16,64").split(",")]
_, evals = D.load_cache("data/tasks.pkl")
SCHEMA = {"Revenue currently impacted?": {"type": "bool"}, "What business impact?": {"type": "choice", "options": ["none", "degraded", "outage"]},
          "Which queue?": {"type": "choice", "options": ["billing", "technical", "sales", "hr", "none of the above"]},
          "Priority level?": {"type": "scale", "legend": {"0": "low", "1": "medium", "2": "high"}}, "Human attention needed?": {"type": "bool"}}
ctxs = [e.context for e in evals["support_tickets"][:400]]


async def client(cl, n, lat, dur):
    t_end = time.monotonic() + dur
    while time.monotonic() < t_end:
        t = time.monotonic()
        r = await cl.post(URL + "/decide", json={"context": random.choice(ctxs), "schema": SCHEMA}, timeout=60)
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
