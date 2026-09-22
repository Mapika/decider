"""Start a /v1/systemone server, replay a row file against it at several concurrencies, stop it.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m decider.bench.run_serving_matrix \
        --model /path/to/model --data sample.jsonl.gz --out-dir runs/serving --rows 1000 \
        --case old:decider.serve_v1: \
        --case new:decider.serve:

Each case is `label:module:VAR=VAL,VAR=VAL` (an empty VAR list means the module's defaults).  The server is started as its
own process, polled on /health, replayed at --conc (default 1,8), then terminated by its pid (SIGTERM, then SIGKILL).
Per-case responses land in <out-dir>/<label>_c<conc>.jsonl and the summaries (latency, errors, start-up time, /stats) in
<out-dir>/summary.json.
"""
import argparse, json, os, subprocess, sys, time
import urllib.request

from decider.bench.replay_systemone import main as replay_main


def wait_health(url, timeout, proc):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} after {time.time()-t0:.0f}s")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if json.load(r).get("ok"):
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"server not healthy after {timeout}s")


def get_json(url, path):
    try:
        with urllib.request.urlopen(url + path, timeout=30) as r:
            return json.load(r)
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--conc", default="1,8")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--startup-timeout", type=float, default=5400)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    url = f"http://127.0.0.1:{a.port}"
    concs = [int(x) for x in a.conc.split(",")]
    summaries = []
    for spec in a.case:
        label, module, envs = spec.split(":", 2)
        env = dict(os.environ, DECIDER_MODEL=a.model)
        for kv in [x for x in envs.split(",") if x]:
            k, v = kv.split("=", 1); env[k] = v
        log = open(os.path.join(a.out_dir, f"{label}_server.log"), "w")
        cmd = [sys.executable, "-m", "uvicorn", module + ":app", "--host", "127.0.0.1", "--port", str(a.port),
               "--log-level", "warning"]
        print(f"\n=== case {label}: {module} {envs or '(defaults)'}", flush=True)
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            boot = wait_health(url, a.startup_timeout, proc)
            print(f"[{label}] healthy after {boot:.0f}s (pid {proc.pid})", flush=True)
            for conc in concs:
                out = os.path.join(a.out_dir, f"{label}_c{conc}.jsonl")
                s = replay_main(["run", "--url", url, "--data", a.data, "--rows", str(a.rows), "--conc", str(conc),
                                 "--out", out, "--label", f"{label}_c{conc}"])
                s = dict(s or {}, case=label, module=module, env=envs, startup_s=round(boot, 1))
                s["server_stats"] = get_json(url, "/stats")
                summaries.append(s)
                path = os.path.join(a.out_dir, "summary.json")
                keep = []                                        # merge with earlier runs into the same directory
                if os.path.exists(path):
                    try: keep = [x for x in json.load(open(path)) if (x.get("case"), x.get("conc")) not in
                                 {(y["case"], y["conc"]) for y in summaries}]
                    except Exception: keep = []
                json.dump(keep + summaries, open(path, "w"), indent=1)
        finally:
            proc.terminate()                                      # the server is one process; signal it by pid only
            try:
                proc.wait(timeout=120)
            except Exception:
                proc.kill(); proc.wait(timeout=60)
            log.close()
            time.sleep(10)
    print("\n" + json.dumps(summaries, indent=1)[:4000])
    print("\nwrote", os.path.join(a.out_dir, "summary.json"))


if __name__ == "__main__":
    main()
