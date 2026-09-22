"""Probe: for which split points does a cached two-pass forward disagree with a single prefill?

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m decider.bench.probe_cache_split --model <path> --lengths 4300,6000

`Engine.score_shared`, `EngineV2.score_shared` and `SchemaEngine` all run a prefix once with `use_cache=True` and then
continue with a suffix.  That is only sound if `core(ids[:L], use_cache=True)` followed by `core(ids[L:],
past_key_values=...)` equals `core(ids)`.  With the cuDNN SDPA backend on (torch 2.14 / CUDA 13 on Blackwell) it does not,
for some (prefix, suffix) pairs: the error appears at the first `full_attention` block and is the size of the signal, or
NaN.  The engines turn that backend off (decider.engine.set_attention_backend_policy), and with it off the probe finds no
failure.  `--cudnn` re-enables the backend to reproduce the fault.  A non-finite hidden state counts as a failure.

Also checks that the single prefill is causal and invariant to right padding, which the bucketed graph path relies on.
Exits 1 when any check fails.
"""
import argparse, sys, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="1200,2500,4300,6000")
    ap.add_argument("--step", type=int, default=128)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cudnn", action="store_true", help="re-enable the cuDNN SDPA backend the engine turns off")
    a = ap.parse_args()

    from decider.engine_v2 import EngineV2
    e = EngineV2(a.model, compile=False, fp8=False, use_graphs=False)
    if a.cudnn:
        torch.backends.cuda.enable_cudnn_sdp(True)
    print("cudnn_sdp_enabled", torch.backends.cuda.cudnn_sdp_enabled(), flush=True)

    def rel(ref, h):
        r = float((ref - h).abs().max() / ref.abs().max())
        return r if r == r else float("inf")          # NaN -> inf, so it is reported as a failure
    g = torch.Generator().manual_seed(a.seed)
    bad_total = 0
    with torch.no_grad():
        for L in [int(x) for x in a.lengths.split(",")]:
            ids = torch.randint(1000, 200000, (L,), generator=g).tolist()
            t = torch.tensor(ids, device=e.dev)[None]
            full = e.core(input_ids=t, use_cache=False).last_hidden_state[0]

            # invariant 1: a truncated prefill agrees with the full prefill at the truncation point
            caus = []
            for P in range(a.step, L, a.step * 4):
                h = e.core(input_ids=t[:, :P], use_cache=False).last_hidden_state[0, -1]
                r = rel(full[P - 1], h)
                if r > a.tol: caus.append((P, round(r, 3)))
            # invariant 2: right padding to a longer shape does not change the last real position
            pad = []
            for T in (L + 64, ((L // 1024) + 2) * 1024):
                p = torch.full((1, T), e.tok.pad_token_id, dtype=torch.long, device=e.dev); p[0, :L] = t[0]
                h = e.core(input_ids=p, use_cache=False).last_hidden_state[0, L - 1]
                r = rel(full[L - 1], h)
                if r > a.tol: pad.append((T, round(r, 3)))
            # the cached two-pass forward
            split = []
            for P in range(a.step, L - 8, a.step):
                c = e.core(input_ids=t[:, :P], use_cache=True).past_key_values
                h = e.core(input_ids=t[:, P:], past_key_values=c, use_cache=True).last_hidden_state[0, -1]
                r = rel(full[-1], h)
                if r > a.tol: split.append((P, L - P, round(r, 3)))
            bad_total += len(split) + len(caus) + len(pad)
            print(f"L={L}: prefill-causality failures {caus or 'none'}; right-padding failures {pad or 'none'}; "
                  f"cached-split failures (prefix, suffix, rel err) {split or 'none'}", flush=True)
    print("VERDICT", f"{bad_total} failures" if bad_total else "no failure found")
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
