"""GPU check that EngineV2 answers what the eager masked forward answers (and what decider.engine.Engine answers).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m decider.bench.verify_engine_v2 --model <path> --data rows.jsonl.gz --rows 200

Reference: DecisionModel.slot_logits with a real attention mask and no padding trickery (the path decider.infer.Decider
takes with use_graphs=False).  Compared against EngineV2.score_items (bucketed graph replay), EngineV2.score_shared
(one state pass, cache forked per question) and, when --engine1 is given, decider.engine.Engine.score_items.
Exits 1 on any argmax mismatch, any non-finite probability, or a max |dp| above --tol (default 0.05, the bf16 batch-shape
floor measured in docs/SERVING.md is about 0.03).
"""
import argparse, gzip, json, sys, time
import torch

from decider import systemone as S1
from decider.prompt_fast import build_rows
from decider.model import collate


def suite_rows(path, n, tok, max_ctx_tokens):
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            if len(out) >= n:
                break
            r = json.loads(line)
            try:
                ctx = S1.render_state(r["state"])
                rqs = {k: S1.render_question(v) for k, v in r["questions"].items()}
                flat, _ = S1.plan_rows(rqs, False)
            except ValueError:
                continue
            rows = [[(x["question"], list(x["options"]))] for x in flat]
            items, _ = build_rows(tok, ctx, rows, max_ctx_tokens=max_ctx_tokens)
            out.append((r["id"], items))
    return out


@torch.no_grad()
def eager_ref(m, items, temperature, dev, max_tokens=65536):
    """The masked eager forward, one chunk at a time."""
    out = []
    per = max(1, max_tokens // max(len(it["ids"]) for it in items))
    for i in range(0, len(items), per):
        chunk = items[i:i + per]
        b = collate(chunk, m.tok.pad_token_id)
        lg = m.slot_logits(*[b[k].to(dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
        p = torch.softmax(lg / temperature, -1).cpu(); c = 0
        for it in chunk:
            out.append(p[c:c + len(it["slots"])]); c += len(it["slots"])
    return out


def report(name, ref, got):
    """-> (argmax mismatches, max |dp|, non-finite rows).  A NaN anywhere counts as a failure, never as a small difference."""
    dmax = 0.0; mism = 0; n = 0; bad = 0
    assert len(ref) == len(got), (name, len(ref), len(got))
    for a, b in zip(ref, got):
        assert a.shape[0] == b.shape[0], (name, a.shape, b.shape)
        a = a[:, :b.shape[1]]
        if not (bool(a.isfinite().all()) and bool(b.isfinite().all())):
            bad += a.shape[0]; n += a.shape[0]; continue
        dmax = max(dmax, float((a - b).abs().max()))
        mism += int((a.argmax(-1) != b.argmax(-1)).sum()); n += a.shape[0]
    print(f"{name:34s} answers {n:6d}  argmax mismatches {mism:5d}  non-finite rows {bad:4d}  max |dp| {dmax:.2e}")
    return mism, dmax, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help=".jsonl(.gz) of {id, state, questions} rows")
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-state-tokens", type=int, default=32768)
    ap.add_argument("--engine1", action="store_true", help="also compare decider.engine.Engine (compile off, fp8 off)")
    ap.add_argument("--fp8", action="store_true")
    a = ap.parse_args()

    from decider.engine_v2 import EngineV2
    e2 = EngineV2(a.model, compile=False, fp8=a.fp8, max_ctx_tokens=a.max_state_tokens)
    t = e2.warmup(log=lambda s: print(s, flush=True)); e2.seal()
    print(f"warm-up: {len(e2.graphs)} graphs in {t:.0f}s, {torch.cuda.memory_reserved()/1e9:.1f} GB reserved", flush=True)

    data = suite_rows(a.data, a.rows, e2.tok, a.max_state_tokens)
    print(f"{len(data)} suite rows, {sum(len(x) for _, x in data)} scoring rows", flush=True)

    ref, v2, sh = [], [], []
    t0 = time.time()
    for _, items in data:
        ref += eager_ref(e2.m, items, a.temperature, e2.dev)
    print(f"eager reference {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    for _, items in data:
        v2 += e2.score_items(items, temperature=a.temperature)
    print(f"engine_v2 score_items {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    for _, items in data:
        sh += (e2.score_shared(items, temperature=a.temperature) if len(items) > 1 else e2.score_items(items, temperature=a.temperature))
    print(f"engine_v2 score_shared {time.time()-t0:.0f}s", flush=True)

    problems = 0
    for name, got in (("EngineV2.score_items vs eager", v2), ("EngineV2.score_shared vs eager", sh)):
        m, d, nf = report(name, ref, got); problems += m + nf + (d > a.tol)
    report("score_shared vs score_items", v2, sh)
    print("engine stats", e2.stats)

    if a.engine1:
        del e2
        torch.cuda.empty_cache()
        from decider.engine import Engine
        e1 = Engine(a.model, compile=False, fp8=a.fp8, conv_patch=False, max_ctx_tokens=a.max_state_tokens)
        v1 = []
        for _, items in data:
            v1 += e1.score_items(items, temperature=a.temperature)
        m, d, nf = report("Engine.score_items vs eager", ref, v1); problems += m + nf + (d > a.tol)
    print("VERDICT", "ok" if problems == 0 else f"{problems} problems")
    sys.exit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
