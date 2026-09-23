"""Answer a row file with the eager masked forward and write it in replay_systemone's format, so a server run can be
compared against the reference with `replay_systemone compare`.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m decider.bench.eager_reference \\
        --model /path/to/model --data rows.jsonl.gz --rows 1000 --out runs/serving/eager.jsonl

The reference is DecisionModel.slot_logits with a real attention mask (no padding trick, no graphs, no cache), one request
at a time, in chunks of at most --max-tokens padded tokens.  Rows are prepared exactly as decider.serve prepares them
(decider.serve.prepare) with the temperature and isolated_levels of the model's decider_config.json, so the answers are
what a correct server should return.  Requests whose questions fail validation are written with status 422.
"""
import argparse, json, os, sys, time
import torch

from decider.bench.replay_systemone import load_rows
from decider.model import DecisionModel, collate
from decider import systemone as S1
from decider.serve import prepare, load_config


@torch.no_grad()
def masked_probs(m, items, temperature, dev, max_tokens):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-state-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=65536, help="padded tokens per forward")
    a = ap.parse_args()
    from decider.engine import set_attention_backend_policy
    set_attention_backend_policy()
    cfg = load_config(a.model)
    temperature = float(os.environ.get("DECIDER_TEMPERATURE", cfg.get("temperature", 1.0)))
    isolated = bool(cfg.get("isolated_levels", False)); name = "decider-" + str(cfg.get("version", "dev"))
    from decider.prompt import chat_for
    m = DecisionModel(a.model, dtype=torch.bfloat16, grad_ckpt=False).to("cuda").eval()
    chat = chat_for(m.tok, cfg)                     # the model's prompt layout, as decider.serve reads it
    rows = load_rows(a.data, a.rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    t0 = time.time(); n_ans = 0; nonfinite = 0
    with open(a.out, "w") as f:
        f.write(json.dumps({"_summary": dict(label="eager_reference", model=a.model, temperature=temperature, isolated_levels=isolated,
                                             rows=len(rows))}) + "\n")
        for j, r in enumerate(rows):
            rid = r.get("id", j)
            try:
                rqs, index, items, ctx_len = prepare(m.tok, r["state"], r["questions"], True, isolated, a.max_state_tokens, chat=chat)
            except ValueError as e:
                f.write(json.dumps(dict(id=rid, ms=0.0, status=422, kind="error", detail=str(e))) + "\n"); continue
            probs = [p for ps in masked_probs(m, items, temperature, "cuda", a.max_tokens) for p in ps] if items else []
            nonfinite += sum(0 if bool(p.isfinite().all()) else 1 for p in probs); n_ans += len(probs)
            resp = {"model": name, "answers": S1.assemble(rqs, index, [p.tolist() for p in probs]),
                    "usage": {"input_tokens": S1.unique_tokens(items), "output_tokens": 0}}
            f.write(json.dumps(dict(id=rid, ms=0.0, status=200, response=resp)) + "\n")
            if (j + 1) % 100 == 0:
                print(f"[eager_reference] {j + 1}/{len(rows)} requests, {n_ans} answers, {time.time() - t0:.0f}s", flush=True)
    print(f"[eager_reference] done: {len(rows)} requests, {n_ans} answers, {nonfinite} non-finite, {time.time() - t0:.0f}s -> {a.out}", flush=True)
    sys.exit(1 if nonfinite else 0)


if __name__ == "__main__":
    main()
