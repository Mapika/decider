"""Synthetic situation->action data from a local teacher (Qwen3.5-27B, bf16, HF generate, batched).
Each item: domain, situation text, question, 3-6 options, correct index, short reason, risk flag.
   python -m decider.data.teacher_situations data/synth.jsonl --n 3000 [--model Qwen/Qwen3.5-27B]"""
import argparse, json, random, re, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DOMAINS = ["a 2D platformer game (gaps, enemies, pipes, power-ups)", "a top-down grid maze with keys, doors and lava", "a text adventure (rooms, items, monsters)",
           "an arcade paddle game (ball, paddle, bricks)", "a road-crossing game with moving cars", "a turn-based tactics game (units, terrain, cover)",
           "a card game against a dealer", "a racing game (corners, opponents, fuel)", "a tower-defense game (waves, towers, gold)", "a survival crafting game (hunger, tools, night)",
           "a household robot doing kitchen tasks", "a warehouse robot moving packages", "driving a car in city traffic", "flying a drone delivery route",
           "an on-call engineer handling a production incident", "a customer-support agent with an open ticket", "a security analyst triaging an alert",
           "a hospital triage nurse", "a trader managing a small portfolio", "a farmer planning the week", "a hiker in changing mountain weather",
           "a chess-like board position (simplified rules given)", "a spaceship captain managing power and shields", "a chef running a busy kitchen line",
           "a firefighter at a house fire", "a lifeguard at a crowded beach", "a shop manager with low stock", "a teacher with a disrupted classroom",
           "a football coach in the final minutes", "a scuba diver with a low tank", "a submarine game (depth, sonar, mines)", "a stealth game (guards, cameras, noise)"]

SYS = ("You write training data for a small model that must pick the best action in a situation. Output strict JSON only.")
PROMPT = ("Domain: {domain}.\nWrite ONE realistic situation for an agent in this domain, in the third person or second person, 2-5 sentences, as a plain state description "
          "(positions, distances, resources, threats, timers) without hints about the right answer. Then a question asking what to do right now, "
          "{k} candidate actions (short phrases, distinct, exactly one clearly best given the situation and common sense; the others plausible but worse or dangerous), "
          "the index of the best action, a one-sentence reason, and whether the agent is in immediate danger (true/false).\n"
          "Vary the scenario: {seed_hint}.\nReturn JSON with keys: situation, question, options (list), answer (int), reason, danger (bool).")
HINTS = ["threat approaching", "resource running low", "a shortcut with risk", "two goals conflicting", "waiting is the right call", "retreat is the right call",
         "a timer about to expire", "an ally needs help", "misleading but harmless obstacle", "the obvious move is a trap", "everything is fine, keep going",
         "must use a tool first", "an unusual rule of this world applies", "information is incomplete", "a chain of two steps is needed"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_new", type=int, default=400); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    rng = random.Random(a.seed); done = 0; t0 = time.time(); bad = 0
    with open(a.out, "a") as f:
        while done < a.n:
            reqs = []
            for _ in range(a.bs):
                d = rng.choice(DOMAINS); k = rng.choice([3, 4, 4, 5, 6]); hint = rng.choice(HINTS)
                msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": PROMPT.format(domain=d, k=k, seed_hint=hint)}]
                reqs.append((d, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)))
            enc = tok([p for _, p in reqs], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = m.generate(**enc, max_new_tokens=a.max_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for (d, _), t in zip(reqs, texts):
                mm = re.search(r"\{.*\}", t, re.S)
                try:
                    j = json.loads(mm.group(0)); opts = j["options"]; ans = int(j["answer"])
                    assert isinstance(opts, list) and 3 <= len(opts) <= 6 and 0 <= ans < len(opts) and j["situation"] and j["question"]
                    j["domain"] = d; f.write(json.dumps(j, ensure_ascii=False) + "\n"); done += 1
                except Exception:
                    bad += 1
            f.flush()
            print(f"[synth] {done}/{a.n} ok, {bad} rejected, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
