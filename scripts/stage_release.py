"""Assemble a Hugging Face release folder: weights + tokenizer + decider_config.json + the inference subset of the package + the card.
   python scripts/stage_release.py runs/decider_full/model runs/release/decider-2b [eval_results.json]"""
import os, shutil, sys
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(f"{dst}/decider", exist_ok=True)
for f in os.listdir(src):
    if os.path.isfile(f"{src}/{f}"): shutil.copy2(f"{src}/{f}", f"{dst}/{f}")
for m in (
    "__init__", "prompt", "model", "systemone", "infer", "batching", "prompt_fast",
    "engine", "engine_v2", "shared_prefix", "schema_engine", "fp8", "serve", "metrics", "mps_ops", "mps_moe", "temperature",
    "calibrate",
):  # all internal modules needed to run and serve, and calibrate (fits temperature_by_type); no training code
    shutil.copy2(f"decider/{m}.py", f"{dst}/decider/{m}.py")
shutil.copy2("MODEL_CARD.md", f"{dst}/README.md")
if len(sys.argv) > 3: shutil.copy2(sys.argv[3], f"{dst}/eval_results.json")
assert os.path.exists(f"{dst}/decider_config.json"), "write decider_config.json (temperature, version, schema_first, isolated_levels) into the model folder first"
print("staged", dst, sorted(os.listdir(dst)))
