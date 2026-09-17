"""Put a fine-tuned text-only checkpoint's language weights into the VLM (vision tower from the base).
   python -m decider.vision.transplant runs/r7_v4/model runs/r7_v4/vlm [Qwen/Qwen3.5-2B-Base]"""
import sys, torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoProcessor
src, dst = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3.5-2B-Base"
vlm = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
txt = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
tsd = txt.state_dict(); vsd = vlm.state_dict(); n = 0; missing = []
for k, v in tsd.items():
    kk = k if k in vsd else k.replace("model.", "model.language_model.", 1)
    if kk in vsd and vsd[kk].shape == v.shape:
        vsd[kk].copy_(v); n += 1
    else:
        missing.append(k)
print(f"copied {n}/{len(tsd)} tensors; unmatched: {missing[:5]}")
vlm.load_state_dict(vsd); vlm.save_pretrained(dst); AutoProcessor.from_pretrained(base).save_pretrained(dst)
print("saved", dst)
