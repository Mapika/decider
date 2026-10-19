"""Image rows for decider.serve (DECIDER_IMAGES=1), for Qwen3.5 deciders released as a text model plus a `vision/` folder.

The repository root holds the text model (Qwen3_5ForCausalLM) that decider.serve loads as always; `vision/` holds the image-text
config (Qwen3_5ForConditionalGeneration), the processor config and the vision tower weights only (visual.safetensors, keys
"model.visual.*").  At start-up the image-text model is built on the meta device, its language model is replaced by the language
model the engine has already loaded (the same modules: no second copy, and FP8 linears when DECIDER_FP8=1); only the vision
tower is built and loaded.

Layout (as trained, decider2/family_qwen/vl_common.py): the row is rendered exactly as a text row, each asset reference in the state
reads "the image" ("image k" with several), and one <|vision_start|> <|image_pad|> x n <|vision_end|> block per image, in reference
order, goes in front of the whole prompt; n is the image processor's grid (at most 768 x 768 pixels) / merge_size^2.  An image row is
one eager forward of the image-text model; its letter logits are read at the slot as for text rows (same letter rows of the LM head,
same softcap, same temperatures).
"""
import os

import torch
import torch.nn.functional as F

MAX_PIXELS = 768 * 768


def reference_text(k, n):
    return "the image" if n == 1 else f"image {k + 1}"


def vision_dir(model):
    if os.path.isdir(model):
        d = os.path.join(model, "vision")
    else:                                            # a Hub repository id
        from huggingface_hub import snapshot_download
        d = os.path.join(snapshot_download(model, allow_patterns=["vision/*"]), "vision")
    if not os.path.isfile(os.path.join(d, "config.json")):
        raise ValueError(f"DECIDER_IMAGES=1 needs a vision/ folder next to the model ({d} has no config.json)")
    return d


class VisionReader:
    def __init__(self, eng, model, dtype=torch.bfloat16):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        d = vision_dir(model)
        cfg = AutoConfig.from_pretrained(d)
        with torch.device("meta"):
            vlm = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
        lm = vlm.model.language_model
        if type(lm) is not type(eng.m.lm.model):
            raise ValueError(f"vision/config.json describes a {type(lm).__name__} language model, the root model is a "
                             f"{type(eng.m.lm.model).__name__}")
        vlm.model.language_model = eng.m.lm.model    # shared: the engine's modules, not a copy
        vlm.lm_head = None
        sd = {k[len("model.visual."):]: v for k, v in load_file(os.path.join(d, "visual.safetensors")).items()}
        vis = type(vlm.model.visual)._from_config(cfg.vision_config, dtype=dtype)   # small: built for real (rotary buffers)
        missing, unexpected = vis.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise ValueError(f"vision/visual.safetensors does not match vision/config.json: missing {missing[:4]}, unexpected {unexpected[:4]}")
        vlm.model.visual = vis.to(eng.dev).eval()
        self.vlm, self.eng = vlm.eval(), eng
        self.ip = AutoProcessor.from_pretrained(d).image_processor
        tok = eng.tok
        self.vs, self.pad, self.ve = (tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"))
        self.merge = self.ip.merge_size
        self.bytes = sum(p.numel() * p.element_size() for p in vis.parameters())

    def image_inputs(self, images):
        return self.ip(images=images, return_tensors="pt", max_pixels=MAX_PIXELS)

    def add_images(self, items, images):
        """Put the image blocks in front of every row (in place); returns the processor inputs shared by the rows."""
        ii = self.image_inputs(images)
        block = []
        for g in ii["image_grid_thw"]:
            block += [self.vs] + [self.pad] * (int(g.prod()) // self.merge ** 2) + [self.ve]
        for it in items:
            it["ids"] = block + list(it["ids"]); it["slots"] = [s + len(block) for s in it["slots"]]
        return ii

    @torch.no_grad()
    def score(self, items, ii, temperature):
        """One eager forward per row (all rows carry the same images); -> one [n_q, MAX_OPTIONS] probability tensor per row."""
        from decider.engine import read_slots
        from decider.model import cap_logits
        from decider.temperature import item_slice, slot_temperatures
        dev = self.eng.dev
        pv = ii["pixel_values"].to(dev, torch.bfloat16); grid = ii["image_grid_thw"].to(dev)
        out = []
        for i, it in enumerate(items):
            ids = torch.tensor([it["ids"]], dtype=torch.long, device=dev)
            h = self.vlm.model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=pv, image_grid_thw=grid,
                               mm_token_type_ids=(ids == self.pad).long(), use_cache=False).last_hidden_state
            lg = cap_logits(self.eng.m, F.linear(h, self.eng.W).float())
            out += read_slots(lg, [0] * len(it["slots"]), list(it["slots"]), list(it["nopts"]),
                              slot_temperatures(item_slice(temperature, i, i + 1), [it]), [len(it["slots"])])
        return out
