"""Vision variant: image(s) + the same lettered prompt, letter logits at the answer slots, one forward pass."""
import io, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .prompt import build, letter_ids, MAX_OPTIONS

IMG = "<|vision_start|><|image_pad|><|vision_end|>"


def to_pil(x):
    if isinstance(x, Image.Image): return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)): return Image.open(io.BytesIO(x)).convert("RGB")
    return Image.fromarray(x).convert("RGB")


class VisionDecisionModel(nn.Module):
    def __init__(self, name, dtype=torch.bfloat16, grad_ckpt=True):
        super().__init__()
        self.proc = AutoProcessor.from_pretrained(name); self.tok = self.proc.tokenizer
        self.lm = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
        if grad_ckpt: self.lm.gradient_checkpointing_enable()
        self.register_buffer("letters", torch.tensor(letter_ids(self.tok)), persistent=False)
        self.slot_tok = self.tok.encode(" (", add_special_tokens=False)[-1]; self.colon = self.tok.encode(":", add_special_tokens=False)[-1]
        self.answer_tok = self.tok.encode("Answer", add_special_tokens=False)[0]

    def prepare(self, examples, max_ctx_tokens=1536):
        """examples: list of (image or None, Example). Returns processor inputs + slot bookkeeping."""
        texts, images, golds, nopts, nq = [], [], [], [], []
        for img, ex in examples:
            b = build(ex, self.tok, _NoShuffle(), max_ctx_tokens=max_ctx_tokens)
            txt = self.tok.decode(b["ids"])
            texts.append((IMG if img is not None else "") + txt); golds.extend(b["golds"]); nopts.extend(b["nopts"]); nq.append(len(b["slots"]))
            if img is not None: images.append(to_pil(img))
        self.tok.padding_side = "right"
        inp = self.proc(images=images or None, text=texts, return_tensors="pt", padding=True)
        ids = inp["input_ids"]
        slot_idx, slot_batch = [], []
        for bi in range(ids.shape[0]):
            row = ids[bi].tolist(); found = []
            for i in range(2, len(row)):
                if row[i] == self.slot_tok and row[i - 1] == self.colon and self.answer_tok in row[max(0, i - 5):i]:
                    found.append(i)
            assert len(found) == nq[bi], (len(found), nq[bi], self.tok.decode(row[-40:]))
            slot_idx.extend(found); slot_batch.extend([bi] * len(found))
        inp["slot_idx"] = torch.tensor(slot_idx); inp["slot_batch"] = torch.tensor(slot_batch)
        inp["golds"] = torch.tensor(golds); inp["nopts"] = torch.tensor(nopts)
        return inp

    def slot_logits(self, inp):
        dev = self.letters.device
        kw = {k: v.to(dev) for k, v in inp.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
        h = self.lm.model(**kw, use_cache=False).last_hidden_state                        # [B, T, H]
        hs = h[inp["slot_batch"].to(dev), inp["slot_idx"].to(dev)]                         # [N, H]
        lg = F.linear(hs, self.lm.lm_head.weight[self.letters]).float()                    # [N, K] letter logits only
        ar = torch.arange(MAX_OPTIONS, device=dev)[None, :]
        return lg.masked_fill(ar >= inp["nopts"].to(dev)[:, None], float("-inf"))


class _NoShuffle:
    def shuffle(self, x): pass
    def sample(self, xs, k): return xs[:k]


if __name__ == "__main__":
    import time, numpy as np
    from .infer import Example, Q
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    m = VisionDecisionModel("Qwen/Qwen3.5-2B-Base", grad_ckpt=False).cuda().eval()
    env = JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-1-1-v0"), SIMPLE_MOVEMENT); obs = env.reset()
    for _ in range(100): obs, _, _, _ = env.step(3)
    ex = Example("The image shows the current Super Mario Bros screen. Mario runs right; jump over enemies and gaps.",
                 [Q("What should Mario do right now?", ["run right", "jump right", "step left", "wait"], 0), Q("Is an enemy visible?", ["no", "yes"], 1)])
    inp = m.prepare([(obs, ex), (None, ex)])
    with torch.no_grad(): lg = m.slot_logits(inp)
    print("slots found:", inp["slot_idx"].tolist(), "batch", inp["slot_batch"].tolist(), "tokens", inp["input_ids"].shape)
    print("probs (image / no image):", [[round(x, 3) for x in torch.softmax(r, -1)[:4].tolist()] for r in lg])
