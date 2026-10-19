"""Jev-style image assets for the System One servers (decider.serve, decider.serve_vllm).

A request may carry `assets`: {reference: {"type": "image", "format": "png" | "jpg" | "jpeg", "data_base64": ...}}.  The state names
an image by its reference string, anywhere in its text (e.g. {"photo": "img1"}); references in the questions are not supported, and an
asset the state never names is an error.  Each server decides what a reference becomes in the prompt (a placeholder token, or text).
"""
import base64
import binascii
import io
import json

FORMATS = ("png", "jpg", "jpeg")


def decode_assets(assets, max_images):
    """{ref: asset} -> {ref: RGB PIL image}; ValueError (the servers' 422) on anything malformed."""
    from PIL import Image
    if not isinstance(assets, dict) or not assets:
        raise ValueError("assets: a non-empty map of reference -> image")
    if len(assets) > max_images:
        raise ValueError(f"too many assets: {len(assets)}, the limit is {max_images} (DECIDER_MAX_IMAGES)")
    out = {}
    for ref, a in assets.items():
        if not isinstance(ref, str) or not ref or not isinstance(a, dict):
            raise ValueError("assets: each entry is reference -> {type, format, data_base64}")
        if a.get("type", "image") != "image" or str(a.get("format", "")).lower() not in FORMATS:
            raise ValueError(f"asset {ref!r}: type must be image and format png, jpg or jpeg")
        try:
            im = Image.open(io.BytesIO(base64.b64decode(a.get("data_base64") or "", validate=True)))
            im.load()
        except (binascii.Error, ValueError, OSError) as e:
            raise ValueError(f"asset {ref!r}: data_base64 is not a readable image ({e})")
        out[ref] = im.convert("RGB")
    return out


def place_images(state_text, questions, images, max_images, placeholder):
    """Replace every occurrence of an asset reference in the rendered state by placeholder(k, n) (k-th reference, n references in
    all; the longest reference wins where references overlap).  -> (state text, images in reference order)."""
    qtext = json.dumps(questions, ensure_ascii=False)
    refs = sorted(images, key=len, reverse=True)
    hits, used, i = [], set(), 0
    while i < len(state_text):
        r = next((r for r in refs if state_text.startswith(r, i)), None)
        if r is None:
            i += 1; continue
        hits.append((i, r)); used.add(r); i += len(r)
    for r in images:
        if r not in used:
            if r in qtext:
                raise ValueError(f"asset {r!r}: images can be referenced from the state only")
            raise ValueError(f"asset {r!r} is not referenced in the state")
    if len(hits) > max_images:
        raise ValueError(f"too many image references: {len(hits)}, the limit is {max_images} (DECIDER_MAX_IMAGES)")
    parts, last = [], 0
    for k, (pos, r) in enumerate(hits):
        parts += [state_text[last:pos], placeholder(k, len(hits))]; last = pos + len(r)
    parts.append(state_text[last:])
    return "".join(parts), [images[r] for _, r in hits]
