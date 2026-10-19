"""decider.images: Jev-style asset decoding and state references (no model, no GPU)."""
import base64
import io

import pytest
from PIL import Image

from decider.images import decode_assets, place_images


def _png(color=(10, 20, 30)):
    b = io.BytesIO(); Image.new("RGB", (8, 6), color).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


def test_decode_ok():
    out = decode_assets({"img1": {"type": "image", "format": "png", "data_base64": _png()}}, 4)
    assert list(out) == ["img1"] and out["img1"].size == (8, 6) and out["img1"].mode == "RGB"


@pytest.mark.parametrize("assets,msg", [
    ({}, "non-empty"),
    ({"a": {"format": "gif", "data_base64": _png()}}, "format"),
    ({"a": {"format": "png", "data_base64": "not base64!"}}, "not a readable image"),
    ({"a": {"format": "png", "data_base64": base64.b64encode(b"text").decode()}}, "not a readable image"),
])
def test_decode_errors(assets, msg):
    with pytest.raises(ValueError, match=msg):
        decode_assets(assets, 4)


def test_decode_limit():
    with pytest.raises(ValueError, match="too many assets"):
        decode_assets({str(i): {"format": "png", "data_base64": _png()} for i in range(3)}, 2)


def test_place_order_and_text():
    ims = {"a": "A", "photo-b": "B"}
    text, order = place_images('{"x": "photo-b", "y": "a + a"}', {}, ims, 4, lambda k, n: f"<{k}/{n}>")
    assert text == '{"x": "<0/3>", "y": "<1/3> + <2/3>"}' and order == ["B", "A", "A"]


def test_place_longest_reference_wins():
    text, order = place_images("img10 img1", {}, {"img1": 1, "img10": 10}, 4, lambda k, n: "#")
    assert text == "# #" and order == [10, 1]


def test_place_errors():
    with pytest.raises(ValueError, match="state only"):
        place_images("nothing here", {"q": {"instructions": "look at pic"}}, {"pic": 1}, 4, lambda k, n: "#")
    with pytest.raises(ValueError, match="not referenced"):
        place_images("nothing here", {}, {"pic": 1}, 4, lambda k, n: "#")
    with pytest.raises(ValueError, match="too many image references"):
        place_images("p p p", {}, {"p": 1}, 2, lambda k, n: "#")
