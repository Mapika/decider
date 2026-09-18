"""The vision variant: decisions from an image plus the same lettered prompt (decider.vision.model)."""


def __getattr__(name):                      # lazy: importing decider.vision must not pull in torch/transformers
    if name in ("VisionDecisionModel", "to_pil", "IMG"):
        from decider.vision import model as m
        return getattr(m, name)
    raise AttributeError(name)
