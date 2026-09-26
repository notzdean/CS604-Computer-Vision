def apply(model, args):
    """BitFit: train only bias parameters (+ the new classifier head)."""
    for name, p in model.named_parameters():
        p.requires_grad = name.endswith(".bias") or name.startswith("head")
    return model
