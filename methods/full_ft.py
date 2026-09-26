def apply(model, args):
    """Full fine-tuning: every parameter is trained."""
    model.requires_grad_(True)
    return model
