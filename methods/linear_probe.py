def apply(model, args):
    """Linear probing: the pre-trained backbone is frozen and only the classifier head is trained (0% of backbone)."""
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    return model
