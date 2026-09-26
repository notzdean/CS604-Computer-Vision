"""Build the model: pre-trained ViT-B/16 -> (optional head init) -> method (full_ft / lora / ...)."""
import timm

BASE_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def build_model(cfg, init_head=None):
    """init_head(model) runs BEFORE the method wraps the model, so PEFT wrappers copy the initialised head."""
    from methods import REGISTRY
    model = timm.create_model(cfg.get("base_model", BASE_MODEL), pretrained=cfg.get("pretrained", True),
                              num_classes=cfg["num_classes"])
    if init_head is not None:
        init_head(model)
    return REGISTRY[cfg["method"]](model, cfg.get("method_args") or {})
