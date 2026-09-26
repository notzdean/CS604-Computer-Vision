from peft import LoraConfig, get_peft_model


def apply(model, args):
    """LoRA on the attention qkv projection (timm ViT has one fused Linear called 'qkv').
    args: r, alpha, dropout, target_modules. The head is fully trained (modules_to_save)."""
    r = args.get("r", 16)
    cfg = LoraConfig(r=r, lora_alpha=args.get("alpha", 2 * r), lora_dropout=args.get("dropout", 0.0),
                     target_modules=args.get("target_modules", ["qkv"]), modules_to_save=["head"], bias="none")
    return get_peft_model(model, cfg)
