"""One file per adaptation method. Each exposes apply(model, args) -> model.
Contract: the classifier head (model.head) must stay trainable; everything else is up to the method.
To add a method: create methods/<name>.py and register it below."""
from methods import adaptformer, bitfit, full_ft, lora, lora_adaptformer

REGISTRY = {
    "full_ft": full_ft.apply,
    "lora": lora.apply,
    "bitfit": bitfit.apply,
    "adaptformer": adaptformer.apply,
    "lora_adaptformer": lora_adaptformer.apply,
}
