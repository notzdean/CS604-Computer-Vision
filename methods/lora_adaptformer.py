import torch.nn as nn

from methods.adaptformer import ParallelAdapterMLP


class LoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank update: out = W x + (alpha / r) * B(A(x)). B starts at zero."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def apply(model, args):
    """LoRA on each block's attention qkv AND an AdaptFormer adapter on each block's FFN, in one model.
    args: lora_r (default 12), lora_alpha (default 2 * lora_r), bottleneck (default 208), scale (default 0.1).
    Only the LoRA matrices, the adapters and the head are trained."""
    r = args.get("lora_r", 12)
    alpha = args.get("lora_alpha", 2 * r)
    bottleneck = args.get("bottleneck", 208)
    scale = args.get("scale", 0.1)
    dim = model.embed_dim
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r, alpha)
        blk.mlp = ParallelAdapterMLP(blk.mlp, dim, bottleneck, scale)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_" in name or ".down." in name or ".up." in name or name.startswith("head"):
            p.requires_grad = True
    return model
