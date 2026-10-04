import torch.nn as nn


class ParallelAdapterMLP(nn.Module):
    """AdaptFormer: a bottleneck adapter in parallel with the frozen FFN of a ViT block.
    out = FFN(x) + scale * up(ReLU(down(x)))   where x is the FFN input (after norm2).
    up is zero-initialised, so the adapted model starts identical to the pre-trained one."""

    def __init__(self, mlp, dim, bottleneck, scale):
        super().__init__()
        self.mlp = mlp
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        self.act = nn.ReLU()
        self.scale = scale
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.mlp(x) + self.scale * self.up(self.act(self.down(x)))


def apply(model, args):
    """AdaptFormer on every transformer block's FFN. Only the adapters and the head are trained.
    args: bottleneck (default 64), scale (default 0.1)."""
    bottleneck = args.get("bottleneck", 64)
    scale = args.get("scale", 0.1)
    dim = model.embed_dim
    for blk in model.blocks:
        blk.mlp = ParallelAdapterMLP(blk.mlp, dim, bottleneck, scale)
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if ".down." in name or ".up." in name or name.startswith("head"):
            p.requires_grad = True
    return model
