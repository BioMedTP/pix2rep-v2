import torch
import torch.nn as nn


def get_norm_layer(norm_type: str, num_channels: int, ndim: int = 2) -> nn.Module:
    if norm_type == "batch":
        if ndim == 3:
            return nn.BatchNorm3d(num_channels)
        else:
            return nn.BatchNorm2d(num_channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=32, num_channels=num_channels)
    elif norm_type == "instance":
        if ndim == 3:
            return nn.InstanceNorm3d(num_channels, affine=True)
        else:
            return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# Final convolution layer for segmentation
class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
