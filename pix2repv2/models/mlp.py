import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    Flexible projection head whose depth is controlled by the length of `inner_dims`.
    - inner_dims = []  -> identity (no change)
    - inner_dims = [d1] -> in -> d1 -> out  (2 convs, matches previous depth==1)
    - inner_dims = [d1, d2] -> in -> d1 -> d2 -> out (3 convs, matches previous depth==2)
    etc.
    Convs use kernel_size=1 and batchnorm+ReLU after each hidden conv.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        inner_dims: list[int] = None,
        norm_type: str = "batch",
    ):
        super().__init__()
        inner_dims = inner_dims or []
        if len(inner_dims) == 0:
            # keep identity behaviour for zero inner dims
            self.net = nn.Identity()
            return

        layers = []
        current_in_channels = channels_in
        # add hidden conv blocks
        for hidden_channels in inner_dims:
            layers.append(nn.Conv2d(current_in_channels, hidden_channels, 1))
            if norm_type == "batch":
                layers.append(nn.BatchNorm2d(hidden_channels))
            elif norm_type == "group":
                layers.append(nn.GroupNorm(num_groups=32, num_channels=hidden_channels))
            layers.append(nn.ReLU())
            current_in_channels = hidden_channels

        # final conv to channels_out (no BN/ReLU after final)
        layers.append(nn.Conv2d(current_in_channels, channels_out, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
