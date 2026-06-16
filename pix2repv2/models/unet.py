import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import get_norm_layer


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels_repr: int | None,
        init_filters: int = 64,
        norm: str = "batch",
        bilinear: bool = False,
    ):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.norm = norm
        self.bilinear = bilinear
        # Attribute to know the number of output channels of the deep representation
        # to be used in the projection head
        if out_channels_repr is None:
            self.n_feature_maps = init_filters
        else:
            self.n_feature_maps = out_channels_repr

        self.inc = DoubleConv(in_channels, init_filters, norm)
        self.down1 = Down(init_filters, init_filters * 2, norm)
        self.down2 = Down(init_filters * 2, init_filters * 4, norm)
        self.down3 = Down(init_filters * 4, init_filters * 8, norm)
        self.down4 = Down(init_filters * 8, init_filters * 16, norm)
        factor = 2 if bilinear else 1
        self.down5 = Down(init_filters * 16, init_filters * 32 // factor, norm)
        self.up1 = Up(init_filters * 32, init_filters * 16 // factor, norm, bilinear)
        self.up2 = Up(init_filters * 16, init_filters * 8 // factor, norm, bilinear)
        self.up3 = Up(init_filters * 8, init_filters * 4 // factor, norm, bilinear)
        self.up4 = Up(init_filters * 4, init_filters * 2 // factor, norm, bilinear)
        if out_channels_repr is not None:
            self.up5 = Up(init_filters * 2, out_channels_repr, norm, bilinear)
        else:
            # Default U-Net behavior: output same number of channels as first layer
            self.up5 = Up(init_filters * 2, init_filters, norm, bilinear)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        return x


class PartialUNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        norm: str,
        bilinear: bool = False,
    ):
        super(PartialUNet3D, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.norm = norm
        self.bilinear = bilinear

        # NOTE: modified kernel_size=(1, 3, 3) and padding=(0, 1, 1) for the first conv layer
        self.inc = DoubleConvAMOS(in_channels, base_channels, norm, ndim=3)
        self.down1 = Down(
            base_channels,
            base_channels * 2,
            norm,
            maxpool_kernel_size=(1, 2, 2),
            ndim=3,
        )
        self.down2 = Down(base_channels * 2, base_channels * 4, norm, ndim=3)
        self.down3 = Down(base_channels * 4, base_channels * 8, norm, ndim=3)
        self.down4 = Down(base_channels * 8, base_channels * 16, norm, ndim=3)
        self.down5 = Down(base_channels * 16, base_channels * 16, norm, ndim=3)
        factor = 2 if bilinear else 1
        self.up1 = UpAMOS(
            base_channels * 32, base_channels * 16 // factor, norm, bilinear, ndim=3
        )
        self.up2 = Up(
            base_channels * 16, base_channels * 8 // factor, norm, bilinear, ndim=3
        )
        self.up3 = Up(
            base_channels * 8, base_channels * 4 // factor, norm, bilinear, ndim=3
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)

        skip_H0 = x1  # First skip connection at full resolution (*128^3, 32 channels)
        skip_H1 = x2  # Second skip connection at half resolution (*64^3, 64 channels)

        return x, skip_H1, skip_H0  # shape: (B, C, D, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, ndim: int = 2):
        super().__init__()
        # store the convolution and RELU layers
        if ndim == 2:
            conv = nn.Conv2d
        elif ndim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")

        self.double_conv = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # Conv2d_WS(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm, out_channels, ndim),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # Conv2d_WS(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm, out_channels, ndim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        maxpool_kernel_size: int | tuple[int] = 2,
        ndim: int = 2,
    ):
        super().__init__()
        if ndim == 2:
            maxpool = nn.MaxPool2d
        elif ndim == 3:
            maxpool = nn.MaxPool3d
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")

        self.maxpool_conv = nn.Sequential(
            maxpool(kernel_size=maxpool_kernel_size),
            DoubleConv(in_channels, out_channels, norm, ndim),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        bilinear: bool = True,
        ndim: int = 2,
    ):
        super().__init__()
        self.ndim = ndim

        if self.ndim == 2:
            convtranspose = nn.ConvTranspose2d
            up_mode = "bilinear"
        elif self.ndim == 3:
            convtranspose = nn.ConvTranspose3d
            up_mode = "trilinear"
        else:
            raise ValueError(f"ndim must be 2 or 3, got {self.ndim}")

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm, self.ndim)
        else:
            self.up = convtranspose(
                in_channels,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels, norm, self.ndim)

    def forward(self, x1, x2):
        # input is (1, C, H, W)
        x1 = self.up(x1)
        if self.ndim == 2:
            diffY = x2.shape[2] - x1.shape[2]
            diffX = x2.shape[3] - x1.shape[3]
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )

        else:  # 3D
            diffZ = x2.shape[2] - x1.shape[2]
            diffY = x2.shape[3] - x1.shape[3]
            diffX = x2.shape[4] - x1.shape[4]

            x1 = F.pad(
                x1,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConvAMOS(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str, ndim: int = 2):
        super().__init__()
        # store the convolution and RELU layers
        if ndim == 2:
            conv = nn.Conv2d
        elif ndim == 3:
            conv = nn.Conv3d
        else:
            raise ValueError(f"ndim must be 2 or 3, got {ndim}")

        self.double_conv = nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            get_norm_layer(norm, out_channels, ndim),
            nn.ReLU(inplace=True),
            conv(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                padding=(0, 1, 1),
                bias=False,
            ),
            get_norm_layer(norm, out_channels, ndim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UpAMOS(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        bilinear: bool = True,
        ndim: int = 2,
    ):
        super().__init__()
        self.ndim = ndim

        if self.ndim == 2:
            convtranspose = nn.ConvTranspose2d
            up_mode = "bilinear"
        elif self.ndim == 3:
            convtranspose = nn.ConvTranspose3d
            up_mode = "trilinear"
        else:
            raise ValueError(f"ndim must be 2 or 3, got {self.ndim}")

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, norm, self.ndim)
        else:
            self.up = convtranspose(
                in_channels // 2,
                in_channels // 2,
                kernel_size=2,
                stride=2,
            )
            self.conv = DoubleConv(in_channels, out_channels, norm, self.ndim)

    def forward(self, x1, x2):
        # input is (1, C, H, W)
        x1 = self.up(x1)
        if self.ndim == 2:
            diffY = x2.shape[2] - x1.shape[2]
            diffX = x2.shape[3] - x1.shape[3]
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
            )

        else:  # 3D
            diffZ = x2.shape[2] - x1.shape[2]
            diffY = x2.shape[3] - x1.shape[3]
            diffX = x2.shape[4] - x1.shape[4]

            x1 = F.pad(
                x1,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
