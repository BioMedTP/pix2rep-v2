from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.blocks import UnetBasicBlock, UnetResBlock
from monai.networks.blocks.dynunet_block import get_conv_layer
from monai.networks.nets.swin_unetr import SwinUNETR


class SwinUNETR_SSL(SwinUNETR):
    def __init__(
        self,
        in_channels: int,
        out_channels_repr: int | None,
        feature_size: int,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 2,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            feature_size=feature_size,
            norm_name=norm_name,
            spatial_dims=spatial_dims,
            **kwargs,
        )

        # Attribute to know the number of output channels of the deep representation to be used in the projection head
        if out_channels_repr is None:
            self.n_feature_maps = feature_size
        else:
            self.n_feature_maps = out_channels_repr

        self.decoder1 = UnetrUpBlockCustom(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            # if out_channels_repr = feature_size, then original behavior. Otherwise, we expand
            # the representation to out_channels_repr channels
            out_channels_repr=out_channels_repr,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = nn.Identity()  # remove the final classification layer

    def forward(self, x_in):
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._check_input_size(x_in.shape[2:])
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        # custom decoder that outputs 1024 feature maps
        out = self.decoder1(dec0, enc0)
        # Remove last classification layer for SSL pretraining by outputing
        # the feature representations directly.
        # logits = self.out(out)
        return out


class UnetrUpBlockCustom(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

    Customized to allow different number of output channels for the deep representation
    (e.g., going from 120 to 1024 feature maps at the highest resolution level).
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        out_channels_repr: int | None,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            out_channels_repr: number of output channels for the deep representation.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if out_channels_repr is None:
            # Original behavior
            out_channels_repr = out_channels

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels_repr,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels_repr,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
