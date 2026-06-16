import torch.nn as nn
from loguru import logger

from .swin_unetr import SwinUNETR_SSL
from .unet import UNet

BACKBONE_REGISTRY = {
    "unet": UNet,
    "swin_unetr": SwinUNETR_SSL,
}


def build_backbone(cfg: dict) -> nn.Module:
    backbone_name = cfg.pretraining.backbone.name.lower()

    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )

    backbone_cls = BACKBONE_REGISTRY[backbone_name]
    logger.info(f"Building backbone: {backbone_name}")

    return backbone_cls(**cfg.pretraining.backbone.params)
