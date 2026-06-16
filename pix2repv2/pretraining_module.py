import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pix2repv2.models import mlp
from pix2repv2.models.registry import build_backbone
from pix2repv2.utils import losses, utils


class Pix2Repv2PretrainingModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        flattened_cfg = utils.flatten_pretraining_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg

        # Backbone architecture
        self.backbone = build_backbone(self.cfg)

        # Projection head architecture
        channels_in = self.backbone.n_feature_maps
        channels_out = self.cfg.pretraining.n_feature_maps_mlp
        depth = self.cfg.pretraining.projection_head_depth
        assert depth >= 0 and isinstance(depth, int), (
            "projection_head_depth must be an int >= 0"
        )

        if depth == 0:
            self.projection_head = nn.Identity()
        else:
            # collect inner dims in order and slice to needed depth
            inner_dims = list()
            if hasattr(self.cfg.pretraining, "inner_dim_1"):
                inner_dims.append(self.cfg.pretraining.inner_dim_1)
            if hasattr(self.cfg.pretraining, "inner_dim_2"):
                inner_dims.append(self.cfg.pretraining.inner_dim_2)
            if hasattr(self.cfg.pretraining, "inner_dim_3"):
                inner_dims.append(self.cfg.pretraining.inner_dim_3)
            if hasattr(self.cfg.pretraining, "inner_dim_4"):
                inner_dims.append(self.cfg.pretraining.inner_dim_4)
            inner_dims = inner_dims[:depth]
            self.projection_head = mlp.ProjectionHead(
                channels_in=channels_in,
                channels_out=channels_out,
                inner_dims=inner_dims,
                norm_type=self.cfg.finetuning.in_context_proj_norm,
            )

    def forward(self, view):
        return self.projection_head(self.backbone(view))

    def training_step(self, batch, batch_idx):
        view1 = rearrange(batch[0], "B C W H 1 -> B C W H")
        view2 = rearrange(batch[1], "B C W H 1 -> B C W H")
        grid = batch[2].squeeze(1)  # remove channel dim (C=1)

        # Apply spatial transformation to the second view
        view2 = F.grid_sample(
            view2,
            grid,
            mode="bicubic",
            align_corners=True,
        )  # shape: (B, C, W, H)

        # Forward pass on the two views
        dense_features_1 = self(view1)
        dense_features_2 = self(view2)

        # Apply spatial transformation to the first view local feature maps
        dense_features_1 = F.grid_sample(
            dense_features_1,
            grid,
            mode="bicubic",
            align_corners=True,
        )

        # Compute Barlow Twins loss
        train_loss = losses.barlow_loss(
            dense_features_1,
            dense_features_2,
            self.cfg.pretraining.lambda_barlow,
            self.device,
        )

        self.log(
            "train_loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size_pretraining,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        view1 = rearrange(batch[0], "B C W H 1 -> B C W H")
        view2 = rearrange(batch[1], "B C W H 1 -> B C W H")
        grid = batch[2].squeeze(1)  # remove channel dim (C=1)

        # Apply spatial transformation to the second view
        view2 = F.grid_sample(
            view2,
            grid,
            mode="bicubic",
            align_corners=True,
        )  # shape: (B, C, W, H)

        # Forward pass on the two views
        dense_features_1 = self(view1)
        dense_features_2 = self(view2)

        # Apply spatial transformation to the first view local feature maps
        dense_features_1 = F.grid_sample(
            dense_features_1,
            grid,
            mode="bicubic",
            align_corners=True,
        )

        # Compute Barlow Twins loss
        val_loss = losses.barlow_loss(
            dense_features_1,
            dense_features_2,
            self.cfg.pretraining.lambda_barlow,
            self.device,
        )

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.cfg.data.batch_size_pretraining,
        )
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.projection_head.parameters()},
            ],
            lr=self.cfg.pretraining.lr_backbone,
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.05 * total_steps)  # warmup for 5% of total steps

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-6,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        cosine_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-8,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine_schedule],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
