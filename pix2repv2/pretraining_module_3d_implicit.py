import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pix2repv2.models.unet import PartialUNet3D
from pix2repv2.utils import losses, utils


class Pix2Repv2_3DPretrainingModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        flattened_cfg = utils.flatten_pretraining_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg

        # Backbone architecture
        self.backbone = PartialUNet3D(
            in_channels=self.cfg.pretraining.n_channels,
            base_channels=self.cfg.pretraining.base_channels,
            norm=self.cfg.pretraining.norm,
        )

        in_channels_implicit_decoder = (
            self.cfg.pretraining.base_channels
            + self.cfg.pretraining.base_channels * 2**1
            + self.cfg.pretraining.base_channels * 2**2
        )  # 32 + 64 + 128 (i.e. H2 + H1 + H0)

        D = self.cfg.pretraining.n_feature_maps_3D
        d = self.cfg.pretraining.proj_dim_3D

        self.implicit_decoder = nn.Sequential(
            nn.BatchNorm1d(in_channels_implicit_decoder),
            nn.Linear(in_channels_implicit_decoder, D),
            nn.BatchNorm1d(D),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.Dropout(p=0.2),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
            nn.BatchNorm1d(D),
            nn.ReLU(inplace=True),
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(D, d),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, view, sparse_grid):
        partial_features, skip_H1, skip_H0 = self.backbone(view)
        point_features_H2 = F.grid_sample(
            partial_features, sparse_grid, mode="bilinear", align_corners=True
        )
        point_features_H1 = F.grid_sample(
            skip_H1, sparse_grid, mode="bilinear", align_corners=True
        )
        point_features_H0 = F.grid_sample(
            skip_H0, sparse_grid, mode="bilinear", align_corners=True
        )

        point_features = torch.cat(
            (
                rearrange(point_features_H2, "B C 1 1 N -> B C N"),
                rearrange(point_features_H1, "B C 1 1 N -> B C N"),
                rearrange(point_features_H0, "B C 1 1 N -> B C N"),
            ),
            dim=1,
        )  # shape: (B, C = C1 + C2 + C3, Npoints)
        point_features = rearrange(point_features, "B C N -> (B N) C")

        point_features = self.implicit_decoder(point_features)
        point_features_proj = self.projection_head(point_features)

        return point_features_proj

    def training_step(self, batch, batch_idx):
        view1_3d = rearrange(batch[0], "B 1 H W D -> B 1 D H W")
        view2_3d = rearrange(batch[1], "B 1 H W D -> B 1 D H W")
        dense_grid_3d = batch[2].squeeze(1)  # shape: (B, D, H, W, 3)
        sparse_grid = batch[3].squeeze(1)  # shape: (B, 1, 1, Npoints, 3)
        sparse_grid_warped = batch[4].squeeze(1)  # shape: (B, 1, 1, Npoints, 3)

        assert self.backbone.training
        assert self.implicit_decoder.training
        assert self.projection_head.training

        # Apply spatial transformation to the second view
        view2_3d = F.grid_sample(
            view2_3d,
            dense_grid_3d,
            mode="bilinear",  # NOTE: mode='bicubic' supports only 4-D input. When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear
            align_corners=True,
        )  # shape: (B, C, D, H, W)

        # Forward pass on the two views
        point_features_1 = self(view1_3d, sparse_grid_warped)
        point_features_2 = self(view2_3d, sparse_grid)

        # Compute loss with the Barlow Twins loss
        train_loss = losses.barlow_loss_3d(
            point_features_1,
            point_features_2,
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
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        view1_3d = rearrange(batch[0], "B 1 H W D -> B 1 D H W")
        view2_3d = rearrange(batch[1], "B 1 H W D -> B 1 D H W")
        dense_grid_3d = batch[2].squeeze(1)  # shape: (B, D, H, W, 3)
        sparse_grid = batch[3].squeeze(1)  # shape: (B, 1, 1, Npoints, 3)
        sparse_grid_warped = batch[4].squeeze(1)  # shape: (B, 1, 1, Npoints, 3)

        # Apply spatial transformation to the second view
        view2_3d = F.grid_sample(
            view2_3d,
            dense_grid_3d,
            mode="bilinear",  # NOTE: mode='bicubic' supports only 4-D input. When mode='bilinear' and the input is 5-D, the interpolation mode used internally will actually be trilinear
            align_corners=True,
        )  # shape: (B, C, D, H, W)

        # Forward pass on the two views
        point_features_1 = self(view1_3d, sparse_grid_warped)
        point_features_2 = self(view2_3d, sparse_grid)

        # Compute loss with the Barlow Twins loss
        val_loss = losses.barlow_loss_3d(
            point_features_1,
            point_features_2,
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
        )
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.backbone.parameters()},
                {"params": self.implicit_decoder.parameters()},
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
