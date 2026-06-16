import lightning as L
import monai.losses
import monai.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from pix2repv2.models.unet import PartialUNet3D
from pix2repv2.utils import utils


class Pix2Repv2_3DFinetuningModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        flattened_cfg = utils.flatten_finetuning_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg
        self.class_labels = {
            "0": "background",
            "1": "spleen",
            "2": "right kidney",
            "3": "left kidney",
            "4": "gall bladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "aorta",
            "9": "postcava",
            "10": "pancreas",
            "11": "right adrenal gland",
            "12": "left adrenal gland",
            "13": "duodenum",
            "14": "bladder",
            "15": "prostate/uterus",
        }
        self.num_classes = len(self.class_labels)

        # Backbone architecture
        self.backbone = PartialUNet3D(
            in_channels=self.cfg.pretraining.n_channels,
            base_channels=self.cfg.pretraining.base_channels,
            norm=self.cfg.pretraining.norm,
        )

        in_channels_implicit_decoder = (
            self.backbone.base_channels
            + self.backbone.base_channels * 2**1
            + self.backbone.base_channels * 2**2
        )  # 32 + 64 + 128 (i.e. H2 + H1 + H0)

        D = self.cfg.pretraining.n_feature_maps_3D

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

        # Finetuning head
        self.finetuning_layer = nn.Linear(D, self.num_classes)

        # Fine-tuning loss (for DiceLoss only, set lambda_ce=0.0 in config)
        self.dice_ce_loss = monai.losses.DiceCELoss(
            include_background=True,  # NOTE: this seems to work better
            to_onehot_y=True,
            softmax=True,  # only used by the `DiceLoss`
            batch=self.cfg.finetuning.batch_dice,
            reduction="mean",
            lambda_dice=self.cfg.finetuning.lambda_dice,
            lambda_ce=self.cfg.finetuning.lambda_ce,
        )

        # Dice metric for validation
        self.train_dice = monai.metrics.DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=self.num_classes,
        )

        self.val_dice = monai.metrics.DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=self.num_classes,
        )

    def forward(self, imgs):
        partial_features, skip_H1, skip_H0 = self.backbone(imgs)
        B, _, D, H, W = imgs.shape

        grid = utils.make_base_grid_5d(
            D, H, W, align_corners=True, dtype=imgs.dtype, device=imgs.device
        )  # shape: (D, H, W, 4)
        grid = grid[..., 0:3]  # shape: (D, H, W, 3)
        grid = grid.repeat(B, 1, 1, 1, 1)

        # Each feature maps is of shape: (B, C, D, H, W)
        feats_H2 = F.grid_sample(
            partial_features, grid, mode="bilinear", align_corners=True
        )
        feats_H1 = F.grid_sample(skip_H1, grid, mode="bilinear", align_corners=True)
        feats_H0 = F.grid_sample(skip_H0, grid, mode="bilinear", align_corners=True)

        point_features = torch.cat(
            [
                rearrange(feats_H2, "B C D H W -> (B D H W) C"),
                rearrange(feats_H1, "B C D H W -> (B D H W) C"),
                rearrange(feats_H0, "B C D H W -> (B D H W) C"),
            ],
            dim=1,
        )

        logits = self.finetuning_layer(self.implicit_decoder(point_features))

        logits = rearrange(
            logits,
            "(B D H W) n_classes -> B n_classes D H W",
            B=B,
            D=D,
            H=H,
            W=W,
        )

        return logits

    def training_step(self, batch, batch_idx):
        imgs = rearrange(batch[0], "B C H W D -> B C D H W")
        masks = rearrange(batch[1], "B 1 H W D -> B 1 D H W")

        assert self.backbone.training
        assert self.implicit_decoder.training
        assert self.finetuning_layer.training

        logits = self(imgs)  # shape: (B, C, D, H, W)

        train_loss = self.dice_ce_loss(logits, masks)

        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True)

        preds = torch.argmax(logits, dim=1, keepdim=True)
        self.train_dice(preds, masks)

        return train_loss

    def validation_step(self, batch, batch_idx):
        imgs = rearrange(batch[0], "B C H W D -> B C D H W")
        masks = rearrange(batch[1], "B 1 H W D -> B 1 D H W")

        logits = self(imgs)  # shape: (B, C, D, H, W)

        val_loss = self.dice_ce_loss(logits, masks)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Computing Dice score for the current batch (to be aggregated at epoch end)
        preds = torch.argmax(logits, dim=1, keepdim=True)
        self.val_dice(preds, masks)

        return val_loss

    def on_validation_epoch_end(self):
        # .aggregate() returns a tensor of per-class Dice scores averaged across all batches
        # (only apply reduction on `not-nan` values)
        val_dice_scores = self.val_dice.aggregate().item()
        self.log("val_dice", val_dice_scores)
        self.val_dice.reset()

    def on_train_epoch_end(self):
        # .aggregate() returns a tensor of per-class Dice scores averaged across all batches
        # (only apply reduction on `not-nan` values)
        train_dice_scores = self.train_dice.aggregate().item()
        self.log("train_dice", train_dice_scores)
        self.train_dice.reset()

    def configure_optimizers(self):
        # Different LRs for backbone and segmentation head
        param_groups = [
            {
                "params": self.backbone.parameters(),
                "lr": self.cfg.finetuning.finetuning_lr_backbone,
            },
            {
                "params": self.implicit_decoder.parameters(),
                "lr": self.cfg.finetuning.finetuning_lr_backbone,
            },
            {
                "params": self.finetuning_layer.parameters(),
                "lr": self.cfg.finetuning.finetuning_lr_outconv,
            },
        ]

        optimizer = torch.optim.AdamW(param_groups)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.03 * total_steps)  # 3% warmup

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
