import lightning as L
import monai.losses
import monai.metrics
import torch
from einops import rearrange

from pix2repv2.models.registry import build_backbone
from pix2repv2.models.layers import OutConv
from pix2repv2.utils import utils


class Pix2Repv2LinearProbingModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        flattened_cfg = utils.flatten_finetuning_cfg(cfg)
        self.save_hyperparameters(flattened_cfg)
        self.cfg = cfg

        # Backbone architecture
        self.backbone = build_backbone(self.cfg)

        # Finetuning head
        self.finetuning_layer = OutConv(
            self.backbone.n_feature_maps,
            self.cfg.finetuning.n_classes,
        )

        # Fine-tuning loss (for DiceLoss only, set lambda_ce=0.0 in config)
        self.dice_ce_loss = monai.losses.DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            reduction="mean",
            softmax=True,
            lambda_dice=self.cfg.finetuning.lambda_dice,
            lambda_ce=self.cfg.finetuning.lambda_ce,
        )
        # Dice metric for validation
        self.dice = monai.metrics.DiceMetric(
            include_background=False,
            reduction="mean",
            num_classes=self.cfg.finetuning.n_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        logits = self.finetuning_layer(features)
        return logits  # shape: (B, num_classes, H, W)

    def training_step(self, batch, batch_idx):
        images = rearrange(batch[0], "B C W H 1 -> B C W H")
        masks = rearrange(batch[1], "B C W H 1 -> B C W H")
        logits = self(images)
        train_loss = self.dice_ce_loss(logits, masks)
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.data.batch_size_finetuning,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        images = rearrange(batch[0], "B C W H 1 -> B C W H")
        masks = rearrange(batch[1], "B C W H 1 -> B C W H")
        logits = self(images)
        val_loss = self.dice_ce_loss(logits, masks)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.cfg.data.batch_size_finetuning,
        )

        # Computing Dice score for the current batch (to be aggregated at epoch end)
        softmax = torch.nn.Softmax(dim=1)
        preds_labels = torch.argmax(softmax(logits), dim=1, keepdim=True)
        self.dice(preds_labels, masks)

        return val_loss

    def on_validation_epoch_end(self):
        # NOTE: .aggregate() returns a tensor of per-class Dice scores averaged across all batches
        # (only apply reduction on `not-nan` values)
        val_dice_scores = self.dice.aggregate().item()
        self.log("val_dice", val_dice_scores)
        self.dice.reset()

    def configure_optimizers(self):
        linear_probing_optim = torch.optim.AdamW(
            self.finetuning_layer.parameters(),
            lr=self.cfg.finetuning.linear_probing_lr_outconv,
        )
        return linear_probing_optim
