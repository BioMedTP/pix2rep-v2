import argparse

import torch
import wandb_osh
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from torch.utils.data import ConcatDataset
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from pix2repv2.data.acdc import (
    ACDCPatchSSL,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.data.mnms import MnMsPatchSSL
from pix2repv2.data.mnms2 import MnMs2PatchSSL
from pix2repv2.pretraining_module import Pix2Repv2PretrainingModule
from pix2repv2.utils.utils import Config, ModelCheckpointSymlink

wandb_osh.set_log_level("ERROR")


def main(
    cfg: dict,
    train_loader_ssl: torch.utils.data.DataLoader,
    val_loader_ssl: torch.utils.data.DataLoader,
    experiment_name: str,
    distributed: bool,
):
    wandb_logger = WandbLogger(
        project="Pix2Repv2-pretraining",
        name=experiment_name,
        save_dir="output",
    )

    last_ckpt_callback = ModelCheckpoint(
        dirpath=f"output/lightning_logs/pretraining/cardiac/{experiment_name}/checkpoints/",
        save_last="link",  # create a symlink to the last checkpoint
        save_top_k=0,  # save full last checkpoint to resume training if needed
        monitor=None,
    )

    best_ckpt_callback = ModelCheckpointSymlink(
        dirpath=f"output/lightning_logs/pretraining/cardiac/{experiment_name}/checkpoints/",
        save_top_k=3,
        save_weights_only=True,  # for intermediate checkpoints, only saving weights
        monitor="val_loss",
        mode="min",
    )

    if distributed:
        logger.info("Distributed training: enabled")
        strategy = "ddp"
        devices = 4
        num_nodes = 2

        # Syncing BatchNorm only for BT loss (+ projection head). Otherwise, overhead is probably not worth it.
        if (
            hasattr(cfg.pretraining.backbone.params, "norm")
            and cfg.pretraining.backbone.params.norm != "batch"
        ):
            sync_batchnorm = True
        else:
            sync_batchnorm = False

    else:
        logger.info("Distributed training: disabled")
        strategy = "auto"
        devices = 1
        num_nodes = 1
        sync_batchnorm = False

    model = Pix2Repv2PretrainingModule(cfg)
    trainer = Trainer(
        accelerator="cuda",
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        deterministic="warn",  # use deterministic algorithms whenever possible
        logger=wandb_logger,
        max_epochs=cfg.pretraining.num_epochs,
        sync_batchnorm=sync_batchnorm,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TriggerWandbSyncLightningCallback(),
            last_ckpt_callback,
            best_ckpt_callback,
        ],
    )
    trainer.fit(model, train_loader_ssl, val_loader_ssl)

    if num_nodes > 1:
        if trainer.strategy.launcher is not None:
            # NOTE: This forces all ranks (including those on other nodes)
            # to wait until the master node finished the checkpointing symlinks (avoid sync errors).
            trainer.strategy.barrier("end_of_training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretraining_dataset",
        type=str,
        default="ACDC",
        help="Which dataset(s) to use for pretraining. Could be ACDC, MnMs, MnMs2 or all",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name for WandB tracking and checkpoint saving",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,  # Sets the value to False if the flag is absent
        help="Whether to use distributed training (DDP)",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    assert args.exp_name is not None, "Each experience should have a name!"

    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed, workers=True)

    if args.pretraining_dataset == "ACDC":
        acdc_ssl_patch_ds = ACDCPatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
        )
        # NOTE: ACDC doesn't have a predefined validation set. We create a validation set
        # by splitting the training set at the slice level (10% of total slices go to validation).
        train_subset, val_subset = create_train_val_subsets(
            patch_dataset=acdc_ssl_patch_ds,
            cfg=cfg,
        )
        logger.info("ACDC dataloaders construction")
        train_loader_ssl = build_patch_loader(
            patch_dataset=train_subset,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=True,
        )
        val_loader_ssl = build_patch_loader(
            patch_dataset=val_subset,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=False,
        )

    elif args.pretraining_dataset == "MnMs":
        mnms_ssl_patch_ds = MnMsPatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
            split="training",
        )
        val_mnms_ssl_patch_ds = MnMsPatchSSL(
            cfg=cfg,
            apply_augmentations=False,
            apply_elastic_transform=False,
            split="validation",
        )
        logger.info("MnMs dataloaders construction")
        train_loader_ssl = build_patch_loader(
            patch_dataset=mnms_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=True,
        )
        val_loader_ssl = build_patch_loader(
            patch_dataset=val_mnms_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=False,
        )

    elif args.pretraining_dataset == "MnMs2":
        mnms2_ssl_patch_ds = MnMs2PatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
            split="training",
        )
        val_mnms2_ssl_patch_ds = MnMs2PatchSSL(
            cfg=cfg,
            apply_augmentations=False,
            apply_elastic_transform=False,
            split="validation",
        )
        logger.info("MnMs2 dataloaders construction")
        train_loader_ssl = build_patch_loader(
            patch_dataset=mnms2_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=True,
        )
        val_loader_ssl = build_patch_loader(
            patch_dataset=val_mnms2_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=False,
        )

    elif args.pretraining_dataset == "all":
        acdc_patch_ds = ACDCPatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
        )
        train_acdc_patch_subset, val_acdc_patch_subset = create_train_val_subsets(
            patch_dataset=acdc_patch_ds,
            cfg=cfg,
        )
        train_mnms_patch_ds = MnMsPatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
            split="training",
        )
        train_mnms2_patch_ds = MnMs2PatchSSL(
            cfg=cfg,
            apply_augmentations=True,
            apply_elastic_transform=True,
            split="training",
        )
        train_ssl_patch_ds = ConcatDataset(
            datasets=[
                train_acdc_patch_subset,
                train_mnms_patch_ds,
                train_mnms2_patch_ds,
            ],
        )

        val_mnms_patch_ds = MnMsPatchSSL(
            cfg=cfg,
            apply_augmentations=False,
            apply_elastic_transform=False,
            split="validation",
        )
        val_mnms2_patch_ds = MnMs2PatchSSL(
            cfg=cfg,
            apply_augmentations=False,
            apply_elastic_transform=False,
            split="validation",
        )
        val_ssl_patch_ds = ConcatDataset(
            datasets=[val_acdc_patch_subset, val_mnms_patch_ds, val_mnms2_patch_ds],
        )

        logger.info("ACDC + MnMs + MnMs2 dataloaders construction")
        train_loader_ssl = build_patch_loader(
            train_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=True,
        )
        val_loader_ssl = build_patch_loader(
            val_ssl_patch_ds,
            batch_size=cfg.data.batch_size_pretraining,
            shuffle=False,
        )

    else:
        raise ValueError(
            f"Unknown dataset(s) for pretraining: {args.pretraining_dataset}. "
            f"Should be ACDC, MnMs, MnMs2 or all"
        )

    main(
        cfg=cfg,
        train_loader_ssl=train_loader_ssl,
        val_loader_ssl=val_loader_ssl,
        experiment_name=args.exp_name,
        distributed=args.distributed,
    )
