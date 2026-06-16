import argparse

import torch
import wandb_osh
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from pix2repv2.data.amos import (
    AMOSPatchSSL,
    build_amos_pretraining_pipeline,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.pretraining_module_3d_implicit import Pix2Repv2_3DPretrainingModule
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
        project="Pix2Repv2-3d-pretraining",
        name=experiment_name,
        save_dir="output",
    )

    last_ckpt_callback = ModelCheckpoint(
        dirpath=f"output/lightning_logs/pretraining/ct/{experiment_name}/checkpoints/",
        save_last="link",  # create a symlink to the last checkpoint
        save_top_k=0,  # save full last checkpoint to resume training if needed
        monitor=None,
    )

    best_ckpt_callback = ModelCheckpointSymlink(
        dirpath=f"output/lightning_logs/pretraining/ct/{experiment_name}/checkpoints/",
        save_top_k=3,
        save_weights_only=True,  # for intermediate checkpoints, only saving weights
        monitor="val_loss",
        mode="min",
    )

    if distributed:
        logger.info("Distributed training: enabled")
        strategy = "ddp"
        devices = 4
        num_nodes = 1

        # Syncing BatchNorm only for BT (and projection head). Otherwise, overhead is not worth it.
        if hasattr(cfg.pretraining, "norm") and cfg.pretraining.norm != "batch":
            sync_batchnorm = True
        else:
            sync_batchnorm = False

    else:
        logger.info("Distributed training: disabled")
        strategy = "auto"
        devices = 1
        num_nodes = 1
        sync_batchnorm = False

    model = Pix2Repv2_3DPretrainingModule(cfg)
    trainer = Trainer(
        accelerator="cuda",
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        deterministic="warn",  # use deterministic algorithms when possible
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
            # This forces all ranks (including those on other nodes)
            # to wait until the master node finished the checkpointing symlinks.
            trainer.strategy.barrier("end_of_training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name for WandB logging and checkpointing",
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

    cfg = Config(config_name="config3D", overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed, workers=True)

    pipeline = build_amos_pretraining_pipeline()
    ds = AMOSPatchSSL(
        cfg,
        pipeline=pipeline,
        apply_augmentations=True,
        apply_elastic_transform=False,
    )
    # Remove broken sample ("6016") whose spatial dims are [623, 0, 0]
    ds.ids = [id for id in ds.ids if id != "6016"]

    train_subset, val_subset = create_train_val_subsets(
        patch_dataset=ds,
        cfg=cfg,
    )

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

    main(
        cfg=cfg,
        train_loader_ssl=train_loader_ssl,
        val_loader_ssl=val_loader_ssl,
        experiment_name=args.exp_name,
        distributed=args.distributed,
    )
