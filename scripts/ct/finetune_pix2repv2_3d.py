import argparse
import json
from pathlib import Path

import torch
import wandb_osh
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

from pix2repv2.data.acdc import create_train_val_subsets
from pix2repv2.data.amos import (
    AMOSPatchSupervised,
    build_amos_finetuning_pipeline,
    build_patch_loader,
)
from pix2repv2.finetuning_module_3d_implicit import Pix2Repv2_3DFinetuningModule
from pix2repv2.utils.utils import (
    Config,
    ModelCheckpointSymlink,
    finetuning_logging_every_n_steps,
)

wandb_osh.set_log_level("ERROR")


def main(
    cfg: dict,
    train_loader_sup: torch.utils.data.DataLoader,
    val_loader_sup: torch.utils.data.DataLoader,
    exp_name: str,
    backbone_name: str,
    ckpt_name: str,
    strategy: str,
    distributed: bool,
):
    wandb_logger = WandbLogger(
        project="Pix2Repv2-3d-finetuning",
        name=exp_name,
        save_dir="output",
    )

    ckpt_dirpath = Path(f"output/lightning_logs/finetuning/ct/{exp_name}/checkpoints/")
    last_ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dirpath,
        save_last="link",  # create a symlink to the last checkpoint
        save_top_k=0,  # save full last checkpoint to resume training if needed
        monitor=None,
    )

    best_ckpt_callback = ModelCheckpointSymlink(
        dirpath=ckpt_dirpath,
        save_top_k=2,
        save_weights_only=True,  # for intermediate checkpoints, only saving weights
        monitor="val_dice",
        mode="max",
    )

    if cfg.finetuning.num_patients == 1:
        cfg.finetuning.num_epochs = 3000
    elif cfg.finetuning.num_patients == 5:
        cfg.finetuning.num_epochs = 1500
    else:
        cfg.finetuning.num_epochs = 1000

    if strategy == "scratch":
        logger.info("Instantiating model from scratch")
        cfg.finetuning.finetuning_lr_backbone = 1e-4  # same as pretraining LR
        model = Pix2Repv2_3DFinetuningModule(cfg=cfg)
        log_every_n_steps = finetuning_logging_every_n_steps(
            cfg.finetuning.num_patients
        )
    elif strategy == "finetuning":
        pretrained_backbone_path = f"output/lightning_logs/pretraining/ct/{backbone_name}/checkpoints/{ckpt_name}.ckpt"
        logger.info(
            f"Loading pretrained backbone from: {pretrained_backbone_path} for full finetuning"
        )
        model = Pix2Repv2_3DFinetuningModule.load_from_checkpoint(
            pretrained_backbone_path,
            cfg=cfg,
            strict=False,
        )
        # LR scheduling used during full finetuning
        log_every_n_steps = finetuning_logging_every_n_steps(
            cfg.finetuning.num_patients
        )
    else:
        raise ValueError(
            f"Unsupported strategy: {strategy}. Should be either 'scratch' or 'finetuning'"
        )

    if distributed:
        logger.info("Distributed training: enabled")
        strategy = "ddp"
        devices = 4
        num_nodes = 1

        # Syncing BatchNorm only for implicit decoder. Otherwise, overhead is not worth it.
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

    trainer = Trainer(
        accelerator="auto",
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision="bf16",  # NOTE: bf16 fully supported on A100 & H100 (avoid on V100 because fallback to fp32 on unsupported hardware)
        deterministic="warn",  # NOTE: Put "warn" to avoid potential issues from non-deterministic ops if set to True
        logger=wandb_logger,
        max_epochs=cfg.finetuning.num_epochs,
        sync_batchnorm=sync_batchnorm,
        log_every_n_steps=log_every_n_steps,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TriggerWandbSyncLightningCallback(),
            last_ckpt_callback,
            best_ckpt_callback,
        ],
    )
    trainer.fit(model, train_loader_sup, val_loader_sup)

    with (ckpt_dirpath / "selected_patients.json").open("w") as f:
        if hasattr(train_loader_sup.dataset, "dataset"):
            json.dump(train_loader_sup.dataset.dataset.selected_patients, f, indent=2)
        else:
            json.dump(train_loader_sup.dataset.selected_patients, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name for WandB",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        help="Which pretrained backbone to load",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best",
        help="Which checkpoint (last, best, intermediate?) to load",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="finetuning",  # {finetuning, linear_probing, scratch}
        help="Which finetuning strategy to use. Could be full finetuning, linear probing or training from scratch",
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

    pipeline = build_amos_finetuning_pipeline()
    selected_patients = AMOSPatchSupervised.select_patient_ids(cfg, pipeline.ids)

    ds = AMOSPatchSupervised(
        cfg,
        pipeline=pipeline,
        selected_patients=selected_patients,
    )

    train_subset, val_subset = create_train_val_subsets(
        patch_dataset=ds,
        cfg=cfg,
    )

    train_loader_sup = build_patch_loader(
        patch_dataset=train_subset,
        batch_size=cfg.data.batch_size_finetuning,
        shuffle=True,
        prefetch_factor=4,
    )
    val_loader_sup = build_patch_loader(
        patch_dataset=val_subset,
        batch_size=cfg.data.batch_size_finetuning,
        shuffle=False,
        prefetch_factor=4,
    )

    main(
        cfg=cfg,
        train_loader_sup=train_loader_sup,
        val_loader_sup=val_loader_sup,
        exp_name=args.exp_name,
        backbone_name=args.backbone_name,
        ckpt_name=args.ckpt_name,
        strategy=args.strategy,
        distributed=args.distributed,
    )
