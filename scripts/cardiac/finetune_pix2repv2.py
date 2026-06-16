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

import pix2repv2.data.acdc as acdc
import pix2repv2.data.mnms as mnms
import pix2repv2.data.mnms2 as mnms2
from pix2repv2.data.acdc import (
    ACDCPatchSupervised,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.data.mnms import MnMsPatchSupervised
from pix2repv2.data.mnms2 import MnMs2PatchSupervised
from pix2repv2.finetuning_module import Pix2Repv2FinetuningModule
from pix2repv2.linear_probing_module import Pix2Repv2LinearProbingModule
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
    strategy: "str",
):
    wandb_logger = WandbLogger(
        project="Pix2Repv2-finetuning",
        name=exp_name,
        save_dir="output",
    )

    ckpt_dirpath = Path(
        f"output/lightning_logs/finetuning/cardiac/{exp_name}/checkpoints/"
    )
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

    pretrained_backbone_path = f"output/lightning_logs/pretraining/cardiac/{backbone_name}/checkpoints/{ckpt_name}.ckpt"
    if strategy == "scratch":
        logger.info("Instantiating model from scratch")
        # use a higher learning rate when training from scratch, compared to pretraining LR
        cfg.finetuning.finetuning_lr_backbone = 1e-4
        model = Pix2Repv2FinetuningModule(cfg=cfg)
        log_every_n_steps = finetuning_logging_every_n_steps(
            cfg.finetuning.num_patients
        )

    elif strategy == "linear_probing":
        logger.info(
            f"Loading pretrained backbone from: {pretrained_backbone_path} for linear probing"
        )
        model = Pix2Repv2LinearProbingModule.load_from_checkpoint(
            pretrained_backbone_path,
            cfg=cfg,
            strict=False,
        )
        for param in model.backbone.parameters():
            param.requires_grad = False  # freeze backbone

        log_every_n_steps = 50  # default Lightning value

    else:
        logger.info(
            f"Loading pretrained backbone from: {pretrained_backbone_path} for full finetuning"
        )
        model = Pix2Repv2FinetuningModule.load_from_checkpoint(
            pretrained_backbone_path,
            cfg=cfg,
            strict=False,
        )
        # LR scheduling used during full finetuning
        log_every_n_steps = finetuning_logging_every_n_steps(
            cfg.finetuning.num_patients
        )

    trainer = Trainer(
        accelerator="cuda",
        devices=1,
        deterministic="warn",
        logger=wandb_logger,
        max_epochs=cfg.finetuning.num_epochs,
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
        "--finetuning_dataset",
        type=str,
        default="ACDC",
        help="Which dataset to use for finetuning {ACDC, MnMs, MnMs2}",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name for WandB and checkpoint saving",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        help="Which pretrained backbone to load. Should correspond to the exp_name used during pretraining",
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
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    assert args.exp_name is not None, "Each experience should have a name!"

    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed, workers=True)

    if args.finetuning_dataset == "ACDC":
        selected_patients = acdc.select_nested_patient_ids(cfg)
        acdc_sup_patch_ds = ACDCPatchSupervised(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
            apply_augmentations=True,
            apply_elastic_transform=True,
            selected_patients=selected_patients,
        )
        # NOTE: ACDC doesn't have a predefined validation set. We create a validation set
        # by splitting the training set at the slice level (10% of total slices go to validation).
        train_subset, val_subset = create_train_val_subsets(
            patch_dataset=acdc_sup_patch_ds,
            cfg=cfg,
        )
        logger.info("ACDC dataloaders construction")
        train_loader_sup = build_patch_loader(
            patch_dataset=train_subset,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=True,
        )
        val_loader_sup = build_patch_loader(
            patch_dataset=val_subset,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=False,
        )

    elif args.finetuning_dataset == "MnMs":
        selected_patients = mnms.select_nested_patient_ids(cfg)
        mnms_train_sup_patch_ds = MnMsPatchSupervised(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
            apply_augmentations=True,
            apply_elastic_transform=True,
            selected_patients=selected_patients,
            split="training",
        )
        mnms_val_sup_patch_ds = MnMsPatchSupervised(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
            apply_augmentations=False,
            split="validation",
        )
        logger.info("MnMs dataloaders construction")
        train_loader_sup = build_patch_loader(
            mnms_train_sup_patch_ds,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=True,
        )
        val_loader_sup = build_patch_loader(
            mnms_val_sup_patch_ds,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=False,
        )

    elif args.finetuning_dataset == "MnMs2":
        selected_patients = mnms2.select_nested_patient_ids(cfg)
        mnms2_train_sup_patch_ds = MnMs2PatchSupervised(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
            apply_augmentations=True,
            apply_elastic_transform=True,
            selected_patients=selected_patients,
            split="training",
        )
        mnms2_val_sup_patch_ds = MnMs2PatchSupervised(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
            apply_augmentations=False,
            split="validation",
        )
        logger.info("MnMs2 dataloaders construction")
        train_loader_sup = build_patch_loader(
            mnms2_train_sup_patch_ds,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=True,
        )
        val_loader_sup = build_patch_loader(
            mnms2_val_sup_patch_ds,
            batch_size=cfg.data.batch_size_finetuning,
            shuffle=False,
        )

    else:
        raise ValueError(
            f"Unknown dataset for finetuning: {args.finetuning_dataset}. "
            f"Should be {{ACDC, MnMs, MnMs2}}"
        )

    main(
        cfg=cfg,
        train_loader_sup=train_loader_sup,
        val_loader_sup=val_loader_sup,
        exp_name=args.exp_name,
        backbone_name=args.backbone_name,
        ckpt_name=args.ckpt_name,
        strategy=args.strategy,
    )
