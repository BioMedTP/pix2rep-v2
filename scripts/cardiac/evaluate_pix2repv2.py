import argparse
import json
import os
from datetime import datetime

import monai.metrics
import torch
import torchio as tio
from einops import rearrange
from lightning import seed_everything
from loguru import logger
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice

from pix2repv2.data.acdc import (
    ACDCPatchSupervised,
    ACDCSupervisedEval,
    build_patch_loader,
    create_train_val_subsets,
)
from pix2repv2.data.mnms import MnMsPatchSupervised, MnMsSupervisedEval
from pix2repv2.data.mnms2 import MnMs2PatchSupervised, MnMs2SupervisedEval
from pix2repv2.finetuning_module import Pix2Repv2FinetuningModule
from pix2repv2.utils import utils
from pix2repv2.utils.utils import Config, setup_pytorch


def main(
    cfg: dict,
    loader: torch.utils.data.DataLoader,
    split: str,
    finetuned_weights_path: str | os.PathLike,
    device: str | torch.device,
    result_out_dir: str | os.PathLike,
    apply_postprocessing: bool,
    inference_strategy: str,
):
    finetuned_model = Pix2Repv2FinetuningModule.load_from_checkpoint(
        finetuned_weights_path,
        cfg=cfg,
        strict=True,
    )
    finetuned_model.eval().to(device)

    global_dice = monai.metrics.DiceMetric(
        include_background=False,
        reduction="mean",
        num_classes=cfg.finetuning.n_classes,
    )
    per_class_dice = monai.metrics.DiceMetric(
        include_background=False,
        reduction="mean_batch",
        num_classes=cfg.finetuning.n_classes,
        return_with_label=True,
    )
    global_dice.reset()
    per_class_dice.reset()

    per_patient_dice_scores: dict[str, list[float]] = dict()
    slices_evaluated = 0  # number of 2D slices used for evaluation

    with torch.no_grad():
        for i, (imgs, masks, pid, orig_spacing, actual_spacing_out) in enumerate(
            loader
        ):
            if split == "val":
                imgs = rearrange(imgs, "B C W H D -> B C W H").to(device)
                logits = finetuned_model(imgs)
                masks = rearrange(masks, "B C W H D -> B C W H").int().to(device)
            elif split == "test":
                # batch_size=1 for test loader (i.e. each batch is a full patient volume)
                imgs = rearrange(imgs, "1 C W H D -> D C W H").to(device)

                if inference_strategy == "full_image":
                    logits = finetuned_model(imgs)
                else:
                    # Sliding window inference from 128x128 patches
                    logits = sliding_window_inference(
                        inputs=imgs,
                        roi_size=(128, 128),
                        sw_batch_size=16,  # need at least a V100-16GB
                        predictor=finetuned_model,
                        overlap=0.5,
                        mode="constant",
                    )
                logits = rearrange(logits, "D C W H -> C W H D")

            # Transform logits to label predictions
            softmax = torch.nn.Softmax(dim=0)(logits)

            # Resample back probability maps to original spacing and size
            sw, sh, sd = (
                orig_spacing[0].item(),
                orig_spacing[1].item(),
                orig_spacing[2].item(),
            )
            actual_spacing_out = (
                actual_spacing_out[0].item(),
                actual_spacing_out[1].item(),
                actual_spacing_out[2].item(),
            )
            softmax, _ = utils.resample(
                softmax,
                spacing_in=actual_spacing_out,
                spacing_out=(sw, sh, sd),
                mode="bicubic",
            )  # shape: (C, W, H, D)

            predicted_masks = torch.argmax(softmax, dim=0, keepdim=True)

            if apply_postprocessing:
                subject = tio.Subject(
                    predicted_mask=tio.LabelMap(tensor=predicted_masks.detach().cpu())
                )
                subject = tio.transforms.KeepLargestComponent()(subject)
                keep_lcc = subject["predicted_mask"].data.to(device)
                predicted_masks = predicted_masks * keep_lcc

            predicted_masks = predicted_masks.unsqueeze(0)  # shape: (B=1, C=1, W, H, D)
            masks = masks.int().to(device)  # shape (B=1, C=1, W, H, D)

            assert predicted_masks.shape == masks.shape, (
                f"Predicted masks shape {predicted_masks.shape} should match "
                f"ground truth masks shape {masks.shape}. Verify resampling step."
            )
            global_dice(predicted_masks, masks)
            per_class_dice(predicted_masks, masks)

            if masks.dim() == 4:
                slices_in_batch = masks.shape[-1]
            elif masks.dim() == 5:
                slices_in_batch = masks.shape[0] * masks.shape[-1]
            else:
                slices_in_batch = masks.shape[-1]
            slices_evaluated += int(slices_in_batch)

            if split == "test":
                # Re-compute Dice to store per-patient Dice scores
                patient_dice = compute_dice(
                    predicted_masks,
                    masks,
                    include_background=False,
                    num_classes=cfg.finetuning.n_classes,
                )
                pid_str = "".join(pid)
                per_patient_dice_scores[f"{pid_str}_{i}"] = patient_dice.cpu().tolist()
            else:
                per_patient_dice_scores = None

        # Aggregate over all batches
        mean_global_dice = global_dice.aggregate().item()  # scalar
        mean_per_class_dice = per_class_dice.aggregate()  # [num_classes] vector

    results = {
        "date": str(datetime.now()),
        "model": finetuned_weights_path,
        "mean_global_dice": float(mean_global_dice),
        "per_class_global_dice": mean_per_class_dice,
        "n_slices": slices_evaluated,  # number of 2D slices used for evaluation
        "n_batches": len(loader),
        "batch_size": loader.batch_size,
        "post_processing": apply_postprocessing,
        "inference_strategy": inference_strategy,
        "per_patient_dice_scores": per_patient_dice_scores,  # Dice Score detail per patient
    }

    os.makedirs(result_out_dir, exist_ok=True)
    filename_base = f"metrics_{args.finetuning_exp_name}"
    if inference_strategy == "sliding_window":
        filename_base += "_sw"
    ext = ".json"
    filename = filename_base + ext
    file_path = os.path.join(result_out_dir, filename)

    version = 1
    while os.path.exists(file_path):
        filename = f"{filename_base}_v{version}{ext}"
        file_path = os.path.join(result_out_dir, filename)
        version += 1

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation_dataset",
        type=str,
        help="Which dataset to use for evaluation of finetuned models {ACDC, MnMs, MnMs2}",
    )
    parser.add_argument(
        "--finetuning_exp_name",
        type=str,
        help="Finetuned model experiment name. Should correspond to the exp_name used during finetuning",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Specify which split to use to evaluate the model (val or test)",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="best",
        help="Which checkpoint (last, best, intermediate?) to load",
    )
    parser.add_argument(
        "--inference_strategy",
        type=str,
        default="full_image",  # full_image or sliding_window
        help="Whether to infer on full image or using Sliding Window Inference (SWI). "
        "SWI is useful when using non-convolutional backbones (e.g., SwinUNETR) that don't support arbitrary input sizes",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,  # Sets the value to False if the flag is absent
        help="Whether to use post-processing (keep largest connected component)",
    )

    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    assert args.split in ("val", "test"), "loader should be either val or test sets"
    assert args.inference_strategy in ("full_image", "sliding_window"), (
        "inference_strategy should be either full_image or sliding_window"
    )
    logger.info(f"You choose to evaluate on {args.split} set")

    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed)

    device = setup_pytorch()
    finetuned_weights_path = f"output/lightning_logs/finetuning/cardiac/{args.finetuning_exp_name}/checkpoints/{args.ckpt_name}.ckpt"
    result_out_dir = f"output/evaluation_results/pix2repv2_{args.split}"

    if args.split == "val":
        if args.evaluation_dataset == "ACDC":
            acdc_sup_patch_ds = ACDCPatchSupervised(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
                apply_augmentations=False,
            )
            _, val_subset = create_train_val_subsets(
                patch_dataset=acdc_sup_patch_ds,
                cfg=cfg,
            )
            logger.info("ACDC validation dataloader construction")
            loader = build_patch_loader(
                patch_dataset=val_subset,
                batch_size=cfg.data.batch_size_pretraining,
                shuffle=False,
            )

        elif args.evaluation_dataset == "MnMs":
            mnms_val_sup_patch_ds = MnMsPatchSupervised(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
                apply_augmentations=False,
                split="validation",
            )
            logger.info("MnMs validation dataloader construction")
            loader = build_patch_loader(
                mnms_val_sup_patch_ds,
                batch_size=cfg.data.batch_size_pretraining,
                shuffle=False,
            )

        elif args.evaluation_dataset == "MnMs2":
            mnms2_val_sup_patch_ds = MnMs2PatchSupervised(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
                apply_augmentations=False,
                split="validation",
            )
            logger.info("MnMs2 validation dataloader construction")
            loader = build_patch_loader(
                mnms2_val_sup_patch_ds,
                batch_size=cfg.data.batch_size_pretraining,
                shuffle=False,
            )

        else:
            raise ValueError(
                f"Unknown dataset for evaluation: {args.evaluation_dataset}. "
                f"Should be {{ACDC, MnMs, MnMs2}}"
            )

    elif args.split == "test":
        # test loader (volumetric) = a single batch correspond to all slices
        # of a given patient's 3D volume at a given instant
        if args.evaluation_dataset == "ACDC":
            test_volume_ds = ACDCSupervisedEval(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
            )
            logger.info("ACDC test dataloader construction")
            loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

        elif args.evaluation_dataset == "MnMs":
            test_volume_ds = MnMsSupervisedEval(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
            )
            logger.info("MnMs test dataloader construction")
            loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

        elif args.evaluation_dataset == "MnMs2":
            test_volume_ds = MnMs2SupervisedEval(
                cfg=cfg,
                is_resampled=True,
                target_spacing=1.0,
            )
            logger.info("MnMs2 test dataloader construction")
            loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

        else:
            raise ValueError(
                f"Unknown dataset for evaluation: {args.evaluation_dataset}. "
                f"Should be {{ACDC, MnMs, MnMs2}}"
            )

    else:
        raise ValueError(f"Unknown split for evaluation: {args.split}")

    main(
        cfg=cfg,
        loader=loader,
        split=args.split,
        finetuned_weights_path=finetuned_weights_path,
        device=device,
        result_out_dir=result_out_dir,
        apply_postprocessing=args.apply_postprocessing,
        inference_strategy=args.inference_strategy,
    )
