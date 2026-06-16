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

from pix2repv2.data.amos import (
    AMOSSupervisedEval,
    build_amos_eval_pipeline,
    build_patch_loader,
)
from pix2repv2.finetuning_module_3d_implicit import Pix2Repv2_3DFinetuningModule
from pix2repv2.utils.utils import Config, setup_pytorch


def main(
    cfg: dict,
    loader: torch.utils.data.DataLoader,
    finetuned_weights_path: str | os.PathLike,
    device: str | torch.device,
    result_out_dir: str | os.PathLike,
    apply_postprocessing: bool,
    inference_strategy: str,
    verbose: bool = True,
):
    logger.info(f"Loading finetuned model from {finetuned_weights_path}")
    finetuned_model = Pix2Repv2_3DFinetuningModule.load_from_checkpoint(
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

    with torch.no_grad():
        for i, (imgs, masks, pid) in enumerate(loader):
            imgs = rearrange(imgs, "1 C H W D -> 1 C D H W").to(device)
            masks = masks.int().to(device)  # shape (B=1, C=1, H, W, D)

            if inference_strategy == "full_image":
                logits = finetuned_model(imgs)
            else:
                # Sliding window inference from 128x128 patches
                logits = sliding_window_inference(
                    inputs=imgs,
                    roi_size=tuple(
                        cfg.data.ROI_SW_SIZE
                    ),  # NOTE: MONAI expects roi_size (D, H, W) in case input is 3D
                    sw_batch_size=1,
                    predictor=finetuned_model,
                    overlap=0.5,
                    mode="gaussian",
                    progress=True,
                )
            logits = rearrange(logits, "1 C D H W -> 1 C H W D")

            # Transform logits to label predictions
            predicted_masks = torch.argmax(logits, dim=1, keepdim=True)

            if apply_postprocessing:
                subject = tio.Subject(
                    predicted_mask=tio.LabelMap(tensor=predicted_masks.detach().cpu())
                )
                subject = tio.transforms.KeepLargestComponent()(subject)
                keep_lcc = subject["predicted_mask"].data.to(device)
                predicted_masks = predicted_masks * keep_lcc

            assert predicted_masks.shape == masks.shape
            global_dice(predicted_masks, masks)
            per_class_dice(predicted_masks, masks)

            # TODO: Fix that because not optimal
            # Re-compute Dice to store per-patient Dice scores
            patient_dice = compute_dice(
                predicted_masks,
                masks,
                include_background=False,
                num_classes=cfg.finetuning.n_classes,
            )
            per_patient_dice_scores[f"{pid}_{i}"] = patient_dice.cpu().tolist()
            if verbose:
                logger.info(f"Processed patient {pid}")

        # Aggregate over all batches
        mean_global_dice = global_dice.aggregate().item()  # scalar
        mean_per_class_dice = per_class_dice.aggregate()  # [num_classes] vector

    results = {
        "date": str(datetime.now()),
        "model": finetuned_weights_path,
        "mean_global_dice": float(mean_global_dice),
        "per_class_global_dice": mean_per_class_dice,
        "n_batches": len(loader),
        "batch_size": loader.batch_size,
        "post_processing": apply_postprocessing,
        "inference_strategy": inference_strategy,
        "per_patient_dice_scores": per_patient_dice_scores,  # detail per patient
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
        "--finetuning_exp_name",
        type=str,
        help="Finetuned model experiment name. Should correspond to the exp_name used during finetuning",
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
        # default="full_image",  # full_image or sliding_window
        default="sliding_window",  # full_image or sliding_window
        help="Whether to infer on full image or using sliding patches window",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,  # Sets the value to False if the flag is absent
        help="Whether to use post-processing (keep largest connected component)",
    )

    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    assert args.inference_strategy in ("full_image", "sliding_window"), (
        "inference_strategy should be either full_image or sliding_window"
    )

    cfg = Config(config_name="config3D", overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed)

    device = setup_pytorch()
    finetuned_weights_path = f"output/lightning_logs/finetuning/ct/{args.finetuning_exp_name}/checkpoints/{args.ckpt_name}.ckpt"
    result_out_dir = "output/evaluation_results/pix2repv2-3d/"

    pipeline = build_amos_eval_pipeline()
    test_volume_ds = AMOSSupervisedEval(
        cfg=cfg,
        pipeline=pipeline,
    )
    logger.info("AMOS evaluation dataloader construction")
    loader = build_patch_loader(
        test_volume_ds,
        batch_size=1,
        shuffle=False,
        prefetch_factor=2,
    )

    main(
        cfg=cfg,
        loader=loader,
        finetuned_weights_path=finetuned_weights_path,
        device=device,
        result_out_dir=result_out_dir,
        apply_postprocessing=args.apply_postprocessing,
        inference_strategy=args.inference_strategy,
    )
