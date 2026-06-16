import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchio as tio
from einops import rearrange
from lightning import seed_everything
from loguru import logger
from monai.inferers import sliding_window_inference

from pix2repv2.data.acdc import ACDCSupervisedEval, build_patch_loader
from pix2repv2.data.mnms import MnMsSupervisedEval
from pix2repv2.data.mnms2 import MnMs2SupervisedEval
from pix2repv2.finetuning_module import Pix2Repv2FinetuningModule
from pix2repv2.utils import utils
from pix2repv2.utils.utils import Config, setup_pytorch
from pix2repv2.utils.viz import pix2rep_prediction_figure


def main(
    cfg: dict,
    loader: torch.utils.data.DataLoader,
    finetuned_weights_path: str | os.PathLike,
    device: str | torch.device,
    out_fig_dir: str | os.PathLike,
    apply_postprocessing: bool,
    inference_strategy: str,
):
    """Plots segmentation predictions of a finetuned Pix2Repv2 model on the test set of a given dataset (ACDC, MnMs or MnMs2).

    Args:
        cfg (dict)
        loader (torch.utils.data.DataLoader): Evaluation dataloader (with batch_size=1, i.e. each batch is a full patient volume)
        finetuned_weights_path (str | os.PathLike): Path to the finetuned model checkpoint to load for inference
        device (str | torch.device)
        out_fig_dir (str | os.PathLike): Directory where to save output figures
        apply_postprocessing (bool): Whether to apply post-processing (keep largest connected component) to predicted masks
        inference_strategy (str): full_image for Pix2Rep-v2+U-Net / sliding_window for Pix2Rep-v2+SwinUNETR
    """
    finetuned_model = Pix2Repv2FinetuningModule.load_from_checkpoint(
        finetuned_weights_path,
        cfg=cfg,
        strict=True,
    )
    finetuned_model.eval().to(device)

    with torch.no_grad():
        for idx, (imgs, masks, pid, orig_spacing, actual_spacing_out) in enumerate(
            loader
        ):
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
            imgs = rearrange(imgs, "D C W H -> C W H D").to(device)
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
            # Resampling back probability map to initial spacing
            softmax, _ = utils.resample(
                softmax,
                spacing_in=actual_spacing_out,
                spacing_out=(sw, sh, sd),
                mode="bicubic",
            )  # shape: (C, W, H, D)

            # Resampling back images (mri_slices) to initial spacing
            imgs, _ = utils.resample(
                imgs,
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

            imgs = rearrange(imgs, "1 W H D -> W H D").cpu()
            predicted_masks = rearrange(predicted_masks, "1 W H D -> W H D").int()
            masks = rearrange(masks, "1 1 W H D -> W H D").int()

            assert predicted_masks.shape == masks.shape, (
                "Prediction and GT masks shape mismatch"
            )
            for slice in range(predicted_masks.shape[-1]):
                fig = pix2rep_prediction_figure(
                    imgs[..., slice],
                    predicted_masks[..., slice],
                    masks[..., slice],
                )
                pid_str = "".join(pid)
                out_dir = os.path.join(out_fig_dir, f"{pid_str}_{idx}")
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(out_dir, f"predicted_slice_{slice}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="ACDC",
        help="Which dataset to use for plotting segmentation predictions {ACDC, MnMs, MnMs2}",
    )
    parser.add_argument(
        "--finetuning_exp_name",
        type=str,
        help="Finetuned model experiment name to know which checkpoint to load for prediction",
    )
    parser.add_argument(
        "--checkpoint_name",
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

    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed)

    device = setup_pytorch()
    finetuned_weights_path = f"output/lightning_logs/finetuning/cardiac/{args.finetuning_exp_name}/checkpoints/{args.checkpoint_name}.ckpt"
    out_fig_dir = (
        f"output/predictions/cardiac/{args.dataset}/pix2rep_{args.finetuning_exp_name}"
    )

    # test loader (volumetric) = a single batch correspond to all slices
    # of a given patient's 3D volume at a given instant
    if args.dataset == "ACDC":
        test_volume_ds = ACDCSupervisedEval(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
        )
        logger.info("ACDC test dataloader construction")
        loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

    elif args.dataset == "MnMs":
        test_volume_ds = MnMsSupervisedEval(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
        )
        logger.info("MnMs test dataloader construction")
        loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

    elif args.dataset == "MnMs2":
        test_volume_ds = MnMs2SupervisedEval(
            cfg=cfg,
            is_resampled=True,
            target_spacing=1.0,
        )
        logger.info("MnMs2 test dataloader construction")
        loader = build_patch_loader(test_volume_ds, batch_size=1, shuffle=False)

    else:
        raise ValueError(
            f"Unknown dataset for prediction: {args.dataset}. "
            f"Should be {{ACDC, MnMs, MnMs2}}"
        )

    main(
        cfg=cfg,
        loader=loader,
        finetuned_weights_path=finetuned_weights_path,
        device=device,
        out_fig_dir=out_fig_dir,
        apply_postprocessing=args.apply_postprocessing,
        inference_strategy=args.inference_strategy,
    )
