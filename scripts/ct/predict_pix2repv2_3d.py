import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchio as tio
from einops import rearrange
from lightning import seed_everything
from loguru import logger
from monai.inferers import sliding_window_inference

from pix2repv2.data.amos import (
    AMOSSupervisedEval,
    build_amos_eval_pipeline,
    build_patch_loader,
)

from pix2repv2.finetuning_module_3d_implicit import Pix2Repv2_3DFinetuningModule
from pix2repv2.utils.utils import Config, setup_pytorch
from pix2repv2.utils.viz import pix2rep_prediction_figure_3d


def main(
    cfg: dict,
    loader: torch.utils.data.DataLoader,
    finetuned_weights_path: str | os.PathLike,
    device: str | torch.device,
    out_fig_dir: str | os.PathLike,
    apply_postprocessing: bool,
    inference_strategy: str,
):
    logger.info(f"Loading finetuned model from {finetuned_weights_path}")
    finetuned_model = Pix2Repv2_3DFinetuningModule.load_from_checkpoint(
        finetuned_weights_path,
        cfg=cfg,
        strict=True,
    )
    finetuned_model.eval().to(device)

    with torch.no_grad():
        for idx, (imgs, masks, pid) in enumerate(loader):
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

            imgs = rearrange(imgs, "1 1 D H W -> H W D").cpu()
            predicted_masks = rearrange(predicted_masks, "1 1 H W D -> H W D").int()
            masks = rearrange(masks, "1 1 H W D -> H W D").int()

            assert predicted_masks.shape == masks.shape, (
                "Prediction and GT masks shape mismatch"
            )
            for slice in range(0, predicted_masks.shape[-1], 10):
                fig = pix2rep_prediction_figure_3d(
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
            logger.info(f"Patient {pid} processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetuning_exp_name",
        type=str,
        help="Finetuned model experiment name",
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
    out_fig_dir = f"output/predictions/pix2repv2-3d/{args.finetuning_exp_name}"

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
        out_fig_dir=out_fig_dir,
        apply_postprocessing=args.apply_postprocessing,
        inference_strategy=args.inference_strategy,
    )
