import argparse
import json
import os
from datetime import datetime

import faiss
import faiss.contrib.torch_utils  # type: ignore (necessary import for FAISS to work with Pytorch tensors)
import matplotlib.pyplot as plt
import monai.metrics
import torch
from einops import rearrange
from loguru import logger
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice

import pix2repv2.utils.utils as utils
from pix2repv2.data.acdc import ACDCSupervised4D, build_patch_loader
from pix2repv2.data.mnms import MnMsSupervised4D
from pix2repv2.data.mnms2 import MnMs2Supervised4D
from pix2repv2.models.swin_unetr import SwinUNETR_SSL
from pix2repv2.pretraining_module import Pix2Repv2PretrainingModule
from pix2repv2.utils.utils import Config, setup_pytorch


def extract_embeddings(
    mr_volume: torch.Tensor,
    model: Pix2Repv2PretrainingModule,
    use_proj_head_feats: int,
    is_resampled: bool,
    spacing_in: float | tuple[float, float, float],
    spacing_out: float | tuple[float, float, float],
    device: str,
) -> torch.Tensor:
    """Extract embeddings of a 3D volume at time t and optionally resample to original spacing."""

    mr_volume = rearrange(mr_volume, "C W H D -> D C W H").to(device)

    with torch.no_grad():
        if use_proj_head_feats == 0:
            predictor = model.backbone
        elif use_proj_head_feats == 1:
            predictor = model  # NOTE: backbone + projection head
        else:
            raise ValueError(
                f"use_proj_head_feats should be either 0 (for backbone features) or 1 (for projection head features), but got {use_proj_head_feats}"
            )

        # NOTE: if you experience OOM issues, you can use SWI only to avoid inferring on the whole volume at once.
        if isinstance(model.backbone, SwinUNETR_SSL):
            deep_embs = sliding_window_inference(
                inputs=mr_volume,
                roi_size=(128, 128),
                sw_batch_size=16,  # need at least a V100-16GB
                predictor=predictor,
                overlap=0.5,
                mode="constant",
            )
        else:
            deep_embs = predictor(mr_volume)

    if is_resampled:
        deep_embs = utils.resample(
            rearrange(deep_embs, "D C W H -> C W H D"),
            spacing_in=spacing_in,
            spacing_out=spacing_out,
            mode="bicubic",
        )
        deep_embs = rearrange(deep_embs, "C W H D -> D C W H")

    return deep_embs


def search_and_assign(
    prev_features: torch.Tensor,
    curr_features: torch.Tensor,
    anchor_features: torch.Tensor | None,
    prev_soft_seg: torch.Tensor,
    anchor_soft_seg: torch.Tensor | None,
    k: int,
    gpu_index_flat: faiss.GpuIndexFlatIP,
    use_distance_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate segmentation between two timesteps using FAISS search and aggregation of soft labels."""

    if anchor_features is not None and not torch.equal(prev_features, anchor_features):
        prev_features = torch.cat([prev_features, anchor_features], dim=0)
        prev_soft_seg = torch.cat([prev_soft_seg, anchor_soft_seg], dim=0)

    torch.nn.functional.normalize(prev_features, p=2, dim=1, out=prev_features)
    torch.nn.functional.normalize(curr_features, p=2, dim=1, out=curr_features)

    gpu_index_flat.reset()
    gpu_index_flat.add(prev_features)

    SIM, IDX = gpu_index_flat.search(curr_features, k)
    nearest_indices = IDX.to(prev_soft_seg.device)

    neighbor_soft = prev_soft_seg[nearest_indices]  # (N, k, C)

    if use_distance_weights:
        weights = torch.softmax(SIM, dim=1).unsqueeze(-1).cpu()  # (N, k, 1)
        curr_soft_seg = (neighbor_soft * weights).sum(dim=1)
    else:
        curr_soft_seg = neighbor_soft.mean(dim=1)

    curr_hard_seg = curr_soft_seg.argmax(dim=1)
    return curr_soft_seg, curr_hard_seg


def propagate_one_timestep(
    prev_deep_embs_volume: torch.Tensor,
    curr_deep_embs_volume: torch.Tensor,
    anchor_deep_embs_volume: torch.Tensor | None,
    prev_soft_seg: torch.Tensor,
    anchor_soft_seg: torch.Tensor | None,
    k: int,
    gpu_index_flat: faiss.GpuIndexFlatIP,
    num_classes: int,
    use_distance_weights: bool = False,
    slice_wise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate segmentation for one timestep across slices (slice-wise) or across the full volume (volume-wise)."""

    D, _, W, H = curr_deep_embs_volume.shape

    if slice_wise:
        curr_soft_seg = torch.zeros(W, H, D, num_classes)
        curr_hard_seg = torch.zeros(W, H, D, dtype=torch.long)

        for d in range(D):
            prev_slice_features = rearrange(
                prev_deep_embs_volume[d], "C W H -> (W H) C"
            ).contiguous()
            curr_slice_features = rearrange(
                curr_deep_embs_volume[d], "C W H -> (W H) C"
            ).contiguous()

            if anchor_deep_embs_volume is not None:
                anchor_features = rearrange(
                    anchor_deep_embs_volume[d], "C W H -> (W H) C"
                ).contiguous()
            else:
                anchor_features = None

            prev_soft_flat = prev_soft_seg[:, :, d].reshape(-1, num_classes)

            if anchor_soft_seg is not None:
                anchor_soft_flat = anchor_soft_seg[:, :, d].reshape(-1, num_classes)
            else:
                anchor_soft_flat = None

            curr_soft_slice, curr_hard_slice = search_and_assign(
                prev_slice_features,
                curr_slice_features,
                anchor_features,
                prev_soft_flat,
                anchor_soft_flat,
                k=k,
                gpu_index_flat=gpu_index_flat,
                use_distance_weights=use_distance_weights,
            )

            curr_soft_seg[:, :, d] = curr_soft_slice.reshape(W, H, num_classes)
            curr_hard_seg[:, :, d] = curr_hard_slice.reshape(W, H)

    else:
        prev_features = rearrange(
            prev_deep_embs_volume, "D C W H -> (W H D) C"
        ).contiguous()
        curr_features = rearrange(
            curr_deep_embs_volume, "D C W H -> (W H D) C"
        ).contiguous()

        if anchor_deep_embs_volume is not None:
            anchor_features = rearrange(
                anchor_deep_embs_volume, "D C W H -> (W H D) C"
            ).contiguous()
        else:
            anchor_features = None

        prev_soft_flat = prev_soft_seg.reshape(-1, num_classes)
        if anchor_soft_seg is not None:
            anchor_soft_flat = anchor_soft_seg.reshape(-1, num_classes)
        else:
            anchor_soft_flat = None

        curr_soft_flat, curr_hard_flat = search_and_assign(
            prev_features,
            curr_features,
            anchor_features,
            prev_soft_flat,
            anchor_soft_flat,
            k=k,
            gpu_index_flat=gpu_index_flat,
            use_distance_weights=use_distance_weights,
        )

        curr_soft_seg = curr_soft_flat.reshape(W, H, D, num_classes)
        curr_hard_seg = curr_hard_flat.reshape(W, H, D)

    return curr_soft_seg, curr_hard_seg


def plot_slice(seg_slice: torch.Tensor, t: int, d: int, fig_folder_path: str):
    os.makedirs(fig_folder_path, exist_ok=True)
    plt.imshow(seg_slice.detach().cpu().swapaxes(0, 1), cmap="gray")
    plt.title(f"Propagated Segmentation at t={t}, slice d={d}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder_path, f"propagated_seg_t{t}_d{d}.png"))
    plt.close()


def main(
    cfg: Config,
    loader: torch.utils.data.DataLoader,
    exp_name: str,
    es_to_ed: int,
    slice_wise: bool,
    use_proj_head_feats: int,
    first_frame_anchor: bool,
    use_distance_weights: bool,
    is_resampled: bool,
    k: int,
    pretrained_backbone: str,
    device: str,
    plot_seg_propagation: bool = False,
):
    """Zero-shot 3D+t segmentation propagation from ED->ES or ES->ED using a pretrained Pix2Repv2 backbone.

    Args:
        cfg (Config): Config dict.
        loader (torch.utils.data.DataLoader): DataLoader for the 3D+t volumes.
        es_to_ed (bool): In which direction to propagate the segmentation.
        slice_wise (bool): Whether to perform slice-wise (i.e., in 2D) propagation.
        first_frame_anchor (bool): Whether to use the first frame as an anchor for propagation.
        use_distance_weights (bool): Whether to weight the neighbor contributions based on similarity scores.
        is_resampled (bool): Whether the 3D+t volume has been resampled in the dataset.
        k (int): Number of nearest neighbors to use during similarity search.
        pretrained_backbone (str): Name of the pretrained backbone to use.
        device (str): Device to run the model on (e.g., "cuda" or "cpu").
        plot_seg_propagation (bool): Whether to plot and save (intermediate) segmentation propagation figures.
    """
    pretrained_backbone_path = f"output/lightning_logs/pretraining/cardiac/{pretrained_backbone}/checkpoints/best.ckpt"
    logger.info(f"Loading pretrained backbone: {pretrained_backbone}")
    model = Pix2Repv2PretrainingModule.load_from_checkpoint(
        pretrained_backbone_path,
        cfg=cfg,
        strict=True,
    )
    model.eval().to(device)

    num_classes = cfg.finetuning.n_classes

    logger.info("Building FAISS index for segmentation propagation...")
    res = faiss.StandardGpuResources()

    if use_proj_head_feats == 0:
        logger.info("Using representation features")
        d = 1024
    elif use_proj_head_feats == 1:
        logger.info("Using representation features")
        d = 256
    else:
        raise ValueError(
            f"use_proj_head_feats should be either 0 (for backbone features) or 1 (for projection head features), but got {use_proj_head_feats}"
        )

    gpu_index_flat = faiss.GpuIndexFlatIP(res, d)  # build the index directly on GPU
    # assert model.backbone.n_feure_maps == 1024
    logger.info("FAISS index built.")
    logger.info(f"Is FAISS index trained? {gpu_index_flat.is_trained}")

    global_dice = monai.metrics.DiceMetric(
        include_background=False,
        reduction="mean",
        num_classes=cfg.finetuning.n_classes,
    )
    per_class_dice = monai.metrics.DiceMetric(
        include_background=True,
        reduction="mean_batch",
        num_classes=cfg.finetuning.n_classes,
        return_with_label=True,
    )

    for i, batch in enumerate(loader):
        mr_4d_volume = batch[0].squeeze(0)  # shape: (W, H, D, T)
        ed_gt_volume = batch[1].squeeze(0)  # GT seg tensor at ED. shape: (W, H, D)
        es_gt_volume = batch[2].squeeze(0)  # GT seg tensor at ES. shape: (W, H, D)
        pid = batch[4][0]
        orig_spacing = batch[5]
        logger.info(f"Processing patient {pid}")

        # Input spacing of the 4D images (resampled in the dataset __getitem__)
        in_spacing = (
            loader.dataset.target_spacing,
            loader.dataset.target_spacing,
            orig_spacing[-1],
        )

        if es_to_ed == 0:
            logger.info("Propagating segmentation from ED -> ES...")
            start_t = 0  # NOTE: 3D+t volumes are already processed in the dataset __getitem__ to only include frames between ED and ES
            end_t = mr_4d_volume.shape[-1]  # propagate until the last time step (ES)
            step = 1
            start_gt_volume = ed_gt_volume
            end_gt_volume = es_gt_volume
            seg_direction = "ED_to_ES"
        elif es_to_ed == 1:
            logger.info("Propagating segmentation from ES -> ED...")
            start_t = mr_4d_volume.shape[-1] - 1
            end_t = -1  # 3D+t volumes are already cropped in the dataset __getitem__ to only include frames between ED and ES
            step = -1
            start_gt_volume = es_gt_volume
            end_gt_volume = ed_gt_volume
            seg_direction = "ES_to_ED"
        else:
            raise ValueError(
                f"es_to_ed should be either 0 (for ED -> ES) or 1 (for ES -> ED), but got {es_to_ed}"
            )

        start_gt_soft = torch.nn.functional.one_hot(
            start_gt_volume.long(), num_classes
        ).float()

        prev_deep_embs_volume = extract_embeddings(
            mr_4d_volume[..., start_t],
            model=model,
            use_proj_head_feats=use_proj_head_feats,
            is_resampled=is_resampled,
            spacing_in=in_spacing,
            spacing_out=orig_spacing,
            device=device,
        )

        if first_frame_anchor:
            anchor_deep_embs_volume = prev_deep_embs_volume
            anchor_soft_seg = start_gt_soft.clone()
        else:
            anchor_deep_embs_volume = None
            anchor_soft_seg = None

        for t in range(start_t + step, end_t, step):
            curr_deep_embs_volume = extract_embeddings(
                mr_4d_volume[..., t],
                model=model,
                use_proj_head_feats=use_proj_head_feats,
                is_resampled=is_resampled,
                spacing_in=in_spacing,
                spacing_out=orig_spacing,
                device=device,
            )

            curr_soft_seg, curr_hard_seg = propagate_one_timestep(
                prev_deep_embs_volume,
                curr_deep_embs_volume,
                anchor_deep_embs_volume,
                prev_soft_seg=start_gt_soft,
                anchor_soft_seg=anchor_soft_seg,
                k=k,
                gpu_index_flat=gpu_index_flat,
                num_classes=num_classes,
                use_distance_weights=use_distance_weights,
                slice_wise=slice_wise,
            )

            # Update for next time step
            start_gt_soft = curr_soft_seg
            prev_deep_embs_volume = curr_deep_embs_volume
            logger.info(f"Propagated segmentation to time step t={t}")

            if plot_seg_propagation:
                fig_folder_path = f"output/figs/segprop/{pretrained_backbone}/k={k}/{pid}/{seg_direction}"
                for d in range(curr_hard_seg.shape[2]):
                    logger.info(f"Visualizing slice d={d} at time t={t}")
                    plot_slice(curr_hard_seg[:, :, d], t, d, fig_folder_path)

        pred_seg = rearrange(curr_hard_seg, "W H D -> 1 1 W H D")
        gt = rearrange(end_gt_volume, "W H D -> 1 1 W H D")
        global_dice(pred_seg, gt)
        per_class_dice(pred_seg, gt)

        patient_dice = compute_dice(
            pred_seg,
            gt,
            include_background=False,
            num_classes=cfg.finetuning.n_classes,
        )
        pid_str = "".join(pid)
        results = {
            "date": str(datetime.now()),
            "exp_name": exp_name,
            "model": pretrained_backbone_path,
            "patient_dice": patient_dice.cpu().tolist(),
            "top_k": k,
            "es_to_ed": es_to_ed,
            "slice_wise": slice_wise,
            "repr_dim": d,
            "use_proj_head_feats": use_proj_head_feats,
            "first_frame_anchor": first_frame_anchor,
            "use_distance_weights": use_distance_weights,
            "num_patients_in_support_set": cfg.finetuning.num_patients,
        }
        file_path = (
            f"output/segprop/{pretrained_backbone}/{exp_name}/k={k}/{seg_direction}/{pid_str}_"
            + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            + ".json"
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=4)

    # Aggregate over all batches
    mean_global_dice = global_dice.aggregate().item()  # scalar
    mean_per_class_dice = per_class_dice.aggregate()  # [num_classes] vector
    results = {
        "date": str(datetime.now()),
        "model": pretrained_backbone_path,
        "use_proj_head_feats": use_proj_head_feats,
        "repr_dim": d,
        "mean_global_dice": mean_global_dice,
        "mean_per_class_dice": mean_per_class_dice,
        "es_to_ed": es_to_ed,
        "k": k,
        "is_resampled": is_resampled,
        "slice_wise": slice_wise,
        "first_frame_anchor": first_frame_anchor,
        "use_distance_weights": use_distance_weights,
    }
    file_path = (
        f"output/segprop/{pretrained_backbone}/{exp_name}/k={k}/{seg_direction}/segprop_results_"
        + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        + ".json"
    )
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        required=True,
        help="Which pretrained backbone to load for extracting deep embeddings",
    )
    parser.add_argument(
        "--es_to_ed",
        type=int,
        default=0,
        help="Whether to propagate segmentation from ES -> ED (1) or ED -> ES (0)",
    )
    parser.add_argument(
        "--slice_wise",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,
        help="Whether to propagate segmentation slice-wise (i.e., in 2D)",
    )
    parser.add_argument(
        "--first_frame_anchor",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,
        help="Whether to use the first frame as an anchor for segmentation propagation",
    )
    parser.add_argument(
        "--use_distance_weights",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,
        help="Whether to weight the neighbor contributions based on similarity scores",
    )
    parser.add_argument(
        "--is_resampled",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,
        help="Whether to resampled the 4D volume in the dataset",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to use for segmentation propagation",
    )
    parser.add_argument(
        "--use_proj_head_feats",
        type=int,
        default=1,
        help="Whether to use a projection head for in-context segmentation (instead of using backbone features directly). 0 means no projection head (D=1024), 1 means use projection head representations (d=256).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,  # Sets the value to False if the flag is absent
        help="Whether to plot and save segmentation propagation figures",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    cfg = Config(overrides=args.overrides).cfg
    device = setup_pytorch()

    if args.dataset == "ACDC":
        ds_4d_volume = ACDCSupervised4D(is_resampled=args.is_resampled, split="testing")
        loader = build_patch_loader(
            patch_dataset=ds_4d_volume, batch_size=1, shuffle=False
        )
    elif args.dataset == "MnMs":
        ds_4d_volume = MnMsSupervised4D(is_resampled=args.is_resampled, split="testing")
        loader = build_patch_loader(
            patch_dataset=ds_4d_volume, batch_size=1, shuffle=False
        )
    elif args.dataset == "MnMs2":
        ds_4d_volume = MnMs2Supervised4D(
            is_resampled=args.is_resampled, split="testing"
        )
        loader = build_patch_loader(
            patch_dataset=ds_4d_volume, batch_size=1, shuffle=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    main(
        cfg=cfg,
        loader=loader,
        exp_name=args.exp_name,
        es_to_ed=args.es_to_ed,
        slice_wise=args.slice_wise,
        use_proj_head_feats=args.use_proj_head_feats,
        first_frame_anchor=args.first_frame_anchor,
        use_distance_weights=args.use_distance_weights,
        is_resampled=args.is_resampled,
        k=args.k,
        pretrained_backbone=args.backbone_name,
        device=device,
        plot_seg_propagation=args.plot,
    )
