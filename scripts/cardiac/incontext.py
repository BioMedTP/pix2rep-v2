import argparse
import json
import os
from datetime import datetime

import faiss
import faiss.contrib.torch_utils  # necessary import for FAISS to work with Pytorch tensors
import matplotlib.pyplot as plt
import monai.metrics
import numpy as np
import torch
from einops import rearrange
from lightning import seed_everything
from loguru import logger
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice

import pix2repv2.data.acdc as acdc
import pix2repv2.data.mnms as mnms
import pix2repv2.data.mnms2 as mnms2
from pix2repv2.data.acdc import ACDCSupervisedEval, build_patch_loader
from pix2repv2.data.mnms import MnMsSupervisedEval
from pix2repv2.data.mnms2 import MnMs2SupervisedEval
from pix2repv2.models.swin_unetr import SwinUNETR_SSL
from pix2repv2.pretraining_module import Pix2Repv2PretrainingModule
from pix2repv2.utils.utils import Config, setup_pytorch


def build_faiss_index(
    db_features: torch.Tensor | np.ndarray,
    nlist: int,
    nprobe: int,
    train_index: bool,
    index_name: str,
    use_gpu: bool = True,
) -> faiss.Index:
    """Build a FAISS index for in-context segmentation

    Args:
        db_features (torch.Tensor | np.ndarray): Database of features to index (need to be L2-normalized beforehand)
        nlist (int): Number of clusters for IVF
        nprobe (int): Number of probes for IVF
        use_gpu (bool, optional): Whether to use GPU for FAISS index. Defaults to True.
        train_index (bool, optional): Whether to train the index or load a pre-trained one. Defaults to True.
        index_name (str, optional): Name of the index file to save/load.

    Returns:
        faiss.Index: FAISS index object
    """
    N, d = db_features.shape

    if train_index:
        # index = faiss.index_factory(d, "IVF8192,Flat", faiss.METRIC_INNER_PRODUCT)
        index = faiss.index_factory(
            d,
            f"OPQ64_256,IVF{nlist},PQ64x8",
            faiss.METRIC_INNER_PRODUCT,
        )
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = nprobe
        clustering_index = faiss.IndexFlatIP(index_ivf.d)
        index_ivf.clustering_index = faiss.index_cpu_to_all_gpus(clustering_index)

        # As a rule of thumb there is no consistent improvement of the k-means quantizer beyond
        # 20 iterations and 1000 * k training points (https://github.com/facebookresearch/faiss/wiki/FAQ/effe69c8b999a8acc0e71ae07f88bff3c7d7d0f4)
        train_size = min(500_000, N)
        train_indices = np.random.choice(N, train_size, replace=False)
        index.train(db_features[train_indices].to("cpu"))
        assert index.is_trained

        index_out_path = os.path.join("output/faiss", index_name)
        os.makedirs(os.path.dirname(index_out_path), exist_ok=True)
        faiss.write_index(index, index_out_path)
        logger.info(f"FAISS index saved under {index_out_path}")
    else:
        index = faiss.read_index(os.path.join("output/faiss", index_name))

    if use_gpu:
        logger.info("Moving FAISS index to GPU")
        co = faiss.GpuMultipleClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co)

    return index


def build_bruteforce_faiss_index(
    db_features: torch.Tensor | np.ndarray,
    use_gpu: bool = True,
) -> faiss.Index:
    N, d = db_features.shape
    index = faiss.IndexFlatIP(d)
    assert index.is_trained

    logger.info("FAISS index built")

    if use_gpu:
        logger.info("Moving FAISS index to GPU")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index


def main(
    cfg: dict,
    query_ds: torch.utils.data.Dataset,
    support_loader: torch.utils.data.DataLoader,
    exp_name: str,
    pretrained_backbone: str,
    use_projection_head: int,
    k: int,
    device: str,
    dataset_name: str,
    plot: bool,
):
    pretrained_backbone_path = f"output/lightning_logs/pretraining/cardiac/{pretrained_backbone}/checkpoints/best.ckpt"
    logger.info(f"Loading pretrained backbone: {pretrained_backbone}")
    model = Pix2Repv2PretrainingModule.load_from_checkpoint(
        pretrained_backbone_path,
        cfg=cfg,
        strict=True,
    )
    model.eval().to(device)

    db_features = []
    support_gts = []

    for batch_s in iter(support_loader):
        support_img = rearrange(batch_s[0], "1 C W H D -> D C W H").to(device)
        support_gt = rearrange(batch_s[1], "1 1 W H D -> D W H")

        with torch.no_grad():
            if use_projection_head == 0:
                predictor = model.backbone
            elif use_projection_head == 1:
                predictor = model  # NOTE: backbone + projection head
            else:
                raise ValueError(
                    f"use_projection_head should be either 0 (for backbone features) or 1 (for projection head features), but got {use_projection_head}"
                )

            if isinstance(model.backbone, SwinUNETR_SSL):
                support_features = sliding_window_inference(
                    inputs=support_img,
                    roi_size=(128, 128),
                    sw_batch_size=16,  # need at least a V100-16GB
                    predictor=predictor,
                    overlap=0.5,
                    mode="constant",
                )
            else:
                support_features = predictor(support_img)  # shape : (D, C, W, H)

        # Move features to CPU immediately to avoid keeping the full database
        # on GPU memory. Only the current batch activations live on the GPU.
        support_features_cpu = support_features.detach().cpu()
        db_features.append(rearrange(support_features_cpu, "D C W H -> (W H D) C"))

        # Ensure support labels are kept on CPU as well
        support_gts.append(rearrange(support_gt, "D W H -> (W H D)"))

        del support_features, support_features_cpu, support_img, support_gt

    db_features = torch.cat(db_features, dim=0)  # shape: (n*(WHD), C)
    support_gts = torch.cat(support_gts, dim=0)  # shape: (n*(WHD),)

    N, d = db_features.shape
    logger.info(f"Database features shape: {db_features.shape}")
    if use_projection_head == 0:
        assert d == 1024, "Feature dimension must be 1024 for Pix2Rep backbone"
    elif use_projection_head == 1:
        assert d == 256, "Feature dimension must be 256 for Pix2Rep projection head"
    else:
        raise ValueError(
            f"use_projection_head should be either 0 (for backbone features) or 1 (for projection head features), but got {use_projection_head}"
        )

    if N <= 3_000_000:
        keep_ratio = 1.0
    elif 3_000_000 < N <= 10_000_000:
        keep_ratio = 0.75
    elif 10_000_000 < N <= 25_000_000:
        keep_ratio = 0.25
    else:
        keep_ratio = 0.125
    subset = np.random.choice(N, int(keep_ratio * N), replace=False)
    db_features = db_features[subset]
    support_gts = support_gts[subset]
    logger.info(
        f"Actually using features of shape: {db_features.shape} after subsampling with keep_ratio={keep_ratio:.2f}"
    )

    torch.nn.functional.normalize(
        db_features,
        p=2,
        dim=1,
        out=db_features,
    )

    # nlist value based on https://github.com/facebookresearch/faiss/issues/112
    # nlist = int(4 * np.sqrt(N))
    # nprobe = min(64, nlist)
    # index_name = f"{dataset_name}_index.faiss"
    # index = build_faiss_index(
    #     db_features,
    #     nlist=nlist,
    #     nprobe=nprobe,
    #     train_index=True,
    #     index_name=index_name,
    #     use_gpu=True,
    # )
    index = build_bruteforce_faiss_index(
        db_features,
        use_gpu=True,
    )
    index.add(db_features.cpu())

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

    for i, batch_q in enumerate(query_ds):
        img = batch_q[0]  # shape: (1, W, H, D)
        gt = batch_q[1].squeeze(0)  # GT seg of shape: (W, H, D)
        pid = batch_q[2]
        logger.info(f"Processing patient {pid}")

        query_img = rearrange(img, "C W H D -> D C W H").to(device)

        with torch.no_grad():
            # query_features = model.backbone(query_img)
            if use_projection_head == 0:
                predictor = model.backbone
            elif use_projection_head == 1:
                predictor = model
            else:
                raise ValueError(
                    f"use_projection_head should be either 0 (for backbone features) or 1 (for projection head features), but got {use_projection_head}"
                )

            if isinstance(model.backbone, SwinUNETR_SSL):
                query_features = sliding_window_inference(
                    inputs=query_img,
                    roi_size=(128, 128),
                    sw_batch_size=16,  # need at least a V100-16GB
                    predictor=predictor,
                    overlap=0.5,
                    mode="constant",
                )
            else:
                query_features = predictor(query_img)  # shape : (D, C, W, H)

        query_features = rearrange(query_features, "D C W H -> (W H) D C")
        torch.nn.functional.normalize(
            query_features,
            p=2,
            dim=2,
            out=query_features,
        )

        W, H, D = gt.shape
        pred_seg = torch.empty((W, H, D), device=gt.device, dtype=gt.dtype)

        for d in range(D):
            # Perform FAISS search directly from GPU query features to
            # avoid repeated CPU<->GPU transfers for each slice.
            SIM, IDX = index.search(query_features[:, d, :].cpu().contiguous(), k=k)
            pred_seg_flat = support_gts[IDX.cpu()]

            pred_seg_flat, _ = torch.mode(pred_seg_flat, dim=1)  # shape: (N,)
            pred_seg[:, :, d] = pred_seg_flat.view(W, H)

            if plot:
                logger.info(f"Visualizing slice d={d} from query patient {pid}")
                fig_folder_path = f"output/figs/incontext/{pretrained_backbone}/{dataset_name}/N={cfg.finetuning.num_patients}/k={k}/{pid}_{i}"
                os.makedirs(fig_folder_path, exist_ok=True)
                plt.imshow(pred_seg[:, :, d].swapaxes(0, 1), cmap="gray")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(fig_folder_path, f"incontext_seg_slice{d}.png")
                )
                plt.close()

        pred_seg = rearrange(pred_seg, "W H D -> 1 1 W H D")
        gt = rearrange(gt, "W H D -> 1 1 W H D")
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
            "num_patients_in_support_set": cfg.finetuning.num_patients,
        }
        file_path = (
            f"output/incontext/{pretrained_backbone}/{dataset_name}/{exp_name}/"
            + f"N={cfg.finetuning.num_patients}/k={k}/use_projection_head={use_projection_head}/{pid_str}_"
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
        "exp_name": exp_name,
        "model": pretrained_backbone_path,
        "mean_global_dice": float(mean_global_dice),
        "mean_per_class_dice": mean_per_class_dice,
        "top_k": k,
        "num_patients_in_support_set": cfg.finetuning.num_patients,
    }
    file_path = (
        f"output/incontext/{pretrained_backbone}/{dataset_name}/{exp_name}/"
        + f"N={cfg.finetuning.num_patients}/k={k}/use_projection_head={use_projection_head}_"
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
        "--support_dataset",
        type=str,
        default="ACDC",
        help="Which dataset to use for creating the support set {ACDC, MnMs, MnMs2}",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        required=True,
        help="Which pretrained backbone to load for extracting deep embeddings",
    )
    parser.add_argument(
        "--use_projection_head",
        type=int,
        default=1,
        help="Whether to use a projection head for in-context segmentation (instead of using backbone features directly). 0 means no projection head (D=1024), 1 means use projection head representations (d=256).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors to use for in-context segmentation assignment",
    )
    parser.add_argument(
        "--plot",
        action="store_true",  # Sets the value to True if the flag is present
        default=False,  # Sets the value to False if the flag is absent
        help="Whether to plot and save segmentation propagation figures",
    )
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()
    device = setup_pytorch()
    cfg = Config(overrides=args.overrides).cfg
    seed_everything(cfg.data.random_seed, workers=True)

    if args.support_dataset == "ACDC":
        query_ds = ACDCSupervisedEval(cfg=cfg, is_resampled=False)
        selected_patients = acdc.select_nested_patient_ids(cfg)
        support_set = ACDCSupervisedEval(
            cfg=cfg,
            is_resampled=False,
            split="training",
            selected_patients=selected_patients,
        )
        logger.info("ACDC support set dataloaders construction")
        support_loader = build_patch_loader(
            patch_dataset=support_set,
            batch_size=1,
            shuffle=False,
        )

    elif args.support_dataset == "MnMs":
        query_ds = MnMsSupervisedEval(cfg=cfg, is_resampled=False)
        selected_patients = mnms.select_nested_patient_ids(cfg)
        support_set = MnMsSupervisedEval(
            cfg=cfg,
            is_resampled=False,
            split="training",
            selected_patients=selected_patients,
        )
        logger.info("MnMs support set dataloaders construction")
        support_loader = build_patch_loader(
            support_set,
            batch_size=1,
            shuffle=False,
        )

    elif args.support_dataset == "MnMs2":
        query_ds = MnMs2SupervisedEval(cfg=cfg, is_resampled=False)
        selected_patients = mnms2.select_nested_patient_ids(cfg)
        support_set = MnMs2SupervisedEval(
            cfg=cfg,
            is_resampled=False,
            split="training",
            selected_patients=selected_patients,
        )
        logger.info("MnMs2 support set dataloaders construction")
        support_loader = build_patch_loader(
            support_set,
            batch_size=1,
            shuffle=False,
        )

    else:
        raise ValueError(
            f"Unknown dataset for finetuning: {args.support_dataset}. "
            f"Should be {{ACDC, MnMs, MnMs2}}"
        )

    main(
        cfg=cfg,
        query_ds=query_ds,
        support_loader=support_loader,
        exp_name=args.exp_name,
        pretrained_backbone=args.backbone_name,
        use_projection_head=args.use_projection_head,
        k=args.k,
        device=device,
        dataset_name=args.support_dataset,
        plot=args.plot,
    )
