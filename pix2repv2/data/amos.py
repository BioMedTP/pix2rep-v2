import math
import os
import random

import numpy as np
import torch
import torchio as tio
from amid.amos import AMOS
from connectome import Apply, CacheToDisk, Chain, Filter, Transform
from einops import rearrange
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from pix2repv2.data.acdc import ValidationWrapper
from pix2repv2.utils import utils
from pix2repv2.utils.augmentations import (
    CropToBox,
    FlipAxesToCanonical,
    RandomBlurOrSharpen,
    RandomInvert,
    RescaleToSpacing,
    get_body_mask,
    hu_clip,
    mask_to_bbox,
)
from pix2repv2.utils.default_params import (
    BODY_THRESHOLD_HU,
    PATCH_SIZE,
    SPACING,
    WINDOW_HU,
)


class AMOSPatchSSL(Dataset):
    """AMOS dataset for pretraining Pix2Rep-v2 on 3D applications (multi-organ abodominal CT scans).
    It consists of 3D (cubic) patches of size (C=1, H=192, W=192, D=64) which are augmented with random
    photometric and geometric augmentations. The two augmented views of the same patch are returned
    together with the dense grid that will be used to warp the second 3D view and the first view's local features (cf. training_step()).
    Points to sample (and their warped version) are also returned to evaluate the loss at these locations (cf. Pix2Rep-v2+Implicit-UNet).
    """

    def __init__(
        self,
        cfg: dict,
        pipeline: Chain,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = False,
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.ids = pipeline.ids
        self.load_example = self.pipeline._compile(["image", "body_voxels"])
        self.min_patch_size_factor = self.cfg.pretraining.min_patch_size_factor
        self.max_patch_size_factor = self.cfg.pretraining.max_patch_size_factor
        self.apply_elastic_transform = apply_elastic_transform
        self.apply_augmentations = apply_augmentations
        self.resizer = tio.Resize(
            target_shape=tuple(self.cfg.data.PATCH_SIZE),  # shape: (W, H, D)
            image_interpolation="linear",
        )

        self.pretraining_transform = tio.Compose(
            [
                RandomBlurOrSharpen(
                    p=self.cfg.pretraining.pretraining_transform.random_blur_or_sharpen_p,
                    std=tuple(
                        self.cfg.pretraining.pretraining_transform.random_blur_std
                    ),
                    alpha=tuple(
                        self.cfg.pretraining.pretraining_transform.random_blur_alpha
                    ),
                ),
                tio.RandomGamma(
                    log_gamma=self.cfg.pretraining.pretraining_transform.log_gamma,
                    p=self.cfg.pretraining.pretraining_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=self.cfg.pretraining.pretraining_transform.random_noise_std,
                    p=self.cfg.pretraining.pretraining_transform.random_noise_p,
                ),
                RandomInvert(
                    p=self.cfg.pretraining.pretraining_transform.random_invert_p
                ),
            ]
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_id = self.ids[idx]
        image, body_voxels = self.load_example(item_id)

        image = torch.from_numpy(image).to(torch.float32)
        body_voxels = torch.from_numpy(body_voxels).to(torch.uint8)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=image.unsqueeze(0)),
            label=tio.LabelMap(tensor=body_voxels.unsqueeze(0)),
        )  # shape (C=1, W, H, D)

        w, h, d = subject["image"].spatial_shape
        patch_size_factor = np.random.uniform(
            self.min_patch_size_factor,
            self.max_patch_size_factor,
        )
        patch_size = min(w, h, d)

        sampler = tio.data.LabelSampler(
            patch_size=(
                round(patch_size * patch_size_factor),
                round(patch_size * patch_size_factor),
                round(patch_size * patch_size_factor),
            ),
            label_name="label",
            label_probabilities={0: 0, 1: 1},
        )

        patch = next(sampler._generate_patches(subject, num_patches=1))
        patch = self.resizer(patch)  # shape: (C=1, W=192, H=192, D=64)

        # Apply photometric augmentations yielding two different 3D patches
        if self.apply_augmentations:
            view1_3d = self.pretraining_transform(patch["image"].data)
            view2_3d = self.pretraining_transform(patch["image"].data)
        else:
            view1_3d = patch["image"].data
            view2_3d = patch["image"].data

        # NOTE: now the two views are in dimension (C=1, H, W, D) which will make life easier later on
        view1_3d = torch.transpose(view1_3d, 2, 1)
        view2_3d = torch.transpose(view2_3d, 2, 1)

        assert view1_3d.shape == view2_3d.shape
        C, H, W, D = view1_3d.shape

        # Rigid geometric transforms for images and feature maps (equivariance)
        affine_matrix = utils.generate_single_affine_spatial_transform_3d(
            p_rot=1,  # always do some rotations in pretraining
            p_flip=1,  # always do some flip in pretraining
            is_rotated=self.cfg.pretraining.is_rotated,
            is_cropped=self.cfg.pretraining.is_cropped,
            is_flipped=self.cfg.pretraining.is_flipped,
            max_angle=self.cfg.pretraining.max_angle,
            max_crop=self.cfg.pretraining.max_crop,
            max_tilt_deg=self.cfg.pretraining.max_tilt_deg,
        )  # shape: (3, 4) affine 3D matrix

        if self.apply_elastic_transform:
            with torch.no_grad():
                # NOTE: Elastic transforms not totally functional in 3D for the moment. Hence, put apply_elastic_transform=False for now.
                # grid that will be applied for the geometric (rigid + elastic) transforms
                dense_grid_3d = utils.generate_random_affine_elastic_grid_3d(
                    p=self.cfg.pretraining.pretraining_transform.elastic_transform_p_3d,
                    H=H,
                    W=W,
                    D=D,
                    affine=affine_matrix,
                )  # shape: (1, D, H, W, 3)
        else:
            dense_grid_3d = F.affine_grid(
                affine_matrix.unsqueeze(0),
                torch.Size((1, C, D, H, W)),
                align_corners=True,
            )  # shape: (1, D, H, W, 3)

        # Number of points to sample per 3D patch (2^17 ~ 131k)
        Npoints = self.cfg.pretraining.n_points

        # sparse_grid will later be applied to view2_3d to get the local features at these sampled points
        # Pytorch coordinates are in [-1, 1]
        sparse_grid = torch.rand(1, 1, Npoints, 3) * 2 - 1
        sparse_grid = sparse_grid.view(1, 1, 1, Npoints, 3)
        sparse_grid_warped = self._apply_affine_to_sparse_grid(
            sparse_grid, affine_matrix
        )
        sparse_grid_warped = sparse_grid_warped.view(1, 1, 1, Npoints, 3)

        return view1_3d, view2_3d, dense_grid_3d, sparse_grid, sparse_grid_warped

    def _apply_affine_to_sparse_grid(
        self,
        sparse_grid: torch.Tensor,
        affine_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        sparse_grid: (1, 1, 1, N, 3) in normalized (Pytorch) coordinates [-1, 1]
        affine_matrix: (3, 4)

        Returns:
            points_transformed: (N, 3)
        """
        N = sparse_grid.shape[-2]

        points = sparse_grid.view(N, 3)

        ones = torch.ones(N, 1, device=points.device, dtype=points.dtype)
        points_h = torch.cat([points, ones], dim=1)  # homogenous coordinates (N, 4)

        points_transformed = torch.matmul(points_h, affine_matrix.T)  # (N, 3)

        return points_transformed


class AMOSPatchSupervised(Dataset):
    """Dataset used for fine-tuning Pix2Rep-v2 for 3D multi-organ abdomical segmentation on AMOS.
    The patch sampling strategy is the following: sample 50% of background patches and 50% of foreground patches.
    The patch_size is similar to the pretraining patch_size (C=1, H=192, W=192, D=64)
    (3D volumes are already resampled during the dataset creation pipeline, cf. below).
    Random photometric and geometric augmentations are applied to the patches.
    """

    def __init__(
        self,
        cfg: dict,
        pipeline: Chain,
        apply_augmentations: bool = True,
        apply_geometric_transform: bool = True,
        selected_patients: list[int] | None = None,
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.load_example = self.pipeline._compile(["image", "mask"])
        self.apply_augmentations = apply_augmentations
        self.apply_geometric_transform = apply_geometric_transform
        self.selected_patients = selected_patients

        self.ids = (
            self.pipeline.ids
            if self.selected_patients is None
            else self.selected_patients
        )

        self.supervised_transform = tio.Compose(
            [
                RandomBlurOrSharpen(
                    p=self.cfg.finetuning.finetuning_transform.random_blur_or_sharpen_p,
                    std=tuple(self.cfg.finetuning.finetuning_transform.random_blur_std),
                    alpha=tuple(
                        self.cfg.finetuning.finetuning_transform.random_blur_alpha
                    ),
                ),
                tio.RandomGamma(
                    log_gamma=self.cfg.finetuning.finetuning_transform.log_gamma,
                    p=self.cfg.finetuning.finetuning_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=self.cfg.finetuning.finetuning_transform.random_noise_std,
                    p=self.cfg.finetuning.finetuning_transform.random_noise_p,
                ),
            ]
        )

        self.label_sampler = tio.data.LabelSampler(
            patch_size=tuple(self.cfg.data.PATCH_SIZE),
            label_name="label",
            label_probabilities={
                0: 0,
                1: 1,
                2: 1,
                3: 1,
                4: 1,
                5: 1,
                6: 1,
                7: 1,
                8: 1,
                9: 1,
                10: 1,
                11: 1,
                12: 1,
                13: 1,
                14: 1,
                15: 1,
            },
        )

        self.uniform_sampler = tio.data.UniformSampler(
            patch_size=tuple(self.cfg.data.PATCH_SIZE)
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        item_id = self.ids[idx]
        img, mask = self.load_example(item_id)

        img = torch.from_numpy(img).to(torch.float32)
        mask = torch.from_numpy(mask).to(torch.int16)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img.unsqueeze(0)),
            label=tio.LabelMap(tensor=mask.unsqueeze(0)),
        )  # shape (C=1, W, H, D)

        if random.random() < 0.5:
            # Sample patch uniformly at random in volume
            patch = next(self.uniform_sampler._generate_patches(subject, num_patches=1))
        else:
            # Sample random patch in foreground
            patch = next(self.label_sampler._generate_patches(subject, num_patches=1))

        # Apply photometric augmentations for finetuning
        if self.apply_augmentations:
            img = self.supervised_transform(patch["image"].data)
        else:
            img = patch["image"].data

        mask = patch["label"].data

        # NOTE: now image and mask are in dimension (C=1, H, W, D)
        img = torch.transpose(img, 2, 1)
        mask = torch.transpose(mask, 2, 1)
        C, H, W, D = img.shape

        # TODO: Add random scale (0.9, 1.1) if possible
        if self.apply_geometric_transform:
            affine_matrix = utils.generate_single_affine_spatial_transform_3d(
                p_rot=self.cfg.finetuning.p_rotation,
                p_flip=self.cfg.finetuning.p_flip,
                is_rotated=self.cfg.finetuning.is_rotated,
                is_cropped=self.cfg.finetuning.is_cropped,
                is_flipped=self.cfg.finetuning.is_flipped,
                max_angle=self.cfg.finetuning.max_angle,  # [-15, 15] degrees
                max_tilt_deg=self.cfg.finetuning.max_tilt_deg,  # 0 (only 2D rotation around z-axis)
            )  # shape: (3, 4) affine 3D matrix
            grid = F.affine_grid(
                affine_matrix.unsqueeze(0),
                torch.Size((1, C, D, H, W)),
                align_corners=True,
            )  # shape: (N=1, D, H, W, 3)
            img = F.grid_sample(
                rearrange(img, "C H W D -> 1 C D H W"),
                grid,
                mode="bilinear",  # NOTE: the interpolation when input is 5D is actually trilinear in this case
                align_corners=True,
            )
            mask = F.grid_sample(
                rearrange(mask, "C H W D -> 1 C D H W").to(torch.float32),
                grid,
                mode="nearest",
                align_corners=True,
            ).to(torch.int16)
            img = rearrange(img, "1 C D H W -> C H W D", C=1)
            mask = rearrange(mask, "1 C D H W -> C H W D", C=1)

        assert img.shape == mask.shape

        return img, mask

    @staticmethod
    def select_patient_ids(cfg: dict, ids: list[str]) -> list[str]:
        """
        Deterministic nested subset selection.

        ids: list[str]
        percent: percentage of data to keep (1–100)
        seed: random seed for reproducibility
        """
        percent = cfg.finetuning.num_patients
        if not (0 < percent <= 100):
            raise ValueError("percent must be in (0, 100]")

        ids_sorted = sorted(ids)
        rng = random.Random(cfg.data.random_seed)
        ids_shuffled = ids_sorted.copy()
        rng.shuffle(ids_shuffled)

        k = max(1, math.ceil(len(ids_shuffled) * percent / 100))

        return ids_shuffled[:k]


class AMOSSupervisedEval(Dataset):
    def __init__(
        self,
        cfg: dict,
        pipeline: Chain,
    ):
        """
        Dataset for evaluating segmentation performance on AMOS test set.
        It only loads 3D CT scans, convert them to torch tensors and reshape
        them to (C=1, H, W, D).
        """
        self.cfg = cfg
        self.pipeline = pipeline
        self.load_example = self.pipeline._compile(["image", "mask"])
        self.ids = self.pipeline.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        pid = self.ids[idx]
        image, mask = self.load_example(pid)

        # shape (C=1, W, H, D)
        image = torch.from_numpy(image).to(torch.float32).unsqueeze(0)
        mask = torch.from_numpy(mask).to(torch.int32).unsqueeze(0)

        # NOTE: now image and mask are in dimension (C=1, H, W, D)
        image = torch.transpose(image, 2, 1)
        mask = torch.transpose(mask, 2, 1)

        assert image.shape == mask.shape

        return image, mask, pid


def build_amos_pretraining_pipeline(
    amos_dir: str = os.environ.get("AMOS_DATASET_FOLDER"),
    cache_dir: str = os.environ.get("AMOS_CACHE_DIR"),
):
    parse_affine = Transform(
        __inherit__=True,
        flipped_axes=lambda affine: (
            tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3)
            if affine is not None
            else ()
        ),
        spacing=lambda affine: (
            tuple(np.abs(np.diag(affine[:3, :3]))) if affine is not None else None
        ),
    )

    amos_ct_ids = AMOS(root=amos_dir).ids
    # NOTE: AMOS does not disclose test set GT labels. Hence, I use AMOS {val | N=100} set as testing set, merge original {train+test | N=400} sets for pretraining, finetune only on {train | N=200} set.
    # Remove test subjects (i.e., subjet from the "official" val set) and MRI scans
    with open(os.path.join(amos_dir, "amos_ids_val.txt"), "r") as f:
        eval_ids = set(line.strip() for line in f)
    with open(os.path.join(amos_dir, "amos_ids_mri.txt"), "r") as f:
        mri_ids = set(line.strip() for line in f)

    amos_ct_ids = [id for id in amos_ct_ids if id not in eval_ids and id not in mri_ids]

    amos = Chain(
        AMOS(root=amos_dir),
        Filter.keep(amos_ct_ids, verbose=True),
        parse_affine,
        FlipAxesToCanonical(),
    )
    pipeline = Chain(
        amos,  # 500 labeled abdominal CTs + 1661 unlabeled CTs (after filtering) for training
        # cache spacing
        CacheToDisk.simple("spacing", root=cache_dir),
        Filter(
            lambda spacing: spacing is not None and spacing[-1] is not None,
            verbose=True,
        ),
        CacheToDisk.simple("ids", root=cache_dir),
        # remove multi-phase CT
        Apply(image=lambda x: x[..., 0] if x.ndim == 4 else x),
        # cropping, rescaling
        Transform(
            __inherit__=True,
            cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU),
        ),
        CropToBox(axis=(-3, -2, -1)),
        RescaleToSpacing(
            to_spacing=SPACING, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)
        ),
        # adding body_voxels
        Transform(__inherit__=True, body_voxels=lambda image: get_body_mask(image)),
        Filter(lambda body_voxels: len(body_voxels) > 0, verbose=True),
        CacheToDisk.simple("ids", root=cache_dir),
        Apply(image=lambda x: hu_clip(x, WINDOW_HU[0], WINDOW_HU[1])),
        Apply(image=lambda x: np.float32(x)),
        CacheToDisk.simple("image", "body_voxels", root=cache_dir),
    )

    return pipeline


def build_amos_finetuning_pipeline(
    amos_dir: str = os.environ.get("AMOS_DATASET_FOLDER"),
    cache_dir: str = os.environ.get("AMOS_CACHE_DIR"),
):
    parse_affine = Transform(
        __inherit__=True,
        flipped_axes=lambda affine: (
            tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3)
            if affine is not None
            else ()
        ),
        spacing=lambda affine: (
            tuple(np.abs(np.diag(affine[:3, :3]))) if affine is not None else None
        ),
    )

    # only the 500 labeled abdominal CTs for finetuning (MRI scans already remove by the [:500] slicing)
    amos_ct_ids = AMOS(root=amos_dir).ids[:500]

    # Keep only train set patients for finetuning
    with open(os.path.join(amos_dir, "amos_ids_train.txt"), "r") as f:
        train_ids = set(line.strip() for line in f)
    amos_ct_ids = [id for id in amos_ct_ids if id in train_ids]

    # 200 labeled abdominal CTs for finetuning (from AMOS initial train set)
    assert len(amos_ct_ids) == 200

    amos = Chain(
        AMOS(root=amos_dir),
        Filter.keep(amos_ct_ids, verbose=True),
        parse_affine,
        FlipAxesToCanonical(),
    )
    pipeline = Chain(
        amos,
        # cache spacing
        CacheToDisk.simple("spacing", root=cache_dir),
        Filter(
            lambda spacing: spacing is not None and spacing[-1] is not None,
            verbose=True,
        ),
        CacheToDisk.simple("ids", root=cache_dir),
        # cropping, rescaling
        Transform(
            __inherit__=True,
            cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU),
        ),
        CropToBox(axis=(-3, -2, -1)),
        RescaleToSpacing(
            to_spacing=SPACING, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)
        ),
        CacheToDisk.simple("image", root=cache_dir),
        # filtering by shape
        Filter(lambda image: np.all(np.array(image.shape) >= PATCH_SIZE), verbose=True),
        CacheToDisk.simple("ids", root=cache_dir),
        Apply(image=lambda x: hu_clip(x, WINDOW_HU[0], WINDOW_HU[1])),
        Apply(image=lambda x: np.float32(x)),
        CacheToDisk.simple("image", "mask", root=cache_dir),
    )

    return pipeline


def build_amos_eval_pipeline(
    amos_dir: str = os.environ.get("AMOS_DATASET_FOLDER"),
    cache_dir: str = os.environ.get("AMOS_CACHE_DIR"),
):
    parse_affine = Transform(
        __inherit__=True,
        flipped_axes=lambda affine: (
            tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3)
            if affine is not None
            else ()
        ),
        spacing=lambda affine: (
            tuple(np.abs(np.diag(affine[:3, :3]))) if affine is not None else None
        ),
    )

    # only the 500 labeled abdominal CTs for evaluation
    amos_ct_ids = AMOS(root=amos_dir).ids[:500]
    # Keep only subjets from the "official" val set that constitutes our test set (since AMOS test set GT are not disclosed)
    with open(os.path.join(amos_dir, "amos_ids_val.txt"), "r") as f:
        eval_ids = set(line.strip() for line in f)
    amos_ct_eval_ids = [id for id in amos_ct_ids if id in eval_ids]

    # 100 labeled CTs are for evaluation (from AMOS initial val set)
    assert len(amos_ct_eval_ids) == 100

    amos = Chain(
        AMOS(root=amos_dir),
        Filter.keep(amos_ct_eval_ids, verbose=True),
        parse_affine,
        FlipAxesToCanonical(),
    )
    pipeline = Chain(
        amos,
        # cache spacing
        CacheToDisk.simple("spacing", root=cache_dir),
        Filter(
            lambda spacing: spacing is not None and spacing[-1] is not None,
            verbose=True,
        ),
        CacheToDisk.simple("ids", root=cache_dir),
        # cropping, rescaling
        Transform(
            __inherit__=True,
            cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU),
        ),
        CropToBox(axis=(-3, -2, -1)),
        RescaleToSpacing(
            to_spacing=SPACING, axis=(-3, -2, -1), image_fill_value=lambda x: np.min(x)
        ),
        # CacheToDisk.simple("image", root=cache_dir),
        Apply(image=lambda x: hu_clip(x, WINDOW_HU[0], WINDOW_HU[1])),
        Apply(image=lambda x: np.float32(x)),
        CacheToDisk.simple("image", "mask", root=cache_dir),
    )

    return pipeline


def create_train_val_subsets(
    patch_dataset: torch.utils.data.Dataset,
    cfg: dict = None,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Create training/validation split for torch Dataset that does not
    have a predefined split, by taking val_ratio of the total (training) volumes for validation.

    Args:
        patch_dataset (torch.utils.data.Dataset): Dataset to split.
        cfg (dict):

    Returns:
        tuple[torch.utils.data.Subset, torch.utils.data.Subset]: training and validation subsets.
    """
    total_volumes = len(patch_dataset)
    val_volumes = max(1, int(round(cfg.data.val_ratio * total_volumes)))
    train_volumes = total_volumes - val_volumes

    generator = torch.Generator().manual_seed(cfg.data.random_seed)
    train_subset, val_subset = random_split(
        dataset=patch_dataset,
        lengths=[train_volumes, val_volumes],
        generator=generator,
    )
    # wrap only validation to disable augmentations
    val_subset = ValidationWrapper(val_subset)

    return train_subset, val_subset


def build_patch_loader(
    patch_dataset: torch.utils.data.Dataset
    | torch.utils.data.Subset
    | torch.utils.data.ConcatDataset,
    batch_size: int,
    shuffle: bool,
    prefetch_factor: int = 4,
) -> torch.utils.data.DataLoader:
    """Build a Dataloader for 3D patch-based SSL pre-training and/or finetuning.

    Args:
        patch_dataset (AMOSDatasetSSL):
            Patch-based dataset from which batches should be drawn.
        batch_size (int): Batch size may changed between pretraining and finetuning.
        shuffle (bool): Whether to shuffle the dataset at each epoch.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the given patch dataset.
    """
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    num_workers = max(n_cpus - 1, 0)
    logger.info(f"Using num_workers = {num_workers}")
    if num_workers == 0:
        logger.warning("num_workers=0. Dataloading will be a bottleneck.")

    loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )

    return loader
