import glob
import math
import os
import random
from typing import List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from einops import rearrange
from loguru import logger
from torch.utils.data import Dataset

from pix2repv2.utils import utils
from pix2repv2.utils.augmentations import RandomRescaleIntensity, RandomInvert


class MnMsPatchSSL(Dataset):
    """Base class for pre-training patch-based Pix2Rep-v2 with MnMs dataset.
    A random 2D patch is extracted on-the-fly from each 2D slice of the 4D MnMs dataset,
    and two (photometric) augmented views are generated for SSL pre-training (50_655 slices).
    2D patches are resized to a fixed size of (128, 128, 1) after extraction.
    By default, patches are not resampled to keep variability in geometric structures scale
    during pretraining.
    The grid to be applied for the geometric transform is also returned at this stage.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("MnMs_DATASET_FOLDER"),
        min_patch_size_factor: float | None = None,
        max_patch_size_factor: float | None = None,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = True,
        split: str = "training",  # "training" or "validation"
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.min_patch_size_factor = (
            min_patch_size_factor
            if min_patch_size_factor is not None
            else self.cfg.pretraining.min_patch_size_factor
        )
        self.max_patch_size_factor = (
            max_patch_size_factor
            if max_patch_size_factor is not None
            else self.cfg.pretraining.max_patch_size_factor
        )
        self.apply_augmentations = apply_augmentations
        self.apply_elastic_transform = apply_elastic_transform
        self.split = split
        # Precompute all (z, t) slice indices for all patients
        self.slice_indices: list[tuple[str, int, int]] = list()

        if split == "training":
            all_patient_dirs = sorted(
                glob.glob(
                    os.path.join(data_folder_path, split, "*", "*", "*_sa_cine.nii.gz")
                )
            )
        elif split == "validation":
            all_patient_dirs = sorted(
                glob.glob(
                    os.path.join(data_folder_path, split, "*", "*_sa_cine.nii.gz")
                )
            )
        else:
            raise ValueError("split must be either 'training' or 'validation'")

        for patient_path in all_patient_dirs:
            img = nib.load(patient_path, mmap=True)
            W, H, D, T = img.shape
            self.slice_indices.extend(
                [(patient_path, d, t) for d in range(D) for t in range(T)]
            )

        self.resizer = tio.Resize(
            target_shape=(128, 128, 1),  # shape: (W, H, D)
            image_interpolation="bspline",
        )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )

        # SSL photometric augmentations
        cfg_transform = self.cfg.pretraining.pretraining_transform
        self.pretraining_transform = tio.Compose(
            [
                tio.RandomBiasField(
                    coefficients=cfg_transform.random_field_coef,
                    p=cfg_transform.random_field_p,
                ),
                # out_min_max=(Imin, Imax) to reduce the gap between rescaling at the
                # patch-level (pretraining/finetuning) and rescaling at the slice-level (inference/eval)
                RandomRescaleIntensity(
                    Imin_range=(0.0, 0.3),
                    Imax_range=(0.7, 1.0),
                    percentiles=(1, 99),
                ),
                tio.RandomBlur(
                    std=cfg_transform.random_blur_std,
                    p=cfg_transform.random_blur_p,
                ),
                tio.RandomGamma(
                    log_gamma=cfg_transform.log_gamma,
                    p=cfg_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=cfg_transform.random_noise_std,
                    p=cfg_transform.random_noise_p,
                ),
                RandomInvert(p=cfg_transform.random_invert_p),
            ]
        )

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, d, t = self.slice_indices[idx]
        # Lazy loading: only the requested slice via mmap to avoid full 4D reads
        # Warning: Nibabel loads image in (W, H, D, T) 'Python' order, i.e., with the LPS orientation (Left, Posterior, Superior).
        # This means that the first dimension of the array corresponds to the image width (x-axis), etc.
        img = nib.load(path, mmap=True)  # shape: (W, H, D, T)
        slice_2d = img.dataobj[:, :, d, t]
        slice_2d = rearrange(slice_2d, "W H -> 1 W H 1")  # shape: (C, W, H, D)

        slice_2d_totensor = torch.tensor(
            slice_2d,
            dtype=torch.float32,
        )
        subject = tio.Subject(slice_2d=tio.ScalarImage(tensor=slice_2d_totensor))

        w, h, _ = subject.spatial_shape
        patch_size_factor = np.random.uniform(
            self.min_patch_size_factor,
            self.max_patch_size_factor,
        )
        patch_size = min(h, w)
        random_sampler = tio.data.UniformSampler(
            patch_size=(
                round(patch_size * patch_size_factor),
                round(patch_size * patch_size_factor),
                1,
            )
        )
        patch = next(random_sampler._generate_patches(subject, num_patches=1))

        resized_patch = self.resizer(patch)  # shape: (C=1, W=128, H=128, D=1)

        # SSL photometric augmentations
        if self.apply_augmentations:
            view1 = self.pretraining_transform(resized_patch["slice_2d"].data)
            view2 = self.pretraining_transform(resized_patch["slice_2d"].data)
        else:
            view1 = self.intensity_rescaler(resized_patch["slice_2d"].data)
            view2 = self.intensity_rescaler(resized_patch["slice_2d"].data)
        assert view1.shape == view2.shape  # shape: (C=1, W=128, H=128, D=1)

        # Rigid geometric transforms for images and feature maps (equivariance)
        affine_matrix = utils.generate_single_affine_spatial_transform(
            is_rotated=self.cfg.pretraining.is_rotated,
            is_cropped=self.cfg.pretraining.is_cropped,
            is_flipped=self.cfg.pretraining.is_flipped,
            is_translation=self.cfg.pretraining.is_translation,
            max_angle=self.cfg.pretraining.max_angle,
            max_crop=self.cfg.pretraining.max_crop,
        )
        if self.apply_elastic_transform:
            with torch.no_grad():
                # grid that will be applied for the geometric (rigid + elastic) transforms
                grid = utils.generate_random_affine_elastic_grid(
                    p=self.cfg.pretraining.pretraining_transform.elastic_transform_p,
                    H=view1.shape[2],  # recall that view1 has shape (C, W, H, D)
                    W=view1.shape[1],
                    affine=affine_matrix,
                )  # shape: (1, W, H, 2)
        else:
            grid = F.affine_grid(
                affine_matrix.unsqueeze(0),  # add 'batch' dim ('B=1')
                rearrange(view1, "C W H D -> D C W H").shape,
                align_corners=True,
            )

        return view1, view2, grid


class MnMsPatchSupervised(Dataset):
    """
    Patch-based supervised MnMs dataset.
    A fixed-size 2D patch is extracted on-the-fly from each *labeled* 2D slice of the MnMs dataset,
    with 50% background patches and 50% heart patches (LabelSampler). Patches are augmented
    through photometric augmentations during finetuning (potentially different from pretraining).
    All images are resampled to a common pixel spacing prior to patch extraction.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("MnMs_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,
        apply_augmentations: bool = True,
        apply_elastic_transform: bool = True,
        # List of patients for finetuning in different data regimes (i.e., len(selected_patients) = |X_tr|% of training set).
        # If None, all training patients are used.
        selected_patients: list[str] | None = None,
        split: str = "training",  # "training" or "validation"
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # target in plane spacing (s=sw=sh) in mm
        self.apply_augmentations = apply_augmentations
        self.apply_elastic_transform = apply_elastic_transform
        self.selected_patients = selected_patients
        self.split = split

        self.slice_indices: list[tuple[str, str, int, str]] = list()

        if split == "training":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "Labeled", "*"))
            )
        elif split == "validation":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "*"))
            )
        else:
            raise ValueError("split must be either 'training' or 'validation'")

        # If a subset is specified, only keep their patient dirs
        if self.selected_patients is not None:
            selected_set = set(self.selected_patients)
            patient_dirs = [
                p_dir
                for p_dir in all_patient_dirs
                if os.path.basename(p_dir) in selected_set
            ]
        else:
            patient_dirs = all_patient_dirs

        for p_dir in patient_dirs:
            pid = os.path.basename(p_dir)

            for frame in ["ed", "es"]:
                img_path = os.path.join(p_dir, f"{pid}_sa_{frame}.nii.gz")
                gt_path = os.path.join(p_dir, f"{pid}_sa_{frame}_gt.nii.gz")

                if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                    logger.warning(
                        f"Image or GT path does not exist for patient {pid}. Skipping..."
                    )
                    continue

                img = nib.load(img_path, mmap=True)
                W, H, D = img.shape

                self.slice_indices.extend(
                    [(img_path, gt_path, d, pid) for d in range(D)]
                )

        self.intensity_rescaler = RandomRescaleIntensity(
            Imin_range=(0.0, 0.3),
            Imax_range=(0.7, 1.0),
            percentiles=(1, 99),
        )

        # Photometric augmentations for supervised stage
        cfg_transform = self.cfg.finetuning.supervised_transform
        self.supervised_transform = tio.Compose(
            [
                tio.RandomBiasField(
                    coefficients=cfg_transform.random_field_coef,
                    p=cfg_transform.random_field_p,
                ),
                # out_min_max=(Imin, Imax) to reduce the gap between rescaling at the
                # patch-level (pretraining/finetuning) and rescaling at the slice-level (inference/eval)
                RandomRescaleIntensity(
                    Imin_range=(0.0, 0.3),
                    Imax_range=(0.7, 1.0),
                    percentiles=(1, 99),
                ),
                tio.RandomBlur(
                    std=cfg_transform.random_blur_std,
                    p=cfg_transform.random_blur_p,
                ),
                tio.RandomGamma(
                    log_gamma=cfg_transform.log_gamma,
                    p=cfg_transform.random_gamma_p,
                ),
                tio.RandomNoise(
                    std=cfg_transform.random_noise_std,
                    p=cfg_transform.random_noise_p,
                ),
                RandomInvert(p=cfg_transform.random_invert_p),
            ]
        )

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path, d, pid = self.slice_indices[idx]
        # Lazy loading of 2d slice and GT mask from '3D' volume
        img_nii = nib.load(img_path, mmap=True)
        gt_nii = nib.load(gt_path, mmap=True)

        orig_spacing = img_nii.header.get_zooms()  # (sw, sh, sd)
        # RandomScale between (0.9, 1.1mm) to augment scale variability
        s_out = np.random.uniform(self.target_spacing - 0.1, self.target_spacing + 0.1)
        target_spacing = (s_out, s_out, orig_spacing[-1])

        img_slice = img_nii.dataobj[:, :, d]
        gt_slice = gt_nii.dataobj[:, :, d]

        img_tensor = torch.tensor(
            rearrange(img_slice, "W H -> 1 W H 1"),
            dtype=torch.float32,
        )
        gt_tensor = torch.tensor(
            rearrange(gt_slice, "W H -> 1 W H 1"),
            dtype=torch.int32,
        )

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor),
            label=tio.LabelMap(tensor=gt_tensor),
        )
        if self.is_resampled:
            img_resampled, actual_spacing_out = utils.resample(
                img=subject["image"].data,
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="bicubic",
            )  # shape: (C, W, H, D)

            # Note: grid_sample() does not support type int. Need to cast the label map
            # to type 'float' prior to resampling and back to type 'int' after.
            label_resampled, _ = utils.resample(
                img=subject["label"].data.float(),
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="nearest",
            )  # shape: (C, W, H, D)

            subject["image"].data = img_resampled
            subject["label"].data = label_resampled.int()

        # During finetuning, sample half background patches and half heart patches
        sampler = tio.data.LabelSampler(
            patch_size=(128, 128, 1),
            label_name="label",
            label_probabilities={
                0: 0.5,  # BACKGROUND
                1: 0.2,  # LV
                2: 0.1,  # MYO
                3: 0.2,  # RV
            },  # same prob for LV and RV
        )
        patch = next(
            sampler._generate_patches(subject, num_patches=1)
        )  # shape: (C=1, W=128, H=128, D=1)

        if self.apply_augmentations:
            if self.apply_elastic_transform:
                affine_matrix = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
                with torch.no_grad():
                    grid = utils.generate_random_affine_elastic_grid(
                        p=self.cfg.finetuning.supervised_transform.elastic_transform_p,
                        # recall that patch has shape (C, W, H, D)
                        H=patch["image"].data.shape[2],
                        W=patch["image"].data.shape[1],
                        affine=affine_matrix,
                    )  # shape: (1, W, H, 2)

                # Apply elastic transform, then photometric augmentations
                patch_to_elastic = rearrange(patch["image"].data, "C W H D -> D C W H")
                patch_to_elastic = F.grid_sample(
                    patch_to_elastic,
                    grid,
                    mode="bicubic",
                    align_corners=True,
                )  # shape: (D=1, C=1, W=128, H=128)
                patch_to_photometric = rearrange(patch_to_elastic, "D C W H -> C W H D")
                img_data = self.supervised_transform(patch_to_photometric)

                gt_to_elastic = rearrange(patch["label"].data, "C W H D -> D C W H")
                gt_to_elastic = F.grid_sample(
                    gt_to_elastic.float(),
                    grid,
                    mode="nearest",
                    align_corners=True,
                ).int()  # shape: (D=1, C=1, W=128, H=128)
                gt_data = rearrange(gt_to_elastic, "D C W H -> C W H D")

            else:
                img_data = self.supervised_transform(patch["image"].data)
                gt_data = patch["label"].data

        else:
            img_data = self.intensity_rescaler(patch["image"].data)
            gt_data = patch["label"].data

        assert img_data.shape == gt_data.shape

        return img_data, gt_data, pid, actual_spacing_out


class MnMsSupervisedEval(Dataset):
    """
    Dataset class used for evaluating the model on '3D' MnMs volumes.
    It considers the D-dim (depth) as the batch dimension and perform inference on each 2d slice
    independently. The evaluation will be performed using 3D Dice Score, considering all slices
    of a given patient at a specific instant (ED or ES).
    The slices are resampled to a common pixel spacing prior to inference to match finetuning
    strategy. The prediction will be later resampled back to the original spacing to compute the
    Dice score in the initial ground truth space.
    """

    def __init__(
        self,
        cfg: dict,
        data_folder_path: str = os.environ.get("MnMs_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,  # in plane spacing (sw, sh) in mm
        split: str = "testing",
        selected_patients: list[str] | None = None,
    ):
        self.cfg = cfg
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # in plane spacing (w, h, d) in mm
        self.split = split
        self.selected_patients = selected_patients
        self.slice_indices: list[tuple[str, str, int, str]] = list()

        if split == "testing" or split == "validation":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "*"))
            )
        elif split == "training":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "Labeled", "*"))
            )
        else:
            raise ValueError("split must be in {'training', 'validation', 'testing'}")

        # If a subset is specified, only keep their patient dirs
        if self.selected_patients is not None:
            selected_set = set(self.selected_patients)
            patient_dirs = [
                p_dir
                for p_dir in all_patient_dirs
                if os.path.basename(p_dir) in selected_set
            ]
        else:
            patient_dirs = all_patient_dirs

        for p_dir in patient_dirs:
            pid = os.path.basename(p_dir)
            ed_idx, es_idx = utils._parse_mnms_ed_es_idx(
                csv_path=os.path.join(
                    data_folder_path,
                    "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
                ),
                patient_id=pid,
            )

            img_path = os.path.join(p_dir, f"{pid}_sa_cine.nii.gz")
            gt_path = os.path.join(p_dir, f"{pid}_sa_cine_gt.nii.gz")

            if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                logger.warning(
                    f"Image or GT path does not exist for patient {pid}. Skipping..."
                )
                continue

            self.slice_indices.extend(
                [(img_path, gt_path, t, pid) for t in [ed_idx, es_idx]]
            )

        self.intensity_rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        )

    def __len__(self):
        # len = 272 since 136 test patients with two '3D' volume (ED/ES) each
        return len(self.slice_indices)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, str, tuple[float, float, float]]:
        img_path, gt_path, t, pid = self.slice_indices[idx]
        # Loading of '3D' volume at a specific instant (ED or ES)
        img_nifti = nib.load(img_path, mmap=True)
        gt_nifti = nib.load(gt_path, mmap=True)

        sw, sh, sd, st = img_nifti.header.get_zooms()  # (sw, sh, sd, st) spacing in mm
        orig_spacing = (sw, sh, sd)
        target_spacing = (self.target_spacing, self.target_spacing, sd)

        img = img_nifti.dataobj[:, :, :, t]
        gt = gt_nifti.dataobj[:, :, :, t]

        img_tensor = torch.tensor(img, dtype=torch.float32)  # shape: (W, H, D)
        gt_tensor = torch.tensor(gt, dtype=torch.int32)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img_tensor.unsqueeze(0)),
            label=tio.LabelMap(tensor=gt_tensor.unsqueeze(0)),
        )
        actual_spacing_out = orig_spacing
        if self.is_resampled:
            img_resampled, actual_spacing_out = utils.resample(
                img=subject["image"].data,
                spacing_in=orig_spacing,
                spacing_out=target_spacing,
                mode="bicubic",
            )  # shape: (C=1, W, H, D)
            subject["image"].data = img_resampled

        img_data = self.intensity_rescaler(subject["image"].data)
        gt_data = subject["label"].data  # shape: (C=1, W, H, D)

        return img_data, gt_data, pid, orig_spacing, actual_spacing_out


class MnMsSupervised4D(Dataset):
    """
    Dataset used for zero-shot 3D+t segmentation propagation on MnMs dataset.
    The dataset returns the full 4D ACDC volume along with the ED and ES ground truth masks.
    The first frame of the returned 4D volume (mr_volume_4d) corresponds to the ED instant and
    the last frame corresponds to the ES instant.

    -   is_resampled boolean argument controls whether the 4D volumes are resampled to a common
    pixel spacing prior to forward pass. If resampled, the predictions will be later resampled
    back to the original spacing.

    The ed_frame_number and es_frame_number are also returned to identify the frames
    corresponding to the ED and ES ground truth masks.
    """

    def __init__(
        self,
        data_folder_path: str = os.environ.get("MnMs_DATASET_FOLDER"),
        is_resampled: bool = True,
        target_spacing: float = 1.0,  # in plane spacing (sw, sh) in mm
        split: str = "testing",
    ):
        self.data_folder_path = data_folder_path
        self.is_resampled = is_resampled
        self.target_spacing = target_spacing  # in plane spacing (w, h, d) in mm
        self.split = split
        self.slice_indices: list[tuple[str, str, int, str]] = list()

        if split == "testing" or split == "validation":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "*"))
            )
        elif split == "training":
            all_patient_dirs = sorted(
                glob.glob(os.path.join(data_folder_path, split, "Labeled", "*"))
            )
        else:
            raise ValueError("split must be in {'training', 'validation', 'testing'}")

        for p_dir in all_patient_dirs:
            pid = os.path.basename(p_dir)
            ed_idx, es_idx = utils._parse_mnms_ed_es_idx(
                csv_path=os.path.join(
                    data_folder_path,
                    "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv",
                ),
                patient_id=pid,
            )

            img_path = os.path.join(p_dir, f"{pid}_sa_cine.nii.gz")
            gt_path = os.path.join(p_dir, f"{pid}_sa_cine_gt.nii.gz")

            ed_gt_path = os.path.join(p_dir, f"{pid}_sa_ed_gt.nii.gz")
            es_gt_path = os.path.join(p_dir, f"{pid}_sa_es_gt.nii.gz")

            if not (os.path.exists(img_path) and os.path.exists(gt_path)):
                logger.warning(
                    f"Image or GT path does not exist for patient {pid}. Skipping..."
                )
                continue

            self.slice_indices.extend(
                [
                    (
                        img_path,
                        ed_gt_path,
                        es_gt_path,
                        pid,
                        (ed_idx, es_idx),
                    )
                ]
            )

        self.intensity_rescaler = tio.RescaleIntensity(
            out_min_max=(0, 1),
            percentiles=(1, 99),
        )

    def __len__(self):
        # len = 136 test patients
        return len(self.slice_indices)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        (
            img_path,
            ed_gt_path,
            es_gt_path,
            pid,
            (ed_frame_number, es_frame_number),
        ) = self.slice_indices[idx]

        # Loading of '4D' volume with ground truth seg at ED
        img_nifti = nib.load(img_path)  # shape: (W, H, D, T)
        gt_ed_nifti = nib.load(ed_gt_path)  # shape: (W, H, D)
        gt_es_nifti = nib.load(es_gt_path)

        sw, sh, sd, st = img_nifti.header.get_zooms()  # (sw, sh, sd, st)
        orig_spacing = (sw, sh, sd)

        img = img_nifti.get_fdata()
        # NOTE: frame numbers are 0-based in MnMs metadata
        img = get_cine_between_frames(img, ed_frame_number, es_frame_number)
        gt_ed = gt_ed_nifti.get_fdata()
        gt_es = gt_es_nifti.get_fdata()

        img_tensor = torch.tensor(img, dtype=torch.float32)  # shape: (W, H, D, T)
        gt_ed_tensor = torch.tensor(gt_ed, dtype=torch.int32)  # shape: (W, H, D)
        gt_es_tensor = torch.tensor(gt_es, dtype=torch.int32)

        if self.is_resampled:
            target_spacing = (self.target_spacing, self.target_spacing, sd)
            resampled_volumes = []

            for t in range(img_tensor.shape[-1]):
                mr_volume = img_tensor[:, :, :, t]
                # Resampling each 3D volume independently to target spacing.
                mr_volume_res, actual_spacing_out = utils.resample(
                    img=mr_volume.unsqueeze(0),
                    spacing_in=orig_spacing,  # only sw, sh is taken into account
                    spacing_out=target_spacing,
                    mode="bicubic",
                )  # shape: (C=1, W, H, D)

                subject = tio.Subject(image=tio.ScalarImage(tensor=mr_volume_res))
                mr_volume_res = self.intensity_rescaler(subject["image"].data)

                resampled_volumes.append(mr_volume_res)

            # Resampled 4D MR volume
            mr_volume_4d = torch.stack(
                resampled_volumes, dim=-1
            )  # shape: (C=1, W, H, D, T)
        else:
            subject = tio.Subject(image=tio.ScalarImage(tensor=img_tensor))
            mr_volume_4d = self.intensity_rescaler(subject["image"].data)
            mr_volume_4d = mr_volume_4d.unsqueeze(0)  # shape: (C=1, W, H, D, T)
            actual_spacing_out = orig_spacing

        return (
            mr_volume_4d,
            gt_ed_tensor,
            gt_es_tensor,
            (ed_frame_number, es_frame_number),
            pid,
            orig_spacing,
            actual_spacing_out,
        )


def select_nested_patient_ids(
    cfg: dict,
    data_folder_path: str = os.environ.get("MnMs_DATASET_FOLDER"),
) -> List[str]:
    """Returns a list of MnMs patient IDs to include for finetuning.

    Properties:
        - Random patient-level selection
        - Fully reproducible given cfg.data.random_seed
        - Nested across data regimes (1% ⊂ 2% ⊂ 5% ⊂ ...)
        - No stratification (random choice, compared to ACDC where we stratify by pathology)

    Args:
        cfg (dict): Hydra config dict.
        data_folder_path (str): Path to MnMs training/Labeled folder.

    Returns:
        list[str]: List of patient IDs to train on.
    """
    labeled_train_path = os.path.join(data_folder_path, "training", "Labeled")

    patient_ids = sorted(
        pid
        for pid in os.listdir(labeled_train_path)
        if os.path.isdir(os.path.join(labeled_train_path, pid))
    )

    rng = random.Random(cfg.data.random_seed)
    rng.shuffle(patient_ids)

    # Percentage of patients to use for finetuning
    pct = float(cfg.finetuning.num_patients)
    assert 0 < pct <= 100, "Percentage of finetuning patients must be in (0, 100]"

    n_total = len(patient_ids)
    num_patients = max(1, math.ceil(pct / 100.0 * n_total))

    selected_patient_ids = patient_ids[:num_patients]

    return selected_patient_ids


def get_cine_between_frames(img_4d, ed, es):
    """
    Extract frames between ED and ES.

    Parameters:
        img_4d: numpy array of shape (W, H, D, T)
        ed: int (0-based index of ED frame in the 4D volume, as per MnMs metadata)
        es: int (same convention as ed)

    Returns:
        4D sub-volume (W, H, D, t_sub)
    """
    if ed <= es:
        # Normal case (no wrap)
        return img_4d[..., ed : es + 1]
    else:
        # Wrap-around case (cyclic time)
        return np.concatenate((img_4d[..., ed:], img_4d[..., : es + 1]), axis=-1)
