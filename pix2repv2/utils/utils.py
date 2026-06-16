import json
import os
import random
import re
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from hydra import compose, initialize
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch._decomp.decompositions import _linspace_from_neg_one
from torch_cubic_spline_grids import CubicBSplineGrid2d, CubicBSplineGrid3d


def load_nii(img_path):
    """
    Shortcut to load a nifti file
    """

    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


def resample(
    img: torch.Tensor,
    spacing_in: float | tuple[float, float, float],
    spacing_out: float | tuple[float, float, float],
    mode: str,
    align_corners: bool = True,
) -> torch.Tensor | tuple[float, float, float]:
    """Resample a given img to a target spacing (spacing_out) based on Pytorch grid_sample().
    The image is assumed to be a torch.Tensor in (C, W, H, D) (Nibabel) format.
    If a 2D or 3D image is provided, it will be reshaped accordingly.

    Args:
        img (torch.Tensor): The image to be resampled (could be an image or a label map).
        Can be 2D (W, H), 3D (C, W, H) or 4D tensor (C, W, H, D).
        spacing_in (float | tuple[float, float, float]): Input spacing of the image (in mm).
        If a single value s is provided, sw = sh = sd = s is assumed.
        spacing_out (float | tuple[float, float, float]): Output spacing of the image (in mm).
        If a single value s is provided, sw = sh = sd = s is assumed.
        mode (str): Interpolation mode to be used for resampling. 'bilinear' or 'bicubic'
        for images and/or probability maps, 'nearest' for segmentation masks.
        align_corners (bool): Whether to align corners in the interpolation. Defaults to True.

    Returns:
        torch.Tensor: The resampled image in the spacing_out spacing (C, W, H, D).
        tuple: actual output spacing after resampling (sw, sh, sd)
    """
    assert isinstance(img, torch.Tensor), (
        f"img should be a torch.Tensor, got {type(img)}"
    )

    if img.ndim == 2:  # (W, H)
        img = rearrange(img, "W H -> 1 W H 1")
    elif img.ndim == 3:  # (C, W, H)
        img = rearrange(img, "C W H -> C W H 1")
    elif img.ndim != 4:  # (C, W, H, D)
        raise ValueError(f"img should be 2D, 3D or 4D tensor, got {img.ndim}D tensor")

    C, W_in, H_in, D = img.shape

    if isinstance(spacing_in, float):
        spacing_in = (spacing_in, spacing_in, spacing_in)
    if isinstance(spacing_out, float):
        spacing_out = (spacing_out, spacing_out, spacing_out)

    # round is used instead of ceil to avoid shape mismatch
    W_out = round(spacing_in[0] / spacing_out[0] * (W_in - 1)) + 1
    H_out = round(spacing_in[1] / spacing_out[1] * (H_in - 1)) + 1

    identity = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    grid = torch.nn.functional.affine_grid(
        theta=rearrange(identity, "W H -> 1 W H").repeat(D, 1, 1),
        size=torch.Size((D, C, W_out, H_out)),
        align_corners=align_corners,
    ).to(img.device)
    img_resampled = torch.nn.functional.grid_sample(
        rearrange(img, "C W H D -> D C W H"),
        grid,
        mode=mode,
        align_corners=align_corners,
    )
    # Rearrange shape to match img input shape
    img_resampled = rearrange(img_resampled, "D C W H -> C W H D")

    actual_spacing_out = (
        spacing_in[0] * (W_in - 1) / (W_out - 1),
        spacing_in[1] * (H_in - 1) / (H_out - 1),
        spacing_in[2],
    )

    return img_resampled, actual_spacing_out


def resample_3d(
    img: torch.Tensor,
    spacing_in: float | tuple[float, float, float],
    spacing_out: float | tuple[float, float, float],
    mode: str,
    align_corners: bool = True,
):
    """
    True 3D resampling using grid_sample.

    img: (C, W, H, D)
    returns: (C, W_out, H_out, D_out)
    """
    assert img.ndim == 4, f"Expected (C,W,H,D), got {img.shape}"
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)

    C, W_in, H_in, D_in = img.shape

    if isinstance(spacing_in, float):
        spacing_in = (spacing_in,) * 3
    if isinstance(spacing_out, float):
        spacing_out = (spacing_out,) * 3

    # --- compute output size ---
    W_out = round(spacing_in[0] / spacing_out[0] * (W_in - 1)) + 1
    H_out = round(spacing_in[1] / spacing_out[1] * (H_in - 1)) + 1
    D_out = round(spacing_in[2] / spacing_out[2] * (D_in - 1)) + 1

    img = rearrange(img, "C W H D -> 1 C D W H")

    theta = torch.eye(3, 4, device=img.device).unsqueeze(0)  # (1,3,4)
    grid = F.affine_grid(
        theta,
        size=(1, C, D_out, W_out, H_out),
        align_corners=align_corners,
    )

    out = F.grid_sample(
        img,
        grid,
        mode=mode,
        align_corners=align_corners,
    )

    out = rearrange(out, "1 C D W H -> C W H D")

    actual_spacing_out = (
        spacing_in[0] * (W_in - 1) / (W_out - 1),
        spacing_in[1] * (H_in - 1) / (H_out - 1),
        spacing_in[2] * (D_in - 1) / (D_out - 1),
    )

    return out, actual_spacing_out


# TODO: do the doc for this function
def generate_affine_spatial_transform(
    batch_size: int,
    is_rotated: bool,
    is_cropped: bool,
    is_flipped: bool,
    is_translation: bool,
    max_angle: float = np.pi / 2,
    max_crop: float = 0.5,
):
    identity_affine = torch.eye(3).float()
    identity_affine = identity_affine.repeat((batch_size, 1, 1))

    zeros = torch.zeros(batch_size)
    ones = torch.ones(batch_size)

    # Rotation
    if is_rotated:
        theta = torch.FloatTensor(batch_size).uniform_(-max_angle, max_angle)
        # angles = torch.tensor([np.pi / 2, np.pi, np.pi * 3/2])
        # theta = angles[torch.multinomial(angles, batch_size, replacement = True)]

        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        rotation = torch.stack(
            (
                torch.stack([cos_theta, -sin_theta, zeros], dim=-1),
                torch.stack([sin_theta, cos_theta, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ),
            dim=1,
        ).float()

        # Remove randmly a certain percentage of the transformation
        # random_indexes = torch.randint(batch_size, (int(batch_size * 0.2), ))
        # rotation[random_indexes] = torch.eye(3)

    else:
        rotation = identity_affine.detach().clone()

    # Cropping
    if is_cropped:
        # Define random crop parameters
        scale_factor = torch.FloatTensor(batch_size).uniform_(max_crop, 0.95)
        translate_height = 2 * (1 - scale_factor) * torch.rand(batch_size) - (
            1 - scale_factor
        )
        translate_width = 2 * (1 - scale_factor) * torch.rand(batch_size) - (
            1 - scale_factor
        )

        # Compute copping matrix
        crop = torch.stack(
            (
                torch.stack([scale_factor, zeros, translate_width], dim=-1),
                torch.stack([zeros, scale_factor, translate_height], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ),
            dim=1,
        ).float()

        # Remove randmly a certain percentage of the transformation
        # random_indexes = torch.randint(batch_size, (int(batch_size * 0.2), ))
        # crop[random_indexes] = torch.eye(3)

    else:
        crop = identity_affine.detach().clone()

    # Flip
    if is_flipped:
        # For certain samples (50%) apply horizontal flip, for others vertical flip
        horizontal_flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()

        flip = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()

        flip = flip.repeat((batch_size, 1, 1))

        random_indexes = torch.randint(
            low=0,
            high=batch_size,
            size=(int(batch_size * 0.5),),
        )
        flip[random_indexes] = horizontal_flip

        # Remove randmly a certain percentage of the transformation (for 1/3 of the images)
        random_indexes = torch.randint(
            low=0,
            high=batch_size,
            size=(int(batch_size * 0.33),),
        )
        flip[random_indexes] = torch.eye(3)

    else:
        flip = identity_affine.detach().clone()

    # Translation
    if is_translation:
        dx = torch.FloatTensor(batch_size).uniform_(-1, 1)
        dy = torch.FloatTensor(batch_size).uniform_(-1, 1)

        translation = torch.stack(
            (
                torch.stack([ones, zeros, dx], dim=-1),
                torch.stack([zeros, ones, dy], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ),
            dim=1,
        ).float()

        # Remove randmly a certain percentage of the transformation
        random_indexes = torch.randint(batch_size, (int(batch_size * 0.5),))
        translation[random_indexes] = torch.eye(3)

    else:
        translation = identity_affine.detach().clone()

    # Combine all transformations
    final_affine_matrix = torch.bmm(
        rotation,
        torch.bmm(crop, torch.bmm(translation, flip)),
    )
    final_affine_matrix = final_affine_matrix[:, :-1, :]

    return final_affine_matrix


def generate_single_affine_spatial_transform(
    is_rotated: bool,
    is_cropped: bool,
    is_flipped: bool,
    is_translation: bool,
    max_angle: float = np.pi / 2,
    max_crop: float = 0.5,
):
    identity_affine = torch.eye(3).float()

    if is_rotated:
        theta = random.uniform(-max_angle, max_angle)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        rotation = torch.tensor(
            [
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1],
            ]
        ).float()

    else:
        rotation = identity_affine.detach().clone()

    if is_cropped:
        scale_factor = random.uniform(max_crop, 0.95)
        translate_height = random.uniform(-(1 - scale_factor), (1 - scale_factor))
        translate_width = random.uniform(-(1 - scale_factor), (1 - scale_factor))

        crop = torch.tensor(
            [
                [scale_factor, 0, translate_width],
                [0, scale_factor, translate_height],
                [0, 0, 1],
            ]
        ).float()

    else:
        crop = identity_affine.detach().clone()

    # Flip
    if is_flipped:
        r = random.random()
        # 1/3 chance to have vertical flip, 1/3 horizontal flip, 1/3 no flip
        if r < 1 / 3:
            # Vertical flip
            flip = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]).float()
        elif r < 2 / 3:
            # Horizontal flip
            flip = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
        else:
            # Identity (no flip)
            flip = identity_affine.detach().clone()

    else:
        flip = identity_affine.detach().clone()

    # Translation
    if is_translation:
        dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(-1, 1)

        r = random.random()
        if r < 1 / 2:
            translation = torch.tensor(
                [
                    [1, 0, dx],
                    [0, 1, dy],
                    [0, 0, 1],
                ]
            ).float()
        else:
            translation = identity_affine.detach().clone()

    else:
        translation = identity_affine.detach().clone()

    # Combine all transformations
    final_affine_matrix = torch.matmul(
        rotation,
        torch.matmul(crop, torch.matmul(translation, flip)),
    )
    final_affine_matrix = final_affine_matrix[:-1, :]

    return final_affine_matrix


def generate_single_affine_spatial_transform_3d(
    p_rot: float,
    p_flip: float,
    is_rotated: bool,
    is_cropped: bool,
    is_flipped: bool = False,  # no flip in 3D
    max_angle: float = np.pi / 4,
    max_crop: float = 0.33,  # 1/3
    max_tilt_deg: float = 0,  # NOTE: 2D rotations only
):
    identity_affine = torch.eye(4).float()
    proba_rotation = random.random()
    proba_flip = random.random()

    if is_rotated and proba_rotation < p_rot:

        def _random_axis_close_to_z(max_tilt_deg):
            """
            Generate a random 3D unit vector axis that is mostly aligned with the z-axis,
            with a small tilt up to `max_tilt_deg` degrees.

            Returns:
                v (np.ndarray): 3D unit vector close to z-axis
            """
            max_tilt_rad = np.deg2rad(max_tilt_deg)
            tilt = np.random.uniform(0, max_tilt_rad)

            z = np.random.randn(2)
            u = z / np.linalg.norm(z)  # unitaire en 2D

            sin_eps = np.sin(tilt)
            v = np.array(
                (u[0] * sin_eps, u[1] * sin_eps, np.cos(tilt))
            )  # unitaire en 3D avec une grosse composante z

            return v

        theta = np.random.uniform(-max_angle, max_angle)
        rot3x3 = R.from_rotvec(theta * _random_axis_close_to_z(max_tilt_deg))
        rot3x3 = rot3x3.as_matrix()
        rotation = torch.eye(4)
        rotation[:3, :3] = torch.tensor(rot3x3).float()

    else:
        rotation = identity_affine.detach().clone()

    if is_cropped:
        scale_factor = random.uniform(max_crop, 0.95)
        translate_width = random.uniform(-(1 - scale_factor), (1 - scale_factor))
        translate_height = random.uniform(-(1 - scale_factor), (1 - scale_factor))
        translate_depth = random.uniform(-(1 - scale_factor), (1 - scale_factor))

        crop = torch.tensor(
            [
                [scale_factor, 0, 0, translate_width],
                [0, scale_factor, 0, translate_height],
                # for no crop in z: [0, 0, 1, 0],
                [0, 0, scale_factor, translate_depth],
                [0, 0, 0, 1],
            ]
        ).float()

    else:
        crop = identity_affine.detach().clone()

    # Flip
    if is_flipped and proba_flip < p_flip:
        r = random.random()
        # 25% chance x-axis flip, 25% y-axis flip, 50% no flip
        if r < 0.25:
            # Flip along x-axis
            flip = torch.tensor(
                [
                    [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ).float()
        elif r < 0.5:
            # Flip along y-axis
            flip = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            ).float()
        elif r < 0.75:
            # Flip along z-axis
            flip = torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            ).float()
        else:
            # No flip
            flip = identity_affine.detach().clone()
    else:
        flip = identity_affine.detach().clone()

    # Combine all transformations
    final_affine_matrix = torch.matmul(
        crop,
        torch.matmul(rotation, flip),
    )
    final_affine_matrix = final_affine_matrix[:-1, :]

    return final_affine_matrix


def generate_random_affine_elastic_grid(
    p: float,
    H: int,
    W: int,
    affine: torch.Tensor | None = None,
    nc_l: int = 5,
    nc_h: int = 9,
    locked_borders: bool = True,
    align_corners: bool = True,
) -> torch.Tensor:
    """Generate a random 2D sampling grid that combines elastic deformation
    and an optional (rigid) affine transformation.

    The elastic deformation is modeled as the sum of two cubic B-spline deformation
    fields:
        - a low-frequency (coarse) field controlling 'large-scale' anatomical
          variations,
        - a high-frequency field introducing local shape deformation.

    The resulting grid is compatible with `torch.nn.functional.grid_sample()`
    and can be applied either directly to images or to (dense) feature maps.

    Args:
        p (float, optional): Probability of applying the elastic deformation.
        H (int): Image height in pixels
        W (int): Image width in pixels
        affine : torch.Tensor or None, optional
            Affine transformation matrix of shape (2, 3) in homogeneous coordinates.
            If provided, the elastic deformation is composed with this affine transformation.
            If None, only the elastic deformation is applied.
        nc_l : int, default to 5.
            Number of control points per spatial dimension for the low-frequency
            (coarse) B-spline grid. Smaller values produce smoother deformations
            at a larger scale.
        nc_h : int, default to 9.
            Number of control points per spatial dimension for the high-frequency
            (fine) B-spline grid. Larger values allow more local deformations.
        locked_borders : bool, default=True
            To be set to True such that the control points on the image borders are set to zero,
            enforcing zero displacement at the image boundaries and preventing folding near the image edges.

        align_corners (bool, optional): Defaults to True.

    Returns:
        grid : torch.Tensor
        Sampling grid of shape (1, H, W, 2) with normalized coordinates
        in the range [-1, 1], to be used with `grid_sample`.
    """
    if affine is None:
        p = 1.0  # always apply elastic deformation if no affine provided
    if torch.rand(1) > p:
        # Only apply affine transformation if provided
        grid = F.affine_grid(
            affine.unsqueeze(0),  # add 'batch' dim ('B=1')
            torch.Size((1, 1, W, H)),
            align_corners=True,
        )
        return grid

    bspline_l = CubicBSplineGrid2d(resolution=(nc_l, nc_l), n_channels=2)
    bspline_h = CubicBSplineGrid2d(resolution=(nc_h, nc_h), n_channels=2)

    Dx_l = Dy_l = 8  # maximum magnitude in pixels along x and y
    Dx_h = Dy_h = 2.5

    max_displacement_l = np.array([[[2 * Dx_l / H]], [[2 * Dy_l / W]]])
    max_displacement_h = np.array([[[2 * Dx_h / H]], [[2 * Dy_h / W]]])

    # Control point value uniform at random between -max and max along every channel
    bspline_l.data = (
        (torch.rand(*bspline_l.data.shape) - 0.5) * 2 * torch.Tensor(max_displacement_l)
    )
    bspline_h.data = (
        (torch.rand(*bspline_h.data.shape) - 0.5) * 2 * torch.Tensor(max_displacement_h)
    )

    if locked_borders:
        bspline_l.data[:, 0, :] = 0
        bspline_l.data[:, nc_l - 1, :] = 0
        bspline_l.data[:, :, 0] = 0
        bspline_l.data[:, :, nc_l - 1] = 0

        bspline_h.data[:, 0, :] = 0
        bspline_h.data[:, nc_h - 1, :] = 0
        bspline_h.data[:, :, 0] = 0
        bspline_h.data[:, :, nc_h - 1] = 0

    # Utilities for building the base sampling grid
    def _linspace_from_neg_one(
        num_steps: int, align_corners: bool, dtype: torch.dtype, device: torch.device
    ):
        if num_steps <= 1:
            return torch.tensor(0, device=device, dtype=dtype)

        a = ((num_steps - 1) / num_steps) if not align_corners else 1
        return torch.linspace(-a, a, steps=num_steps, device=device, dtype=dtype)

    def _make_base_grid_4d(theta: torch.Tensor, h: int, w: int, align_corners: bool):
        dtype = theta.dtype
        device = theta.device

        # Using padding and summation generates a single kernel vs using torch.stack where 3 kernels generated
        # corresponding to each individual tensor: grid_x, grid_y, grid_one
        grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
        grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
        grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

        # this is just a temporary hack and we should use torch.stack here once #104480 is merged
        grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
        grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
        grid_one = torch.nn.functional.pad(
            grid_one, pad=(2, 0), mode="constant", value=0
        )
        return grid_x + grid_y + grid_one

    grid = _make_base_grid_4d(bspline_l.data, H, W, align_corners)
    # Convert grid from [-1, 1] (make_base_grid coordinates sytem) to [0, 1] (cubicbsplinegrid2D format)
    base_grid_01 = grid[:, :, 0:2] / 2 + 0.5  # from [-1, 1] to [0, 1]

    # Combining both displacement field
    displacement = bspline_l(base_grid_01) + bspline_h(base_grid_01)
    grid[:, :, 0:2] += displacement

    if affine is None:
        grid = grid[:, :, 0:2].view(1, W, H, 2)
        return grid
    else:
        # if composing the elastic deformation with an affine part as in our case
        # adapted from https://github.com/pytorch/pytorch/blob/9b3e34d8589b29f7b4e7fab6f78711b7ca6e4639/torch/_decomp/decompositions.py#L4278
        # affine should be (2, 3) in homogeneous coords for a single subject
        grid = (grid.float().view(-1, 3, 1) * affine.mT.unsqueeze(0)).sum(-2)
        # you can return that in the dataset getitem to be collated
        grid = grid.view(1, W, H, 2)

        return grid


def generate_random_affine_elastic_grid_3d(
    p: float,
    H: int,
    W: int,
    D: int,
    affine: torch.Tensor | None = None,
    nc_l: int = 7,
    nc_h: int = 9,
    locked_borders: bool = True,
    align_corners: bool = True,
) -> torch.Tensor:
    """Generate a random 3D sampling grid that combines elastic deformation
    and an optional (rigid) affine transformation.

    The elastic deformation is modeled as the sum of two cubic B-spline deformation
    fields:
        - a low-frequency (coarse) field controlling 'large-scale' anatomical
          variations,
        - a high-frequency field introducing local shape deformation.

    The resulting grid is compatible with `torch.nn.functional.grid_sample()`
    and can be applied either directly to images or to (dense) feature maps.

    Args:
        p (float, optional): Probability of applying the elastic deformation.
        H (int): Image height in pixels
        W (int): Image width in pixels
        D (int): Image depth in pixels
        affine : torch.Tensor or None, optional
            Affine transformation matrix of shape (3, 4) in homogeneous coordinates.
            If provided, the elastic deformation is composed with this affine transformation.
            If None, only the elastic deformation is applied.
        nc_l : int, default to 7.
            Number of control points per spatial dimension for the low-frequency
            (coarse) B-spline grid. Smaller values produce smoother deformations
            at a larger scale.
        nc_h : int, default to 9.
            Number of control points per spatial dimension for the high-frequency
            (fine) B-spline grid. Larger values allow more local deformations.
        locked_borders : bool, default=True
            To be set to True such that the control points on the image borders are set to zero,
            enforcing zero displacement at the image boundaries and preventing folding near the image edges.

        align_corners (bool, optional): Defaults to True.

    Returns:
        grid : torch.Tensor
        Sampling grid of shape (1, D, W, H, 3) with normalized coordinates
        in the range [-1, 1], to be used with `grid_sample`.
    """
    if affine is None:
        p = 1.0  # always apply elastic deformation if no affine provided
    if torch.rand(1) > p:
        # Only apply affine transformation if provided
        grid = F.affine_grid(
            affine.unsqueeze(0),  # add 'batch' dim ('B=1')
            torch.Size((1, 1, D, W, H)),
            align_corners=True,
        )
        return grid

    bspline_l = CubicBSplineGrid3d(resolution=(nc_l, nc_l, nc_l), n_channels=3)
    bspline_h = CubicBSplineGrid3d(resolution=(nc_h, nc_h, nc_h), n_channels=3)

    Dx_l = Dy_l = 9  # maximum magnitude in pixels along x, y, and z
    Dz_l = 5

    Dx_h = Dy_h = 5
    Dz_h = 2.5

    max_displacement_l = torch.Tensor(
        [
            [[[2 * Dx_l / H]]],
            [[[2 * Dy_l / W]]],
            [[[2 * Dz_l / D]]],
        ]
    )
    max_displacement_h = torch.Tensor(
        [
            [[[2 * Dx_h / H]]],
            [[[2 * Dy_h / W]]],
            [[[2 * Dz_h / D]]],
        ]
    )
    # Control point value uniform at random between -max and max along every channel
    bspline_l.data = (torch.rand(*bspline_l.data.shape) - 0.5) * 2 * max_displacement_l
    bspline_h.data = (torch.rand(*bspline_h.data.shape) - 0.5) * 2 * max_displacement_h

    if locked_borders:
        bspline_l.data[:, 0, :, :] = 0
        bspline_l.data[:, nc_l - 1, :, :] = 0
        bspline_l.data[:, :, 0, :] = 0
        bspline_l.data[:, :, nc_l - 1, :] = 0
        bspline_l.data[:, :, :, 0] = 0
        bspline_l.data[:, :, :, nc_l - 1] = 0

        bspline_h.data[:, 0, :, :] = 0
        bspline_h.data[:, nc_h - 1, :, :] = 0
        bspline_h.data[:, :, 0, :] = 0
        bspline_h.data[:, :, nc_h - 1, :] = 0
        bspline_h.data[:, :, :, 0] = 0
        bspline_h.data[:, :, :, nc_h - 1] = 0

    def _make_base_grid_5d(
        theta: torch.Tensor, d: int, h: int, w: int, align_corners: bool
    ):
        dtype = theta.dtype
        device = theta.device

        grid_x = _linspace_from_neg_one(h, align_corners, dtype, device).view(
            1, 1, h, 1
        )
        grid_y = _linspace_from_neg_one(w, align_corners, dtype, device).view(
            1, w, 1, 1
        )
        grid_z = _linspace_from_neg_one(d, align_corners, dtype, device).view(
            d, 1, 1, 1
        )
        grid_one = torch.ones((1, 1, 1, 1), dtype=dtype, device=device)

        # this is just a temporary hack and we should use torch.stack here once #104480 is merged
        grid_x = torch.nn.functional.pad(grid_x, pad=(0, 3), mode="constant", value=0)
        grid_y = torch.nn.functional.pad(grid_y, pad=(1, 2), mode="constant", value=0)
        grid_z = torch.nn.functional.pad(grid_z, pad=(2, 1), mode="constant", value=0)
        grid_one = torch.nn.functional.pad(
            grid_one, pad=(3, 0), mode="constant", value=0
        )
        return grid_x + grid_y + grid_z + grid_one

    grid = _make_base_grid_5d(bspline_l.data, D, H, W, align_corners)
    # Convert grid from [-1, 1] (make_base_grid coordinates sytem) to [0, 1] (cubicbsplinegrid3D format)
    base_grid_01 = grid[..., 0:3] / 2 + 0.5  # from [-1, 1] to [0, 1]

    # Combining both displacement field
    displacement = bspline_l(base_grid_01) + bspline_h(base_grid_01)
    grid[..., 0:3] += displacement

    if affine is None:
        grid = grid[..., 0:3].view(1, D, W, H, 3)
        return grid
    else:
        # if composing the elastic deformation with an affine part as in our case
        # adapted from https://github.com/pytorch/pytorch/blob/9b3e34d8589b29f7b4e7fab6f78711b7ca6e4639/torch/_decomp/decompositions.py#L4278
        # affine should be (3, 4) in homogeneous coords for a single subject
        grid = (grid.float().view(-1, 4, 1) * affine.mT.unsqueeze(0)).sum(-2)
        # you can return that in the dataset getitem to be collated
        grid = grid.view(1, D, W, H, 3)

        return grid


# NOTE: from https://github.com/pytorch/pytorch/blob/9b57aa93fa76cbb069a05211c865a9df0323627a/torch/_decomp/decompositions.py#L4360
def make_base_grid_5d(
    d: int,
    h: int,
    w: int,
    align_corners: bool,
    dtype: torch.dtype,
    device: str | torch.device,
):
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, 1, w, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(1, h, 1, 1)
    grid_z = _linspace_from_neg_one(d, align_corners, dtype, device).view(d, 1, 1, 1)
    grid_one = torch.ones((1, 1, 1, 1), dtype=dtype, device=device)

    # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 3), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 2), mode="constant", value=0)
    grid_z = torch.nn.functional.pad(grid_z, pad=(2, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(3, 0), mode="constant", value=0)
    return grid_x + grid_y + grid_z + grid_one


def set_seed(seed: int, device: str, cudnn_benchmark: bool | None = None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                logger.warning(
                    (
                        "torch.backends.cudnn.benchmark was set to True which may"
                        " results in lack of reproducibility. In some cases to ensure"
                        " reproducibility you may need to set"
                        " torch.backends.cudnn.benchmark to False."
                    ),
                    UserWarning,
                )
        else:
            raise ValueError(
                f"cudnn_benchmark expected to be bool or None, got '{cudnn_benchmark}'"
            )
        torch.cuda.manual_seed_all(seed)


def setup_pytorch(random_seed: int | None = None) -> str | int:
    torch.multiprocessing.set_sharing_strategy("file_system")
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        # torch.backends.cudnn.benchmark = True
        logger.info(f"using cuda device {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"using {device} device")
    else:
        device = "cpu"
        logger.warning("no GPU available")
    if random_seed is not None:
        set_seed(seed=random_seed, device=device)
    # torch.set_default_device(f"{device}:0")
    return device


def _parse_acdc_info_cfg(info_path: str) -> tuple[int, int]:
    """Parse ED and ES frame numbers from ACDC Info.cfg file."""
    ed = es = None
    with open(info_path, "r") as f:
        for line in f:
            if "ED:" in line:
                ed = int(line.strip().split(":")[-1])
            elif "ES:" in line:
                es = int(line.strip().split(":")[-1])

    if ed is None or es is None:
        raise RuntimeError(f"Could not parse ED/ES from {info_path}")

    return ed, es


def _parse_mnms_ed_es_idx(csv_path: str, patient_id: str) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    row = df.loc[df["External code"] == patient_id]

    if row.empty:
        raise ValueError(f"Patient ID '{patient_id}' not found")

    ed_idx, es_idx = int(row["ED"].iloc[0]), int(row["ES"].iloc[0])

    return ed_idx, es_idx


def _parse_mnms2_ed_es_idx(csv_path: str, patient_id: str) -> tuple[int, int]:
    """
    Parse ED/ES frame indices for MnMs-2 from a saved correspondence CSV.

    Expected CSV columns (from your matching script):
        - pid
        - ED_idx
        - ES_idx

    Parameters
    ----------
    csv_path : str
        Path to the CSV file storing ED/ES indices.
    patient_id : str
        Patient ID (e.g. "201", "202", ...)

    Returns
    -------
    tuple[int, int]
        (ed_idx, es_idx) as 0-based frame indices in the CINE sequence.
    """
    df = pd.read_csv(csv_path)

    # Ensure consistent typing (patient IDs like "201")
    df["pid"] = df["pid"].astype(str)
    patient_id = str(patient_id)
    row = df.loc[df["pid"] == patient_id]

    if row.empty:
        raise ValueError(f"Patient ID '{patient_id}' not found in MnMs-2 CSV")

    ed_idx = int(row["ED_idx"].iloc[0])
    es_idx = int(row["ES_idx"].iloc[0])

    return ed_idx, es_idx


def flatten_pretraining_cfg(cfg):
    assert "data" in cfg and "pretraining" in cfg, (
        "The cfg dictionary must contain 'data' and 'pretraining' keys"
    )
    flattened_cfg = {
        **{f"data.{k}": v for k, v in cfg["data"].items()},
        **{f"pretraining.{k}": v for k, v in cfg["pretraining"].items()},
    }
    return flattened_cfg


def flatten_finetuning_cfg(cfg):
    assert "data" in cfg and "finetuning" in cfg, (
        "The cfg dictionary must contain 'data' and 'finetuning' keys"
    )
    flattened_cfg = {
        **{f"data.{k}": v for k, v in cfg["data"].items()},
        **{f"finetuning.{k}": v for k, v in cfg["finetuning"].items()},
    }
    return flattened_cfg


def aggregate_results(root_dir: str, experiment_prefix: str):
    """
    Aggregates mean_global_dice values for all experiment files matching the prefix.

    Args:
        root_dir (str): Directory containing the JSON files.
        experiment_prefix (str): For example: "metrics_finetune_pix2rep_BT_{exp_suffix}"

    Returns:
        pd.DataFrame: rows = runs, columns = np (num_patients) values.
    """

    # Regex to extract parameters from file names
    pattern = re.compile(rf"metrics_{experiment_prefix}_np=(\d+)_run(\d+)\.json$")

    # Mapping: results[np][run] = dice value
    results = defaultdict(dict)

    for filename in os.listdir(root_dir):
        match = pattern.match(filename)
        if not match:
            continue

        np_value = int(match.group(1))
        run_id = int(match.group(2))

        filepath = os.path.join(root_dir, filename)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            dice = data.get("mean_global_dice", None)
            if dice is not None:
                results[np_value][run_id] = dice

        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    # Convert nested dict to DataFrame (rows = runs, columns = num_patients)
    df = pd.DataFrame(results)

    # Sort columns (np values)
    df = df.sort_index(axis=1)

    # Sort rows (run numbers)
    df = df.sort_index(axis=0)
    df.index = [f"run{idx}" for idx in df.index]

    return df


def finetuning_logging_every_n_steps(num_patients: int):
    if num_patients <= 5:
        return 1
    elif 5 < num_patients <= 10:
        return 5
    elif 10 < num_patients <= 20:
        return 10
    else:
        return 50


class Config:
    def __init__(
        self,
        config_path: str = "../../config",
        config_name: str = "config",
        overrides: list[str] | None = None,
    ):
        self.config_name = config_name
        self.cfg = self.load_config(config_path, overrides)

    def load_config(self, config_path: str, overrides: list[str] | None = None):
        with initialize(config_path=config_path, version_base=None):
            cfg = compose(self.config_name, overrides=overrides or [])
        return cfg


class ModelCheckpointSymlink(ModelCheckpoint):
    """Creates/updates a symlink 'best.ckpt' pointing to the best model checkpoint."""

    def on_validation_end(self, trainer, pl_module):
        # Call standard checkpoint hook first
        super().on_validation_end(trainer, pl_module)

        # Run only on rank 0
        if not trainer.is_global_zero:
            return

        # If no best checkpoint yet --> skip
        if not self.best_model_path:
            return

        best_ckpt_path = os.path.join(self.dirpath, "best.ckpt")

        # Remove existing symlink if any
        if os.path.islink(best_ckpt_path) or os.path.exists(best_ckpt_path):
            try:
                os.remove(best_ckpt_path)
            except FileNotFoundError:
                pass  # in case another process deleted it first

        # Create symlink to current best checkpoint
        target = os.path.abspath(self.best_model_path)
        os.symlink(target, best_ckpt_path)
