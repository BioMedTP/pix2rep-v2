import itertools
import warnings
from collections import defaultdict
from typing import Callable, Sequence, Union

import numpy as np
import torchio as tio
from connectome import Input, Mixin, Transform, inverse, optional, positional
from imops import crop_to_box, pad, zoom, zoom_to_shape
from numpy.lib.array_utils import normalize_axis_tuple
from skimage.segmentation import flood
from torchio.data.subject import Subject
from torchio.transforms import Blur, IntensityTransform
from torchio.transforms.augmentation import RandomTransform

from pix2repv2.utils.default_params import BODY_THRESHOLD_HU


class RandomInvert(tio.Transform):
    """Reimplementation of torchvision RandomInvert augmentation under the torchIO format.
    Image intensities are clamped to prevent negative values after inversion.
    """

    def __init__(self, p: float, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__(p=p)
        self.min_val = min_val
        self.max_val = max_val

    def apply_transform(self, subject):
        for image in subject.get_images(intensity_only=True):
            x = image.data
            x = x.clamp(self.min_val, self.max_val)
            image.set_data(self.max_val - x)
        return subject


class RandomRescaleIntensity(tio.Transform):
    def __init__(
        self,
        Imin_range=(0.0, 0.3),
        Imax_range=(0.7, 1.0),
        percentiles=(1, 99),
    ):
        super().__init__()
        self.Imin_range = Imin_range
        self.Imax_range = Imax_range
        self.percentiles = percentiles

    def apply_transform(self, subject):
        Imin = np.random.uniform(*self.Imin_range)
        Imax = np.random.uniform(*self.Imax_range)

        # Safety: ensure valid ordering
        if Imin >= Imax:
            return subject

        rescale = tio.RescaleIntensity(
            out_min_max=(Imin, Imax),
            percentiles=self.percentiles,
        )

        return rescale(subject)


class RandomBlurOrSharpen(RandomTransform, IntensityTransform):
    """Randomly blur or sharpen an image.

    Sharpening is implemented as an unsharp mask using TorchIO's Blur transform:
        output = image + alpha * (image - blurred_image)
    """

    def __init__(
        self,
        p: float,
        std: Union[float, tuple[float, float]] = (0.25, 1.5, 0.25, 1.5, 0.0, 0.0),
        alpha: tuple[float, float] = (0.2, 0.99),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.std_ranges = self.parse_params(std, None, "std", min_constraint=0)
        self.alpha_range = self._parse_range(alpha, "alpha")

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        if self.sample_uniform(0, 1) > self.p:
            return subject

        # Decide once per subject: blur OR sharpen with equal chance
        do_blur = self.sample_uniform(0, 1) < 0.5

        arguments: dict[str, dict] = defaultdict(dict)
        alpha_dict: dict[str, float] = {}

        for name in images_dict:
            std = self.get_params(self.std_ranges)
            arguments["std"][name] = std

            if not do_blur:
                alpha_dict[name] = self.sample_uniform(*self.alpha_range)

        if do_blur:
            transform = Blur(**self.add_base_args(arguments))
            transformed = transform(subject)
        else:
            # --- Unsharp mask ---
            blurred = Blur(**self.add_base_args(arguments))(subject)
            assert isinstance(blurred, Subject)

            for name, image in images_dict.items():
                # original = subject[name].data
                original = image.data
                blurred_data = blurred[name].data
                # alpha = arguments["alpha"][name]
                alpha = alpha_dict[name]

                image.set_data(original + alpha * (original - blurred_data))

            transformed = subject

        assert isinstance(transformed, Subject)
        return transformed

    def get_params(self, std_ranges):
        sx, sy, sz = self.sample_uniform_sextet(std_ranges)
        return sx, sy, sz


# NOTE: all following augmentations have been taken from Vox2Vec repository (cf. https://github.com/mishgon/vox2vec)
class FlipAxesToCanonical(Transform):
    __inherit__ = True

    @positional
    def image(x, flipped_axes):
        if not flipped_axes:
            return x
        return np.flip(x, flipped_axes).copy()

    @optional
    @positional
    def mask(x, flipped_axes):
        if not flipped_axes or x is None:
            return x
        return np.flip(x, flipped_axes).copy()

    def flipped_axes(flipped_axes):
        return ()

    @inverse
    def sgm(sgm, flipped_axes: Input):
        if not flipped_axes:
            return sgm
        return np.flip(sgm, flipped_axes).copy()


class _Rescale(Mixin):
    def image(image, _scale_factor, _axis, _image_fill_value):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return zoom(
                image,
                _scale_factor,
                _axis,
                order=3,
                fill_value=_image_fill_value,
                backend="Scipy",
            )

    @optional
    def spacing(spacing, _scale_factor, _axis):
        axis = normalize_axis_list(_axis, len(spacing))
        spacing = np.array(spacing)
        spacing[axis] /= _scale_factor
        return tuple(spacing)

    @optional
    def shape(shape, _scale_factor, _axis):
        axis = normalize_axis_list(_axis, len(shape))
        shape = np.array(shape)
        shape[axis] = np.round(shape[axis] * _scale_factor).astype(int)
        return tuple(shape)

    @optional
    @positional
    def mask(x, _scale_factor, _axis):
        if x is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return zoom(x, _scale_factor, _axis, order=0, backend="Scipy")

    body_mask = mask

    @inverse
    def sgm(sgm, image: Input, _axis, _sgm_interp_ord):
        shape = np.array(image.shape)[normalize_axis_list(_axis, image.ndim)]
        return zoom_to_shape(sgm, shape, _axis, order=_sgm_interp_ord, backend="Scipy")


class RescaleToSpacing(Transform, _Rescale):
    __inherit__ = "id"
    _to_spacing: Union[float, Sequence[float]]
    _axis: Union[int, Sequence[int]]
    _image_fill_value: Union[float, Callable]
    _sgm_interp_ord: int = 3

    def _scale_factor(spacing, _to_spacing, _axis):
        axis = normalize_axis_list(_axis, len(spacing))
        old_spacing = np.array(spacing, dtype=float)[axis]
        return np.nan_to_num(old_spacing / _to_spacing, nan=1)


class _CropToBox(Mixin):
    @positional
    def image(x, _box, _axis):
        if x is not None:
            return crop_to_box(x, _box, _axis)

    mask = body_mask = optional(image)

    def shape(_box):
        return tuple(_box[1] - _box[0])

    @inverse
    def sgm(sgm, image: Input, _box, _axis):
        shape = np.array(image.shape)[normalize_axis_list(_axis, image.ndim)]
        return restore_crop(sgm, _box, shape, _axis)


class CropToBox(Transform, _CropToBox):
    __inherit__ = "spacing", "id"
    _axis: Union[Sequence[int], int]

    def _box(cropping_box):
        return cropping_box


def hu_clip(
    image: np.ndarray,
    min_window_hu: float,
    max_window_hu: float,
) -> np.ndarray:
    image = (image - min_window_hu) / (max_window_hu - min_window_hu)
    return np.clip(image, 0.0, 1.0)


def get_body_mask(image: np.ndarray) -> np.ndarray:
    air = image < BODY_THRESHOLD_HU
    return ~flood(
        pad(air, padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0, 0)
    )[1:-1, 1:-1]


def normalize_axis_list(axis, ndim):
    return list(normalize_axis_tuple(axis, ndim))


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError("The mask is empty.")

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)

    return np.array([start, stop])


def restore_crop(x, box, shape, axis, padding_values=0):
    axis = normalize_axis_list(axis, x.ndim)

    assert box.shape[1] == len(shape) == len(axis)

    padding = np.array([box[0], shape - box[1]], dtype=int).T
    return pad(x, padding, axis, padding_values)
