"""Utilities for data augmentation methods."""

__all__ = ["_get_data_augmentation_function"]

from data_aug_4_tsc.data_augs._amplitude_warping import amplitude_warping
from data_aug_4_tsc.data_augs._discriminative_guided_warping import (
    discriminative_guided_warping,
)
from data_aug_4_tsc.data_augs._jittering import add_noise
from data_aug_4_tsc.data_augs._random_guided_warping import random_guided_warping
from data_aug_4_tsc.data_augs._scaling import scale
from data_aug_4_tsc.data_augs._weighted_barycenter_averaging import (
    weighted_barycenter_averaging,
)
from data_aug_4_tsc.data_augs._window_warping import window_warping


def _get_data_augmentation_function(method: str):
    if method == "Scaling":
        return scale
    elif method == "Jittering":
        return add_noise
    elif method == "WW":
        return window_warping
    elif method == "AW":
        return amplitude_warping
    elif method == "RGW":
        return random_guided_warping
    elif method == "DGW":
        return discriminative_guided_warping
    elif method == "WBA":
        return weighted_barycenter_averaging
