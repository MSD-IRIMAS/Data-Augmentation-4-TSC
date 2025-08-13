"""Data augmentation functions for time series data."""

__all__ = [
    "scale",
    "add_noise",
    "amplitude_warping",
    "window_warping",
    "random_guided_warping",
    "_random_guided_warping",
    "discriminative_guided_warping",
    "_discriminative_guided_warping",
    "weighted_barycenter_averaging",
    "_get_data_augmentation_function",
]

from data_aug_4_tsc.data_augs._amplitude_warping import amplitude_warping
from data_aug_4_tsc.data_augs._data_augs import _get_data_augmentation_function
from data_aug_4_tsc.data_augs._discriminative_guided_warping import (
    _discriminative_guided_warping,
    discriminative_guided_warping,
)
from data_aug_4_tsc.data_augs._jittering import add_noise
from data_aug_4_tsc.data_augs._random_guided_warping import (
    _random_guided_warping,
    random_guided_warping,
)
from data_aug_4_tsc.data_augs._scaling import scale
from data_aug_4_tsc.data_augs._weighted_barycenter_averaging import (
    weighted_barycenter_averaging,
)
from data_aug_4_tsc.data_augs._window_warping import window_warping
