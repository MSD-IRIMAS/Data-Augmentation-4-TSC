"""Test if augmentation functions do not crash."""

import inspect

import numpy as np
import pytest

from data_aug_4_tsc import data_augs

_data_aug_functions = [
    member[1]
    for member in inspect.getmembers(data_augs, lambda x: callable(x))
    if member[0] != "_get_data_augmentation_function"
    and member[0] != "_pad_ts_collection_edges"
    and member[0] != "_random_guided_warping"
    and member[0] != "_discriminative_guided_warping"
]


@pytest.mark.parametrize("data_augmentation_function", _data_aug_functions)
def test_all_data_augmentations(data_augmentation_function):
    """Test all data augmentation functions."""
    X = np.random.normal(size=(5, 7, 2))
    y = np.array([0, 0, 1, 1, 1])

    X_augmented = data_augmentation_function(X, y=y)

    assert X_augmented is not None
    assert X_augmented.shape[0] == X.shape[0]
    assert X_augmented.shape[1] == X.shape[1]
    assert X_augmented.shape[2] == X.shape[2]


if __name__ == "__main__":
    from data_aug_4_tsc.data_augs import random_guided_warping

    test_all_data_augmentations(random_guided_warping)
