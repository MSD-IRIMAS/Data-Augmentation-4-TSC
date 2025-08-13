"""Discriminative guided warping augmentation method."""

import numpy as np
from aeon.distances import create_bounding_matrix
from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._dtw import (
    _dtw_cost_matrix,
    _dtw_from_multiple_to_multiple_distance,
    _dtw_pairwise_distance,
)
from aeon.distances.elastic._msm import (
    _msm_from_multiple_to_multiple_distance,
    _msm_independent_cost_matrix,
    _msm_pairwise_distance,
)
from aeon.distances.elastic._shape_dtw import (
    _pad_ts_edges,
    _shape_dtw_cost_matrix,
    _shape_dtw_from_multiple_to_multiple_distance,
    _shape_dtw_pairwise_distance,
)
from numba import njit


def discriminative_guided_warping(
    X: np.ndarray,
    y: np.ndarray,
    distance: str = "dtw",
    window: float = None,
    reach: int = 15,
    c: float = 1.0,
):
    """Warp the input time series using the warping path with a discriminative series.

    Proposed in [1], using the warping path between each series
    and another discriminative selected, re-sample the input time series
    with the alignment path. The discriminative series is chosen in such way
    that it has the maximum distance between itself and its positive and
    negative neighbors.

    Parameters
    ----------
    X: np.ndarray, shape (batch_size, length_TS, n_channels)
        The input set of time series.
    y: np.ndarray, shape (batch_size,)
        The labels of each input time series.
    distance: str, default = "dtw"
        The distance measure used for the elastic barycenter.
        Default behavior will be using Dynamic Time Warping.
        Possible distances:
        ============================================================
        distance           name
        ============================================================
        dtw                Dynamic Time Warping
        adtw               Amerced Dynamic Time Warping
        erp                Edit Real Penalty
        edr                Edit distance for real sequences
        euclidean          Euclidean Distance
        LCSS               Longest Common Subsequence
        manhattan          Manhattan Distance
        minkowski          Minkowski Distance
        msm                Move-Split-Merge
        sbd                Shape-based Distance
        shape_dtw          Shape Dynamic Time Warping
        squared            Squared Distance
        twe                Time Warp Edit
        wddtw              Weighted Derivative Dynamic Time Warping
        wdtw               Weighted Dynamic Time Warping
        ============================================================
    window : float or None, default=None
        For DTW, the window to use for the bounding matrix.
        If None, no bounding matrix is used. window is a
        percentage deviation, so if ``window = 0.1`` then
        10% of the series length is the max warping allowed.
        is used.
    reach : int, default=15.
        For ShapeDTW, length of the sub-sequences. Default is 15,
        meaning a sliding window of length 31 centered at a point.
    c : float, default=1.
        For MSM, cost for split or merge operation. Default is 1.

    Returns
    -------
    np.ndarray, shape (batch_size, length_TS, n_channels)
        The warped series.

    References
    ----------
    [1] Iwana, Brian Kenji, and Seiichi Uchida. "Time series data
        augmentation for neural networks by time warping with a
        discriminative teacher." In 2020 25th International
        Conference on Pattern Recognition (ICPR), pp. 3558-3565. IEEE, 2021.
    """
    _X = np.swapaxes(X, axis1=1, axis2=2)

    augmented_X = _discriminative_guided_warping(
        X=_X, y=y, distance=distance, window=window, reach=reach, c=c
    )

    return np.swapaxes(augmented_X, axis1=1, axis2=2)


@njit(nopython=True, fastmath=True, parallel=True, cache=True)
def _discriminative_guided_warping(
    X: np.ndarray,
    y: np.ndarray,
    distance: str = "dtw",
    window: float = None,
    reach: int = 15,
    c: float = 1.0,
):
    batch_size = int(X.shape[0])
    length_TS = int(X.shape[2])
    n_channels = int(X.shape[1])

    bounding_matrix = create_bounding_matrix(
        length_TS, length_TS, window=window, itakura_max_slope=None
    )

    X_agumented = np.zeros_like(X)

    for i in range(batch_size):
        positive_choices = np.where((y == y[i]) & (np.arange(batch_size) != i))[0]
        negative_choices = np.where(y != y[i])[0]

        pos_numbers = min(5, len(positive_choices))
        neg_numbers = min(5, len(negative_choices))

        if pos_numbers == 0 or neg_numbers == 0:
            X_agumented[i] = X[i]
            continue

        positive_prototypes = X[
            np.random.choice(positive_choices, pos_numbers, replace=False)
        ]
        negative_prototypes = X[
            np.random.choice(negative_choices, neg_numbers, replace=False)
        ]

        pos_distances = np.zeros(shape=(pos_numbers,))
        neg_distances = np.zeros(shape=(neg_numbers,))

        if distance == "dtw":
            pos_distance_matrix = _dtw_pairwise_distance(
                X=positive_prototypes,
                window=window,
                itakura_max_slope=None,
                unequal_length=False,
            )
            neg_distance_matrix = _dtw_from_multiple_to_multiple_distance(
                x=positive_prototypes,
                y=negative_prototypes,
                window=window,
                itakura_max_slope=None,
                unequal_length=False,
            )
        elif distance == "msm":
            pos_distance_matrix = _msm_pairwise_distance(
                X=positive_prototypes,
                window=window,
                independent=True,
                c=c,
                itakura_max_slope=None,
                unequal_length=False,
            )
            neg_distance_matrix = _msm_from_multiple_to_multiple_distance(
                x=positive_prototypes,
                y=negative_prototypes,
                window=window,
                independent=True,
                c=c,
                itakura_max_slope=None,
                unequal_length=False,
            )
        elif distance == "shape_dtw":
            pos_distance_matrix = _shape_dtw_pairwise_distance(
                X=positive_prototypes,
                window=window,
                descriptor="identity",
                reach=reach,
                itakura_max_slope=None,
                transformation_precomputed=False,
                transformed_x=None,
                unequal_length=False,
            )

            neg_distance_matrix = _shape_dtw_from_multiple_to_multiple_distance(
                x=positive_prototypes,
                y=negative_prototypes,
                window=window,
                descriptor="identity",
                reach=reach,
                itakura_max_slope=None,
                transformation_precomputed=False,
                transformed_x=None,
                transformed_y=None,
                unequal_length=False,
            )

        pos_distances = (1 / (max(1.0, pos_numbers - 1))) * np.sum(
            pos_distance_matrix, axis=1
        )
        neg_distances = (1 / (neg_numbers)) * np.sum(neg_distance_matrix, axis=1)

        neg_diff_pos_distances = neg_distances - pos_distances

        discriminative_series = positive_prototypes[np.argmax(neg_diff_pos_distances)]

        if distance == "dtw":
            cost_matrix = _dtw_cost_matrix(
                x=X[i],
                y=discriminative_series,
                bounding_matrix=bounding_matrix,
            )
            warping_path = compute_min_return_path(cost_matrix)
        elif distance == "msm":
            cost_matrix = _msm_independent_cost_matrix(
                x=X[i],
                y=discriminative_series,
                bounding_matrix=bounding_matrix,
                c=c,
            )
            warping_path = compute_min_return_path(cost_matrix)
        elif distance == "shape_dtw":
            Xi_pad = _pad_ts_edges(x=X[i], reach=reach)
            discriminative_series_pad = _pad_ts_edges(
                x=discriminative_series, reach=reach
            )

            cost_matrix = _shape_dtw_cost_matrix(
                x=Xi_pad,
                y=discriminative_series_pad,
                bounding_matrix=bounding_matrix,
                reach=reach,
            )
            warping_path = compute_min_return_path(cost_matrix)

        transformation_indices = np.array([x[0] for x in warping_path])

        for dim in range(n_channels):
            X_agumented[i, dim, :] = np.interp(
                np.arange(length_TS),
                np.linspace(0, length_TS - 1, num=len(transformation_indices)),
                X[i, dim, transformation_indices],
            )

    return X_agumented
