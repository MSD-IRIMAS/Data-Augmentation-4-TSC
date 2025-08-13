"""Random guided warping augmentation method."""

import numpy as np
from aeon.distances import create_bounding_matrix
from aeon.distances.elastic._alignment_paths import compute_min_return_path
from aeon.distances.elastic._dtw import _dtw_cost_matrix
from aeon.distances.elastic._msm import _msm_independent_cost_matrix
from aeon.distances.elastic._shape_dtw import _pad_ts_edges, _shape_dtw_cost_matrix
from numba import njit


def random_guided_warping(
    X: np.ndarray,
    y: np.ndarray = None,
    distance: str = "dtw",
    window: float = None,
    reach: int = 15,
    c: float = 1.0,
):
    """Warp the input time series using the warping path with a random series.

    Proposed in [1], using the warping path between each series
    and another randomly selected, re-sample the input time series
    with the alignment path.

    Parameters
    ----------
    X: tf.Tensor, shape (batch_size, length_TS, n_channels)
        The input set of time series.
    y: np.ndarray, shape (batch_size,)
        The labels of each input time series. Ignored, here for
        code structure reasons.
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

    augmented_X = _random_guided_warping(
        X=_X, y=y, distance=distance, window=window, reach=reach, c=c
    )

    return np.swapaxes(augmented_X, axis1=1, axis2=2)


@njit(nopython=True, fastmath=True, parallel=True, cache=True)
def _random_guided_warping(
    X: np.ndarray,
    y: np.ndarray = None,
    distance: str = "dtw",
    window: float = None,
    reach: int = 15,
    c: float = 1.0,
):
    batch_size = int(X.shape[0])
    length_TS = int(X.shape[2])
    n_channels = int(X.shape[1])

    X_agumented = np.zeros_like(X)

    bounding_matrix = create_bounding_matrix(
        length_TS, length_TS, window=window, itakura_max_slope=None
    )

    for i in range(batch_size):
        choices = np.delete(np.arange(batch_size), i)
        randomly_chosen_series = X[np.random.choice(choices)]

        if distance == "dtw":
            cost_matrix = _dtw_cost_matrix(
                x=X[i],
                y=randomly_chosen_series,
                bounding_matrix=bounding_matrix,
            )
            warping_path = compute_min_return_path(cost_matrix)
        elif distance == "msm":
            cost_matrix = _msm_independent_cost_matrix(
                x=X[i],
                y=randomly_chosen_series,
                bounding_matrix=bounding_matrix,
                c=c,
            )
            warping_path = compute_min_return_path(cost_matrix)
        elif distance == "shape_dtw":
            Xi_pad = _pad_ts_edges(x=X[i], reach=reach)
            randomly_chosen_series_pad = _pad_ts_edges(
                x=randomly_chosen_series, reach=reach
            )

            cost_matrix = _shape_dtw_cost_matrix(
                x=Xi_pad,
                y=randomly_chosen_series_pad,
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
