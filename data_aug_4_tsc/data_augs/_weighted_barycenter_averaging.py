"""Weighted Barycenter Averaging Method."""

import math
from typing import Dict, Optional

import numpy as np
from aeon.clustering.averaging import elastic_barycenter_average
from aeon.distances import pairwise_distance


# @njit(nopython=True, fastmath=True, parallel=True, cache=True)
def weighted_barycenter_averaging(
    X: np.ndarray,
    y: np.ndarray,
    distance: str = "dtw",
    distance_params: Optional[dict] = None,
):
    """Find the weighted barycenter average between a set of series.

    Proposed in [1], using the Dynamic Time Warping Barycenter Averaging
    algorithm [2] or ShapeDTW Barycenter Averaging [3,4], find the
    weighted prototype to augment one sample using its neighbors in a
    batch.

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
    distance_params: dict, default = None
        The parameters of the distance function used.

    Returns
    -------
    np.ndarray, shape (batch_size, length_TS, n_channels)
        The augmented set of series containing weighted prototypes.

    References
    ----------
    .. [1] Forestier, G., Petitjean, F., Dau, H. A., Webb, G. I., & Keogh, E.
       (2017, November). Generating synthetic time series to augment sparse
       datasets. In 2017 IEEE international conference on data mining (ICDM)
       (pp. 865-870). IEEE.
    .. [2] F. Petitjean, A. Ketterlin & P. Gancarski. A global averaging method
       for dynamic time warping, with applications to clustering. Pattern
       Recognition, Elsevier, 2011, Vol. 44, Num. 3, pp. 678-693
    .. [3] Ismail-Fawaz, Ali, Hassan Ismail Fawaz, François Petitjean,
       Maxime Devanne, Jonathan Weber, Stefano Berretti, Geoffrey I. Webb,
       and Germain Forestier. "ShapeDBA: generating effective time series
       prototypes using shapeDTW barycenter averaging." In International
       Workshop on Advanced Analytics and Learning on Temporal Data, pp.
       127-142. Cham: Springer Nature Switzerland, 2023.
    .. [4] Ismail-Fawaz, Ali, Maxime Devanne, Stefano Berretti, Jonathan Weber,
       and Germain Forestier. "Weighted Average of Human Motion Sequences for
       Improving Rehabilitation Assessment." In International Workshop on
       Advanced Analytics and Learning on Temporal Data, pp. 131-146. Cham:
       Springer Nature Switzerland, 2024.
    """
    batch_size = len(X)

    X_augmented = np.zeros_like(X)

    for i in range(batch_size):
        indices_samples_from_same_class = np.where(
            (y == y[i]) & (np.arange(len(y)) != i)
        )[0]

        if len(indices_samples_from_same_class) == 0:
            X_augmented[i] = X[i]
        else:
            samples_from_same_class = X[
                np.random.choice(
                    indices_samples_from_same_class,
                    min(4, len(indices_samples_from_same_class)),
                    replace=False,
                )
            ]

            X_augmented[i] = _augment_one_series(
                X=samples_from_same_class,
                reference_series=X[i],
                distance=distance,
                distance_params=distance_params,
            )

    return X_augmented


def _augment_one_series(
    X: np.ndarray,
    reference_series: np.ndarray,
    distance: str = "dtw",
    distance_params: Optional[Dict] = None,
):
    if distance_params is not None:
        pairwise_distances_to_reference = pairwise_distance(
            x=np.swapaxes(X, axis1=1, axis2=2),
            y=np.expand_dims(np.swapaxes(reference_series, axis1=0, axis2=1), axis=0),
            method=distance,
            **distance_params,
        )
    else:
        pairwise_distances_to_reference = pairwise_distance(
            x=np.swapaxes(X, axis1=1, axis2=2),
            y=np.expand_dims(np.swapaxes(reference_series, axis1=0, axis2=1), axis=0),
            method=distance,
        )
    neighbors_indices = np.argsort(pairwise_distances_to_reference, axis=0)

    _distance_to_nearest = pairwise_distances_to_reference[neighbors_indices[0]]
    _distance_to_nearest = _distance_to_nearest[0, 0]

    weights = np.zeros(shape=(len(X),))

    weights[neighbors_indices[0]] = 0.5

    for i in range(1, len(X)):

        _distance_from_i_to_ref = pairwise_distances_to_reference[neighbors_indices[i]]
        _distance_from_i_to_ref = _distance_from_i_to_ref[0, 0]

        _weight = math.exp(
            math.log(0.5) * _distance_from_i_to_ref / _distance_to_nearest
        )

        weights[neighbors_indices[i]] = _weight

    X_combined = np.concatenate(
        (np.expand_dims(reference_series, axis=0), X[neighbors_indices[:, 0]]), axis=0
    )
    weights_combined = np.concatenate((np.array([1.0]), weights), axis=0)

    if distance_params is not None:

        _distance_params = distance_params.copy()

        augmented_series = elastic_barycenter_average(
            X=np.swapaxes(X_combined, axis1=1, axis2=2),
            distance=distance,
            init_barycenter=np.swapaxes(reference_series, axis1=0, axis2=1),
            weights=weights_combined,
            **_distance_params,
        )
    else:
        augmented_series = elastic_barycenter_average(
            X=np.swapaxes(X_combined, axis1=1, axis2=2),
            distance=distance,
            init_barycenter=np.swapaxes(reference_series, axis1=0, axis2=1),
            weights=weights_combined,
        )

    return np.swapaxes(augmented_series, axis1=0, axis2=1)
