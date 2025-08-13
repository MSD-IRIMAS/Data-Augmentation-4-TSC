"""Coverage metric."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class COVERAGE(BaseMetricCalculator):
    """The Coverage diversity metric [1]_ [2]_.

    The Coverage metric counts the number of real samples that
    include at least one generated sample. This work was
    proposed by [2]_, this implementation is motivated from
    there code in
    https://github.com/clovaai/generative-evaluation-prdc/
    Copyright (c) 2020-present NAVER Corp. MIT license

    Parameters
    ----------
    classifier : tf.keras.models.Model, default = None
        The keras classification model used as feature extractor,
        including the last layer of classification that will be
        automatically remvoved internally.
    batch_size : int, default = 128
        The batch size used for when computing latent
        representations.
    n_neighbors : int, default = 5
        The size of neighborhood around real samples.

    Returns
    -------
    None

    References
    ----------
    .. [1] Ismail-Fawaz, Ali, Maxime Devanne, Stefano Berretti,
    Jonathan Weber, and Germain Forestier. "Establishing a
    unified evaluation framework for human motion generation:
    A comparative analysis of metrics." Computer Vision and
    Image Understanding 254 (2025): 104337.

    .. [2] Naeem, Muhammad Ferjad, Seong Joon Oh, Youngjung
    Uh, Yunjey Choi, and Jaejun Yoo. "Reliable fidelity and
    diversity metrics for generative models." In International
    conference on machine learning, pp. 7176-7185. PMLR, 2020.
    """

    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
        n_neighbors: int = 5,
    ) -> None:
        self.n_neighbors = n_neighbors

        super().__init__(classifier=classifier, batch_size=batch_size)

    def _get_distances_k_neighbors(self, x: np.ndarray, k: int):
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X=x)

        distances_neighbors, _ = nn.kneighbors(X=x)

        return distances_neighbors[:, k - 1]

    def calculate(
        self,
        xgenerated: np.ndarray = None,
        ygenerated: np.ndarray = None,
        xreal: np.ndarray = None,
        yreal: np.ndarray = None,
    ):
        """Calculate the metric.

        Parameters
        ----------
        xgenerated : np.ndarray, default = None
            The generated samples.
        ygenerated : np.ndarray, default = None
            The labels of the generated samples.
            Not used.
        xreal : np.ndarray, default = None
            The real samples.
        yreal : np.ndarray, default = None
            The labels of the real samples.

        Returns
        -------
        coverage : float, value of the metric.
        """
        if xreal is not None:
            if xgenerated is None:
                xgenerated, xreal = train_test_split(
                    xreal, stratify=yreal, test_size=0.5
                )

        real_latent = self._to_latent(x=xreal)
        gen_latent = self._to_latent(x=xgenerated)
        if np.isnan(gen_latent).any():
            gen_latent = gen_latent[~np.isnan(gen_latent).any(axis=1)]

        real_gen_distance_matrix = pairwise_distances(X=real_latent, Y=gen_latent)
        real_distances_k_neighbors = self._get_distances_k_neighbors(
            x=real_latent, k=self.n_neighbors + 1
        )

        distances_nearest_neighbor_real_to_gen = np.min(
            real_gen_distance_matrix, axis=1
        )

        exists_inside_neighborhood = (
            distances_nearest_neighbor_real_to_gen < real_distances_k_neighbors
        )

        coverage = np.mean(exists_inside_neighborhood)

        return coverage
