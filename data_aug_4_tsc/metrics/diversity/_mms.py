"""Mean Maximum Similarity metric."""

import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class MMS(BaseMetricCalculator):
    """The Mean Maximum Similarity (MMS) diversity metric [1]_ [2]_.

    The MMS quantifies novelty/diversity by
    averaging the distances of each of the
    generated samples to its real nearest samples.

    Parameters
    ----------
    classifier : tf.keras.models.Model, default = None
        The keras classification model used as feature extractor,
        including the last layer of classification that will be
        automatically remvoved internally.
    batch_size : int, default = 128
        The batch size used for when computing latent
        representations.

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

    .. [2] Cervantes, Pablo, Yusuke Sekikawa, Ikuro Sato,
    and Koichi Shinoda. "Implicit neural representations
    for variable length human motion generation." In
    European Conference on Computer Vision, pp. 356-372.
    Cham: Springer Nature Switzerland, 2022.
    """

    def __init__(
        self, classifier: tf.keras.models.Model, batch_size: int = 128
    ) -> None:
        super().__init__(classifier=classifier, batch_size=batch_size)

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
            Not used.

        Returns
        -------
        coverage : float, value of the metric.
        """
        k = 1
        if xgenerated is None:
            xgenerated = xreal
            k = 2

        real_latent = self._to_latent(x=xreal)
        gen_latent = self._to_latent(x=xgenerated)
        if np.isnan(gen_latent).any():
            gen_latent = gen_latent[~np.isnan(gen_latent).any(axis=1)]

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X=real_latent)

        distances, _ = nn.kneighbors(X=gen_latent, return_distance=True)

        if distances.shape[-1] > 1:
            distances = distances[:, 1]

        return np.mean(distances)
