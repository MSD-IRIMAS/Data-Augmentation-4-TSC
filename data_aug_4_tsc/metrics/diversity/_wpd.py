"""Warping Path Diversity metric."""

import numpy as np
import tensorflow as tf
from aeon.distances import dtw_alignment_path

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class WPD(BaseMetricCalculator):
    """The Warping Path Diversity (WPD) diversity metric [1]_.

    The value of the WPD metric is the average
    distance of a warping path to the diagonal (perfect
    warping) between two samples.

    Parameters
    ----------
    classifier : tf.keras.models.Model, default = None
        The keras classification model used as feature extractor,
        including the last layer of classification that will be
        automatically remvoved internally.
    batch_size : int, default = 128
        The batch size used for when computing latent
        representations.
    Swpd : int, default = 200
        The size of random selected set.
    runs : int, default = 5
        The number of random initialization of random
        selection of size Swpd.

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
    """

    def __init__(
        self,
        classifier: tf.keras.models.Model = None,
        batch_size: int = 128,
        Swpd: int = 200,
        runs: int = 5,
    ) -> None:
        self.Swpd = Swpd
        self.runs = runs

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
        wpd : float, value of WPD.
        """
        if xreal is not None:
            if xgenerated is None:
                xgenerated = xreal

        if np.isnan(xgenerated).any():
            xgenerated = xgenerated[~np.isnan(xgenerated).any(axis=(1, 2))]

        if self.Swpd > len(xgenerated):
            self._Swpd = len(xgenerated)
        else:
            self._Swpd = self.Swpd

        wpd_values = []

        if len(xgenerated.shape) > 3:
            xgenerated = np.reshape(
                xgenerated, (xgenerated.shape[0], xgenerated.shape[1], -1)
            )

        for _ in range(self.runs):
            all_indices = np.arange(len(xgenerated))

            G = xgenerated[np.random.choice(a=all_indices, size=self._Swpd)]
            G_prime = xgenerated[np.random.choice(a=all_indices, size=self._Swpd)]

            for i in range(self._Swpd):
                dtw_path, dtw_dist = dtw_alignment_path(
                    x=np.swapaxes(G[i], axis1=0, axis2=1),
                    y=np.swapaxes(G_prime[i], axis1=0, axis2=1),
                )
                dtw_path = np.asarray(dtw_path)

                wpd_values.append(
                    (np.sqrt(2) / (2 * len(dtw_path)))
                    * np.sum(np.abs(dtw_path[:, 0] - dtw_path[:, 1]))
                )

        wpd = np.mean(wpd_values)

        return wpd
