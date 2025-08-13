"""Fréchet Inception Distance metric."""

import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class FID(BaseMetricCalculator):
    """The Fréchet Inception Distance (FID) fidelity metric [1]_ [2]_.

    The FID is calculated between the Cumulative Distribution
    Functions (CDFs) of both real and generated distributions.

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

    .. [2] Heusel, Martin, Hubert Ramsauer, Thomas Unterthiner,
    Bernhard Nessler, and Sepp Hochreiter. "Gans trained by a
    two time-scale update rule converge to a local nash
    equilibrium." Advances in neural information processing
    systems 30 (2017).
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
        xreal : np.ndarray, default = None
            The real samples.
        yreal : np.ndarray, default = None
            The labels of the real samples.

        Returns
        -------
        fid : float, value of FID.
        """
        if xreal is not None:
            if xgenerated is None:
                xgenerated, xreal = train_test_split(
                    xreal, stratify=yreal, test_size=0.5
                )

        real_latent = self._to_latent(x=xreal)

        mean_real = np.mean(real_latent, axis=0)
        cov_real = np.cov(real_latent, rowvar=False)

        gen_latent = self._to_latent(x=xgenerated)
        if np.isnan(gen_latent).any():
            gen_latent = gen_latent[~np.isnan(gen_latent).any(axis=1)]

        mean_gen = np.mean(gen_latent, axis=0)
        cov_gen = np.cov(gen_latent, rowvar=False)

        diff_means = np.sum(np.square(mean_real - mean_gen))
        cov_prod = sqrtm(cov_real.dot(cov_gen))

        if np.iscomplexobj(cov_prod):
            cov_prod = cov_prod.real

        fid = diff_means + np.trace(cov_real + cov_gen - 2.0 * cov_prod)

        return fid
