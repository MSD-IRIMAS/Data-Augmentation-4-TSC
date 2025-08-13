"""Average Pair Distance metric."""

import numpy as np
import tensorflow as tf

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class APD(BaseMetricCalculator):
    """The Average Pair Distance (APD) diversity metric [1]_ [2]_ [3]_.

    The APD metric can evaluate the diversity of
    any set of data, not necessarily generated ones.
    This is done through calculating the average
    Euclidean distance between randomly selected
    pairs of samples from the same dataset.

    Parameters
    ----------
    classifier : tf.keras.models.Model, default = None
        The keras classification model used as feature extractor,
        including the last layer of classification that will be
        automatically remvoved internally.
    batch_size : int, default = 128
        The batch size used for when computing latent
        representations.
    Sapd : int, default = 200
        The size of random selected set.
    runs : int, default = 5
        The number of random initialization of random
        selection of size Sapd.

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

    .. [2] Guo, Chuan, Xinxin Zuo, Sen Wang, Shihao Zou, Qingyao Sun,
    Annan Deng, Minglun Gong, and Li Cheng. "Action2motion:
    Conditioned generation of 3d human motions." In Proceedings
    of the 28th ACM international conference on multimedia, pp.
    2021-2029. 2020.

    .. [3] Zhang, Richard, Phillip Isola, Alexei A. Efros,
    Eli Shechtman, and Oliver Wang. "The unreasonable effectiveness
    of deep features as a perceptual metric." In Proceedings of
    the IEEE conference on computer vision and pattern recognition,
    pp. 586-595. 2018.
    """

    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
        Sapd: int = 200,
        runs: int = 5,
    ) -> None:
        self.Sapd = Sapd
        self.runs = runs

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
        apd : float, value of APD.
        """
        if xreal is not None:
            if xgenerated is None:
                xgenerated = xreal

        x_latent = self._to_latent(x=xgenerated)
        if np.isnan(x_latent).any():
            x_latent = x_latent[~np.isnan(x_latent).any(axis=1)]

        if self.Sapd > len(x_latent):
            self._Sapd = len(x_latent)
        else:
            self._Sapd = self.Sapd

        apd_values = []

        for _ in range(self.runs):
            all_indices = np.arange(len(x_latent))

            V = x_latent[np.random.choice(a=all_indices, size=self._Sapd)]
            V_prime = x_latent[np.random.choice(a=all_indices, size=self._Sapd)]

            apd_values.append(np.mean(np.linalg.norm(V - V_prime, axis=1)))

        apd = np.mean(apd_values)

        return apd
