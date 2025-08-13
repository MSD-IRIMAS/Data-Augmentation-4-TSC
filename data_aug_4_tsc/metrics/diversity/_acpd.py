"""Average per Class Pair Distance metric."""

import numpy as np
import tensorflow as tf

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class ACPD(BaseMetricCalculator):
    """The Average per Class Pair Distance (ACPD) diversity metric [1]_ [2]_ [3]_.

    Similar to the APD, the Average per Class Pair
    Distance ACPD evaluates the diversity of the
    generated samples, with the same interpretation as
    for the APD.

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
        xreal : np.ndarray, default = None
            The real samples.
        yreal : np.ndarray, default = None
            The labels of the real samples.

        Returns
        -------
        acpd : float, value of ACPD.
        """
        if xreal is not None:
            if xgenerated is None:
                xgenerated = xreal
                ygenerated = yreal

        self.n_classes = len(np.unique(ygenerated))
        acpd_values = []

        for c in range(self.n_classes):
            apd_per_class_values = []
            x_c = xgenerated[ygenerated == c]

            x_c_latent = self._to_latent(x=x_c)
            if np.isnan(x_c_latent).any():
                x_c_latent = x_c_latent[~np.isnan(x_c_latent).any(axis=1)]

            if self.Sapd > len(x_c_latent):
                self._Sapd = len(x_c_latent)
            else:
                self._Sapd = self.Sapd

            for _ in range(self.runs):
                all_indices = np.arange(len(x_c_latent))

                V = x_c_latent[np.random.choice(a=all_indices, size=self._Sapd)]
                V_prime = x_c_latent[np.random.choice(a=all_indices, size=self._Sapd)]

                apd_per_class_values.append(
                    np.mean(np.linalg.norm(V - V_prime, axis=1))
                )

            acpd_values.append(np.mean(apd_per_class_values))

        acpd = np.mean(acpd_values)

        return acpd
