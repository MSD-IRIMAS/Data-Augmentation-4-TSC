"""Accuracy On Generated metric."""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from data_aug_4_tsc.metrics.base import BaseMetricCalculator


class AOG(BaseMetricCalculator):
    """The Accuracy On Generated (AOG) fidelity metric [1]_.

    The value of the AOG metric is a classifier's
    ability to classify generated samples to their
    intended class label used for generaiton.

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
    """

    def __init__(
        self,
        classifier: tf.keras.models.Model,
        batch_size: int = 128,
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
        aog : float, value of AOG.
        """
        if xreal is not None and yreal is not None:
            if xgenerated is None and ygenerated is None:
                xgenerated = xreal
                ygenerated = yreal

        ypred = self.classifier.predict(xgenerated, batch_size=self.batch_size)
        ypred = np.argmax(ypred, axis=1)

        aog = accuracy_score(y_true=ygenerated, y_pred=ypred, normalize=True)

        return aog
