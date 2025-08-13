"""Base class metric calculator."""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseMetricCalculator(ABC):
    """Base model for metric calculation.

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
    """

    def __init__(self, classifier=None, batch_size=128):
        self.classifier = classifier
        self.feature_extractor = self.remove_classification_layer(model=self.classifier)
        self.batch_size = batch_size

    def remove_classification_layer(self, model):
        """Remove classification layer from self.classifier.

        Parameters
        ----------
        model : tf.keras.models.Model, default = None
            A keras classification model.

        Returns
        -------
        new_model : tf.keras.models.Model
            The same model without the classificaiton head.

        """
        input_model = model.input
        output_model = model.layers[-2].output

        new_model = tf.keras.models.Model(inputs=input_model, outputs=output_model)

        return new_model

    def _split_to_batches(self, x):
        n = int(x.shape[0])

        if self.batch_size > n:
            return [x]

        batches = []

        for i in range(0, n - self.batch_size + 1, self.batch_size):
            batches.append(x[i : i + self.batch_size])

        if n % self.batch_size > 0:
            batches.append(x[i + self.batch_size : n])

        return batches

    def _rejoin_batches(self, x, n):
        m = len(x)
        d = x[0].shape[-1]

        x_rejoin = np.zeros(shape=(n, d))
        filled = 0

        for i in range(m):
            _stop = len(x[i])

            x_rejoin[filled : filled + _stop, :] = x[i]

            filled += len(x[i])

        return x_rejoin

    def _to_latent(self, x):
        x_batches = self._split_to_batches(x=x)
        x_latent = []

        for batch in x_batches:
            x_latent.append(self.feature_extractor(batch, training=False))

        return self._rejoin_batches(x=x_latent, n=len(x))

    @abstractmethod
    def calculate(
        self,
        xgenerated=None,
        ygenerated=None,
        xreal=None,
        yreal=None,
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
        """
        ...
