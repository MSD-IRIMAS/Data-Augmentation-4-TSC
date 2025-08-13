"""Amplitude warping augmentation method."""

import numpy as np
import tensorflow as tf


@tf.function
def amplitude_warping(X: tf.Tensor, y: tf.Tensor = None):
    """Adjust the amplitude of the input time series.

    This augmentation method was proposed in [1].

    Parameters
    ----------
    X: tf.Tensor, shape (batch_size, length_TS, n_channels)
        The input set of time series.
    y: np.ndarray, shape (batch_size,)
        The labels of each input time series. Ignored, here for
        code structure reasons.

    Returns
    -------
    np.ndarray, shape (batch_size, length_TS, n_channels)
        The adjusted series.

    References
    ----------
    [1] Um, Terry T., Franz MJ Pfister, Daniel Pichler, Satoshi Endo,
        Muriel Lang, Sandra Hirche, Urban Fietzek, and Dana Kulić.
        "Data augmentation of wearable sensor data for parkinson's disease
        monitoring using convolutional neural networks." In Proceedings of
        the 19th ACM international conference on multimodal interaction,
        pp. 216-220. 2017.
    """
    length_TS = X.shape[1]
    x_values = tf.tile(
        tf.reshape(
            tf.linspace(np.float64(-1.0), np.float64(1.0), num=length_TS),
            (1, length_TS, 1),
        ),
        [len(X), 1, 1],
    )
    random_cubic_scaling_factors = _get_random_cubic(x_values=x_values)

    return X * random_cubic_scaling_factors


@tf.function
def _get_random_cubic(x_values):
    a = 0.2

    a = tf.random.uniform(
        shape=(len(x_values), 1, 1), minval=-a, maxval=a, dtype="float64"
    )
    b = tf.random.uniform(
        shape=(len(x_values), 1, 1), minval=-a, maxval=a, dtype="float64"
    )
    c = tf.random.uniform(
        shape=(len(x_values), 1, 1), minval=-a, maxval=a, dtype="float64"
    )
    d = tf.random.uniform(
        shape=(len(x_values), 1, 1), minval=-a, maxval=a, dtype="float64"
    )

    random_cubic_function = a * x_values**3 + b * x_values**2 + c * x_values + d

    min_value = tf.reduce_min(random_cubic_function, axis=1, keepdims=True)
    random_cubic_function += tf.abs(min_value)

    max_value = tf.reduce_max(tf.abs(random_cubic_function), axis=1, keepdims=True)
    random_cubic_function /= max_value
    random_cubic_function += 0.5

    return random_cubic_function
