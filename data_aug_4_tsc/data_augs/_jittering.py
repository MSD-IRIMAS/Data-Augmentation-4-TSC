"""Jittering augmentation method."""

import tensorflow as tf


@tf.function
def add_noise(X: tf.Tensor, y: tf.Tensor = None):
    """Adding noise to input time series.

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
        The noised series.

    References
    ----------
    [1] Um, Terry T., Franz MJ Pfister, Daniel Pichler, Satoshi Endo,
        Muriel Lang, Sandra Hirche, Urban Fietzek, and Dana Kulić.
        "Data augmentation of wearable sensor data for parkinson's disease
        monitoring using convolutional neural networks." In Proceedings of
        the 19th ACM international conference on multimodal interaction,
        pp. 216-220. 2017.
    """
    sigma = tf.random.uniform(shape=(), minval=0, maxval=0.1, dtype="float64")
    noise = tf.random.normal(shape=X.shape, mean=0.0, stddev=sigma, dtype="float64")
    return X + noise
