"""Scaling augmentation method."""

import tensorflow as tf


@tf.function
def scale(X: tf.Tensor, y: tf.Tensor = None):
    """Scaling the input time series.

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
        The scaled series.

    References
    ----------
    [1] Um, Terry T., Franz MJ Pfister, Daniel Pichler, Satoshi Endo,
        Muriel Lang, Sandra Hirche, Urban Fietzek, and Dana Kulić.
        "Data augmentation of wearable sensor data for parkinson's disease
        monitoring using convolutional neural networks." In Proceedings of
        the 19th ACM international conference on multimodal interaction,
        pp. 216-220. 2017.
    """
    scale_factors = tf.random.normal(
        mean=1.0,
        stddev=0.1,
        shape=(int(X.shape[0]), 1, int(X.shape[2])),
        dtype="float64",
    )

    return X * scale_factors
