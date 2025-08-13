"""Window warping augmentation method."""

import numpy as np
from numba import njit, prange


@njit(nopython=True, fastmath=True, parallel=True, cache=True)
def window_warping(
    X: np.ndarray,
    y: np.ndarray = None,
    window_size_ratio: float = None,
    warp_scale: float = None,
    window_start: int = None,
):
    """Warp the input time series on the time axis.

    This augmentation method was proposed in [1]. The window size,
    warp scale and start of the window is randomly generated for each
    time series in the input batch, if and only if, the three parameters
    window_size_ration, warp_scale and window_start are set to None as
    by default. If these parameters are set to actual values then the same
    value is used all through the batch.

    Parameters
    ----------
    X: tf.Tensor, shape (batch_size, length_TS, n_channels)
        The input set of time series.
    y: np.ndarray, shape (batch_size,)
        The labels of each input time series. Ignored, here for
        code structure reasons.
    window_size_ratio: float, default = None
        The ratio size of the window to adjust by warping, if None,
        then it is randomly selected (between 5% and 30%) per
        series. If float, then the same value is used for all
        series in the batch.
    warp_scale: float, default = None
        The scale of the window warping, if None then it is randomly
        selected per series (between 0.5 and 2.0). If float, then
        the same value is used for all series in the batch.
    window_start: int, default = None
        The index of the start of the windows, if None then it is
        randomly selected per series. If in then the same value
        is used for all series in the batch.

    Returns
    -------
    np.ndarray, shape (batch_size, length_TS, n_channels)
        The warped series.

    References
    ----------
    [1] Le Guennec, Arthur, Simon Malinowski, and Romain Tavenard.
        "Data augmentation for time series classification using
        convolutional neural networks." In ECML/PKDD workshop on
        advanced analytics and learning on temporal data. 2016.
    """
    batch_size = len(X)
    length_TS = len(X[0])
    n_channels = len(X[0, 0])

    if window_size_ratio is None:
        window_size_ratios = np.random.choice(
            np.linspace(0.05, 0.3, num=6), size=batch_size
        )
    else:
        window_size_ratios = np.repeat(window_size_ratio, repeats=batch_size).reshape(
            (batch_size,)
        )
    window_sizes = np.ceil(window_size_ratios * length_TS).astype(np.int32)

    if warp_scale is None:
        warp_scales = np.random.choice(np.linspace(0.1, 2.0, num=20), size=batch_size)
    else:
        warp_scales = np.repeat(warp_scale, repeats=batch_size).reshape((batch_size,))

    warped_series = np.zeros_like(X)

    for i in prange(batch_size):
        _X = X[i]
        if window_start is None:
            _window_start = np.random.randint(
                low=1, high=length_TS - window_sizes[i], size=1
            )[0]
        window_end = _window_start + window_sizes[i]
        window_steps = np.arange(window_sizes[i])

        for dim in prange(n_channels):
            start_seg = _X[:_window_start, dim]
            window_seg = np.interp(
                np.linspace(
                    0, window_sizes[i] - 1, num=int(window_sizes[i] * warp_scales[i])
                ),
                window_steps,
                _X[_window_start:window_end, dim],
            )
            end_seg = _X[window_end:, dim]

            warped = np.concatenate((start_seg, window_seg, end_seg))

            warped_series[i, :, dim] = np.interp(
                np.arange(length_TS),
                np.linspace(0, length_TS - 1.0, num=len(warped)),
                warped,
            )

    return warped_series
