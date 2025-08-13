"""The LITE classification model."""

import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import IndividualLITEClassifier


class LITE_CLASSIFIER:
    """The Light Inception with boosTing tEchniques (LITE).

    The LITE [1]_ classificaiton model uses a CNN backbone with
    an optimized number of parameters to classify time series
    samples. We use aeon-toolkit [2]_ 's implementation.

    Parameters
    ----------
    output_directory : str
        The output directory.
    n_epochs : int, default = 1500
        The number of epochs to train the model.
    batch_size : int, default = 64
        The batch size during training.

    Returns
    -------
    None

    References
    ----------
    .. [1] Ismail-Fawaz, Ali, Maxime Devanne, Stefano Berretti,
    Jonathan Weber, and Germain Forestier. "Lite: Light inception
    with boosting techniques for time series classification." In
    2023 IEEE 10th International Conference on Data Science and
    Advanced Analytics (DSAA), pp. 1-10. IEEE, 2023.

    .. [2] Middlehurst, Matthew, Ali Ismail-Fawaz, Antoine Guillaume,
    Christopher Holder, David Guijo-Rubio, Guzal Bulatova, Leonidas
    Tsaprounis et al. "aeon: a Python toolkit for learning from
    time series." Journal of Machine Learning Research 25, no.
    289 (2024): 1-10.
    """

    def __init__(
        self, output_directory: str, n_epochs: int = 1500, batch_size: int = 64
    ):

        self.output_directory = output_directory
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.feature_extractor = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on labeled real samples.

        Parameters
        ----------
        X : np.ndarray
            The real samples.
        y : np.ndarray
            The labels of the real samples.

        Returns
        -------
        A trained keras model.

        """
        self.feature_extractor = IndividualLITEClassifier(
            file_path=self.output_directory,
            save_best_model=True,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            verbose=True,
        )
        self.feature_extractor.fit(X=np.swapaxes(X, axis1=1, axis2=2), y=y)

        tf.keras.backend.clear_session()

        return self.feature_extractor
