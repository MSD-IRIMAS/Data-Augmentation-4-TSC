import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import IndividualLITEClassifier


class LITE_CLASSIFIER:

    def __init__(
        self, output_directory: str, n_epochs: int = 1500, batch_size: int = 64
    ):

        self.output_directory = output_directory
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.feature_extractor = None

    def fit(self, X: np.ndarray, y: np.ndarray):

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
