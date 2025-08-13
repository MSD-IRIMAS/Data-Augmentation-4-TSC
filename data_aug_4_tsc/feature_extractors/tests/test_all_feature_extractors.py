"""Test if feature extractors training does not crash."""

import inspect
import tempfile

import numpy as np
import pytest

from data_aug_4_tsc import feature_extractors

_feature_extractors = [
    member[1] for member in inspect.getmembers(feature_extractors, inspect.isclass)
]


@pytest.mark.parametrize("feature_extractor", _feature_extractors)
def test_all_feature_extractors(feature_extractor):
    """Test all feature extractors training."""
    with tempfile.TemporaryDirectory() as tmp:
        X = np.random.normal(size=(5, 7, 2))
        y = np.array([0, 0, 1, 1, 1])

        if feature_extractor.__name__ == "LITE_CLASSIFIER":
            _model = feature_extractor(output_directory=tmp, n_epochs=2)

        _model.fit(X, y)

        assert _model.feature_extractor is not None
