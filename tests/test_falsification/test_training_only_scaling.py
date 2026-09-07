import numpy as np
import pytest
from clusterdrift.falsification.models import TrainingOnlyPreprocessor

def test_training_only_scaling():
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    prep = TrainingOnlyPreprocessor(with_scaling=True)
    prep.fit(X_train)

    expected_means = np.array([20.0, 200.0])
    expected_stds = np.std(X_train, axis=0)

    np.testing.assert_allclose(prep.means_, expected_means)
    np.testing.assert_allclose(prep.scales_, expected_stds)

    X_test = np.array([
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    X_test_trans = prep.transform(X_test)
    assert X_test_trans[0, 0] == 0.0
    assert X_test_trans[0, 1] == 0.0
    np.testing.assert_allclose(X_test_trans[1, 0], (30.0 - 20.0) / expected_stds[0])