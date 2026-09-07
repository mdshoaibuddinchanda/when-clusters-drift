import numpy as np
import pytest
from clusterdrift.falsification.models import TrainingOnlyPreprocessor

def test_training_only_imputation():
    X_train = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
        [5.0, np.nan],  # median for col 1 should be 20.0 (from 10.0 and 30.0)
    ])
    prep = TrainingOnlyPreprocessor(with_scaling=False)
    prep.fit(X_train)

    assert prep.medians_[0] == 3.0
    assert prep.medians_[1] == 20.0

    X_test = np.array([
        [np.nan, 50.0],
        [2.0, np.nan],
    ])
    X_test_trans = prep.transform(X_test)

    # Missing in col 0 replaced with training median 3.0
    assert X_test_trans[0, 0] == 3.0
    assert X_test_trans[0, 1] == 50.0
    # Missing in col 1 replaced with training median 20.0
    assert X_test_trans[1, 0] == 2.0
    assert X_test_trans[1, 1] == 20.0


def test_fallback_zero_imputation():
    X_train = np.array([
        [np.nan],
        [np.nan],
    ])
    prep = TrainingOnlyPreprocessor(with_scaling=False)
    prep.fit(X_train)
    assert prep.medians_[0] == 0.0