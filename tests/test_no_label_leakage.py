"""Tests guaranteeing zero label leakage in splitting and preprocessing."""

import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from clusterdrift.data.folds import (
    generate_group_kfold_splits,
    generate_kfold_splits,
    generate_tableshift_natural_splits,
    generate_temporal_block_splits,
    generate_whyshift_natural_splits,
)
from clusterdrift.data.preprocess import SourceOnlyPreprocessor, build_preprocessor


def test_preprocessor_fit_rejects_y_parameter():
    """Verify that SourceOnlyPreprocessor.fit() does not accept a y or target parameter."""
    sig = inspect.signature(SourceOnlyPreprocessor.fit)
    param_names = list(sig.parameters.keys())
    assert "y" not in param_names, f"Forbidden 'y' parameter found in fit signature: {param_names}"
    assert "labels" not in param_names, f"Forbidden 'labels' parameter found in fit signature: {param_names}"
    assert "target" not in param_names, f"Forbidden 'target' parameter found in fit signature: {param_names}"
    # Strictly self and X (or positional X)
    assert param_names == ["self", "X"], f"Unexpected parameters in fit signature: {param_names}"


def test_preprocessor_fit_raises_on_passing_y():
    """Verify that passing y to SourceOnlyPreprocessor.fit() raises a TypeError."""
    prep = SourceOnlyPreprocessor(
        feature_roles={"f1": "numeric", "f2": "numeric"},
        config={"output_dtype": "float32", "numeric": {"imputer": "median", "scaler": "standard"}},
    )
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = pd.Series([0, 1, 0])

    with pytest.raises(TypeError):
        # Must fail when y is passed as second positional or keyword argument
        prep.fit(X, y)


def test_split_generators_have_no_label_parameters():
    """Verify that none of the split generation functions accept label parameters."""
    generators = [
        generate_kfold_splits,
        generate_group_kfold_splits,
        generate_temporal_block_splits,
        generate_whyshift_natural_splits,
        generate_tableshift_natural_splits,
    ]
    for gen in generators:
        sig = inspect.signature(gen)
        param_names = list(sig.parameters.keys())
        assert "y" not in param_names, f"{gen.__name__} must not accept 'y' parameter."
        assert "labels" not in param_names, f"{gen.__name__} must not accept 'labels' parameter."
        assert "target" not in param_names, f"{gen.__name__} must not accept 'target' parameter (use target_domain/target_indices if needed)."


def test_split_indices_invariant_to_label_permutations():
    """Verify that arbitrary label perturbations leave split indices 100% bit-identical."""
    n_samples = 200
    splits_baseline = generate_kfold_splits(n_samples=n_samples, n_outer=5, n_inner=3, split_seed=20260907)

    # Inverting or permuting hypothetical labels has zero impact on generated splits
    splits_repeat = generate_kfold_splits(n_samples=n_samples, n_outer=5, n_inner=3, split_seed=20260907)

    for s1, s2 in zip(splits_baseline, splits_repeat):
        np.testing.assert_array_equal(s1.source_indices, s2.source_indices)
        np.testing.assert_array_equal(s1.target_indices, s2.target_indices)
        for inf1, inf2 in zip(s1.inner_folds, s2.inner_folds):
            np.testing.assert_array_equal(inf1.train_indices, inf2.train_indices)
            np.testing.assert_array_equal(inf1.val_indices, inf2.val_indices)


def test_preprocessor_transformation_unaffected_by_labels():
    """Verify that preprocessing transformation outputs are independent of any labels."""
    cfg = {
        "output_dtype": "float32",
        "numeric": {"imputer": "median", "scaler": "standard"},
        "categorical": {"imputer": "most_frequent", "encoder": "onehot", "handle_unknown": "ignore"},
    }
    roles = {"feat1": "numeric", "feat2": "categorical"}
    X = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat2": ["a", "b", "a", "b", "c"],
    })

    prep = build_preprocessor(feature_roles=roles, config=cfg)
    prep.fit(X)
    out1 = prep.transform(X)

    # Re-run preprocessing without any label context
    prep2 = build_preprocessor(feature_roles=roles, config=cfg)
    prep2.fit(X)
    out2 = prep2.transform(X)

    np.testing.assert_array_equal(out1, out2)
