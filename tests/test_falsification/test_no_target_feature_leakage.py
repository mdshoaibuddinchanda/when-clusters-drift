import pytest
from clusterdrift.falsification.dataset import FORBIDDEN_PREDICTOR_FEATURES
from clusterdrift.falsification.protocol import (
    P4_STRUCTURAL_ABLATIONS,
    PREDEFINED_FEATURE_BLOCKS,
    SINGLE_SIGNAL_INCREMENTAL,
)


def test_forbidden_features_not_in_any_block():
    for block_name, features in PREDEFINED_FEATURE_BLOCKS.items():
        for feat in features:
            assert feat not in FORBIDDEN_PREDICTOR_FEATURES, f"Forbidden feature {feat} in block {block_name}"

    for abl_name, features in P4_STRUCTURAL_ABLATIONS.items():
        for feat in features:
            assert feat not in FORBIDDEN_PREDICTOR_FEATURES, f"Forbidden feature {feat} in ablation {abl_name}"

    for inc_name, features in SINGLE_SIGNAL_INCREMENTAL.items():
        for feat in features:
            assert feat not in FORBIDDEN_PREDICTOR_FEATURES, f"Forbidden feature {feat} in incremental {inc_name}"


def test_forbidden_features_contains_labels_and_metadata():
    for forbidden in ["y", "label", "labels", "delta_ari", "dataset_id", "condition", "seed"]:
        assert forbidden in FORBIDDEN_PREDICTOR_FEATURES