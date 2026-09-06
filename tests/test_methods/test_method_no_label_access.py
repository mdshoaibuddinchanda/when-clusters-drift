"""Tests ensuring that clustering methods are strictly unsupervised and have zero label access."""

import inspect
from pathlib import Path
import numpy as np
import pytest

from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.gmm import GMM
from clusterdrift.methods.gustafson_kessel import GustafsonKessel
from clusterdrift.methods.kmeans import KMeans
from clusterdrift.methods.pfcm import PFCM

METHODS = [KMeans, FCM, GMM, PFCM, GustafsonKessel]


@pytest.mark.parametrize("model_cls", METHODS)
def test_fit_signature_rejects_y(model_cls):
    """Verify that fit() signature only accepts X and does NOT have a 'y' parameter."""
    sig = inspect.signature(model_cls.fit)
    param_names = list(sig.parameters.keys())
    assert "self" in param_names
    assert "X" in param_names
    assert "y" not in param_names, f"{model_cls.__name__}.fit has forbidden 'y' parameter!"


@pytest.mark.parametrize("model_cls", METHODS)
def test_fit_raises_type_error_if_y_passed(model_cls):
    """Verify that calling fit(X, y) raises TypeError at runtime."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([0, 1, 0])
    model = model_cls(n_clusters=2)

    with pytest.raises(TypeError):
        model.fit(X, y)


def test_methods_directory_has_zero_label_references():
    """Verify that no source file under src/clusterdrift/methods/ accesses label artifacts."""
    methods_dir = Path(__file__).resolve().parent.parent.parent / "src" / "clusterdrift" / "methods"
    forbidden_tokens = [
        "labels.parquet",
        "hard_labels.parquet",
        "y_target",
        "target labels",
        "target_labels",
    ]

    for py_file in methods_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8").lower()
        for token in forbidden_tokens:
            assert token not in content, (
                f"Forbidden label token '{token}' detected in {py_file.name}!"
            )
