"""External evaluation metrics comparing cluster assignments against ground truth labels."""

from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def _ensure_1d_labels(y: Union[np.ndarray, pd.Series, list]) -> np.ndarray:
    """Ensure label array is a flat 1D numpy array."""
    if isinstance(y, (pd.Series, pd.DataFrame)):
        arr = y.to_numpy().ravel()
    else:
        arr = np.asarray(y).ravel()
    return arr


def adjusted_rand_index(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Adjusted Rand Index (ARI) between ground truth labels and predicted clusters.

    ARI is symmetric and invariant to label permutations.
    """
    yt = _ensure_1d_labels(y_true)
    yp = _ensure_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Label lengths do not match: {len(yt)} vs {len(yp)}")
    return float(adjusted_rand_score(yt, yp))


def normalized_mutual_info(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Normalized Mutual Information (NMI)."""
    yt = _ensure_1d_labels(y_true)
    yp = _ensure_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Label lengths do not match: {len(yt)} vs {len(yp)}")
    return float(normalized_mutual_info_score(yt, yp, average_method="arithmetic"))


def adjusted_mutual_info(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
) -> float:
    """Compute Adjusted Mutual Information (AMI)."""
    yt = _ensure_1d_labels(y_true)
    yp = _ensure_1d_labels(y_pred)
    if len(yt) != len(yp):
        raise ValueError(f"Label lengths do not match: {len(yt)} vs {len(yp)}")
    return float(adjusted_mutual_info_score(yt, yp, average_method="arithmetic"))
