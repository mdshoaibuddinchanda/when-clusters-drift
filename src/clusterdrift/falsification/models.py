"""Models, Training-Only Transformers, and Inner GroupKFold Tuning for Phase 7."""

from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


class TrainingOnlyPreprocessor:
    """Strictly training-only median imputer and optional standard scaler."""

    def __init__(self, with_scaling: bool = False):
        self.with_scaling = with_scaling
        self.medians_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "TrainingOnlyPreprocessor":
        arr = np.asarray(X, dtype=np.float64)
        n_features = arr.shape[1]
        medians = np.zeros(n_features, dtype=np.float64)

        for j in range(n_features):
            col = arr[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                medians[j] = float(np.median(valid))
            else:
                medians[j] = 0.0

        self.medians_ = medians

        if self.with_scaling:
            # Impute training copy to compute mean and std
            imputed = arr.copy()
            for j in range(n_features):
                mask = np.isnan(imputed[:, j])
                imputed[mask, j] = self.medians_[j]
            means = np.mean(imputed, axis=0)
            stds = np.std(imputed, axis=0)
            stds[stds < 1e-12] = 1.0
            self.means_ = means
            self.scales_ = stds

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor must be fitted before transform")
        arr = np.asarray(X, dtype=np.float64).copy()
        n_features = arr.shape[1]

        for j in range(n_features):
            mask = np.isnan(arr[:, j])
            arr[mask, j] = self.medians_[j]

        if self.with_scaling:
            if self.means_ is None or self.scales_ is None:
                raise RuntimeError("Scaler was not fitted properly")
            arr = (arr - self.means_) / self.scales_

        return arr

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


import hashlib
import json


@dataclass(frozen=True)
class TuningSelection:
    selected_params: Dict[str, Any]
    best_score: float
    candidates: List[Dict[str, Any]]
    tie_broken: bool
    inner_splits_sha256: str = ""


def tune_and_fit_hgbr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    grid: Dict[str, List[Any]],
    random_state: int = 2026090707,
    n_splits: int = 5,
) -> Tuple[HistGradientBoostingRegressor, TrainingOnlyPreprocessor, TuningSelection]:
    """Inner GroupKFold hyperparameter selection for HistGradientBoostingRegressor."""
    # Deterministic grid combinations
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Outer preprocessor
    outer_prep = TrainingOnlyPreprocessor(with_scaling=False)
    outer_prep.fit(X_train)

    # Setup GroupKFold inner CV
    unique_groups = len(np.unique(groups_train))
    actual_splits = min(n_splits, unique_groups)
    assert actual_splits == n_splits, f"Expected exactly {n_splits} inner folds, got {actual_splits}"
    gkf = GroupKFold(n_splits=actual_splits)

    # Pre-verify splits and compute inner split hash
    inner_splits_repr = []
    for f_idx, (in_tr, in_val) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        tr_g = sorted(list(set(str(g) for g in groups_train[in_tr])))
        val_g = sorted(list(set(str(g) for g in groups_train[in_val])))
        assert set(tr_g).isdisjoint(set(val_g)), f"Inner fold {f_idx} train/val groups overlap: {set(tr_g) & set(val_g)}"
        inner_splits_repr.append(f"fold_{f_idx}:tr={tr_g}:val={val_g}")
    inner_splits_sha256 = hashlib.sha256("|".join(inner_splits_repr).encode("utf-8")).hexdigest()

    candidate_results: List[Dict[str, Any]] = []

    for params in param_combos:
        fold_maes = []
        fold_records = []
        for fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_in_tr = X_train[inner_tr_idx]
            y_in_tr = y_train[inner_tr_idx]
            X_in_val = X_train[inner_val_idx]
            y_in_val = y_train[inner_val_idx]

            in_prep = TrainingOnlyPreprocessor(with_scaling=False)
            X_in_tr_trans = in_prep.fit_transform(X_in_tr)
            X_in_val_trans = in_prep.transform(X_in_val)

            model = HistGradientBoostingRegressor(
                loss="squared_error",
                random_state=random_state,
                early_stopping=False,
                min_samples_leaf=20,
                **params,
            )
            model.fit(X_in_tr_trans, y_in_tr)
            preds = model.predict(X_in_val_trans)
            f_mae = float(mean_absolute_error(y_in_val, preds))
            fold_maes.append(f_mae)

            tr_g_sha = hashlib.sha256(json.dumps(sorted(list(set(str(g) for g in groups_train[inner_tr_idx])))).encode("utf-8")).hexdigest()
            val_g_sha = hashlib.sha256(json.dumps(sorted(list(set(str(g) for g in groups_train[inner_val_idx])))).encode("utf-8")).hexdigest()
            fold_records.append({
                "inner_fold": fold_idx,
                "inner_training_group_sha256": tr_g_sha,
                "inner_validation_group_sha256": val_g_sha,
                "inner_fold_mae": f_mae,
            })

        mean_mae = float(np.mean(fold_maes))
        candidate_results.append({
            "params": params,
            "mean_mae": mean_mae,
            "fold_maes": fold_maes,
            "fold_records": fold_records,
            "selected": False,
            "tie_status": False,
        })

    # Find best score
    best_score = min(c["mean_mae"] for c in candidate_results)

    # Deterministic tie-breaking within 1e-7:
    # Prefer lower max_leaf_nodes, lower max_iter, higher l2_regularization, lower learning_rate
    tied_candidates = [
        c for c in candidate_results
        if abs(c["mean_mae"] - best_score) <= 1e-7
    ]

    for c in tied_candidates:
        c["tie_status"] = len(tied_candidates) > 1

    def _hgbr_tie_key(c: Dict[str, Any]) -> Tuple[int, int, float, float]:
        p = c["params"]
        return (
            p.get("max_leaf_nodes", 31),
            p.get("max_iter", 100),
            -p.get("l2_regularization", 0.0),
            p.get("learning_rate", 0.1),
        )

    tied_candidates.sort(key=_hgbr_tie_key)
    selected = tied_candidates[0]
    selected["selected"] = True
    best_params = selected["params"]

    # Final fit on full outer training set
    X_train_trans = outer_prep.transform(X_train)
    final_model = HistGradientBoostingRegressor(
        loss="squared_error",
        random_state=random_state,
        early_stopping=False,
        min_samples_leaf=20,
        **best_params,
    )
    final_model.fit(X_train_trans, y_train)

    selection = TuningSelection(
        selected_params=best_params,
        best_score=best_score,
        candidates=candidate_results,
        tie_broken=len(tied_candidates) > 1,
        inner_splits_sha256=inner_splits_sha256,
    )
    return final_model, outer_prep, selection


def tune_and_fit_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alphas: List[float],
    random_state: int = 2026090707,
    n_splits: int = 5,
) -> Tuple[Ridge, TrainingOnlyPreprocessor, TuningSelection]:
    """Inner GroupKFold hyperparameter selection for Ridge linear control."""
    outer_prep = TrainingOnlyPreprocessor(with_scaling=True)
    outer_prep.fit(X_train)

    unique_groups = len(np.unique(groups_train))
    actual_splits = min(n_splits, unique_groups)
    assert actual_splits == n_splits, f"Expected exactly {n_splits} inner folds, got {actual_splits}"
    gkf = GroupKFold(n_splits=actual_splits)

    # Pre-verify splits and compute inner split hash
    inner_splits_repr = []
    for f_idx, (in_tr, in_val) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
        tr_g = sorted(list(set(str(g) for g in groups_train[in_tr])))
        val_g = sorted(list(set(str(g) for g in groups_train[in_val])))
        assert set(tr_g).isdisjoint(set(val_g)), f"Inner fold {f_idx} train/val groups overlap: {set(tr_g) & set(val_g)}"
        inner_splits_repr.append(f"fold_{f_idx}:tr={tr_g}:val={val_g}")
    inner_splits_sha256 = hashlib.sha256("|".join(inner_splits_repr).encode("utf-8")).hexdigest()

    candidate_results: List[Dict[str, Any]] = []

    for alpha in sorted(alphas):
        fold_maes = []
        fold_records = []
        for fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            X_in_tr = X_train[inner_tr_idx]
            y_in_tr = y_train[inner_tr_idx]
            X_in_val = X_train[inner_val_idx]
            y_in_val = y_train[inner_val_idx]

            in_prep = TrainingOnlyPreprocessor(with_scaling=True)
            X_in_tr_trans = in_prep.fit_transform(X_in_tr)
            X_in_val_trans = in_prep.transform(X_in_val)

            model = Ridge(alpha=alpha, random_state=random_state)
            model.fit(X_in_tr_trans, y_in_tr)
            preds = model.predict(X_in_val_trans)
            f_mae = float(mean_absolute_error(y_in_val, preds))
            fold_maes.append(f_mae)

            tr_g_sha = hashlib.sha256(json.dumps(sorted(list(set(str(g) for g in groups_train[inner_tr_idx])))).encode("utf-8")).hexdigest()
            val_g_sha = hashlib.sha256(json.dumps(sorted(list(set(str(g) for g in groups_train[inner_val_idx])))).encode("utf-8")).hexdigest()
            fold_records.append({
                "inner_fold": fold_idx,
                "inner_training_group_sha256": tr_g_sha,
                "inner_validation_group_sha256": val_g_sha,
                "inner_fold_mae": f_mae,
            })

        mean_mae = float(np.mean(fold_maes))
        candidate_results.append({
            "params": {"alpha": alpha},
            "mean_mae": mean_mae,
            "fold_maes": fold_maes,
            "fold_records": fold_records,
            "selected": False,
            "tie_status": False,
        })

    best_score = min(c["mean_mae"] for c in candidate_results)

    # Tie-breaking: prefer higher alpha (stronger regularization)
    tied_candidates = [
        c for c in candidate_results
        if abs(c["mean_mae"] - best_score) <= 1e-7
    ]
    for c in tied_candidates:
        c["tie_status"] = len(tied_candidates) > 1

    tied_candidates.sort(key=lambda c: -c["params"]["alpha"])
    selected = tied_candidates[0]
    selected["selected"] = True
    best_params = selected["params"]

    X_train_trans = outer_prep.transform(X_train)
    final_model = Ridge(alpha=best_params["alpha"], random_state=random_state)
    final_model.fit(X_train_trans, y_train)

    selection = TuningSelection(
        selected_params=best_params,
        best_score=best_score,
        candidates=candidate_results,
        tie_broken=len(tied_candidates) > 1,
        inner_splits_sha256=inner_splits_sha256,
    )
    return final_model, outer_prep, selection
