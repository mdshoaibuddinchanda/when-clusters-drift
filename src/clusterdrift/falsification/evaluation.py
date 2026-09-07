"""Evaluation Routines: LODO, LOSFO, Linear Controls, Ablations, and Diagnostics."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from clusterdrift.falsification.models import (
    tune_and_fit_hgbr,
    tune_and_fit_ridge,
    TuningSelection,
)
from clusterdrift.falsification.protocol import (
    ALL_SHIFT_FAMILIES,
    P4_STRUCTURAL_ABLATIONS,
    PREDEFINED_FEATURE_BLOCKS,
    SINGLE_SIGNAL_INCREMENTAL,
    compute_falsification_protocol_sha256,
)


def compute_prediction_record_sha256(rec: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of a single prediction record."""
    keys = [
        "dataset_id",
        "outer_fold",
        "condition",
        "shift_family",
        "severity",
        "seed",
        "feature_block",
        "regressor",
        "observed_delta_ari",
        "predicted_delta_ari",
        "absolute_error",
        "squared_error",
        "outer_test_group",
        "inner_protocol_sha",
    ]
    vals = [str(rec.get(k, "")) for k in keys]
    canonical_repr = "|".join(vals)
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute primary and secondary evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    med_ae = float(np.median(np.abs(y_true - y_pred)))
    # For r2_score, if y_true variance is 0, handle edge case
    var = float(np.var(y_true))
    r2 = float(r2_score(y_true, y_pred)) if var > 1e-12 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": med_ae,
        "r2": r2,
    }


def run_lodo_evaluation(
    joined_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute Leave-One-Dataset-Out (LODO) nested evaluation across 12 datasets."""
    protocol_sha = compute_falsification_protocol_sha256(cfg)
    datasets = cfg["datasets"]
    feature_blocks = cfg["feature_blocks"]
    hgbr_grid = cfg["primary_regressor"]["grid"]
    ridge_alphas = cfg["linear_control"]["alphas"]
    global_seed = cfg["global_seed"]

    # Primary analysis excludes clean
    primary_df = joined_df[joined_df["condition"] != "clean"].copy()

    predictions_records: List[Dict[str, Any]] = []
    hyperparams_records: List[Dict[str, Any]] = []

    # Models and feature blocks to evaluate
    eval_blocks: Dict[str, List[str]] = dict(feature_blocks)
    # Add ablations
    for k, v in P4_STRUCTURAL_ABLATIONS.items():
        eval_blocks[k] = v
    # Add single signals
    for k, v in SINGLE_SIGNAL_INCREMENTAL.items():
        eval_blocks[k] = v

    for test_ds in datasets:
        train_mask = (primary_df["dataset_id"] != test_ds)
        test_mask = (primary_df["dataset_id"] == test_ds)

        train_data = primary_df[train_mask]
        test_data = primary_df[test_mask]

        groups_train = train_data["dataset_id"].values
        y_train = train_data["delta_ari"].values.astype(np.float64)
        y_test = test_data["delta_ari"].values.astype(np.float64)

        for block_name, features in eval_blocks.items():
            X_train = train_data[features].values.astype(np.float64)
            X_test = test_data[features].values.astype(np.float64)

            # Fit HGBR
            model, prep, selection = tune_and_fit_hgbr(
                X_train, y_train, groups_train, hgbr_grid, random_state=global_seed
            )
            X_test_trans = prep.transform(X_test)
            preds = model.predict(X_test_trans)

            # Record hyperparameter selection
            hyperparams_records.append({
                "outer_eval_type": "LODO",
                "test_group": test_ds,
                "feature_block": block_name,
                "regressor": "hist_gradient_boosting",
                "training_groups_count": len(np.unique(groups_train)),
                "best_inner_mae": selection.best_score,
                "selected_params": json.dumps(selection.selected_params),
                "tie_broken": selection.tie_broken,
            })

            # Record predictions
            for idx, (_, row) in enumerate(test_data.iterrows()):
                obs = float(row["delta_ari"])
                pred = float(preds[idx])
                abs_err = abs(obs - pred)
                sq_err = (obs - pred) ** 2
                rec = {
                    "dataset_id": row["dataset_id"],
                    "outer_fold": int(row["outer_fold"]),
                    "condition": row["condition"],
                    "shift_family": row["shift_family"],
                    "severity": row["severity"],
                    "seed": int(row["seed"]),
                    "feature_block": block_name,
                    "regressor": "hist_gradient_boosting",
                    "observed_delta_ari": obs,
                    "predicted_delta_ari": pred,
                    "absolute_error": abs_err,
                    "squared_error": sq_err,
                    "outer_test_group": test_ds,
                    "inner_protocol_sha": protocol_sha,
                }
                rec["prediction_record_sha256"] = compute_prediction_record_sha256(rec)
                predictions_records.append(rec)

            # For standard blocks P0-P5, also fit Ridge linear control
            if block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
                r_model, r_prep, r_selection = tune_and_fit_ridge(
                    X_train, y_train, groups_train, ridge_alphas, random_state=global_seed
                )
                r_test_trans = r_prep.transform(X_test)
                r_preds = r_model.predict(r_test_trans)

                hyperparams_records.append({
                    "outer_eval_type": "LODO",
                    "test_group": test_ds,
                    "feature_block": block_name,
                    "regressor": "ridge",
                    "training_groups_count": len(np.unique(groups_train)),
                    "best_inner_mae": r_selection.best_score,
                    "selected_params": json.dumps(r_selection.selected_params),
                    "tie_broken": r_selection.tie_broken,
                })

                for idx, (_, row) in enumerate(test_data.iterrows()):
                    obs = float(row["delta_ari"])
                    pred = float(r_preds[idx])
                    abs_err = abs(obs - pred)
                    sq_err = (obs - pred) ** 2
                    rec = {
                        "dataset_id": row["dataset_id"],
                        "outer_fold": int(row["outer_fold"]),
                        "condition": row["condition"],
                        "shift_family": row["shift_family"],
                        "severity": row["severity"],
                        "seed": int(row["seed"]),
                        "feature_block": block_name,
                        "regressor": "ridge",
                        "observed_delta_ari": obs,
                        "predicted_delta_ari": pred,
                        "absolute_error": abs_err,
                        "squared_error": sq_err,
                        "outer_test_group": test_ds,
                        "inner_protocol_sha": protocol_sha,
                    }
                    rec["prediction_record_sha256"] = compute_prediction_record_sha256(rec)
                    predictions_records.append(rec)

    pred_df = pd.DataFrame(predictions_records)
    hyper_df = pd.DataFrame(hyperparams_records)

    # Compute overall LODO metrics for HGBR P0-P5
    hgbr_lodo = pred_df[pred_df["regressor"] == "hist_gradient_boosting"]
    lodo_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub = hgbr_lodo[hgbr_lodo["feature_block"] == blk]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        m["feature_block"] = blk
        m["regressor"] = "hist_gradient_boosting"
        m["n_predictions"] = len(sub)
        lodo_metrics.append(m)
    lodo_metrics_df = pd.DataFrame(lodo_metrics)

    # Compute per-dataset metrics for HGBR
    lodo_ds_metrics = []
    for ds in datasets:
        row_dict = {"dataset_id": ds}
        for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            sub = hgbr_lodo[(hgbr_lodo["dataset_id"] == ds) & (hgbr_lodo["feature_block"] == blk)]
            mae = float(mean_absolute_error(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values))
            row_dict[f"mae_{blk}"] = mae
        lodo_ds_metrics.append(row_dict)
    lodo_ds_df = pd.DataFrame(lodo_ds_metrics)

    # Paired dataset deltas: Delta_04^(d) = MAE_P0^(d) - MAE_P4^(d), Delta_34^(d) = MAE_P3^(d) - MAE_P4^(d)
    paired_ds_deltas = []
    for _, r in lodo_ds_df.iterrows():
        d_04 = r["mae_P0"] - r["mae_P4"]
        d_34 = r["mae_P3"] - r["mae_P4"]
        paired_ds_deltas.append({
            "dataset_id": r["dataset_id"],
            "mae_P0": r["mae_P0"],
            "mae_P3": r["mae_P3"],
            "mae_P4": r["mae_P4"],
            "delta_04": d_04,
            "delta_34": d_34,
        })
    paired_ds_df = pd.DataFrame(paired_ds_deltas)

    # Ridge control metrics
    ridge_lodo = pred_df[pred_df["regressor"] == "ridge"]
    ridge_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub_r = ridge_lodo[ridge_lodo["feature_block"] == blk]
        sub_h = hgbr_lodo[hgbr_lodo["feature_block"] == blk]
        m_r = compute_metrics(sub_r["observed_delta_ari"].values, sub_r["predicted_delta_ari"].values)
        m_h = compute_metrics(sub_h["observed_delta_ari"].values, sub_h["predicted_delta_ari"].values)
        ridge_metrics.append({
            "feature_block": blk,
            "hgbr_mae": m_h["mae"],
            "ridge_mae": m_r["mae"],
            "mae_diff_ridge_minus_hgbr": m_r["mae"] - m_h["mae"],
            "ridge_rmse": m_r["rmse"],
            "ridge_r2": m_r["r2"],
        })
    ridge_df = pd.DataFrame(ridge_metrics)

    # Structural ablations
    p4_mae = float(lodo_metrics_df[lodo_metrics_df["feature_block"] == "P4"]["mae"].values[0])
    abl_metrics = []
    for k in P4_STRUCTURAL_ABLATIONS.keys():
        sub = hgbr_lodo[hgbr_lodo["feature_block"] == k]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        abl_metrics.append({
            "variant": k,
            "mae": m["mae"],
            "mae_p4": p4_mae,
            "delta_mae_ablation_minus_p4": m["mae"] - p4_mae,
            "rmse": m["rmse"],
        })
    abl_df = pd.DataFrame(abl_metrics)

    # Single-signal incrementals
    p0_mae = float(lodo_metrics_df[lodo_metrics_df["feature_block"] == "P0"]["mae"].values[0])
    inc_metrics = []
    for k in SINGLE_SIGNAL_INCREMENTAL.keys():
        sub = hgbr_lodo[hgbr_lodo["feature_block"] == k]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        inc_metrics.append({
            "variant": k,
            "mae": m["mae"],
            "mae_p0": p0_mae,
            "delta_mae_p0_minus_incremental": p0_mae - m["mae"],
            "rmse": m["rmse"],
        })
    inc_df = pd.DataFrame(inc_metrics)

    # Dimension strata audit
    strata_metrics = []
    for stratum in ["low_D", "mid_D", "high_D"]:
        stratum_sub_p0 = hgbr_lodo[
            (hgbr_lodo["feature_block"] == "P0") &
            (hgbr_lodo["dataset_id"].isin(primary_df[primary_df["dimension_stratum"] == stratum]["dataset_id"].unique()))
        ]
        stratum_sub_p4 = hgbr_lodo[
            (hgbr_lodo["feature_block"] == "P4") &
            (hgbr_lodo["dataset_id"].isin(primary_df[primary_df["dimension_stratum"] == stratum]["dataset_id"].unique()))
        ]
        if len(stratum_sub_p0) > 0 and len(stratum_sub_p4) > 0:
            m_p0 = compute_metrics(stratum_sub_p0["observed_delta_ari"].values, stratum_sub_p0["predicted_delta_ari"].values)
            m_p4 = compute_metrics(stratum_sub_p4["observed_delta_ari"].values, stratum_sub_p4["predicted_delta_ari"].values)
            
            # Signal distribution in this stratum
            orig_stratum = primary_df[primary_df["dimension_stratum"] == stratum]
            strata_metrics.append({
                "dimension_stratum": stratum,
                "n_datasets": len(orig_stratum["dataset_id"].unique()),
                "p0_mae": m_p0["mae"],
                "p4_mae": m_p4["mae"],
                "delta_mae_p0_minus_p4": m_p0["mae"] - m_p4["mae"],
                "D_U_R_mean": float(orig_stratum["D_U_R"].mean()),
                "D_U_C_mean": float(orig_stratum["D_U_C"].mean()),
                "D_V_mean": float(orig_stratum["D_V"].mean()),
                "D_H_mean": float(orig_stratum["D_H"].mean()),
                "D_M_mean": float(orig_stratum["D_M"].mean()),
            })
    strata_df = pd.DataFrame(strata_metrics)

    return {
        "predictions_df": pred_df,
        "hyperparameters_df": hyper_df,
        "lodo_metrics_df": lodo_metrics_df,
        "lodo_dataset_metrics_df": lodo_ds_df,
        "paired_dataset_deltas_df": paired_ds_df,
        "ridge_control_metrics_df": ridge_df,
        "structural_ablation_df": abl_df,
        "single_signal_incremental_df": inc_df,
        "dimension_strata_audit_df": strata_df,
    }


def run_losfo_evaluation(
    joined_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute Leave-One-Shift-Family-Out (LOSFO) nested evaluation across 7 families."""
    protocol_sha = compute_falsification_protocol_sha256(cfg)
    feature_blocks = cfg["feature_blocks"]
    hgbr_grid = cfg["primary_regressor"]["grid"]
    global_seed = cfg["global_seed"]

    # Primary analysis excludes clean
    primary_df = joined_df[joined_df["condition"] != "clean"].copy()

    predictions_records: List[Dict[str, Any]] = []
    hyperparams_records: List[Dict[str, Any]] = []

    for test_fam in ALL_SHIFT_FAMILIES:
        train_mask = (primary_df["shift_family"] != test_fam)
        test_mask = (primary_df["shift_family"] == test_fam)

        train_data = primary_df[train_mask]
        test_data = primary_df[test_mask]

        groups_train = train_data["dataset_id"].values
        y_train = train_data["delta_ari"].values.astype(np.float64)
        y_test = test_data["delta_ari"].values.astype(np.float64)

        for block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            features = feature_blocks[block_name]
            X_train = train_data[features].values.astype(np.float64)
            X_test = test_data[features].values.astype(np.float64)

            model, prep, selection = tune_and_fit_hgbr(
                X_train, y_train, groups_train, hgbr_grid, random_state=global_seed
            )
            X_test_trans = prep.transform(X_test)
            preds = model.predict(X_test_trans)

            hyperparams_records.append({
                "outer_eval_type": "LOSFO",
                "test_group": test_fam,
                "feature_block": block_name,
                "regressor": "hist_gradient_boosting",
                "training_groups_count": len(np.unique(groups_train)),
                "best_inner_mae": selection.best_score,
                "selected_params": json.dumps(selection.selected_params),
                "tie_broken": selection.tie_broken,
            })

            for idx, (_, row) in enumerate(test_data.iterrows()):
                obs = float(row["delta_ari"])
                pred = float(preds[idx])
                abs_err = abs(obs - pred)
                sq_err = (obs - pred) ** 2
                rec = {
                    "dataset_id": row["dataset_id"],
                    "outer_fold": int(row["outer_fold"]),
                    "condition": row["condition"],
                    "shift_family": row["shift_family"],
                    "severity": row["severity"],
                    "seed": int(row["seed"]),
                    "feature_block": block_name,
                    "regressor": "hist_gradient_boosting",
                    "observed_delta_ari": obs,
                    "predicted_delta_ari": pred,
                    "absolute_error": abs_err,
                    "squared_error": sq_err,
                    "outer_test_group": test_fam,
                    "inner_protocol_sha": protocol_sha,
                }
                rec["prediction_record_sha256"] = compute_prediction_record_sha256(rec)
                predictions_records.append(rec)

    pred_df = pd.DataFrame(predictions_records)
    hyper_df = pd.DataFrame(hyperparams_records)

    # Compute overall LOSFO metrics
    losfo_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub = pred_df[pred_df["feature_block"] == blk]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        m["feature_block"] = blk
        m["regressor"] = "hist_gradient_boosting"
        m["n_predictions"] = len(sub)
        losfo_metrics.append(m)
    losfo_metrics_df = pd.DataFrame(losfo_metrics)

    # Compute per-family metrics
    losfo_fam_metrics = []
    for fam in ALL_SHIFT_FAMILIES:
        row_dict = {"shift_family": fam}
        for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            sub = pred_df[(pred_df["shift_family"] == fam) & (pred_df["feature_block"] == blk)]
            mae = float(mean_absolute_error(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values))
            row_dict[f"mae_{blk}"] = mae
        losfo_fam_metrics.append(row_dict)
    losfo_fam_df = pd.DataFrame(losfo_fam_metrics)

    # Paired family deltas: Delta_04^(f) = MAE_P0^(f) - MAE_P4^(f), Delta_34^(f) = MAE_P3^(f) - MAE_P4^(f)
    paired_fam_deltas = []
    for _, r in losfo_fam_df.iterrows():
        d_04 = r["mae_P0"] - r["mae_P4"]
        d_34 = r["mae_P3"] - r["mae_P4"]
        paired_fam_deltas.append({
            "shift_family": r["shift_family"],
            "mae_P0": r["mae_P0"],
            "mae_P3": r["mae_P3"],
            "mae_P4": r["mae_P4"],
            "delta_04": d_04,
            "delta_34": d_34,
        })
    paired_fam_df = pd.DataFrame(paired_fam_deltas)

    return {
        "predictions_df": pred_df,
        "hyperparameters_df": hyper_df,
        "losfo_metrics_df": losfo_metrics_df,
        "losfo_family_metrics_df": losfo_fam_df,
        "paired_family_deltas_df": paired_fam_df,
    }