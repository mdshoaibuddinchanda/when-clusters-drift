"""Evaluation Routines: Primary LODO/LOSFO, Secondary Analyses, Linear Controls, and Diagnostics.

Execution-Only Reliability, Caching, Resumability, and Determinism Upgrades:
- Durable atomic per-job checkpointing with cryptographic envelope
- Full inner-CV candidate score logging for complete audit trail
- Secondary clean-inclusive LODO and LOSFO analyses
- Strict training-only preprocessing and thread-bounded execution
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from clusterdrift.falsification.dataset import (
    assign_dimension_stratum,
    validate_feature_block_isolation,
)
from clusterdrift.falsification.execution.checkpoint import EvaluationCheckpointManager
from clusterdrift.falsification.execution.environment import limit_inner_threads
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
    compute_prediction_record_sha256,
)
from clusterdrift.shifts.hashing import atomic_write_csv, compute_file_sha256


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute primary and secondary evaluation metrics."""
    arr_true = np.asarray(y_true, dtype=np.float64)
    arr_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(arr_true, arr_pred))
    mse = float(mean_squared_error(arr_true, arr_pred))
    rmse = float(np.sqrt(mse))
    med_ae = float(np.median(np.abs(arr_true - arr_pred)))

    var = float(np.var(arr_true))
    r2 = float(r2_score(arr_true, arr_pred)) if var > 1e-12 else 0.0
    return {
        "mae": mae,
        "rmse": rmse,
        "median_absolute_error": med_ae,
        "r2": r2,
    }


def _compute_key_universe_sha256(df: pd.DataFrame) -> str:
    """Compute deterministic SHA256 of sorted (dataset_id, fold, condition, seed) keys."""
    keys = [
        f"{r['dataset_id']}_{int(r['outer_fold'])}_{r['condition']}_{r['method']}_{int(r['seed'])}"
        for _, r in df.iterrows()
    ]
    keys.sort()
    return hashlib.sha256("|".join(keys).encode("utf-8")).hexdigest()


def evaluate_single_job(
    analysis_scope: str,
    outer_eval_type: str,
    outer_test_group: str,
    feature_block: str,
    features: List[str],
    regressor: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    cfg: Dict[str, Any],
    checkpoint_mgr: Optional[EvaluationCheckpointManager] = None,
    protocol_sha: str = "",
    pass_a_sha: str = "",
    pass_b_sha: str = "",
    joined_sha: str = "",
) -> Dict[str, Any]:
    """Evaluate a single atomic outer unit job with caching and provenance binding."""
    validate_feature_block_isolation(features)

    train_key_sha = _compute_key_universe_sha256(train_data)
    test_key_sha = _compute_key_universe_sha256(test_data)

    # Check cache/checkpoint
    if checkpoint_mgr is not None:
        cached = checkpoint_mgr.load_checkpoint(
            analysis_scope,
            outer_eval_type,
            outer_test_group,
            feature_block,
            regressor,
            expected_train_sha=train_key_sha,
            expected_test_sha=test_key_sha,
        )
        if cached is not None:
            return cached

    # Prepare numpy arrays
    groups_train = train_data["dataset_id"].values
    y_train = train_data["delta_ari"].values.astype(np.float64)
    y_test = test_data["delta_ari"].values.astype(np.float64)

    X_train = train_data[features].values.astype(np.float64)
    X_test = test_data[features].values.astype(np.float64)

    global_seed = cfg.get("global_seed", 2026090707)

    with limit_inner_threads(1):
        if regressor == "hist_gradient_boosting":
            hgbr_grid = cfg["primary_regressor"]["grid"]
            model, prep, selection = tune_and_fit_hgbr(
                X_train, y_train, groups_train, hgbr_grid, random_state=global_seed
            )
        elif regressor == "ridge":
            ridge_alphas = cfg["linear_control"]["alphas"]
            model, prep, selection = tune_and_fit_ridge(
                X_train, y_train, groups_train, ridge_alphas, random_state=global_seed
            )
        else:
            raise ValueError(f"Unknown regressor: {regressor}")

        X_test_trans = prep.transform(X_test)
        preds = model.predict(X_test_trans)

    # Build prediction records
    predictions: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(test_data.iterrows()):
        obs = float(row["delta_ari"])
        pred = float(preds[idx])
        abs_err = abs(obs - pred)
        sq_err = (obs - pred) ** 2
        rec = {
            "analysis_scope": analysis_scope,
            "outer_eval_type": outer_eval_type,
            "outer_test_group": outer_test_group,
            "dataset_id": str(row["dataset_id"]),
            "outer_fold": int(row["outer_fold"]),
            "condition": str(row["condition"]),
            "shift_family": str(row["shift_family"]),
            "severity": str(row["severity"]),
            "method": str(row["method"]),
            "seed": int(row["seed"]),
            "feature_block": feature_block,
            "regressor": regressor,
            "observed_delta_ari": obs,
            "predicted_delta_ari": pred,
            "absolute_error": abs_err,
            "squared_error": sq_err,
            "phase7_protocol_sha": protocol_sha,
            "pass_a_signals_sha": pass_a_sha,
            "pass_b_quality_sha": pass_b_sha,
        }
        rec["prediction_record_sha256"] = compute_prediction_record_sha256(rec)
        predictions.append(rec)

    envelope: Dict[str, Any] = {
        "phase7_protocol_sha256": protocol_sha,
        "pass_a_signals_sha256": pass_a_sha,
        "pass_b_quality_sha256": pass_b_sha,
        "joined_table_sha256": joined_sha,
        "analysis_scope": analysis_scope,
        "outer_eval_type": outer_eval_type,
        "outer_test_group": outer_test_group,
        "feature_block": feature_block,
        "regressor": regressor,
        "training_key_universe_sha256": train_key_sha,
        "test_key_universe_sha256": test_key_sha,
        "inner_split_sha256": selection.inner_splits_sha256,
        "selected_params": selection.selected_params,
        "best_inner_score": selection.best_score,
        "tie_broken": selection.tie_broken,
        "inner_cv_candidates": selection.candidates,
        "predictions": predictions,
    }

    if checkpoint_mgr is not None:
        checkpoint_mgr.save_checkpoint(envelope)

    return envelope


def run_lodo_evaluation(
    joined_df: pd.DataFrame,
    cfg: Dict[str, Any],
    checkpoint_mgr: Optional[EvaluationCheckpointManager] = None,
    protocol_sha: str = "",
    pass_a_sha: str = "",
    pass_b_sha: str = "",
    joined_sha: str = "",
) -> Dict[str, Any]:
    """Execute Primary Leave-One-Dataset-Out (LODO) nested evaluation across 12 datasets."""
    datasets = cfg["datasets"]
    feature_blocks = cfg["feature_blocks"]

    # Primary analysis strictly excludes clean
    primary_df = joined_df[joined_df["condition"] != "clean"].copy()
    assert len(primary_df) == 4200, f"Expected 4200 primary rows, got {len(primary_df)}"

    eval_blocks: Dict[str, List[str]] = dict(feature_blocks)
    for k, v in P4_STRUCTURAL_ABLATIONS.items():
        eval_blocks[k] = v
    for k, v in SINGLE_SIGNAL_INCREMENTAL.items():
        eval_blocks[k] = v

    all_envelopes: List[Dict[str, Any]] = []

    for test_ds in datasets:
        train_mask = (primary_df["dataset_id"] != test_ds)
        test_mask = (primary_df["dataset_id"] == test_ds)

        train_data = primary_df[train_mask]
        test_data = primary_df[test_mask]

        assert test_ds not in train_data["dataset_id"].values
        assert set(train_data["dataset_id"].values).isdisjoint(set(test_data["dataset_id"].values))
        assert len(train_data) == 3850, f"Expected 3850 LODO train rows, got {len(train_data)}"
        assert len(test_data) == 350, f"Expected 350 LODO test rows, got {len(test_data)}"

        # 1. HGBR for all blocks
        for block_name, features in eval_blocks.items():
            env = evaluate_single_job(
                analysis_scope="primary",
                outer_eval_type="LODO",
                outer_test_group=test_ds,
                feature_block=block_name,
                features=features,
                regressor="hist_gradient_boosting",
                train_data=train_data,
                test_data=test_data,
                cfg=cfg,
                checkpoint_mgr=checkpoint_mgr,
                protocol_sha=protocol_sha,
                pass_a_sha=pass_a_sha,
                pass_b_sha=pass_b_sha,
                joined_sha=joined_sha,
            )
            all_envelopes.append(env)

        # 2. Ridge linear control for standard blocks P0-P5
        for block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            env = evaluate_single_job(
                analysis_scope="primary",
                outer_eval_type="LODO",
                outer_test_group=test_ds,
                feature_block=block_name,
                features=feature_blocks[block_name],
                regressor="ridge",
                train_data=train_data,
                test_data=test_data,
                cfg=cfg,
                checkpoint_mgr=checkpoint_mgr,
                protocol_sha=protocol_sha,
                pass_a_sha=pass_a_sha,
                pass_b_sha=pass_b_sha,
                joined_sha=joined_sha,
            )
            all_envelopes.append(env)

    # Assemble tables
    pred_records = []
    hyper_records = []
    inner_cv_records = []

    for env in all_envelopes:
        pred_records.extend(env["predictions"])
        hyper_records.append({
            "analysis_scope": env["analysis_scope"],
            "outer_eval_type": env["outer_eval_type"],
            "outer_test_group": env["outer_test_group"],
            "feature_block": env["feature_block"],
            "regressor": env["regressor"],
            "best_inner_mae": env["best_inner_score"],
            "selected_params": json.dumps(env["selected_params"]),
            "tie_broken": env["tie_broken"],
        })
        for cand in env["inner_cv_candidates"]:
            p_json = json.dumps(cand["params"])
            for f_rec in cand.get("fold_records", []):
                inner_cv_records.append({
                    "analysis_scope": env["analysis_scope"],
                    "outer_eval_type": env["outer_eval_type"],
                    "outer_test_group": env["outer_test_group"],
                    "feature_block": env["feature_block"],
                    "regressor": env["regressor"],
                    "candidate_params": p_json,
                    "inner_fold": f_rec["inner_fold"],
                    "inner_training_group_sha256": f_rec["inner_training_group_sha256"],
                    "inner_validation_group_sha256": f_rec["inner_validation_group_sha256"],
                    "inner_fold_mae": f_rec["inner_fold_mae"],
                    "mean_candidate_mae": cand["mean_mae"],
                    "selected": cand.get("selected", False),
                    "tie_status": cand.get("tie_status", False),
                })

    pred_df = pd.DataFrame(pred_records)
    hyper_df = pd.DataFrame(hyper_records)
    inner_cv_df = pd.DataFrame(inner_cv_records)

    assert len(pred_df) == 92400, f"Expected 92,400 primary LODO predictions, got {len(pred_df)}"

    # Primary LODO metrics for HGBR P0-P5
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

    # Per-dataset metrics for HGBR
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
            (hgbr_lodo["feature_block"] == "P0")
            & (hgbr_lodo["dataset_id"].isin(primary_df[primary_df["dimension_stratum"] == stratum]["dataset_id"].unique()))
        ]
        stratum_sub_p4 = hgbr_lodo[
            (hgbr_lodo["feature_block"] == "P4")
            & (hgbr_lodo["dataset_id"].isin(primary_df[primary_df["dimension_stratum"] == stratum]["dataset_id"].unique()))
        ]
        if len(stratum_sub_p0) > 0 and len(stratum_sub_p4) > 0:
            m_p0 = compute_metrics(stratum_sub_p0["observed_delta_ari"].values, stratum_sub_p0["predicted_delta_ari"].values)
            m_p4 = compute_metrics(stratum_sub_p4["observed_delta_ari"].values, stratum_sub_p4["predicted_delta_ari"].values)

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
        "inner_cv_scores_df": inner_cv_df,
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
    checkpoint_mgr: Optional[EvaluationCheckpointManager] = None,
    protocol_sha: str = "",
    pass_a_sha: str = "",
    pass_b_sha: str = "",
    joined_sha: str = "",
) -> Dict[str, Any]:
    """Execute Primary Leave-One-Shift-Family-Out (LOSFO) nested evaluation across 7 families."""
    feature_blocks = cfg["feature_blocks"]

    # Primary analysis strictly excludes clean
    primary_df = joined_df[joined_df["condition"] != "clean"].copy()
    assert len(primary_df) == 4200, f"Expected 4200 primary rows, got {len(primary_df)}"

    all_envelopes: List[Dict[str, Any]] = []

    for test_fam in ALL_SHIFT_FAMILIES:
        train_mask = (primary_df["shift_family"] != test_fam)
        test_mask = (primary_df["shift_family"] == test_fam)

        train_data = primary_df[train_mask]
        test_data = primary_df[test_mask]

        assert test_fam not in train_data["shift_family"].values
        assert set(train_data["shift_family"].values).isdisjoint(set(test_data["shift_family"].values))
        assert len(train_data) == 3600, f"Expected 3600 LOSFO train rows, got {len(train_data)}"
        assert len(test_data) == 600, f"Expected 600 LOSFO test rows, got {len(test_data)}"

        for block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            env = evaluate_single_job(
                analysis_scope="primary",
                outer_eval_type="LOSFO",
                outer_test_group=test_fam,
                feature_block=block_name,
                features=feature_blocks[block_name],
                regressor="hist_gradient_boosting",
                train_data=train_data,
                test_data=test_data,
                cfg=cfg,
                checkpoint_mgr=checkpoint_mgr,
                protocol_sha=protocol_sha,
                pass_a_sha=pass_a_sha,
                pass_b_sha=pass_b_sha,
                joined_sha=joined_sha,
            )
            all_envelopes.append(env)

    pred_records = []
    hyper_records = []
    inner_cv_records = []

    for env in all_envelopes:
        pred_records.extend(env["predictions"])
        hyper_records.append({
            "analysis_scope": env["analysis_scope"],
            "outer_eval_type": env["outer_eval_type"],
            "outer_test_group": env["outer_test_group"],
            "feature_block": env["feature_block"],
            "regressor": env["regressor"],
            "best_inner_mae": env["best_inner_score"],
            "selected_params": json.dumps(env["selected_params"]),
            "tie_broken": env["tie_broken"],
        })
        for cand in env["inner_cv_candidates"]:
            p_json = json.dumps(cand["params"])
            for f_rec in cand.get("fold_records", []):
                inner_cv_records.append({
                    "analysis_scope": env["analysis_scope"],
                    "outer_eval_type": env["outer_eval_type"],
                    "outer_test_group": env["outer_test_group"],
                    "feature_block": env["feature_block"],
                    "regressor": env["regressor"],
                    "candidate_params": p_json,
                    "inner_fold": f_rec["inner_fold"],
                    "inner_training_group_sha256": f_rec["inner_training_group_sha256"],
                    "inner_validation_group_sha256": f_rec["inner_validation_group_sha256"],
                    "inner_fold_mae": f_rec["inner_fold_mae"],
                    "mean_candidate_mae": cand["mean_mae"],
                    "selected": cand.get("selected", False),
                    "tie_status": cand.get("tie_status", False),
                })

    pred_df = pd.DataFrame(pred_records)
    hyper_df = pd.DataFrame(hyper_records)
    inner_cv_df = pd.DataFrame(inner_cv_records)

    assert len(pred_df) == 25200, f"Expected 25,200 primary LOSFO predictions, got {len(pred_df)}"

    # Primary LOSFO metrics
    losfo_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub = pred_df[pred_df["feature_block"] == blk]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        m["feature_block"] = blk
        m["regressor"] = "hist_gradient_boosting"
        m["n_predictions"] = len(sub)
        losfo_metrics.append(m)
    losfo_metrics_df = pd.DataFrame(losfo_metrics)

    # Per-family metrics
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
        "inner_cv_scores_df": inner_cv_df,
        "losfo_metrics_df": losfo_metrics_df,
        "losfo_family_metrics_df": losfo_fam_df,
        "paired_family_deltas_df": paired_fam_df,
    }


def run_secondary_lodo_evaluation(
    joined_df: pd.DataFrame,
    cfg: Dict[str, Any],
    checkpoint_mgr: Optional[EvaluationCheckpointManager] = None,
    protocol_sha: str = "",
    pass_a_sha: str = "",
    pass_b_sha: str = "",
    joined_sha: str = "",
) -> Dict[str, Any]:
    """Execute Secondary Clean-Inclusive LODO nested evaluation across 12 datasets (4500 rows)."""
    datasets = cfg["datasets"]
    feature_blocks = cfg["feature_blocks"]

    assert len(joined_df) == 4500, f"Expected 4500 joined rows for secondary LODO, got {len(joined_df)}"

    all_envelopes: List[Dict[str, Any]] = []

    for test_ds in datasets:
        train_mask = (joined_df["dataset_id"] != test_ds)
        test_mask = (joined_df["dataset_id"] == test_ds)

        train_data = joined_df[train_mask]
        test_data = joined_df[test_mask]

        assert test_ds not in train_data["dataset_id"].values
        assert set(train_data["dataset_id"].values).isdisjoint(set(test_data["dataset_id"].values))
        assert len(train_data) == 4125, f"Expected 4125 secondary LODO train rows, got {len(train_data)}"
        assert len(test_data) == 375, f"Expected 375 secondary LODO test rows, got {len(test_data)}"

        for block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            env = evaluate_single_job(
                analysis_scope="secondary",
                outer_eval_type="LODO",
                outer_test_group=test_ds,
                feature_block=block_name,
                features=feature_blocks[block_name],
                regressor="hist_gradient_boosting",
                train_data=train_data,
                test_data=test_data,
                cfg=cfg,
                checkpoint_mgr=checkpoint_mgr,
                protocol_sha=protocol_sha,
                pass_a_sha=pass_a_sha,
                pass_b_sha=pass_b_sha,
                joined_sha=joined_sha,
            )
            all_envelopes.append(env)

    pred_records = []
    hyper_records = []
    inner_cv_records = []

    for env in all_envelopes:
        pred_records.extend(env["predictions"])
        hyper_records.append({
            "analysis_scope": env["analysis_scope"],
            "outer_eval_type": env["outer_eval_type"],
            "outer_test_group": env["outer_test_group"],
            "feature_block": env["feature_block"],
            "regressor": env["regressor"],
            "best_inner_mae": env["best_inner_score"],
            "selected_params": json.dumps(env["selected_params"]),
            "tie_broken": env["tie_broken"],
        })
        for cand in env["inner_cv_candidates"]:
            p_json = json.dumps(cand["params"])
            for f_rec in cand.get("fold_records", []):
                inner_cv_records.append({
                    "analysis_scope": env["analysis_scope"],
                    "outer_eval_type": env["outer_eval_type"],
                    "outer_test_group": env["outer_test_group"],
                    "feature_block": env["feature_block"],
                    "regressor": env["regressor"],
                    "candidate_params": p_json,
                    "inner_fold": f_rec["inner_fold"],
                    "inner_training_group_sha256": f_rec["inner_training_group_sha256"],
                    "inner_validation_group_sha256": f_rec["inner_validation_group_sha256"],
                    "inner_fold_mae": f_rec["inner_fold_mae"],
                    "mean_candidate_mae": cand["mean_mae"],
                    "selected": cand.get("selected", False),
                    "tie_status": cand.get("tie_status", False),
                })

    pred_df = pd.DataFrame(pred_records)
    hyper_df = pd.DataFrame(hyper_records)
    inner_cv_df = pd.DataFrame(inner_cv_records)

    assert len(pred_df) == 27000, f"Expected 27,000 secondary LODO predictions, got {len(pred_df)}"

    # Metrics
    sec_lodo_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub = pred_df[pred_df["feature_block"] == blk]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        m["feature_block"] = blk
        m["regressor"] = "hist_gradient_boosting"
        m["n_predictions"] = len(sub)
        sec_lodo_metrics.append(m)
    sec_lodo_metrics_df = pd.DataFrame(sec_lodo_metrics)

    # Per-dataset metrics
    sec_ds_metrics = []
    for ds in datasets:
        row_dict = {"dataset_id": ds}
        for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            sub = pred_df[(pred_df["dataset_id"] == ds) & (pred_df["feature_block"] == blk)]
            mae = float(mean_absolute_error(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values))
            row_dict[f"mae_{blk}"] = mae
        sec_ds_metrics.append(row_dict)
    sec_ds_df = pd.DataFrame(sec_ds_metrics)

    return {
        "predictions_df": pred_df,
        "hyperparameters_df": hyper_df,
        "inner_cv_scores_df": inner_cv_df,
        "secondary_lodo_metrics_df": sec_lodo_metrics_df,
        "secondary_lodo_dataset_metrics_df": sec_ds_df,
    }


def run_secondary_losfo_evaluation(
    joined_df: pd.DataFrame,
    cfg: Dict[str, Any],
    checkpoint_mgr: Optional[EvaluationCheckpointManager] = None,
    protocol_sha: str = "",
    pass_a_sha: str = "",
    pass_b_sha: str = "",
    joined_sha: str = "",
) -> Dict[str, Any]:
    """Execute Secondary Clean-Inclusive LOSFO nested evaluation across 7 families.

    Test set: 600 shifted rows for held-out family.
    Train set: 3600 shifted rows from other families + 300 clean rows = 3900 rows.
    """
    feature_blocks = cfg["feature_blocks"]

    all_envelopes: List[Dict[str, Any]] = []

    for test_fam in ALL_SHIFT_FAMILIES:
        test_mask = (joined_df["shift_family"] == test_fam)
        train_mask = (joined_df["shift_family"] != test_fam)

        train_data = joined_df[train_mask]
        test_data = joined_df[test_mask]

        assert test_fam not in train_data["shift_family"].values
        assert set(train_data["shift_family"].values).isdisjoint(set(test_data["shift_family"].values))
        assert len(train_data) == 3900, f"Expected 3900 secondary LOSFO train rows, got {len(train_data)}"
        assert len(test_data) == 600, f"Expected 600 secondary LOSFO test rows, got {len(test_data)}"

        for block_name in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            env = evaluate_single_job(
                analysis_scope="secondary",
                outer_eval_type="LOSFO",
                outer_test_group=test_fam,
                feature_block=block_name,
                features=feature_blocks[block_name],
                regressor="hist_gradient_boosting",
                train_data=train_data,
                test_data=test_data,
                cfg=cfg,
                checkpoint_mgr=checkpoint_mgr,
                protocol_sha=protocol_sha,
                pass_a_sha=pass_a_sha,
                pass_b_sha=pass_b_sha,
                joined_sha=joined_sha,
            )
            all_envelopes.append(env)

    pred_records = []
    hyper_records = []
    inner_cv_records = []

    for env in all_envelopes:
        pred_records.extend(env["predictions"])
        hyper_records.append({
            "analysis_scope": env["analysis_scope"],
            "outer_eval_type": env["outer_eval_type"],
            "outer_test_group": env["outer_test_group"],
            "feature_block": env["feature_block"],
            "regressor": env["regressor"],
            "best_inner_mae": env["best_inner_score"],
            "selected_params": json.dumps(env["selected_params"]),
            "tie_broken": env["tie_broken"],
        })
        for cand in env["inner_cv_candidates"]:
            p_json = json.dumps(cand["params"])
            for f_rec in cand.get("fold_records", []):
                inner_cv_records.append({
                    "analysis_scope": env["analysis_scope"],
                    "outer_eval_type": env["outer_eval_type"],
                    "outer_test_group": env["outer_test_group"],
                    "feature_block": env["feature_block"],
                    "regressor": env["regressor"],
                    "candidate_params": p_json,
                    "inner_fold": f_rec["inner_fold"],
                    "inner_training_group_sha256": f_rec["inner_training_group_sha256"],
                    "inner_validation_group_sha256": f_rec["inner_validation_group_sha256"],
                    "inner_fold_mae": f_rec["inner_fold_mae"],
                    "mean_candidate_mae": cand["mean_mae"],
                    "selected": cand.get("selected", False),
                    "tie_status": cand.get("tie_status", False),
                })

    pred_df = pd.DataFrame(pred_records)
    hyper_df = pd.DataFrame(hyper_records)
    inner_cv_df = pd.DataFrame(inner_cv_records)

    assert len(pred_df) == 25200, f"Expected 25,200 secondary LOSFO predictions, got {len(pred_df)}"

    # Metrics
    sec_losfo_metrics = []
    for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
        sub = pred_df[pred_df["feature_block"] == blk]
        m = compute_metrics(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values)
        m["feature_block"] = blk
        m["regressor"] = "hist_gradient_boosting"
        m["n_predictions"] = len(sub)
        sec_losfo_metrics.append(m)
    sec_losfo_metrics_df = pd.DataFrame(sec_losfo_metrics)

    # Per-family metrics
    sec_fam_metrics = []
    for fam in ALL_SHIFT_FAMILIES:
        row_dict = {"shift_family": fam}
        for blk in ["P0", "P1", "P2", "P3", "P4", "P5"]:
            sub = pred_df[(pred_df["shift_family"] == fam) & (pred_df["feature_block"] == blk)]
            mae = float(mean_absolute_error(sub["observed_delta_ari"].values, sub["predicted_delta_ari"].values))
            row_dict[f"mae_{blk}"] = mae
        sec_fam_metrics.append(row_dict)
    sec_fam_df = pd.DataFrame(sec_fam_metrics)

    return {
        "predictions_df": pred_df,
        "hyperparameters_df": hyper_df,
        "inner_cv_scores_df": inner_cv_df,
        "secondary_losfo_metrics_df": sec_losfo_metrics_df,
        "secondary_losfo_family_metrics_df": sec_fam_df,
    }
