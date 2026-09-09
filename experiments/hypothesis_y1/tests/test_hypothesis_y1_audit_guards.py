"""Executable guards for the isolated Hypothesis-Y1 audit."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from clusterdrift.falsification.execution.cache import (
    PersistentPhase7Cache,
    compute_model_cache_fingerprint,
    compute_source_cache_fingerprint,
)
from clusterdrift.data.loaders import _load_or_download_pinned_archive
from clusterdrift.methods.fcm import FCM
from clusterdrift.shifts.hashing import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npy,
    atomic_write_parquet,
)


ROOT = Path(__file__).resolve().parents[3]
Y1 = ROOT / "experiments/hypothesis_y1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_inputs_and_independent_reproduction_are_exact() -> None:
    assert digest(ROOT / "results/falsification/signals_label_free.csv") == "3a9e6c68cbe34c8763d70769715dba0d34976dfc35c8877c5680561da5c0e80a"
    assert digest(ROOT / "results/falsification/quality_evaluation_only.csv") == "1b3873bd92233702e761791e7b0f4c3e5f5a9f3f41320641fa4105e11c6a5fa5"
    result = json.loads((Y1 / "artifacts/hypothesis_y1_independent_replication.json").read_text(encoding="utf-8"))
    assert result["agreement"]["status"] == "PASSED"
    assert result["verdict"] == "FAILS_PRIMARY_FALSIFICATION"


def test_independent_reimplementation_has_no_forbidden_imports() -> None:
    path = Y1 / "scripts/hypothesis_y1_independent_phase7_replication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "clusterdrift.falsification.evaluation",
        "clusterdrift.falsification.models",
        "clusterdrift.falsification.bootstrap",
        "clusterdrift.falsification.verification",
    }
    assert imported.isdisjoint(forbidden)


def test_original_usable_flag_did_not_enforce_convergence() -> None:
    joined = pd.read_csv(ROOT / "results/falsification/joined_evaluation_table.csv")
    accepted_nonconverged = joined["usable"].astype(bool) & ~(
        joined["source_converged"].astype(bool) & joined["candidate_converged"].astype(bool)
    )
    assert int(accepted_nonconverged.sum()) == 635


def test_corrected_split_verification_reports_zero_group_crossing() -> None:
    summary = json.loads((Y1 / "artifacts/hypothesis_y1_corrected_split_summary.json").read_text(encoding="utf-8"))
    for dataset in summary["datasets"].values():
        for fold in dataset["folds"]:
            assert fold["outer_group_leakage"] is False
            assert fold["inner_group_leakage"] is False


def test_numeric_model_cache_roundtrip_and_tamper_rejection(tmp_path: Path) -> None:
    cache = PersistentPhase7Cache(tmp_path, "protocol", ROOT)
    rng = np.random.default_rng(7)
    x = np.r_[rng.normal(-1, 0.1, (30, 2)), rng.normal(1, 0.1, (30, 2))]
    model = FCM(n_clusters=2, random_state=1, fuzzifier_policy="dimension_adaptive").fit(x)
    cache.put_source_model("iris", 0, "fcm_adaptive", 1, 2, "input-fingerprint", model)
    restored = cache.get_source_model("iris", 0, "fcm_adaptive", 1, "input-fingerprint")
    assert restored is not None
    np.testing.assert_allclose(restored.predict_membership(x), model.predict_membership(x), rtol=0, atol=0)
    payload = cache.models_dir / "iris_fold_0_fcm_adaptive_seed_1.npz"
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 1
    payload.write_bytes(raw)
    assert cache.get_source_model("iris", 0, "fcm_adaptive", 1, "input-fingerprint") is None
    payload.unlink()
    legacy = cache.models_dir / "iris_fold_0_fcm_adaptive_seed_1.joblib"
    legacy.write_bytes(b"not deserialized")
    assert cache.get_source_model("iris", 0, "fcm_adaptive", 1, "input-fingerprint") is None
    assert cache.validate_cache()["corrupt_items"] >= 1


def test_array_cache_digest_and_fingerprint_are_required(tmp_path: Path) -> None:
    cache = PersistentPhase7Cache(tmp_path, "protocol", ROOT)
    cache.put_source_data("iris", 0, "fp", np.ones((4, 2)), np.ones((2, 2)))
    assert cache.get_source_data("iris", 0, "fp") is not None
    assert cache.get_source_data("iris", 0, "stale") is None
    array = cache.datasets_dir / "iris_fold_0_X_src.npy"
    value = np.load(array, allow_pickle=False)
    value[0, 0] = 99
    np.save(array, value, allow_pickle=False)
    assert cache.get_source_data("iris", 0, "fp") is None


def test_scalar_caches_require_expected_provenance(tmp_path: Path) -> None:
    cache = PersistentPhase7Cache(tmp_path, "protocol", ROOT)
    cache.put_mmd_sigma("iris", 0, "source-a", 1.2, "SUCCESS", 3.4)
    assert cache.get_mmd_sigma("iris", 0, "source-a") == (1.2, "SUCCESS", 3.4)
    assert cache.get_mmd_sigma("iris", 0, "source-b") is None
    cache.put_dx("iris", 0, "location_mild", "scenario-a", 0.3)
    assert cache.get_dx("iris", 0, "location_mild", "scenario-a") == 0.3
    assert cache.get_dx("iris", 0, "location_mild", "scenario-b") is None


def test_source_and_model_fingerprints_bind_input_artifacts(tmp_path: Path) -> None:
    dataset = "fixture"
    dataset_dir = tmp_path / f"data/canonical/controlled/{dataset}"
    split_dir = tmp_path / f"data/splits/controlled/{dataset}"
    probe_dir = tmp_path / f"data/probes/reference/{dataset}"
    config_dir = tmp_path / "configs"
    for directory in (dataset_dir, split_dir, probe_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "features.parquet").write_bytes(b"features-a")
    (dataset_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (split_dir / "fold_0.npz").write_bytes(b"split-a")
    (probe_dir / "fold_0.json").write_text("{}", encoding="utf-8")
    (config_dir / "methods.yaml").write_text("fcm: a", encoding="utf-8")
    source_a = compute_source_cache_fingerprint(tmp_path, dataset, 0, "prep")
    (dataset_dir / "features.parquet").write_bytes(b"features-b")
    source_b = compute_source_cache_fingerprint(tmp_path, dataset, 0, "prep")
    assert source_a != source_b
    model_a = compute_model_cache_fingerprint(tmp_path, source_b, "fcm_adaptive", 1, 2)
    (config_dir / "methods.yaml").write_text("fcm: b", encoding="utf-8")
    model_b = compute_model_cache_fingerprint(tmp_path, source_b, "fcm_adaptive", 1, 2)
    assert model_a != model_b


@pytest.mark.parametrize("bad", ["../escape", "a/b", "a\\b", ".."])
def test_cache_rejects_path_traversal_components(tmp_path: Path, bad: str) -> None:
    cache = PersistentPhase7Cache(tmp_path, "protocol", ROOT)
    with pytest.raises(ValueError):
        cache.get_source_data(bad, 0, "fp")


def test_cache_clear_rejects_escaped_root(tmp_path: Path) -> None:
    cache_parent = tmp_path / "cache"
    cache = PersistentPhase7Cache(cache_parent, "protocol", ROOT)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    cache.root = outside
    with pytest.raises(ValueError, match="escapes configured directory"):
        cache.clear_cache(confirmed=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_grouped_no_skill_baseline_is_recorded() -> None:
    metrics = pd.read_csv(Y1 / "artifacts/hypothesis_y1_no_skill_metrics.csv")
    median_lodo = metrics.query("evaluation == 'LODO' and feature_block == 'B0_median'").iloc[0]
    assert median_lodo["mae"] == pytest.approx(0.03800816450584381, abs=1e-15)
    assert median_lodo["mae"] < 0.04670466347898161


def test_archive_loader_enforces_pinned_digest_without_network(tmp_path: Path) -> None:
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"fixed scientific source bytes")
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert _load_or_download_pinned_archive("https://invalid.example/sample.zip", archive, expected) == archive.read_bytes()
    with pytest.raises(ValueError, match="digest mismatch"):
        _load_or_download_pinned_archive("https://invalid.example/sample.zip", archive, "0" * 64)
    with pytest.raises(ValueError, match="non-HTTPS"):
        _load_or_download_pinned_archive("http://invalid.example/missing.zip", tmp_path / "missing.zip", "0" * 64)


def test_atomic_scientific_writers_publish_readable_files(tmp_path: Path) -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0], "group": ["a", "b"]})
    csv_path = tmp_path / "table.csv"
    parquet_path = tmp_path / "table.parquet"
    array_path = tmp_path / "array.npy"
    json_path = tmp_path / "metadata.json"
    atomic_write_csv(csv_path, frame, index=False)
    atomic_write_parquet(parquet_path, frame, index=False, engine="pyarrow")
    atomic_write_npy(array_path, np.array([[1.0, 2.0]]))
    atomic_write_json(json_path, {"complete": True})
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
    pd.testing.assert_frame_equal(pd.read_parquet(parquet_path), frame)
    np.testing.assert_array_equal(np.load(array_path, allow_pickle=False), [[1.0, 2.0]])
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"complete": True}
    assert not list(tmp_path.glob(".*.tmp_*"))


def test_corrected_replication_is_complete_and_provenance_bound() -> None:
    corrected = Y1 / "corrected_replication"
    rows_path = corrected / "hypothesis_y1_corrected_rows.csv"
    metrics_path = corrected / "hypothesis_y1_corrected_model_metrics.csv"
    summary_path = corrected / "hypothesis_y1_corrected_replication_summary.json"
    provenance_path = Y1 / "artifacts/hypothesis_y1_corrected_execution_provenance.json"
    rows = pd.read_csv(rows_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert len(rows) == 9000
    assert set(rows["replication_arm"]) == {"duplicate_corrected_150_all", "fully_corrected_600"}
    assert rows["task_input_fingerprint"].nunique() == 60
    assert summary["row_file_sha256"] == digest(rows_path)
    assert provenance["result_sha256"]["corrected_rows"] == digest(rows_path)
    assert provenance["result_sha256"]["corrected_model_metrics"] == digest(metrics_path)
    assert provenance["result_sha256"]["corrected_summary"] == digest(summary_path)
    assert Path(provenance["interpreter"]).resolve() == Path(r"D:\Conda\p12\python.exe").resolve()


def test_decision_matrix_has_exact_required_mechanisms_and_columns() -> None:
    matrix = pd.read_csv(Y1 / "artifacts/hypothesis_y1_decision_matrix.csv")
    expected_columns = [
        "mechanism", "supporting evidence", "contradictory evidence", "datasets supporting",
        "families supporting", "effect magnitude", "confidence", "would it invalidate Phase 7?",
        "does it justify new hypothesis?", "untouched test required?",
    ]
    expected_mechanisms = {
        "Phase-7 evaluator implementation bug",
        "non-convergence contaminated structural signals",
        "duplicate leakage contaminated folds",
        "missing target classes invalidated folds",
        "raw validity controls were unfair",
        "raw structural signals are dataset-specific",
        "D_V contains real transferable signal",
        "D_H destroys transfer",
        "dimension-adaptive fuzzification causes instability",
        "exact Delta ARI regression is unsuitable",
        "risk ranking works while magnitude regression fails",
        "classification labels are poor cluster truth",
        "structural signals are genuinely weak",
        "dataset generalization differs from shift-family generalization",
        "signal redundancy hurts model transfer",
    }
    assert matrix.columns.tolist() == expected_columns
    assert len(matrix) == 15
    assert set(matrix["mechanism"]) == expected_mechanisms


def test_final_summary_and_master_report_are_decision_consistent() -> None:
    required_keys = {
        "original_phase7_reproduced", "original_P0", "original_P3", "original_P4",
        "original_verdict", "source_convergence_rate", "candidate_convergence_rate",
        "both_converged_rate", "datasets_with_severe_convergence_problem",
        "duplicate_leakage_datasets", "largest_duplicate_leakage_rate", "class_K_problem_folds",
        "no_skill_mean_mae", "no_skill_median_mae", "corrected_P0",
        "corrected_validity_control", "corrected_P4", "corrected_result_direction",
        "within_dataset_structural_help_count", "lodo_structural_help_count",
        "shift_family_structural_help_count", "strongest_mechanism", "counterevidence",
        "security_issue_count", "methodological_issue_count", "issues_fixed", "final_decision",
        "new_hypothesis_if_any", "untouched_confirmation_plan",
    }
    allowed = {
        "PHASE7_RESULT_INVALID_REQUIRES_NEW_CONFIRMATORY_REPLICATION",
        "FAILURE_CONFIRMED_NEW_HYPOTHESIS_JUSTIFIED",
        "FAILURE_CONFIRMED_STOP_PROJECT",
    }
    summary = json.loads((Y1 / "artifacts/hypothesis_y1_final_summary.json").read_text(encoding="utf-8"))
    master = (Y1 / "reports/hypothesis_y1_master_report.md").read_text(encoding="utf-8")
    assert required_keys <= summary.keys()
    assert summary["final_decision"] in allowed
    assert summary["original_phase7_reproduced"] is True
    assert summary["original_verdict"] == "FAILS_PRIMARY_FALSIFICATION"
    assert f"`{summary['final_decision']}`" in master
    assert "DO NOT CONTINUE ORIGINAL PHASE 8–15 ROADMAP." in master
    headings = re.findall(r"^## (\d+)\.", master, flags=re.MULTILINE)
    assert headings == [str(number) for number in range(1, 36)]
