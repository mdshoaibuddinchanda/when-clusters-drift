"""Canonical Hashing and Input-Lock Construction for Phase 6.

Guarantees cryptographic provenance across all Phase 6 signals and upstream dependencies.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file reading in 64KB blocks."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for sha256 computation: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_canonical_dict_sha256(data: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of a dictionary with sorted keys and canonical JSON formatting."""
    canon_bytes = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(canon_bytes).hexdigest()


def compute_signal_protocol_sha256(signals_cfg: Dict[str, Any]) -> str:
    """Compute SHA-256 hash of the frozen signal protocol definition."""
    protocol_keys = [
        "protocol_version",
        "signal_definition_version",
        "global_signal_seed",
        "js",
        "prototype_movement",
        "entropy",
        "cluster_mass",
        "mmd",
        "validity",
    ]
    sub_dict = {k: signals_cfg[k] for k in protocol_keys if k in signals_cfg}
    return compute_canonical_dict_sha256(sub_dict)


def compute_signal_record_sha256(rec: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of a single signal record."""
    binding_fields = {
        "signal_protocol_sha256": rec["signal_protocol_sha256"],
        "dataset_id": rec["dataset_id"],
        "outer_fold": int(rec["outer_fold"]),
        "condition": rec["condition"],
        "method": rec["method"],
        "seed": int(rec["seed"]),
        "reference_bank_sha256": rec["reference_bank_sha256"],
        "current_bank_sha256": rec["current_bank_sha256"],
        "shift_spec_sha256": rec["shift_spec_sha256"],
        "shift_spec_file_sha256": rec.get("shift_spec_file_sha256", ""),
        "shift_replay_sha256": rec.get("shift_replay_sha256") or "",
        "source_model_fingerprint": rec["source_model_fingerprint"],
        "candidate_model_fingerprint": rec["candidate_model_fingerprint"],
        "alignment_permutation": rec["alignment_permutation"] if isinstance(rec["alignment_permutation"], list) else json.loads(rec["alignment_permutation"]),
        "alignment_global_margin": round(float(rec["alignment_global_margin"]), 7),
        "D_U_R": round(float(rec["D_U_R"]), 7) if rec["D_U_R"] is not None else None,
        "D_U_C": round(float(rec["D_U_C"]), 7) if rec["D_U_C"] is not None else None,
        "D_V": round(float(rec["D_V"]), 7) if rec["D_V"] is not None else None,
        "D_H": round(float(rec["D_H"]), 7) if rec["D_H"] is not None else None,
        "D_M": round(float(rec["D_M"]), 7) if rec["D_M"] is not None else None,
        "D_X": round(float(rec["D_X"]), 7) if rec["D_X"] is not None else None,
        "FPC": round(float(rec["FPC"]), 7) if rec.get("FPC") is not None and not (isinstance(rec["FPC"], float) and (rec["FPC"] != rec["FPC"])) else None,
        "PE_norm": round(float(rec["PE_norm"]), 7) if rec.get("PE_norm") is not None and not (isinstance(rec["PE_norm"], float) and (rec["PE_norm"] != rec["PE_norm"])) else None,
        "XB_soft_m2": round(float(rec["XB_soft_m2"]), 7) if rec.get("XB_soft_m2") is not None and not (isinstance(rec["XB_soft_m2"], float) and (rec["XB_soft_m2"] != rec["XB_soft_m2"])) else None,
        "silhouette": round(float(rec["silhouette"]), 7) if rec.get("silhouette") is not None and not (isinstance(rec["silhouette"], float) and (rec["silhouette"] != rec["silhouette"])) else None,
        "usable": bool(rec["usable"]),
    }
    return compute_canonical_dict_sha256(binding_fields)


def compute_quality_record_sha256(rec: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of an evaluation-only quality record."""
    binding_fields = {
        "dataset_id": rec["dataset_id"],
        "outer_fold": int(rec["outer_fold"]),
        "condition": rec["condition"],
        "method": rec["method"],
        "seed": int(rec["seed"]),
        "quality_target": rec["quality_target"],
        "ari_clean": round(float(rec["ari_clean"]), 7),
        "ari_condition": round(float(rec["ari_condition"]), 7),
        "delta_ari": round(float(rec["delta_ari"]), 7),
        "nmi_condition": round(float(rec["nmi_condition"]), 7),
        "ami_condition": round(float(rec["ami_condition"]), 7),
        "n_evaluation_rows": int(rec["n_evaluation_rows"]),
    }
    return compute_canonical_dict_sha256(binding_fields)


def build_phase6_input_lock(
    project_root: Path,
    producer_commit: str,
    signals_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct complete Phase-6 cryptographic input lock document."""
    root = Path(project_root)
    signal_proto_sha = compute_signal_protocol_sha256(signals_cfg)

    lock = {
        "protocol_version": 1,
        "generated_from_commit": producer_commit,
        "phase5_freeze_commit": "5780b98836d4c8239e5c6d085a3d495dbd8484d4",
        "phase5_producer_commit": "412e20f6b8d0f54b21557dfece01e88fdac0ab47",
        "phase5_artifact_commit": "ca9dbb9b386b9a629af55d0482b451de371b6e1b",
        "phase5_input_lock_sha256": compute_file_sha256(root / "data" / "probes" / "phase5_input_lock.json"),
        "phase5_probe_manifest_sha256": compute_file_sha256(root / "data" / "probes" / "probe_manifest.json"),
        "phase4_input_lock_sha256": compute_file_sha256(root / "data" / "shifts" / "phase4_input_lock.json"),
        "phase4_shift_manifest_sha256": compute_file_sha256(root / "data" / "shifts" / "shift_manifest.json"),
        "phase2_split_manifest_sha256": compute_file_sha256(root / "data" / "splits" / "split_manifest.json"),
        "phase2_preprocessing_config_sha256": compute_file_sha256(root / "configs" / "preprocessing.yaml"),
        "phase1_datasets_manifest_sha256": compute_file_sha256(root / "data" / "manifests" / "datasets.json"),
        "methods_config_sha256": compute_file_sha256(root / "configs" / "methods.yaml"),
        "signals_config_sha256": compute_file_sha256(root / "configs" / "signals.yaml"),
        "alignment_config_sha256": compute_file_sha256(root / "configs" / "alignment.yaml"),
        "probe_config_sha256": compute_file_sha256(root / "configs" / "probes.yaml"),
        "signal_protocol_sha256": signal_proto_sha,
        "global_signal_seed": signals_cfg.get("global_signal_seed", 2026090706),
    }

    lock["phase6_input_lock_sha256"] = compute_canonical_dict_sha256(lock)
    return lock
