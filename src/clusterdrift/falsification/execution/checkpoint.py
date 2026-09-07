"""Fine-Grained Atomic Checkpointing and Artifact Assembly for Phase 7."""

import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from clusterdrift.falsification.execution.resources import assert_not_frozen_data_path
from clusterdrift.shifts.hashing import atomic_write_csv, atomic_write_json
from clusterdrift.signals.hashing import (
    compute_quality_record_sha256,
    compute_signal_record_sha256,
)


class SignalCheckpointManager:
    """Manages 4,500 fine-grained atomic signal checkpoints."""

    def __init__(self, work_dir: Path, project_root: Optional[Path] = None):
        self.project_root = Path(project_root).resolve() if project_root else None
        self.work_dir = assert_not_frozen_data_path(work_dir, self.project_root)
        self.checkpoints_dir = self.work_dir / "checkpoints" / "signals"
        self.quarantine_dir = self.work_dir / "checkpoints" / "quarantine"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, dataset_id: str, outer_fold: int, condition: str, seed: int) -> Path:
        return self.checkpoints_dir / dataset_id / f"fold_{outer_fold}" / condition / f"seed_{seed}.json"

    def is_checkpoint_valid(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        method: str,
        seed: int,
        expected_protocol_sha: Optional[str] = None,
    ) -> bool:
        """Check if a checkpoint exists and passes strict cryptographic & scientific validation."""
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        if not p.exists() or p.stat().st_size == 0:
            return False

        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)

            # Key check
            if (
                rec.get("dataset_id") != dataset_id
                or rec.get("outer_fold") != outer_fold
                or rec.get("condition") != condition
                or rec.get("method") != method
                or rec.get("seed") != seed
            ):
                return False

            # Protocol check
            if expected_protocol_sha and rec.get("signal_protocol_sha256") != expected_protocol_sha:
                return False

            # Hash recomputation check
            stored_sha = rec.get("signal_record_sha256")
            recomputed_sha = compute_signal_record_sha256(rec)
            if stored_sha != recomputed_sha:
                # Quarantine corrupt checkpoint
                q_path = self.quarantine_dir / f"corrupt_{p.name}"
                shutil.move(p, q_path)
                return False

            return True
        except Exception:
            # Quarantine broken JSON
            try:
                q_path = self.quarantine_dir / f"broken_{p.name}"
                shutil.move(p, q_path)
            except Exception:
                pass
            return False

    def save_checkpoint(self, rec: Dict[str, Any]) -> Path:
        """Atomically persist a single completed signal row checkpoint."""
        p = self._get_checkpoint_path(
            rec["dataset_id"],
            int(rec["outer_fold"]),
            rec["condition"],
            int(rec["seed"]),
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(p, rec, indent=2)
        return p

    def load_checkpoint(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        method: str,
        seed: int,
    ) -> Optional[Dict[str, Any]]:
        """Load and return valid checkpoint record if exists."""
        if not self.is_checkpoint_valid(dataset_id, outer_fold, condition, method, seed):
            return None
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_completed_checkpoint_keys(
        self,
        expected_protocol_sha: Optional[str] = None,
    ) -> Set[Tuple[str, int, str, str, int]]:
        """Enumerate all currently valid completed checkpoint keys."""
        valid_keys = set()
        for p in self.checkpoints_dir.glob("*/*/*/*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if expected_protocol_sha and rec.get("signal_protocol_sha256") != expected_protocol_sha:
                    continue
                if compute_signal_record_sha256(rec) == rec.get("signal_record_sha256"):
                    valid_keys.add((
                        rec["dataset_id"],
                        int(rec["outer_fold"]),
                        rec["condition"],
                        rec["method"],
                        int(rec["seed"]),
                    ))
            except Exception:
                continue
        return valid_keys

    def assemble_signals_csv(
        self,
        target_csv_path: Path,
        expected_keys: List[Tuple[str, int, str, str, int]],
    ) -> pd.DataFrame:
        """Assemble all verified checkpoints into canonical CSV artifact."""
        records: List[Dict[str, Any]] = []
        missing_keys = []

        for ds, fold, cond, meth, seed in expected_keys:
            rec = self.load_checkpoint(ds, fold, cond, meth, seed)
            if rec is None:
                missing_keys.append((ds, fold, cond, meth, seed))
            else:
                records.append(rec)

        if missing_keys:
            raise ValueError(
                f"Cannot assemble signals artifact: {len(missing_keys)} missing/corrupt checkpoints out of {len(expected_keys)}"
            )

        df = pd.DataFrame(records)
        df.sort_values(
            by=["dataset_id", "outer_fold", "condition", "method", "seed"],
            inplace=True,
        )
        atomic_write_csv(target_csv_path, df)
        return df


class QualityCheckpointManager:
    """Manages 4,500 fine-grained atomic evaluation-only quality checkpoints with provenance envelope."""

    def __init__(
        self,
        work_dir: Path,
        project_root: Optional[Path] = None,
        phase7_protocol_sha256: Optional[str] = None,
        pass_a_signals_sha256: Optional[str] = None,
        pass_a_freeze_commit: Optional[str] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else None
        self.work_dir = assert_not_frozen_data_path(work_dir, self.project_root)
        self.phase7_protocol_sha256 = phase7_protocol_sha256
        self.pass_a_signals_sha256 = pass_a_signals_sha256
        self.pass_a_freeze_commit = pass_a_freeze_commit
        self.checkpoints_dir = self.work_dir / "checkpoints" / "quality"
        self.quarantine_dir = self.work_dir / "checkpoints" / "quarantine_quality"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, dataset_id: str, outer_fold: int, condition: str, seed: int) -> Path:
        return self.checkpoints_dir / dataset_id / f"fold_{outer_fold}" / condition / f"seed_{seed}.json"

    def is_checkpoint_valid(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        method: str,
        seed: int,
        expected_protocol_sha: Optional[str] = None,
        expected_signals_sha: Optional[str] = None,
        expected_freeze_commit: Optional[str] = None,
    ) -> bool:
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        if not p.exists() or p.stat().st_size == 0:
            return False

        try:
            with open(p, "r", encoding="utf-8") as f:
                envelope = json.load(f)

            if not isinstance(envelope, dict):
                return False

            # Provenance checks
            proto_sha = expected_protocol_sha or self.phase7_protocol_sha256
            if proto_sha and envelope.get("phase7_protocol_sha256") != proto_sha:
                return False

            sig_sha = expected_signals_sha or self.pass_a_signals_sha256
            if sig_sha and envelope.get("pass_a_signals_sha256") != sig_sha:
                return False

            freeze_com = expected_freeze_commit or self.pass_a_freeze_commit
            if freeze_com and envelope.get("pass_a_freeze_commit") != freeze_com:
                return False

            quality_rec = envelope.get("quality_record")
            if not isinstance(quality_rec, dict):
                return False

            # Key coordinate checks
            if (
                quality_rec.get("dataset_id") != dataset_id
                or int(quality_rec.get("outer_fold")) != int(outer_fold)
                or quality_rec.get("condition") != condition
                or quality_rec.get("method") != method
                or int(quality_rec.get("seed")) != int(seed)
            ):
                return False

            # Cryptographic quality record hash recomputation check
            stored_sha = envelope.get("quality_record_sha256")
            recomputed_sha = compute_quality_record_sha256(quality_rec)
            if stored_sha != recomputed_sha or quality_rec.get("quality_record_sha256") != recomputed_sha:
                q_path = self.quarantine_dir / f"corrupt_{p.name}"
                shutil.move(p, q_path)
                return False

            return True
        except Exception:
            try:
                q_path = self.quarantine_dir / f"broken_{p.name}"
                shutil.move(p, q_path)
            except Exception:
                pass
            return False

    def save_checkpoint(self, rec: Dict[str, Any]) -> Path:
        p = self._get_checkpoint_path(
            rec["dataset_id"],
            int(rec["outer_fold"]),
            rec["condition"],
            int(rec["seed"]),
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        rec_copy = dict(rec)
        sha = compute_quality_record_sha256(rec_copy)
        rec_copy["quality_record_sha256"] = sha
        envelope = {
            "phase7_protocol_sha256": self.phase7_protocol_sha256 or "",
            "pass_a_signals_sha256": self.pass_a_signals_sha256 or "",
            "pass_a_freeze_commit": self.pass_a_freeze_commit or "",
            "quality_record_sha256": sha,
            "quality_record": rec_copy,
        }
        atomic_write_json(p, envelope, indent=2)
        return p

    def load_checkpoint(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        method: str,
        seed: int,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_checkpoint_valid(dataset_id, outer_fold, condition, method, seed):
            return None
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        with open(p, "r", encoding="utf-8") as f:
            envelope = json.load(f)
        return envelope.get("quality_record")

    def get_completed_checkpoint_keys(
        self,
        expected_protocol_sha: Optional[str] = None,
        expected_signals_sha: Optional[str] = None,
        expected_freeze_commit: Optional[str] = None,
    ) -> Set[Tuple[str, int, str, str, int]]:
        valid_keys = set()
        for p in self.checkpoints_dir.glob("*/*/*/*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    envelope = json.load(f)

                proto_sha = expected_protocol_sha or self.phase7_protocol_sha256
                if proto_sha and envelope.get("phase7_protocol_sha256") != proto_sha:
                    continue
                sig_sha = expected_signals_sha or self.pass_a_signals_sha256
                if sig_sha and envelope.get("pass_a_signals_sha256") != sig_sha:
                    continue
                freeze_com = expected_freeze_commit or self.pass_a_freeze_commit
                if freeze_com and envelope.get("pass_a_freeze_commit") != freeze_com:
                    continue

                rec = envelope.get("quality_record")
                if not isinstance(rec, dict):
                    continue

                stored_sha = envelope.get("quality_record_sha256")
                if compute_quality_record_sha256(rec) == stored_sha:
                    valid_keys.add((
                        rec["dataset_id"],
                        int(rec["outer_fold"]),
                        rec["condition"],
                        rec["method"],
                        int(rec["seed"]),
                    ))
            except Exception:
                continue
        return valid_keys

    def assemble_quality_csv(
        self,
        target_csv_path: Path,
        expected_keys: List[Tuple[str, int, str, str, int]],
    ) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        missing_keys = []

        for ds, fold, cond, meth, seed in expected_keys:
            rec = self.load_checkpoint(ds, fold, cond, meth, seed)
            if rec is None:
                missing_keys.append((ds, fold, cond, meth, seed))
            else:
                records.append(rec)

        if missing_keys:
            raise ValueError(
                f"Cannot assemble quality artifact: {len(missing_keys)} missing checkpoints out of {len(expected_keys)}"
            )

        df = pd.DataFrame(records)
        df.sort_values(
            by=["dataset_id", "outer_fold", "condition", "method", "seed"],
            inplace=True,
        )
        atomic_write_csv(target_csv_path, df)
        return df