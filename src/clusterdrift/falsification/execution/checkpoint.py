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
    """Manages 4,500 fine-grained atomic evaluation-only quality checkpoints."""

    def __init__(self, work_dir: Path, project_root: Optional[Path] = None):
        self.project_root = Path(project_root).resolve() if project_root else None
        self.work_dir = assert_not_frozen_data_path(work_dir, self.project_root)
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
    ) -> bool:
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        if not p.exists() or p.stat().st_size == 0:
            return False

        try:
            with open(p, "r", encoding="utf-8") as f:
                rec = json.load(f)

            if (
                rec.get("dataset_id") != dataset_id
                or rec.get("outer_fold") != outer_fold
                or rec.get("condition") != condition
                or rec.get("method") != method
                or rec.get("seed") != seed
            ):
                return False

            stored_sha = rec.get("quality_record_sha256")
            recomputed_sha = compute_quality_record_sha256(rec)
            if stored_sha != recomputed_sha:
                q_path = self.quarantine_dir / f"corrupt_{p.name}"
                shutil.move(p, q_path)
                return False

            return True
        except Exception:
            return False

    def save_checkpoint(self, rec: Dict[str, Any]) -> Path:
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
        if not self.is_checkpoint_valid(dataset_id, outer_fold, condition, method, seed):
            return None
        p = self._get_checkpoint_path(dataset_id, outer_fold, condition, seed)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_completed_checkpoint_keys(self) -> Set[Tuple[str, int, str, str, int]]:
        valid_keys = set()
        for p in self.checkpoints_dir.glob("*/*/*/*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                if compute_quality_record_sha256(rec) == rec.get("quality_record_sha256"):
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