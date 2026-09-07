"""Thread-safe in-memory caching for preprocessors, probe matrices, and memberships."""

from threading import Lock
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ProbeCache:
    """Thread-safe multi-tier in-memory cache for Phase 5 representation pipeline."""

    def __init__(self):
        self._lock = Lock()
        self._preprocessors: Dict[Tuple[str, int, str], Any] = {}
        self._transformed_banks: Dict[str, np.ndarray] = {}
        self._memberships: Dict[Tuple[str, str], np.ndarray] = {}

        self.hits = {
            "preprocessor": 0,
            "transformed_bank": 0,
            "membership": 0,
        }
        self.misses = {
            "preprocessor": 0,
            "transformed_bank": 0,
            "membership": 0,
        }

    def get_preprocessor(self, dataset_id: str, outer_fold: int, prep_config_sha: str) -> Optional[Any]:
        key = (dataset_id, outer_fold, prep_config_sha)
        with self._lock:
            if key in self._preprocessors:
                self.hits["preprocessor"] += 1
                return self._preprocessors[key]
            self.misses["preprocessor"] += 1
            return None

    def put_preprocessor(self, dataset_id: str, outer_fold: int, prep_config_sha: str, prep: Any) -> None:
        key = (dataset_id, outer_fold, prep_config_sha)
        with self._lock:
            self._preprocessors[key] = prep

    def get_transformed_bank(self, bank_sha256: str) -> Optional[np.ndarray]:
        with self._lock:
            if bank_sha256 in self._transformed_banks:
                self.hits["transformed_bank"] += 1
                return self._transformed_banks[bank_sha256].copy()
            self.misses["transformed_bank"] += 1
            return None

    def put_transformed_bank(self, bank_sha256: str, matrix: np.ndarray) -> None:
        with self._lock:
            self._transformed_banks[bank_sha256] = matrix.copy()

    def get_membership(self, model_fingerprint: str, bank_sha256: str) -> Optional[np.ndarray]:
        key = (model_fingerprint, bank_sha256)
        with self._lock:
            if key in self._memberships:
                self.hits["membership"] += 1
                return self._memberships[key].copy()
            self.misses["membership"] += 1
            return None

    def put_membership(self, model_fingerprint: str, bank_sha256: str, U: np.ndarray) -> None:
        key = (model_fingerprint, bank_sha256)
        with self._lock:
            self._memberships[key] = U.copy()

    def clear(self) -> None:
        with self._lock:
            self._preprocessors.clear()
            self._transformed_banks.clear()
            self._memberships.clear()
            for k in self.hits:
                self.hits[k] = 0
            for k in self.misses:
                self.misses[k] = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": dict(self.hits),
                "misses": dict(self.misses),
                "cached_preprocessors": len(self._preprocessors),
                "cached_transformed_banks": len(self._transformed_banks),
                "cached_memberships": len(self._memberships),
            }
