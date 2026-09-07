"""Deterministic sampling and position selection algorithms for probe banks."""

from typing import Any, Optional, Tuple
import numpy as np


def select_reference_probe_positions(
    n_source: int,
    max_size: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Select 0-indexed positions within the outer-source partition [0..n_source-1]."""
    if n_source <= 0:
        raise ValueError(f"n_source must be positive, got {n_source}")

    if n_source <= max_size:
        return np.arange(n_source, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_source, size=max_size, replace=False)
    return np.sort(chosen).astype(np.int64)


def select_reference_probe_indices(
    n_source: int,
    max_size: int = 2048,
    seed: int = 0,
    phase2_source_indices: Optional[np.ndarray] = None,
) -> Any:
    """Select reference probe positions and optionally map to canonical dataset rows.

    Parameters
    ----------
    n_source : int
        Number of available outer source rows.
    max_size : int, default 2048
        Maximum reference bank size.
    seed : int
        Deterministic seed derived from dataset, fold, and split hash.
    phase2_source_indices : np.ndarray, optional
        Array of canonical dataset row IDs for the outer source partition.

    Returns
    -------
    If phase2_source_indices is provided:
        Tuple[np.ndarray, np.ndarray] : (selected_source_positions, canonical_row_indices)
    Else:
        np.ndarray : selected_source_positions
    """
    positions = select_reference_probe_positions(n_source, max_size=max_size, seed=seed)
    if phase2_source_indices is not None:
        if len(phase2_source_indices) != n_source:
            raise ValueError(
                f"phase2_source_indices length {len(phase2_source_indices)} does not match n_source {n_source}"
            )
        canonical_rows = np.asarray(phase2_source_indices[positions], dtype=np.int64)
        return positions, canonical_rows
    return positions


def select_current_probe_positions(
    n_target: int,
    row_index_map: np.ndarray,
    max_size: int = 2048,
    seed: int = 0,
    phase2_target_indices: Optional[np.ndarray] = None,
) -> Any:
    """Select target batch positions for current probe bank A_t^C.

    CRITICAL RULE:
    For class_prevalence shifts (which use bootstrap resampling), duplicates in the
    observed shifted target are scientifically meaningful. DO NOT deduplicate.

    Parameters
    ----------
    n_target : int
        Number of observed target points in the shifted scenario.
    row_index_map : np.ndarray
        Array mapping shifted positions [0..n_target-1] to original target-partition positions.
    max_size : int, default 2048
        Maximum current bank size.
    seed : int
        Deterministic seed derived from dataset, fold, condition, and shift spec hash.
    phase2_target_indices : np.ndarray, optional
        Array mapping target-partition positions to canonical dataset rows.

    Returns
    -------
    If phase2_target_indices is provided:
        Tuple[selected_current_positions, target_partition_positions, canonical_row_indices]
    Else:
        Tuple[selected_current_positions, target_partition_positions]
    """
    if n_target <= 0:
        raise ValueError(f"n_target must be positive, got {n_target}")
    if len(row_index_map) != n_target:
        raise ValueError(f"row_index_map length {len(row_index_map)} does not match n_target {n_target}")

    if n_target <= max_size:
        positions = np.arange(n_target, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(n_target, size=max_size, replace=False)).astype(np.int64)

    target_positions = np.asarray(row_index_map[positions], dtype=np.int64)
    if phase2_target_indices is not None:
        canonical_rows = np.asarray(phase2_target_indices[target_positions], dtype=np.int64)
        return positions, target_positions, canonical_rows
    return positions, target_positions
