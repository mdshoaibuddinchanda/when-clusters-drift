"""Deterministic sampling and position selection algorithms for probe banks."""

from typing import Tuple
import numpy as np


def select_reference_probe_indices(
    n_source: int,
    max_size: int = 2048,
    seed: int = 0,
) -> np.ndarray:
    """Select canonical source rows for historical reference probe bank A^R.

    Parameters
    ----------
    n_source : int
        Number of available outer source rows.
    max_size : int, default 2048
        Maximum reference bank size.
    seed : int
        Deterministic seed derived from dataset, fold, and split hash.

    Returns
    -------
    canonical_indices : np.ndarray, shape (min(n_source, max_size),)
        Sorted array of 0-indexed canonical source row indices.
    """
    if n_source <= 0:
        raise ValueError(f"n_source must be positive, got {n_source}")

    if n_source <= max_size:
        return np.arange(n_source, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_source, size=max_size, replace=False)
    return np.sort(chosen).astype(np.int64)


def select_current_probe_positions(
    n_target: int,
    row_index_map: np.ndarray,
    max_size: int = 2048,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select target batch positions for current probe bank A_t^C.

    CRITICAL RULE:
    For class_prevalence shifts (which use bootstrap resampling), duplicates in the
    observed shifted target are scientifically meaningful. DO NOT deduplicate.

    Parameters
    ----------
    n_target : int
        Number of observed target points in the shifted scenario.
    row_index_map : np.ndarray
        Array mapping target positions [0..n_target-1] to underlying canonical source/target rows.
    max_size : int, default 2048
        Maximum current bank size.
    seed : int
        Deterministic seed derived from dataset, fold, condition, and shift spec hash.

    Returns
    -------
    selected_positions : np.ndarray
        Sorted indices into the current shifted target batch [0..n_target-1].
    canonical_row_indices : np.ndarray
        Underlying canonical dataset rows mapped from selected_positions. May contain duplicates.
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

    canonical_rows = np.asarray(row_index_map[positions], dtype=np.int64)
    return positions, canonical_rows
