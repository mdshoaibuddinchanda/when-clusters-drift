"""Test ambiguity detection for low-margin assignment pairs."""

import numpy as np
import pytest

from clusterdrift.alignment.hungarian import align_clusters


def test_ambiguity_detection_on_identical_prototypes():
    """Identical or nearly identical candidate clusters must trigger ambiguous=True."""
    # Reference has two separated clusters
    centers_ref = np.array([
        [-5.0, 0.0],
        [5.0, 0.0],
    ])
    scales_ref = np.array([1.0, 1.0])
    U_ref = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    # Candidate has twin clusters at the midpoint (equidistant from both reference clusters)
    centers_cand = np.array([
        [0.0, 0.0],
        [0.0, 1e-10],  # Difference < 1e-8 tolerance
    ])
    U_cand = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    res = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        cost_margin_tolerance=1e-8,
    )

    assert res.ambiguous is True
    assert res.global_assignment_margin <= 1e-8
    assert res.best_assignment_cost >= 0.0
    assert res.second_best_assignment_cost >= res.best_assignment_cost


def test_global_ambiguity_exact_tie():
    """Verify that an exact tie between two complete assignments is flagged ambiguous=True."""
    # Symmetrical configuration creating identical costs for permutations [0, 1] and [1, 0]
    centers_ref = np.array([
        [-1.0, 0.0],
        [1.0, 0.0],
    ])
    scales_ref = np.array([1.0, 1.0])
    U_ref = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    # Candidate equidistant
    centers_cand = np.array([
        [0.0, -1.0],
        [0.0, 1.0],
    ])
    U_cand = np.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    res = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        cost_margin_tolerance=1e-8,
    )

    assert res.ambiguous is True
    assert res.global_assignment_margin <= 1e-8
    assert res.second_best_assignment_cost == pytest.approx(res.best_assignment_cost, abs=1e-8)


def test_unambiguous_case():
    """Distinct, well-separated clusters must have ambiguous=False."""
    centers_ref = np.array([
        [-10.0, 0.0],
        [10.0, 0.0],
    ])
    scales_ref = np.array([1.0, 1.0])
    U_ref = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    centers_cand = np.array([
        [-9.9, 0.0],
        [10.1, 0.0],
    ])
    U_cand = np.array([
        [0.99, 0.01],
        [0.01, 0.99],
    ])

    res = align_clusters(
        centers_ref=centers_ref,
        centers_cand=centers_cand,
        scales_ref=scales_ref,
        U_ref=U_ref,
        U_cand=U_cand,
        cost_margin_tolerance=1e-8,
    )

    assert res.ambiguous is False
    assert res.minimum_assignment_margin > 1.0
