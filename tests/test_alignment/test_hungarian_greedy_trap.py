"""Test that Hungarian assignment escapes greedy nearest-matching suboptimality traps."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def test_hungarian_greedy_trap_lower_global_cost():
    """Verify that Hungarian bipartite matching returns lower total cost than greedy matching.

    Classic 2x2 cost matrix:
        C = [[1.0, 2.0],
             [2.0, 10.0]]

    Greedy algorithm:
        Row 0 chooses min cost column: Col 0 (cost = 1.0).
        Row 1 is forced to take remaining Col 1 (cost = 10.0).
        Greedy total cost = 1.0 + 10.0 = 11.0.

    Hungarian algorithm:
        Assignment: (Row 0 -> Col 1) and (Row 1 -> Col 0).
        Cost = 2.0 + 2.0 = 4.0.
        4.0 < 11.0.
    """
    C = np.array([
        [1.0, 2.0],
        [2.0, 10.0],
    ])

    # Greedy simulation
    # Row 0 picks min
    greedy_col_0 = int(np.argmin(C[0]))
    greedy_col_1 = 1 - greedy_col_0
    greedy_cost = C[0, greedy_col_0] + C[1, greedy_col_1]

    assert greedy_col_0 == 0
    assert greedy_col_1 == 1
    assert greedy_cost == 11.0

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(C)
    hungarian_cost = np.sum(C[row_ind, col_ind])

    assert hungarian_cost == 4.0
    assert hungarian_cost < greedy_cost
    np.testing.assert_array_equal(col_ind, [1, 0])


def test_hungarian_greedy_trap_not_falsely_ambiguous(monkeypatch):
    """Verify that C = [[1, 2], [2, 10]] with optimal cost 4.0 is NOT marked ambiguous.

    Row 0 has assigned cost 2.0 (while row minimum is 1.0).
    Under the old row-local margin, this resulted in a negative margin (-1.0) and false ambiguity.
    Under the global assignment margin:
        J* = 4.0
        Forbidding (0, 1) -> alternative assignment (0, 0) + (1, 1) has cost 11.0.
        Forbidding (1, 0) -> alternative assignment (0, 0) + (1, 1) has cost 11.0.
        J^(2) = 11.0
        global_assignment_margin = 11.0 - 4.0 = 7.0 >= 0.
        ambiguous = False.
    """
    C = np.array([
        [1.0, 2.0],
        [2.0, 10.0],
    ])
    from clusterdrift.alignment import hungarian
    monkeypatch.setattr(
        hungarian,
        "compute_combined_alignment_cost",
        lambda **kwargs: (C, C, np.eye(2)),
    )
    res = hungarian.align_clusters(
        centers_ref=np.zeros((2, 2)),
        centers_cand=np.zeros((2, 2)),
        scales_ref=np.ones(2),
        U_ref=np.eye(2),
        U_cand=np.eye(2),
        cost_margin_tolerance=1e-8,
    )
    assert res.assignment_cost == 4.0
    assert res.best_assignment_cost == 4.0
    assert res.second_best_assignment_cost == 11.0
    assert res.global_assignment_margin == 7.0
    assert res.ambiguous is False
    assert res.forbidden_edge_producing_second_best in [(0, 1), (1, 0)]
