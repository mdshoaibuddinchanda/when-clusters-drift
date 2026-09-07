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
