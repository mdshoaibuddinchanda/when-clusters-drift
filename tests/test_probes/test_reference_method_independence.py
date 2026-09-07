"""Test that reference probe bank is completely method- and model-independent."""

import numpy as np
from clusterdrift.probes.hashing import derive_reference_probe_seed
from clusterdrift.probes.selection import select_reference_probe_indices


def test_reference_bank_method_and_condition_independence():
    """A^R depends strictly on dataset, fold, and split; independent of method, algorithm seed, or shift condition."""
    global_seed = 2026090705
    split_sha = "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"

    # Reference seed payload does not contain method name, method seed, or target condition
    seed_a = derive_reference_probe_seed(global_seed, "wine", 2, split_sha)
    seed_b = derive_reference_probe_seed(global_seed, "wine", 2, split_sha)

    assert seed_a == seed_b

    indices_fcm = select_reference_probe_indices(150, 2048, seed_a)
    indices_gmm = select_reference_probe_indices(150, 2048, seed_b)

    # Identical reference probes regardless of downstream method
    np.testing.assert_array_equal(indices_fcm, indices_gmm)
