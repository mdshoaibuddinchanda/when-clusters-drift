"""Fuzzy C-Means (FCM) clustering implementation."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.utils import check_simplex_constraint, ensure_feature_array


class FCM(BaseClusteringMethod):
    """Direct implementation of standard Fuzzy C-Means clustering with rigorous numerical safeguards."""

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        random_state: Optional[int] = None,
        max_iter: int = 150,
        tol: float = 1e-5,
    ):
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        if m <= 1.0:
            raise ValueError(f"Fuzzifier m must be strictly > 1.0, got {m}")
        self.m: float = float(m)
        self.membership_semantics: str = "fuzzy"
        self.center_shift_history_: List[float] = []

    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix between data samples and prototypes.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        centers : np.ndarray, shape (K, d)

        Returns
        -------
        np.ndarray, shape (n, K)
            Distance d_ik = ||x_i - v_k||_2.
        """
        # diff: (n, K, d)
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return dist

    def _compute_memberships_from_distances(self, dist: np.ndarray) -> np.ndarray:
        """Compute fuzzy membership matrix U from distance matrix with exact zero-distance handling.

        Parameters
        ----------
        dist : np.ndarray, shape (n, K)

        Returns
        -------
        np.ndarray, shape (n, K)
        """
        N, K = dist.shape
        power = 2.0 / (self.m - 1.0)
        U = np.zeros((N, K), dtype=np.float64)

        # Identify zero distance cases where a point coincides with one or more prototypes
        zero_mask = dist == 0.0
        has_zero = np.any(zero_mask, axis=1)

        # 1. Non-coincident points (standard update equation)
        if np.any(~has_zero):
            normal_dist = dist[~has_zero]
            inv_dist = 1.0 / np.maximum(normal_dist, 1e-15)
            inv_dist_pow = inv_dist ** power  # (n_normal, K)
            row_sums = np.sum(inv_dist_pow, axis=1, keepdims=True)
            U[~has_zero] = inv_dist_pow / np.maximum(row_sums, 1e-15)

        # 2. Coincident points: assign 1/c to coincident prototypes, 0 to others
        if np.any(has_zero):
            for i in np.where(has_zero)[0]:
                coincident_k = np.where(zero_mask[i])[0]
                c = len(coincident_k)
                U[i, coincident_k] = 1.0 / c

        return U

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Update prototypes given current membership matrix U."""
        U_m = U ** self.m  # (n, K)
        fuzzy_mass = np.sum(U_m, axis=0)  # (K,)

        # Check for empty / collapsed fuzzy mass
        if np.any(fuzzy_mass < 1e-15):
            return np.zeros((self.n_clusters, X.shape[1])), False

        # v_k = sum_i(u_ik^m * x_i) / sum_i(u_ik^m)
        numerator = U_m.T @ X  # (K, d)
        centers = numerator / fuzzy_mass[:, np.newaxis]
        return centers, True

    def _compute_objective(self, X: np.ndarray, centers: np.ndarray, U: np.ndarray) -> float:
        """Compute FCM objective function J_m(U, V)."""
        dist = self._compute_distances(X, centers)
        dist_sq = dist ** 2
        U_m = U ** self.m
        return float(np.sum(U_m * dist_sq))

    def fit(self, X: np.ndarray | pd.DataFrame) -> "FCM":
        """Fit FCM prototypes and memberships strictly on unsupervised features X."""
        arr_X = ensure_feature_array(X)
        N, d = arr_X.shape

        if N < self.n_clusters:
            self.status_ = "EMPTY_CLUSTER"
            raise ValueError(f"n_samples ({N}) must be >= n_clusters ({self.n_clusters})")

        # Deterministic initialization of memberships or prototypes from random_state
        rng = np.random.default_rng(self.random_state)
        # Initialize memberships using Dirichlet distribution to guarantee simplex
        U = rng.dirichlet(np.ones(self.n_clusters), size=N)

        # Compute initial centers from initial U
        centers, valid = self._update_centers(arr_X, U)
        if not valid:
            self.status_ = "EMPTY_CLUSTER"
            raise RuntimeError("Initial fuzzy mass collapsed into empty cluster.")

        self.objective_history_ = []
        self.center_shift_history_ = []
        self.converged_ = False

        initial_obj = self._compute_objective(arr_X, centers, U)
        self.objective_history_.append(initial_obj)

        for iteration in range(1, self.max_iter + 1):
            # Update memberships
            dist = self._compute_distances(arr_X, centers)
            U = self._compute_memberships_from_distances(dist)

            # Update prototypes
            new_centers, valid = self._update_centers(arr_X, U)
            if not valid:
                self.status_ = "EMPTY_CLUSTER"
                self.warnings_.append(f"Iteration {iteration}: fuzzy mass collapsed.")
                break

            shift = float(np.linalg.norm(new_centers - centers, ord="fro"))
            self.center_shift_history_.append(shift)

            obj = self._compute_objective(arr_X, new_centers, U)
            self.objective_history_.append(obj)

            centers = new_centers
            self.n_iter_ = iteration

            if shift < self.tol:
                self.converged_ = True
                break

        self.cluster_centers_ = centers
        self.membership_ = U

        # Verify simplex constraint on final memberships
        valid_simplex, max_dev = check_simplex_constraint(self.membership_)
        if not valid_simplex:
            self.warnings_.append(f"Final membership simplex deviation: {max_dev}")

        if self.converged_:
            self.status_ = "SUCCESS"
        elif self.status_ == "INVALID_INPUT":
            self.status_ = "MAX_ITER_REACHED"

        return self

    def predict_membership(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute fuzzy membership for samples in X using fixed source prototypes.

        Guaranteed not to modify cluster_centers_.
        """
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        dist = self._compute_distances(arr_X, self.cluster_centers_)
        U = self._compute_memberships_from_distances(dist)
        return U

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict hard cluster assignments by taking argmax of soft memberships."""
        membership = self.predict_membership(X)
        return np.argmax(membership, axis=1)
