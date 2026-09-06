"""Fuzzy C-Means (FCM) clustering implementation."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import kmeans_plusplus

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.diagnostics import assess_fuzzy_partition_degeneracy
from clusterdrift.methods.fuzzifier import resolve_fuzzifier
from clusterdrift.methods.utils import check_simplex_constraint, ensure_feature_array


class FCM(BaseClusteringMethod):
    """Direct implementation of standard Fuzzy C-Means clustering with rigorous numerical safeguards."""

    def __init__(
        self,
        n_clusters: int,
        m: float | str = 2.0,
        random_state: Optional[int] = None,
        max_iter: int = 150,
        tol: float = 1e-5,
        initialization: str = "kmeans++",
        fuzzifier_policy: str = "fixed",
    ):
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        if isinstance(m, str) and m in ("dimension_adaptive", "winkler_dimension_rule"):
            fuzzifier_policy = m
            m = 2.0

        if fuzzifier_policy == "fixed":
            if float(m) <= 1.0:
                raise ValueError(f"Fuzzifier m must be strictly > 1.0, got {m}")

        if initialization not in ("kmeans++", "random_membership"):
            raise ValueError(
                f"Unknown initialization method '{initialization}'. Must be 'kmeans++' or 'random_membership'."
            )
        self.m: float = float(m)
        self.fuzzifier_policy: str = fuzzifier_policy
        self.initialization: str = initialization
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
        """Compute fuzzy membership matrix U from distance matrix in the log domain.

        Implements numerically stable log-domain Softmax to prevent floating-point
        overflow/underflow as m -> 1.01, preserving exact zero-distance handling.

        Parameters
        ----------
        dist : np.ndarray, shape (n, K)

        Returns
        -------
        np.ndarray, shape (n, K)
        """
        N, K = dist.shape
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        power = 2.0 / (m_eff - 1.0)
        U = np.zeros((N, K), dtype=np.float64)

        # Identify zero distance cases where a point coincides with one or more prototypes
        zero_mask = dist == 0.0
        has_zero = np.any(zero_mask, axis=1)

        # 1. Non-coincident points (numerically stable log-domain Softmax)
        if np.any(~has_zero):
            normal_dist = dist[~has_zero]
            safe_dist = np.maximum(normal_dist, 1e-15)
            log_w = - power * np.log(safe_dist)  # (n_normal, K)
            M = np.max(log_w, axis=1, keepdims=True)
            w = np.exp(log_w - M)
            row_sums = np.sum(w, axis=1, keepdims=True)
            U[~has_zero] = w / np.maximum(row_sums, 1e-15)

        # 2. Coincident points: assign 1/c to coincident prototypes, 0 to others
        if np.any(has_zero):
            for i in np.where(has_zero)[0]:
                coincident_k = np.where(zero_mask[i])[0]
                c = len(coincident_k)
                U[i, coincident_k] = 1.0 / c

        return U

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Update prototypes given current membership matrix U."""
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        U_m = U ** m_eff  # (n, K)
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
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        U_m = U ** m_eff
        return float(np.sum(U_m * dist_sq))

    def fit(self, X: np.ndarray | pd.DataFrame) -> "FCM":
        """Fit FCM prototypes and memberships strictly on unsupervised features X."""
        arr_X = ensure_feature_array(X)
        N, d = arr_X.shape

        if N < self.n_clusters:
            self.status_ = "EMPTY_CLUSTER"
            raise ValueError(f"n_samples ({N}) must be >= n_clusters ({self.n_clusters})")

        # Resolve fuzzifier policy strictly from unsupervised feature dimensionality d
        res = resolve_fuzzifier(
            policy=self.fuzzifier_policy,
            value=self.m,
            n_features=d,
        )
        self.fuzzifier_policy_ = res.policy
        self.effective_m_ = res.effective_m
        self.effective_dimension_ = res.dimension
        self.fuzzifier_clipped_ = res.clipped

        self.initialization_method_ = self.initialization

        if self.initialization == "kmeans++":
            # Deterministic prototype seeding using kmeans++
            init_centers, _ = kmeans_plusplus(
                arr_X, n_clusters=self.n_clusters, random_state=self.random_state
            )
            centers = np.ascontiguousarray(init_centers, dtype=np.float64)
            self.initial_centers_ = centers.copy()
            # Compute initial memberships from initial prototypes
            dist = self._compute_distances(arr_X, centers)
            U = self._compute_memberships_from_distances(dist)
        elif self.initialization == "random_membership":
            # Historical random Dirichlet membership initialization (preserved for tests)
            rng = np.random.default_rng(self.random_state)
            U = rng.dirichlet(np.ones(self.n_clusters), size=N)
            centers, valid = self._update_centers(arr_X, U)
            if not valid:
                self.status_ = "EMPTY_CLUSTER"
                raise RuntimeError("Initial fuzzy mass collapsed into empty cluster.")
            self.initial_centers_ = centers.copy()
        else:
            raise ValueError(f"Unsupported initialization: {self.initialization}")

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

        # Assess fuzzy solution degeneracy
        self.diagnostics_ = assess_fuzzy_partition_degeneracy(
            arr_X, self.cluster_centers_, self.membership_, self.n_clusters
        )
        self.degenerate_solution_ = self.diagnostics_["is_degenerate"]

        if self.degenerate_solution_:
            self.status_ = "DEGENERATE_SOLUTION"
            self.warnings_.append(
                "Degenerate fuzzy solution detected: near-uniform memberships and collapsed prototypes."
            )
        elif self.converged_:
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
