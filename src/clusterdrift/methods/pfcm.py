"""Possibilistic Fuzzy C-Means (PFCM) clustering implementation.

Reference:
Pal, N. R., Pal, K., Keller, J. M., & Bezdek, J. C. (2005).
"A possibilistic fuzzy c-means clustering algorithm."
IEEE Transactions on Fuzzy Systems, 13(4), 517-530.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.utils import check_simplex_constraint, ensure_feature_array


class PFCM(BaseClusteringMethod):
    """Possibilistic Fuzzy C-Means (Pal et al., 2005) clustering implementation."""

    def __init__(
        self,
        n_clusters: int,
        a: float = 1.0,
        b: float = 1.0,
        m: float = 2.0,
        eta: float = 2.0,
        k_scale: float = 1.0,
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
        if a <= 0.0 or b <= 0.0:
            raise ValueError(f"Parameters a and b must be strictly positive, got a={a}, b={b}")
        if m <= 1.0 or eta <= 1.0:
            raise ValueError(f"Exponents m and eta must be strictly > 1.0, got m={m}, eta={eta}")

        self.a: float = float(a)
        self.b: float = float(b)
        self.m: float = float(m)
        self.eta: float = float(eta)
        self.k_scale: float = float(k_scale)
        self.membership_semantics: str = "fuzzy"

        self.typicality_: Optional[np.ndarray] = None
        self.gamma_: Optional[np.ndarray] = None
        self.center_shift_history_: List[float] = []

    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix d_ik = ||x_i - v_k||_2."""
        diff = X[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    def _compute_fuzzy_memberships(self, dist: np.ndarray) -> np.ndarray:
        """Compute fuzzy membership matrix U satisfying row simplex constraint."""
        N, K = dist.shape
        power = 2.0 / (self.m - 1.0)
        U = np.zeros((N, K), dtype=np.float64)

        zero_mask = dist == 0.0
        has_zero = np.any(zero_mask, axis=1)

        if np.any(~has_zero):
            normal_dist = dist[~has_zero]
            inv_dist = 1.0 / np.maximum(normal_dist, 1e-15)
            inv_dist_pow = inv_dist ** power
            row_sums = np.sum(inv_dist_pow, axis=1, keepdims=True)
            U[~has_zero] = inv_dist_pow / np.maximum(row_sums, 1e-15)

        if np.any(has_zero):
            for i in np.where(has_zero)[0]:
                coincident_k = np.where(zero_mask[i])[0]
                c = len(coincident_k)
                U[i, coincident_k] = 1.0 / c

        return U

    def _compute_typicalities(self, dist: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """Compute possibilistic typicality matrix T with values in [0, 1].

        Crucial: typicality has NO row-sum constraint across clusters.
        """
        N, K = dist.shape
        dist_sq = dist ** 2
        power = 1.0 / (self.eta - 1.0)

        # t_ik = [1 + (b / gamma_k * dist_sq_ik)^(1 / (eta - 1))]^(-1)
        safe_gamma = np.maximum(gamma, 1e-12)
        ratio = (self.b / safe_gamma[np.newaxis, :]) * dist_sq
        T = 1.0 / (1.0 + np.maximum(ratio, 0.0) ** power)

        # For coincident points (dist == 0), t_ik = 1.0
        T = np.clip(T, 0.0, 1.0)
        return T

    def _estimate_gamma(self, X: np.ndarray, centers: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Estimate scale parameter gamma_k from initial partition following Pal et al. (2005)."""
        dist = self._compute_distances(X, centers)
        dist_sq = dist ** 2
        U_m = U ** self.m
        fuzzy_mass = np.sum(U_m, axis=0)  # (K,)
        safe_mass = np.maximum(fuzzy_mass, 1e-12)
        gamma = self.k_scale * np.sum(U_m * dist_sq, axis=0) / safe_mass
        return np.maximum(gamma, 1e-6)

    def _update_centers(
        self, X: np.ndarray, U: np.ndarray, T: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """Update prototypes v_k = sum_i(w_ik * x_i) / sum_i(w_ik), where w_ik = a*u_ik^m + b*t_ik^eta."""
        weight = self.a * (U ** self.m) + self.b * (T ** self.eta)  # (n, K)
        col_mass = np.sum(weight, axis=0)  # (K,)

        if np.any(col_mass < 1e-15):
            return np.zeros((self.n_clusters, X.shape[1])), False

        numerator = weight.T @ X  # (K, d)
        centers = numerator / col_mass[:, np.newaxis]
        return centers, True

    def _compute_objective(
        self, X: np.ndarray, centers: np.ndarray, U: np.ndarray, T: np.ndarray, gamma: np.ndarray
    ) -> float:
        """Compute PFCM objective J_PFCM."""
        dist = self._compute_distances(X, centers)
        dist_sq = dist ** 2
        comp1 = np.sum((self.a * (U ** self.m) + self.b * (T ** self.eta)) * dist_sq)
        comp2 = np.sum(gamma * np.sum((1.0 - T) ** self.eta, axis=0))
        return float(comp1 + comp2)

    def fit(self, X: np.ndarray | pd.DataFrame) -> "PFCM":
        """Fit PFCM prototypes, memberships, and typicalities strictly on unsupervised features X."""
        arr_X = ensure_feature_array(X)
        N, d = arr_X.shape

        if N < self.n_clusters:
            self.status_ = "EMPTY_CLUSTER"
            raise ValueError(f"n_samples ({N}) must be >= n_clusters ({self.n_clusters})")

        rng = np.random.default_rng(self.random_state)
        # Initial memberships via Dirichlet distribution
        U = rng.dirichlet(np.ones(self.n_clusters), size=N)

        # Initial prototypes from U
        U_m = U ** self.m
        mass = np.maximum(np.sum(U_m, axis=0), 1e-12)
        centers = (U_m.T @ arr_X) / mass[:, np.newaxis]

        # Initial gamma estimation
        self.gamma_ = self._estimate_gamma(arr_X, centers, U)
        dist = self._compute_distances(arr_X, centers)
        T = self._compute_typicalities(dist, self.gamma_)

        self.objective_history_ = []
        self.center_shift_history_ = []
        self.converged_ = False

        init_obj = self._compute_objective(arr_X, centers, U, T, self.gamma_)
        self.objective_history_.append(init_obj)

        for iteration in range(1, self.max_iter + 1):
            # Update U and T
            dist = self._compute_distances(arr_X, centers)
            U = self._compute_fuzzy_memberships(dist)
            T = self._compute_typicalities(dist, self.gamma_)

            # Update prototypes
            new_centers, valid = self._update_centers(arr_X, U, T)
            if not valid:
                self.status_ = "EMPTY_CLUSTER"
                self.warnings_.append(f"Iteration {iteration}: fuzzy-possibilistic mass collapsed.")
                break

            shift = float(np.linalg.norm(new_centers - centers, ord="fro"))
            self.center_shift_history_.append(shift)

            obj = self._compute_objective(arr_X, new_centers, U, T, self.gamma_)
            self.objective_history_.append(obj)

            centers = new_centers
            self.n_iter_ = iteration

            if shift < self.tol:
                self.converged_ = True
                break

        self.cluster_centers_ = centers
        self.membership_ = U
        self.typicality_ = T

        valid_simplex, max_dev = check_simplex_constraint(self.membership_)
        if not valid_simplex:
            self.warnings_.append(f"PFCM membership simplex deviation: {max_dev}")

        if self.converged_:
            self.status_ = "SUCCESS"
        elif self.status_ == "INVALID_INPUT":
            self.status_ = "MAX_ITER_REACHED"

        return self

    def predict_membership(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute fuzzy membership for samples in X using fixed source prototypes."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        dist = self._compute_distances(arr_X, self.cluster_centers_)
        return self._compute_fuzzy_memberships(dist)

    def predict_typicality(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute possibilistic typicality for samples in X using fixed source prototypes and gamma."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        dist = self._compute_distances(arr_X, self.cluster_centers_)
        return self._compute_typicalities(dist, self.gamma_)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Assign samples to highest fuzzy membership cluster."""
        membership = self.predict_membership(X)
        return np.argmax(membership, axis=1)
