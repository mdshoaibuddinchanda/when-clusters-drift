"""Gustafson-Kessel (GK) fuzzy clustering implementation with adaptive covariance metrics.

References:
- Gustafson, D. E., & Kessel, W. C. (1979). Fuzzy clustering with a fuzzy covariance matrix.
- Babuska, R. (1998). Fuzzy Modeling for Control. Springer.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.cluster import kmeans_plusplus

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.diagnostics import assess_fuzzy_partition_degeneracy
from clusterdrift.methods.fuzzifier import resolve_fuzzifier
from clusterdrift.methods.utils import (
    check_simplex_constraint,
    compute_log_det_root,
    compute_pairwise_mahalanobis_sq,
    ensure_feature_array,
    regularize_covariance_matrix,
)


class GustafsonKessel(BaseClusteringMethod):
    """Gustafson-Kessel fuzzy clustering with cluster-adaptive covariance metrics and rigorous regularization."""

    def __init__(
        self,
        n_clusters: int,
        m: float | str = 2.0,
        random_state: Optional[int] = None,
        max_iter: int = 150,
        tol: float = 1e-5,
        min_eig: float = 1e-6,
        max_cond: float = 1e8,
        ridge_factor: float = 1e-5,
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
        self.min_eig: float = float(min_eig)
        self.max_cond: float = float(max_cond)
        self.ridge_factor: float = float(ridge_factor)
        self.initialization: str = initialization
        self.membership_semantics: str = "fuzzy"

        self.covariances_: Optional[List[np.ndarray]] = None
        self.metric_matrices_: Optional[List[np.ndarray]] = None
        self.condition_numbers_: List[float] = []
        self.regularization_applied_: List[bool] = []
        self.center_shift_history_: List[float] = []

    def _compute_mahalanobis_distances(
        self, X: np.ndarray, centers: np.ndarray, metric_matrices: List[np.ndarray]
    ) -> np.ndarray:
        """Compute Mahalanobis distance matrix d_ik = sqrt((x_i - v_k)^T A_k (x_i - v_k))."""
        N, d = X.shape
        K = len(centers)
        dist = np.zeros((N, K), dtype=np.float64)

        for k in range(K):
            d_sq_k = compute_pairwise_mahalanobis_sq(X, centers[k], metric_matrices[k])
            dist[:, k] = np.sqrt(np.maximum(d_sq_k, 0.0))

        return dist

    def _compute_memberships_from_distances(self, dist: np.ndarray) -> np.ndarray:
        """Compute fuzzy membership matrix U from Mahalanobis distances in the log domain."""
        N, K = dist.shape
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        power = 2.0 / (m_eff - 1.0)
        U = np.zeros((N, K), dtype=np.float64)

        zero_mask = dist == 0.0
        has_zero = np.any(zero_mask, axis=1)

        if np.any(~has_zero):
            normal_dist = dist[~has_zero]
            safe_dist = np.maximum(normal_dist, 1e-15)
            log_w = - power * np.log(safe_dist)
            M = np.max(log_w, axis=1, keepdims=True)
            w = np.exp(log_w - M)
            row_sums = np.sum(w, axis=1, keepdims=True)
            U[~has_zero] = w / np.maximum(row_sums, 1e-15)

        if np.any(has_zero):
            for i in np.where(has_zero)[0]:
                coincident_k = np.where(zero_mask[i])[0]
                c = len(coincident_k)
                U[i, coincident_k] = 1.0 / c

        return U

    def _update_centers(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Update cluster prototypes v_k = sum_i(u_ik^m * x_i) / sum_i(u_ik^m)."""
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        U_m = U ** m_eff
        fuzzy_mass = np.sum(U_m, axis=0)

        if np.any(fuzzy_mass < 1e-15):
            return np.zeros((self.n_clusters, X.shape[1])), False

        numerator = U_m.T @ X
        centers = numerator / fuzzy_mass[:, np.newaxis]
        return centers, True

    def _update_covariances_and_metrics(
        self, X: np.ndarray, centers: np.ndarray, U: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[bool], bool]:
        """Compute cluster fuzzy covariance matrices and norm-inducing metric matrices A_k."""
        N, d = X.shape
        K = self.n_clusters
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        U_m = U ** m_eff
        fuzzy_mass = np.sum(U_m, axis=0)

        covariances = []
        metrics = []
        cond_numbers = []
        reg_flags = []

        for k in range(K):
            if fuzzy_mass[k] < 1e-15:
                return [], [], [], [], False

            diff = X - centers[k]  # (n, d)
            # Weighted outer product: (diff.T * u_ik^m) @ diff / sum_i u_ik^m
            weighted_diff = diff.T * U_m[:, k]  # (d, n)
            cov_k = (weighted_diff @ diff) / fuzzy_mass[k]  # (d, d)

            # Regularize covariance to guarantee positive definiteness
            cov_reg, cond, was_reg = regularize_covariance_matrix(
                cov_k,
                min_eig=self.min_eig,
                max_cond=self.max_cond,
                ridge_factor=self.ridge_factor,
            )

            # Compute A_k = [det(cov_reg)]^(1/d) * inv(cov_reg)
            det_root = compute_log_det_root(cov_reg)
            try:
                inv_cov = np.linalg.inv(cov_reg)
                A_k = det_root * inv_cov
                # Symmetrize A_k
                A_k = 0.5 * (A_k + A_k.T)
                # Verify positive definiteness of A_k
                min_eig_A = np.linalg.eigvalsh(A_k)[0]
                if min_eig_A <= 0:
                    A_k += (abs(min_eig_A) + 1e-6) * np.eye(d)
            except np.linalg.LinAlgError:
                return [], [], [], [], False

            covariances.append(cov_reg)
            metrics.append(A_k)
            cond_numbers.append(cond)
            reg_flags.append(was_reg)

        return covariances, metrics, cond_numbers, reg_flags, True

    def _compute_objective(self, dist: np.ndarray, U: np.ndarray) -> float:
        """Compute Gustafson-Kessel objective J_GK."""
        m_eff = self.effective_m_ if getattr(self, "effective_m_", None) is not None else self.m
        U_m = U ** m_eff
        dist_sq = dist ** 2
        return float(np.sum(U_m * dist_sq))

    def fit(self, X: np.ndarray | pd.DataFrame) -> "GustafsonKessel":
        """Fit Gustafson-Kessel prototypes and adaptive covariance metrics strictly on unsupervised features X."""
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
            init_centers, _ = kmeans_plusplus(
                arr_X, n_clusters=self.n_clusters, random_state=self.random_state
            )
            centers = np.ascontiguousarray(init_centers, dtype=np.float64)
            self.initial_centers_ = centers.copy()
            # Initial memberships from Euclidean distances to initial prototypes
            diff = arr_X[:, np.newaxis, :] - centers[np.newaxis, :, :]
            init_euc_dist = np.sqrt(np.maximum(np.sum(diff ** 2, axis=-1), 0.0))
            U = self._compute_memberships_from_distances(init_euc_dist)
        elif self.initialization == "random_membership":
            rng = np.random.default_rng(self.random_state)
            U = rng.dirichlet(np.ones(self.n_clusters), size=N)
            centers, valid = self._update_centers(arr_X, U)
            if not valid:
                self.status_ = "EMPTY_CLUSTER"
                raise RuntimeError("Initial fuzzy mass collapsed into empty cluster.")
            self.initial_centers_ = centers.copy()
        else:
            raise ValueError(f"Unsupported initialization: {self.initialization}")

        # Initial covariances and metric matrices
        covs, metrics, conds, regs, valid_cov = self._update_covariances_and_metrics(arr_X, centers, U)
        if not valid_cov:
            # Fallback to identity metrics if initial covariance estimation is singular
            self.warnings_.append("Initial covariance singular; initialized with identity metrics.")
            metrics = [np.eye(d) for _ in range(self.n_clusters)]
            covs = [np.eye(d) for _ in range(self.n_clusters)]
            conds = [1.0 for _ in range(self.n_clusters)]
            regs = [True for _ in range(self.n_clusters)]

        self.objective_history_ = []
        self.center_shift_history_ = []
        self.converged_ = False

        init_dist = self._compute_mahalanobis_distances(arr_X, centers, metrics)
        init_obj = self._compute_objective(init_dist, U)
        self.objective_history_.append(init_obj)

        for iteration in range(1, self.max_iter + 1):
            # 1. Update memberships
            dist = self._compute_mahalanobis_distances(arr_X, centers, metrics)
            U = self._compute_memberships_from_distances(dist)

            # 2. Update prototypes
            new_centers, valid_c = self._update_centers(arr_X, U)
            if not valid_c:
                self.status_ = "EMPTY_CLUSTER"
                self.warnings_.append(f"Iteration {iteration}: fuzzy mass collapsed.")
                break

            # 3. Update covariances and metric matrices
            covs, metrics, conds, regs, valid_cov = self._update_covariances_and_metrics(
                arr_X, new_centers, U
            )
            if not valid_cov:
                self.status_ = "SINGULAR_METRIC"
                self.warnings_.append(f"Iteration {iteration}: covariance matrix singular after regularization.")
                break

            shift = float(np.linalg.norm(new_centers - centers, ord="fro"))
            self.center_shift_history_.append(shift)

            dist_new = self._compute_mahalanobis_distances(arr_X, new_centers, metrics)
            obj = self._compute_objective(dist_new, U)
            self.objective_history_.append(obj)

            centers = new_centers
            self.n_iter_ = iteration

            if shift < self.tol:
                self.converged_ = True
                break

        self.cluster_centers_ = centers
        self.membership_ = U
        self.covariances_ = covs
        self.metric_matrices_ = metrics
        self.condition_numbers_ = conds
        self.regularization_applied_ = regs

        valid_simplex, max_dev = check_simplex_constraint(self.membership_)
        if not valid_simplex:
            self.warnings_.append(f"GK membership simplex deviation: {max_dev}")

        if any(regs):
            self.warnings_.append(f"Regularization applied to {sum(regs)}/{self.n_clusters} clusters.")

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
        """Compute fuzzy membership for samples in X using fixed source prototypes and metric matrices."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        dist = self._compute_mahalanobis_distances(arr_X, self.cluster_centers_, self.metric_matrices_)
        return self._compute_memberships_from_distances(dist)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Assign samples to highest fuzzy membership cluster."""
        membership = self.predict_membership(X)
        return np.argmax(membership, axis=1)
