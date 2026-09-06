"""Gaussian Mixture Model (GMM) clustering implementation wrapping scikit-learn."""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture as SklearnGMM

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.utils import check_simplex_constraint, ensure_feature_array


class GMM(BaseClusteringMethod):
    """Gaussian Mixture Model using scikit-learn with locked configuration."""

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        covariance_type: str = "full",
        n_init: int = 1,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        tol: float = 1e-3,
    ):
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        self.covariance_type: str = covariance_type
        self.n_init: int = int(n_init)
        self.reg_covar: float = float(reg_covar)
        self.membership_semantics: str = "probabilistic"

        self.means_: Optional[np.ndarray] = None
        self.weights_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.lower_bound_: Optional[float] = None
        self._sklearn_model: Optional[SklearnGMM] = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> "GMM":
        """Fit Gaussian Mixture Model on unsupervised features X."""
        arr_X = ensure_feature_array(X)
        N, d = arr_X.shape

        if N < self.n_clusters:
            self.status_ = "EMPTY_CLUSTER"
            raise ValueError(f"n_samples ({N}) must be >= n_clusters ({self.n_clusters})")

        self._sklearn_model = SklearnGMM(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=self.n_init,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        try:
            self._sklearn_model.fit(arr_X)
        except Exception as exc:
            self.status_ = "NUMERICAL_FAILURE"
            self.warnings_.append(f"GMM fit failure: {str(exc)}")
            raise

        self.means_ = self._sklearn_model.means_.copy()
        self.cluster_centers_ = self.means_
        self.weights_ = self._sklearn_model.weights_.copy()
        self.covariances_ = self._sklearn_model.covariances_.copy()
        self.converged_ = bool(self._sklearn_model.converged_)
        self.n_iter_ = int(self._sklearn_model.n_iter_)
        self.lower_bound_ = float(self._sklearn_model.lower_bound_)
        self.objective_history_ = [self.lower_bound_]

        self.membership_ = self._sklearn_model.predict_proba(arr_X)
        valid_simplex, max_dev = check_simplex_constraint(self.membership_)
        if not valid_simplex:
            self.warnings_.append(f"GMM membership simplex deviation: {max_dev}")

        self.status_ = "SUCCESS" if self.converged_ else "MAX_ITER_REACHED"
        return self

    def predict_membership(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute posterior probabilities for samples in X."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        proba = self._sklearn_model.predict_proba(arr_X)
        return proba

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict highest-probability cluster components for samples in X."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        return self._sklearn_model.predict(arr_X)
