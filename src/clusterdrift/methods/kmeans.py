"""K-Means clustering implementation wrapping scikit-learn."""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SklearnKMeans

from clusterdrift.methods.base import BaseClusteringMethod
from clusterdrift.methods.utils import ensure_feature_array


class KMeans(BaseClusteringMethod):
    """K-Means clustering algorithm using scikit-learn with locked configuration."""

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
        )
        self.n_init: int = int(n_init)
        self.algorithm: str = algorithm
        self.membership_semantics: str = "hard_one_hot"
        self.inertia_: Optional[float] = None
        self._sklearn_model: Optional[SklearnKMeans] = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> "KMeans":
        """Fit K-Means model on unsupervised features X."""
        arr_X = ensure_feature_array(X)
        if arr_X.shape[0] < self.n_clusters:
            self.status_ = "EMPTY_CLUSTER"
            raise ValueError(
                f"n_samples ({arr_X.shape[0]}) must be >= n_clusters ({self.n_clusters})"
            )

        self._sklearn_model = SklearnKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            algorithm=self.algorithm,
        )

        self._sklearn_model.fit(arr_X)

        self.cluster_centers_ = self._sklearn_model.cluster_centers_.copy()
        self.inertia_ = float(self._sklearn_model.inertia_)
        self.n_iter_ = int(self._sklearn_model.n_iter_)
        self.converged_ = bool(self.n_iter_ < self.max_iter)
        self.objective_history_ = [self.inertia_]

        # One-hot hard membership
        labels = self._sklearn_model.labels_
        N = arr_X.shape[0]
        self.membership_ = np.zeros((N, self.n_clusters), dtype=np.float64)
        self.membership_[np.arange(N), labels] = 1.0

        self.status_ = "SUCCESS" if self.converged_ else "MAX_ITER_REACHED"
        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict closest cluster for samples in X."""
        self._check_is_fitted()
        arr_X = ensure_feature_array(X)
        return self._sklearn_model.predict(arr_X)

    def predict_membership(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Return one-hot hard membership representation for samples in X."""
        self._check_is_fitted()
        labels = self.predict(X)
        N = len(labels)
        membership = np.zeros((N, self.n_clusters), dtype=np.float64)
        membership[np.arange(N), labels] = 1.0
        return membership
