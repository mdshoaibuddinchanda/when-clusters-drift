"""Base interface for unsupervised clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np
import pandas as pd


class BaseClusteringMethod(ABC):
    """Abstract base class for all unsupervised clustering models in clusterdrift.

    Enforces strict unsupervised fitting: fit(X) cannot accept labels y.
    Provides standard attributes:
    - cluster_centers_: np.ndarray of shape (n_clusters, n_features)
    - n_clusters: int
    - converged_: bool
    - n_iter_: int
    - objective_history_: List[float]
    - random_state: Optional[int]
    - membership_: np.ndarray of shape (n_samples, n_clusters)
    - membership_semantics: str ('fuzzy', 'probabilistic', 'hard_one_hot')
    - status_: str ('SUCCESS', 'MAX_ITER_REACHED', 'NUMERICAL_FAILURE', 'SINGULAR_METRIC', 'EMPTY_CLUSTER', 'INVALID_INPUT', 'DEGENERATE_SOLUTION')
    - warnings_: List[str]
    - degenerate_solution_: bool
    - diagnostics_: Dict[str, Any]
    - initialization_method_: str
    - initial_centers_: Optional[np.ndarray]
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: Optional[int] = None,
        max_iter: int = 300,
        tol: float = 1e-4,
    ):
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        self.n_clusters: int = int(n_clusters)
        self.random_state: Optional[int] = random_state
        self.max_iter: int = int(max_iter)
        self.tol: float = float(tol)

        # Fitted attributes
        self.cluster_centers_: Optional[np.ndarray] = None
        self.converged_: bool = False
        self.n_iter_: int = 0
        self.objective_history_: List[float] = []
        self.membership_: Optional[np.ndarray] = None
        self.membership_semantics: str = "fuzzy"
        self.status_: str = "INVALID_INPUT"
        self.warnings_: List[str] = []
        self.degenerate_solution_: bool = False
        self.diagnostics_: dict = {}
        self.initialization_method_: str = ""
        self.initial_centers_: Optional[np.ndarray] = None
        self.fuzzifier_policy_: str = ""
        self.effective_m_: Optional[float] = None
        self.effective_dimension_: int = 0
        self.fuzzifier_clipped_: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray | pd.DataFrame) -> "BaseClusteringMethod":
        """Fit clustering model strictly on unsupervised features X.

        Labels y are strictly forbidden in unsupervised fit.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Assign data samples in X to nearest cluster index in [0, K-1].

        Returns
        -------
        np.ndarray
            1D array of cluster indices of shape (n_samples,).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_membership(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Compute soft membership or posterior probability distribution for samples in X.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, n_clusters).
        """
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        """Verify that model has been fitted before inference."""
        if self.cluster_centers_ is None:
            raise RuntimeError("This clustering model instance is not fitted yet. Call 'fit' first.")
