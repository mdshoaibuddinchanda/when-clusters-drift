"""Source-only preprocessing module guaranteeing zero label or target leakage."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class SourceOnlyPreprocessor:
    """Preprocessor fitted strictly on source feature data with zero target or label access."""

    def __init__(
        self,
        feature_roles: Dict[str, str],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.feature_roles = feature_roles
        self.config = config.get("preprocessing", config)
        self.metadata = metadata or {}

        # Feature partitioning
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.ordinal_cols: List[str] = []
        self.boolean_cols: List[str] = []

        category_orderings = self.metadata.get("category_orderings") or {}
        ordinal_policy = self.config.get("ordinal", {}).get("policy", "onehot_unless_explicit_order")

        for col, role in self.feature_roles.items():
            if role == "numeric":
                self.numeric_cols.append(col)
            elif role == "categorical":
                self.categorical_cols.append(col)
            elif role == "ordinal":
                if ordinal_policy == "onehot_unless_explicit_order" and col not in category_orderings:
                    # Policy dictating one-hot encoding when explicit verified category order is absent
                    self.categorical_cols.append(col)
                else:
                    self.ordinal_cols.append(col)
            elif role == "boolean":
                self.boolean_cols.append(col)
            else:
                raise ValueError(f"Unknown feature role '{role}' for feature '{col}'")

        # Fitted transformers
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.numeric_scaler: Optional[StandardScaler] = None

        self.categorical_imputer: Optional[SimpleImputer] = None
        self.categorical_encoder: Optional[OneHotEncoder] = None

        self.ordinal_imputer: Optional[SimpleImputer] = None
        self.ordinal_encoders: Dict[str, OrdinalEncoder] = {}
        self.ordinal_scaler: Optional[StandardScaler] = None

        self.boolean_imputer: Optional[SimpleImputer] = None

        self.is_fitted: bool = False
        self.output_feature_names: List[str] = []
        self.source_category_vocabularies_: Dict[str, set] = {}

    def fit(self, X: pd.DataFrame) -> "SourceOnlyPreprocessor":
        """Fit preprocessing transformations strictly on source features without target or label access."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        # 1. Numeric features
        if self.numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy="median")
            X_num_imp = self.numeric_imputer.fit_transform(X[self.numeric_cols])

            # Safe handling for entirely-empty features in source: fill nan statistics with 0.0
            if np.isnan(self.numeric_imputer.statistics_).any():
                self.numeric_imputer.statistics_ = np.nan_to_num(self.numeric_imputer.statistics_, nan=0.0)
                X_num_imp = np.nan_to_num(X_num_imp, nan=0.0)

            self.numeric_scaler = StandardScaler()
            self.numeric_scaler.fit(X_num_imp)

        # 2. Categorical features (including ordinals without explicit order)
        if self.categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            X_cat_df = X[self.categorical_cols].astype(str)
            X_cat_imp = self.categorical_imputer.fit_transform(X_cat_df)

            # Record source category vocabulary for unseen category auditing
            self.source_category_vocabularies_ = {
                col: set(X_cat_df[col].dropna().unique()) for col in self.categorical_cols
            }

            min_freq = self.config.get("categorical", {}).get("min_frequency", 0.01)
            self.categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_freq,
                sparse_output=False,
                dtype=np.float32,
            )
            self.categorical_encoder.fit(X_cat_imp)

        # 3. Explicitly ordered ordinal features
        if self.ordinal_cols:
            category_orderings = self.metadata.get("category_orderings") or {}
            self.ordinal_imputer = SimpleImputer(strategy="most_frequent")
            X_ord_df = X[self.ordinal_cols].astype(str)
            X_ord_imp = self.ordinal_imputer.fit_transform(X_ord_df)

            ord_encoded = []
            for idx, col in enumerate(self.ordinal_cols):
                order = category_orderings[col]
                encoder = OrdinalEncoder(
                    categories=[order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                col_enc = encoder.fit_transform(X_ord_imp[:, [idx]])
                self.ordinal_encoders[col] = encoder
                ord_encoded.append(col_enc)

            X_ord_mat = np.hstack(ord_encoded)
            self.ordinal_scaler = StandardScaler()
            self.ordinal_scaler.fit(X_ord_mat)

        # 4. Boolean features
        if self.boolean_cols:
            self.boolean_imputer = SimpleImputer(strategy="most_frequent")
            X_bool_df = X[self.boolean_cols].copy()
            self.boolean_imputer.fit(X_bool_df)

        self.is_fitted = True
        self._build_feature_names_out()
        return self

    def _build_feature_names_out(self) -> None:
        """Construct deterministic output feature names."""
        names: List[str] = []
        if self.numeric_cols:
            names.extend(self.numeric_cols)
        if self.categorical_cols and self.categorical_encoder is not None:
            names.extend(self.categorical_encoder.get_feature_names_out(self.categorical_cols).tolist())
        if self.ordinal_cols:
            names.extend(self.ordinal_cols)
        if self.boolean_cols:
            names.extend(self.boolean_cols)
        self.output_feature_names = names

    def get_feature_names_out(self) -> List[str]:
        """Return the exact names of transformed output features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling get_feature_names_out().")
        return list(self.output_feature_names)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input features using source-fitted parameters."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        parts: List[np.ndarray] = []

        # 1. Numeric transform
        if self.numeric_cols:
            X_num = X[self.numeric_cols]
            X_num_imp = self.numeric_imputer.transform(X_num)
            X_num_imp = np.nan_to_num(X_num_imp, nan=0.0)
            X_num_scaled = self.numeric_scaler.transform(X_num_imp)
            parts.append(np.nan_to_num(X_num_scaled, nan=0.0).astype(np.float32))

        # 2. Categorical transform
        if self.categorical_cols:
            X_cat_df = X[self.categorical_cols].astype(str)
            X_cat_imp = self.categorical_imputer.transform(X_cat_df)
            X_cat_ohe = self.categorical_encoder.transform(X_cat_imp)
            parts.append(X_cat_ohe.astype(np.float32))

        # 3. Ordinal transform
        if self.ordinal_cols:
            X_ord_df = X[self.ordinal_cols].astype(str)
            X_ord_imp = self.ordinal_imputer.transform(X_ord_df)
            ord_encoded = []
            for idx, col in enumerate(self.ordinal_cols):
                encoder = self.ordinal_encoders[col]
                col_enc = encoder.transform(X_ord_imp[:, [idx]])
                ord_encoded.append(col_enc)
            X_ord_mat = np.hstack(ord_encoded)
            X_ord_scaled = self.ordinal_scaler.transform(X_ord_mat)
            parts.append(np.nan_to_num(X_ord_scaled, nan=0.0).astype(np.float32))

        # 4. Boolean transform
        if self.boolean_cols:
            X_bool_df = X[self.boolean_cols].copy()
            X_bool_imp = self.boolean_imputer.transform(X_bool_df)
            # Map truthy/falsy to 0.0 / 1.0
            X_bool_arr = (X_bool_imp.astype(str) == "True").astype(np.float32)
            parts.append(X_bool_arr)

        if not parts:
            raise ValueError("No features available to transform.")

        out = np.hstack(parts).astype(np.float32)

        # Enforce finite representation
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        return out

    def count_unseen_categories(self, X_target: pd.DataFrame) -> int:
        """Count total occurrences of unseen categorical levels in target data."""
        if not self.categorical_cols or not self.source_category_vocabularies_:
            return 0

        unseen_count = 0
        for col in self.categorical_cols:
            if col in X_target.columns:
                source_vocab = self.source_category_vocabularies_.get(col, set())
                target_vals = X_target[col].dropna().astype(str).tolist()
                for val in target_vals:
                    if val not in source_vocab:
                        unseen_count += 1
        return unseen_count


def build_preprocessor(
    feature_roles: Dict[str, str],
    config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> SourceOnlyPreprocessor:
    """Instantiate a source-only preprocessor adhering to Phase 2 configuration."""
    return SourceOnlyPreprocessor(
        feature_roles=feature_roles,
        config=config,
        metadata=metadata,
    )
