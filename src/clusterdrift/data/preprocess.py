"""Source-only preprocessing module guaranteeing zero label or target leakage."""

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


BOOLEAN_TRUE_VALUES = frozenset({True, 1, 1.0, "true", "1", "1.0", "t", "yes", "y"})
BOOLEAN_FALSE_VALUES = frozenset({False, 0, 0.0, "false", "0", "0.0", "f", "no", "n"})


def normalize_boolean_value(val: Any) -> Any:
    """Normalize acceptable boolean representations or raise ValueError for invalid inputs."""
    if pd.isna(val) or val is None or val == "":
        return np.nan

    if isinstance(val, (bool, np.bool_)):
        return bool(val)

    if isinstance(val, (int, float, np.integer, np.floating)):
        if val == 1:
            return True
        if val == 0:
            return False
        raise ValueError(f"Invalid numeric value for boolean feature: {val}")

    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in BOOLEAN_TRUE_VALUES:
            return True
        if val_clean in BOOLEAN_FALSE_VALUES:
            return False
        raise ValueError(f"Invalid string value for boolean feature: '{val}'")

    raise ValueError(f"Unrecognized type for boolean feature: {type(val)} ({val})")


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
        self.source_category_vocabularies_: Dict[str, Set[str]] = {}

    def fit(self, X: pd.DataFrame) -> "SourceOnlyPreprocessor":
        """Fit preprocessing transformations strictly on source features without target or label access."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")

        # 1. Numeric features
        if self.numeric_cols:
            X_num = X[self.numeric_cols]
            # Disallow raw infinity
            arr_num = X_num.to_numpy(dtype=float, copy=False)
            if np.isneginf(arr_num).any() or np.isposinf(arr_num).any():
                raise ValueError("Infinite values (+inf or -inf) detected in raw numeric features.")

            self.numeric_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            X_num_imp = self.numeric_imputer.fit_transform(X_num)

            # Preserve entirely missing source features with documented fallback (0.0 before scaler)
            fallback_num = self.config.get("numeric", {}).get("all_missing_fallback", 0.0)
            for i, col in enumerate(self.numeric_cols):
                if X_num[col].isna().all() or (
                    hasattr(self.numeric_imputer, "statistics_")
                    and np.isnan(self.numeric_imputer.statistics_[i])
                ):
                    self.numeric_imputer.statistics_[i] = fallback_num
                    X_num_imp[:, i] = fallback_num

            self.numeric_scaler = StandardScaler()
            self.numeric_scaler.fit(X_num_imp)

        # 2. Categorical features
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype("object")
            # Normalize pandas missing values to np.nan
            X_cat = X_cat.where(pd.notna(X_cat), np.nan)

            # Record genuine source category vocabulary (excluding missing values)
            self.source_category_vocabularies_ = {}
            for col in self.categorical_cols:
                non_null = X_cat[col].dropna()
                self.source_category_vocabularies_[col] = set(non_null.astype(str).unique())

            self.categorical_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
            X_cat_imp = self.categorical_imputer.fit_transform(X_cat)

            # Preserve entirely missing categorical source features with deterministic sentinel
            sentinel = self.config.get("categorical", {}).get("all_missing_sentinel", "__MISSING_SOURCE__")
            for i, col in enumerate(self.categorical_cols):
                if X_cat[col].isna().all():
                    self.categorical_imputer.statistics_[i] = sentinel
                    X_cat_imp[:, i] = sentinel
                    self.source_category_vocabularies_[col] = {sentinel}

            # Only after imputation normalize values to strings for encoding
            X_cat_str = X_cat_imp.astype(str)

            min_freq = self.config.get("categorical", {}).get("min_frequency", 0.01)
            if min_freq is not None and min_freq <= 0.0:
                min_freq = None
            self.categorical_encoder = OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_freq,
                sparse_output=False,
                dtype=np.float32,
            )
            self.categorical_encoder.fit(X_cat_str)

        # 3. Explicitly ordered ordinal features
        if self.ordinal_cols:
            category_orderings = self.metadata.get("category_orderings") or {}
            X_ord = X[self.ordinal_cols].astype("object")
            X_ord = X_ord.where(pd.notna(X_ord), np.nan)

            self.ordinal_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
            X_ord_imp = self.ordinal_imputer.fit_transform(X_ord)

            ord_sentinel = self.config.get("ordinal", {}).get("all_missing_sentinel", "__MISSING_SOURCE__")
            for i, col in enumerate(self.ordinal_cols):
                if X_ord[col].isna().all():
                    order = category_orderings.get(col, [])
                    fallback_ord = order[0] if order else ord_sentinel
                    self.ordinal_imputer.statistics_[i] = fallback_ord
                    X_ord_imp[:, i] = fallback_ord

            X_ord_str = X_ord_imp.astype(str)
            ord_encoded = []
            for idx, col in enumerate(self.ordinal_cols):
                order = category_orderings[col]
                encoder = OrdinalEncoder(
                    categories=[order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                col_enc = encoder.fit_transform(X_ord_str[:, [idx]])
                self.ordinal_encoders[col] = encoder
                ord_encoded.append(col_enc)

            X_ord_mat = np.hstack(ord_encoded)
            self.ordinal_scaler = StandardScaler()
            self.ordinal_scaler.fit(X_ord_mat)

        # 4. Boolean features
        if self.boolean_cols:
            # Map canonical booleans, rejecting unrecognized values
            X_bool_df = pd.DataFrame(index=X.index)
            for col in self.boolean_cols:
                X_bool_df[col] = X[col].apply(normalize_boolean_value)

            self.boolean_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
            X_bool_imp = self.boolean_imputer.fit_transform(X_bool_df)

            fallback_bool = bool(self.config.get("boolean", {}).get("all_missing_fallback", False))
            for i, col in enumerate(self.boolean_cols):
                if X_bool_df[col].isna().all():
                    self.boolean_imputer.statistics_[i] = fallback_bool
                    X_bool_imp[:, i] = fallback_bool

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
            arr_num = X_num.to_numpy(dtype=float, copy=False)
            if np.isneginf(arr_num).any() or np.isposinf(arr_num).any():
                raise ValueError("Infinite values (+inf or -inf) detected in numeric features during transform.")

            X_num_imp = self.numeric_imputer.transform(X_num)
            X_num_scaled = self.numeric_scaler.transform(X_num_imp)
            parts.append(X_num_scaled.astype(np.float32))

        # 2. Categorical transform
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype("object")
            X_cat = X_cat.where(pd.notna(X_cat), np.nan)
            X_cat_imp = self.categorical_imputer.transform(X_cat)
            X_cat_str = X_cat_imp.astype(str)
            X_cat_ohe = self.categorical_encoder.transform(X_cat_str)
            parts.append(X_cat_ohe.astype(np.float32))

        # 3. Ordinal transform
        if self.ordinal_cols:
            X_ord = X[self.ordinal_cols].astype("object")
            X_ord = X_ord.where(pd.notna(X_ord), np.nan)
            X_ord_imp = self.ordinal_imputer.transform(X_ord)
            X_ord_str = X_ord_imp.astype(str)
            ord_encoded = []
            for idx, col in enumerate(self.ordinal_cols):
                encoder = self.ordinal_encoders[col]
                col_enc = encoder.transform(X_ord_str[:, [idx]])
                ord_encoded.append(col_enc)
            X_ord_mat = np.hstack(ord_encoded)
            X_ord_scaled = self.ordinal_scaler.transform(X_ord_mat)
            parts.append(X_ord_scaled.astype(np.float32))

        # 4. Boolean transform
        if self.boolean_cols:
            X_bool_df = pd.DataFrame(index=X.index)
            for col in self.boolean_cols:
                X_bool_df[col] = X[col].apply(normalize_boolean_value)
            X_bool_imp = self.boolean_imputer.transform(X_bool_df)
            X_bool_arr = (X_bool_imp == True).astype(np.float32)
            parts.append(X_bool_arr)

        if not parts:
            raise ValueError("No features available to transform.")

        out = np.hstack(parts).astype(np.float32)

        # Enforce finite representation: do NOT silently hide raw/transformed infinity!
        if np.isnan(out).any():
            raise ValueError("NaN values detected in preprocessed output after imputation.")
        if np.isneginf(out).any() or np.isposinf(out).any():
            raise ValueError("Infinite values (+inf or -inf) detected in preprocessed output.")

        return out

    def get_unseen_categories(self, X_target: pd.DataFrame) -> Tuple[int, List[str]]:
        """Count total occurrences of unseen categorical levels and list affected column names."""
        if not self.categorical_cols or not self.source_category_vocabularies_:
            return 0, []

        X_cat_tgt = X_target[self.categorical_cols].astype("object")
        X_cat_tgt = X_cat_tgt.where(pd.notna(X_cat_tgt), np.nan)

        unseen_count = 0
        affected_cols = []

        for col in self.categorical_cols:
            if col in X_cat_tgt.columns:
                source_vocab = self.source_category_vocabularies_.get(col, set())
                target_vals = X_cat_tgt[col].dropna().astype(str).tolist()
                col_has_unseen = False
                for val in target_vals:
                    if val not in source_vocab:
                        unseen_count += 1
                        col_has_unseen = True
                if col_has_unseen:
                    affected_cols.append(col)

        return unseen_count, affected_cols

    def count_unseen_categories(self, X_target: pd.DataFrame) -> int:
        """Count total occurrences of unseen categorical levels in target data."""
        return self.get_unseen_categories(X_target)[0]


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
