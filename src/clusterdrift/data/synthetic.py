"""Deterministic synthetic benchmark data generation with exact soft posterior memberships."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import gamma, gammaln, logsumexp

from clusterdrift.data.schemas import SyntheticDatasetSpec


def _compute_gaussian_log_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Compute log N(X | mean, cov) in a numerically stable manner."""
    d = X.shape[1]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    inv_cov = np.linalg.inv(cov)
    quad = np.sum(diff @ inv_cov * diff, axis=1)
    log_pdf = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
    return log_pdf


def _compute_student_t_log_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, df: float) -> np.ndarray:
    """Compute multivariate Student-t log pdf."""
    d = X.shape[1]
    diff = X - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    inv_cov = np.linalg.inv(cov)
    quad = np.sum(diff @ inv_cov * diff, axis=1)

    log_const = gammaln((df + d) / 2.0) - gammaln(df / 2.0) - 0.5 * (d * np.log(df * np.pi) + logdet)
    log_pdf = log_const - 0.5 * (df + d) * np.log1p(quad / df)
    return log_pdf


def compute_posterior_memberships(
    X: np.ndarray,
    priors: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
    df: Optional[float] = None,
) -> np.ndarray:
    """
    Calculate the exact true posterior membership matrix tau:
      tau_ik = (pi_k * p(x_i | theta_k)) / sum_j (pi_j * p(x_i | theta_j))
    using logsumexp for machine-precision stability.
    """
    n, d = X.shape
    K = len(priors)
    log_weighted_densities = np.zeros((n, K))

    for k in range(K):
        log_prior = np.log(priors[k])
        if df is None:
            log_dens = _compute_gaussian_log_pdf(X, means[k], covariances[k])
        else:
            log_dens = _compute_student_t_log_pdf(X, means[k], covariances[k], df)
        log_weighted_densities[:, k] = log_prior + log_dens

    # Normalize via logsumexp across clusters
    log_normalizer = logsumexp(log_weighted_densities, axis=1, keepdims=True)
    tau = np.exp(log_weighted_densities - log_normalizer)

    # Enforce strictly valid probability simplex
    assert np.allclose(tau.sum(axis=1), 1.0, atol=1e-5), "Posterior memberships do not sum to 1.0."
    assert (tau >= 0.0).all(), "Negative posterior probabilities encountered."
    return tau


def generate_synthetic_family(spec: SyntheticDatasetSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Deterministically generate one of the eight synthetic benchmark families:
      s01: balanced GMM
      s02: overlapping GMM
      s03: imbalanced GMM
      s04: heteroscedastic GMM
      s05: anisotropic GMM
      s06: high-dimensional GMM
      s07: irrelevant features GMM
      s08: heavy-tailed Student-t mixture
    """
    rng = np.random.default_rng(spec.generator_seed)
    K = spec.K
    n = spec.n
    d = spec.d
    p = spec.parameters

    # 1. Determine cluster priors
    if "priors" in p:
        priors = np.array(p["priors"], dtype=float)
        priors /= priors.sum()
    else:
        priors = np.full(K, 1.0 / K)

    # 2. Determine cluster centers
    separation = p.get("separation", 3.5)
    means = np.zeros((K, d))

    if spec.family_id == "s07_irrelevant_features_gmm":
        # First 20 dimensions are informative, remaining 80 are pure noise
        inf_d = p.get("informative_d", 20)
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
        for k in range(K):
            means[k, :inf_d] = rng.standard_normal(inf_d) * 0.5
            means[k, 0] = separation * np.cos(angles[k])
            means[k, 1] = separation * np.sin(angles[k])
    else:
        # Place centroids symmetrically or in orthogonal coordinate directions
        if d >= K:
            for k in range(K):
                means[k, k] = separation
        else:
            angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
            for k in range(K):
                means[k, 0] = separation * np.cos(angles[k])
                means[k, 1] = separation * np.sin(angles[k])
                if d > 2:
                    means[k, 2:] = rng.standard_normal(d - 2) * 0.2

    # 3. Determine covariance matrices
    covariances = np.zeros((K, d, d))
    if spec.family_id == "s04_heteroscedastic_gmm":
        variances = p.get("variances", [0.5, 1.5, 3.0])
        for k in range(K):
            var_k = variances[k % len(variances)]
            covariances[k] = np.eye(d) * var_k
    elif spec.family_id == "s05_anisotropic_gmm":
        eccentricity = p.get("eccentricity", 4.0)
        for k in range(K):
            # Create random orthonormal rotation matrix
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            # Diagonal eigenvalues with high eccentricity
            eigenvals = np.ones(d)
            eigenvals[0] = eccentricity
            eigenvals[1] = 1.0 / eccentricity
            covariances[k] = Q @ np.diag(eigenvals) @ Q.T
    else:
        cov_scale = p.get("covariance_scale", 1.0)
        for k in range(K):
            covariances[k] = np.eye(d) * cov_scale

    # 4. Sample samples according to priors
    cluster_counts = rng.multinomial(n, priors)
    X = np.zeros((n, d))
    y = np.zeros(n, dtype=int)
    idx = 0

    df = p.get("df") if spec.family_id == "s08_student_t_mixture" else None

    for k in range(K):
        count_k = cluster_counts[k]
        if count_k == 0:
            continue
        if df is not None:
            # Sample from multivariate Student-t: mu + Z / sqrt(u / df) where u ~ chi2(df)
            Z = rng.multivariate_normal(np.zeros(d), covariances[k], size=count_k)
            u = rng.chisquare(df, size=(count_k, 1))
            samples = means[k] + Z / np.sqrt(u / df)
        else:
            samples = rng.multivariate_normal(means[k], covariances[k], size=count_k)

        X[idx : idx + count_k] = samples
        y[idx : idx + count_k] = k
        idx += count_k

    # 5. Compute exact true soft posterior memberships
    tau = compute_posterior_memberships(X, priors, means, covariances, df=df)

    parameters_record = {
        "family_id": spec.family_id,
        "generator": spec.generator,
        "generator_seed": spec.generator_seed,
        "K": K,
        "n": n,
        "d": d,
        "priors": priors.tolist(),
        "means": means.tolist(),
        "covariances": [cov.tolist() for cov in covariances],
        "df": df,
    }

    return X, y, tau, parameters_record


def save_synthetic_dataset(
    spec: SyntheticDatasetSpec,
    output_root: Optional[Path] = None,
) -> Dict[str, str]:
    """Generate and serialize synthetic dataset artifacts into standardized directory."""
    root = output_root or (Path.cwd() / "data" / "synthetic")
    target_dir = root / spec.family_id
    target_dir.mkdir(parents=True, exist_ok=True)

    X, y, tau, params = generate_synthetic_family(spec)

    feature_names = [f"f_{i}" for i in range(spec.d)]
    df_X = pd.DataFrame(X, columns=feature_names)
    df_y = pd.DataFrame({"true_label": y})

    features_path = target_dir / "features.parquet"
    labels_path = target_dir / "hard_labels.parquet"
    tau_path = target_dir / "soft_memberships.npy"
    params_path = target_dir / "parameters.json"
    meta_path = target_dir / "metadata.json"

    df_X.to_parquet(features_path, index=False, engine="pyarrow")
    df_y.to_parquet(labels_path, index=False, engine="pyarrow")
    np.save(tau_path, tau)

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    metadata = {
        "dataset_id": spec.family_id,
        "dataset_type": "synthetic",
        "generator": spec.generator,
        "generator_seed": spec.generator_seed,
        "K": spec.K,
        "n_rows": spec.n,
        "n_features": spec.d,
        "soft_ground_truth_available": True,
        "notes": spec.notes,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "features": str(features_path),
        "hard_labels": str(labels_path),
        "soft_memberships": str(tau_path),
        "parameters": str(params_path),
        "metadata": str(meta_path),
    }
