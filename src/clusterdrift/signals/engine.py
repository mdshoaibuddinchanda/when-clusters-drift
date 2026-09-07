"""Label-Free Structural Signal Engine and Multi-Tier Cache.

Coordinates:
    - Label-free model fitting (source and shadow candidate)
    - Single reference-derived Hungarian cluster alignment
    - Computation of primary structural signal vector Z_t = [D_U^R, D_U^C, D_V, D_H, D_M, D_X]
    - Conventional validity controls on current probe bank
    - Cryptographic record hashing and usability determination
"""

from threading import Lock
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from clusterdrift.alignment.costs import compute_reference_cluster_scales
from clusterdrift.alignment.hungarian import align_clusters
from clusterdrift.alignment.state import compute_model_fingerprint
from clusterdrift.methods.fcm import FCM
from clusterdrift.methods.gmm import GMM
from clusterdrift.probes.evaluation import ProbeEvaluation, verify_probe_identity
from clusterdrift.signals.covariate import compute_mmd_b2, derive_source_median_bandwidth
from clusterdrift.signals.divergence import check_probability_simplex
from clusterdrift.signals.hashing import (
    compute_signal_protocol_sha256,
    compute_signal_record_sha256,
)
from clusterdrift.signals.result import SignalResult
from clusterdrift.signals.structural import compute_all_structural_signals
from clusterdrift.signals.validity import compute_validity_controls


class SignalCache:
    """Thread-safe multi-tier cache for models, representations, probe memberships, and MMD."""

    def __init__(self):
        self._lock = Lock()
        self._preprocessors: Dict[Tuple[str, int, str], Any] = {}
        self._source_models: Dict[Tuple[str, int, str, int], Any] = {}
        self._reference_scales: Dict[str, np.ndarray] = {}
        self._transformed_banks: Dict[str, np.ndarray] = {}
        self._memberships: Dict[Tuple[str, str], np.ndarray] = {}
        self._mmd_sigmas: Dict[Tuple[str, int], Tuple[float, str]] = {}
        self._mmd_source_sums: Dict[Tuple[str, int], float] = {}
        self._dx_cache: Dict[Tuple[str, int, str], float] = {}

        self.hits = {
            "preprocessor": 0,
            "source_model": 0,
            "reference_scales": 0,
            "transformed_bank": 0,
            "membership": 0,
            "mmd_sigma": 0,
            "mmd_source_sum": 0,
            "dx": 0,
        }
        self.misses = {
            "preprocessor": 0,
            "source_model": 0,
            "reference_scales": 0,
            "transformed_bank": 0,
            "membership": 0,
            "mmd_sigma": 0,
            "mmd_source_sum": 0,
            "dx": 0,
        }

    def get_preprocessor(self, dataset_id: str, outer_fold: int, prep_sha: str) -> Optional[Any]:
        key = (dataset_id, outer_fold, prep_sha)
        with self._lock:
            if key in self._preprocessors:
                self.hits["preprocessor"] += 1
                return self._preprocessors[key]
            self.misses["preprocessor"] += 1
            return None

    def put_preprocessor(self, dataset_id: str, outer_fold: int, prep_sha: str, prep: Any) -> None:
        key = (dataset_id, outer_fold, prep_sha)
        with self._lock:
            self._preprocessors[key] = prep

    def get_source_model(self, dataset_id: str, outer_fold: int, method: str, seed: int) -> Optional[Any]:
        key = (dataset_id, outer_fold, method, seed)
        with self._lock:
            if key in self._source_models:
                self.hits["source_model"] += 1
                return self._source_models[key]
            self.misses["source_model"] += 1
            return None

    def put_source_model(self, dataset_id: str, outer_fold: int, method: str, seed: int, model: Any) -> None:
        key = (dataset_id, outer_fold, method, seed)
        with self._lock:
            self._source_models[key] = model

    def get_reference_scales(self, model_fp: str) -> Optional[np.ndarray]:
        with self._lock:
            if model_fp in self._reference_scales:
                self.hits["reference_scales"] += 1
                return self._reference_scales[model_fp].copy()
            self.misses["reference_scales"] += 1
            return None

    def put_reference_scales(self, model_fp: str, scales: np.ndarray) -> None:
        with self._lock:
            self._reference_scales[model_fp] = scales.copy()

    def get_transformed_bank(self, bank_sha256: str) -> Optional[np.ndarray]:
        with self._lock:
            if bank_sha256 in self._transformed_banks:
                self.hits["transformed_bank"] += 1
                return self._transformed_banks[bank_sha256].copy()
            self.misses["transformed_bank"] += 1
            return None

    def put_transformed_bank(self, bank_sha256: str, matrix: np.ndarray) -> None:
        with self._lock:
            self._transformed_banks[bank_sha256] = matrix.copy()

    def get_membership(self, model_fp: str, bank_sha256: str) -> Optional[np.ndarray]:
        key = (model_fp, bank_sha256)
        with self._lock:
            if key in self._memberships:
                self.hits["membership"] += 1
                return self._memberships[key].copy()
            self.misses["membership"] += 1
            return None

    def put_membership(self, model_fp: str, bank_sha256: str, U: np.ndarray) -> None:
        key = (model_fp, bank_sha256)
        with self._lock:
            self._memberships[key] = U.copy()

    def get_mmd_sigma(self, dataset_id: str, outer_fold: int) -> Optional[Tuple[float, str]]:
        key = (dataset_id, outer_fold)
        with self._lock:
            if key in self._mmd_sigmas:
                self.hits["mmd_sigma"] += 1
                return self._mmd_sigmas[key]
            self.misses["mmd_sigma"] += 1
            return None

    def put_mmd_sigma(self, dataset_id: str, outer_fold: int, sigma: float, status: str) -> None:
        key = (dataset_id, outer_fold)
        with self._lock:
            self._mmd_sigmas[key] = (sigma, status)

    def get_mmd_source_kernel_sum(self, dataset_id: str, outer_fold: int) -> Optional[float]:
        key = (dataset_id, outer_fold)
        with self._lock:
            if key in self._mmd_source_sums:
                self.hits["mmd_source_sum"] += 1
                return self._mmd_source_sums[key]
            self.misses["mmd_source_sum"] += 1
            return None

    def put_mmd_source_kernel_sum(self, dataset_id: str, outer_fold: int, sum_xx: float) -> None:
        key = (dataset_id, outer_fold)
        with self._lock:
            self._mmd_source_sums[key] = sum_xx

    def get_dx(self, dataset_id: str, outer_fold: int, condition: str) -> Optional[float]:
        key = (dataset_id, outer_fold, condition)
        with self._lock:
            if key in self._dx_cache:
                self.hits["dx"] += 1
                return self._dx_cache[key]
            self.misses["dx"] += 1
            return None

    def put_dx(self, dataset_id: str, outer_fold: int, condition: str, dx: float) -> None:
        key = (dataset_id, outer_fold, condition)
        with self._lock:
            self._dx_cache[key] = dx

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "hits": dict(self.hits),
                "misses": dict(self.misses),
                "cached_preprocessors": len(self._preprocessors),
                "cached_source_models": len(self._source_models),
                "cached_reference_scales": len(self._reference_scales),
                "cached_transformed_banks": len(self._transformed_banks),
                "cached_memberships": len(self._memberships),
                "cached_mmd_sigmas": len(self._mmd_sigmas),
                "cached_mmd_source_sums": len(self._mmd_source_sums),
                "cached_dx": len(self._dx_cache),
            }


class SignalEngine:
    """Orchestrates label-free structural signal generation and conventional controls."""

    def __init__(
        self,
        signals_cfg: Dict[str, Any],
        alignment_cfg: Optional[Dict[str, Any]] = None,
        methods_cfg: Optional[Dict[str, Any]] = None,
        cache: Optional[SignalCache] = None,
    ):
        self.signals_cfg = signals_cfg
        self.alignment_cfg = alignment_cfg or {}
        self.methods_cfg = methods_cfg or {}
        self.cache = cache or SignalCache()
        self.signal_protocol_sha256 = compute_signal_protocol_sha256(signals_cfg)

    def _fit_model(
        self,
        method: str,
        K: int,
        seed: int,
        X: np.ndarray,
    ) -> Any:
        """Fit unsupervised clustering model with locked hyperparameter policy."""
        if method == "fcm_adaptive":
            m = FCM(
                n_clusters=K,
                random_state=seed,
                fuzzifier_policy="dimension_adaptive",
            )
        elif method == "gmm":
            m = GMM(
                n_clusters=K,
                random_state=seed,
                covariance_type="full",
                n_init=1,
                reg_covar=1e-6,
            )
        else:
            raise ValueError(f"Unsupported clustering method for Phase 6: {method}")

        m.fit(X)
        return m

    def compute_signals(
        self,
        dataset_id: str,
        outer_fold: int,
        condition: str,
        method: str,
        seed: int,
        K: int,
        X_source_trans: np.ndarray,
        X_target_shifted_trans: np.ndarray,
        A_R: np.ndarray,
        A_C: np.ndarray,
        ref_bank_sha256: str,
        cur_bank_sha256: str,
        shift_spec_sha256: str,
    ) -> SignalResult:
        """Compute all label-free structural signals and conventional controls for a scenario."""
        t_tot_start = time.perf_counter()

        # 1. Source Model (retrieved from cache or fit once)
        t_src_start = time.perf_counter()
        m_src = self.cache.get_source_model(dataset_id, outer_fold, method, seed)
        if m_src is None:
            m_src = self._fit_model(method, K, seed, X_source_trans)
            self.cache.put_source_model(dataset_id, outer_fold, method, seed, m_src)
            runtime_source_fit = time.perf_counter() - t_src_start
        else:
            runtime_source_fit = 0.0

        src_status = getattr(m_src, "status_", "SUCCESS" if getattr(m_src, "converged_", True) else "FAILED")
        src_conv = bool(getattr(m_src, "converged_", True))
        src_degen = bool(getattr(m_src, "degenerate_solution_", False))
        fp_src = compute_model_fingerprint(method, {}, seed, m_src.cluster_centers_)

        # 2. Reference Cluster Scales
        scales_ref = self.cache.get_reference_scales(fp_src)
        if scales_ref is None:
            U_src = m_src.predict_membership(X_source_trans)
            scales_ref, _, _ = compute_reference_cluster_scales(
                X_source_trans, U_src, m_src.cluster_centers_
            )
            self.cache.put_reference_scales(fp_src, scales_ref)

        # 3. Source reference bank memberships
        U_0_R = self.cache.get_membership(fp_src, ref_bank_sha256)
        if U_0_R is None:
            U_0_R = m_src.predict_membership(A_R)
            self.cache.put_membership(fp_src, ref_bank_sha256, U_0_R)

        # 4. Candidate Model (fitted independently and unsupervised on shifted target)
        t_cand_start = time.perf_counter()
        m_cand = self._fit_model(method, K, seed, X_target_shifted_trans)
        runtime_candidate_fit = time.perf_counter() - t_cand_start

        cand_status = getattr(m_cand, "status_", "SUCCESS" if getattr(m_cand, "converged_", True) else "FAILED")
        cand_conv = bool(getattr(m_cand, "converged_", True))
        cand_degen = bool(getattr(m_cand, "degenerate_solution_", False))
        fp_cand = compute_model_fingerprint(method, {}, seed, m_cand.cluster_centers_)

        # 5. Candidate reference bank memberships
        U_t_R = m_cand.predict_membership(A_R)

        # 6. Verify probe identity on reference bank
        eval_ref_0 = ProbeEvaluation(
            bank_sha256=ref_bank_sha256,
            bank_type="reference",
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition="reference",
            model_fingerprint=fp_src,
            memberships=U_0_R,
            n_samples=len(A_R),
            n_clusters=K,
        )
        eval_ref_t = ProbeEvaluation(
            bank_sha256=ref_bank_sha256,
            bank_type="reference",
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
            model_fingerprint=fp_cand,
            memberships=U_t_R,
            n_samples=len(A_R),
            n_clusters=K,
        )
        verify_probe_identity(eval_ref_0, eval_ref_t)

        # 7. Cluster Alignment (Hungarian solver on reference probe bank A^R)
        t_align_start = time.perf_counter()
        eta = float(self.alignment_cfg.get("eta", 0.50))
        align_res = align_clusters(
            centers_ref=m_src.cluster_centers_,
            centers_cand=m_cand.cluster_centers_,
            scales_ref=scales_ref,
            U_ref=U_0_R,
            U_cand=U_t_R,
            eta=eta,
        )
        runtime_alignment = time.perf_counter() - t_align_start

        # Aligned candidate centers and reference memberships
        centers_cand_aligned = align_res.aligned_centers
        U_t_R_aligned = align_res.apply_to_memberships(U_t_R)

        # 8. Current Probe Bank Evaluations
        U_0_C = m_src.predict_membership(A_C)
        U_t_C = m_cand.predict_membership(A_C)

        eval_cur_0 = ProbeEvaluation(
            bank_sha256=cur_bank_sha256,
            bank_type="current",
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
            model_fingerprint=fp_src,
            memberships=U_0_C,
            n_samples=len(A_C),
            n_clusters=K,
        )
        eval_cur_t = ProbeEvaluation(
            bank_sha256=cur_bank_sha256,
            bank_type="current",
            dataset_id=dataset_id,
            outer_fold=outer_fold,
            condition=condition,
            model_fingerprint=fp_cand,
            memberships=U_t_C,
            n_samples=len(A_C),
            n_clusters=K,
        )
        verify_probe_identity(eval_cur_0, eval_cur_t)

        # Apply SAME reference-derived permutation to current candidate memberships
        U_t_C_aligned = align_res.apply_to_memberships(U_t_C)

        # 9. Primary Structural Signals (D_U^R, D_U^C, D_V, D_H, D_M)
        t_sig_start = time.perf_counter()
        struct_stats = compute_all_structural_signals(
            U_source_reference=U_0_R,
            U_candidate_reference_aligned=U_t_R_aligned,
            U_source_current=U_0_C,
            U_candidate_current_aligned=U_t_C_aligned,
            centers_source=m_src.cluster_centers_,
            centers_candidate_aligned=centers_cand_aligned,
            reference_scales=scales_ref,
            epsilon_v=float(self.signals_cfg.get("prototype_movement", {}).get("epsilon", 1e-12)),
            validate_simplex=True,
        )
        runtime_signals = time.perf_counter() - t_sig_start

        # 10. Covariate Shift Signal D_X = MMD_b^2(A^R, A_t^C)
        t_mmd_start = time.perf_counter()
        cached_dx = self.cache.get_dx(dataset_id, outer_fold, condition)
        if cached_dx is not None:
            D_X = cached_dx
            sigma_info = self.cache.get_mmd_sigma(dataset_id, outer_fold)
            sigma, mmd_status = sigma_info if sigma_info else (1.0, "CACHED")
            runtime_mmd = 0.0
        else:
            sigma_info = self.cache.get_mmd_sigma(dataset_id, outer_fold)
            if sigma_info is None:
                max_pts = int(self.signals_cfg.get("mmd", {}).get("bandwidth_max_points", 1024))
                global_seed = int(self.signals_cfg.get("global_signal_seed", 2026090706))
                sigma, mmd_status = derive_source_median_bandwidth(
                    A_R=A_R,
                    dataset_id=dataset_id,
                    outer_fold=outer_fold,
                    ref_bank_sha256=ref_bank_sha256,
                    global_signal_seed=global_seed,
                    max_points=max_pts,
                )
                self.cache.put_mmd_sigma(dataset_id, outer_fold, sigma, mmd_status)
            else:
                sigma, mmd_status = sigma_info

            cached_sum_xx = self.cache.get_mmd_source_kernel_sum(dataset_id, outer_fold)
            chunk_size = int(self.signals_cfg.get("mmd", {}).get("chunk_size", 512))
            neg_tol = float(self.signals_cfg.get("mmd", {}).get("negative_tolerance", 1e-10))

            D_X, sum_xx = compute_mmd_b2(
                X=A_R,
                Y=A_C,
                sigma=sigma,
                chunk_size=chunk_size,
                negative_tolerance=neg_tol,
                cached_sum_xx=cached_sum_xx,
            )
            if cached_sum_xx is None:
                self.cache.put_mmd_source_kernel_sum(dataset_id, outer_fold, sum_xx)
            self.cache.put_dx(dataset_id, outer_fold, condition, D_X)
            runtime_mmd = time.perf_counter() - t_mmd_start

        # 11. Conventional Validity Controls on A_t^C under M_0
        validity_stats = compute_validity_controls(
            X_current=A_C,
            centers_source=m_src.cluster_centers_,
            U_source_current=U_0_C,
            validate_simplex=True,
        )

        # 12. Determine Usability (PART Z)
        is_finite_src = np.all(np.isfinite(m_src.cluster_centers_)) and np.all(np.isfinite(U_0_R)) and np.all(np.isfinite(U_0_C))
        is_finite_cand = np.all(np.isfinite(m_cand.cluster_centers_)) and np.all(np.isfinite(U_t_R)) and np.all(np.isfinite(U_t_C))
        is_non_degen = (not src_degen) and (not cand_degen)
        is_not_ambig = not align_res.ambiguous

        all_signals_finite = (
            np.isfinite(struct_stats["D_U_R"])
            and np.isfinite(struct_stats["D_U_C"])
            and np.isfinite(struct_stats["D_V"])
            and np.isfinite(struct_stats["D_H"])
            and np.isfinite(struct_stats["D_M"])
            and np.isfinite(D_X)
        )

        usable = bool(is_finite_src and is_finite_cand and is_non_degen and is_not_ambig and all_signals_finite)

        runtime_total = time.perf_counter() - t_tot_start

        rec_dict = {
            "signal_protocol_sha256": self.signal_protocol_sha256,
            "dataset_id": dataset_id,
            "outer_fold": outer_fold,
            "condition": condition,
            "method": method,
            "seed": seed,
            "K": K,
            "source_model_status": src_status,
            "candidate_model_status": cand_status,
            "source_converged": src_conv,
            "candidate_converged": cand_conv,
            "source_degenerate": src_degen,
            "candidate_degenerate": cand_degen,
            "usable": usable,
            "reference_bank_sha256": ref_bank_sha256,
            "current_bank_sha256": cur_bank_sha256,
            "shift_spec_sha256": shift_spec_sha256,
            "source_model_fingerprint": fp_src,
            "candidate_model_fingerprint": fp_cand,
            "alignment_permutation": [int(x) for x in align_res.permutation],
            "alignment_best_cost": round(float(align_res.best_assignment_cost), 7),
            "alignment_second_best_cost": round(float(align_res.second_best_assignment_cost), 7),
            "alignment_global_margin": round(float(align_res.global_assignment_margin), 7),
            "alignment_ambiguous": bool(align_res.ambiguous),
            "D_U_R": struct_stats["D_U_R"],
            "D_U_R_median": struct_stats["D_U_R_median"],
            "D_U_R_p95": struct_stats["D_U_R_p95"],
            "D_U_C": struct_stats["D_U_C"],
            "D_U_C_median": struct_stats["D_U_C_median"],
            "D_U_C_p95": struct_stats["D_U_C_p95"],
            "D_V": struct_stats["D_V"],
            "D_V_median": struct_stats["D_V_median"],
            "D_V_max": struct_stats["D_V_max"],
            "entropy_source_current": struct_stats["entropy_source_current"],
            "entropy_candidate_current": struct_stats["entropy_candidate_current"],
            "entropy_signed_change": struct_stats["entropy_signed_change"],
            "D_H": struct_stats["D_H"],
            "mass_source_current": struct_stats["mass_source_current"],
            "mass_candidate_current": struct_stats["mass_candidate_current"],
            "D_M": struct_stats["D_M"],
            "D_X": round(float(D_X), 7),
            "mmd_sigma": round(float(sigma), 7),
            "mmd_bandwidth_status": mmd_status,
            "FPC": validity_stats["FPC"],
            "PE": validity_stats["PE"],
            "PE_norm": validity_stats["PE_norm"],
            "XB_soft_m2": validity_stats["XB_soft_m2"],
            "XB_status": validity_stats["XB_status"],
            "silhouette": validity_stats["silhouette"],
            "silhouette_status": validity_stats["silhouette_status"],
            "runtime_source_fit": round(float(runtime_source_fit), 5),
            "runtime_candidate_fit": round(float(runtime_candidate_fit), 5),
            "runtime_alignment": round(float(runtime_alignment), 5),
            "runtime_signals": round(float(runtime_signals), 5),
            "runtime_mmd": round(float(runtime_mmd), 5),
            "runtime_total": round(float(runtime_total), 5),
        }

        signal_record_sha = compute_signal_record_sha256(rec_dict)
        rec_dict["signal_record_sha256"] = signal_record_sha

        return SignalResult(**rec_dict)
