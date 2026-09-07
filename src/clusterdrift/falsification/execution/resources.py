"""Resource guards, memory estimation, and path safety assertions."""

import os
from pathlib import Path
import psutil
from typing import Optional, Union


FORBIDDEN_DATA_SUBDIRS = [
    Path("data/canonical"),
    Path("data/splits"),
    Path("data/shifts"),
    Path("data/probes"),
    Path("data/synthetic"),
    Path("results"),
]


def assert_not_frozen_data_path(target_path: Union[str, Path], project_root: Optional[Path] = None) -> Path:
    """Hard safety guard: REFUSE to use any path resolving inside frozen research data directories."""
    resolved_target = Path(target_path).resolve()

    # Refuse filesystem roots
    if resolved_target == Path(resolved_target.anchor) or str(resolved_target) in ["/", "\\"]:
        raise ValueError(f"Path safety violation: Target path cannot be a root directory: {resolved_target}")

    if project_root is not None:
        resolved_proj = Path(project_root).resolve()
        if resolved_target == resolved_proj:
            raise ValueError(f"Path safety violation: Target path cannot be the project repository root: {resolved_target}")

        for sub in FORBIDDEN_DATA_SUBDIRS:
            forbidden_full = (resolved_proj / sub).resolve()
            try:
                resolved_target.relative_to(forbidden_full)
                raise ValueError(
                    f"Path safety violation: Target path '{resolved_target}' resolves inside frozen data directory '{forbidden_full}'"
                )
            except ValueError as e:
                if "Path safety violation" in str(e):
                    raise
                # Not relative to forbidden, safe

    return resolved_target


def get_default_work_dir() -> Path:
    """Determine OS-appropriate default ephemeral runtime directory."""
    env_dir = os.environ.get("CLUSTERDRIFT_WORK_DIR") or os.environ.get("CLUSTERDRIFT_CACHE_DIR")
    if env_dir:
        return Path(env_dir).resolve()

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "clusterdrift" / "when-clusters-drift" / "phase7"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "clusterdrift" / "when-clusters-drift" / "phase7"

    return Path.home() / ".cache" / "clusterdrift" / "when-clusters-drift" / "phase7"


def estimate_task_memory_bytes(N: int, D: int, K: int) -> int:
    """Estimate memory needed for single-task FCM fitting and signal computation."""
    # Distance matrix N x K x 8 bytes
    dist_bytes = N * K * 8
    # Membership matrix N x K x 8 bytes
    u_bytes = N * K * 8
    # Center matrix K x D x 8 bytes
    v_bytes = K * D * 8
    # Intermediate buffers (approx 5x)
    total_est = (dist_bytes + u_bytes + v_bytes) * 5 + (N * D * 8 * 2)
    return max(total_est, 10 * 1024 * 1024)  # Minimum 10MB baseline


def check_memory_headroom(estimated_bytes_per_worker: int, n_workers: int) -> bool:
    """Check if physical RAM can safely accommodate concurrent workers without thrashing."""
    mem = psutil.virtual_memory()
    total_needed = estimated_bytes_per_worker * n_workers
    # Require at least 20% safety margin beyond needed memory
    return mem.available >= (total_needed * 1.2)