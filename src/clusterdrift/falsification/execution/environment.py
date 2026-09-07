"""Environment inspection, CPU topology discovery, and threadpool bounding."""

import os
import platform
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import scipy
import sklearn
import joblib
import psutil
from threadpoolctl import threadpool_info, threadpool_limits


def get_library_versions() -> Dict[str, str]:
    """Capture exact versions of runtime scientific libraries."""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
    }


def get_cpu_topology() -> Dict[str, Any]:
    """Inspect CPU topology, physical/logical cores, RAM, and active BLAS/OpenMP layers."""
    mem = psutil.virtual_memory()
    phys_cores = psutil.cpu_count(logical=False) or 1
    log_cores = psutil.cpu_count(logical=True) or 1
    tb_info = threadpool_info()

    return {
        "physical_cores": phys_cores,
        "logical_cores": log_cores,
        "total_ram_gb": round(mem.total / (1024 ** 3), 2),
        "available_ram_gb": round(mem.available / (1024 ** 3), 2),
        "platform": platform.platform(),
        "threadpool_layers": tb_info,
    }


def set_blas_thread_env(n_threads: int = 1) -> None:
    """Set process-wide environment variables to bound BLAS/OpenMP oversubscription."""
    t_str = str(n_threads)
    for env_var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ]:
        os.environ[env_var] = t_str


def limit_inner_threads(limits: int = 1):
    """Context manager bounding inner BLAS/OpenMP library threads to prevent oversubscription."""
    return threadpool_limits(limits=limits)