"""Phase 7 Falsification Execution Layer: Reliability, Caching, Resumability, and Performance."""

from clusterdrift.falsification.execution.environment import (
    get_cpu_topology,
    limit_inner_threads,
    set_blas_thread_env,
)
from clusterdrift.falsification.execution.resources import (
    assert_not_frozen_data_path,
    estimate_task_memory_bytes,
    get_default_work_dir,
)
from clusterdrift.falsification.execution.cache import (
    PersistentPhase7Cache,
    compute_content_fingerprint,
)
from clusterdrift.falsification.execution.checkpoint import (
    SignalCheckpointManager,
    QualityCheckpointManager,
)
from clusterdrift.falsification.execution.progress import (
    ProgressJournal,
    GracefulInterruptHandler,
)
from clusterdrift.falsification.execution.scheduler import (
    compute_task_complexity,
    sort_tasks_lpt,
)

__all__ = [
    "get_cpu_topology",
    "limit_inner_threads",
    "set_blas_thread_env",
    "assert_not_frozen_data_path",
    "estimate_task_memory_bytes",
    "get_default_work_dir",
    "PersistentPhase7Cache",
    "compute_content_fingerprint",
    "SignalCheckpointManager",
    "QualityCheckpointManager",
    "ProgressJournal",
    "GracefulInterruptHandler",
    "compute_task_complexity",
    "sort_tasks_lpt",
]