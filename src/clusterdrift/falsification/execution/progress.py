"""Progress Journal, Observable Status Reporting, and Graceful Interruption."""

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional

from clusterdrift.falsification.execution.resources import assert_not_frozen_data_path
from clusterdrift.shifts.hashing import atomic_write_json


class GracefulInterruptHandler:
    """Handles SIGINT / SIGTERM gracefully allowing in-flight checkpoints to persist."""

    def __init__(self):
        self.interrupted = False
        self._orig_sigint = signal.getsignal(signal.SIGINT)
        self._orig_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.interrupted = True
        print("\n[INTERRUPT RECEIVED] Finishing in-flight row checkpoint before clean exit...")

    def restore(self):
        signal.signal(signal.SIGINT, self._orig_sigint)
        signal.signal(signal.SIGTERM, self._orig_sigterm)


class ProgressJournal:
    """Durable progress journaling, ETA estimation, and runtime logging."""

    def __init__(
        self,
        work_dir: Path,
        protocol_sha: str,
        total_rows: int = 4500,
        project_root: Optional[Path] = None,
    ):
        self.project_root = Path(project_root).resolve() if project_root else None
        self.work_dir = assert_not_frozen_data_path(work_dir, self.project_root)
        self.protocol_sha = protocol_sha
        self.total_rows = total_rows

        self.journal_path = self.work_dir / "progress.json"
        self.log_path = self.work_dir / "phase7_execution.log"

        self.start_time = time.perf_counter()
        self.failures: List[Dict[str, Any]] = []

    def log(self, message: str, to_console: bool = True) -> None:
        """Write timestamped entry to durable execution log and optionally console."""
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {message}\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)
        if to_console:
            print(message)

    def update(
        self,
        completed_rows: int,
        completed_tasks: int,
        total_tasks: int = 60,
        active_task: Optional[str] = None,
        cache_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically update progress.json with measured throughput and ETA."""
        elapsed = time.perf_counter() - self.start_time
        remaining_rows = max(0, self.total_rows - completed_rows)

        rows_per_sec = (completed_rows / elapsed) if elapsed > 0 else 0.0
        rows_per_hour = rows_per_sec * 3600.0

        if rows_per_sec > 0:
            rem_sec = remaining_rows / rows_per_sec
            eta_dt = datetime.now(timezone.utc) + timedelta(seconds=rem_sec)
            eta_str = eta_dt.isoformat()
        else:
            rem_sec = 0.0
            eta_str = "calculating..."

        doc = {
            "protocol_sha": self.protocol_sha,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "total_rows": self.total_rows,
            "completed_rows": completed_rows,
            "remaining_rows": remaining_rows,
            "percent_complete": round((completed_rows / self.total_rows) * 100.0, 2),
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "active_task": active_task or "none",
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_hour": round(rows_per_hour, 2),
            "estimated_remaining_seconds": round(rem_sec, 2),
            "estimated_completion_time": eta_str,
            "failures_count": len(self.failures),
            "cache_metrics": cache_metrics or {},
        }

        atomic_write_json(self.journal_path, doc, indent=2)
        return doc

    def record_failure(self, task_key: str, error_msg: str) -> None:
        fail_entry = {
            "task_key": task_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error_msg,
        }
        self.failures.append(fail_entry)
        self.log(f"[ERROR] Task failed: {task_key}: {error_msg}")

    def get_status_summary(self) -> str:
        """Format human-readable progress report for --status."""
        if not self.journal_path.exists():
            return "No active execution journal found. Run with --signals-only or --resume to start."

        try:
            with open(self.journal_path, "r", encoding="utf-8") as f:
                doc = json.load(f)

            lines = [
                "============================================================",
                "PHASE 7 FALSIFICATION RUNTIME STATUS",
                "============================================================",
                f"Protocol SHA:    {doc.get('protocol_sha', 'unknown')[:16]}...",
                f"Signal rows:     {doc.get('completed_rows', 0)} / {doc.get('total_rows', 4500)}",
                f"Percent:         {doc.get('percent_complete', 0.0)}%",
                f"Dataset-folds:   {doc.get('completed_tasks', 0)} / {doc.get('total_tasks', 60)}",
                f"Active task:     {doc.get('active_task', 'none')}",
                f"Throughput:      {doc.get('rows_per_hour', 0.0)} rows/hour",
                f"Elapsed:         {doc.get('elapsed_seconds', 0.0):.1f}s",
                f"Est. remaining:  {doc.get('estimated_remaining_seconds', 0.0):.1f}s",
                f"Est. completion: {doc.get('estimated_completion_time', 'unknown')}",
                f"Failures:        {doc.get('failures_count', 0)}",
                "============================================================",
            ]
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading progress journal: {e}"