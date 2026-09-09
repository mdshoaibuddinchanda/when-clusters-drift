"""Run and capture the exact Hypothesis-Y1 validation suites with the active P12 interpreter."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/hypothesis_y1"
LOGS = BASE / "logs"
RESULTS = BASE / "artifacts/hypothesis_y1_test_results.json"
EXPECTED_PYTHON = Path(r"D:\Conda\p12\python.exe")
SUITES = {
    "y1": ("experiments/hypothesis_y1/tests/", LOGS / "hypothesis_y1_pytest_y1.txt"),
    "falsification": ("tests/test_falsification/", LOGS / "hypothesis_y1_pytest_falsification.txt"),
    "full": ("tests/", LOGS / "hypothesis_y1_pytest_full.txt"),
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def count(pattern: str, text: str) -> int:
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else 0


def parse_pytest(text: str, exit_code: int, wall_seconds: float) -> dict[str, object]:
    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    passed = count(r"(\d+)\s+passed", clean)
    failed = count(r"(\d+)\s+failed", clean)
    skipped = count(r"(\d+)\s+skipped", clean)
    errors = count(r"(\d+)\s+errors?", clean)
    xfailed = count(r"(\d+)\s+xfailed", clean)
    xpassed = count(r"(\d+)\s+xpassed", clean)
    deselected = count(r"(\d+)\s+deselected", clean)
    collected_matches = re.findall(r"collected\s+(\d+)\s+items?", clean, flags=re.IGNORECASE)
    collected = int(collected_matches[-1]) if collected_matches else passed + failed + skipped + errors + xfailed + xpassed
    duration_matches = re.findall(r"\bin\s+([0-9.]+)s\b", clean)
    pytest_duration = float(duration_matches[-1]) if duration_matches else None
    return {
        "collected": collected,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "deselected": deselected,
        "exit_code": int(exit_code),
        "duration": pytest_duration if pytest_duration is not None else wall_seconds,
        "wall_duration_seconds": wall_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    args = parser.parse_args()
    actual = Path(sys.executable).resolve()
    if actual != EXPECTED_PYTHON.resolve():
        raise RuntimeError(f"Y1 tests must use {EXPECTED_PYTHON}; active interpreter is {actual}")
    target, log_path = SUITES[args.suite]
    command = [str(actual), "-m", "pytest", target, "-q"]
    started = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - started
    combined = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
    header = (
        f"suite: {args.suite}\n"
        f"command: {' '.join(command)}\n"
        f"working_directory: {ROOT}\n"
        f"exit_code: {proc.returncode}\n"
        f"wall_duration_seconds: {wall:.6f}\n\n"
    )
    atomic_text(log_path, header + combined)
    record = parse_pytest(combined, proc.returncode, wall)
    record.update({
        "command": command,
        "executable": str(actual),
        "log": log_path.relative_to(ROOT).as_posix(),
    })
    existing = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {
        "audit": "hypothesis_y1_test_results",
        "required_interpreter": str(EXPECTED_PYTHON),
        "suites": {},
    }
    existing["suites"][args.suite] = record
    atomic_text(RESULTS, json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"suite": args.suite, **record}, indent=2))
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
