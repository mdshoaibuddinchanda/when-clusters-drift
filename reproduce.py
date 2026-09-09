"""Entry point for reproducing the Phase-7 falsification results and Hypothesis 1 verdict."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the Phase-7 falsification experiment and Hypothesis 1 audit verdict."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all models and bootstrap intervals from scratch instead of summarizing existing predictions.",
    )
    args, unknown = parser.parse_known_args()

    script = ROOT / "experiments" / "hypothesis_y1" / "scripts" / "hypothesis_y1_independent_phase7_replication.py"
    cmd = [sys.executable, str(script)]
    if not args.recompute:
        cmd.append("--summarize-existing")
    cmd.extend(unknown)

    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
