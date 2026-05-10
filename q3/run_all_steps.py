"""
Run the full Q3 pipeline for a single session.

Usage:
    python q3/run_all_steps.py 7_4
    python q3/run_all_steps.py --skip-time
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Add project root to path so we can import config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from q3.config import RESULTS_DIR, ensure_results_dir, CHOSEN_SESSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "session",
        nargs="?",
        default=CHOSEN_SESSION,
        help=f"Session id (default: {CHOSEN_SESSION})",
    )
    parser.add_argument(
        "--skip-time",
        action="store_true",
        help="Run only Q3 core steps (0-4), skip Q3.2 time steps (5-7).",
    )
    return parser.parse_args()


def run_step(script_path: Path, session: str, log_file) -> bool:
    cmd = [sys.executable, str(script_path), session]
    header = f"\n→ Running {script_path.name}\n  $ {' '.join(cmd)}\n"
    print(header, end="", flush=True)
    log_file.write(header)
    log_file.flush()

    start = time.time()
    # Stream output to both terminal and log file
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        if proc.stdout:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
        proc.wait()
        returncode = proc.returncode

    elapsed = time.time() - start
    status = (
        f"  OK ({elapsed/60:.1f} min)\n"
        if returncode == 0
        else f"  FAILED ({elapsed/60:.1f} min)\n"
    )
    print(status, end="", flush=True)
    log_file.write(status)
    log_file.flush()

    return returncode == 0


def main() -> int:
    args = parse_args()
    q3_dir = Path(__file__).parent

    # Ensure the session-specific results directory exists
    # We override CHOSEN_SESSION globally via env var to match args.session
    import os

    os.environ["CHOSEN_SESSION"] = args.session
    ensure_results_dir()

    log_path = RESULTS_DIR / "terminal.log"

    steps = [
        "step0_labels.py",
        "step1_features.py",
        "step1b_features_clean.py",
        "step2_natural_vs_parametric.py",
        "step3_natural_subgroups.py",
        "step4_plots.py",
    ]
    if not args.skip_time:
        steps.extend([
            "step5_time_features.py",
            "step5b_time_features_clean.py",
            "step6_time_decode.py",
            "step7_time_plots.py",
        ])

    with open(log_path, "w", encoding="utf-8") as log_file:

        def log_print(msg):
            print(msg, flush=True)
            log_file.write(msg + "\n")
            log_file.flush()

        log_print("=" * 70)
        log_print(f"Q3 pipeline session: {args.session}")
        log_print(f"Steps: {', '.join(steps)}")
        log_print(f"Log file: {log_path}")
        log_print("=" * 70)

        pipeline_start = time.time()
        for step in steps:
            ok = run_step(q3_dir / step, args.session, log_file)
            if not ok:
                log_print(f"\nStopping pipeline at {step}.")
                return 1

        total = time.time() - pipeline_start
        log_print("\n" + "=" * 70)
        log_print(
            f"Q3 pipeline completed for session {args.session} in {total/60:.1f} min"
        )
        log_print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
