"""
Run the full Q1 pipeline across multiple MICrONS sessions.

Usage:
    python run_all_sessions.py                    # runs all viable sessions
    python run_all_sessions.py 7_4 5_6            # runs only the listed sessions

Each session produces a separate subdirectory under ../results/<session>/.
"""
import subprocess
import sys
import time
from pathlib import Path

import h5py

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_PATH = r"C:\data\microns\microns.h5"
MIN_AL_NEURONS = 200   # skip sessions with too-small AL populations

SCRIPTS = [
    "step0_explore_session.py",
    "step1_features.py",
    "step1b_behavioral_clean.py",
    "step4_learning_curves.py",
    "step4_learning_curves_CLEAN.py",
    "step5_confusion.py",
    "step5_confusion_CLEAN.py",
]

# ------------------------------------------------------------------
# Decide which sessions to run
# ------------------------------------------------------------------
if len(sys.argv) > 1:
    sessions_to_run = sys.argv[1:]
    print(f"User-specified sessions: {sessions_to_run}")
else:
    with h5py.File(DATA_PATH, "r") as f:
        all_sessions = list(f["sessions"].keys())
        sessions_to_run = []
        for sess in all_sessions:
            al_path = f"sessions/{sess}/meta/area_indices/AL"
            if al_path in f:
                n_al = f[al_path].shape[0]
                if n_al >= MIN_AL_NEURONS:
                    sessions_to_run.append(sess)
                else:
                    print(f"Skipping {sess}: only {n_al} AL neurons")
    print(f"\nViable sessions ({len(sessions_to_run)}): {sessions_to_run}")

# ------------------------------------------------------------------
# Loop
# ------------------------------------------------------------------
src_dir = Path(__file__).parent
overall_start = time.time()

for i, sess in enumerate(sessions_to_run):
    print(f"\n{'='*70}")
    print(f"SESSION {i+1}/{len(sessions_to_run)}: {sess}")
    print(f"{'='*70}")
    sess_start = time.time()

    for script in SCRIPTS:
        script_path = src_dir / script
        print(f"\n  → {script}")
        script_start = time.time()

        result = subprocess.run(
            ["python", str(script_path), sess],
            capture_output=False,     # show output live
            text=True,
        )

        elapsed = time.time() - script_start
        if result.returncode != 0:
            print(f"  ERROR: {script} failed for session {sess}. "
                  f"Skipping remaining steps.")
            break
        print(f"  ✓ {script} completed in {elapsed/60:.1f} min")

    sess_elapsed = time.time() - sess_start
    print(f"\nSession {sess} total: {sess_elapsed/60:.1f} min")

total_elapsed = time.time() - overall_start
print(f"\n{'='*70}")
print(f"All sessions complete. Total runtime: {total_elapsed/3600:.1f} hours")
print(f"{'='*70}")