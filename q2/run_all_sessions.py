"""
Run the full Q2 pipeline across multiple sessions.

Usage:
    python run_all_sessions.py                    # all viable sessions
    python run_all_sessions.py 7_4 5_6            # specific sessions
    python run_all_sessions.py --clean-only       # skip raw decode, run clean only
    python run_all_sessions.py --skip-pool        # don't run step4 at the end

A session is considered viable if AL has at least MIN_AL_NEURONS neurons.
"""
import subprocess
import sys
import time
import argparse
from pathlib import Path

PYTHON = sys.executable

import h5py

# -------------------------
# ARGS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("sessions", nargs="*",
                    help="Sessions to run (default: all viable)")
parser.add_argument("--clean-only", action="store_true",
                    help="Skip step1/step2 raw; only run step1b and step2 --clean")
parser.add_argument("--skip-pool", action="store_true")
args = parser.parse_args()

# -------------------------
# CONFIG
# -------------------------
DATA_PATH    = Path("/Users/bea/microns_decoding/data/1621/raw/microns.h5")
Q2_DIR       = Path(__file__).parent
MIN_AL_NEURONS = 200

# -------------------------
# DECIDE SESSIONS
# -------------------------
if args.sessions:
    sessions = args.sessions
    print(f"User-specified sessions: {sessions}")
else:
    with h5py.File(DATA_PATH, "r") as f:
        all_sessions = sorted(f["sessions"].keys())
        sessions = []
        for sess in all_sessions:
            al_path = f"sessions/{sess}/meta/area_indices/AL"
            if al_path in f and f[al_path].shape[0] >= MIN_AL_NEURONS:
                sessions.append(sess)
            else:
                n = f[al_path].shape[0] if al_path in f else 0
                print(f"Skipping {sess}: only {n} AL neurons")
    print(f"\nViable sessions ({len(sessions)}): {sessions}")

# -------------------------
# PIPELINE STEPS
# -------------------------
def run(script, *extra_args):
    cmd = [PYTHON, str(Q2_DIR / script)] + list(extra_args)
    print(f"    $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0

overall_start = time.time()

for i, sess in enumerate(sessions):
    print(f"\n{'='*65}")
    print(f"SESSION {i+1}/{len(sessions)}: {sess}")
    print(f"{'='*65}")
    sess_start = time.time()
    ok = True

    # Step 0: trial labels (always needed)
    print("\n  [step0] Building trial label table...")
    ok = run("step0_trial_labels.py", sess)
    if not ok:
        print(f"  step0 failed for {sess}, skipping.")
        continue

    if not args.clean_only:
        # Step 1: raw time-resolved features
        print("\n  [step1] Extracting raw features...")
        ok = run("step1_features.py", sess)
        if not ok:
            print(f"  step1 failed for {sess}, skipping.")
            continue

        # Step 2: decode (raw) — LR + SVM
        print("\n  [step2] Decoding (raw)...")
        ok = run("step2_decode.py", sess)
        if not ok:
            print(f"  step2 (raw) failed for {sess}.")

    # Step 1b: behavioral regression features
    print("\n  [step1b] Behavioral regression...")
    ok = run("step1b_features_clean.py", sess)
    if not ok:
        print(f"  step1b failed for {sess}, skipping clean decode.")
        continue

    # Step 2 clean: decode (clean) — LR + SVM
    print("\n  [step2 --clean] Decoding (clean)...")
    ok = run("step2_decode.py", sess, "--clean")
    if not ok:
        print(f"  step2 (clean) failed for {sess}.")

    elapsed = time.time() - sess_start
    print(f"\n  Session {sess} done in {elapsed/60:.1f} min")

# -------------------------
# POOL RESULTS
# -------------------------
if not args.skip_pool:
    print(f"\n{'='*65}")
    print("Pooling results across sessions...")
    run("step4_pool_sessions.py", "--both")

total = time.time() - overall_start
print(f"\n{'='*65}")
print(f"All done. Total runtime: {total/3600:.1f} hours")
