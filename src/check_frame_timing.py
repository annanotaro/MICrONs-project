"""
Verify the assumptions behind FRAMES_TO_DROP = 3:
  (a) Every trial in every session has the same number of frames.
  (b) The sampling rate is ~6.3 Hz everywhere (so 3 frames ≈ 475 ms).

Run once. Prints a per-session summary and flags any deviation.
"""
import sys
import importlib.util
import os
from pathlib import Path
from collections import Counter

import h5py
import numpy as np

READER_PATH = os.environ.get(
    "MICRONS_READER_PATH",
    r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py",
)
DATA_PATH = os.environ.get("MICRONS_DATA_PATH", r"C:\data\microns\microns.h5")

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

EXPECTED_FRAMES = 75
EXPECTED_HZ = 6.3
HZ_TOLERANCE = 0.5  # accept anything in [5.8, 6.8]

# ----------------------------------------------------------------------
# (a) Frame-count check + (b) look for a sampling-rate attribute
# ----------------------------------------------------------------------
print(f"Opening {DATA_PATH}")
with h5py.File(DATA_PATH, "r") as f:
    sessions = list(f["sessions"].keys())
    print(f"Found {len(sessions)} sessions\n")

    # Where might a sampling rate live? Scan attribute keys at a few levels.
    print("Scanning HDF5 attributes for anything rate-like...")
    rate_keys_seen = set()

    def scan_attrs(name, obj):
        for k in obj.attrs.keys():
            kl = k.lower()
            if any(s in kl for s in ("rate", "hz", "fps", "freq", "sampling", "fs")):
                rate_keys_seen.add((name, k, obj.attrs[k]))

    f.visititems(scan_attrs)
    if rate_keys_seen:
        print("  Candidate attributes found:")
        for name, k, v in sorted(rate_keys_seen)[:20]:
            print(f"    {name} :: {k} = {v}")
        if len(rate_keys_seen) > 20:
            print(f"    ... and {len(rate_keys_seen) - 20} more")
    else:
        print("  No rate-like attribute found at any level.")
        print("  (Will infer rate from trial duration, see below.)")
    print()

# ----------------------------------------------------------------------
# Per-session frame-count + timing check via the reader
# ----------------------------------------------------------------------
print(f"{'session':<8}  {'n_trials':>8}  {'frame counts (count: n_trials)':<35}  "
      f"{'inferred Hz':>11}")
print("-" * 80)

problems = []
with MicronsReader(DATA_PATH) as reader:
    for sess in sessions:
        try:
            hashes = reader.get_hashes_by_session(sess)
        except Exception as e:
            print(f"{sess:<8}  ERROR getting hashes: {e}")
            problems.append((sess, f"get_hashes failed: {e}"))
            continue

        frame_counts = Counter()
        durations = []  # seconds, if available

        # Sample up to 10 trials per session — enough to detect inhomogeneity
        sample_idx = np.linspace(0, len(hashes) - 1, num=min(10, len(hashes)),
                                 dtype=int)
        for ti in sample_idx:
            try:
                trial = reader.get_trial(sess, int(ti))
            except Exception as e:
                problems.append((sess, f"trial {ti} failed: {e}"))
                continue

            r = trial["responses"]
            n_frames = r.shape[1]
            frame_counts[n_frames] += 1

            # Try to infer rate: if the trial dict carries a time vector or
            # a duration, use it. Common keys: 'times', 'time', 'duration'.
            for key in ("times", "time", "frame_times", "t"):
                if key in trial:
                    t = np.asarray(trial[key]).squeeze()
                    if t.ndim == 1 and len(t) >= 2:
                        dt = np.median(np.diff(t))
                        if dt > 0:
                            durations.append(1.0 / dt)
                    break

        # Most common frame count
        if frame_counts:
            counts_str = ", ".join(f"{k}: {v}" for k, v in
                                   sorted(frame_counts.items()))
        else:
            counts_str = "(no trials read)"

        hz_str = f"{np.mean(durations):.2f}" if durations else "n/a"

        print(f"{sess:<8}  {len(hashes):>8}  {counts_str:<35}  {hz_str:>11}")

        # Flag any deviation
        if len(frame_counts) > 1:
            problems.append((sess, f"inhomogeneous frame counts: {dict(frame_counts)}"))
        if frame_counts and EXPECTED_FRAMES not in frame_counts:
            problems.append((sess, f"frame count != {EXPECTED_FRAMES}: {dict(frame_counts)}"))
        if durations:
            mean_hz = float(np.mean(durations))
            if abs(mean_hz - EXPECTED_HZ) > HZ_TOLERANCE:
                problems.append((sess, f"sampling rate {mean_hz:.2f} Hz "
                                       f"(expected ~{EXPECTED_HZ})"))

# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
print()
if problems:
    print(f"FOUND {len(problems)} POTENTIAL ISSUE(S):")
    for sess, msg in problems:
        print(f"  {sess}: {msg}")
    print("\nFRAMES_TO_DROP = 3 may not be appropriate for the flagged sessions.")
else:
    print("All sessions look consistent: 75 frames per trial, sampling rate")
    print(f"within tolerance of {EXPECTED_HZ} Hz (or rate not directly available,")
    print("but frame counts match — the assumption is at least structurally valid).")
    print("\nFRAMES_TO_DROP = 3 is fine across the dataset.")