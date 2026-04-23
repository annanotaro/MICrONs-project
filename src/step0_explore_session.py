import sys
import importlib.util
import h5py
import pandas as pd
from collections import Counter

import os
READER_PATH = os.environ.get(
    "MICRONS_READER_PATH",
    r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py"
)
DATA_PATH = os.environ.get(
    "MICRONS_DATA_PATH",
    r"C:\data\microns\microns.h5"
)

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

import os
import sys
from pathlib import Path

if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

# All outputs go to results/<session>/
RESULTS_DIR = Path(__file__).parent.parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "csv").mkdir(exist_ok=True)

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")

with MicronsReader(DATA_PATH) as reader:
    hashes = reader.get_hashes_by_session(CHOSEN_SESSION)
    print(f"Session {CHOSEN_SESSION}: {len(hashes)} trials")

    # Use the reader's own encoder to build H5 paths correctly
    stim_info = []
    missing = 0
    for trial_idx, h in enumerate(hashes):
        h_key = reader._encode_hash(h)
        video_path = f"videos/{h_key}"
        if video_path not in reader.f:
            missing += 1
            stim_info.append((trial_idx, h, "UNKNOWN", "UNKNOWN", "UNKNOWN"))
            continue
        attrs = dict(reader.f[video_path].attrs)
        stim_info.append((
            trial_idx,
            h,
            attrs.get("type", "UNKNOWN"),
            attrs.get("short_movie_name", "UNKNOWN"),
            attrs.get("movie_name", "UNKNOWN"),
        ))

    print(f"Missing video entries: {missing}")

trials_df = pd.DataFrame(
    stim_info,
    columns=["trial_idx", "hash", "type", "short_name", "movie_name"]
)

print("\nCounts by type:")
print(trials_df["type"].value_counts())

print("\nCounts by short_movie_name:")
print(trials_df["short_name"].value_counts())

print("\nUnique clips per short_movie_name:")
for name, group in trials_df.groupby("short_name"):
    n_unique = group["hash"].nunique()
    n_trials = len(group)
    print(f"  {name:15s}  {n_unique:3d} unique clips, {n_trials:3d} trials")

# Save for use in Step 1
trials_df.to_csv(RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv", index=False)
print(f"\nSaved to {RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"}")

def unified_label(row):
    if row["type"] == "Clip":
        return row["short_name"]
    return row["type"]

trials_df["label"] = trials_df.apply(unified_label, axis=1)

# Normalize casing — "sports1m" should match the others stylistically
trials_df["label"] = trials_df["label"].replace({"sports1m": "Sports1M"})

# Binary natural/parametric flag for Q1a
trials_df["is_natural"] = trials_df["label"].isin(["Cinematic", "Sports1M", "Rendered"])

print("\nFinal label distribution:")
print(trials_df["label"].value_counts())
print("\nNatural vs parametric:")
print(trials_df["is_natural"].value_counts())

trials_df.to_csv(RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv", index=False)