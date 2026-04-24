import os
import sys
import importlib.util
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

READER_PATH = "/Users/gaiagr/.cache/huggingface/hub/datasets--NeuroBLab--MICrONS/snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03/reader.py"
DATA_PATH = "/Users/gaiagr/.cache/huggingface/hub/datasets--NeuroBLab--MICrONS/snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03/microns.h5"

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AREAS = ["V1", "LM", "AL", "RL"]
FRAMES_TO_DROP = 3

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs -> {RESULTS_DIR}")

# -------------------------
# LOAD READER
# -------------------------
spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

# -------------------------
# LOAD TRIAL LABELS
# -------------------------
trials_path = RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"
trials_df = pd.read_csv(trials_path)

# Keep only the 3 natural classes for Q2
trials_df = trials_df[trials_df["label"].isin(["Cinematic", "Sports1M", "Rendered"])].reset_index(drop=True)

print(f"Loaded {len(trials_df)} natural trials from {trials_path}")
print("\nLabel counts:")
print(trials_df["label"].value_counts())

# -------------------------
# LOAD AREA INDICES
# -------------------------
area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        path = f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"
        area_indices[area] = f[path][:]
        print(f"  {area}: {len(area_indices[area])} neurons")

n_trials = len(trials_df)

# Determine number of timepoints after dropping onset frames
with MicronsReader(DATA_PATH) as reader:
    example_trial = reader.get_trial(CHOSEN_SESSION, int(trials_df.iloc[0]["trial_idx"]))
    n_time = example_trial["responses"].shape[1] - FRAMES_TO_DROP

print(f"\nTrials: {n_trials}")
print(f"Timepoints per trial after dropping onset: {n_time}")

# -------------------------
# PREALLOCATE ARRAYS
# -------------------------
X = {
    area: np.zeros((n_trials, len(idx), n_time), dtype=np.float32)
    for area, idx in area_indices.items()
}

# -------------------------
# LOAD RESPONSES
# -------------------------
print("\nLoading time-resolved responses...")

with MicronsReader(DATA_PATH) as reader:
    for i, row in enumerate(trials_df.itertuples(index=False)):
        trial = reader.get_trial(CHOSEN_SESSION, int(row.trial_idx))
        responses = trial["responses"]  # (n_neurons, n_time)

        ts = responses[:, FRAMES_TO_DROP:]

        for area, idx in area_indices.items():
            X[area][i] = ts[idx]

        if (i + 1) % 50 == 0 or i == n_trials - 1:
            print(f"  {i+1}/{n_trials}")

# -------------------------
# BUILD LABELS / GROUPS
# -------------------------
y = trials_df["label"].values
groups = trials_df["hash"].values

# -------------------------
# SANITY CHECKS
# -------------------------
print("\nShapes:")
for area in AREAS:
    print(f"{area}: {X[area].shape}")

unique_labels, counts = np.unique(y, return_counts=True)
print("\nLabel counts:")
for lab, cnt in zip(unique_labels, counts):
    print(f"{lab}: {cnt}")

assert len(unique_labels) == 3, "Expected exactly 3 classes"
print(f"\nUnique groups (hashes): {len(np.unique(groups))}")

# -------------------------
# SAVE
# -------------------------
out_path = RESULTS_DIR / f"q2_features_{CHOSEN_SESSION}.npz"

np.savez_compressed(
    out_path,
    X_V1=X["V1"],
    X_LM=X["LM"],
    X_AL=X["AL"],
    X_RL=X["RL"],
    y=y,
    groups=groups
)

print(f"\nSaved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")