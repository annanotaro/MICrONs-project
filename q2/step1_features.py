import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MICRONS_DATA_PATH, MICRONS_READER_PATH

if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

READER_PATH = MICRONS_READER_PATH
DATA_PATH = MICRONS_DATA_PATH

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AREAS = ["V1", "LM", "AL", "RL"]
FRAMES_TO_DROP = 3

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs -> {RESULTS_DIR}")

# -------------------------
# LOAD TRIAL LABELS
# -------------------------
trials_path = RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"
trials_df = pd.read_csv(trials_path)

trials_df = trials_df[trials_df["label"].isin(["Cinematic", "Sports1M", "Rendered"])].reset_index(drop=True)

print(f"Loaded {len(trials_df)} natural trials from {trials_path}")
print("\nLabel counts:")
print(trials_df["label"].value_counts())

# -------------------------
# LOAD AREA INDICES + PREALLOCATE
# -------------------------
with h5py.File(DATA_PATH, "r") as f:
    area_indices = {}
    for area in AREAS:
        path = f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"
        area_indices[area] = f[path][:]
        print(f"  {area}: {len(area_indices[area])} neurons")

    # get n_time from first trial
    t0_idx = int(trials_df.iloc[0]["trial_idx"])
    r0 = f[f"sessions/{CHOSEN_SESSION}/trials/{t0_idx}/responses"][:]
    n_time = r0.shape[1] - FRAMES_TO_DROP

n_trials = len(trials_df)
print(f"\nTrials: {n_trials}")
print(f"Timepoints per trial after dropping onset: {n_time}")

X = {
    area: np.zeros((n_trials, len(idx), n_time), dtype=np.float32)
    for area, idx in area_indices.items()
}

# -------------------------
# LOAD RESPONSES
# -------------------------
print("\nLoading time-resolved responses...")

with h5py.File(DATA_PATH, "r") as f:
    for i, row in enumerate(trials_df.itertuples(index=False)):
        responses = f[f"sessions/{CHOSEN_SESSION}/trials/{int(row.trial_idx)}/responses"][:]
        ts = responses[:, FRAMES_TO_DROP:]

        for area, idx in area_indices.items():
            X[area][i] = ts[idx, :n_time]

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
