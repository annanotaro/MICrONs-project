import sys, importlib.util
import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

READER_PATH = os.environ.get(
    "MICRONS_READER_PATH",
    r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py"
)
DATA_PATH = os.environ.get(
    "MICRONS_DATA_PATH",
    r"C:\data\microns\microns.h5"
)


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

AREAS = ["V1", "LM", "AL", "RL"]
FRAMES_TO_DROP = 3

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

trials_df = pd.read_csv(RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv")

# Load all trials into memory: responses, pupil, treadmill concatenated along time
all_responses = []
all_pupil = []
all_treadmill = []
trial_boundaries = []  # (start_frame, end_frame) per trial in concat coordinates

with MicronsReader(DATA_PATH) as reader:
    cursor = 0
    for trial_idx in trials_df["trial_idx"]:
        t = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        r = t["responses"]                 # (n_neurons, 75)
        p = t["pupil"]                     # (4, 75) — 4 features, use all
        tm = t["treadmill"].squeeze()      # (75,)
        all_responses.append(r)
        all_pupil.append(p)
        all_treadmill.append(tm)
        trial_boundaries.append((cursor, cursor + r.shape[1]))
        cursor += r.shape[1]

# Concatenate
R = np.concatenate(all_responses, axis=1)   # (n_neurons, total_frames)
P = np.concatenate(all_pupil, axis=1).T     # (total_frames, 4)
T = np.concatenate(all_treadmill)[:, None]  # (total_frames, 1)
behavior = np.hstack([P, T])                # (total_frames, 5)
print(f"Concatenated: responses {R.shape}, behavior {behavior.shape}")

# ---------------------------------------------------------------
# Handle NaNs before regression
# ---------------------------------------------------------------
print("\nNaN check:")
print(f"  responses R:  {np.isnan(R).sum()} NaNs")
print(f"  behavior:     {np.isnan(behavior).sum()} NaNs")
print(f"  rows with any NaN in behavior: {np.isnan(behavior).any(axis=1).sum()}")
print(f"  rows with any NaN in R^T:      {np.isnan(R.T).any(axis=1).sum()}")

# Rows valid for fitting: no NaN in behavior AND no NaN in responses
valid = ~(np.isnan(behavior).any(axis=1) | np.isnan(R).any(axis=0))
print(f"  valid timepoints for fit: {valid.sum()} / {len(valid)}")

# Fit linear regression on clean rows only
lr = LinearRegression()
lr.fit(behavior[valid], R[:, valid].T)   # train on clean subset

# To predict across ALL timepoints (including NaN rows), we need behavior
# with no NaNs. Replace NaN rows with column means so their contribution
# is only the intercept (effectively no behavioral correction where data is missing).
behavior_filled = behavior.copy()
col_means = np.nanmean(behavior, axis=0)
for j in range(behavior.shape[1]):
    mask = np.isnan(behavior_filled[:, j])
    behavior_filled[mask, j] = col_means[j]

R_predicted = lr.predict(behavior_filled).T   # (n_neurons, total_frames)
R_clean = R - R_predicted

# If any neuron has NaN responses in certain frames, keep those NaNs out of
# the subtracted trace (preserves the NaN pattern; trial-mean below uses nanmean).
R_clean = np.where(np.isnan(R), np.nan, R_clean)

# Now rebuild trial-mean features from R_clean
area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        area_indices[area] = f[f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"][:]

n_trials = len(trials_df)
X_clean = {area: np.zeros((n_trials, len(idx)), dtype=np.float32)
           for area, idx in area_indices.items()}

for i, (start, end) in enumerate(trial_boundaries):
    trial_slice = R_clean[:, start + FRAMES_TO_DROP : end]
    trial_mean = np.nanmean(trial_slice, axis=1)
    for area, idx in area_indices.items():
        X_clean[area][i, :] = trial_mean[idx]

# Final sanity check — if any X_clean entries ended up as NaN, something's wrong
for area in AREAS:
    n_nan = np.isnan(X_clean[area]).sum()
    if n_nan > 0:
        print(f"  WARNING: X_clean[{area}] has {n_nan} NaN entries")

# Save
np.savez_compressed(
    RESULTS_DIR / f"features_clean_{CHOSEN_SESSION}.npz",
    X_V1=X_clean["V1"], X_LM=X_clean["LM"],
    X_AL=X_clean["AL"], X_RL=X_clean["RL"],
    y_label=trials_df["label"].values,
    y_natural=trials_df["is_natural"].values.astype(int),
)
print(f"Saved to {RESULTS_DIR / f'features_clean_{CHOSEN_SESSION}.npz'} ")