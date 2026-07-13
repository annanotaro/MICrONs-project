"""
Q3 Step 1b: Behavioral regression (Cleaning).
Subtracts treadmill and pupil effects from trial-mean responses.

Outputs:
  q3/results/<session>/features_q3_clean_<session>.npz
"""

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path

from config import (
    AREAS,
    CHOSEN_SESSION,
    DATA_PATH,
    FRAMES_TO_DROP,
    RESULTS_DIR,
    ensure_results_dir,
    get_reader_class,
    load_trials,
)

ensure_results_dir()
print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")

MicronsReader = get_reader_class()
trials_df = load_trials()

# -------------------------
# Load all trial data for regression
# -------------------------
print("Loading session data for behavioral regression...")
all_responses = []
all_pupil = []
all_treadmill = []
trial_boundaries = []  # (start_frame, end_frame) per trial in concat coordinates

with MicronsReader(DATA_PATH) as reader:
    cursor = 0
    for trial_idx in trials_df["trial_idx"]:
        t = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        r = t["responses"]                 # (n_neurons, T)
        p = t["pupil"]                     # (4, T)
        tm = t["treadmill"].squeeze()      # (T,)
        
        all_responses.append(r)
        all_pupil.append(p)
        all_treadmill.append(tm)
        trial_boundaries.append((cursor, cursor + r.shape[1]))
        cursor += r.shape[1]

# Concatenate for session-wide fit
R = np.concatenate(all_responses, axis=1)   # (n_neurons, total_frames)
P = np.concatenate(all_pupil, axis=1).T     # (total_frames, 4)
T = np.concatenate(all_treadmill)[:, None]  # (total_frames, 1)
behavior = np.hstack([P, T])                # (total_frames, 5)

# -------------------------
# Behavioral Regression
# -------------------------
print(f"Fitting linear regression (behavior {behavior.shape} -> responses {R.shape})...")

# Filter out NaNs
valid = ~(np.isnan(behavior).any(axis=1) | np.isnan(R).any(axis=0))
print(f"  Valid timepoints for fit: {valid.sum()} / {len(valid)}")

lr = LinearRegression()
lr.fit(behavior[valid], R[:, valid].T)

# Predict and subtract
# Fill behavior NaNs with mean for prediction safety
behavior_filled = behavior.copy()
col_means = np.nanmean(behavior, axis=0)
for j in range(behavior.shape[1]):
    mask = np.isnan(behavior_filled[:, j])
    behavior_filled[mask, j] = col_means[j]

R_predicted = lr.predict(behavior_filled).T
R_clean = R - R_predicted
R_clean = np.where(np.isnan(R), np.nan, R_clean) # Keep original NaN pattern

# -------------------------
# Rebuild trial-mean cleaned features
# -------------------------
print("Extracting cleaned trial-mean features per area...")
area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        area_indices[area] = f[f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"][:]

X_clean = {area: np.zeros((len(trials_df), len(idx)), dtype=np.float32)
           for area, idx in area_indices.items()}

for i, (start, end) in enumerate(trial_boundaries):
    trial_slice = R_clean[:, start + FRAMES_TO_DROP : end]
    trial_mean = np.nanmean(trial_slice, axis=1)
    for area, idx in area_indices.items():
        X_clean[area][i, :] = trial_mean[idx]

# -------------------------
# Save
# -------------------------
out_path = RESULTS_DIR / f"features_q3_clean_{CHOSEN_SESSION}.npz"
np.savez_compressed(
    out_path,
    X_V1=X_clean["V1"],
    X_LM=X_clean["LM"],
    X_AL=X_clean["AL"],
    X_RL=X_clean["RL"],
    y_label=trials_df["label"].values,
    y_natural=trials_df["is_natural"].values.astype(int),
    y_origin=trials_df["origin"].values,
    y_naturalness_group=trials_df["naturalness_group"].values,
    trial_idx=trials_df["trial_idx"].values,
    hash=trials_df["hash"].values,
)

print(f"\nSaved to {out_path}")
