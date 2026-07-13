"""
Q3.2 Step 5b: Behavioral regression for Time-Resolved features.
Subtracts treadmill and pupil effects from time-varying responses.

Outputs:
  q3/results/<session>/q3_time_features_clean_<session>.npz
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
# Load session data for regression
# -------------------------
print("Loading session data for behavioral regression...")
all_responses = []
all_pupil = []
all_treadmill = []
trial_info = [] # (n_frames) per trial

with MicronsReader(DATA_PATH) as reader:
    for trial_idx in trials_df["trial_idx"]:
        t = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        r = t["responses"]
        p = t["pupil"]
        tm = t["treadmill"].squeeze()
        
        all_responses.append(r)
        all_pupil.append(p)
        all_treadmill.append(tm)
        trial_info.append(r.shape[1])

# Concatenate for session-wide fit
R = np.concatenate(all_responses, axis=1)
P = np.concatenate(all_pupil, axis=1).T
T = np.concatenate(all_treadmill)[:, None]
behavior = np.hstack([P, T])

# -------------------------
# Behavioral Regression
# -------------------------
print("Fitting behavioral regression...")
valid = ~(np.isnan(behavior).any(axis=1) | np.isnan(R).any(axis=0))
lr = LinearRegression()
lr.fit(behavior[valid], R[:, valid].T)

behavior_filled = behavior.copy()
col_means = np.nanmean(behavior, axis=0)
for j in range(behavior.shape[1]):
    mask = np.isnan(behavior_filled[:, j])
    behavior_filled[mask, j] = col_means[j]

R_predicted = lr.predict(behavior_filled).T
R_clean = R - R_predicted
R_clean = np.where(np.isnan(R), np.nan, R_clean)

# -------------------------
# Extract cleaned time-resolved trials
# -------------------------
print("Extracting cleaned time-resolved trials...")
n_time = min(trial_info) - FRAMES_TO_DROP
n_trials = len(trials_df)

area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        area_indices[area] = f[f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"][:]

X_clean = {
    area: np.zeros((n_trials, len(idx), n_time), dtype=np.float32)
    for area, idx in area_indices.items()
}

cursor = 0
for i, length in enumerate(trial_info):
    trial_slice = R_clean[:, cursor + FRAMES_TO_DROP : cursor + FRAMES_TO_DROP + n_time]
    for area, idx in area_indices.items():
        X_clean[area][i] = trial_slice[idx]
    cursor += length

# -------------------------
# Recompute Stability (CV) on cleaned traces
# -------------------------
print("Recomputing stability (CV) on cleaned traces...")
MEAN_FLOOR = 1e-10
stability = {}

for area in AREAS:
    neuron_mean = X_clean[area].mean(axis=2)
    neuron_std = X_clean[area].std(axis=2)
    neuron_mean_safe = np.where(np.abs(neuron_mean) < MEAN_FLOOR, MEAN_FLOOR, neuron_mean)
    neuron_cv = neuron_std / np.abs(neuron_mean_safe)
    stability[area] = neuron_cv.mean(axis=1)

# -------------------------
# Save
# -------------------------
out_path = RESULTS_DIR / f"q3_time_features_clean_{CHOSEN_SESSION}.npz"
save_dict = {
    "y_label": trials_df["label"].values,
    "y_natural": trials_df["is_natural"].values.astype(int),
    "y_origin": trials_df["origin"].values,
    "y_naturalness_group": trials_df["naturalness_group"].values,
    "trial_idx": trials_df["trial_idx"].values,
    "hash": trials_df["hash"].values,
}
for area in AREAS:
    save_dict[f"X_{area}"] = X_clean[area]
    save_dict[f"stability_{area}"] = stability[area]

np.savez_compressed(out_path, **save_dict)
print(f"\nSaved to {out_path}")
