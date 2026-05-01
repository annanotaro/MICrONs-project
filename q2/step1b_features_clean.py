"""
Behavioral regression for Q2 time-resolved features.

Fits a linear model (pupil + treadmill -> responses) across all trials in the
session, takes the residuals, then rebuilds the per-trial time-resolved arrays
(trials x neurons x time) for the 3 natural classes only.

Mirrors Q1's step1b_behavioral_clean.py but keeps the temporal dimension intact.
"""
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression

# -------------------------
# CONFIG
# -------------------------
CHOSEN_SESSION = sys.argv[1] if len(sys.argv) > 1 else "7_4"

DATA_PATH = Path("/Users/bea/microns_decoding/data/1621/raw/microns.h5")
RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AREAS = ["V1", "LM", "AL", "RL"]
FRAMES_TO_DROP = 3

print(f"Session:  {CHOSEN_SESSION}")
print(f"Outputs -> {RESULTS_DIR}")

# -------------------------
# LOAD ALL TRIALS (all labels, for a more robust regression fit)
# -------------------------
trials_path = RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"
trials_df_all = pd.read_csv(trials_path)

all_responses = []
all_pupil = []
all_treadmill = []
trial_boundaries = []   # (start, end) in concatenated frame coordinates

print(f"\nLoading all {len(trials_df_all)} trials for regression fit...")

with h5py.File(DATA_PATH, "r") as f:
    cursor = 0
    for row in trials_df_all.itertuples():
        grp = f[f"sessions/{CHOSEN_SESSION}/trials/{int(row.trial_idx)}"]
        r  = grp["responses"][:]           # (n_neurons, T)
        p  = grp["pupil"][:]               # (4, T)
        tm = grp["treadmill"][:].squeeze() # (T,)
        all_responses.append(r)
        all_pupil.append(p)
        all_treadmill.append(tm)
        trial_boundaries.append((cursor, cursor + r.shape[1]))
        cursor += r.shape[1]

# -------------------------
# CONCATENATE
# -------------------------
R = np.concatenate(all_responses, axis=1)    # (n_neurons, total_T)
P = np.concatenate(all_pupil, axis=1).T      # (total_T, 4)
TM = np.concatenate(all_treadmill)[:, None]  # (total_T, 1)
behavior = np.hstack([P, TM])                # (total_T, 5)

print(f"\nConcatenated: responses {R.shape}, behavior {behavior.shape}")

# -------------------------
# NaN HANDLING
# -------------------------
print("\nNaN check:")
print(f"  responses: {np.isnan(R).sum()} NaNs")
print(f"  behavior:  {np.isnan(behavior).sum()} NaNs")

valid = ~(np.isnan(behavior).any(axis=1) | np.isnan(R).any(axis=0))
print(f"  valid timepoints for fit: {valid.sum()} / {len(valid)}")

# -------------------------
# FIT REGRESSION ON CLEAN ROWS
# -------------------------
lr = LinearRegression()
lr.fit(behavior[valid], R[:, valid].T)   # (valid_T, n_neurons)

# Fill NaN rows in behavior with column means so prediction covers all timepoints
behavior_filled = behavior.copy()
col_means = np.nanmean(behavior, axis=0)
for j in range(behavior.shape[1]):
    mask = np.isnan(behavior_filled[:, j])
    behavior_filled[mask, j] = col_means[j]

R_pred = lr.predict(behavior_filled).T   # (n_neurons, total_T)
R_clean = R - R_pred
R_clean = np.where(np.isnan(R), np.nan, R_clean)  # preserve original NaN pattern

print("Regression done.")

# -------------------------
# LOAD AREA INDICES
# -------------------------
area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        area_indices[area] = f[f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"][:]

# -------------------------
# REBUILD TIME-RESOLVED ARRAYS FOR NATURAL TRIALS ONLY
# -------------------------
trials_df_all["orig_idx"] = np.arange(len(trials_df_all))
natural_mask = trials_df_all["label"].isin(["Cinematic", "Sports1M", "Rendered"])
trials_nat = trials_df_all[natural_mask].reset_index(drop=True)

print(f"\nRebuilding time-resolved arrays for {len(trials_nat)} natural trials...")

# Determine minimum n_time across natural trials (after dropping onset frames)
n_times = []
for row in trials_nat.itertuples():
    start, end = trial_boundaries[row.orig_idx]
    n_times.append((end - start) - FRAMES_TO_DROP)
n_time = min(n_times)
print(f"  n_time = {n_time} (min across natural trials)")

n_trials = len(trials_nat)
X_clean = {area: np.zeros((n_trials, len(idx), n_time), dtype=np.float32)
           for area, idx in area_indices.items()}

for i, row in enumerate(trials_nat.itertuples()):
    start, end = trial_boundaries[row.orig_idx]
    # Drop onset frames, then take exactly n_time frames
    ts = R_clean[:, start + FRAMES_TO_DROP : start + FRAMES_TO_DROP + n_time]
    for area, idx in area_indices.items():
        X_clean[area][i] = ts[idx]

# -------------------------
# SANITY CHECKS
# -------------------------
print("\nShapes:")
for area in AREAS:
    n_nan = np.isnan(X_clean[area]).sum()
    print(f"  {area}: {X_clean[area].shape}   NaNs: {n_nan}")

# -------------------------
# SAVE
# -------------------------
y = trials_nat["label"].values
groups = trials_nat["hash"].values

out_path = RESULTS_DIR / f"q2_features_clean_{CHOSEN_SESSION}.npz"
np.savez_compressed(
    out_path,
    X_V1=X_clean["V1"], X_LM=X_clean["LM"],
    X_AL=X_clean["AL"], X_RL=X_clean["RL"],
    y=y, groups=groups
)

print(f"\nSaved to {out_path}")
print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")
