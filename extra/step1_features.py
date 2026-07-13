"""
Q3 Step 1: Extract trial-mean neural features per area.

Outputs:
  q3/results/<session>/features_q3_<session>.npz
"""

import h5py
import numpy as np
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

# -------------------------
# Load trial labels from Step 0
# -------------------------
trials_df = load_trials()
print(f"Loaded {len(trials_df)} trials")
print(trials_df["label"].value_counts().to_string())
print()

# -------------------------
# Get per-area neuron indices
# -------------------------
area_indices: dict[str, np.ndarray] = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        path = f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"
        if path in f:
            area_indices[area] = f[path][:]
            print(f"  {area}: {len(area_indices[area])} neurons")
        else:
            print(f"  {area}: NOT FOUND")

n_trials = len(trials_df)
n_total_neurons = sum(len(idx) for idx in area_indices.values())
print(f"\nTotal neurons across {len(AREAS)} areas: {n_total_neurons}")

# -------------------------
# Build feature matrices: trial-averaged vector per trial
# -------------------------
X: dict[str, np.ndarray] = {
    area: np.zeros((n_trials, len(idx)), dtype=np.float32)
    for area, idx in area_indices.items()
}

print(f"\nLoading trial responses (dropping first {FRAMES_TO_DROP} frames)...")
with MicronsReader(DATA_PATH) as reader:
    for i, trial_idx in enumerate(trials_df["trial_idx"]):
        trial = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        responses = trial["responses"]
        trial_mean = responses[:, FRAMES_TO_DROP:].mean(axis=1)

        for area, idx in area_indices.items():
            X[area][i, :] = trial_mean[idx]

        if (i + 1) % 50 == 0 or i == n_trials - 1:
            print(f"  {i+1}/{n_trials}")

# -------------------------
# Build label vectors
# -------------------------
y_label = trials_df["label"].values
y_natural = trials_df["is_natural"].values.astype(int)
y_origin = trials_df["origin"].values
y_naturalness_group = trials_df["naturalness_group"].values

# -------------------------
# Save
# -------------------------
out_path = RESULTS_DIR / f"features_q3_{CHOSEN_SESSION}.npz"
np.savez_compressed(
    out_path,
    X_V1=X["V1"],
    X_LM=X["LM"],
    X_AL=X["AL"],
    X_RL=X["RL"],
    y_label=y_label,
    y_natural=y_natural,
    y_origin=y_origin,
    y_naturalness_group=y_naturalness_group,
    trial_idx=trials_df["trial_idx"].values,
    hash=trials_df["hash"].values,
)

print(f"\nSaved to {out_path}")
print(f"File size: {Path(out_path).stat().st_size / 1e6:.1f} MB")

# -------------------------
# Sanity checks
# -------------------------
print("\n" + "=" * 60)
print("Sanity checks")
print("=" * 60)
for area in AREAS:
    mat = X[area]
    print(f"{area}: shape={mat.shape}  "
          f"mean={mat.mean():.3f}  std={mat.std():.3f}  "
          f"min={mat.min():.3f}  max={mat.max():.1f}")

for area in AREAS:
    mat = X[area]
    nat_mean = mat[y_natural == 1].mean()
    par_mean = mat[y_natural == 0].mean()
    print(f"{area}: natural mean={nat_mean:.3f}, parametric mean={par_mean:.3f}, "
          f"diff={nat_mean - par_mean:+.3f}")
