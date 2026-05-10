"""
Q3.2 Step 5: Time-resolved feature extraction + stability metric.

Extracts full temporal traces (trial x neurons x time) for ALL stimuli
and computes per-trial firing-rate stability (coefficient of variation).

Outputs:
  q3/results/<session>/q3_time_features_<session>.npz
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

# ======================================================================
# Load trial labels from Step 0
# ======================================================================
trials_df = load_trials()
print(f"Loaded {len(trials_df)} trials")
print(trials_df["label"].value_counts().to_string())
print()

# ======================================================================
# Get per-area neuron indices
# ======================================================================
area_indices: dict[str, np.ndarray] = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        path = f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"
        if path in f:
            area_indices[area] = f[path][:]
            print(f"  {area}: {len(area_indices[area])} neurons")
        else:
            raise RuntimeError(f"Area indices not found: {path}")

n_trials = len(trials_df)

# Determine minimum number of timepoints across all trials after dropping onset frames
print("Determining common trial length...")
all_lengths = []
with MicronsReader(DATA_PATH) as reader:
    for trial_idx in trials_df["trial_idx"]:
        trial = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        all_lengths.append(trial["responses"].shape[1] - FRAMES_TO_DROP)

n_time = min(all_lengths)
print(f"\nTrials: {n_trials}")
print(f"Timepoints per trial (after dropping {FRAMES_TO_DROP} onset frames): {n_time}")
if len(set(all_lengths)) > 1:
    print(f"  Note: Trials had varying lengths. Truncating all to min length {n_time}.")
    print(f"  Max length was {max(all_lengths)}.")

# ======================================================================
# Preallocate arrays
# ======================================================================
X: dict[str, np.ndarray] = {
    area: np.zeros((n_trials, len(idx), n_time), dtype=np.float32)
    for area, idx in area_indices.items()
}

# ======================================================================
# Load time-resolved responses
# ======================================================================
print("\nLoading time-resolved responses...")

with MicronsReader(DATA_PATH) as reader:
    for i, trial_idx in enumerate(trials_df["trial_idx"]):
        trial = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        responses = trial["responses"][:, FRAMES_TO_DROP : FRAMES_TO_DROP + n_time]

        for area, idx in area_indices.items():
            X[area][i] = responses[idx]

        if (i + 1) % 50 == 0 or i == n_trials - 1:
            print(f"  {i+1}/{n_trials}")

# ======================================================================
# Compute per-trial stability metric (coefficient of variation)
# ======================================================================
print("\nComputing per-trial stability (CV over time)...")

MEAN_FLOOR = 1e-10

stability: dict[str, np.ndarray] = {}
for area in AREAS:
    neuron_mean = X[area].mean(axis=2)  # (n_trials, n_neurons)
    neuron_std = X[area].std(axis=2)    # (n_trials, n_neurons)

    neuron_mean_safe = np.where(np.abs(neuron_mean) < MEAN_FLOOR, MEAN_FLOOR, neuron_mean)
    neuron_cv = neuron_std / np.abs(neuron_mean_safe)  # (n_trials, n_neurons)

    stability[area] = neuron_cv.mean(axis=1)  # (n_trials,)

    print(f"  {area}: mean CV={stability[area].mean():.3f} ± {stability[area].std():.3f}")

# ======================================================================
# Build label vectors
# ======================================================================
y_label = trials_df["label"].values
y_natural = trials_df["is_natural"].values.astype(int)
y_origin = trials_df["origin"].values
y_naturalness_group = trials_df["naturalness_group"].values

# ======================================================================
# Save
# ======================================================================
out_path = RESULTS_DIR / f"q3_time_features_{CHOSEN_SESSION}.npz"

save_dict = {
    "y_label": y_label,
    "y_natural": y_natural,
    "y_origin": y_origin,
    "y_naturalness_group": y_naturalness_group,
    "trial_idx": trials_df["trial_idx"].values,
    "hash": trials_df["hash"].values,
}

for area in AREAS:
    save_dict[f"X_{area}"] = X[area]
    save_dict[f"stability_{area}"] = stability[area]

np.savez_compressed(out_path, **save_dict)

print(f"\nSaved to {out_path}")
print(f"File size: {Path(out_path).stat().st_size / 1e6:.1f} MB")

# ======================================================================
# Sanity checks
# ======================================================================
print("\n" + "=" * 60)
print("Sanity checks")
print("=" * 60)
for area in AREAS:
    mat = X[area]
    print(f"{area}: shape={mat.shape}  "
          f"mean={mat.mean():.3f}  std={mat.std():.3f}  "
          f"min={mat.min():.3f}  max={mat.max():.1f}")

print("\nStability by stimulus type:")
for label in sorted(trials_df["label"].unique()):
    mask = y_label == label
    for area in AREAS:
        cv = stability[area][mask].mean()
        print(f"  {label:12s}  {area}: CV={cv:.3f}")
