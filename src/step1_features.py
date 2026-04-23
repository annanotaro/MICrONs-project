import sys
import importlib.util
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

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

# Drop the first N frames before averaging (calcium/spike response onset lag)
# At ~6.3 Hz, 3 frames ≈ 475 ms
FRAMES_TO_DROP = 3

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

# ---------------------------------------------------------------
# Load trial labels from Step 0
# ---------------------------------------------------------------
trials_df = pd.read_csv(RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv")
print(f"Loaded {len(trials_df)} trials from {RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"}")
print(trials_df["label"].value_counts().to_string())
print()

# ---------------------------------------------------------------
# Get per-area neuron indices once
# ---------------------------------------------------------------
area_indices = {}
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
print(f"\nTotal neurons across 4 areas: {n_total_neurons}")
print()

# ---------------------------------------------------------------
# Build feature matrices: one trial-averaged vector per trial
# ---------------------------------------------------------------
# Pre-allocate one matrix per area
X = {area: np.zeros((n_trials, len(idx)), dtype=np.float32)
     for area, idx in area_indices.items()}

print(f"Loading trial responses (dropping first {FRAMES_TO_DROP} frames before mean)...")
with MicronsReader(DATA_PATH) as reader:
    for i, trial_idx in enumerate(trials_df["trial_idx"]):
        trial = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        responses = trial["responses"]  # (n_all_neurons, 75)

        # Mean over time, skipping onset lag frames
        trial_mean = responses[:, FRAMES_TO_DROP:].mean(axis=1)

        # Split by area
        for area, idx in area_indices.items():
            X[area][i, :] = trial_mean[idx]

        if (i + 1) % 50 == 0 or i == n_trials - 1:
            print(f"  {i+1}/{n_trials}")

# ---------------------------------------------------------------
# Build label vectors
# ---------------------------------------------------------------
y_label = trials_df["label"].values               # 5-class: Cinematic, Sports1M, Rendered, Monet2, Trippy
y_natural = trials_df["is_natural"].values.astype(int)  # Q1a: 1=natural, 0=parametric

# ---------------------------------------------------------------
# Save everything as a single .npz file
# ---------------------------------------------------------------
out_path = RESULTS_DIR / f"features_{CHOSEN_SESSION}.npz"
np.savez_compressed(
    out_path,
    X_V1=X["V1"], X_LM=X["LM"], X_AL=X["AL"], X_RL=X["RL"],
    y_label=y_label,
    y_natural=y_natural,
    trial_idx=trials_df["trial_idx"].values,
    hash=trials_df["hash"].values,
)

print(f"\nSaved to {out_path}")
print(f"File size: {Path(out_path).stat().st_size / 1e6:.1f} MB")

# ---------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("Sanity checks")
print("=" * 60)
for area in AREAS:
    mat = X[area]
    print(f"{area}: shape={mat.shape}  "
          f"mean={mat.mean():.3f}  std={mat.std():.3f}  "
          f"min={mat.min():.3f}  max={mat.max():.1f}")

# Check that there's actual stimulus-driven variance
# Natural vs parametric mean response should differ slightly
for area in AREAS:
    mat = X[area]
    nat_mean = mat[y_natural == 1].mean()
    par_mean = mat[y_natural == 0].mean()
    print(f"{area}: natural mean={nat_mean:.3f}, parametric mean={par_mean:.3f}, "
          f"diff={nat_mean - par_mean:+.3f}")