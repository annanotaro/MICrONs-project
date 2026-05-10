import sys, importlib.util
import h5py
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MICRONS_DATA_PATH, MICRONS_READER_PATH

READER_PATH = MICRONS_READER_PATH
DATA_PATH = MICRONS_DATA_PATH


if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

# All outputs go to results/<session>/
RESULTS_DIR = Path(__file__).parent.parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")

AREAS = ["V1", "LM", "AL", "RL"]
FRAMES_TO_DROP = 3

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

# ---------------------------------------------------------------
# 1. Load data for regression (pupil + treadmill -> responses)
# ---------------------------------------------------------------
trials_df = pd.read_csv(RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv")

all_responses = []
all_pupil = []
all_treadmill = []

print(f"Loading {len(trials_df)} trials for behavioral regression...")
with MicronsReader(DATA_PATH) as reader:
    for trial_idx in trials_df["trial_idx"]:
        t = reader.get_trial(CHOSEN_SESSION, int(trial_idx))
        all_responses.append(t["responses"]) # (n_neurons, T)
        all_pupil.append(t["pupil"])        # (4, T)
        all_treadmill.append(t["treadmill"])  # (T,)

# Concatenate everything over time
R = np.concatenate(all_responses, axis=1)    # (n_neurons, total_T)
P = np.concatenate(all_pupil, axis=1).T      # (total_T, 4)
TM = np.concatenate(all_treadmill)[:, None]  # (total_T, 1)

# Regression: response = beta * behavior + residual
behavior = np.hstack([P, TM]) # (total_T, 5)

print(f"Concatenated shape: responses {R.shape}, behavior {behavior.shape}")

# Handle NaNs (pupil can have NaNs if eye tracking lost)
valid = ~(np.isnan(behavior).any(axis=1) | np.isnan(R).any(axis=0))
print(f"Valid timepoints: {valid.sum()} / {len(valid)}")

lr = LinearRegression()
lr.fit(behavior[valid], R[:, valid].T)

# Get residuals (Cleaned responses)
# Use the full behavior (filling NaNs with means so we don't lose trials)
behavior_filled = behavior.copy()
for j in range(behavior.shape[1]):
    mask = np.isnan(behavior_filled[:, j])
    behavior_filled[mask, j] = np.nanmean(behavior[:, j])

R_pred = lr.predict(behavior_filled).T
R_clean = R - R_pred

print("Regression complete.")

# ---------------------------------------------------------------
# 2. Re-extract trial-averaged features from cleaned responses
# ---------------------------------------------------------------
# We need to know where each trial starts/ends in the concatenated array
trial_lengths = [res.shape[1] for res in all_responses]
trial_starts = [0] + list(np.cumsum(trial_lengths)[:-1])

area_indices = {}
with h5py.File(DATA_PATH, "r") as f:
    for area in AREAS:
        area_indices[area] = f[f"sessions/{CHOSEN_SESSION}/meta/area_indices/{area}"][:]

X_clean = {area: np.zeros((len(trials_df), len(idx)), dtype=np.float32)
           for area, idx in area_indices.items()}

for i, start in enumerate(trial_starts):
    end = start + trial_lengths[i]
    # Re-apply the onset drop
    ts_clean = R_clean[:, start + FRAMES_TO_DROP : end]
    trial_mean = ts_clean.mean(axis=1)
    
    for area, idx in area_indices.items():
        X_clean[area][i, :] = trial_mean[idx]

# ---------------------------------------------------------------
# 3. Save
# ---------------------------------------------------------------
out_path = RESULTS_DIR / f"features_clean_{CHOSEN_SESSION}.npz"
np.savez_compressed(
    out_path,
    X_V1=X_clean["V1"], X_LM=X_clean["LM"], X_AL=X_clean["AL"], X_RL=X_clean["RL"],
    y_label=trials_df["label"].values,
    y_natural=trials_df["is_natural"].values.astype(int),
    trial_idx=trials_df["trial_idx"].values,
    hash=trials_df["hash"].values,
)

print(f"Saved cleaned features to {out_path}")
