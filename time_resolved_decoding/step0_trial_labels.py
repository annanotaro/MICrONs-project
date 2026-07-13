import os
import sys
import h5py
import pandas as pd
from pathlib import Path

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

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs -> {RESULTS_DIR}")

# -------------------------
# BUILD LABEL TABLE
# -------------------------
def encode_hash(h):
    return h.replace("/", "%2F")

rows = []
missing = 0

with h5py.File(DATA_PATH, "r") as f:
    trials_grp = f[f"sessions/{CHOSEN_SESSION}/trials"]
    n_trials = len(trials_grp)
    print(f"Session {CHOSEN_SESSION}: {n_trials} trials")

    for trial_idx in range(n_trials):
        t = trials_grp[str(trial_idx)]
        h = t.attrs["condition_hash"]
        video_key = encode_hash(h)

        if f"videos/{video_key}" not in f:
            missing += 1
            rows.append((trial_idx, h, "UNKNOWN", "UNKNOWN", "UNKNOWN"))
            continue

        attrs = dict(f[f"videos/{video_key}"].attrs)
        rows.append((
            trial_idx,
            h,
            attrs.get("type", "UNKNOWN"),
            attrs.get("short_movie_name", "UNKNOWN"),
            attrs.get("movie_name", "UNKNOWN"),
        ))

print(f"Missing video entries: {missing}")

# -------------------------
# ASSEMBLE TABLE
# -------------------------
trials_df = pd.DataFrame(
    rows,
    columns=["trial_idx", "hash", "type", "short_name", "movie_name"]
)

print("\nCounts by type:")
print(trials_df["type"].value_counts())

print("\nCounts by short_movie_name:")
print(trials_df["short_name"].value_counts())

# -------------------------
# LABELS
# -------------------------
def unified_label(row):
    if row["type"] == "Clip":
        return row["short_name"]
    return row["type"]

trials_df["label"] = trials_df.apply(unified_label, axis=1)
trials_df["label"] = trials_df["label"].replace({"sports1m": "Sports1M"})
trials_df["is_natural"] = trials_df["label"].isin(["Cinematic", "Sports1M", "Rendered"])

print("\nFinal label distribution:")
print(trials_df["label"].value_counts())

print("\nNatural vs parametric:")
print(trials_df["is_natural"].value_counts())

# -------------------------
# SAVE
# -------------------------
out_path = RESULTS_DIR / f"trials_{CHOSEN_SESSION}.csv"
trials_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
