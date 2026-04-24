import os
import sys
import importlib.util
import pandas as pd
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

READER_PATH = "/Users/gaiagr/.cache/huggingface/hub/datasets--NeuroBLab--MICrONS/snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03/reader.py"
DATA_PATH = "/Users/gaiagr/.cache/huggingface/hub/datasets--NeuroBLab--MICrONS/snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03/microns.h5"

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs -> {RESULTS_DIR}")

# -------------------------
# LOAD READER
# -------------------------
spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

# -------------------------
# BUILD LABEL TABLE
# -------------------------
with MicronsReader(DATA_PATH) as reader:
    hashes = reader.get_hashes_by_session(CHOSEN_SESSION)
    print(f"Session {CHOSEN_SESSION}: {len(hashes)} trials")

    rows = []
    missing = 0

    for trial_idx, h in enumerate(hashes):
        h_key = reader._encode_hash(h)
        video_path = f"videos/{h_key}"

        if video_path not in reader.f:
            missing += 1
            rows.append((trial_idx, h, "UNKNOWN", "UNKNOWN", "UNKNOWN"))
            continue

        attrs = dict(reader.f[video_path].attrs)

        rows.append((
            trial_idx,
            h,
            attrs.get("type", "UNKNOWN"),
            attrs.get("short_movie_name", "UNKNOWN"),
            attrs.get("movie_name", "UNKNOWN"),
        ))

    print(f"Missing video entries: {missing}")

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

# Normalize naming
trials_df["label"] = trials_df["label"].replace({"sports1m": "Sports1M"})

# Binary flag
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