"""
Q3 Step 0: Build enriched trial table with fine-grained grouping columns.

Outputs:
  q3/results/<session>/trials_q3_<session>.csv
"""

import pandas as pd

from config import (
    CHOSEN_SESSION,
    DATA_PATH,
    FILMED_LABELS,
    NATURAL_LABELS,
    RESULTS_DIR,
    ensure_results_dir,
    get_reader_class,
)

ensure_results_dir()
print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")

MicronsReader = get_reader_class()

# -------------------------
# Build raw trial table
# -------------------------
with MicronsReader(DATA_PATH) as reader:
    hashes = reader.get_hashes_by_session(CHOSEN_SESSION)
    print(f"Session {CHOSEN_SESSION}: {len(hashes)} trials")

    rows: list[tuple] = []
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
    columns=["trial_idx", "hash", "type", "short_name", "movie_name"],
)

# -------------------------
# Unified label (same logic as src/step0)
# -------------------------
def unified_label(row: pd.Series) -> str:
    if row["type"] == "Clip":
        return row["short_name"]
    return row["type"]

trials_df["label"] = trials_df.apply(unified_label, axis=1)
trials_df["label"] = trials_df["label"].replace({"sports1m": "Sports1M"})

# -------------------------
# Q3-specific grouping columns
# -------------------------
trials_df["is_natural"] = trials_df["label"].isin(NATURAL_LABELS)


def assign_origin(label: str) -> str:
    if label in FILMED_LABELS:
        return "filmed"
    if label == "Rendered":
        return "rendered"
    return "parametric"


trials_df["origin"] = trials_df["label"].map(assign_origin)


def assign_naturalness_group(label: str) -> str:
    if label in FILMED_LABELS:
        return "human_filmed"
    if label == "Rendered":
        return "cg_natural"
    return "parametric"


trials_df["naturalness_group"] = trials_df["label"].map(assign_naturalness_group)

# -------------------------
# Summary
# -------------------------
print("\nLabel distribution:")
print(trials_df["label"].value_counts())

print("\nNatural vs parametric:")
print(trials_df["is_natural"].value_counts())

print("\nOrigin groups:")
print(trials_df["origin"].value_counts())

print("\nNaturalness groups:")
print(trials_df["naturalness_group"].value_counts())

# -------------------------
# Save
# -------------------------
out_path = RESULTS_DIR / f"trials_q3_{CHOSEN_SESSION}.csv"
trials_df.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")
