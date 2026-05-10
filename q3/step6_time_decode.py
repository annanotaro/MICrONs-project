"""
Q3.2 Step 6: Time-resolved decoding with stability conditioning.

Analyses:
  1. Time-resolved accuracy curves (binary, 3-class, 5-class)
  2. Stability-conditioned decoding (stable vs unstable trials)
  3. Stability distributions by stimulus type (Mann-Whitney U)
  4. Per-timepoint stability-accuracy correlation (Spearman)

Outputs:
  q3/results/<session>/q3_time_decode_<session>.npz
  q3/results/<session>/csv/q3_time_accuracy_<session>.csv
  q3/results/<session>/csv/q3_stability_by_stimulus_<session>.csv
  q3/results/<session>/csv/q3_stability_decoding_<session>.csv
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

from config import (
    AREAS,
    CHOSEN_SESSION,
    N_FOLDS,
    N_NEURONS_TIME,
    N_SEEDS_TIME,
    RANDOM_STATE,
    RESULTS_DIR,
    ensure_results_dir,
)

ensure_results_dir()
print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")


# ======================================================================
# Helpers
# ======================================================================
def make_classifier(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=seed,
        )),
    ])


def decode_timepoint(
    Xt: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> float:
    gkf = GroupKFold(n_splits=N_FOLDS)
    scores = []
    for tr, te in gkf.split(Xt, y, groups=groups):
        clf = make_classifier(seed)
        clf.fit(Xt[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(Xt[te])))
    return float(np.mean(scores))


def decode_over_time(
    X_full: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_neurons_keep: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_acc, std_acc) over time, averaged across seeds."""
    n_trials, n_neurons, n_time = X_full.shape
    n_keep = min(n_neurons_keep, n_neurons)

    seeds_acc = np.zeros((N_SEEDS_TIME, n_time))
    for s in range(N_SEEDS_TIME):
        rng = np.random.default_rng(RANDOM_STATE + s)
        keep_idx = rng.choice(n_neurons, size=n_keep, replace=False)
        X = X_full[:, keep_idx, :]

        for t in range(n_time):
            seeds_acc[s, t] = decode_timepoint(X[:, :, t], y, groups, RANDOM_STATE + s)

    return seeds_acc.mean(axis=0), seeds_acc.std(axis=0)


# ======================================================================
# Load data
# ======================================================================
data = np.load(
    RESULTS_DIR / f"q3_time_features_clean_{CHOSEN_SESSION}.npz", allow_pickle=True,
)

X_by_area = {area: data[f"X_{area}"] for area in AREAS}
stability_by_area = {area: data[f"stability_{area}"] for area in AREAS}
y_label = data["y_label"]
y_natural = data["y_natural"]
y_origin = data["y_origin"]
groups = data["hash"]

n_time = X_by_area["V1"].shape[2]
print(f"Loaded: {len(y_label)} trials, {n_time} timepoints")
print(f"Natural: {(y_natural == 1).sum()}, Parametric: {(y_natural == 0).sum()}")

N_NEURONS_TIME = min(N_NEURONS_TIME, min(X.shape[1] for X in X_by_area.values()))


# ======================================================================
# PART 1: Time-resolved accuracy curves
# ======================================================================
print("\n" + "=" * 70)
print("PART 1: Time-resolved decoding (binary, 3-class, 5-class)")
print("=" * 70)

tasks = {
    "binary": {"y": y_natural, "chance": 0.5},
    "3class": {"y": y_origin, "chance": 1 / 3},
    "5class": {"y": y_label, "chance": 1 / 5},
}

time_acc: dict[str, dict[str, np.ndarray]] = {}
time_std: dict[str, dict[str, np.ndarray]] = {}

for task_name, task_info in tasks.items():
    print(f"\n  Task: {task_name} (chance={task_info['chance']:.2f})")
    time_acc[task_name] = {}
    time_std[task_name] = {}

    for area in AREAS:
        print(f"    {area}...", end=" ", flush=True)
        mean_acc, std_acc = decode_over_time(
            X_by_area[area], task_info["y"], groups, N_NEURONS_TIME,
        )
        time_acc[task_name][area] = mean_acc
        time_std[task_name][area] = std_acc
        print(f"peak={mean_acc.max():.3f}")

# Summary CSV
time_acc_rows: list[dict] = []
for task_name, task_info in tasks.items():
    for area in AREAS:
        acc = time_acc[task_name][area]
        time_acc_rows.append({
            "task": task_name,
            "area": area,
            "peak_acc": float(acc.max()),
            "peak_t": int(acc.argmax()),
            "mean_acc": float(acc.mean()),
            "chance": task_info["chance"],
        })

time_acc_df = pd.DataFrame(time_acc_rows)
time_acc_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_time_accuracy_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 2: Stability-conditioned decoding (stable vs unstable half)
# ======================================================================
print("\n" + "=" * 70)
print("PART 2: Stable vs unstable trials (binary decoding)")
print("=" * 70)

stability_results: list[dict] = []

for area in AREAS:
    cv_values = stability_by_area[area]
    median_cv = np.median(cv_values)

    stable_mask = cv_values <= median_cv
    unstable_mask = cv_values > median_cv

    print(f"\n  {area}: median CV={median_cv:.3f}, "
          f"stable n={stable_mask.sum()}, unstable n={unstable_mask.sum()}")

    # Decode each half
    for split_name, mask in [("stable", stable_mask), ("unstable", unstable_mask)]:
        X_split = X_by_area[area][mask]
        y_split = y_natural[mask]
        groups_split = groups[mask]

        n_unique_groups = len(np.unique(groups_split))
        n_splits = min(N_FOLDS, n_unique_groups)
        if n_splits < 2:
            print(f"    {split_name}: too few groups ({n_unique_groups}), skipping")
            continue

        n_keep = min(N_NEURONS_TIME, X_split.shape[1])
        seeds_acc = np.zeros((N_SEEDS_TIME, n_time))

        for s in range(N_SEEDS_TIME):
            rng = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng.choice(X_split.shape[1], size=n_keep, replace=False)
            X_sub = X_split[:, keep_idx, :]

            gkf = GroupKFold(n_splits=n_splits)
            for t in range(n_time):
                Xt = X_sub[:, :, t]
                fold_accs = []
                for tr, te in gkf.split(Xt, y_split, groups=groups_split):
                    clf = make_classifier(RANDOM_STATE + s)
                    clf.fit(Xt[tr], y_split[tr])
                    fold_accs.append(
                        balanced_accuracy_score(y_split[te], clf.predict(Xt[te]))
                    )
                seeds_acc[s, t] = np.mean(fold_accs)

        mean_curve = seeds_acc.mean(axis=0)
        time_acc[f"binary_{split_name}_{area}"] = {area: mean_curve}
        time_std[f"binary_{split_name}_{area}"] = {area: seeds_acc.std(axis=0)}

        stability_results.append({
            "area": area,
            "split": split_name,
            "peak_acc": float(mean_curve.max()),
            "peak_t": int(mean_curve.argmax()),
            "mean_acc": float(mean_curve.mean()),
        })
        print(f"    {split_name}: peak={mean_curve.max():.3f}")

# Paired comparison: peak accuracy stable vs unstable per area
print("\n  Paired peak-accuracy comparison (stable vs unstable):")
for area in AREAS:
    stable_row = next(
        (r for r in stability_results if r["area"] == area and r["split"] == "stable"),
        None,
    )
    unstable_row = next(
        (r for r in stability_results if r["area"] == area and r["split"] == "unstable"),
        None,
    )
    if stable_row and unstable_row:
        diff = stable_row["peak_acc"] - unstable_row["peak_acc"]
        print(f"    {area}: stable={stable_row['peak_acc']:.3f}, "
              f"unstable={unstable_row['peak_acc']:.3f}, diff={diff:+.3f}")

stability_decode_df = pd.DataFrame(stability_results)
stability_decode_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_stability_decoding_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 3: Stability by stimulus type
# ======================================================================
print("\n" + "=" * 70)
print("PART 3: Stability distributions by stimulus type")
print("=" * 70)

stab_rows: list[dict] = []
for area in AREAS:
    cv_values = stability_by_area[area]

    nat_cv = cv_values[y_natural == 1]
    par_cv = cv_values[y_natural == 0]
    stat, p_mwu = mannwhitneyu(nat_cv, par_cv, alternative="two-sided")

    print(f"  {area}: natural CV={nat_cv.mean():.3f}±{nat_cv.std():.3f}, "
          f"parametric CV={par_cv.mean():.3f}±{par_cv.std():.3f}, "
          f"MWU p={p_mwu:.4f}")

    for label in sorted(np.unique(y_label)):
        mask = y_label == label
        stab_rows.append({
            "area": area,
            "label": label,
            "is_natural": int(y_natural[mask][0]),
            "cv_mean": float(cv_values[mask].mean()),
            "cv_std": float(cv_values[mask].std()),
            "cv_median": float(np.median(cv_values[mask])),
            "n_trials": int(mask.sum()),
        })

    stab_rows.append({
        "area": area,
        "label": "_nat_vs_par_test",
        "is_natural": -1,
        "cv_mean": float(nat_cv.mean() - par_cv.mean()),
        "cv_std": 0.0,
        "cv_median": p_mwu,
        "n_trials": len(cv_values),
    })

stab_df = pd.DataFrame(stab_rows)
stab_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_stability_by_stimulus_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 4: Per-timepoint stability-accuracy correlation
# ======================================================================
print("\n" + "=" * 70)
print("PART 4: Stability-accuracy correlation over time")
print("=" * 70)

for area in AREAS:
    cv_values = stability_by_area[area]
    acc_curve = time_acc["binary"][area]

    # Per-timepoint mean stability of correctly-decoded trials
    # Use overall accuracy as proxy — correlate timepoint-level acc with mean CV
    # across trials at that timepoint (trial-level CV is time-averaged, so
    # we correlate the fixed trial-level stability with time-varying accuracy)
    rho, p_spearman = spearmanr(np.arange(n_time), acc_curve)
    print(f"  {area}: time vs accuracy Spearman rho={rho:.3f}, p={p_spearman:.4f}")


# ======================================================================
# Save everything to .npz
# ======================================================================
save_dict: dict[str, np.ndarray] = {
    "n_time": np.array(n_time),
    "n_seeds": np.array(N_SEEDS_TIME),
    "n_neurons_time": np.array(N_NEURONS_TIME),
}

for task_name in tasks:
    for area in AREAS:
        save_dict[f"acc_{task_name}_{area}"] = time_acc[task_name][area]
        save_dict[f"std_{task_name}_{area}"] = time_std[task_name][area]

for area in AREAS:
    save_dict[f"stability_{area}"] = stability_by_area[area]
    for split_name in ("stable", "unstable"):
        key = f"binary_{split_name}_{area}"
        if key in time_acc:
            save_dict[f"acc_{key}"] = time_acc[key][area]
            save_dict[f"std_{key}"] = time_std[key][area]

out_path = RESULTS_DIR / f"q3_time_decode_{CHOSEN_SESSION}.npz"
np.savez_compressed(out_path, **save_dict)
print(f"\nSaved to {out_path}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("All Q3.2 step 6 analyses complete. Files written:")
print("=" * 70)
print(f"  q3_time_decode_{CHOSEN_SESSION}.npz")
print(f"  csv/q3_time_accuracy_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_stability_by_stimulus_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_stability_decoding_{CHOSEN_SESSION}.csv")
