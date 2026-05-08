"""
Q3 Step 2: Binary decoding deep-dive — natural vs parametric.

Analyses:
  1. Binary classification per area (balanced accuracy, stratified 5-fold CV)
  2. Learning curves (accuracy vs neuron count)
  3. Permutation null distributions (200 shuffles)
  4. Paired area comparisons (Wilcoxon signed-rank)
  5. Per-neuron discriminability (Cohen's d distributions)

Outputs:
  q3/results/<session>/q3_binary_decode_<session>.npz
  q3/results/<session>/csv/q3_learning_curves_<session>.csv
  q3/results/<session>/csv/q3_paired_comparisons_<session>.csv
  q3/results/<session>/csv/q3_permutation_nulls_<session>.csv
  q3/results/<session>/csv/q3_neuron_discriminability_<session>.csv
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

from config import (
    AREAS,
    CHOSEN_SESSION,
    N_FOLDS,
    N_NEURONS_MATCHED,
    N_PAIRED_SEEDS,
    N_PERM_SHUFFLES,
    N_SUBSAMPLES,
    NEURON_COUNTS,
    RANDOM_STATE,
    RESULTS_DIR,
    ensure_results_dir,
)

ensure_results_dir()
print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")


# ======================================================================
# Shared helpers
# ======================================================================
def make_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])


def evaluate(X: np.ndarray, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in skf.split(X, y):
        clf = make_classifier()
        clf.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores))


def evaluate_subsampled(
    X_full: np.ndarray,
    y: np.ndarray,
    n_neurons: int,
    n_repeats: int,
    base_seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(base_seed)
    scores = [
        evaluate(X_full[:, rng.choice(X_full.shape[1], size=n_neurons, replace=False)], y)
        for _ in range(n_repeats)
    ]
    return float(np.mean(scores)), float(np.std(scores))


# ======================================================================
# Load data
# ======================================================================
data = np.load(RESULTS_DIR / f"features_q3_clean_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_natural = data["y_natural"]

print(f"Loaded features: {sum(X.shape[1] for X in X_by_area.values())} total neurons")
print(f"Natural: {(y_natural == 1).sum()}, Parametric: {(y_natural == 0).sum()}")

N_NEURONS_MATCHED = min(N_NEURONS_MATCHED, min(X.shape[1] for X in X_by_area.values()))


# ======================================================================
# PART 1: Learning curves
# ======================================================================
print("\n" + "=" * 70)
print("PART 1: Learning curves — natural vs parametric")
print("=" * 70)

lc_results: list[dict] = []
for area in AREAS:
    X_full = X_by_area[area]
    n_available = X_full.shape[1]
    for n in NEURON_COUNTS:
        if n > n_available:
            continue
        if n == n_available:
            mean_acc, std_acc = evaluate(X_full, y_natural), 0.0
        else:
            mean_acc, std_acc = evaluate_subsampled(
                X_full, y_natural, n, N_SUBSAMPLES,
                base_seed=(hash("nat_vs_par" + area) + n) % 2**31,
            )
        lc_results.append({
            "area": area, "n_neurons": n,
            "acc_mean": mean_acc, "acc_std": std_acc, "chance": 0.5,
        })
        print(f"  {area:3s}  n={n:4d}  acc={mean_acc:.3f} ± {std_acc:.3f}")

lc_df = pd.DataFrame(lc_results)
lc_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_learning_curves_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 2: Paired area comparisons at matched neuron count
# ======================================================================
print("\n" + "=" * 70)
print(f"PART 2: Paired comparisons at n={N_NEURONS_MATCHED}, {N_PAIRED_SEEDS} seeds")
print("=" * 70)


def paired_area_comparison(
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    n_neurons: int,
    n_seeds: int,
) -> tuple[float, float, float]:
    diffs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx_a = rng.choice(X_a.shape[1], size=n_neurons, replace=False)
        idx_b = rng.choice(X_b.shape[1], size=n_neurons, replace=False)
        diffs.append(evaluate(X_a[:, idx_a], y) - evaluate(X_b[:, idx_b], y))
    diffs_arr = np.array(diffs)
    _, p = wilcoxon(diffs_arr, alternative="two-sided")
    return float(diffs_arr.mean()), float(diffs_arr.std()), float(p)


paired_results: list[dict] = []
for a, b in combinations(AREAS, 2):
    mean_diff, std_diff, p = paired_area_comparison(
        X_by_area[a], X_by_area[b], y_natural,
        n_neurons=N_NEURONS_MATCHED, n_seeds=N_PAIRED_SEEDS,
    )
    paired_results.append({
        "area_a": a, "area_b": b, "n_neurons": N_NEURONS_MATCHED,
        "mean_diff": mean_diff, "std_diff": std_diff, "p_raw": p,
    })
    print(f"  {a} - {b}: {mean_diff:+.4f} ± {std_diff:.4f}  p={p:.4f}")

paired_df = pd.DataFrame(paired_results)
paired_df["p_bonferroni"] = np.minimum(paired_df["p_raw"] * len(paired_df), 1.0)
paired_df["significant"] = paired_df["p_bonferroni"] < 0.05
paired_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_paired_comparisons_{CHOSEN_SESSION}.csv", index=False,
)

print("\nSignificant pairs after Bonferroni correction:")
sig = paired_df[paired_df["significant"]]
if len(sig) == 0:
    print("  None.")
else:
    print(sig[["area_a", "area_b", "mean_diff", "p_bonferroni"]].to_string(index=False))


# ======================================================================
# PART 3: Permutation null distributions
# ======================================================================
print("\n" + "=" * 70)
print(f"PART 3: Permutation nulls ({N_PERM_SHUFFLES} shuffles per area)")
print("=" * 70)
print("This is the slow part — expect several minutes.")

null_results: list[dict] = []
null_distributions: dict[str, np.ndarray] = {}

for area in AREAS:
    X = X_by_area[area]
    observed = evaluate(X, y_natural)

    rng = np.random.default_rng(hash("perm_null" + area) % 2**31)
    null_acc = np.array([evaluate(X, rng.permutation(y_natural)) for _ in range(N_PERM_SHUFFLES)])
    null_distributions[area] = null_acc

    p_value = float((null_acc >= observed).mean())
    null_results.append({
        "area": area, "observed": observed,
        "null_mean": float(null_acc.mean()), "null_std": float(null_acc.std()),
        "p_value": p_value,
    })
    print(f"  {area:3s}  obs={observed:.3f}  "
          f"null={null_acc.mean():.3f}±{null_acc.std():.3f}  p={p_value:.3f}")

null_df = pd.DataFrame(null_results)
null_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_permutation_nulls_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 4: Per-neuron discriminability (Cohen's d)
# ======================================================================
print("\n" + "=" * 70)
print("PART 4: Per-neuron Cohen's d (natural vs parametric)")
print("=" * 70)

discriminability: dict[str, np.ndarray] = {}
disc_rows: list[dict] = []

for area in AREAS:
    X = X_by_area[area]
    nat_responses = X[y_natural == 1]
    par_responses = X[y_natural == 0]

    nat_mean = nat_responses.mean(axis=0)
    par_mean = par_responses.mean(axis=0)
    pooled_std = np.sqrt(
        (nat_responses.var(axis=0) * (nat_responses.shape[0] - 1)
         + par_responses.var(axis=0) * (par_responses.shape[0] - 1))
        / (nat_responses.shape[0] + par_responses.shape[0] - 2)
    )
    pooled_std = np.where(pooled_std == 0, 1e-10, pooled_std)

    d_values = (nat_mean - par_mean) / pooled_std
    discriminability[area] = d_values

    disc_rows.append({
        "area": area, "n_neurons": len(d_values),
        "mean_abs_d": float(np.abs(d_values).mean()),
        "median_abs_d": float(np.median(np.abs(d_values))),
        "max_abs_d": float(np.abs(d_values).max()),
        "frac_above_02": float((np.abs(d_values) > 0.2).mean()),
    })
    print(f"  {area:3s}  mean|d|={np.abs(d_values).mean():.3f}  "
          f"median|d|={np.median(np.abs(d_values)):.3f}  "
          f"frac>0.2={((np.abs(d_values) > 0.2).mean()):.2%}")

disc_df = pd.DataFrame(disc_rows)
disc_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_neuron_discriminability_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# Save everything to .npz
# ======================================================================
save_dict = {
    "y_natural": y_natural,
    "neuron_counts": np.array(NEURON_COUNTS),
    "n_perm_shuffles": np.array(N_PERM_SHUFFLES),
}

for area in AREAS:
    save_dict[f"lc_mean_{area}"] = lc_df[lc_df["area"] == area]["acc_mean"].values
    save_dict[f"lc_std_{area}"] = lc_df[lc_df["area"] == area]["acc_std"].values
    save_dict[f"lc_n_{area}"] = lc_df[lc_df["area"] == area]["n_neurons"].values
    save_dict[f"null_{area}"] = null_distributions[area]
    save_dict[f"observed_{area}"] = np.array(
        null_df[null_df["area"] == area]["observed"].values[0]
    )
    save_dict[f"cohens_d_{area}"] = discriminability[area]

out_path = RESULTS_DIR / f"q3_binary_decode_{CHOSEN_SESSION}.npz"
np.savez_compressed(out_path, **save_dict)
print(f"\nSaved to {out_path}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("All step 2 analyses complete. Files written:")
print("=" * 70)
print(f"  q3_binary_decode_{CHOSEN_SESSION}.npz")
print(f"  csv/q3_learning_curves_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_paired_comparisons_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_permutation_nulls_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_neuron_discriminability_{CHOSEN_SESSION}.csv")
