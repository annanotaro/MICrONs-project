"""
Full Q1 analysis: learning curves + paired area comparisons + permutation nulls.
Produces:
  - learning_curves_<session>.png / .pdf
  - results_step4_learning_curves_<session>.csv
  - results_step4_paired_comparisons_<session>.csv
  - results_step4_permutation_nulls_<session>.csv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

# ======================================================================
# Config
# ======================================================================

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
AREA_COLORS = {"V1": "#1f77b4", "LM": "#ff7f0e", "AL": "#2ca02c", "RL": "#d62728"}

N_FOLDS = 5
N_SUBSAMPLES = 10
N_PAIRED_SEEDS = 50
N_PERM_SHUFFLES = 100
NEURON_COUNTS = [25, 50, 100, 200, 400, 575]
N_MIN = 575  # bottleneck area (AL) count
RANDOM_STATE = 42

# ======================================================================
# Shared helpers
# ======================================================================
def make_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])

def evaluate(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for tr, te in skf.split(X, y):
        clf = make_classifier()
        clf.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return np.mean(scores)

def evaluate_subsampled(X_full, y, n_neurons, n_repeats, base_seed):
    rng = np.random.default_rng(base_seed)
    scores = []
    for _ in range(n_repeats):
        idx = rng.choice(X_full.shape[1], size=n_neurons, replace=False)
        scores.append(evaluate(X_full[:, idx], y))
    return np.mean(scores), np.std(scores)

# ======================================================================
# Load data & define questions
# ======================================================================
data = np.load(RESULTS_DIR / f"features_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_label = data["y_label"]
y_natural = data["y_natural"]

questions = [
    ("Q1a", "natural vs parametric",
     np.ones(len(y_label), dtype=bool), y_natural, 0.5),
    ("Q1b", "Monet2 vs Trippy",
     np.isin(y_label, ["Monet2", "Trippy"]),
     (y_label[np.isin(y_label, ["Monet2", "Trippy"])] == "Trippy").astype(int),
     0.5),
    ("Q1c", "3 natural clips",
     np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"]),
     y_label[np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"])],
     1/3),
]

# ======================================================================
# PART 1: Learning curves (sweep over neuron counts)
# ======================================================================
print("\n" + "="*70)
print("PART 1: Learning curves")
print("="*70)

lc_results = []
for qname, qdesc, mask, y_q, chance in questions:
    print(f"\n{qname}: {qdesc}")
    for area in AREAS:
        X_full = X_by_area[area][mask]
        n_available = X_full.shape[1]
        for n in NEURON_COUNTS:
            if n > n_available:
                continue
            if n == n_available:
                mean, std = evaluate(X_full, y_q), 0.0
            else:
                mean, std = evaluate_subsampled(
                    X_full, y_q, n, N_SUBSAMPLES,
                    base_seed=(hash(qname + area) + n) % 2**31,
                )
            lc_results.append({
                "question": qname, "area": area, "n_neurons": n,
                "acc_mean": mean, "acc_std": std, "chance": chance,
            })
            print(f"  {area:3s}  n={n:4d}  acc={mean:.3f} ± {std:.3f}")

lc_df = pd.DataFrame(lc_results)
lc_df.to_csv(RESULTS_DIR / f"results_step4_learning_curves_{CHOSEN_SESSION}.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (qname, qdesc, _, _, chance) in zip(axes, questions):
    qdf = lc_df[lc_df["question"] == qname]
    for area in AREAS:
        adf = qdf[qdf["area"] == area].sort_values("n_neurons")
        ax.errorbar(adf["n_neurons"], adf["acc_mean"], yerr=adf["acc_std"],
                    marker="o", capsize=3, lw=1.8, ms=6,
                    color=AREA_COLORS[area], label=area)
    ax.axhline(chance, ls="--", color="gray", lw=1, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("neurons")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"{qname} — {qdesc}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
fig.suptitle(f"Learning curves — session {CHOSEN_SESSION}", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"learning_curves_{CHOSEN_SESSION}.png", dpi=150, bbox_inches="tight")
fig.savefig(RESULTS_DIR / f"learning_curves_{CHOSEN_SESSION}.pdf", bbox_inches="tight")
plt.close()

# ======================================================================
# PART 2: Paired cross-area comparisons at matched n
# ======================================================================
print("\n" + "="*70)
print(f"PART 2: Paired comparisons at n={N_MIN}, {N_PAIRED_SEEDS} seeds")
print("="*70)

def paired_area_comparison(X_a, X_b, y, n_neurons, n_seeds):
    diffs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx_a = rng.choice(X_a.shape[1], size=n_neurons, replace=False)
        idx_b = rng.choice(X_b.shape[1], size=n_neurons, replace=False)
        diffs.append(evaluate(X_a[:, idx_a], y) - evaluate(X_b[:, idx_b], y))
    diffs = np.array(diffs)
    _, p = wilcoxon(diffs, alternative="two-sided")
    return diffs.mean(), diffs.std(), p

paired_results = []
for qname, qdesc, mask, y_q, _ in questions:
    print(f"\n{qname}:")
    for a, b in combinations(AREAS, 2):
        mean_diff, std_diff, p = paired_area_comparison(
            X_by_area[a][mask], X_by_area[b][mask],
            y_q, n_neurons=N_MIN, n_seeds=N_PAIRED_SEEDS,
        )
        paired_results.append({
            "question": qname, "area_a": a, "area_b": b,
            "mean_diff": mean_diff, "std_diff": std_diff, "p_raw": p,
        })
        print(f"  {a} - {b}: {mean_diff:+.4f} ± {std_diff:.4f}  p={p:.4f}")

paired_df = pd.DataFrame(paired_results)
# Bonferroni correction within each question
paired_df["p_bonferroni"] = (
    paired_df.groupby("question")["p_raw"]
    .transform(lambda x: np.minimum(x * len(x), 1.0))
)
paired_df["significant"] = paired_df["p_bonferroni"] < 0.05
paired_df.to_csv(RESULTS_DIR / f"results_step4_paired_comparisons_{CHOSEN_SESSION}.csv", index=False)

print("\nSignificant pairs after Bonferroni correction:")
sig = paired_df[paired_df["significant"]]
if len(sig) == 0:
    print("  None.")
else:
    print(sig[["question", "area_a", "area_b", "mean_diff", "p_bonferroni"]].to_string(index=False))

# ======================================================================
# PART 3: Shuffle-label permutation nulls (full population, each area)
# ======================================================================
print("\n" + "="*70)
print(f"PART 3: Permutation nulls ({N_PERM_SHUFFLES} shuffles per cell)")
print("="*70)
print("This is the slow part — expect several minutes.")

def permutation_null(X, y, n_shuffles, seed):
    rng = np.random.default_rng(seed)
    return np.array([evaluate(X, rng.permutation(y)) for _ in range(n_shuffles)])

null_results = []
for qname, qdesc, mask, y_q, chance in questions:
    print(f"\n{qname}:")
    for area in AREAS:
        X = X_by_area[area][mask]
        observed = evaluate(X, y_q)
        null = permutation_null(X, y_q, N_PERM_SHUFFLES,
                                seed=(hash(qname + area)) % 2**31)
        p = (null >= observed).mean()
        null_results.append({
            "question": qname, "area": area,
            "observed": observed, "null_mean": null.mean(),
            "null_std": null.std(), "p_value": p,
        })
        print(f"  {area:3s}  obs={observed:.3f}  "
              f"null={null.mean():.3f}±{null.std():.3f}  p={p:.3f}")

null_df = pd.DataFrame(null_results)
null_df.to_csv(RESULTS_DIR /  f"results_step4_permutation_nulls_{CHOSEN_SESSION}.csv", index=False)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "="*70)
print("All analyses complete. Files written:")
print("="*70)
print(f"  learning_curves_{CHOSEN_SESSION}.png / .pdf")
print(f"  results_step4_learning_curves_{CHOSEN_SESSION}.csv")
print(f"  results_step4_paired_comparisons_{CHOSEN_SESSION}.csv")
print(f"  results_step4_permutation_nulls_{CHOSEN_SESSION}.csv")