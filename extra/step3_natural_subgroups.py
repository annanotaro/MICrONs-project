"""
Q3 Step 3: 3-way origin classification and pairwise contrasts.

Analyses:
  1. 3-class: filmed vs rendered vs parametric
  2. Pairwise: filmed↔rendered, filmed↔parametric, rendered↔parametric
  3. Confusion matrices per area (3-class)
  4. 5-class full classification (all individual labels)
  5. 5-class confusion matrices per area

Outputs:
  q3/results/<session>/q3_subgroup_decode_<session>.npz
  q3/results/<session>/csv/q3_threeclass_accuracy_<session>.csv
  q3/results/<session>/csv/q3_pairwise_accuracy_<session>.csv
  q3/results/<session>/csv/q3_fiveclass_accuracy_<session>.csv
  q3/results/<session>/csv/q3_confusion_3class_<session>.csv
  q3/results/<session>/csv/q3_confusion_5class_<session>.csv
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from config import (
    AREAS,
    CHOSEN_SESSION,
    N_FOLDS,
    N_NEURONS_MATCHED,
    N_SUBSAMPLES,
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


def cv_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Collect cross-validated predictions for confusion matrix."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = make_classifier()
        clf.fit(X[tr], y[tr])
        y_pred[te] = clf.predict(X[te])
    return y_pred


def subsample_neurons(
    X: np.ndarray, n_target: int, rng: np.random.Generator,
) -> np.ndarray:
    if X.shape[1] <= n_target:
        return X
    idx = rng.choice(X.shape[1], size=n_target, replace=False)
    return X[:, idx]


# ======================================================================
# Load data
# ======================================================================
data = np.load(RESULTS_DIR / f"features_q3_clean_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_label = data["y_label"]
y_origin = data["y_origin"]
y_naturalness_group = data["y_naturalness_group"]

print(f"Loaded features: {len(y_label)} trials")
print(f"Origin counts: {dict(zip(*np.unique(y_origin, return_counts=True)))}")
print(f"Label counts: {dict(zip(*np.unique(y_label, return_counts=True)))}")

N_NEURONS_MATCHED = min(N_NEURONS_MATCHED, min(X.shape[1] for X in X_by_area.values()))


# ======================================================================
# PART 1: 3-class classification (filmed vs rendered vs parametric)
# ======================================================================
print("\n" + "=" * 70)
print("PART 1: 3-class origin classification")
print("=" * 70)

origin_labels = ["filmed", "rendered", "parametric"]
threeclass_results: list[dict] = []
cm_3class: dict[str, np.ndarray] = {}
rng = np.random.default_rng(RANDOM_STATE)

for area in AREAS:
    X_full = X_by_area[area]
    X_sub = subsample_neurons(X_full, N_NEURONS_MATCHED, rng)

    if X_full.shape[1] > N_NEURONS_MATCHED:
        mean_acc, std_acc = evaluate_subsampled(
            X_full, y_origin, N_NEURONS_MATCHED, N_SUBSAMPLES,
            base_seed=(hash("3class" + area)) % 2**31,
        )
    else:
        mean_acc, std_acc = evaluate(X_full, y_origin), 0.0

    threeclass_results.append({
        "area": area, "task": "3class_origin",
        "acc_mean": mean_acc, "acc_std": std_acc,
        "chance": 1 / 3, "n_neurons": min(N_NEURONS_MATCHED, X_full.shape[1]),
    })

    y_pred = cv_predictions(X_sub, y_origin)
    cm_3class[area] = confusion_matrix(y_origin, y_pred, labels=origin_labels, normalize="true")

    print(f"  {area:3s}  acc={mean_acc:.3f} ± {std_acc:.3f}")

threeclass_df = pd.DataFrame(threeclass_results)
threeclass_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_threeclass_accuracy_{CHOSEN_SESSION}.csv", index=False,
)

cm3_rows: list[dict] = []
for area in AREAS:
    cm = cm_3class[area]
    for i, true_lbl in enumerate(origin_labels):
        for j, pred_lbl in enumerate(origin_labels):
            cm3_rows.append({
                "area": area, "true": true_lbl, "predicted": pred_lbl,
                "proportion": cm[i, j],
            })
pd.DataFrame(cm3_rows).to_csv(
    RESULTS_DIR / "csv" / f"q3_confusion_3class_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 2: Pairwise origin comparisons
# ======================================================================
print("\n" + "=" * 70)
print("PART 2: Pairwise origin contrasts")
print("=" * 70)

pairwise_results: list[dict] = []
pair_names = list(combinations(origin_labels, 2))

for origin_a, origin_b in pair_names:
    mask = np.isin(y_origin, [origin_a, origin_b])
    y_pair = (y_origin[mask] == origin_b).astype(int)

    print(f"\n  {origin_a} vs {origin_b} (n={mask.sum()}):")
    for area in AREAS:
        X_full = X_by_area[area][mask]
        if X_full.shape[1] > N_NEURONS_MATCHED:
            mean_acc, std_acc = evaluate_subsampled(
                X_full, y_pair, N_NEURONS_MATCHED, N_SUBSAMPLES,
                base_seed=(hash(origin_a + origin_b + area)) % 2**31,
            )
        else:
            mean_acc, std_acc = evaluate(X_full, y_pair), 0.0

        pairwise_results.append({
            "pair": f"{origin_a}_vs_{origin_b}",
            "origin_a": origin_a, "origin_b": origin_b,
            "area": area, "acc_mean": mean_acc, "acc_std": std_acc,
            "chance": 0.5,
            "n_neurons": min(N_NEURONS_MATCHED, X_full.shape[1]),
        })
        print(f"    {area:3s}  acc={mean_acc:.3f} ± {std_acc:.3f}")

pairwise_df = pd.DataFrame(pairwise_results)
pairwise_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_pairwise_accuracy_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# PART 3: 5-class full classification
# ======================================================================
print("\n" + "=" * 70)
print("PART 3: 5-class full classification")
print("=" * 70)

class_labels_5 = ["Cinematic", "Sports1M", "Rendered", "Monet2", "Trippy"]
fiveclass_results: list[dict] = []
cm_5class: dict[str, np.ndarray] = {}

for area in AREAS:
    X_full = X_by_area[area]
    X_sub = subsample_neurons(X_full, N_NEURONS_MATCHED, rng)

    if X_full.shape[1] > N_NEURONS_MATCHED:
        mean_acc, std_acc = evaluate_subsampled(
            X_full, y_label, N_NEURONS_MATCHED, N_SUBSAMPLES,
            base_seed=(hash("5class" + area)) % 2**31,
        )
    else:
        mean_acc, std_acc = evaluate(X_full, y_label), 0.0

    fiveclass_results.append({
        "area": area, "task": "5class_full",
        "acc_mean": mean_acc, "acc_std": std_acc,
        "chance": 1 / 5, "n_neurons": min(N_NEURONS_MATCHED, X_full.shape[1]),
    })

    y_pred = cv_predictions(X_sub, y_label)
    cm_5class[area] = confusion_matrix(y_label, y_pred, labels=class_labels_5, normalize="true")

    print(f"  {area:3s}  acc={mean_acc:.3f} ± {std_acc:.3f}")

fiveclass_df = pd.DataFrame(fiveclass_results)
fiveclass_df.to_csv(
    RESULTS_DIR / "csv" / f"q3_fiveclass_accuracy_{CHOSEN_SESSION}.csv", index=False,
)

cm5_rows: list[dict] = []
for area in AREAS:
    cm = cm_5class[area]
    for i, true_lbl in enumerate(class_labels_5):
        for j, pred_lbl in enumerate(class_labels_5):
            cm5_rows.append({
                "area": area, "true": true_lbl, "predicted": pred_lbl,
                "proportion": cm[i, j],
            })
pd.DataFrame(cm5_rows).to_csv(
    RESULTS_DIR / "csv" / f"q3_confusion_5class_{CHOSEN_SESSION}.csv", index=False,
)


# ======================================================================
# Save everything to .npz
# ======================================================================
save_dict: dict[str, np.ndarray] = {
    "origin_labels": np.array(origin_labels),
    "class_labels_5": np.array(class_labels_5),
    "y_origin": y_origin,
    "y_label": y_label,
}

for area in AREAS:
    save_dict[f"cm_3class_{area}"] = cm_3class[area]
    save_dict[f"cm_5class_{area}"] = cm_5class[area]

out_path = RESULTS_DIR / f"q3_subgroup_decode_{CHOSEN_SESSION}.npz"
np.savez_compressed(out_path, **save_dict)
print(f"\nSaved to {out_path}")

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print("All step 3 analyses complete. Files written:")
print("=" * 70)
print(f"  q3_subgroup_decode_{CHOSEN_SESSION}.npz")
print(f"  csv/q3_threeclass_accuracy_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_pairwise_accuracy_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_fiveclass_accuracy_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_confusion_3class_{CHOSEN_SESSION}.csv")
print(f"  csv/q3_confusion_5class_{CHOSEN_SESSION}.csv")
