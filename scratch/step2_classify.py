import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline

CHOSEN_SESSION = "7_4"
AREAS = ["V1", "LM", "AL", "RL"]
N_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Load features
# ---------------------------------------------------------------
data = np.load(f"features_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_label = data["y_label"]
y_natural = data["y_natural"]

print("Loaded feature matrices:")
for area in AREAS:
    print(f"  {area}: {X_by_area[area].shape}")
print(f"  y_label distribution: {pd.Series(y_label).value_counts().to_dict()}")
print()

# ---------------------------------------------------------------
# Classifier factory — L2 logistic regression, standardized features
# ---------------------------------------------------------------
def make_classifier(multiclass=False):
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

# ---------------------------------------------------------------
# CV evaluation function
# ---------------------------------------------------------------
def evaluate(X, y, multiclass=False):
    """Return balanced accuracy over stratified K-fold CV."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        clf = make_classifier(multiclass=multiclass)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], y_pred))
    return np.mean(scores), np.std(scores)

# ---------------------------------------------------------------
# Define the three sub-questions
# ---------------------------------------------------------------
# Q1a: natural vs parametric (all trials)
q1a_mask = np.ones(len(y_label), dtype=bool)
q1a_y = y_natural

# Q1b: Monet2 vs Trippy (only parametric trials)
q1b_mask = np.isin(y_label, ["Monet2", "Trippy"])
q1b_y = (y_label[q1b_mask] == "Trippy").astype(int)

# Q1c: Cinematic vs Sports1M vs Rendered (only natural trials)
q1c_mask = np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"])
q1c_y = y_label[q1c_mask]

questions = [
    ("Q1a", "natural vs parametric",  q1a_mask, q1a_y, False),
    ("Q1b", "Monet2 vs Trippy",       q1b_mask, q1b_y, False),
    ("Q1c", "3 natural clips",        q1c_mask, q1c_y, True),
]

# ---------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------
results = []

for qname, qdesc, mask, y_q, multiclass in questions:
    chance = 1.0 / len(np.unique(y_q))
    print(f"\n{'='*70}")
    print(f"{qname}: {qdesc}")
    print(f"  n_trials={mask.sum()}, n_classes={len(np.unique(y_q))}, chance={chance:.3f}")
    print(f"{'='*70}")

    for area in AREAS:
        X = X_by_area[area][mask]

        # Full-population decoder
        full_mean, full_std = evaluate(X, y_q, multiclass=multiclass)

        # Population-mean baseline: single scalar feature
        X_mean = X.mean(axis=1, keepdims=True)
        base_mean, base_std = evaluate(X_mean, y_q, multiclass=multiclass)

        # Lift = full - baseline
        lift = full_mean - base_mean

        print(f"  {area:3s}  full={full_mean:.3f}±{full_std:.3f}  "
              f"mean-baseline={base_mean:.3f}±{base_std:.3f}  "
              f"lift={lift:+.3f}")

        results.append({
            "question": qname,
            "area": area,
            "n_neurons": X.shape[1],
            "n_trials": X.shape[0],
            "full_acc": full_mean,
            "full_std": full_std,
            "baseline_acc": base_mean,
            "baseline_std": base_std,
            "lift": lift,
            "chance": chance,
        })

# ---------------------------------------------------------------
# Save results table
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"results_step2_{CHOSEN_SESSION}.csv", index=False)
print(f"\n\nResults saved to results_step2_{CHOSEN_SESSION}.csv")
print("\nSummary table:")
print(results_df.to_string(index=False))
