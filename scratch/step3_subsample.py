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
N_SUBSAMPLES = 20
RANDOM_STATE = 42

data = np.load(f"features_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_label = data["y_label"]
y_natural = data["y_natural"]

# Bottleneck = smallest area
n_min = min(X.shape[1] for X in X_by_area.values())
print(f"n_min across areas: {n_min}")
print()

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
    for train_idx, test_idx in skf.split(X, y):
        clf = make_classifier()
        clf.fit(X[train_idx], y[train_idx])
        scores.append(balanced_accuracy_score(y[test_idx], clf.predict(X[test_idx])))
    return np.mean(scores)

def evaluate_subsampled(X_full, y, n_neurons, n_repeats, base_seed=0):
    """Run CV n_repeats times, each with a different random neuron subsample."""
    rng = np.random.default_rng(base_seed)
    scores = []
    for rep in range(n_repeats):
        idx = rng.choice(X_full.shape[1], size=n_neurons, replace=False)
        scores.append(evaluate(X_full[:, idx], y))
    return np.mean(scores), np.std(scores), scores

questions = [
    ("Q1a", np.ones(len(y_label), dtype=bool), y_natural),
    ("Q1b", np.isin(y_label, ["Monet2", "Trippy"]),
            (y_label[np.isin(y_label, ["Monet2", "Trippy"])] == "Trippy").astype(int)),
    ("Q1c", np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"]),
            y_label[np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"])]),
]

results = []
for qname, mask, y_q in questions:
    print(f"\n{'='*70}")
    print(f"{qname}: n_trials={mask.sum()}, n_classes={len(np.unique(y_q))}")
    print(f"{'='*70}")
    for area in AREAS:
        X = X_by_area[area][mask]
        # For the smallest area, subsampling to n_min == len(X.T) means 1 "repeat"
        # gives you the full data every time. Skip subsampling there.
        if X.shape[1] == n_min:
            acc = evaluate(X, y_q)
            mean, std = acc, 0.0
            print(f"  {area:3s}  (full pop, n={n_min}): {acc:.3f}")
        else:
            mean, std, _ = evaluate_subsampled(X, y_q, n_min, N_SUBSAMPLES,
                                                base_seed=hash(qname + area) % 2**31)
            print(f"  {area:3s}  (subsampled to n={n_min}, {N_SUBSAMPLES} reps): "
                  f"{mean:.3f} ± {std:.3f}")
        results.append({
            "question": qname, "area": area, "n_neurons": n_min,
            "acc_mean": mean, "acc_std": std,
        })

df = pd.DataFrame(results)
df.to_csv(f"results_step3_{CHOSEN_SESSION}.csv", index=False)
print("\n\nSummary:")
print(df.to_string(index=False))