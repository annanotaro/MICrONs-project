import os
import sys
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

# -------------------------
# CONFIG
# -------------------------
# Configuration: session selection and file paths
if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
FEATURES_PATH = RESULTS_DIR / f"q2_features_{CHOSEN_SESSION}.npz"
OUT_PATH = RESULTS_DIR / f"q2_decode_{CHOSEN_SESSION}.npz"

AREAS = ["V1", "LM", "AL", "RL"]
N_SPLITS = 5

# Use a smaller matched neuron count for a fast first pass
N_NEURONS_SUBSAMPLE = 575
RANDOM_STATE = 42

print(f"Session: {CHOSEN_SESSION}")
print(f"Loading features from: {FEATURES_PATH}")

# -------------------------
# LOAD FEATURES
# -------------------------
# Load precomputed time-resolved features (output of step1)
data = np.load(FEATURES_PATH, allow_pickle=True)

X_by_area = {
    "V1": data["X_V1"],
    "LM": data["X_LM"],
    "AL": data["X_AL"],
    "RL": data["X_RL"],
}
y = data["y"]
groups = data["groups"]

print("\nUnique labels:", np.unique(y))
assert len(np.unique(y)) == 3, "Expected 3 classes (Cinematic, Sports1M, Rendered)"

print("\nLoaded arrays:")
for area in AREAS:
    print(f"{area}: {X_by_area[area].shape}")
print(f"y: {y.shape}")
print(f"groups: {groups.shape}")

# -------------------------
# SUBSAMPLE NEURONS
# -------------------------
# Use the same number of neurons in each area for a faster and fairer comparison
rng = np.random.default_rng(RANDOM_STATE)

for area in AREAS:
    X = X_by_area[area]
    n_trials, n_neurons, n_time = X.shape

    n_keep = min(N_NEURONS_SUBSAMPLE, n_neurons)
    keep_idx = rng.choice(n_neurons, size=n_keep, replace=False)

    X_by_area[area] = X[:, keep_idx, :]
    print(f"{area}: using {n_keep} neurons")

# -------------------------
# GROUPED CV
# -------------------------
# GroupKFold ensures trials from the same clip (same hash)
# are never split across train/test (no leakage)
gkf = GroupKFold(n_splits=N_SPLITS)

# -------------------------
# DECODE OVER TIME
# -------------------------
# For each timepoint t, train/test a classifier on X[:, :, t]
acc_by_area = {}

for area in AREAS:
    X = X_by_area[area]   # (n_trials, n_neurons, n_time)
    n_trials, n_neurons, n_time = X.shape

    print(f"\nDecoding area: {area}")
    print(f"Trials={n_trials}, neurons={n_neurons}, timepoints={n_time}")

    acc_t = np.zeros(n_time, dtype=float)

    for t in range(n_time):
        # Extract population activity at time t
        Xt = X[:, :, t]   # (n_trials, n_neurons)

        fold_accs = []

        for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
            X_train, X_test = Xt[train_idx], Xt[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fresh model for each fold
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = balanced_accuracy_score(y_test, y_pred)
            fold_accs.append(acc)

        acc_t[t] = np.mean(fold_accs)

        if (t + 1) % 10 == 0 or t == n_time - 1:
            print(f"  time {t+1}/{n_time}  mean bal acc = {acc_t[t]:.3f}")

    acc_by_area[area] = acc_t
    print(f"{area} done. Peak acc = {acc_t.max():.3f}")

# -------------------------
# CONFUSION MATRICES
# -------------------------
print("\nComputing confusion matrices (last timepoint)...")

cm_by_area = {}

for area in AREAS:
    X = X_by_area[area]
    Xt = X[:, :, -1]  # use last timepoint

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
        X_train, X_test = Xt[train_idx], Xt[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE
            )
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    labels = np.unique(y)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)

    cm_by_area[area] = cm

    print(f"{area} confusion matrix:\n{cm}")

# -------------------------
# SAVE
# -------------------------
# Save accuracy vs time curves for each area
np.savez_compressed(
    OUT_PATH,
    acc_V1=acc_by_area["V1"],
    acc_LM=acc_by_area["LM"],
    acc_AL=acc_by_area["AL"],
    acc_RL=acc_by_area["RL"],
    cm_V1=cm_by_area["V1"],
    cm_LM=cm_by_area["LM"],
    cm_AL=cm_by_area["AL"],
    cm_RL=cm_by_area["RL"],
    labels=labels,
    n_time=n_time,
    n_neurons_subsample=N_NEURONS_SUBSAMPLE
)

print(f"\nSaved decoding results to: {OUT_PATH}")

# ---------------------------------------------------------------
# TODO: Add SVM baseline
# ---------------------------------------------------------------
# Later: replicate the same pipeline using LinearSVC
# to check robustness across linear classifiers.