import os
import sys
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
N_SEEDS = 10  # number of random subsamplings to average over
NEURON_COUNTS = [25, 50, 100, 150, 200, 300, 400, 575]  # for neuron-count sweep

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
# Subsampling is done per seed inside the decode loop below
# (multi-seed averaging stabilises estimates across random subsets)
for area in AREAS:
    n_neurons = X_by_area[area].shape[1]
    n_keep = min(N_NEURONS_SUBSAMPLE, n_neurons)
    print(f"{area}: will use {n_keep}/{n_neurons} neurons per seed")

# -------------------------
# GROUPED CV
# -------------------------
# GroupKFold ensures trials from the same clip (same hash)
# are never split across train/test (no leakage)
gkf = GroupKFold(n_splits=N_SPLITS)

# -------------------------
# DECODE OVER TIME
# -------------------------
# For each timepoint t, train/test LR and SVM on X[:, :, t]
# Repeated N_SEEDS times with different random neuron subsets
# acc_seeds[clf][area] -> (N_SEEDS, n_time)
acc_seeds = {"lr": {}, "svm": {}}

for area in AREAS:
    X_full = X_by_area[area]   # (n_trials, n_neurons, n_time)
    n_trials, n_neurons, n_time = X_full.shape

    print(f"\nDecoding area: {area}")
    print(f"Trials={n_trials}, neurons={n_neurons}, timepoints={n_time}")

    for clf_name in ("lr", "svm"):
        seeds_acc = np.zeros((N_SEEDS, n_time), dtype=float)

        for s in range(N_SEEDS):
            # Fresh random neuron subset per seed
            rng = np.random.default_rng(RANDOM_STATE + s)
            n_keep = min(N_NEURONS_SUBSAMPLE, n_neurons)
            keep_idx = rng.choice(n_neurons, size=n_keep, replace=False)
            X = X_full[:, keep_idx, :]

            for t in range(n_time):
                Xt = X[:, :, t]   # (n_trials, n_neurons)

                fold_accs = []
                for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                    X_train, X_test = Xt[train_idx], Xt[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if clf_name == "lr":
                        clf = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                penalty="l2", C=1.0, class_weight="balanced",
                                max_iter=2000, random_state=RANDOM_STATE + s
                            )
                        )
                    else:
                        clf = make_pipeline(
                            StandardScaler(),
                            LinearSVC(
                                C=1.0, class_weight="balanced",
                                max_iter=2000, random_state=RANDOM_STATE + s
                            )
                        )

                    clf.fit(X_train, y_train)
                    fold_accs.append(balanced_accuracy_score(y_test, clf.predict(X_test)))

                seeds_acc[s, t] = np.mean(fold_accs)

            if (s + 1) % 5 == 0:
                print(f"  {clf_name.upper()} seed {s+1}/{N_SEEDS}  peak={seeds_acc[s].max():.3f}")

        acc_seeds[clf_name][area] = seeds_acc
        mean_acc = seeds_acc.mean(axis=0)
        print(f"  {clf_name.upper()} {area}: peak={mean_acc.max():.3f} ± {seeds_acc.max(axis=1).std():.3f}")

# convenience: mean and std across seeds per area
acc_by_area    = {area: acc_seeds["lr"][area].mean(axis=0) for area in AREAS}  # LR mean (backward-compat)
acc_std_by_area = {area: acc_seeds["lr"][area].std(axis=0) for area in AREAS}

# -------------------------
# CONFUSION MATRICES
# -------------------------
# At peak timepoint (from mean LR curve), summed across seeds for stability
print("\nComputing confusion matrices (peak timepoint, LR + SVM)...")

labels = np.unique(y)
n_classes = len(labels)
cm_by_area = {"lr": {}, "svm": {}}

for area in AREAS:
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape

    # Use peak timepoint from mean LR curve
    peak_t = acc_by_area[area].argmax()
    print(f"  {area}: peak_t={peak_t}")

    for clf_name in ("lr", "svm"):
        cm_sum = np.zeros((n_classes, n_classes), dtype=int)

        for s in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_STATE + s)
            n_keep = min(N_NEURONS_SUBSAMPLE, n_neurons)
            keep_idx = rng.choice(n_neurons, size=n_keep, replace=False)
            Xt = X_full[:, keep_idx, peak_t]

            y_true_all, y_pred_all = [], []
            for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                X_train, X_test = Xt[train_idx], Xt[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if clf_name == "lr":
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="l2", C=1.0, class_weight="balanced",
                            max_iter=2000, random_state=RANDOM_STATE + s
                        )
                    )
                else:
                    clf = make_pipeline(
                        StandardScaler(),
                        LinearSVC(
                            C=1.0, class_weight="balanced",
                            max_iter=2000, random_state=RANDOM_STATE + s
                        )
                    )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

            cm_sum += confusion_matrix(y_true_all, y_pred_all, labels=labels)

        cm_by_area[clf_name][area] = cm_sum
        print(f"  {clf_name.upper()} {area} CM (summed):\n{cm_sum}")

# -------------------------
# NEURON COUNT SWEEP
# -------------------------
# Accuracy vs number of neurons (LR, mean over time and seeds)
nc_acc = {}   # area -> (n_counts, N_SEEDS)

for area in AREAS:
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape
    print(f"\nNeuron sweep {area} ({n_neurons} neurons)")

    results = np.zeros((len(NEURON_COUNTS), N_SEEDS))

    for ni, n_count in enumerate(NEURON_COUNTS):
        n_keep = min(n_count, n_neurons)
        for s in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng.choice(n_neurons, size=n_keep, replace=False)
            X = X_full[:, keep_idx, :]

            acc_t = np.zeros(n_time)
            for t in range(n_time):
                Xt = X[:, :, t]
                fold_accs = []
                for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="l2", C=1.0, class_weight="balanced",
                            max_iter=2000, random_state=RANDOM_STATE + s
                        )
                    )
                    clf.fit(Xt[train_idx], y[train_idx])
                    fold_accs.append(balanced_accuracy_score(y[test_idx], clf.predict(Xt[test_idx])))
                acc_t[t] = np.mean(fold_accs)

            results[ni, s] = acc_t.mean()

        print(f"  n={n_keep:4d}: {results[ni].mean():.3f} ± {results[ni].std():.3f}")

    nc_acc[area] = results

# -------------------------
# SAVE
# -------------------------
save_dict = dict(
    labels=labels,
    n_time=n_time,
    n_neurons_subsample=N_NEURONS_SUBSAMPLE,
    neuron_counts=np.array(NEURON_COUNTS),
)

for area in AREAS:
    save_dict[f"acc_lr_{area}"]      = acc_by_area[area]
    save_dict[f"acc_lr_std_{area}"]  = acc_std_by_area[area]
    save_dict[f"acc_svm_{area}"]     = acc_seeds["svm"][area].mean(axis=0)
    save_dict[f"acc_svm_std_{area}"] = acc_seeds["svm"][area].std(axis=0)
    save_dict[f"cm_lr_{area}"]       = cm_by_area["lr"][area]
    save_dict[f"cm_svm_{area}"]      = cm_by_area["svm"][area]
    save_dict[f"nc_acc_mean_{area}"] = nc_acc[area].mean(axis=1)
    save_dict[f"nc_acc_std_{area}"]  = nc_acc[area].std(axis=1)

np.savez_compressed(OUT_PATH, **save_dict)
print(f"\nSaved decoding results to: {OUT_PATH}")