import os
import sys
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# -------------------------
# CONFIG
# -------------------------
if len(sys.argv) > 1:
    CHOSEN_SESSION = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "5_6")

W_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 1

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
FEATURES_PATH = RESULTS_DIR / f"q2_features_{CHOSEN_SESSION}.npz"
if W_ARG == 1:
    OUT_PATH = RESULTS_DIR / f"q2_decode_{CHOSEN_SESSION}.npz"
else:
    OUT_PATH = RESULTS_DIR / f"q2_decode_w{W_ARG}_{CHOSEN_SESSION}.npz"

AREAS = ["V1", "LM", "AL", "RL"]
N_SPLITS = 5
RANDOM_STATE = 42
N_SEEDS = 10

print(f"Session: {CHOSEN_SESSION}")
print(f"Loading features from: {FEATURES_PATH}")

# -------------------------
# LOAD FEATURES
# -------------------------
data = np.load(FEATURES_PATH, allow_pickle=True)

X_by_area = {
    "V1": data["X_V1"],
    "LM": data["X_LM"],
    "AL": data["X_AL"],
    "RL": data["X_RL"],
}
y = data["y"]
groups = data["groups"]

N_NEURONS_SUBSAMPLE = min(X_by_area[area].shape[1] for area in AREAS)

# Neuron-count sweep, capped at matched neuron count
NEURON_COUNTS = [25, 50, 100, 150, 200, 300, 400, N_NEURONS_SUBSAMPLE]
NEURON_COUNTS = sorted(set([n for n in NEURON_COUNTS if n <= N_NEURONS_SUBSAMPLE]))

print("\nUnique labels:", np.unique(y))
assert len(np.unique(y)) == 3, "Expected 3 classes (Cinematic, Sports1M, Rendered)"

print("\nLoaded arrays:")
for area in AREAS:
    print(f"{area}: {X_by_area[area].shape}")
print(f"y: {y.shape}")
print(f"groups: {groups.shape}")

print(f"\nMatched neuron count: {N_NEURONS_SUBSAMPLE}")

for area in AREAS:
    n_neurons = X_by_area[area].shape[1]
    print(f"{area}: will use {N_NEURONS_SUBSAMPLE}/{n_neurons} neurons per seed")

# -------------------------
# GROUPED CV
# -------------------------
gkf = GroupKFold(n_splits=N_SPLITS)

# -------------------------
# DECODE OVER TIME
# -------------------------
acc_seeds = {"lr": {}, "svm": {}}

for area in AREAS:
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape

    print(f"\nDecoding area: {area}")
    print(f"Trials={n_trials}, neurons={n_neurons}, timepoints={n_time}")

    for clf_name in ("lr", "svm"):
        seeds_acc = np.zeros((N_SEEDS, n_time), dtype=float)

        for s in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng.choice(n_neurons, size=N_NEURONS_SUBSAMPLE, replace=False)
            X = X_full[:, keep_idx, :]

            for t in range(n_time):
                if W_ARG == 1:
                    Xt = X[:, :, t]
                else:
                    t0 = max(0, t - W_ARG + 1)
                    Xt = X[:, :, t0:t+1].mean(axis=2)

                fold_accs = []
                for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                    X_train, X_test = Xt[train_idx], Xt[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    if clf_name == "lr":
                        clf = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(
                                penalty="l2",
                                C=1.0,
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=RANDOM_STATE + s,
                            ),
                        )
                    else:
                        clf = make_pipeline(
                            StandardScaler(),
                            LinearSVC(
                                C=1.0,
                                class_weight="balanced",
                                max_iter=2000,
                                random_state=RANDOM_STATE + s,
                            ),
                        )

                    clf.fit(X_train, y_train)
                    pred = clf.predict(X_test)
                    fold_accs.append(balanced_accuracy_score(y_test, pred))

                seeds_acc[s, t] = np.mean(fold_accs)

            if (s + 1) % 5 == 0:
                print(
                    f"  {clf_name.upper()} seed {s+1}/{N_SEEDS} "
                    f"peak={seeds_acc[s].max():.3f}"
                )

        acc_seeds[clf_name][area] = seeds_acc
        mean_acc = seeds_acc.mean(axis=0)
        print(
            f"  {clf_name.upper()} {area}: "
            f"peak={mean_acc.max():.3f} ± {seeds_acc.max(axis=1).std():.3f}"
        )

acc_by_area = {
    area: acc_seeds["lr"][area].mean(axis=0)
    for area in AREAS
}
acc_std_by_area = {
    area: acc_seeds["lr"][area].std(axis=0)
    for area in AREAS
}

# -------------------------
# CONFUSION MATRICES
# -------------------------
print("\nComputing confusion matrices (peak timepoint, LR + SVM)...")

labels = np.unique(y)
n_classes = len(labels)
cm_by_area = {"lr": {}, "svm": {}}

for area in AREAS:
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape

    peak_t = acc_by_area[area].argmax()
    print(f"  {area}: peak_t={peak_t}")

    for clf_name in ("lr", "svm"):
        cm_sum = np.zeros((n_classes, n_classes), dtype=int)

        for s in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng.choice(n_neurons, size=N_NEURONS_SUBSAMPLE, replace=False)
            if W_ARG == 1:
                Xt = X_full[:, keep_idx, peak_t]
            else:
                t0 = max(0, peak_t - W_ARG + 1)
                Xt = X_full[:, keep_idx, t0:peak_t+1].mean(axis=2)

            y_true_all, y_pred_all = [], []

            for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                X_train, X_test = Xt[train_idx], Xt[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if clf_name == "lr":
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE + s,
                        ),
                    )
                else:
                    clf = make_pipeline(
                        StandardScaler(),
                        LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE + s,
                        ),
                    )

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                y_true_all.extend(y_test)
                y_pred_all.extend(y_pred)

            cm_sum += confusion_matrix(y_true_all, y_pred_all, labels=labels)

        cm_by_area[clf_name][area] = cm_sum
        print(f"  {clf_name.upper()} {area} CM summed:\n{cm_sum}")

# -------------------------
# NEURON COUNT SWEEP
# -------------------------
nc_acc = {}

for area in AREAS:
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape

    print(f"\nNeuron sweep {area} ({n_neurons} neurons)")

    results = np.zeros((len(NEURON_COUNTS), N_SEEDS))

    for ni, n_count in enumerate(NEURON_COUNTS):
        for s in range(N_SEEDS):
            rng = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng.choice(n_neurons, size=n_count, replace=False)
            X = X_full[:, keep_idx, :]

            acc_t = np.zeros(n_time)

            for t in range(n_time):
                if W_ARG == 1:
                    Xt = X[:, :, t]
                else:
                    t0 = max(0, t - W_ARG + 1)
                    Xt = X[:, :, t0:t+1].mean(axis=2)
                fold_accs = []

                for train_idx, test_idx in gkf.split(Xt, y, groups=groups):
                    clf = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            penalty="l2",
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE + s,
                        ),
                    )

                    clf.fit(Xt[train_idx], y[train_idx])
                    pred = clf.predict(Xt[test_idx])
                    fold_accs.append(balanced_accuracy_score(y[test_idx], pred))

                acc_t[t] = np.mean(fold_accs)

            results[ni, s] = acc_t.mean()

        print(f"  n={n_count:4d}: {results[ni].mean():.3f} ± {results[ni].std():.3f}")

    nc_acc[area] = results

# -------------------------
# SAVE
# -------------------------
save_dict = dict(
    labels=labels,
    n_time=n_time,
    n_neurons_subsample=N_NEURONS_SUBSAMPLE,
    neuron_counts=np.array(NEURON_COUNTS),
    window=W_ARG,
)

for area in AREAS:
    save_dict[f"acc_lr_{area}"] = acc_by_area[area]
    save_dict[f"acc_lr_std_{area}"] = acc_std_by_area[area]
    save_dict[f"acc_svm_{area}"] = acc_seeds["svm"][area].mean(axis=0)
    save_dict[f"acc_svm_std_{area}"] = acc_seeds["svm"][area].std(axis=0)

    save_dict[f"acc_lr_seeds_{area}"] = acc_seeds["lr"][area]
    save_dict[f"acc_svm_seeds_{area}"] = acc_seeds["svm"][area]

    save_dict[f"cm_lr_{area}"] = cm_by_area["lr"][area]
    save_dict[f"cm_svm_{area}"] = cm_by_area["svm"][area]

    save_dict[f"nc_acc_mean_{area}"] = nc_acc[area].mean(axis=1)
    save_dict[f"nc_acc_std_{area}"] = nc_acc[area].std(axis=1)

np.savez_compressed(OUT_PATH, **save_dict)

print(f"\nSaved decoding results to: {OUT_PATH}")