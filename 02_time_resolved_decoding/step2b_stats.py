import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_1samp

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

# -------------------------
# CONFIG
# -------------------------
# Session and temporal window (w=1, w=3, w=5, etc.)
CHOSEN_SESSION = sys.argv[1] if len(sys.argv) > 1 else "5_6"
W_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 1

AREAS = ["V1", "LM", "AL", "RL"]
CHANCE = 1 / 3  # 3-class decoding baseline
N_SPLITS = 5
RANDOM_STATE = 42

# Number of shuffle repetitions for null distribution
N_SHUFFLES = 50

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
FEATURES_PATH = RESULTS_DIR / f"q2_features_{CHOSEN_SESSION}.npz"

# Load correct decode file depending on window size
if W_ARG == 1:
    DECODE_PATH = RESULTS_DIR / f"q2_decode_{CHOSEN_SESSION}.npz"
else:
    DECODE_PATH = RESULTS_DIR / f"q2_decode_w{W_ARG}_{CHOSEN_SESSION}.npz"

OUT_CSV = RESULTS_DIR / f"q2_stats_lr_w{W_ARG}_{CHOSEN_SESSION}.csv"
OUT_NULL = RESULTS_DIR / f"q2_shuffle_null_lr_w{W_ARG}_{CHOSEN_SESSION}.npz"

print(f"Session: {CHOSEN_SESSION}")
print(f"Window: w={W_ARG}")
print(f"Loading decode: {DECODE_PATH}")
print(f"Loading features: {FEATURES_PATH}")

# -------------------------
# LOAD DATA
# -------------------------
# Load decode results (accuracy curves, seeds, etc.)
decode = np.load(DECODE_PATH, allow_pickle=True)

# Load features (neural activity)
features = np.load(FEATURES_PATH, allow_pickle=True)

X_by_area = {
    "V1": features["X_V1"],
    "LM": features["X_LM"],
    "AL": features["X_AL"],
    "RL": features["X_RL"],
}
y = features["y"]
groups = features["groups"]

# Matched neuron count used during decoding
N_NEURONS_SUBSAMPLE = int(decode["n_neurons_subsample"])

# GroupKFold to avoid leakage across repeated clips
gkf = GroupKFold(n_splits=N_SPLITS)

# -------------------------
# STATISTICS PER AREA
# -------------------------
rows = []
null_by_area = {}

for area in AREAS:
    print(f"\nStats for area: {area}")

    # Seed-level accuracy curves (N_SEEDS x n_time)
    acc_seeds = decode[f"acc_lr_seeds_{area}"]

    # Peak accuracy per seed
    peak_by_seed = acc_seeds.max(axis=1)

    # Mean accuracy curve across seeds
    mean_curve = acc_seeds.mean(axis=0)

    # Timepoint of maximum decoding performance
    peak_t = int(mean_curve.argmax())

    # True peak accuracy (from mean curve)
    true_peak = float(mean_curve[peak_t])

    # -------------------------
    # T-TEST VS CHANCE
    # -------------------------
    # One-sample t-test: is decoding > chance?
    t_stat, p_ttest = ttest_1samp(
        peak_by_seed,
        popmean=CHANCE,
        alternative="greater",
    )

    # -------------------------
    # SHUFFLE-LABEL NULL
    # -------------------------
    # Build null distribution by permuting labels
    X_full = X_by_area[area]
    n_trials, n_neurons, n_time = X_full.shape

    null_peaks = np.zeros(N_SHUFFLES, dtype=float)

    for sh in range(N_SHUFFLES):
        rng = np.random.default_rng(RANDOM_STATE + 10000 + sh)

        # Shuffle labels (keep groups unchanged!)
        y_shuff = rng.permutation(y)

        seed_accs = []

        # Repeat decoding with same neuron subsampling strategy
        for s in range(acc_seeds.shape[0]):
            rng_seed = np.random.default_rng(RANDOM_STATE + s)
            keep_idx = rng_seed.choice(
                n_neurons,
                size=N_NEURONS_SUBSAMPLE,
                replace=False,
            )

            # Evaluate only at peak timepoint (fast approximation)
            if W_ARG == 1:
                Xt = X_full[:, keep_idx, peak_t]
            else:
                t0 = max(0, peak_t - W_ARG + 1)
                Xt = X_full[:, keep_idx, t0:peak_t+1].mean(axis=2)

            fold_accs = []

            # Same GroupKFold as real decoding
            for train_idx, test_idx in gkf.split(Xt, y_shuff, groups=groups):
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

                clf.fit(Xt[train_idx], y_shuff[train_idx])
                pred = clf.predict(Xt[test_idx])

                fold_accs.append(
                    balanced_accuracy_score(y_shuff[test_idx], pred)
                )

            seed_accs.append(np.mean(fold_accs))

        # Average across seeds → one value per shuffle
        null_peaks[sh] = np.mean(seed_accs)

        if (sh + 1) % 10 == 0:
            print(f"  shuffle {sh+1}/{N_SHUFFLES}")

    # Empirical p-value (right-tailed)
    p_shuffle = (np.sum(null_peaks >= true_peak) + 1) / (N_SHUFFLES + 1)

    null_by_area[area] = null_peaks

    # -------------------------
    # STORE RESULTS
    # -------------------------
    rows.append({
        "session": CHOSEN_SESSION,
        "window": W_ARG,
        "classifier": "LogisticRegression",
        "area": area,
        "chance": CHANCE,
        "n_seeds": acc_seeds.shape[0],
        "n_shuffles": N_SHUFFLES,
        "peak_time": peak_t,
        "true_peak_accuracy": true_peak,
        "peak_accuracy_mean_across_seeds": float(peak_by_seed.mean()),
        "peak_accuracy_std_across_seeds": float(peak_by_seed.std(ddof=1)),
        "t_statistic_vs_chance": float(t_stat),
        "p_value_ttest_vs_chance": float(p_ttest),
        "shuffle_null_mean": float(null_peaks.mean()),
        "shuffle_null_std": float(null_peaks.std(ddof=1)),
        "p_value_shuffle": float(p_shuffle),
    })

# -------------------------
# SAVE RESULTS
# -------------------------
df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

# Save full null distributions for later analysis/plots
save_dict = {
    "areas": np.array(AREAS),
    "window": W_ARG,
    "n_shuffles": N_SHUFFLES,
}

for area in AREAS:
    save_dict[f"null_peaks_{area}"] = null_by_area[area]

np.savez_compressed(OUT_NULL, **save_dict)

# -------------------------
# SUMMARY
# -------------------------
print("\nStats summary:")
print(df.to_string(index=False))
print(f"\nSaved stats CSV to: {OUT_CSV}")
print(f"Saved shuffle null to: {OUT_NULL}")