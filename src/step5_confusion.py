"""
Q1c confusion matrices: one per area at matched neuron count.
Reveals which pairs of natural clips are confusable in each area.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

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
N_MIN = 575
N_FOLDS = 5
RANDOM_STATE = 42

def make_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])

# Load features
data = np.load(RESULTS_DIR / f"features_{CHOSEN_SESSION}.npz", allow_pickle=True)
X_by_area = {area: data[f"X_{area}"] for area in AREAS}
y_label = data["y_label"]

# Restrict to Q1c (three natural classes)
q1c_mask = np.isin(y_label, ["Cinematic", "Sports1M", "Rendered"])
y_q1c = y_label[q1c_mask]
class_labels = ["Cinematic", "Sports1M", "Rendered"]

# Collect cross-validated predictions per area
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
rng = np.random.default_rng(RANDOM_STATE)

for ax, area in zip(axes, AREAS):
    X_full = X_by_area[area][q1c_mask]
    # Subsample to matched count (AL uses its full population)
    if X_full.shape[1] > N_MIN:
        idx = rng.choice(X_full.shape[1], size=N_MIN, replace=False)
        X = X_full[:, idx]
    else:
        X = X_full

    # Collect predictions from CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_all = np.empty_like(y_q1c)
    for tr, te in skf.split(X, y_q1c):
        clf = make_classifier()
        clf.fit(X[tr], y_q1c[tr])
        y_pred_all[te] = clf.predict(X[te])

    # Row-normalised confusion matrix (each row sums to 1)
    cm = confusion_matrix(y_q1c, y_pred_all, labels=class_labels, normalize="true")

    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels,
                vmin=0, vmax=1, cbar=(area == AREAS[-1]), ax=ax,
                square=True, annot_kws={"size": 10})
    ax.set_title(f"{area}  (n={min(N_MIN, X_full.shape[1])})")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true" if area == AREAS[0] else "")

fig.suptitle(f"Q1c confusion matrices — session {CHOSEN_SESSION}", y=1.02)
fig.tight_layout()
fig.savefig(RESULTS_DIR / f"confusion_q1c_{CHOSEN_SESSION}.png", dpi=150, bbox_inches="tight")
fig.savefig(RESULTS_DIR / f"confusion_q1c_{CHOSEN_SESSION}.pdf", bbox_inches="tight")
plt.close()

# Also save the raw confusion matrices to CSV for the report
rows = []
for area in AREAS:
    X_full = X_by_area[area][q1c_mask]
    if X_full.shape[1] > N_MIN:
        rng2 = np.random.default_rng(RANDOM_STATE)
        idx = rng2.choice(X_full.shape[1], size=N_MIN, replace=False)
        X = X_full[:, idx]
    else:
        X = X_full
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_all = np.empty_like(y_q1c)
    for tr, te in skf.split(X, y_q1c):
        clf = make_classifier()
        clf.fit(X[tr], y_q1c[tr])
        y_pred_all[te] = clf.predict(X[te])
    cm = confusion_matrix(y_q1c, y_pred_all, labels=class_labels, normalize="true")
    for i, true_lbl in enumerate(class_labels):
        for j, pred_lbl in enumerate(class_labels):
            rows.append({
                "area": area, "true": true_lbl, "predicted": pred_lbl,
                "proportion": cm[i, j],
            })

pd.DataFrame(rows).to_csv(RESULTS_DIR / f"results_step5_confusion_{CHOSEN_SESSION}.csv", index=False)
print(f"Saved confusion_q1c_{CHOSEN_SESSION}.png/.pdf and results CSV")