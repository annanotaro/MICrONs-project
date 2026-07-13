import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files without opening windows
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "lines.linewidth": 2.0,
})

COLORS = {
    "LR": "#1f77b4",
    "SVM": "#d95f02",
    "V1": "#1f77b4",
    "LM": "#ff7f0e",
    "AL": "#2ca02c",
    "RL": "#d62728",
}

# -------------------------
# CONFIG
# -------------------------
import sys
CHOSEN_SESSION = sys.argv[1] if len(sys.argv) > 1 else "5_6"
W_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 1

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
if W_ARG == 1:
    DATA_PATH = RESULTS_DIR / f"q2_decode_{CHOSEN_SESSION}.npz"
else:
    DATA_PATH = RESULTS_DIR / f"q2_decode_w{W_ARG}_{CHOSEN_SESSION}.npz"

data = np.load(DATA_PATH, allow_pickle=True)

areas = ["V1", "LM", "AL", "RL"]
chance = 1 / 3

WINDOW = int(data["window"]) if "window" in data else W_ARG
W_LABEL = f"w={WINDOW} frame{'s' if WINDOW > 1 else ''}"

OUT_DIR = Path(__file__).parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    plt.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"Saved: {name}")
    plt.close()

def get_std(key):
    if key in data:
        return data[key]
    base = key.replace("_lr_std_", "_lr_").replace("_svm_std_", "_svm_")
    return np.zeros(len(data[base]))

# -------------------------
# PLOT 1: LR vs SVM over time — one area per row
# -------------------------
fig, axes = plt.subplots(4, 1, figsize=(7, 13), sharex=True, sharey=True)

for ax, area in zip(axes, areas):
    for clf, color, lbl in [
        ("acc_lr", COLORS["LR"], "LR"),
        ("acc_svm", COLORS["SVM"], "SVM"),
    ]:
        acc = data[f"{clf}_{area}"]
        t = np.arange(len(acc))
        ax.plot(t, acc, color=color, label=lbl, linewidth=2.0)

    ax.axhline(chance, linestyle="--", color="black", linewidth=1.1, label="Chance")
    ax.set_title(area, fontweight="bold")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(frameon=True, fontsize=9, loc="best")

axes[-1].set_xlabel("Time (frames)")
fig.suptitle(f"Q2: LR vs SVM accuracy over time  [{W_LABEL}]")
plt.tight_layout()
savefig(f"q2_acc_time_lr_vs_svm_w{WINDOW}")

# -------------------------
# PLOT 2: Peak accuracy per area — LR vs SVM with error bars
# -------------------------
x = np.arange(len(areas))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

for i, (clf, color, lbl) in enumerate([
    ("acc_lr", COLORS["LR"], "LR"),
    ("acc_svm", COLORS["SVM"], "SVM"),
]):
    peaks = [data[f"{clf}_{a}"].max() for a in areas]
    errs = [get_std(f"{clf}_std_{a}")[data[f"{clf}_{a}"].argmax()] for a in areas]

    offset = (i - 0.5) * width
    ax.bar(x + offset, peaks, width, label=lbl, color=color, alpha=0.85)
    ax.errorbar(
        x + offset,
        peaks,
        yerr=errs,
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.1,
    )

ax.axhline(chance, linestyle="--", color="black", label="Chance (1/3)", linewidth=1.1)
ax.set_xticks(x)
ax.set_xticklabels(areas)
ax.set_ylabel("Peak balanced accuracy")
ax.set_ylim(0.30, 0.55)
ax.set_title(f"Peak decoding accuracy per area  [{W_LABEL}]")
ax.legend(frameon=True)
plt.tight_layout()
savefig(f"q2_peak_acc_w{WINDOW}")

# -------------------------
# PLOT 3: Smoothed accuracy curves (LR and SVM)
# -------------------------
def smooth(x, k=5):
    return np.convolve(x, np.ones(k) / k, mode="same")

for clf_key, clf_label in [("acc_lr", "LR"), ("acc_svm", "SVM")]:
    plt.figure(figsize=(10, 6))

    for area in areas:
        acc = data[f"{clf_key}_{area}"]
        plt.plot(
            smooth(acc, k=5),
            label=area,
            color=COLORS[area],
            linewidth=2.0,
        )

    plt.axhline(chance, linestyle="--", color="black", label="Chance", linewidth=1.1)
    plt.xlabel("Time (frames)")
    plt.ylabel(f"Balanced accuracy — {clf_label} (smoothed)")
    plt.title(f"Q2: Smoothed decoding accuracy over time — {clf_label}  [{W_LABEL}]")
    plt.legend(frameon=True)
    plt.tight_layout()
    savefig(f"q2_acc_time_smoothed_{clf_key.split('_')[1]}_w{WINDOW}")

# -------------------------
# PLOT 4: Confusion matrices — 2 columns per row
# -------------------------
labels = data["labels"]
n_classes = len(labels)

fig, axes = plt.subplots(4, 2, figsize=(10, 16))

plot_positions = [
    ("cm_lr", "LR", "V1", 0, 0),
    ("cm_lr", "LR", "LM", 0, 1),
    ("cm_lr", "LR", "AL", 1, 0),
    ("cm_lr", "LR", "RL", 1, 1),
    ("cm_svm", "SVM", "V1", 2, 0),
    ("cm_svm", "SVM", "LM", 2, 1),
    ("cm_svm", "SVM", "AL", 3, 0),
    ("cm_svm", "SVM", "RL", 3, 1),
]

for clf_key, clf_label, area, r, c in plot_positions:
    ax = axes[r, c]
    cm_raw = data[f"{clf_key}_{area}"].astype(float)

    row_sums = cm_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_raw / row_sums

    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if val > 0.6 else "black",
            )

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(f"{clf_label} — {area}", fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Confusion matrices at peak timepoint (row-normalised)  [{W_LABEL}]")
plt.tight_layout()
savefig(f"q2_confusion_matrices_w{WINDOW}")