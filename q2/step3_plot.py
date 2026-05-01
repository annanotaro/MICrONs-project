import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files without opening windows
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

# -------------------------
# CONFIG
# -------------------------
import sys
CHOSEN_SESSION = sys.argv[1] if len(sys.argv) > 1 else "7_4"
W_ARG = int(sys.argv[2]) if len(sys.argv) > 2 else 1  # window size, e.g. python step3_plot.py 7_4 5

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
if W_ARG == 1:
    DATA_PATH = RESULTS_DIR / f"q2_decode_raw_{CHOSEN_SESSION}.npz"
else:
    DATA_PATH = RESULTS_DIR / f"q2_decode_raw_w{W_ARG}_{CHOSEN_SESSION}.npz"

data = np.load(DATA_PATH, allow_pickle=True)

areas = ["V1", "LM", "AL", "RL"]
chance = 1 / 3  # 3-class problem

WINDOW = int(data["window"]) if "window" in data else 1
W_LABEL = f"w={WINDOW} frame{'s' if WINDOW > 1 else ''}"

OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

def savefig(name):
    for ext in ("png", "pdf"):
        plt.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    print(f"Saved: {name}")
    plt.close()

def get_std(key):
    """Return std array if present, else zeros (for pre-existing files without std keys)."""
    if key in data:
        return data[key]
    base = key.replace("_lr_std_", "_lr_").replace("_svm_std_", "_svm_")
    return np.zeros(len(data[base]))

# -------------------------
# PLOT 1: Accuracy vs time — LR per area with std band
# -------------------------
plt.figure(figsize=(10, 6))

for area in areas:
    acc = data[f"acc_lr_{area}"]
    std = get_std(f"acc_lr_std_{area}")
    t = np.arange(len(acc))
    line, = plt.plot(t, acc, label=area, linewidth=2)
    plt.fill_between(t, acc - std, acc + std, alpha=0.20, color=line.get_color())

plt.axhline(chance, linestyle="--", color="black", label="Chance")
plt.xlabel("Time (frames)")
plt.ylabel("Balanced accuracy (LR, mean ± std)")
plt.title(f"Q2: Decoding accuracy over time  [{W_LABEL}]")
plt.legend()
plt.tight_layout()
savefig(f"q2_acc_time_lr_w{WINDOW}")
plt.show()

# -------------------------
# PLOT 1b: LR vs SVM side-by-side (one panel per area)
# -------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for ax, area in zip(axes, areas):
    for clf, color, lbl in [("acc_lr", "#2166ac", "LR"), ("acc_svm", "#d6604d", "SVM")]:
        acc = data[f"{clf}_{area}"]
        std = get_std(f"{clf}_std_{area}")
        t = np.arange(len(acc))
        ax.plot(t, acc, color=color, label=lbl, linewidth=2)
        ax.fill_between(t, acc - std, acc + std, color=color, alpha=0.20)

    ax.axhline(chance, linestyle="--", color="black", linewidth=1, label="Chance")
    ax.set_title(area, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(fontsize=8)

fig.suptitle(f"Q2: LR vs SVM accuracy over time  [{W_LABEL}]", fontsize=13)
plt.tight_layout()
savefig(f"q2_acc_time_lr_vs_svm_w{WINDOW}")
plt.show()

# -------------------------
# PLOT 2: Peak accuracy per area — LR vs SVM with error bars
# -------------------------
x = np.arange(len(areas))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))

for i, (clf, color, lbl) in enumerate([("acc_lr", "#2166ac", "LR"), ("acc_svm", "#d6604d", "SVM")]):
    peaks = [data[f"{clf}_{a}"].max() for a in areas]
    errs  = [get_std(f"{clf}_std_{a}")[data[f"{clf}_{a}"].argmax()] for a in areas]
    offset = (i - 0.5) * width
    ax.bar(x + offset, peaks, width, label=lbl, color=color, alpha=0.85)
    ax.errorbar(x + offset, peaks, yerr=errs,
                fmt="none", color="black", capsize=4, linewidth=1.2)

ax.axhline(chance, linestyle="--", color="black", label="Chance (1/3)")
ax.set_xticks(x)
ax.set_xticklabels(areas, fontsize=12)
ax.set_ylabel("Peak balanced accuracy")
ax.set_ylim(0, 1)
ax.set_title(f"Peak decoding accuracy per area  [{W_LABEL}]")
ax.legend()
plt.tight_layout()
savefig(f"q2_peak_acc_w{WINDOW}")
plt.show()

# -------------------------
# PLOT 3: Smoothed accuracy curves (LR and SVM)
# -------------------------
def smooth(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode='same')

for clf_key, clf_label, color in [("acc_lr", "LR", "#2166ac"), ("acc_svm", "SVM", "#d6604d")]:
    plt.figure(figsize=(10, 6))

    for area in areas:
        acc = data[f"{clf_key}_{area}"]
        plt.plot(smooth(acc, k=5), label=area)

    plt.axhline(chance, linestyle="--", color="black", label="Chance")
    plt.xlabel("Time (frames)")
    plt.ylabel(f"Balanced accuracy — {clf_label} (smoothed)")
    plt.title(f"Q2: Smoothed decoding accuracy over time — {clf_label}  [{W_LABEL}]")
    plt.legend()
    plt.tight_layout()
    savefig(f"q2_acc_time_smoothed_{clf_key.split('_')[1]}_w{WINDOW}")

# -------------------------
# PLOT 4: Confusion matrices — LR (top) and SVM (bottom), row-normalised
# -------------------------
labels = data["labels"]
n_classes = len(labels)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for row, (clf_key, clf_label) in enumerate([("cm_lr", "LR"), ("cm_svm", "SVM")]):
    for col, area in enumerate(areas):
        ax = axes[row, col]
        cm_raw = data[f"{clf_key}_{area}"].astype(float)

        # Row-normalise to get recall per class
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_raw / row_sums

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")

        for i in range(n_classes):
            for j in range(n_classes):
                val = cm_norm[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, color="white" if val > 0.6 else "black")

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        if col == 0:
            ax.set_ylabel(clf_label, fontsize=11, fontweight="bold")
        if row == 0:
            ax.set_title(area, fontsize=12, fontweight="bold")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Confusion matrices at peak timepoint (row-normalised)  [{W_LABEL}]", fontsize=13)
plt.tight_layout()
savefig(f"q2_confusion_matrices_w{WINDOW}")
plt.show()

# -------------------------
# PLOT 5: Accuracy vs number of neurons (LR)
# -------------------------
if "neuron_counts" not in data:
    print("Skipping neuron count plot — run step2 to generate nc data.")
else:
    nc = data["neuron_counts"]
    area_colors = {"V1": "#1b7837", "LM": "#762a83", "AL": "#e08214", "RL": "#2166ac"}

    plt.figure(figsize=(8, 5))

    for area in areas:
        mean = data[f"nc_acc_mean_{area}"]
        std  = data[f"nc_acc_lr_std_{area}"]
        plt.plot(nc, mean, marker="o", markersize=5, linewidth=2,
                 label=area, color=area_colors[area])
        plt.fill_between(nc, mean - std, mean + std,
                         color=area_colors[area], alpha=0.15)

    plt.axhline(chance, linestyle="--", color="black", label="Chance (1/3)")
    plt.xlabel("Number of neurons")
    plt.ylabel("Mean balanced accuracy (across time & seeds)")
    plt.title(f"Accuracy vs neuron count (LR)  [{W_LABEL}]")
    plt.legend()
    plt.tight_layout()
    savefig(f"q2_neuron_count_w{WINDOW}")
