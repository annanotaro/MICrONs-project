import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
SESSION = "5_6"

RESULTS_DIR = Path(__file__).parent / "results" / SESSION
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

NULL_W1 = np.load(RESULTS_DIR / f"q2_shuffle_null_lr_w1_{SESSION}.npz")
NULL_W5 = np.load(RESULTS_DIR / f"q2_shuffle_null_lr_w5_{SESSION}.npz")

df1 = pd.read_csv(RESULTS_DIR / f"q2_stats_lr_w1_{SESSION}.csv")
df5 = pd.read_csv(RESULTS_DIR / f"q2_stats_lr_w5_{SESSION}.csv")

AREAS = df1["area"].values
chance = 1 / 3

# -------------------------
# PLOT: Real vs Shuffle
# -------------------------
fig, axes = plt.subplots(4, 2, figsize=(10, 14), sharey=True)
axes = axes.flatten()

plot_idx = 0

for window_label, null_data, df in [
    ("w=1", NULL_W1, df1),
    ("w=5", NULL_W5, df5),
]:
    for area in AREAS:
        ax = axes[plot_idx]

        null_vals = null_data[f"null_peaks_{area}"]
        true_val = df[df["area"] == area]["true_peak_accuracy"].values[0]

        ax.hist(null_vals, bins=15, color="lightgray", edgecolor="black")
        ax.axvline(true_val, color="red", linewidth=2, label="Real")
        ax.axvline(chance, color="black", linestyle="--", label="Chance")

        ax.set_title(f"{area} ({window_label})")
        ax.set_xlabel("Balanced accuracy")

        if plot_idx % 2 == 0:
            ax.set_ylabel("Count")

        ax.legend(fontsize=8)

        plot_idx += 1

plt.suptitle("Real decoding accuracy vs shuffle-label null distribution", fontsize=14)
plt.tight_layout()

plt.savefig(PLOTS_DIR / "q2_stats_real_vs_shuffle.pdf", bbox_inches="tight")
print("Saved: q2_stats_real_vs_shuffle.pdf")