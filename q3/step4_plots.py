"""
Q3 Step 4: All publication-quality figures.

Reads .npz and CSV outputs from steps 2–3 and produces:
  1. Brain area bar chart — binary natural vs parametric with permutation nulls
  2. Learning curves — accuracy vs neuron count
  3. 3-way confusion matrices (one per area)
  4. Pairwise decoding heatmap (area × pair)
  5. Neuron discriminability violin plots (Cohen's d per area)
  6. 5-class confusion matrices (one per area)
  7. Summary radar chart

All saved as PNG (150 dpi) + PDF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from config import (
    AREAS,
    AREA_COLORS,
    CHOSEN_SESSION,
    RESULTS_DIR,
    ensure_results_dir,
)

ensure_results_dir()
print(f"Session: {CHOSEN_SESSION}")
print(f"Outputs → {RESULTS_DIR}")

# ======================================================================
# Load all data
# ======================================================================
binary_data = np.load(
    RESULTS_DIR / f"q3_binary_decode_{CHOSEN_SESSION}.npz", allow_pickle=True,
)
subgroup_data = np.load(
    RESULTS_DIR / f"q3_subgroup_decode_{CHOSEN_SESSION}.npz", allow_pickle=True,
)
lc_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_learning_curves_{CHOSEN_SESSION}.csv")
pairwise_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_pairwise_accuracy_{CHOSEN_SESSION}.csv")
threeclass_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_threeclass_accuracy_{CHOSEN_SESSION}.csv")
fiveclass_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_fiveclass_accuracy_{CHOSEN_SESSION}.csv")
null_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_permutation_nulls_{CHOSEN_SESSION}.csv")

origin_labels = list(subgroup_data["origin_labels"])
class_labels_5 = list(subgroup_data["class_labels_5"])


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


# ======================================================================
# FIGURE 1: Binary accuracy bar chart with permutation null violins
# ======================================================================
print("\nFigure 1: Binary accuracy with null distributions")

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(AREAS))
bar_width = 0.5

observed_vals = [float(binary_data[f"observed_{area}"]) for area in AREAS]
null_dists = [binary_data[f"null_{area}"] for area in AREAS]

parts = ax.violinplot(null_dists, positions=x_pos, showmedians=True, widths=0.6)
for pc in parts["bodies"]:
    pc.set_facecolor("lightgray")
    pc.set_alpha(0.6)
for key in ("cmins", "cmaxes", "cmedians", "cbars"):
    if key in parts:
        parts[key].set_color("gray")

ax.bar(x_pos, observed_vals, width=bar_width, alpha=0.7,
       color=[AREA_COLORS[a] for a in AREAS], edgecolor="black", linewidth=0.8,
       zorder=3)

ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
ax.set_xticks(x_pos)
ax.set_xticklabels(AREAS)
ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Q3: Natural vs Parametric — session {CHOSEN_SESSION}")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, f"q3_binary_accuracy_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 2: Learning curves
# ======================================================================
print("Figure 2: Learning curves")

fig, ax = plt.subplots(figsize=(8, 5))
for area in AREAS:
    adf = lc_df[lc_df["area"] == area].sort_values("n_neurons")
    ax.errorbar(adf["n_neurons"], adf["acc_mean"], yerr=adf["acc_std"],
                marker="o", capsize=3, lw=1.8, ms=6,
                color=AREA_COLORS[area], label=area)

ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
ax.set_xscale("log")
ax.set_xlabel("Number of neurons")
ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Q3: Learning curves (natural vs parametric) — session {CHOSEN_SESSION}")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
save_fig(fig, f"q3_learning_curves_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 3: 3-way confusion matrices
# ======================================================================
print("Figure 3: 3-class confusion matrices")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
for ax, area in zip(axes, AREAS):
    cm = subgroup_data[f"cm_3class_{area}"]
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=origin_labels, yticklabels=origin_labels,
                vmin=0, vmax=1, cbar=(area == AREAS[-1]), ax=ax,
                square=True, annot_kws={"size": 11})
    ax.set_title(f"{area}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true" if area == AREAS[0] else "")

fig.suptitle(f"Q3: 3-class confusion (filmed/rendered/parametric) — session {CHOSEN_SESSION}",
             y=1.02)
fig.tight_layout()
save_fig(fig, f"q3_confusion_3class_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 4: Pairwise decoding heatmap
# ======================================================================
print("Figure 4: Pairwise decoding heatmap")

pair_labels = pairwise_df["pair"].unique()
heatmap_data = np.zeros((len(AREAS), len(pair_labels)))
for i, area in enumerate(AREAS):
    for j, pair in enumerate(pair_labels):
        row = pairwise_df[(pairwise_df["area"] == area) & (pairwise_df["pair"] == pair)]
        heatmap_data[i, j] = row["acc_mean"].values[0]

pair_display = [p.replace("_vs_", " vs ") for p in pair_labels]

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=pair_display, yticklabels=AREAS,
            vmin=0.4, vmax=1.0, ax=ax, square=True,
            annot_kws={"size": 12})
ax.set_title(f"Q3: Pairwise decoding accuracy — session {CHOSEN_SESSION}")
ax.set_xlabel("Stimulus pair")
ax.set_ylabel("Brain area")
fig.tight_layout()
save_fig(fig, f"q3_pairwise_heatmap_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 5: Neuron discriminability violin plots
# ======================================================================
print("Figure 5: Neuron discriminability (Cohen's d)")

fig, ax = plt.subplots(figsize=(8, 5))
d_data = []
d_labels = []
for area in AREAS:
    d_vals = binary_data[f"cohens_d_{area}"]
    d_data.append(np.abs(d_vals))
    d_labels.extend([area] * len(d_vals))

parts = ax.violinplot(d_data, positions=range(len(AREAS)), showmedians=True, widths=0.7)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(AREA_COLORS[AREAS[i]])
    pc.set_alpha(0.6)

ax.axhline(0.2, ls="--", color="gray", lw=1, alpha=0.7, label="|d| = 0.2 (small)")
ax.set_xticks(range(len(AREAS)))
ax.set_xticklabels(AREAS)
ax.set_ylabel("|Cohen's d|")
ax.set_title(f"Q3: Per-neuron discriminability (natural vs parametric) — session {CHOSEN_SESSION}")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save_fig(fig, f"q3_discriminability_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 6: 5-class confusion matrices
# ======================================================================
print("Figure 6: 5-class confusion matrices")

fig, axes = plt.subplots(1, 4, figsize=(22, 5))
for ax, area in zip(axes, AREAS):
    cm = subgroup_data[f"cm_5class_{area}"]
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_labels_5, yticklabels=class_labels_5,
                vmin=0, vmax=1, cbar=(area == AREAS[-1]), ax=ax,
                square=True, annot_kws={"size": 9})
    ax.set_title(f"{area}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true" if area == AREAS[0] else "")
    ax.tick_params(axis="x", rotation=45)

fig.suptitle(f"Q3: 5-class confusion matrices — session {CHOSEN_SESSION}", y=1.02)
fig.tight_layout()
save_fig(fig, f"q3_confusion_5class_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 7: Summary radar chart
# ======================================================================
print("Figure 7: Summary radar chart")

task_names = ["Nat vs Par", "3-class origin", "5-class full"]
task_chances = [0.5, 1 / 3, 1 / 5]

scores_by_area: dict[str, list[float]] = {}
for area in AREAS:
    binary_acc = float(binary_data[f"observed_{area}"])
    three_acc = float(threeclass_df[threeclass_df["area"] == area]["acc_mean"].values[0])
    five_acc = float(fiveclass_df[fiveclass_df["area"] == area]["acc_mean"].values[0])
    scores_by_area[area] = [binary_acc, three_acc, five_acc]

# Add pairwise accuracies
for pair in pairwise_df["pair"].unique():
    pair_display_name = pair.replace("_vs_", " vs ")
    task_names.append(pair_display_name)
    task_chances.append(0.5)
    for area in AREAS:
        row = pairwise_df[(pairwise_df["area"] == area) & (pairwise_df["pair"] == pair)]
        scores_by_area[area].append(float(row["acc_mean"].values[0]))

n_tasks = len(task_names)
angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
angles.append(angles[0])

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
for area in AREAS:
    values = scores_by_area[area] + [scores_by_area[area][0]]
    ax.plot(angles, values, "o-", lw=1.8, ms=5,
            color=AREA_COLORS[area], label=area)
    ax.fill(angles, values, alpha=0.08, color=AREA_COLORS[area])

chance_values = task_chances + [task_chances[0]]
ax.plot(angles, chance_values, "--", color="gray", lw=1, label="chance")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(task_names, size=8)
ax.set_ylim(0, 1)
ax.set_title(f"Q3: Summary — session {CHOSEN_SESSION}", y=1.08, size=13)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
fig.tight_layout()
save_fig(fig, f"q3_radar_summary_{CHOSEN_SESSION}")


# ======================================================================
# Done
# ======================================================================
print("\n" + "=" * 70)
print("All figures generated:")
print("=" * 70)
print(f"  q3_binary_accuracy_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_learning_curves_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_confusion_3class_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_pairwise_heatmap_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_discriminability_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_confusion_5class_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_radar_summary_{CHOSEN_SESSION}.png/.pdf")
