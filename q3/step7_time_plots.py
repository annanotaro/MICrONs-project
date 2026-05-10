"""
Q3.2 Step 7: Publication-quality figures for time-resolved stability analysis.

Figures:
  1. Time-resolved accuracy — binary (natural vs parametric)
  2. Time-resolved accuracy — 3-class origin
  3. Stability distributions by stimulus type
  4. Stable vs unstable decoding comparison
  5. Stability heatmap (area x stimulus type)
  6. Peak accuracy vs mean stability scatter

All saved as PNG (150 dpi) + PDF.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# Load data
# ======================================================================
decode_data = np.load(
    RESULTS_DIR / f"q3_time_decode_{CHOSEN_SESSION}.npz", allow_pickle=True,
)
time_acc_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_time_accuracy_{CHOSEN_SESSION}.csv")
stab_df = pd.read_csv(RESULTS_DIR / "csv" / f"q3_stability_by_stimulus_{CHOSEN_SESSION}.csv")
stab_decode_df = pd.read_csv(
    RESULTS_DIR / "csv" / f"q3_stability_decoding_{CHOSEN_SESSION}.csv",
)

n_time = int(decode_data["n_time"])
time_axis = np.arange(n_time)


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


# ======================================================================
# FIGURE 1: Time-resolved accuracy — binary (natural vs parametric)
# ======================================================================
print("\nFigure 1: Time-resolved accuracy (binary)")

fig, ax = plt.subplots(figsize=(10, 5))
for area in AREAS:
    acc = decode_data[f"acc_binary_{area}"]
    std = decode_data[f"std_binary_{area}"]
    ax.plot(time_axis, acc, lw=2, color=AREA_COLORS[area], label=area)
    ax.fill_between(time_axis, acc - std, acc + std, alpha=0.15, color=AREA_COLORS[area])

ax.axhline(0.5, ls="--", color="gray", lw=1, label="chance")
ax.set_xlabel("Time (frames after onset)")
ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Q3.2: Natural vs Parametric — time-resolved — session {CHOSEN_SESSION}")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.35, 1.0)
fig.tight_layout()
save_fig(fig, f"q3_time_accuracy_binary_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 2: Time-resolved accuracy — 3-class origin
# ======================================================================
print("Figure 2: Time-resolved accuracy (3-class origin)")

fig, ax = plt.subplots(figsize=(10, 5))
for area in AREAS:
    acc = decode_data[f"acc_3class_{area}"]
    std = decode_data[f"std_3class_{area}"]
    ax.plot(time_axis, acc, lw=2, color=AREA_COLORS[area], label=area)
    ax.fill_between(time_axis, acc - std, acc + std, alpha=0.15, color=AREA_COLORS[area])

ax.axhline(1 / 3, ls="--", color="gray", lw=1, label="chance")
ax.set_xlabel("Time (frames after onset)")
ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Q3.2: Origin (filmed/rendered/parametric) — time-resolved — session {CHOSEN_SESSION}")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.2, 1.0)
fig.tight_layout()
save_fig(fig, f"q3_time_accuracy_3class_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 3: Stability distributions by stimulus type
# ======================================================================
print("Figure 3: Stability distributions by stimulus type")

stab_plot_df = stab_df[stab_df["label"] != "_nat_vs_par_test"].copy()

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, area in zip(axes, AREAS):
    area_data = stab_plot_df[stab_plot_df["area"] == area]

    nat_data = area_data[area_data["is_natural"] == 1].sort_values("label")
    par_data = area_data[area_data["is_natural"] == 0].sort_values("label")

    labels_all = list(nat_data["label"]) + list(par_data["label"])
    means_all = list(nat_data["cv_mean"]) + list(par_data["cv_mean"])
    stds_all = list(nat_data["cv_std"]) + list(par_data["cv_std"])

    colors = (["#2196F3"] * len(nat_data)) + (["#FF5722"] * len(par_data))

    x_pos = np.arange(len(labels_all))
    ax.bar(x_pos, means_all, yerr=stds_all, color=colors, alpha=0.7,
           edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_all, rotation=45, ha="right", fontsize=8)
    ax.set_title(area)
    ax.grid(axis="y", alpha=0.3)

    if area == AREAS[0]:
        ax.set_ylabel("Mean CV (stability)")

fig.suptitle(
    f"Q3.2: Firing-rate stability by stimulus type — session {CHOSEN_SESSION}\n"
    "(blue = natural, orange = parametric)",
    y=1.02,
)
fig.tight_layout()
save_fig(fig, f"q3_stability_distributions_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 4: Stable vs unstable decoding comparison
# ======================================================================
print("Figure 4: Stable vs unstable decoding comparison")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
axes_flat = axes.flatten()

for ax, area in zip(axes_flat, AREAS):
    stable_key = f"acc_binary_stable_{area}"
    unstable_key = f"acc_binary_unstable_{area}"

    if stable_key in decode_data and unstable_key in decode_data:
        acc_stable = decode_data[stable_key]
        acc_unstable = decode_data[unstable_key]

        ax.plot(time_axis, acc_stable, lw=2, color=AREA_COLORS[area],
                label="Stable (low CV)")
        ax.plot(time_axis, acc_unstable, lw=2, color=AREA_COLORS[area],
                ls="--", alpha=0.7, label="Unstable (high CV)")
        ax.axhline(0.5, ls=":", color="gray", lw=1)

        peak_diff = acc_stable.max() - acc_unstable.max()
        ax.annotate(
            f"$\\Delta$peak = {peak_diff:+.3f}",
            xy=(0.95, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.7},
        )
    else:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_title(area)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

axes[1, 0].set_xlabel("Time (frames after onset)")
axes[1, 1].set_xlabel("Time (frames after onset)")
axes[0, 0].set_ylabel("Balanced accuracy")
axes[1, 0].set_ylabel("Balanced accuracy")

fig.suptitle(
    f"Q3.2: Stable vs Unstable trials — nat/par decoding — session {CHOSEN_SESSION}",
    y=1.01,
)
fig.tight_layout()
save_fig(fig, f"q3_stable_vs_unstable_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 5: Stability heatmap (area x stimulus type)
# ======================================================================
print("Figure 5: Stability heatmap")

stim_labels = sorted(stab_plot_df["label"].unique())
heatmap_data = np.zeros((len(AREAS), len(stim_labels)))

for i, area in enumerate(AREAS):
    for j, label in enumerate(stim_labels):
        row = stab_plot_df[(stab_plot_df["area"] == area) & (stab_plot_df["label"] == label)]
        if len(row) > 0:
            heatmap_data[i, j] = row["cv_mean"].values[0]

fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(
    heatmap_data, annot=True, fmt=".2f", cmap="RdYlBu_r",
    xticklabels=stim_labels, yticklabels=AREAS,
    ax=ax, square=True, annot_kws={"size": 11},
)
ax.set_title(f"Q3.2: Mean CV by area × stimulus — session {CHOSEN_SESSION}")
ax.set_xlabel("Stimulus type")
ax.set_ylabel("Brain area")
fig.tight_layout()
save_fig(fig, f"q3_stability_heatmap_{CHOSEN_SESSION}")


# ======================================================================
# FIGURE 6: Peak accuracy vs mean stability scatter
# ======================================================================
print("Figure 6: Peak accuracy vs mean stability")

fig, ax = plt.subplots(figsize=(8, 6))

markers = {"Cinematic": "o", "Sports1M": "s", "Rendered": "^", "Monet2": "D", "Trippy": "v"}

for area in AREAS:
    for label in stim_labels:
        stab_row = stab_plot_df[
            (stab_plot_df["area"] == area) & (stab_plot_df["label"] == label)
        ]
        if len(stab_row) == 0:
            continue

        cv_val = stab_row["cv_mean"].values[0]

        # Peak accuracy for binary decoding at this area
        acc_row = time_acc_df[
            (time_acc_df["area"] == area) & (time_acc_df["task"] == "binary")
        ]
        if len(acc_row) == 0:
            continue
        peak_acc = acc_row["peak_acc"].values[0]

        ax.scatter(
            cv_val, peak_acc,
            color=AREA_COLORS[area],
            marker=markers.get(label, "o"),
            s=80, alpha=0.8, edgecolors="black", linewidths=0.5,
        )

# Legend for areas
for area in AREAS:
    ax.scatter([], [], color=AREA_COLORS[area], s=60, label=area)
ax.legend(loc="upper right", title="Area", fontsize=9)

# Legend for stimulus markers (separate)
for label, marker in markers.items():
    ax.scatter([], [], color="gray", marker=marker, s=60, label=label)

handles, labels_leg = ax.get_legend_handles_labels()
n_areas = len(AREAS)
leg1 = ax.legend(handles[:n_areas], labels_leg[:n_areas],
                 loc="upper right", title="Area", fontsize=8)
ax.add_artist(leg1)
ax.legend(handles[n_areas:], labels_leg[n_areas:],
          loc="lower left", title="Stimulus", fontsize=8)

ax.set_xlabel("Mean CV (higher = less stable)")
ax.set_ylabel("Peak binary decoding accuracy")
ax.set_title(f"Q3.2: Stability vs Decodability — session {CHOSEN_SESSION}")
ax.grid(alpha=0.3)
fig.tight_layout()
save_fig(fig, f"q3_stability_scatter_{CHOSEN_SESSION}")


# ======================================================================
# Done
# ======================================================================
print("\n" + "=" * 70)
print("All Q3.2 figures generated:")
print("=" * 70)
print(f"  q3_time_accuracy_binary_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_time_accuracy_3class_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_stability_distributions_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_stable_vs_unstable_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_stability_heatmap_{CHOSEN_SESSION}.png/.pdf")
print(f"  q3_stability_scatter_{CHOSEN_SESSION}.png/.pdf")
