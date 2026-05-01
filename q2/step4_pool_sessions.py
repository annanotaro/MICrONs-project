"""
Pool Q2 decoding results across all sessions.

For each area and classifier (LR / SVM), averages the per-timepoint accuracy
curves across sessions, then plots:
  1. Pooled accuracy curves (mean +/- SEM across sessions)
  2. Per-session peak accuracy to show variability

Run after step2_decode.py has been executed for each session.
Pass --clean to pool clean (behaviorally-regressed) results.
Pass --both to overlay raw and clean.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------
# ARGS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--clean", action="store_true")
parser.add_argument("--both",  action="store_true")
args = parser.parse_args()

Q2_DIR     = Path(__file__).parent
RESULTS_DIR = Q2_DIR / "results"
OUT_DIR     = Q2_DIR / "results" / "pooled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AREAS  = ["V1", "LM", "AL", "RL"]
CHANCE = 1 / 3

# -------------------------
# COLLECT SESSION RESULTS
# -------------------------
def collect(clean):
    suffix = "clean" if clean else "raw"
    session_dirs = sorted([d for d in RESULTS_DIR.iterdir()
                           if d.is_dir() and d.name != "pooled"])
    pool = {area: {"lr": [], "svm": []} for area in AREAS}
    sessions_found = []

    for sess_dir in session_dirs:
        f = sess_dir / f"q2_decode_{suffix}_{sess_dir.name}.npz"
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        sessions_found.append(sess_dir.name)
        for area in AREAS:
            pool[area]["lr"].append(d[f"acc_lr_{area}"])
            pool[area]["svm"].append(d[f"acc_svm_{area}"])

    print(f"[{suffix}] found {len(sessions_found)} sessions: {sessions_found}")
    return pool, sessions_found

# -------------------------
# POOL (truncate to minimum n_time, then average)
# -------------------------
def pool_curves(curves):
    T = min(c.shape[0] for c in curves)
    mat = np.stack([c[:T] for c in curves])   # (n_sessions, T)
    return mat.mean(axis=0), mat.std(axis=0) / np.sqrt(len(curves)), mat

if args.both:
    data_raw,   sess_raw   = collect(clean=False)
    data_clean, sess_clean = collect(clean=True)
    datasets = {"raw": data_raw, "clean": data_clean}
elif args.clean:
    data, sessions = collect(clean=True)
    datasets = {"clean": data}
else:
    data, sessions = collect(clean=False)
    datasets = {"raw": data}

# -------------------------
# PLOT 1: Pooled accuracy over time (mean +/- SEM)
# -------------------------
colors_clf  = {"lr": "#2166ac", "svm": "#d6604d"}
linestyles  = {"raw": "-", "clean": "--"}
labels_clf  = {"lr": "LR", "svm": "SVM"}

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for ax, area in zip(axes, AREAS):
    for tag, pool in datasets.items():
        ls = linestyles[tag]
        for clf in ["lr", "svm"]:
            mean, sem, _ = pool_curves(pool[area][clf])
            t = np.arange(len(mean))
            label = f"{labels_clf[clf]} ({tag})" if args.both else labels_clf[clf]
            ax.plot(t, mean, color=colors_clf[clf], linestyle=ls,
                    label=label, linewidth=1.8)
            ax.fill_between(t, mean - sem, mean + sem,
                            color=colors_clf[clf], alpha=0.15)

    ax.axhline(CHANCE, color="black", linestyle=":", linewidth=1, label="Chance")
    ax.set_title(area, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time (frames)")
    ax.set_ylabel("Balanced accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

suffix_str = "both" if args.both else ("clean" if args.clean else "raw")
n_sess = len(list(datasets.values())[0][AREAS[0]]["lr"])
fig.suptitle(f"Q2 Pooled Temporal Decoding   |   {n_sess} sessions   |   {suffix_str}",
             fontsize=13, fontweight="bold")
plt.tight_layout()

out = OUT_DIR / f"q2_pooled_accuracy_time_{suffix_str}"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}.pdf/.png")

# -------------------------
# PLOT 2: Per-session peak accuracy (shows variability across sessions)
# -------------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

for ax, area in zip(axes, AREAS):
    for tag, pool in datasets.items():
        ls = linestyles[tag]
        for clf in ["lr", "svm"]:
            _, _, mat = pool_curves(pool[area][clf])
            peaks = mat.max(axis=1)   # (n_sessions,)
            x = np.arange(len(peaks))
            offset = 0.1 if clf == "svm" else -0.1
            label = f"{labels_clf[clf]} ({tag})" if args.both else labels_clf[clf]
            ax.scatter(x + offset, peaks, color=colors_clf[clf],
                       marker="o" if tag == "raw" else "^",
                       label=label, s=40, zorder=3)
            ax.plot(x + offset, peaks, color=colors_clf[clf],
                    linestyle=ls, alpha=0.5, linewidth=1)

    ax.axhline(CHANCE, color="black", linestyle=":", linewidth=1)
    ax.set_title(area, fontweight="bold")
    ax.set_xlabel("Session index")
    ax.set_ylabel("Peak balanced accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)

fig.suptitle(f"Peak Decoding Accuracy per Session   |   {suffix_str}",
             fontsize=12, fontweight="bold")
plt.tight_layout()

out = OUT_DIR / f"q2_peak_per_session_{suffix_str}"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}.pdf/.png")
