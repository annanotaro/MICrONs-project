# MICrONS Project Data and File Structure

This document is a single source of truth for how this repository is organized, where data comes from, and what each pipeline step reads/writes.

## What this project does

This repo contains two related decoding pipelines on MICrONS visual-session data:

- Q1 pipeline in `src/`: trial-level feature extraction, behavioral-regressed variants, learning curves, paired tests, permutation nulls, and confusion matrices.
- Q2 pipeline in `q2/`: time-resolved decoding on the 3 natural classes (`Cinematic`, `Sports1M`, `Rendered`).

The core raw dataset is an HDF5 file (`microns.h5`) plus the dataset reader module (`reader.py`) from the HuggingFace MICrONS snapshot.

---

## Repository layout

- `README.md` - quick run instructions and high-level pipeline summary.
- `requirements.txt` - Python dependencies.
- `.gitignore` - excludes virtual envs, raw/derived data artifacts, and `.env`.
- `.env` (local only, gitignored) - optional environment overrides.
- `CLAUDE.md` - this file.

- `src/` - main Q1 scripts (primary workflow).
  - `step0_explore_session.py`
  - `step1_features.py`
  - `step1b_behavioral_clean.py`
  - `step4_learning_curves.py`
  - `step4_learning_curves_CLEAN.py`
  - `step5_confusion.py`
  - `step5_confusion_CLEAN.py`
  - `run_all_sessions.py`

- `q2/` - Q2 time-resolved decoding workflow.
  - `step0_trial_labels.py`
  - `step1_features.py`
  - `step2_decode.py`
  - `step3_plot.py`
  - `results/<session>/...` (Q2 outputs are nested under `q2/results`)

- `q3/` - Q3 natural vs parametric deep-dive with central config.
  - `config.py` (single source of truth for paths, params, colors)
  - `step0_labels.py`
  - `step1_features.py`
  - `step2_natural_vs_parametric.py`
  - `step3_natural_subgroups.py`
  - `step4_plots.py`
  - `results/<session>/...` (Q3 outputs are nested under `q3/results`)

- `results/<session>/` - Q1 outputs by session (example: `results/7_4/`).
  - Root: figures (`.png`, `.pdf`) and feature bundles (`.npz`)
  - `csv/`: tabular result files (`results_step*.csv`)
  - `trials_<session>.csv` (in root of session folder)

- `utils/` - diagnostics/helpers for environment and dataset access checks.
- `scratch/` - exploratory/deprecated scripts retained for reference.

---

## Data locations and configuration

Scripts use two required external paths:

- `MICRONS_DATA_PATH` -> absolute path to `microns.h5`
- `MICRONS_READER_PATH` -> absolute path to `reader.py`

Behavior in code:

- In `src/step0_explore_session.py`, `src/step1_features.py`, and `src/step1b_behavioral_clean.py`, these are read from environment variables with hardcoded defaults.
- Other Q1 scripts read only derived files from `results/<session>/`.
- `q2` scripts currently use hardcoded local paths inside the script, not env vars.

Session selection:

- Most scripts accept a session id as first CLI arg (example: `7_4`).
- If omitted, they fall back to `CHOSEN_SESSION` env var or `"7_4"`.

---

## End-to-end data flow (Q1, `src/`)

All Q1 scripts write to:

- `results/<session>/`
- and create `results/<session>/csv/` when needed.

### 1) `step0_explore_session.py`

Inputs:

- Raw MICRONS dataset (`microns.h5`) via `MicronsReader`

Outputs:

- `results/<session>/trials_<session>.csv`

Adds trial metadata and labels:

- `trial_idx`, `hash`, `type`, `short_name`, `movie_name`, `label`, `is_natural`

### 2) `step1_features.py`

Inputs:

- `results/<session>/trials_<session>.csv`
- Raw responses from `microns.h5`

Outputs:

- `results/<session>/features_<session>.npz`

Saved arrays:

- `X_V1`, `X_LM`, `X_AL`, `X_RL` (trial x neurons)
- `y_label`, `y_natural`
- `trial_idx`, `hash`

### 3) `step1b_behavioral_clean.py`

Inputs:

- `results/<session>/trials_<session>.csv`
- Raw responses + pupil + treadmill from `microns.h5`

Outputs:

- `results/<session>/features_clean_<session>.npz`

Saved arrays:

- `X_V1`, `X_LM`, `X_AL`, `X_RL` (behavior-regressed features)
- `y_label`, `y_natural`

### 4) `step4_learning_curves.py`

Inputs:

- `results/<session>/features_<session>.npz`

Outputs:

- `results/<session>/learning_curves_<session>.png`
- `results/<session>/learning_curves_<session>.pdf`
- `results/<session>/results_step4_learning_curves_<session>.csv`
- `results/<session>/results_step4_paired_comparisons_<session>.csv`
- `results/<session>/results_step4_permutation_nulls_<session>.csv`

### 5) `step4_learning_curves_CLEAN.py`

Inputs:

- `results/<session>/features_clean_<session>.npz`

Outputs:

- `results/<session>/learning_curves_CLEAN_<session>.png`
- `results/<session>/learning_curves_CLEAN_<session>.pdf`
- `results/<session>/results_step4_learning_curves_CLEAN_<session>.csv`
- `results/<session>/results_step4_paired_comparisons_CLEAN_<session>.csv`
- `results/<session>/results_step4_permutation_nulls_CLEAN_<session>.csv`

### 6) `step5_confusion.py`

Inputs:

- `results/<session>/features_<session>.npz`

Outputs:

- `results/<session>/confusion_q1c_<session>.png`
- `results/<session>/confusion_q1c_<session>.pdf`
- `results/<session>/results_step5_confusion_<session>.csv`

### 7) `step5_confusion_CLEAN.py`

Inputs:

- `results/<session>/features_clean_<session>.npz`

Outputs:

- `results/<session>/confusion_q1c_CLEAN_<session>.png`
- `results/<session>/confusion_q1c_CLEAN_<session>.pdf`
- `results/<session>/results_step5_confusion_CLEAN_<session>.csv`

### Batch runner: `run_all_sessions.py`

- Runs all Q1 steps for either:
  - all viable sessions in the H5 file (filtered by minimum AL neuron count), or
  - user-specified sessions from CLI args.

---

## End-to-end data flow (Q2, `q2/`)

Q2 writes to:

- `q2/results/<session>/`

### 1) `q2/step0_trial_labels.py`

Inputs:

- Raw MICRONS data via `reader.py` + `microns.h5`

Outputs:

- `q2/results/<session>/trials_<session>.csv`

### 2) `q2/step1_features.py`

Inputs:

- `q2/results/<session>/trials_<session>.csv`
- Raw responses from `microns.h5`

Outputs:

- `q2/results/<session>/q2_features_<session>.npz`

Saved arrays:

- `X_V1`, `X_LM`, `X_AL`, `X_RL` (trial x neurons x time)
- `y` (3-class labels)
- `groups` (clip hash for grouped CV)

### 3) `q2/step2_decode.py`

Inputs:

- `q2/results/<session>/q2_features_<session>.npz`

Outputs:

- `q2/results/<session>/q2_decode_<session>.npz`

Saved arrays:

- `acc_V1`, `acc_LM`, `acc_AL`, `acc_RL` (accuracy over time)
- `cm_V1`, `cm_LM`, `cm_AL`, `cm_RL` (confusion matrices)
- `labels`, `n_time`, `n_neurons_subsample`

### 4) `q2/step3_plot.py`

Inputs:

- `q2/results/<session>/q2_decode_<session>.npz`

Outputs:

- Interactive matplotlib windows (`plt.show()`); no files written by default.

---

## End-to-end data flow (Q3, `q3/`)

Q3 deep-dives into natural vs parametric differentiation, with sub-analyses of stimulus origin (filmed vs rendered vs parametric). All scripts import from `q3/config.py` (central configuration — no hardcoded paths in individual scripts).

Q3 writes to:

- `q3/results/<session>/`
- `q3/results/<session>/csv/`

### 1) `q3/step0_labels.py`

Inputs:

- Raw MICRONS data via `reader.py` + `microns.h5`

Outputs:

- `q3/results/<session>/trials_q3_<session>.csv`

Adds columns beyond Q1: `origin` (filmed/rendered/parametric), `naturalness_group` (human_filmed/cg_natural/parametric).

### 2) `q3/step1_features.py`

Inputs:

- `q3/results/<session>/trials_q3_<session>.csv`
- Raw responses from `microns.h5`

Outputs:

- `q3/results/<session>/features_q3_<session>.npz`

Saved arrays: `X_V1`, `X_LM`, `X_AL`, `X_RL` (trial x neurons), `y_label`, `y_natural`, `y_origin`, `y_naturalness_group`.

### 3) `q3/step2_natural_vs_parametric.py`

Inputs:

- `q3/results/<session>/features_q3_<session>.npz`

Outputs:

- `q3/results/<session>/q3_binary_decode_<session>.npz`
- `q3/results/<session>/csv/q3_learning_curves_<session>.csv`
- `q3/results/<session>/csv/q3_paired_comparisons_<session>.csv`
- `q3/results/<session>/csv/q3_permutation_nulls_<session>.csv`
- `q3/results/<session>/csv/q3_neuron_discriminability_<session>.csv`

### 4) `q3/step3_natural_subgroups.py`

Inputs:

- `q3/results/<session>/features_q3_<session>.npz`

Outputs:

- `q3/results/<session>/q3_subgroup_decode_<session>.npz`
- `q3/results/<session>/csv/q3_threeclass_accuracy_<session>.csv`
- `q3/results/<session>/csv/q3_pairwise_accuracy_<session>.csv`
- `q3/results/<session>/csv/q3_fiveclass_accuracy_<session>.csv`
- `q3/results/<session>/csv/q3_confusion_3class_<session>.csv`
- `q3/results/<session>/csv/q3_confusion_5class_<session>.csv`

### 5) `q3/step4_plots.py`

Inputs:

- All .npz and CSV files from steps 2–3

Outputs (all as PNG + PDF):

- `q3/results/<session>/q3_binary_accuracy_<session>`
- `q3/results/<session>/q3_learning_curves_<session>`
- `q3/results/<session>/q3_confusion_3class_<session>`
- `q3/results/<session>/q3_pairwise_heatmap_<session>`
- `q3/results/<session>/q3_discriminability_<session>`
- `q3/results/<session>/q3_confusion_5class_<session>`
- `q3/results/<session>/q3_radar_summary_<session>`

### 6) `q3/step5_time_features.py` (Q3.2)

Inputs:

- `q3/results/<session>/trials_q3_<session>.csv`
- Raw responses from `microns.h5`

Outputs:

- `q3/results/<session>/q3_time_features_<session>.npz`

Saved arrays: `X_V1`, `X_LM`, `X_AL`, `X_RL` (trial x neurons x time), `stability_V1`, `stability_LM`, `stability_AL`, `stability_RL` (per-trial CV), `y_label`, `y_natural`, `y_origin`, `y_naturalness_group`, `trial_idx`, `hash`.

### 7) `q3/step6_time_decode.py` (Q3.2)

Inputs:

- `q3/results/<session>/q3_time_features_<session>.npz`

Outputs:

- `q3/results/<session>/q3_time_decode_<session>.npz`
- `q3/results/<session>/csv/q3_time_accuracy_<session>.csv`
- `q3/results/<session>/csv/q3_stability_by_stimulus_<session>.csv`
- `q3/results/<session>/csv/q3_stability_decoding_<session>.csv`

### 8) `q3/step7_time_plots.py` (Q3.2)

Inputs:

- All .npz and CSV files from steps 5–6

Outputs (all as PNG + PDF):

- `q3/results/<session>/q3_time_accuracy_binary_<session>`
- `q3/results/<session>/q3_time_accuracy_3class_<session>`
- `q3/results/<session>/q3_stability_distributions_<session>`
- `q3/results/<session>/q3_stable_vs_unstable_<session>`
- `q3/results/<session>/q3_stability_heatmap_<session>`
- `q3/results/<session>/q3_stability_scatter_<session>`

---

## File naming conventions

Session id is always embedded in filenames (example `7_4`) to keep outputs session-scoped.

Common suffixes:

- `_CLEAN` -> behavior-regressed variant (from `step1b` features).
- `results_step*.csv` -> table outputs intended for analysis/reporting.
- `features*.npz` -> machine-readable feature bundles for downstream scripts.

---

## What is tracked vs local-only

Tracked in git:

- Source scripts and existing checked-in results under `results/7_4/...`

Ignored in git (important):

- `.env`
- `*.h5`
- `*.npz`
- `q2/results/`
- `q3/results/`
- virtual env folders and cache files

Practical implication:

- Raw dataset and most generated binaries are local artifacts.
- CSV/figure outputs may be tracked depending on location and workflow.

---

## Minimal run order reference

Q1 single session:

1. `python src/step0_explore_session.py <session>`
2. `python src/step1_features.py <session>`
3. `python src/step1b_behavioral_clean.py <session>`
4. `python src/step4_learning_curves.py <session>`
5. `python src/step4_learning_curves_CLEAN.py <session>`
6. `python src/step5_confusion.py <session>`
7. `python src/step5_confusion_CLEAN.py <session>`

Q2 single session:

1. `python q2/step0_trial_labels.py <session>`
2. `python q2/step1_features.py <session>`
3. `python q2/step2_decode.py <session>`
4. `python q2/step3_plot.py`

Q3 single session:

1. `cd q3 && uv run python step0_labels.py <session>`
2. `uv run python step1_features.py <session>`
3. `uv run python step2_natural_vs_parametric.py <session>`
4. `uv run python step3_natural_subgroups.py <session>`
5. `uv run python step4_plots.py <session>`

Q3.2 (time-resolved + stability, requires step0 from Q3):

1. `cd q3 && uv run python step5_time_features.py <session>`
2. `uv run python step6_time_decode.py <session>`
3. `uv run python step7_time_plots.py <session>`

