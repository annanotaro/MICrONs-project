# MICrONS Q1 — Neural Decoding Pipeline

## Setup

1. Install Python 3.13 and create a virtual environment:
```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```

2. Download `microns.h5` and place it at `C:\data\microns\microns.h5`
   (or update the `DATA_PATH` constant at the top of each script in `src/`).

3. Ensure the HuggingFace reader is available at the path specified in
   `READER_PATH` inside each step0/step1 script, or edit the path to
   point to your local copy.

## Running the full pipeline for one session

```powershell
cd src
python step0_explore_session.py 7_4
python step1_features.py 7_4
python step1b_behavioral_clean.py 7_4
python step4_learning_curves.py 7_4
python step4_learning_curves_CLEAN.py 7_4
python step5_confusion.py 7_4
python step5_confusion_CLEAN.py 7_4
```

Runtime: approximately 4 hours per session on a standard laptop.

Outputs go to `results/7_4/` (figures, npz, CSVs).

## Running multiple sessions

```powershell
cd src
python run_all_sessions.py                # all viable sessions (~48 hr)
python run_all_sessions.py 7_4 5_6 7_5    # specific sessions
```

## Directory guide

- `src/` — analysis pipeline (run these)
- `utils/` — one-off diagnostic scripts (safe to ignore)
- `scratch/` — deprecated / superseded code (kept for reference)
- `results/<session>/` — all generated outputs, organized per session
- `report/` — LaTeX source and compiled PDF

## Pipeline overview

1. **step0_explore_session** — parse session metadata, build trial table with labels
2. **step1_features** — extract trial-mean neural responses per area → `.npz`
3. **step1b_behavioral_clean** — same as step1 but regresses out pupil and running
4. **step4_learning_curves** (+CLEAN) — learning curves, paired comparisons, permutation nulls
5. **step5_confusion** (+CLEAN) — Q1c confusion matrices per area

"CLEAN" versions use behaviorally-regressed features from step1b.