# Neural Decoding of Visual Stimuli in Mouse Visual Cortex

Linear decoding of visual stimulus category from two-photon population activity in four mouse
visual areas (V1, LM, AL, RL), using the MICrONS functional dataset. We ask whether decodability
varies systematically along the putative cortical hierarchy, and whether any such differences
survive controls for population size and behavioural state.

> **Main result.** On the finest contrast — discriminating three *natural* video categories
> (Cinematic / Sports1M / Rendered) — **LM is the best decoder in 9/10 sessions**, beating V1 by
> Δ = +0.031 (significant in 10/10 sessions) and AL/RL by Δ ≈ +0.05 (9–10/10), Bonferroni-corrected.
> This **inverts** the naive expectation of a strict V1 → higher-area gradient of categorical
> abstraction. Coarse contrasts (natural vs. parametric; Monet2 vs. Trippy) are at ceiling in every
> area and cannot distinguish the hierarchy at all.

![Pairwise area differences in balanced accuracy](figures/fig2_pairwise_area_differences.png)

*Pairwise area differences in balanced accuracy (clean features, mean across 10 sessions).
Right panel (Q1c, the fine natural contrast) carries the effect: LM beats every other area.
Parenthesised counts = sessions significant after Bonferroni correction.*

📄 **[Full report (PDF)](documents/report.pdf)** · 11 pages, 14 figures

---

## Research questions

| | Question | Chance | Folder |
|---|---|---|---|
| **Q1a** | Natural vs. parametric stimuli | 0.50 | [`q1/`](q1) |
| **Q1b** | Parametric discrimination (Monet2 vs. Trippy) | 0.50 | [`q1/`](q1) |
| **Q1c** | Natural discrimination (Cinematic vs. Sports1M vs. Rendered) | 0.33 | [`q1/`](q1) |
| **Q2** | Time-resolved, per-frame clip-category decoding | 0.33 | [`q2/`](q2) |

## Results

**Q1 — trial-mean decoding, balanced accuracy at matched neuron count (mean ± SD, 10 sessions):**

| | V1 | LM | AL | RL |
|---|---|---|---|---|
| Q1a *(chance 0.50)* | **0.959** ± 0.016 | 0.951 ± 0.015 | 0.918 ± 0.020 | 0.936 ± 0.022 |
| Q1b *(chance 0.50)* | **0.994** ± 0.006 | 0.988 ± 0.009 | 0.979 ± 0.019 | 0.980 ± 0.015 |
| Q1c *(chance 0.33)* | 0.647 ± 0.033 | **0.677** ± 0.036 | 0.621 ± 0.061 | 0.622 ± 0.058 |

All 120/120 (session × area × question) cells decode significantly above the shuffle-label null.
Q1a and Q1b saturate; **Q1c is the only contrast that separates the areas**, and there LM leads.

**Q2 — time-resolved decoding (Session 5_6, peak balanced accuracy, chance 33.3%):**

| Window | Clf. | V1 | LM | AL | RL | Avg. |
|---|---|---|---|---|---|---|
| *w* = 1 | LR | 41.4% | 44.2% | **47.4%** | 42.6% | 43.9% |
| *w* = 1 | SVM | 40.2% | 43.5% | 44.6% | 41.2% | 42.5% |
| *w* = 5 | LR | 46.5% | 48.4% | **50.2%** | 48.5% | 48.4% |
| *w* = 5 | SVM | 44.2% | 46.6% | 47.5% | 47.2% | 46.4% |

Per-frame decoding is significant everywhere (*p* < 10⁻⁹; 0/50 shuffles exceeded the true accuracy)
but stays below 50%. Five-frame temporal averaging adds a uniform **≈ +4.5 points** in every area —
consistent with category information being distributed over time rather than locked to onset — and
cross-area differences shrink under averaging, suggesting broadly distributed representation.

**Behavioural confounds.** Regressing out pupil (4 features) and treadmill velocity before
trial-averaging costs ≈ 0.03 accuracy on Q1a (largest in AL, −0.038), i.e. part of the coarse
natural-vs-parametric contrast reflects covariation of arousal/locomotion with stimulus class.
For Q1b and Q1c the effect is < 0.01 and inconsistent in sign — **the LM advantage is not a
behavioural artefact.** All headline results are reported on cleaned features as the conservative
estimate.

**Confusion structure.** Cinematic ↔ Rendered is the dominant error in every area (off-diagonals
0.19–0.23); Sports1M is the most reliably classified class (diagonal 0.64–0.71), plausibly because
of its distinctive fast coherent motion. LM's advantage is spread across all three classes, not
driven by one.

## Methods

- **Data.** 10 of 14 MICrONS sessions, selected at a matched imaging rate (~6.30 Hz; sessions 9_3,
  9_4, 9_6 excluded at 8.62–9.62 Hz; 7_4 excluded as corrupted). 464 trials/session
  (128 Cinematic, 128 Sports1M, 128 Rendered, 40 Monet2, 40 Trippy).
- **Preprocessing.** First 3 frames (≈475 ms) discarded for response-onset lag; per-neuron trial
  means computed per anatomical area. "Clean" features are residuals after regressing each neuron
  on 4 pupil channels + treadmill velocity.
- **Decoder.** `StandardScaler → LogisticRegression` (ℓ2, *C* = 1, balanced class weights),
  balanced accuracy under 5-fold stratified CV.
- **Population-size control.** Areas differ in recorded neuron count, which inflates accuracy
  independently of coding quality. All cross-area comparisons are made at the matched count
  *N*min (287–468 per session) over **50 random subsamples**.
- **Statistics.** Shuffle-label nulls (100 permutations per cell); paired Wilcoxon signed-rank
  across sessions, Bonferroni-corrected within question.
- **Q2.** Session 5_6 (8,592 neurons; 384 natural trials; 72 timepoints; 468 neurons/area). Per-frame
  response vectors decoded independently at each timepoint with LR and linear SVM.
  **GroupKFold by clip hash** prevents the same clip appearing in train and test. Temporal averaging
  tested at *w* = 5 frames.

## Limitations

Linear decoders cannot recover nonlinearly-formatted information — higher accuracy in LM means
category information is more *linearly accessible* there, not that LM "represents" categories more
than V1. Class counts are imbalanced (384 natural vs. 80 parametric), and *N*min varies across
sessions, so within-session contrasts are matched but cross-session pooling is not.

## Reproducing

```bash
git clone https://github.com/annanotaro/<repo> && cd <repo>
uv sync                      # or: pip install -r requirements.txt
cp .env.example .env         # point DATA_PATH at microns.h5
python main_runner.py --question 1
```

Data is pulled from [`NeuroBLab/MICrONS`](https://huggingface.co/datasets/NeuroBLab/MICrONS) on
first run. See **[docs/DATASET.md](docs/DATASET.md)** for the HDF5 schema and the `MicronsReader` API.

## Authors

Course research project supervised by **Prof. Alessandro Sanzeni**, Bocconi University
(March–April 2026).

Gaia Grossi · Max David · Leo Arthur Morvan · **Anna Notaro** · Beatrice Porta

*Anna Notaro: Q1 in full — decoding pipeline, neuron-count-matched subsampling,
behavioural regression, and the cross-area statistical comparisons (Figures 1–4).*

## References

Stringer et al., *Nature* 2019 · Goltstein et al., *Nat. Neurosci.* 2021 · Chen et al., *PLOS Comp. Biol.* 2024 · Ding et al., *Nature* 2025 (MICrONS functional connectomics)