# Dataset and reader

The analyses in this repository run on a curated subset of the MICrONS Phase 3 functional dataset:
two-photon calcium imaging from mouse visual cortex during presentation of natural video clips
(`Clip`) and parametric stimuli (`Monet2`, `Trippy`).

The data is redistributed as a single indexed HDF5 file, hosted at
[`NeuroBLab/MICrONS`](https://huggingface.co/datasets/NeuroBLab/MICrONS), together with the
`MicronsReader` class used throughout this project. This document describes the file layout and the
reader API.

## Contents

| Property | Detail |
|---|---|
| Sessions | 14 sessions of registered neural activity |
| Stimuli | 3 categories (`Clip`, `Monet2`, `Trippy`), identified by condition hash |
| Neural data | Calcium traces (ΔF/F) from thousands of neurons across V1, LM, AL, RL |
| Behaviour | Treadmill running speed (cm/s); pupil centre (x, y) and ellipse radii (major, minor) |

Note that the analyses in this repository use 10 of the 14 sessions — see the report for the
exclusion criteria.

## Temporal alignment

Neural, pupil and treadmill streams are independently interpolated to 30 Hz to match the stimulus
frame rate, placing every signal on a common stimulus clock. Each frame corresponds to the stimulus
frame presented at least 66 ms before scan time. The 30 Hz series is then uniformly downsampled by a
factor of 4, giving a stored rate of **7.5 Hz**.

Consequently `responses`, `treadmill`, `pupil` and `stim_times` share an identical time axis within
every trial, with samples spaced roughly 133 ms apart.

## File layout

Each unique stimulus video is stored exactly once under `/videos/`. The `/sessions/` group is the
sole source of truth for neural and behavioural data; `/brain_areas/` and `/types/` contain only
HDF5 SoftLinks and exist as lookup indices.

```
root/
├── brain_areas/                     # Anatomical index (SoftLinks only)
│   └── <area_name>/                 # V1, LM, AL, RL
│       └── <session_id>             -> /sessions/<session_id>
│
├── sessions/                        # Primary neural and behavioural data
│   └── <session_id>/                # e.g. 4_7, 5_6
│       ├── meta/
│       │   ├── area_indices/
│       │   │   └── <area_name>      (N_area_neurons,)  int    neuron index mask
│       │   ├── brain_areas          (N_neurons,)       bytes  area label per neuron
│       │   ├── coordinates          (N_neurons, 3)     float  motor coordinates x/y/z
│       │   ├── unit_ids             (N_neurons,)       int    unique neuron IDs
│       │   ├── condition_hashes     (N_trials,)        bytes  hash per trial, in trial order
│       │   └── @fps                                    float  scan acquisition rate
│       └── trials/
│           └── <trial_idx>/
│               ├── responses        (N_neurons, F)     float  ΔF/F calcium traces
│               ├── treadmill        (F,)               float  running speed, cm/s
│               ├── pupil            (4, F)             float  rows: x, y, major_r, minor_r
│               ├── stim_times       (F,)               float  absolute timestamps, seconds
│               └── @condition_hash                     str    video shown in this trial
│
├── types/                           # Stimulus-category index (SoftLinks only)
│   └── <stim_type>/                 # Clip, Monet2, Trippy
│       └── <encoded_hash>           -> /videos/<encoded_hash>
│
└── videos/                          # Stimulus library, one entry per video
    └── <encoded_hash>/              # condition hash, URL-encoded ("/" -> "%2F")
        ├── clip                     (F, H, W)          uint8  grayscale frames
        ├── times                    (F,)               float  frame times, relative, from 0
        ├── instances/               # Reverse index: every trial that showed this video
        │   └── <session_id>_tr<trial_idx>  -> /sessions/<session_id>/trials/<trial_idx>
        ├── @original_hash                              str    raw unencoded hash
        ├── @type                                       str    Clip | Monet2 | Trippy
        ├── @duration, @fps                             float
        │
        │   # Clip only
        ├── @movie_name, @short_movie_name              str
        │
        │   # Monet2 only
        ├── directions               (N_orientations,)  float  grating directions, degrees
        ├── onsets                   (N_orientations,)  float  onset time per grating
        ├── @ori_coherence                              float
        │
        │   # Trippy only
        ├── @temp_freq                                  float  temporal frequency, Hz
        └── @spatial_freq                               float  spatial frequency, cycles/degree
```

`@` denotes an HDF5 attribute; everything else is a dataset, with shape and dtype given.

### Notes

- `condition_hashes` is stored in trial order and may contain duplicates: the same video can be
  shown several times within a session.
- `videos/<hash>/instances/` supports the reverse query — given a video, find every trial that
  presented it, across all sessions.
- `pupil` rows are ordered `[x, y, major_r, minor_r]` consistently in every session.
- `stim_times` are absolute (seconds); `times` under `videos/` are relative, computed as
  `stim_times - stim_times.min()`.

## Installation

The HDF5 file is large, so a clone requires Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/NeuroBLab/MICrONS
cd MICrONS
pip install -r requirements.txt
```

To download programmatically without cloning:

```python
from huggingface_hub import hf_hub_download

reader_path = hf_hub_download(repo_id="NeuroBLab/MICrONS", filename="reader.py")
data_path   = hf_hub_download(repo_id="NeuroBLab/MICrONS", filename="microns.h5")
```

## Reader API

All access goes through `MicronsReader`, used as a context manager:

```python
from reader import MicronsReader

with MicronsReader("microns.h5") as reader:
    ...
```

**Inspect the file.** SoftLinks are shown without dereferencing unless requested:

```python
reader.print_structure(max_items=3, follow_links=False)
```

**Query metadata.**

```python
types            = list(reader.f["types"].keys())            # ['Clip', 'Monet2', 'Trippy']
monet_hashes     = reader.get_hashes_by_type("Monet2")
all_hashes       = reader.get_hashes_by_session("4_7")
unique_hashes    = reader.get_hashes_by_session("4_7", return_unique=True)
areas_in_session = reader.get_available_brain_areas("4_7")   # ['AL', 'LM', 'RL', 'V1']
all_areas        = reader.get_available_brain_areas()
```

**Load a video with every trial that presented it.** This is the primary access method: it
aggregates the stimulus and all associated neural and behavioural trials across sessions, optionally
restricted to one anatomical area.

```python
data = reader.get_full_data_by_hash("0JcYLY6eaQxNgD0AqyHf", brain_area="V1")

data["clip"].shape          # (F, H, W)   grayscale frames
data["stim_type"]           # 'Clip'

for trial in data["trials"]:
    trial["session"]        # '4_7'
    trial["trial_idx"]      # '12'
    trial["responses"]      # (N_V1_neurons, F)
    trial["treadmill"]      # (F,)
    trial["pupil"]          # (4, F)
    trial["stim_times"]     # (F,)
```

**Narrower access.**

```python
# Responses only, for one stimulus
trials = reader.get_responses_by_hash(target_hash, brain_area="LM")
# -> [{'session': str, 'trial_idx': str, 'responses': np.ndarray}, ...]

# A single trial
trial = reader.get_trial("4_7", trial_idx=12, brain_area="V1")

# A stimulus video only
clip, stim_type = reader.get_video_data("0JcYLY6eaQxNgD0AqyHf")
```

## Citation

If you use this dataset or the reader, please cite the original MICrONS Phase 3 release
(Ding et al., *Nature* 640:459–469, 2025) alongside this repository.

Licence: CC BY 4.0.