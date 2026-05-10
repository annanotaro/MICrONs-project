---
license: cc-by-4.0
task_categories:
- feature-extraction
- time-series-forecasting
viewer: false
language:
- en
tags:
- neuroscience
- calcium-imaging
- biology
- multimodal
pretty_name: MICrONS Functional Activity Dataset (14 Sessions)
size_categories:
- 1K<n<10K
---

# MICrONS Functional Activity Dataset & Reader

This repository contains a curated portion of the MICrONS (Multi-Scale Networked Analysis of Cellular Responding Order) dataset. It consists of functional calcium imaging data from the visual cortex of mice in response to various visual stimuli: natural clips (`Clip`) and parametric videos (`Monet2`, `Trippy`).

Videos have been downsampled to match the neural activity scan frequency, with each frame corresponding to the stimulus frame appearing at least 66 ms before scan time. Neural responses are linearly interpolated at each stimulus frame timestamp to align with the video.

The data is organized into a highly efficient, indexed HDF5 format, allowing for rapid cross-session analysis based on either stimulus identity or brain anatomy.

---

## 📊 Dataset Overview

| Property | Detail |
|---|---|
| Sessions | 14 sessions of registered neural activity |
| Stimuli | 3 categories (`Clip`, `Monet2`, `Trippy`) identified by unique condition hashes |
| Neural Data | Calcium traces from thousands of neurons across V1, LM, RL, AL |
| Treadmill | Running speed (cm/s) synchronized to stimulus frames |
| Pupil Size | Major and minor radius of the fitted pupil ellipse (in pixels) |
| Eye Tracking | Pupil center coordinates (x, y) for gaze analysis |

### Temporal alignment & sampling

All data streams — neural responses, pupil, and treadmill — were independently interpolated to 30 Hz to match the stimulus frame rate, aligning every signal to a common stimulus clock. The full 30 Hz timeseries was then uniformly downsampled by a factor of 4, yielding a final sampling rate of **7.5 Hz** for all stored signals. As a result, `responses`, `treadmill`, `pupil`, and `stim_times` share an identical time axis within every trial, with each sample corresponding to a stimulus frame spaced ~133 ms apart.

---

## 🗂️ Internal HDF5 Structure

The file is organized to minimize redundancy: each unique video stimulus is stored once under `/videos/` and linked to every trial across all sessions via HDF5 SoftLinks. The `/sessions/` group is the source of truth for all neural and behavioral recordings.
```text
root/
├── 📂 BRAIN_AREAS/                              # Anatomical index (SoftLinks only)
│   └── 📂 <area_name>/                          # e.g., V1, AL, LM, RL
│       └── 🔗 <session_id>                      -> /sessions/<session_id>
│
├── 📂 SESSIONS/                                 # Primary neural data storage
│   └── 📂 <session_id>/                         # e.g., 4_7, 5_6
│       ├── 📂 META/                             # Session-wide metadata
│       │   ├── 📂 AREA_INDICES/                 # Pre-computed neuron index masks per area
│       │   │   └── 📄 <area_name>               [Dataset: (N_area_neurons,), int]
│       │   ├── 📄 brain_areas                   [Dataset: (N_neurons,), bytes — area label per neuron]
│       │   ├── 📄 coordinates                   [Dataset: (N_neurons, 3), float — motor coordinates x/y/z]
│       │   ├── 📄 unit_ids                      [Dataset: (N_neurons,), int — unique neuron IDs]
│       │   ├── 📄 condition_hashes              [Dataset: (N_trials,), bytes — hash per trial, in order]
│       │   └── (Attr) fps                       [Float — scan acquisition rate]
│       └── 📂 TRIALS/                           # One group per trial, indexed chronologically
│           └── 📂 <trial_idx>/
│               ├── 📄 responses                 [Dataset: (N_neurons, F), float — ΔF/F calcium traces]
│               ├── 📄 treadmill                 [Dataset: (F,), float — running speed in cm/s]
│               ├── 📄 pupil                     [Dataset: (4, F), float — rows: x, y, major_r, minor_r]
│               ├── 📄 stim_times               [Dataset: (F,), float — stimulus frame timestamps in seconds]
│               └── (Attr) condition_hash        [String — identifies the video shown in this trial]
│
├── 📂 TYPES/                                    # Stimulus category index (SoftLinks only)
│   └── 📂 <stim_type>/                          # e.g., Clip, Monet2, Trippy
│       └── 🔗 <encoded_hash>                    -> /videos/<encoded_hash>
│
└── 📂 VIDEOS/                                   # Stimulus library — each video stored once
    └── 📂 <encoded_hash>/                       # URL-encoded condition hash (/ → %2F)
        ├── 📄 clip                              [Dataset: (F, H, W), uint8 — grayscale frames]
        ├── 📄 times                             [Dataset: (F,), float — relative frame timestamps in seconds, starting at 0]
        ├── 📂 INSTANCES/                        # Reverse index: all trials that showed this video
        │   └── 🔗 <session_id>_tr<trial_idx>    -> /sessions/<session_id>/trials/<trial_idx>
        ├── (Attr) original_hash                 [String — raw unencoded hash]
        ├── (Attr) type                          [String — one of: Clip, Monet2, Trippy]
        │
        │   # Clip-specific attributes/datasets:
        ├── (Attr) movie_name                    [String]
        ├── (Attr) short_movie_name              [String]
        ├── (Attr) duration                      [Float — clip duration in seconds]
        ├── (Attr) fps                           [Float — original stimulus frame rate]
        │
        │   # Monet2-specific attributes/datasets:
        ├── 📄 directions                        [Dataset: (N_orientations,), float — grating directions in degrees]
        ├── 📄 onsets                            [Dataset: (N_orientations,), float — onset times per grating]
        ├── (Attr) duration                      [Float]
        ├── (Attr) ori_coherence                 [Float — orientation coherence parameter]
        ├── (Attr) fps                           [Float]
        │
        │   # Trippy-specific attributes/datasets:
        ├── (Attr) temp_freq                     [Float — temporal frequency in Hz]
        ├── (Attr) spatial_freq                  [Float — spatial frequency in cycles/degree]
        ├── (Attr) duration                      [Float]
        └── (Attr) fps                           [Float]
```

### Key design notes

- **`/brain_areas/` and `/types/`** contain only SoftLinks — no data. They serve as fast lookup indices.
- **`/videos/<hash>/instances/`** allows reverse lookup: given a video, find every session and trial that presented it.
- **`condition_hashes`** in session metadata are stored in trial order and may contain duplicates (the same video can be shown multiple times per session).
- **`pupil`** rows are ordered `[x, y, major_r, minor_r]` consistently across all sessions.
- **`stim_times`** are absolute timestamps in seconds; `times` inside `/videos/` are relative (starting at 0), computed as `stim_times - stim_times.min()`.

---

## ⚙️ Setup & Installation

### Option 1 — Clone the repository

Since the dataset is stored as a large HDF5 file, Git LFS is required.
```bash
git lfs install
git clone https://huggingface.co/datasets/NeuroBLab/MICrONS
cd MICrONS
pip install -r requirements.txt
```

### Option 2 — Programmatic download (no clone)
```python
import sys
import importlib.util
from huggingface_hub import hf_hub_download

REPO_ID = "NeuroBLab/MICrONS"

print("Downloading files from Hugging Face...")
reader_path = hf_hub_download(repo_id=REPO_ID, filename="reader.py")
data_path   = hf_hub_download(repo_id=REPO_ID, filename="microns.h5")

spec = importlib.util.spec_from_file_location("reader", reader_path)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["reader"] = reader_module
spec.loader.exec_module(reader_module)

from reader import MicronsReader

with MicronsReader(data_path) as reader:
    reader.print_structure(max_items=2)
```

---

## 🛠️ Reader API

### Initialize
```python
from reader import MicronsReader

with MicronsReader("microns.h5") as reader:
    # all calls go here
    pass
```

### Explore structure
```python
with MicronsReader("microns.h5") as reader:
    # Tree view — SoftLinks are shown without dereferencing by default
    reader.print_structure(max_items=3, follow_links=False)
```

### Query available metadata
```python
with MicronsReader("microns.h5") as reader:
    # All stimulus types in the file
    types = list(reader.f['types'].keys())          # ['Clip', 'Monet2', 'Trippy']

    # All hashes for a given stimulus type
    monet_hashes = reader.get_hashes_by_type('Monet2')

    # All (or unique) hashes shown in a session
    all_hashes    = reader.get_hashes_by_session('4_7')
    unique_hashes = reader.get_hashes_by_session('4_7', return_unique=True)

    # Brain areas recorded in a session, or all areas in the file
    areas_in_session = reader.get_available_brain_areas('4_7')  # ['AL', 'LM', 'RL', 'V1']
    all_areas        = reader.get_available_brain_areas()
```

### Load video + all associated trials

`get_full_data_by_hash` is the primary access method. It aggregates the video and every neural/behavioral trial across all sessions that presented that stimulus.
```python
target_hash = "0JcYLY6eaQxNgD0AqyHf"

with MicronsReader("microns.h5") as reader:
    # Optionally filter responses to a single brain area
    data = reader.get_full_data_by_hash(target_hash, brain_area='V1')

    print(data['clip'].shape)       # (F, H, W)  — grayscale video frames
    print(data['stim_type'])        # e.g. 'Clip'

    for trial in data['trials']:
        print(trial['session'])             # e.g. '4_7'
        print(trial['trial_idx'])           # e.g. '12'
        print(trial['responses'].shape)     # (N_V1_neurons, F)
        print(trial['treadmill'].shape)     # (F,)   — running speed in cm/s
        print(trial['pupil'].shape)         # (4, F) — rows: x, y, major_r, minor_r
        print(trial['stim_times'].shape)    # (F,)   — absolute timestamps in seconds
```

### Load only neural responses
```python
with MicronsReader("microns.h5") as reader:
    trials = reader.get_responses_by_hash(target_hash, brain_area='LM')
    # Returns: [{'session': str, 'trial_idx': str, 'responses': np.array}, ...]
```

### Direct single-trial access
```python
with MicronsReader("microns.h5") as reader:
    trial = reader.get_trial('4_7', trial_idx=12, brain_area='V1')
    print(trial['responses'].shape)   # (N_V1_neurons, F)
    print(trial['stim_times'])        # absolute timestamps for this trial
```

### Load only the video
```python
with MicronsReader("microns.h5") as reader:
    clip, stim_type = reader.get_video_data("0JcYLY6eaQxNgD0AqyHf")
    print(clip.shape)    # (F, H, W)
    print(stim_type)     # 'Clip'
```

---

## 📝 Citation

If you use this dataset or reader in your research, please cite the original MICrONS Phase 3 release and this repository.