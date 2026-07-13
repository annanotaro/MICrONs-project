"""Central configuration for the Q3 pipeline."""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

# -------------------------
# Paths — override via environment variables
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MICRONS_DATA_PATH, MICRONS_READER_PATH

READER_PATH = MICRONS_READER_PATH
DATA_PATH = MICRONS_DATA_PATH

# -------------------------
# Session selection (CLI arg > env var > default)
# -------------------------
if len(sys.argv) > 1:
    CHOSEN_SESSION: str = sys.argv[1]
else:
    CHOSEN_SESSION = os.environ.get("CHOSEN_SESSION", "7_4")

# -------------------------
# Output directory
# -------------------------
RESULTS_DIR: Path = Path(__file__).parent / "results" / CHOSEN_SESSION

# -------------------------
# Brain areas
# -------------------------
AREAS: list[str] = ["V1", "LM", "AL", "RL"]
AREA_COLORS: dict[str, str] = {
    "V1": "#1f77b4",
    "LM": "#ff7f0e",
    "AL": "#2ca02c",
    "RL": "#d62728",
}

# -------------------------
# Stimulus groupings
# -------------------------
NATURAL_LABELS: list[str] = ["Cinematic", "Sports1M", "Rendered"]
PARAMETRIC_LABELS: list[str] = ["Monet2", "Trippy"]
FILMED_LABELS: list[str] = ["Cinematic", "Sports1M"]
RENDERED_LABELS: list[str] = ["Rendered"]

# -------------------------
# Classifier & CV parameters
# -------------------------
N_FOLDS: int = 5
N_SUBSAMPLES: int = 10
N_PAIRED_SEEDS: int = 50
N_PERM_SHUFFLES: int = 200
N_NEURONS_MATCHED: int = 575
NEURON_COUNTS: list[int] = [25, 50, 100, 200, 400, 575]
FRAMES_TO_DROP: int = 3
RANDOM_STATE: int = 42

# Q3.2: Time-resolved decoding parameters
N_SEEDS_TIME: int = 10
N_NEURONS_TIME: int = 575


# -------------------------
# Helpers
# -------------------------
def get_reader_class() -> Any:
    """Import and return the MicronsReader class from the external reader module."""
    spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
    reader_module = importlib.util.module_from_spec(spec)
    sys.modules["microns_reader"] = reader_module
    spec.loader.exec_module(reader_module)
    return reader_module.MicronsReader


def load_trials() -> pd.DataFrame:
    """Load the Q3 trial CSV for the current session."""
    path = RESULTS_DIR / f"trials_q3_{CHOSEN_SESSION}.csv"
    return pd.read_csv(path)


def ensure_results_dir() -> None:
    """Create the results directory tree if it doesn't exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "csv").mkdir(exist_ok=True)
