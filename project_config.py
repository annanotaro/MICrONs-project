"""Global project configuration for shared external paths."""

import os

MICRONS_SNAPSHOT_DIR = os.environ.get(
    "MICRONS_SNAPSHOT_DIR",
    "/Users/arthurm/.cache/huggingface/hub/datasets--NeuroBLab--MICrONS/"
    "snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03",
)

MICRONS_READER_PATH = os.environ.get(
    "MICRONS_READER_PATH",
    f"{MICRONS_SNAPSHOT_DIR}/reader.py",
)

MICRONS_DATA_PATH = os.environ.get(
    "MICRONS_DATA_PATH",
    f"{MICRONS_SNAPSHOT_DIR}/microns.h5",
)
