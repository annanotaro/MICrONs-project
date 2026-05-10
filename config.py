import os
from pathlib import Path

# Path to the local MICrONs HDF5 data file.
# Override by setting the MICRONS_DATA environment variable, e.g.:
#   export MICRONS_DATA="/path/to/your/microns.h5"
DATA_PATH = Path(os.environ.get(
    "MICRONS_DATA",
    "/Users/bea/microns_decoding/data/1621/raw/microns.h5"
))
