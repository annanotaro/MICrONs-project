import sys
import importlib.util
import h5py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MICRONS_DATA_PATH, MICRONS_READER_PATH

READER_PATH = MICRONS_READER_PATH
DATA_PATH = MICRONS_DATA_PATH

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

with MicronsReader(DATA_PATH) as reader:
    hashes = reader.get_hashes_by_session("7_4")
    print(f"First 3 hashes: {hashes[:3]}")
    print()

    # Try get_video_data on one of the previously-UNKNOWN hashes
    # (we'll find one by checking which ones didn't match above)
    for i, h in enumerate(hashes[:5]):
        try:
            vd = reader.get_video_data(h)
            print(f"Trial {i} hash={h[:20]}...")
            print(f"  keys: {list(vd.keys()) if hasattr(vd, 'keys') else type(vd)}")
            if hasattr(vd, 'keys'):
                for k in vd.keys():
                    v = vd[k]
                    if hasattr(v, 'shape'):
                        print(f"  {k}: shape={v.shape}")
                    else:
                        print(f"  {k}: {v}")
            print()
        except Exception as e:
            print(f"Trial {i} hash={h[:20]}... FAILED: {e}")
            print()