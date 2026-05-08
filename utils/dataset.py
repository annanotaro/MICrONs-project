import sys
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MICRONS_DATA_PATH, MICRONS_READER_PATH

reader_path = MICRONS_READER_PATH
data_path = MICRONS_DATA_PATH

spec = importlib.util.spec_from_file_location("microns_reader", reader_path)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)

MicronsReader = reader_module.MicronsReader

with MicronsReader(data_path) as reader:
    reader.print_structure(max_items=2)