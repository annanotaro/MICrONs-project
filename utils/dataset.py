import sys
import importlib.util

reader_path = r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py"
data_path = r"C:\data\microns\microns.h5"

spec = importlib.util.spec_from_file_location("microns_reader", reader_path)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)

MicronsReader = reader_module.MicronsReader

with MicronsReader(data_path) as reader:
    reader.print_structure(max_items=2)