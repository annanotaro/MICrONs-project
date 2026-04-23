import sys
import importlib.util

READER_PATH = r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py"
DATA_PATH = r"C:\data\microns\microns.h5"

spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
reader_module = importlib.util.module_from_spec(spec)
sys.modules["microns_reader"] = reader_module
spec.loader.exec_module(reader_module)
MicronsReader = reader_module.MicronsReader

# Also inspect the source of get_video_data to understand its return
import inspect
print("=" * 60)
print("get_video_data source:")
print("=" * 60)
print(inspect.getsource(MicronsReader.get_video_data))
print()

print("=" * 60)
print("Inspecting actual return values:")
print("=" * 60)
with MicronsReader(DATA_PATH) as reader:
    hashes = reader.get_hashes_by_session("7_4")
    for i, h in enumerate(hashes[:3]):
        result = reader.get_video_data(h)
        print(f"Trial {i}: hash={h[:25]}...")
        print(f"  type: {type(result)}")
        print(f"  length: {len(result)}")
        for j, item in enumerate(result):
            if hasattr(item, 'shape'):
                print(f"  [{j}]: array shape={item.shape} dtype={item.dtype}")
            elif isinstance(item, dict):
                print(f"  [{j}]: dict keys={list(item.keys())}")
            elif isinstance(item, str):
                preview = item[:80] if len(item) > 80 else item
                print(f"  [{j}]: str = {preview!r}")
            else:
                print(f"  [{j}]: {type(item).__name__} = {item}")
        print()

    # Also try get_full_data_by_hash — might give us the short_movie_name directly
    print("=" * 60)
    print("get_full_data_by_hash:")
    print("=" * 60)
    full = reader.get_full_data_by_hash(hashes[0])
    print(f"type: {type(full)}")
    if hasattr(full, 'keys'):
        print(f"keys: {list(full.keys())}")
        for k in full.keys():
            v = full[k]
            if hasattr(v, 'shape'):
                print(f"  {k}: array shape={v.shape}")
            else:
                print(f"  {k}: {type(v).__name__} = {v!r}"[:120])