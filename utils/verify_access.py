import sys
import importlib.util
import traceback

# ---------------------------------------------------------------
# Paths — adjust if yours differ
# ---------------------------------------------------------------
READER_PATH = r"C:\Users\Anna Notaro\.cache\huggingface\hub\datasets--NeuroBLab--MICrONS\snapshots\62869ddcb42d06b4436383d2e56201429d919c34\reader.py"
DATA_PATH = r"C:\data\microns\microns.h5"

# ---------------------------------------------------------------
# 1. Load the reader module
# ---------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load MicronsReader")
print("=" * 60)
try:
    spec = importlib.util.spec_from_file_location("microns_reader", READER_PATH)
    reader_module = importlib.util.module_from_spec(spec)
    sys.modules["microns_reader"] = reader_module
    spec.loader.exec_module(reader_module)
    MicronsReader = reader_module.MicronsReader
    print("OK: MicronsReader loaded")
except Exception as e:
    print(f"FAIL: {e}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------
# 2. Open the H5 file and list what's available
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 2: Open dataset and inspect top-level structure")
print("=" * 60)
with MicronsReader(DATA_PATH) as reader:
    # What methods does the reader expose?
    methods = [m for m in dir(reader) if not m.startswith('_')]
    print("Available methods/attributes on reader:")
    for m in methods:
        print(f"  - {m}")
    print()

    # Full H5 tree (first few items per group)
    print("H5 file structure (max 2 items per group):")
    reader.print_structure(max_items=2)

# ---------------------------------------------------------------
# 3. Try to get sessions / trials / responses
# ---------------------------------------------------------------
print()
print("=" * 60)
print("STEP 3: Access a session and pull one trial's data")
print("=" * 60)
with MicronsReader(DATA_PATH) as reader:
    # Try common method names — one of these should work.
    # Adjust based on what you saw in STEP 2.
    try:
        sessions = reader.get_sessions()
        print(f"Sessions available ({len(sessions)}): {sessions[:10]}")
        chosen = sessions[0]
    except AttributeError:
        print("No get_sessions() — trying alternatives")
        # fallback: look directly in the H5
        import h5py
        with h5py.File(DATA_PATH, "r") as f:
            print("Top-level keys:", list(f.keys()))
        chosen = None

    if chosen is not None:
        print(f"\nExploring session: {chosen}")

        # Try to get hashes (trial stimulus IDs)
        try:
            hashes = reader.get_hashes_by_session(chosen)
            print(f"  {len(hashes)} trials in this session")
            print(f"  First 3 hashes: {hashes[:3]}")
        except Exception as e:
            print(f"  get_hashes_by_session failed: {e}")

        # Try to get one trial's responses
        try:
            trial = reader.get_trial(chosen, 0)
            print(f"  Trial 0 keys: {list(trial.keys()) if hasattr(trial, 'keys') else type(trial)}")
            if 'responses' in trial:
                r = trial['responses']
                print(f"  responses shape: {r.shape}  (n_neurons, n_frames)")
        except Exception as e:
            print(f"  get_trial failed: {e}")

        # Try to get stimulus type for the first hash
        try:
            stim = reader.get_video_type(hashes[0])
            print(f"  First trial stimulus type: {stim}")
        except Exception as e:
            print(f"  get_video_type failed: {e}")

print()
print("=" * 60)
print("DONE. If STEP 3 printed shapes and stimulus types, you're good.")
print("=" * 60)