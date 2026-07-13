import h5py

data_path = r"C:\data\microns\microns.h5"

with h5py.File(data_path, "r") as f:
    print(list(f.keys()))