import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
CHOSEN_SESSION = "7_4"

RESULTS_DIR = Path(__file__).parent / "results" / CHOSEN_SESSION
DATA_PATH = RESULTS_DIR / f"q2_decode_{CHOSEN_SESSION}.npz"

data = np.load(DATA_PATH)

areas = ["V1", "LM", "AL", "RL"]
chance = 1 / 3  # 3-class problem

# -------------------------
# PLOT 1: Accuracy vs time (main figure)
# -------------------------
plt.figure(figsize=(10, 6))

for area in areas:
    acc = data[f"acc_{area}"]
    plt.plot(acc, label=area)

# Chance level reference
plt.axhline(chance, linestyle="--", color="black", label="Chance")

plt.xlabel("Time (frames)")
plt.ylabel("Balanced accuracy")
plt.title("Q2: Decoding accuracy over time")
plt.legend()
plt.tight_layout()

plt.show()

# -------------------------
# PLOT 2: Peak accuracy per area
# -------------------------
peak_acc = []
for area in areas:
    acc = data[f"acc_{area}"]
    peak_acc.append(acc.max())

plt.figure(figsize=(6, 5))
plt.bar(areas, peak_acc)

plt.axhline(chance, linestyle="--", color="black", label="Chance")

plt.ylabel("Peak balanced accuracy")
plt.title("Peak decoding accuracy per area")
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# PLOT 3: Smoothed accuracy curves
# -------------------------
def smooth(x, k=5):
    return np.convolve(x, np.ones(k)/k, mode='same')

plt.figure(figsize=(10, 6))

for area in areas:
    acc = data[f"acc_{area}"]
    acc_smooth = smooth(acc, k=5)
    plt.plot(acc_smooth, label=area)

plt.axhline(chance, linestyle="--", color="black", label="Chance")

plt.xlabel("Time (frames)")
plt.ylabel("Balanced accuracy (smoothed)")
plt.title("Q2: Smoothed decoding accuracy over time")
plt.legend()

plt.tight_layout()
plt.show()
