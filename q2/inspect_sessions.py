import os
import sys
import h5py
import pandas as pd

DEFAULT_H5 = (
    "/Users/gaiagr/.cache/huggingface/hub/"
    "datasets--NeuroBLab--MICrONS/"
    "snapshots/79c7c55fec8484ebffd1cef67cfa433e63f32a03/"
    "microns.h5"
)

AREAS = ["V1", "LM", "AL", "RL"]
BAD_SESSIONS = {"7_4"}


def count_area_neurons(h5, session, area):
    path = f"sessions/{session}/meta/area_indices/{area}"
    if path not in h5:
        return 0
    return len(h5[path][()])


def main():
    h5_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_H5

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find microns.h5: {h5_path}")

    rows = []

    with h5py.File(h5_path, "r") as h5:
        sessions = sorted(h5["sessions"].keys())

        for session in sessions:
            counts = {area: count_area_neurons(h5, session, area) for area in AREAS}

            all_present = all(counts[a] > 0 for a in AREAS)
            total = sum(counts.values())
            min_area = min(counts.values())

            rows.append({
                "session": session,
                **counts,
                "total_V1_LM_AL_RL": total,
                "min_area_count": min_area,
                "all_four_present": all_present,
                "excluded": session in BAD_SESSIONS,
            })

    df = pd.DataFrame(rows)

    df_sorted = df.sort_values(
        by=["all_four_present", "excluded", "min_area_count", "total_V1_LM_AL_RL"],
        ascending=[False, True, False, False],
    )

    print("\nAll sessions ranked:")
    print(df_sorted.to_string(index=False))

    valid = df_sorted[
        (df_sorted["all_four_present"]) &
        (~df_sorted["excluded"])
    ].copy()

    if valid.empty:
        print("\nNo valid replacement session found with all four areas present.")
        return

    best = valid.iloc[0]

    print("\nBest replacement session:")
    print(best.to_string())

    out_csv = "q2/session_area_counts.csv"
    os.makedirs("q2", exist_ok=True)
    df_sorted.to_csv(out_csv, index=False)
    print(f"\nSaved table to: {out_csv}")


if __name__ == "__main__":
    main()