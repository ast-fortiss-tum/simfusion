import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# -------------------------
# Map critical CSV rows to simout files (5-digit precision) with tqdm
# -------------------------
def match_critical_simouts(csv_path, simout_folder, precision=6, key_cols=None):
    """
    Returns the paths of critical simouts based on CSV filtering.

    Matching is done by building a canonical key string using fixed-decimal
    formatting, equivalent to:
        "_".join(("%.6f" % a) for a in values)
    (with precision configurable).

    Args:
        csv_path (str): path to all_testcases.csv
        simout_folder (str): folder containing simout JSON files
        precision (int): number of decimals in the fixed formatting (e.g., 6)
    """
    import os
    import pandas as pd
    from tqdm import tqdm

    critical_df = pd.read_csv(csv_path)

    if key_cols is None:
        key_cols = [
            "goal_center_x","goal_center_y","goal_velocity_min","goal_velocity_max",
            "num_npc","lane_count","npc_x0","npc_v_max","npc_seed","cutin_tx","cutin_delay"
        ]

    fmt = f"%.{precision}f"

    # Build lookup of CSV keys using the SAME formatting op as filenames
    csv_keys = set()
    for _, row in critical_df.iterrows():
        vals = row[key_cols].values
        key = "_".join(fmt % float(a) for a in vals)
        csv_keys.add(key)

    simout_files = [
        os.path.join(simout_folder, f)
        for f in os.listdir(simout_folder)
        if f.endswith(".json")
    ]

    critical_paths = []

    print(f"[INFO] Processing {len(simout_files)} simout files...")
    for f in tqdm(simout_files, desc="Matching simouts"):
        fname = os.path.basename(f)
        fname_parts = fname.split("_")

        # Expected: something like <prefix>_S<val0>_<val1>_..._<valN>.json
        if len(fname_parts) < 2:
            print(f"[WARNING] Skipping file with unexpected format: {fname}")
            continue

        first_val = fname_parts[1][1:]   # strip leading 'S'
        other_vals = fname_parts[2:]

        parts = [first_val] + other_vals
        parts[-1] = parts[-1].removesuffix(".json")
        parts = parts[:len(key_cols)]

        try:
            file_key = "_".join(fmt % float(p) for p in parts)
        except ValueError:
            print(f"[WARNING] Skipping file with unexpected format: {fname}")
            continue

        if file_key in csv_keys:
            critical_paths.append(f)

    print(f"[INFO] Found {len(critical_paths)}/{len(critical_df)} critical simout files")
    return critical_paths
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    csv_path = "/home/user/testing/topic1-opensbt-aw/results_wandb/LoFi_GA_pop10_t06_00_00_seed15/xnrjawkh/all_testcases.csv"
    simout_folder = "/home/user/testing/topic1-opensbt-aw/results_wandb/LoFi_GA_pop10_t06_00_00_seed15/xnrjawkh/simout/"
    
    critical_simouts = match_critical_simouts(csv_path, simout_folder)
    
    print("\nCritical simout paths:")
    for path in critical_simouts:
        print(path)