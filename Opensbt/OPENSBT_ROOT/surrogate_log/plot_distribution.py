"""
Plot the distribution of F1 (min distance) and F2 (velocity at min distance)
training data to diagnose high MAE in surrogate model training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob
import argparse
import os

# === MATCH YOUR EXISTING CONFIG ===
input_columns = ["PedSpeed", "EgoSpeed", "PedDist"]
optional_columns = ["Fitness_Min distance", "Fitness_Velocity at min distance"]
desired_columns = ["Fitness_Min distance_HiFi", "Fitness_Velocity at min distance_HiFi"]

output_folder = "./surrogate_log/plots/"


def load_data(data_folder):
    csv_files = glob(f"{data_folder}/*.csv")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_folder}")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if optional_columns[0] in df.columns:
        target_cols = optional_columns
    else:
        target_cols = desired_columns

    df = df[input_columns + target_cols].dropna()
    for col in target_cols:
        df = df[df[col] != 1000].reset_index(drop=True)

    return df, target_cols


def plot_fitness_distributions(df, target_cols, output_folder, data_folder_name=""):
    os.makedirs(output_folder, exist_ok=True)

    f1_col = target_cols[0]  # Min distance — critical interval [0, 1]
    f2_col = target_cols[1]  # Velocity at min distance — critical: > 0

    f1_values = df[f1_col].values
    f2_values = df[f2_col].values

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── F1 Histogram ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(f1_values, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax1.axvspan(0, 1, alpha=0.15, color="red", label="Critical zone [0, 1]")
    ax1.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Critical boundary (0)")
    ax1.axvline(1, color="darkred", linestyle="--", linewidth=1.5, label="Critical boundary (1)")
    ax1.set_xlabel(f1_col, fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("F1 — Min Distance\nHistogram", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.4)

    # ── F1 Boxplot ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    bp1 = ax2.boxplot(f1_values, vert=True, patch_artist=True,
                      boxprops=dict(facecolor="steelblue", alpha=0.7),
                      medianprops=dict(color="black", linewidth=2))
    ax2.axhspan(0, 1, alpha=0.15, color="red", label="Critical zone [0, 1]")
    ax2.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax2.axhline(1, color="darkred", linestyle="--", linewidth=1.5)
    ax2.set_ylabel(f1_col, fontsize=11)
    ax2.set_title("F1 — Min Distance\nBox Plot", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.set_xticks([])
    ax2.grid(axis="y", alpha=0.4)

    # ── F2 Histogram ────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(f2_values, bins=50, color="darkorange", edgecolor="white", alpha=0.85)
    ax3.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Critical boundary (> 0)")
    f2_pos = np.sum(f2_values > 0)
    f2_nonpos = np.sum(f2_values <= 0)
    ax3.set_xlabel(f2_col, fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title(
        f"F2 — Velocity at Min Distance\nHistogram  |  >0 (critical): {f2_pos}  |  ≤0 (non-crit): {f2_nonpos}",
        fontsize=11, fontweight="bold"
    )
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.4)

    # ── F2 Boxplot ───────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    bp2 = ax4.boxplot(f2_values, vert=True, patch_artist=True,
                      boxprops=dict(facecolor="darkorange", alpha=0.7),
                      medianprops=dict(color="black", linewidth=2))
    ax4.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Critical boundary (> 0)")
    ax4.set_ylabel(f2_col, fontsize=11)
    ax4.set_title("F2 — Velocity at Min Distance\nBox Plot", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.set_xticks([])
    ax4.grid(axis="y", alpha=0.4)

    # ── Summary stats as subtitle ────────────────────────────────────────────────
    f1_crit = np.sum((f1_values >= 0) & (f1_values <= 1))
    f1_total = len(f1_values)
    summary = (
        f"Dataset: {data_folder_name}  |  Total samples: {f1_total}\n"
        f"F1 — min={f1_values.min():.4f}, max={f1_values.max():.4f}, "
        f"mean={f1_values.mean():.4f}, std={f1_values.std():.4f}, "
        f"critical [0,1]: {f1_crit} ({100*f1_crit/f1_total:.1f}%)\n"
        f"F2 — min={f2_values.min():.4f}, max={f2_values.max():.4f}, "
        f"mean={f2_values.mean():.4f}, std={f2_values.std():.4f}, "
        f"critical (>0): {f2_pos} ({100*f2_pos/f1_total:.1f}%)"
    )
    fig.suptitle(summary, fontsize=9, y=1.01, ha="center",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8))

    out_path = os.path.join(output_folder, f"fitness_distribution_{data_folder_name}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"\n[INFO] Distribution plot saved to: {out_path}")

    # ── Print full stats to console ───────────────────────────────────────────────
    print("\n=== F1 (Min Distance) Statistics ===")
    print(f"  Range:    [{f1_values.min():.4f}, {f1_values.max():.4f}]")
    print(f"  Mean:     {f1_values.mean():.4f}")
    print(f"  Median:   {np.median(f1_values):.4f}")
    print(f"  Std:      {f1_values.std():.4f}")
    print(f"  % in [0,1] (critical): {100*f1_crit/f1_total:.1f}%")
    print(f"  Percentiles [5, 25, 50, 75, 95]: "
          f"{np.percentile(f1_values, [5,25,50,75,95]).round(4)}")

    print("\n=== F2 (Velocity at Min Distance) Statistics ===")
    print(f"  Range:    [{f2_values.min():.4f}, {f2_values.max():.4f}]")
    print(f"  Mean:     {f2_values.mean():.4f}")
    print(f"  Median:   {np.median(f2_values):.4f}")
    print(f"  Std:      {f2_values.std():.4f}")
    print(f"  % > 0 (critical): {100*f2_pos/f1_total:.1f}%")
    print(f"  Percentiles [5, 25, 50, 75, 95]: "
          f"{np.percentile(f2_values, [5,25,50,75,95]).round(4)}")


def main():
    parser = argparse.ArgumentParser(description="Plot F1/F2 training data distributions")
    parser.add_argument("--data_folder", type=str, default="./surrogate_log/data/batch0/",
                        help="Path to the folder containing training CSV files")
    args = parser.parse_args()

    data_folder_name = os.path.basename(os.path.normpath(args.data_folder))
    print(f"[INFO] Loading data from: {args.data_folder}")

    df, target_cols = load_data(args.data_folder)
    print(f"[INFO] Loaded {len(df)} samples. Target columns: {target_cols}")

    plot_fitness_distributions(df, target_cols, output_folder, data_folder_name)


if __name__ == "__main__":
    main()