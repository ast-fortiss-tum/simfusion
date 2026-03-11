"""
Plot the distribution of F1 (min distance) and F2 (velocity at min distance)
BEFORE and AFTER the log1p transformation applied in train_model(),
to diagnose high MAE in surrogate model training.
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


# === SAME TRANSFORMS AS train_model() ===
def log1p_transform(y, shift=0.0):
    return np.log1p(y + shift)

def log1p_inverse_transform(y_scaled, shift=0.0):
    return np.expm1(y_scaled) - shift


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


def _add_critical_vlines(ax, values_raw, axis="x"):
    """Adds critical-zone shading in the ORIGINAL (raw) space only."""
    pass  # handled inline per-axis below


def _stats_text(values):
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return (
        f"n={len(values)}  min={values.min():.3f}  max={values.max():.3f}\n"
        f"mean={values.mean():.3f}  std={values.std():.3f}\n"
        f"p5={p5:.3f}  p25={p25:.3f}  p50={p50:.3f}  p75={p75:.3f}  p95={p95:.3f}"
    )


def plot_fitness_distributions(df, target_cols, output_folder, data_folder_name=""):
    os.makedirs(output_folder, exist_ok=True)

    f1_col = target_cols[0]  # Min distance — critical interval [0, 1]
    f2_col = target_cols[1]  # Velocity at min distance — critical: > 0

    f1_raw = df[f1_col].values.astype(float)
    f2_raw = df[f2_col].values.astype(float)
    y_raw = np.column_stack([f1_raw, f2_raw])

    # ── Replicate EXACTLY what train_model() does ────────────────────────────
    # y_shift is computed on the FULL y_reg before train/val split,
    # but here we compute it on the full dataset to approximate training behaviour.
    #y_shift = float(abs(y_raw.min())) if y_raw.min() < 0 else 0.0
    y_shift=0
    print(f"[INFO] y_shift = {y_shift:.6f}  (matches train_model() logic)")

    f1_transformed = log1p_transform(f1_raw, shift=y_shift)
    f2_transformed = log1p_transform(f2_raw, shift=y_shift)

    # Critical boundaries mapped through the SAME transform
    f1_crit_lo_t = log1p_transform(np.array([0.0]), shift=y_shift)[0]
    f1_crit_hi_t = log1p_transform(np.array([1.0]), shift=y_shift)[0]
    f2_crit_lo_t = log1p_transform(np.array([0.0]), shift=y_shift)[0]  # > 0 boundary

    # ── Layout: 4 rows × 2 cols ──────────────────────────────────────────────
    #   Row 0: F1 raw histogram    | F1 transformed histogram
    #   Row 1: F1 raw boxplot      | F1 transformed boxplot
    #   Row 2: F2 raw histogram    | F2 transformed histogram
    #   Row 3: F2 raw boxplot      | F2 transformed boxplot
    fig = plt.figure(figsize=(16, 18))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    col_labels = ["Before log1p transform (raw)", "After log1p transform"]
    row_colors = {"F1": "steelblue", "F2": "darkorange"}

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 0 — F1 histograms
    # ─────────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(f1_raw, bins=60, color=row_colors["F1"], edgecolor="white", alpha=0.85)
    ax.axvspan(0, 1, alpha=0.18, color="red", label="Critical [0, 1]")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(1, color="darkred", linestyle="--", linewidth=1.4, label="Boundaries")
    ax.set_xlabel(f1_col, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"F1  ·  {col_labels[0]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.01, -0.28, _stats_text(f1_raw), transform=ax.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4ff", alpha=0.8))

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(f1_transformed, bins=60, color=row_colors["F1"], edgecolor="white", alpha=0.85)
    ax.axvspan(f1_crit_lo_t, f1_crit_hi_t, alpha=0.18, color="red", label=f"Critical [{f1_crit_lo_t:.3f}, {f1_crit_hi_t:.3f}]")
    ax.axvline(f1_crit_lo_t, color="red", linestyle="--", linewidth=1.4)
    ax.axvline(f1_crit_hi_t, color="darkred", linestyle="--", linewidth=1.4, label="Boundaries (transformed)")
    ax.set_xlabel("log1p(F1 + shift)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"F1  ·  {col_labels[1]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.01, -0.28, _stats_text(f1_transformed), transform=ax.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f4ff", alpha=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 1 — F1 boxplots
    # ─────────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.boxplot(f1_raw, vert=False, patch_artist=True,
               boxprops=dict(facecolor=row_colors["F1"], alpha=0.7),
               medianprops=dict(color="black", linewidth=2),
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.axvspan(0, 1, alpha=0.18, color="red")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.4, label="Critical boundary")
    ax.axvline(1, color="darkred", linestyle="--", linewidth=1.4)
    ax.set_xlabel(f1_col, fontsize=10)
    ax.set_title("F1  ·  Box plot (raw)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.35)

    ax = fig.add_subplot(gs[1, 1])
    ax.boxplot(f1_transformed, vert=False, patch_artist=True,
               boxprops=dict(facecolor=row_colors["F1"], alpha=0.7),
               medianprops=dict(color="black", linewidth=2),
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.axvspan(f1_crit_lo_t, f1_crit_hi_t, alpha=0.18, color="red")
    ax.axvline(f1_crit_lo_t, color="red", linestyle="--", linewidth=1.4, label="Boundary (transformed)")
    ax.axvline(f1_crit_hi_t, color="darkred", linestyle="--", linewidth=1.4)
    ax.set_xlabel("log1p(F1 + shift)", fontsize=10)
    ax.set_title("F1  ·  Box plot (transformed)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.35)

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 2 — F2 histograms
    # ─────────────────────────────────────────────────────────────────────────
    f2_pos_raw = int(np.sum(f2_raw > 0))
    f2_pos_t   = int(np.sum(f2_transformed > f2_crit_lo_t))

    ax = fig.add_subplot(gs[2, 0])
    ax.hist(f2_raw, bins=60, color=row_colors["F2"], edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.4,
               label=f"Critical boundary (>0)\n{f2_pos_raw}/{len(f2_raw)} critical ({100*f2_pos_raw/len(f2_raw):.1f}%)")
    ax.set_xlabel(f2_col, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"F2  ·  {col_labels[0]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.01, -0.28, _stats_text(f2_raw), transform=ax.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e0", alpha=0.8))

    ax = fig.add_subplot(gs[2, 1])
    ax.hist(f2_transformed, bins=60, color=row_colors["F2"], edgecolor="white", alpha=0.85)
    ax.axvline(f2_crit_lo_t, color="red", linestyle="--", linewidth=1.4,
               label=f"Critical boundary (>{f2_crit_lo_t:.3f})\n{f2_pos_t}/{len(f2_transformed)} critical ({100*f2_pos_t/len(f2_transformed):.1f}%)")
    ax.set_xlabel("log1p(F2 + shift)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"F2  ·  {col_labels[1]}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.35)
    ax.text(0.01, -0.28, _stats_text(f2_transformed), transform=ax.transAxes,
            fontsize=7.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e0", alpha=0.8))

    # ─────────────────────────────────────────────────────────────────────────
    # ROW 3 — F2 boxplots
    # ─────────────────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    ax.boxplot(f2_raw, vert=False, patch_artist=True,
               boxprops=dict(facecolor=row_colors["F2"], alpha=0.7),
               medianprops=dict(color="black", linewidth=2),
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.axvline(0, color="red", linestyle="--", linewidth=1.4, label="Critical boundary (> 0)")
    ax.set_xlabel(f2_col, fontsize=10)
    ax.set_title("F2  ·  Box plot (raw)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.35)

    ax = fig.add_subplot(gs[3, 1])
    ax.boxplot(f2_transformed, vert=False, patch_artist=True,
               boxprops=dict(facecolor=row_colors["F2"], alpha=0.7),
               medianprops=dict(color="black", linewidth=2),
               flierprops=dict(marker="o", markersize=3, alpha=0.4))
    ax.axvline(f2_crit_lo_t, color="red", linestyle="--", linewidth=1.4,
               label=f"Critical boundary (>{f2_crit_lo_t:.3f})")
    ax.set_xlabel("log1p(F2 + shift)", fontsize=10)
    ax.set_title("F2  ·  Box plot (transformed)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.35)

    # ─────────────────────────────────────────────────────────────────────────
    # Global title
    # ─────────────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Fitness Distribution: Before vs After log1p Transform\n"
        f"Dataset: {data_folder_name}  |  n={len(df)}  |  y_shift={y_shift:.6f}",
        fontsize=12, fontweight="bold", y=1.01
    )

    out_path = os.path.join(output_folder, f"fitness_distribution_transform_{data_folder_name}.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    print(f"\n[INFO] Plot saved to: {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Console summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  y_shift applied = {y_shift:.6f}")
    print(f"{'='*60}")

    for label, raw, transformed, crit_lo_t, crit_hi_t, crit_desc in [
        ("F1 (Min Distance)", f1_raw, f1_transformed,
         f1_crit_lo_t, f1_crit_hi_t,
         f"raw critical: (0 ≤ F1 ≤ 1) → {int(np.sum((f1_raw >= 0) & (f1_raw <= 1)))} samples"),
        ("F2 (Velocity at Min Dist)", f2_raw, f2_transformed,
         f2_crit_lo_t, f2_crit_lo_t,
         f"raw critical: (F2 > 0) → {int(np.sum(f2_raw > 0))} samples"),
    ]:
        print(f"\n--- {label} ---")
        print(f"  RAW        min={raw.min():.4f}  max={raw.max():.4f}  "
              f"mean={raw.mean():.4f}  std={raw.std():.4f}")
        print(f"  TRANSFORMED min={transformed.min():.4f}  max={transformed.max():.4f}  "
              f"mean={transformed.mean():.4f}  std={transformed.std():.4f}")
        print(f"  {crit_desc}")
        print(f"  Critical boundary in transformed space: > {crit_lo_t:.4f}")
        compression = (raw.max() - raw.min()) / max((transformed.max() - transformed.min()), 1e-9)
        print(f"  Range compression factor (raw/transformed): {compression:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Plot F1/F2 distributions before and after log1p transform"
    )
    parser.add_argument(
        "--data_folder", type=str,
        default="./surrogate_log/data/batch0/",
        help="Path to the folder containing training CSV files"
    )
    args = parser.parse_args()

    data_folder_name = os.path.basename(os.path.normpath(args.data_folder))
    print(f"[INFO] Loading data from: {args.data_folder}")

    df, target_cols = load_data(args.data_folder)
    print(f"[INFO] Loaded {len(df)} samples. Target columns: {target_cols}")

    plot_fitness_distributions(df, target_cols, output_folder, data_folder_name)


if __name__ == "__main__":
    main()