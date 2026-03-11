import os
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from mycoverage.compute_vector import compute_interaction_vector
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.cm as cm
from mycoverage.plot_bundle_io import save_plot_bundle

# -------------------------
# Load all simouts from multiple folders or list of files
# -------------------------
def load_all_simouts(sources):
    """
    sources: list of folder paths or list of lists of file paths per approach
    Returns:
        - all_simouts: list of loaded simout dicts
        - simout_mapping: np.array of approach index per scenario
        - simout_paths: list of file paths
    """
    all_simouts = []
    simout_mapping = []
    simout_paths = []

    for approach_idx, src in enumerate(sources):
        # Determine if src is a folder or a list of files
        if isinstance(src, str) and os.path.isdir(src):
            simout_files = [os.path.join(src, f) for f in os.listdir(src) if f.endswith(".json")]
            simout_files.sort()
        elif isinstance(src, list):
            simout_files = src
        else:
            raise ValueError(f"Invalid source: {src}")

        for f in simout_files:
            with open(f, "r") as fin:
                simout = json.load(fin)
            all_simouts.append(simout)
            simout_mapping.append(approach_idx)
            simout_paths.append(f)

    return all_simouts, np.array(simout_mapping), simout_paths


# -------------------------
# Compute standardized vectors
# -------------------------
def compute_standardized_vectors(simout_list, project_name="planer_final"):
    vectors = np.array([compute_interaction_vector(simout, project_name=project_name) for simout in simout_list])
    mean = np.mean(vectors, axis=0)
    std = np.std(vectors, axis=0)
    std[std == 0] = 1.0
    vectors_std = (vectors - mean) / std
    return vectors_std


# -------------------------
# Compute global DBSCAN clusters
# -------------------------
def compute_global_clusters(vectors_std, eps=1.5, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    db.fit(vectors_std)
    return db.labels_


# -------------------------
# Compute coverage and normalized entropy
# -------------------------
def compute_coverage_entropy(labels, simout_mapping, critical_indices):
    """
    labels: global cluster labels from DBSCAN
    simout_mapping: approach index per scenario
    critical_indices: list of indices of critical simouts in all_simouts
    """
    approaches = np.unique(simout_mapping)
    cluster_set = set(labels[labels != -1])
    n_clusters = len(cluster_set)
    
    coverage_dict = {}
    entropy_dict = {}
    
    for app in approaches:
        app_indices = np.where(simout_mapping == app)[0]
        
        # Coverage
        clusters_hit = set(labels[app_indices])
        clusters_hit.discard(-1)  # ignore noise
        coverage = len(clusters_hit) / n_clusters if n_clusters > 0 else 0
        coverage_dict[app] = coverage
        
        # Entropy (only considering critical failures in this approach)
        crit_in_app = list(set(app_indices) & set(critical_indices))
        if len(crit_in_app) == 0:
            entropy_dict[app] = 0.0
            continue
        
        clusters_of_failures = labels[crit_in_app]
        clusters_of_failures = clusters_of_failures[clusters_of_failures != -1]  # ignore noise
        if len(clusters_of_failures) == 0:
            entropy_dict[app] = 0.0
            continue
        counts = np.array(list(Counter(clusters_of_failures).values()))
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log2(probs))
        H_norm = H / np.log2(n_clusters) if n_clusters > 1 else 0
        entropy_dict[app] = H_norm
    
    return coverage_dict, entropy_dict

def plot_tsne_embedding(
    vectors_std,
    simout_mapping,
    labels,
    simout_paths,
    approach_names=None,
    perplexity=5,
    random_state=42,
    save_folder="./mycoverage/plots",
    annotate=False,
    annotate_max=200,
    annotate_with="basename",  # "basename" | "stem" | "full"
    alpha=0.9,
    s=260,                    # scatter area in points^2
    X_2d=None,

    # Paper sizing/styling
    figsize=(11, 8),
    dpi=600,
    font_size=22,
    label_size=24,
    title_size=26,
    legend_fontsize=16,
    legend_title_fontsize=17,
    line_width=1.4,

    # HiFi half-fill direction: "left"|"right"|"top"|"bottom"
    hifi_fillstyle="left",

    # always save two plots (with noise + without noise)
    save_without_noise=True,
):
    """
    Paper-ready t-SNE plot:
      - Color = DBSCAN cluster label
      - Approach encoding:
          * Lofi      (0): STAR markers (distinct from others)
          * HiFi      (1): HALF-FILLED markers (cluster color on one half)
          * SimFusion (2): filled markers
      - Noise points (label=-1): black 'x'

    Saves:
      - ..._with_noise.(png|pdf)
      - ..._no_noise.(png|pdf)  (if save_without_noise=True)
    """

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "legend.fontsize": legend_fontsize,
        "legend.title_fontsize": legend_title_fontsize,
        "axes.linewidth": line_width,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
    })

    if X_2d is None:
        print("[INFO] Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        X_2d = tsne.fit_transform(vectors_std)

    n = len(labels)
    assert len(simout_mapping) == n
    assert len(simout_paths) == n

    if approach_names is None:
        approach_names = ["Lofi", "HiFi", "SimFusion"]

    approach_markers = {
        0: "*",  # Lofi -> star
        1: "s",  # HiFi
        2: "^",  # SimFusion
    }

    cluster_ids = sorted([c for c in set(labels) if c != -1])
    cmap = cm.get_cmap("tab20", max(len(cluster_ids), 1))
    cluster_color = {cid: cmap(i) for i, cid in enumerate(cluster_ids)}

    markersize = float(np.sqrt(s + 1))
    n_approaches = int(np.max(simout_mapping)) + 1 if n > 0 else 0

    def _draw_one(ax, *, include_noise: bool):
        for app_idx in range(n_approaches):
            idxs = np.where(simout_mapping == app_idx)[0]
            if len(idxs) == 0:
                continue

            marker = approach_markers.get(app_idx, "o")

            # Noise points
            noise_mask = labels[idxs] == -1
            if include_noise and np.any(noise_mask):
                pts = X_2d[idxs][noise_mask]
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    c="k", marker="x", s=s * 0.9, alpha=alpha,
                    linewidths=2.2,
                    zorder=3,
                )

            clustered_mask = labels[idxs] != -1
            clustered_idxs = idxs[clustered_mask]

            for cid in cluster_ids:
                cid_mask = labels[clustered_idxs] == cid
                if not np.any(cid_mask):
                    continue

                pts = X_2d[clustered_idxs][cid_mask]
                col = cluster_color[cid]

                if app_idx == 2:
                    # SimFusion: filled
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        facecolors=[col],
                        edgecolors="black",
                        marker=marker,
                        s=s,
                        linewidths=1.2,
                        alpha=alpha,
                        zorder=2,
                    )

                elif app_idx == 0:
                    # Lofi: star marker (filled with cluster color for visibility)
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        facecolors=[col],
                        edgecolors="black",
                        marker=marker,
                        s=s,
                        linewidths=1.2,
                        alpha=alpha,
                        zorder=2,
                    )

                elif app_idx == 1:
                    # HiFi: half-filled
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        facecolors="none",
                        edgecolors="black",
                        marker=marker,
                        s=s,
                        linewidths=1.8,
                        alpha=1.0,
                        zorder=2,
                    )
                    ax.plot(
                        pts[:, 0], pts[:, 1],
                        linestyle="None",
                        marker=marker,
                        markersize=markersize,
                        markeredgecolor="black",
                        markeredgewidth=1.2,
                        markerfacecolor=col,
                        markerfacecoloralt="white",
                        fillstyle=hifi_fillstyle,
                        alpha=alpha,
                        zorder=4,
                    )
                else:
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        facecolors=[col],
                        edgecolors="black",
                        marker=marker,
                        s=s,
                        linewidths=1.2,
                        alpha=alpha,
                        zorder=2,
                    )

        # --- Legends (outside) ---
        h_lofi = plt.Line2D(
            [0], [0],
            marker=approach_markers[0],
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=16,
            label="Lofi",
        )

        h_hifi = plt.Line2D(
            [0], [0],
            marker=approach_markers[1],
            linestyle="None",
            color="black",
            markeredgecolor="black",
            markeredgewidth=1.8,
            markerfacecolor="black",
            markerfacecoloralt="white",
            fillstyle=hifi_fillstyle,
            markersize=14,
            label="HiFi",
        )

        h_simfusion = plt.Line2D(
            [0], [0],
            marker=approach_markers[2],
            linestyle="None",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markeredgewidth=1.2,
            markersize=14,
            label="SimFusion",
        )

        cluster_handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                color="w",
                markerfacecolor=cluster_color[cid],
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=12,
                label=f"Cluster {cid}",
            )
            for cid in cluster_ids
        ]

        ax.figure.subplots_adjust(right=0.74)

        leg1 = ax.legend(
            handles=[h_lofi, h_hifi, h_simfusion],
            title="Approach",
            loc="upper left",
            bbox_to_anchor=(1.02, 1.00),
            borderaxespad=0.0,
            frameon=True,
        )
        ax.add_artist(leg1)

        if include_noise:
            noise_handle = plt.Line2D(
                [0], [0],
                marker="x",
                linestyle="None",
                color="k",
                markersize=12,
                label="Noise (-1)",
            )
            cluster_legend_handles = cluster_handles + [noise_handle]
        else:
            cluster_legend_handles = cluster_handles

        ax.legend(
            handles=cluster_legend_handles,
            title="Clusters",
            loc="upper left",
            bbox_to_anchor=(1.02, 0.55),
            borderaxespad=0.0,
            frameon=True,
        )

        # Annotation
        if annotate:
            if include_noise:
                candidate_idxs = np.arange(n)
            else:
                candidate_idxs = np.where(labels != -1)[0]

            to_annotate = candidate_idxs
            if len(candidate_idxs) > annotate_max:
                to_annotate = np.random.RandomState(random_state).choice(
                    candidate_idxs, size=annotate_max, replace=False
                )

            for i in to_annotate:
                p = simout_paths[i]
                if annotate_with == "full":
                    txt = p
                elif annotate_with == "stem":
                    txt = Path(p).stem
                else:
                    txt = os.path.basename(p)
                ax.text(X_2d[i, 0], X_2d[i, 1], txt, fontsize=10, alpha=0.85)

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linewidth=0.7, alpha=0.35)

    # Plot 1: WITH noise
    fig1, ax1 = plt.subplots(figsize=figsize)
    _draw_one(ax1, include_noise=True)
    fig1.tight_layout()

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    out_png = Path(save_folder) / "tsne_clusters_and_approaches_paper_with_noise.png"
    out_pdf = Path(save_folder) / "tsne_clusters_and_approaches_paper_with_noise.pdf"
    fig1.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig1.savefig(out_pdf, bbox_inches="tight")

    # Plot 2: WITHOUT noise
    if save_without_noise:
        fig2, ax2 = plt.subplots(figsize=figsize)
        _draw_one(ax2, include_noise=False)
        fig2.tight_layout()

        out_png2 = Path(save_folder) / "tsne_clusters_and_approaches_paper_no_noise.png"
        out_pdf2 = Path(save_folder) / "tsne_clusters_and_approaches_paper_no_noise.pdf"
        fig2.savefig(out_png2, dpi=dpi, bbox_inches="tight")
        fig2.savefig(out_pdf2, bbox_inches="tight")

    plt.show()
    return X_2d
# -------------------------
# Full pipeline with coverage & entropy
# -------------------------
def coverage_pipeline(folder_list, critical_simout_paths=[], eps=1.5, min_samples=2, save_folder="./mycoverage/plots", project_name="planer_final"):
    all_simouts, simout_mapping, simout_paths = load_all_simouts(folder_list)
    vectors_std = compute_standardized_vectors(all_simouts, project_name=project_name)
    labels = compute_global_clusters(vectors_std, eps=eps, min_samples=min_samples)

    # cluster_to_scenarios ...
    cluster_to_scenarios = {}
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cluster_to_scenarios.setdefault(label, []).append(idx)

    print(f"[INFO] Total clusters (excluding noise): {len(cluster_to_scenarios)}")

    critical_indices = [i for i, p in enumerate(simout_paths) if p in critical_simout_paths]

    coverage_dict, entropy_dict = compute_coverage_entropy(labels, simout_mapping, critical_indices)
    for app in coverage_dict:
        print(f"[INFO] Approach {app+1}: Coverage={coverage_dict[app]:.2f}, Entropy={entropy_dict[app]:.2f}")
        
    for p in [5,10,20,30,50]:
        tsne_params = {"perplexity": p, "random_state": 42}
        X_2d = plot_tsne_embedding(
            vectors_std=vectors_std,
            simout_mapping=simout_mapping,
            labels=labels,
            simout_paths=simout_paths,
            approach_names=[f"Approach {i+1}" for i in range(len(folder_list))],
            annotate=False,
            annotate_with="stem",
            save_folder=Path(save_folder) / str(p),
            perplexity=tsne_params["perplexity"],
            random_state=tsne_params["random_state"],
            X_2d=None,  # computed inside
        )

        # Persist everything needed to replot
        bundle_name = "embedding_bundle"
        save_plot_bundle(
            save_folder=Path(save_folder) / str(p),
            bundle_name=bundle_name,
            vectors_std=vectors_std,
            labels=labels,
            simout_mapping=simout_mapping,
            simout_paths=simout_paths,
            folder_list=folder_list,
            critical_simout_paths=critical_simout_paths,
            tsne_params=tsne_params,
            X_2d=X_2d,
        )
        print(f"[INFO] Saved plot bundle: {Path(save_folder) / str(p) / (bundle_name + '.npz')}")

    return vectors_std, labels, simout_mapping, cluster_to_scenarios, simout_paths, coverage_dict, entropy_dict

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    folders = [
        "/path/to/approach1/simouts",
        "/path/to/approach2/simouts",
        "/path/to/approach3/simouts"
    ]
    
    vectors_std, labels, simout_mapping, cluster_to_scenarios, simout_paths, coverage_dict, entropy_dict = coverage_pipeline(folders, eps=1.5, min_samples=2)