from mycoverage.plot_bundle_io import load_plot_bundle
from mycoverage.compute_coverage import plot_tsne_embedding

bundle = load_plot_bundle("/home/user/testing/topic1-opensbt-aw/wandb_analysis/planer_final/diversity/plots", "embedding_bundle")

meta = bundle["meta"]
folder_list = meta["folder_list"]

plot_tsne_embedding(
    vectors_std=bundle["vectors_std"],           # not used if X_2d provided, but fine
    simout_mapping=bundle["simout_mapping"],
    labels=bundle["labels"],
    simout_paths=meta["simout_paths"],
    approach_names=[f"Approach {i+1}" for i in range(len(folder_list))],
    annotate=False,
    annotate_with="stem",
    save_folder=f"/home/user/testing/topic1-opensbt-aw/wandb_analysis/planer_final/diversity/plots/",
    X_2d=bundle["X_2d"],   
)