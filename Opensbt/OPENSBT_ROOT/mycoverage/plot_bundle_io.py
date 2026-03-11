import json
from pathlib import Path
import numpy as np
import sklearn

def save_plot_bundle(
    save_folder,
    bundle_name,
    *,
    vectors_std,
    labels,
    simout_mapping,
    simout_paths,
    folder_list,
    critical_simout_paths,
    tsne_params=None,
    X_2d=None,
):
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # Numeric arrays in one compressed archive
    npz_path = save_folder / f"{bundle_name}.npz"
    np.savez_compressed(
        npz_path,
        vectors_std=vectors_std,
        labels=labels,
        simout_mapping=simout_mapping,
        X_2d=X_2d if X_2d is not None else np.array([]),
    )

    # Metadata in JSON (paths/lists/params/versions)
    meta = {
        "folder_list": list(folder_list),
        "critical_simout_paths": list(critical_simout_paths),
        "simout_paths": list(simout_paths),
        "tsne_params": tsne_params or {},
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    meta_path = save_folder / f"{bundle_name}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return str(npz_path), str(meta_path)


def load_plot_bundle(save_folder, bundle_name):
    save_folder = Path(save_folder)

    npz_path = save_folder / f"{bundle_name}.npz"
    meta_path = save_folder / f"{bundle_name}.json"

    data = np.load(npz_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text())

    X_2d = data["X_2d"]
    if X_2d.size == 0:
        X_2d = None

    return {
        "vectors_std": data["vectors_std"],
        "labels": data["labels"],
        "simout_mapping": data["simout_mapping"],
        "X_2d": X_2d,
        "meta": meta,
    }