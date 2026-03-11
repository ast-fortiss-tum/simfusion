import wandb
import os
import re

api = wandb.Api()

entity = "lofi-hifi"
project = "autoware_final"

runs = api.runs(f"{entity}/{project}")

download_root = "results_aw_wandb"
os.makedirs(download_root, exist_ok=True)

def safe_folder_name(name):
    return re.sub(r'[^\w\-_. ]', '_', name)

for run in runs:
    run_folder = safe_folder_name(run.name)
    run_base_path = os.path.join(download_root, run_folder)

    # Final folder = run_id (no extra artifact folder)
    run_id_path = os.path.join(run_base_path, run.id)

    for artifact in run.logged_artifacts():
        base_name = artifact.name.split(":")[0]

        if base_name == "results_folder":

            # ✅ Skip if already downloaded
            if os.path.exists(run_id_path):
                print(f"Skipping {run.name} ({run.id}) — already exists")
                continue

            print(f"Downloading {artifact.name} from run {run.name} ({run.id})")

            os.makedirs(run_id_path, exist_ok=True)
            artifact.download(root=run_id_path)

print("Done.")