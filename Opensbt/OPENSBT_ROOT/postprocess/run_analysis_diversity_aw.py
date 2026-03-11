from postprocess.wandb import download_run_artifacts_relative
from datetime import datetime, timezone
from typing import List
from postprocess.get_figures_here import *
from postprocess.get_diversity import diversity_report_from_simout
from wandb import Run


def planer_final_filter(runs: List[Run], after: datetime = None) -> List[Run]:
    res = []
    for run in runs:
        print("run name:", run.name)

        # Ensure created_at is timezone-aware
        created = run.created_at
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))

        # ---- seed in [10, 15] check (inclusive) ----
        seed = None
        for t in (run.tags or []):
            if t.startswith("seed:"):
                try:
                    seed = int(t.split(":", 1)[1].strip())
                except ValueError:
                    seed = None
                break
        if seed is None or seed not in [1,2,3,4,5,6]: # 15,16,17
            continue
 
        # --------------------------------------------
        print("run.state:", run.state)
        print("run.name:", run.name)
        if run.state != "crashed" and "GA" in (run.name or ""):
            # Time-based filtering
            if after and not (created > after):
                continue
            res.append(run)

    return res

run_filters = {"planer_final": planer_final_filter}

df = diversity_report_from_simout(
    wb_project_path="lofi-hifi/autoware_final",
    save_csv="aw_final_diversity.csv",
    experiments_folder="results_download",
    run_filter=planer_final_filter,  # or set a datetime
    one_per_name=False,
    output_dir="./wandb_analysis/autoware_final/diversity",
    eps=3,
    min_samples=6,
    entity="lofi-hifi",
    validation_project="lofi-validation",
)
print(df)
