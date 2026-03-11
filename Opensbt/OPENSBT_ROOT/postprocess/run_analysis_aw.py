from postprocess.wandb import download_run_artifacts_relative
from datetime import datetime, timezone
from typing import List
from postprocess.get_figures_here import *

from wandb import Run


def aw_final_filter(runs: List[Run], after: datetime = None) -> List[Run]:
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
        if seed is None or seed not in [1, 2, 3, 4, 5, 6]:
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

run_filters = {"aw_final": aw_final_filter}

# Example: only finished runs, after date, seed 10..15 etc. via your planer_final_filter
df = overview_table(
    wb_project_path="lofi-hifi/autoware_final",
    save_csv="wandb_analysis/aw_final/aw_final_overview.csv",
    experiments_folder="results_download",
    run_filter=aw_final_filter,  # or set a datetime
    one_per_name=False,
)
print(df)