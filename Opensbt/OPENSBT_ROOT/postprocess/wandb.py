from __future__ import annotations

from collections import defaultdict
from typing import Optional, Callable, List, Dict, Iterable

import json
import os

import tqdm
from wandb import Api
from wandb.apis.public import Run

import io
import time
import os
import json
from typing import Callable, Optional, List, Dict, Tuple
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
from pymoo.core.algorithm import Algorithm

import wandb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.wandb_run import Run
import re

def safe_tag_map(run: Run) -> Dict[str, str]:
    """
    Converts tags like ["approach:hifi", "seed:3"] into {"approach":"hifi","seed":"3"}.
    Ignores tags without ":".
    """
    out: Dict[str, str] = {}
    for t in (run.tags or []):
        if ":" in t:
            k, v = t.split(":", 1)  # split once
            out[k] = v
    return out


def infer_approach(run: Run, tags: Dict[str, str]) -> str:
    """
    Decide SUT folder name: hifi / lofi / predict (fallback: unknown).
    Preference order:
      1) tags["approach"]
      2) run.name contains a keyword
    """
    if tags.get("approach"):
        return tags["approach"].strip().lower()

    name = (run.name or "").lower()
    if "hifi" in name or "hi-fi" in name or "hi_fi" in name:
        return "hifi"
    if "lofi" in name or "lo-fi" in name or "lo_fi" in name:
        return "lofi"
    if "predict" in name or "prediction" in name:
        return "predict"
    return "unknown"

def download_run_artifacts_relative(
    wb_project_path: str,
    local_root: str = "./opentest",
    filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None,
    one_per_name: bool = False,
    artifact_base_names: Iterable[str] = ("results_folder", "report", "rerun_folder"),
) -> Dict[str, List[str]]:
    """
    Download selected W&B run artifacts, using approach (hifi/lofi/predict) as the SUT folder.

    Local layout:
      <local_root>/<project>/artifacts/<approach>/<seed>/<run.name>/<run.id>/

    Artifact-specific layout:
      - "results_folder" → no extra subfolder
      - "rerun_folder"   → <local_path>/rerun_hifi/
      - "report"         → <local_path>/report/

    Always returns the local_path for each processed run (even if already downloaded).
    """
    artifact_base_names = set(artifact_base_names)

    api = Api()
    runs = list(api.runs(wb_project_path, per_page=1000))
    project_name = wb_project_path.split("/")[-1]

    if filter_runs is not None:
        runs = list(filter_runs(runs))

    print(f"Found {len(runs)} runs before grouping.")

    if one_per_name:
        grouped = defaultdict(list)
        for run in runs:
            grouped[run.name].append(run)
        selected_runs = [max(group, key=lambda r: r.created_at) for group in grouped.values()]
        print(f"Reduced to {len(selected_runs)} unique run names.")
    else:
        selected_runs = runs
        print(f"Downloading all {len(selected_runs)} runs.")

    downloaded_paths: Dict[str, List[str]] = defaultdict(list)

    for run in tqdm.tqdm(selected_runs):
        tags = safe_tag_map(run)
        approach = infer_approach(run, tags)
        seed = tags.get("seed", "no_seed")

        local_path = os.path.join(
            local_root,
            project_name,
            "artifacts",
            approach,
            seed,
            run.name,
            run.id,
        )
        os.makedirs(local_path, exist_ok=True)

        # Always record path
        downloaded_paths[approach].append(local_path)

        # Skip download if already populated
        if any(os.scandir(local_path)):
            continue

        for artifact in run.logged_artifacts():
            base_name = artifact.name.split(":", 1)[0]

            if base_name not in artifact_base_names:
                continue

            if base_name == "results_folder":
                target_path = local_path

            elif base_name == "rerun_folder":
                target_path = os.path.join(local_path, "rerun_hifi")
                os.makedirs(target_path, exist_ok=True)

            elif base_name == "report":
                target_path = os.path.join(local_path, "report")
                os.makedirs(target_path, exist_ok=True)

            else:
                target_path = local_path

            artifact.download(target_path)

        # Save metadata
        with open(os.path.join(local_path, "tags.json"), "w") as f:
            json.dump(tags, f, indent=2)

        history_df = run.history(keys=None, pandas=True)
        history_df.to_csv(os.path.join(local_path, "run_history.csv"), index=False)

    print("Done.")
    return dict(downloaded_paths)

def wandb_log_csv(filename):
    # log only if file is not empty
    if (
        Path(filename).exists()
        and Path(filename).is_file()
        and os.stat(filename).st_size > 0
    ):
        try:
            df = pd.read_csv(filename)
            metric_table = wandb.Table(dataframe=df)
            metric_table_artifact = wandb.Artifact("metric_history", type="dataset")
            metric_table_artifact.add(metric_table, "metric_table")
            metric_table_artifact.add_file(filename)
            wandb.log({"log": metric_table})
            wandb.log_artifact(metric_table_artifact, name=filename)
        except io.UnsupportedOperation:
            print(f"Cannot log {filename}. Check if it is in .csv format.")
    else:
        print(f"{filename} does not exist or it is empty.")



def logging_callback(algorithm: Algorithm):
    all_population = algorithm.pop
    critical_all, _ = all_population.divide_critical_non_critical()
    wandb.log(
        {
            "population_size": len(all_population),
            "failures": len(critical_all),
            "critical_ratio": len(critical_all) / len(all_population),
            "timestamp": time.time()
        }
    )

def logging_callback_archive(algorithm: Algorithm):
    if hasattr(algorithm, "archive") and algorithm.archive is not None:
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        wandb.log(
            {
                "test_size": len(all_population),
                "failures": len(critical_all),
                "critical_ratio": len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
                "timestamp": time.time()
            }
        )

class TableCallback:
    def __init__(self):
        self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")

    def log(self, algorithm: Algorithm):
        all_population = algorithm.archive
        critical_all, _ = all_population.divide_critical_non_critical()
        self.table.add_data(
            len(all_population),
            len(critical_all),
            len(critical_all) / len(all_population) if len(all_population) > 0 else 0.0,
            time.time()
        )
        wandb.log({"Summary Table": self.table})

    def __getstate__(self):
        """Return state for pickling (skip unpicklable parts)."""
        state = self.__dict__.copy()
        state["table"] = None
        return state

    def __setstate__(self, state):
        """Recreate skipped attributes after unpickling."""
        self.__dict__.update(state)
        if self.table is None:
            self.table = wandb.Table(columns=["test_size", "failures", "critical_ratio", "timestamp"],
                                 log_mode="MUTABLE")
            

def get_sut(project, run, tags):
    if project == "SafeLLM":
        return tags.get("sut", "unknown sut")
    else:
        # Regex to match everything up to "_<number>n"
        match = re.match(r"^(.*)_\d+n", run.name)
        if match:
            return match.group(1)
        else:
            raise Exception("SUT name could not be extracted from run name")
        
def download_run_artifacts(path: str, 
                           filter_runs: Optional[Callable[[List[Run]], List[Run]]] = None) -> Dict[str, Dict[str, List[str]]]:
    runs = Api().runs(
        path,
        per_page = 1000
    )
    project = path.split("/")[-1]
    if filter_runs is not None:
        runs = filter_runs(runs)

    all_paths = defaultdict(lambda: defaultdict(list))
    missing_runs = []
    for run in runs:
        tags = {t.split(":")[0]: t.split(":")[1] for t in run.tags}
        local_path = os.path.join(
            "results",
            "artifacts",
            path,
            tags.get("sut", "unknown sut"),
            tags.get("algorithm", "unknown algo"),
            tags.get("seed", "no seed"),
            run.name,
            run.id,
        )
        os.makedirs(local_path, exist_ok=True)
        if not os.path.exists(os.path.join(local_path, "run_history.csv")):
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
        all_paths[get_sut(project, run, tags)][(tags.get("algorithm", "unknown algo"), tags.get("features", "default"))].append(local_path)
        if not os.path.exists(os.path.join(local_path, "config.txt")):
            missing_runs.append((run, local_path))
    for run, path in tqdm.tqdm(missing_runs):
        artifacts = run.logged_artifacts()
        if len(artifacts) > 5:
            previous = artifacts[0]
            for a in artifacts:
                if a.type == "output":
                    a.download(path)
                    previous.download(path)
                    break
                else:
                    previous = a
            with open(os.path.join(path, "tags.json"), "w") as f:
                json.dump(tags, f)
            run.history().to_csv(os.path.join(local_path, "run_history.csv"), index=False)
    return all_paths



def get_summary(artifact_directory_path: str):
    summary_table = pd.read_csv(os.path.join(artifact_directory_path, "summary_results.csv"))
    summary = {
        attr: value for attr, value in
        zip(summary_table.Attribute, summary_table.Value)
    }
    return summary


def get_run_table(artifact_directory_path: str, freq: str = "1min", interpolate_duplicates: bool = True):
    if os.path.exists(os.path.join(artifact_directory_path, "Summary Table.table.json")):
        with open(os.path.join(artifact_directory_path, "Summary Table.table.json")) as f:
            table_dict = json.load(f)
            df = pd.DataFrame(data=table_dict["data"], columns=table_dict["columns"])
    else:
        df = pd.read_csv(os.path.join(artifact_directory_path, "run_history.csv"))
        df = df[["failures", "test_size", "critical_ratio", "timestamp"]].dropna()
    summary = get_summary(artifact_directory_path)
    if interpolate_duplicates:
        no_duplicates_ratio = float(summary["Number Critical Scenarios (duplicate free)"]) \
            / float(summary["Number Critical Scenarios"])
        df["failures"] = df.failures * no_duplicates_ratio
        df["critical_ratio"] = df["critical_ratio"] * no_duplicates_ratio
    df["time"] = df.timestamp.mul(1e9).apply(pd.Timestamp)
    df.time = df.time - df.time.iloc[0]
    df = df.set_index("time")
    df = pd.concat([df, df.resample(freq).asfreq()]).sort_index().interpolate("index").resample(freq).first()
    return df

if __name__ == "__main__":
    paths = download_run_artifacts_relative(
        wb_project_path="lofi-hifi/planer_final",
        local_root="results_download",
        one_per_name=False,
        artifact_base_names=("results_folder", "validation"),
    )
    print(paths)