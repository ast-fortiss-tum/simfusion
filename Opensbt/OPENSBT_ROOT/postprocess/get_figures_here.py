import os
import csv
import json
from pathlib import Path
import re
import pandas as pd
from typing import Tuple, Iterable, Optional

from wandb import Api
from postprocess.wandb import download_run_artifacts_relative
from wandb.apis.public import Run
import numpy as np

def overview_table(
    wb_project_path: str,
    save_csv: str = "table.csv",
    experiments_folder: str = "results_wandb",
    run_filter=None,
    one_per_name: bool = False,
    save_raw: bool = True,
):
    artifact_paths = download_run_artifacts_relative(
        wb_project_path=wb_project_path,
        local_root=experiments_folder,
        filter_runs=run_filter,
        one_per_name=one_per_name,
        artifact_base_names=("results_folder", "report", "rerun_folder"),
    )

    summary_rows = []

    for sut, paths in artifact_paths.items():
        tests = []
        fails = []
        ratios = []
        raw_rows = []

        for p in paths:
            n_tests, n_fail = get_real_tests(p, wandb_project=wb_project_path.split("/")[1])
            ratio = (n_fail / n_tests) if n_tests else 0.0

            tests.append(n_tests)
            fails.append(n_fail)
            ratios.append(ratio)

            raw_rows.append({
                "SUT": sut,
                "run_path": p,
                "tests": n_tests,
                "failures": n_fail,
                "failure_ratio": ratio,
            })

        # ---- Save raw data per SUT ----
        if save_raw and raw_rows:
            raw_df = pd.DataFrame(raw_rows)
            raw_dir = os.path.splitext(save_csv)[0] + "_raw"
            os.makedirs(raw_dir, exist_ok=True)

            raw_path = os.path.join(raw_dir, f"{sut}_raw.csv")
            raw_df.to_csv(raw_path, index=False)
            print(f"Saved raw data: {raw_path}")

        # ---- Aggregation ----
        row = {"SUT": sut, "n_runs": len(paths)}

        row["tests_mean"] = float(np.mean(tests)) if tests else np.nan
        row["tests_std"] = float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0

        row["failures_mean"] = float(np.mean(fails)) if fails else np.nan
        row["failures_std"] = float(np.std(fails, ddof=1)) if len(fails) > 1 else 0.0

        row["failure_ratio_mean"] = float(np.mean(ratios)) if ratios else np.nan
        row["failure_ratio_std"] = float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df = df.sort_values(by=["SUT"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
    df.to_csv(save_csv, index=False)
    print(f"Saved: {save_csv}")

    latex_path = os.path.splitext(save_csv)[0] + ".tex"
    df.to_latex(latex_path, index=False, float_format="%.3f")
    print(f"Saved: {latex_path}")

    return df
# -------------------------
# Shared helpers (single copy)
# -------------------------

def _safe_folder_name(name: str) -> str:
    return re.sub(r"[^\w\-_. ]", "_", name or "")


def _run_kind(path_name: str) -> str:
    parts = {p.lower() for p in os.path.normpath(path_name).split(os.sep)}
    if "predict" in parts:
        return "predict"
    if "lofi" in parts:
        return "lofi"
    if "hifi" in parts:
        return "hifi"
    else:
        raise ValueError(f"Could not determine run kind from path: {path_name}")


def _load_summary_results_csv(path_name: str) -> dict:
    candidates = [
        os.path.join(path_name, "summary_results.csv"),
        os.path.join(path_name, "summary_results"),
    ]
    fp = next((p for p in candidates if os.path.exists(p)), None)
    if fp is None:
        raise FileNotFoundError(f"Could not find summary_results(.csv) in: {path_name}")

    out = {}
    with open(fp, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attr = (row.get("Attribute") or "").strip()
            val = (row.get("Value") or "").strip()
            if attr:
                out[attr] = val
    return out


def _read_all_testcases(path_name: str):
    fp = os.path.join(path_name, "all_testcases.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing all_testcases.csv in {path_name}")
    with open(fp, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_report_json(path_name: str) -> str:
    # report.json is inside downloaded artifacts; search recursively
    for root, _, files in os.walk(path_name):
        if "report.json" in files:
            return os.path.join(root, "report.json")
    raise FileNotFoundError(f"report.json not found under: {path_name}")


def _extract_failures_from_report(report: dict) -> int:
    """
    Your script indicates the key is 'estimated_failures'.
    If it's actually validated failures, rename the key here.
    """
    v = report.get("estimated_failures", None)
    print("estimated failures found:", v)
    if isinstance(v, (int, float)):
        return int(v)
    try:
        raise KeyError("Could not find 'estimated_failures' in report.json")
    except KeyError:
        print("Falling back to 'total_failures' as estimated failures not found")
        v = report.get("total_failures", 0)
        return int(v) if isinstance(v, (int, float)) else 0


def _read_total_failures_from_agree_summary(path_name: str) -> int:
    """
    Predict runs: rerun_hifi/agree_summary_combined.csv contains:
      metric,count
      ...
      total_failures,<number>

    Returns int(total_failures).
    """
    fp = os.path.join(path_name, "rerun_hifi", "agree_summary_combined.csv")
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing {fp}")

    with open(fp, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("metric") or "").strip() == "critical_hifi":
                return int(float(row["count"]))

    raise KeyError(f"'total_failures' row not found in {fp}")


def _parse_local_run_components(run_local_path: str):
    """
    Expect:
      <local_root>/<source_project>/artifacts/<kind>/<seed>/<run.name>/<run.id>/
    Returns:
      (local_root, source_project, kind, seed, run_name)
    """
    parts = os.path.normpath(run_local_path).split(os.sep)
    try:
        artifacts_i = parts.index("artifacts")
    except ValueError:
        raise ValueError(f"Path does not contain 'artifacts': {run_local_path}")

    source_project = parts[artifacts_i - 1]
    kind = parts[artifacts_i + 1]
    seed = parts[artifacts_i + 2]
    run_name = parts[artifacts_i + 3]

    local_root = os.sep.join(parts[: artifacts_i - 1])
    return local_root, source_project, kind.lower(), seed, run_name


def _download_validation_run_for_lofi(
    *,
    lofi_run_local_path: str,
    entity: str,
    validation_project: str,
    validation_prefix: str = "Validation_",
    local_validation_folder: str = "lofi-validation",
    artifact_base_names: Iterable[str] = ("validation",),
) -> str:
    """
    Download the W&B validation run for a LoFi run:
      run.name == f"{validation_prefix}{lofi_run.name}"
    and download artifact(s) (typically "validation") that contain report.json.
    """
    local_root, source_project, kind, seed, run_name = _parse_local_run_components(lofi_run_local_path)
    if kind != "lofi":
        raise ValueError(f"Expected lofi run, got kind={kind}: {lofi_run_local_path}")

    # If already downloaded locally, skip W&B lookup/download.
    # Mirror the same sanitization used for the W&B run.name matching.
    expected_validation_name = f"{validation_prefix}{run_name}".replace(":", "_")
    local_validation_dir = (
        Path(local_root) / source_project / "artifacts" / "lofi-validation" / str(seed) / expected_validation_name
    )
    print("local_validation_dir:", local_validation_dir)
    if Path(local_validation_dir).is_dir(): 
        print(f"Validation run already exists locally")
        _find_report_json(local_validation_dir)
        return local_validation_dir

    print("Looking for validation run with name:", expected_validation_name)
    api = Api()
    matches = [r for r in api.runs(f"{entity}/{validation_project}", per_page=1000) if r.name == expected_validation_name]
    if not matches:
        raise FileNotFoundError(
            f"No matching validation run found: {entity}/{validation_project} run.name == {expected_validation_name}"
        )
    # max one run (newest)
    vrun: Run = max(matches, key=lambda r: r.created_at)

    v_local_path = os.path.join(
        local_root,
        source_project,
        "artifacts",
        local_validation_folder,
        seed,
        _safe_folder_name(vrun.name),
        vrun.id,
    )
    os.makedirs(v_local_path, exist_ok=True)

    # If report.json is already present, no need to download
    try:
        _find_report_json(v_local_path)
        return v_local_path
    except FileNotFoundError:
        pass

    wanted = {a.split(":", 1)[0] for a in artifact_base_names}
    for artifact in vrun.logged_artifacts():
        base = artifact.name.split(":", 1)[0]
        if base in wanted:
            artifact.download(root=v_local_path)

    # Verify we now have report.json
    _find_report_json(v_local_path)
    return v_local_path


# -------------------------
# Final get_real_tests
# -------------------------

def get_real_tests(
    path_name: str,
    th_obstacle: float = 1.0,
    th_goal: float = 1.0,
    *,
    entity: str = "lofi-hifi",
    validation_project: str = "lofi-validation",
    wandb_project = "planer_final"
) -> Tuple[int, int]:
    """
    Returns (num_duplicate_free_tests, num_duplicate_free_failures).

    - Predict runs:
        tests  = from summary_results.csv in the predict run folder
        fails  = from rerun_hifi/agree_summary_combined.csv, row metric == 'total_failures'

    - Non-lofi (hifi/other) runs:
        tests  = "Number All Scenarios (duplicate free)"
        fails  = "Number Critical Scenarios (duplicate free)"

    - LoFi runs:
        tests  = "Number All Scenarios (duplicate free)" from LoFi summary_results.csv
        fails  = from report.json of downloaded validation run:
                  {entity}/{validation_project} run.name == "Validation_<lofi_run.name>"
                (artifact type "validation" is downloaded to get report.json)
        fallback = old threshold logic on all_testcases.csv:
                  obs < th_obstacle OR goal > th_goal
    """
    s = _load_summary_results_csv(path_name)
    num_tests_dupfree = int(float(s["Number All Scenarios (duplicate free)"]))

    kind = _run_kind(path_name)

    if kind == "predict":
        print("[INFO] Predict run detected, using agree_summary_combined.csv for failures.")
        num_fail_dupfree = _read_total_failures_from_agree_summary(path_name)
        return num_tests_dupfree, int(num_fail_dupfree)
    elif kind == "hifi":
        print("[INFO] Non-LoFi run detected, using summary_results.csv values for tests and failures.")
        num_fail_dupfree = int(float(s["Number Critical Scenarios (duplicate free)"]))
        return num_tests_dupfree, num_fail_dupfree
    else:
        print("[INFO] LoFi run detected, attempting to download validation run artifact for failures.")
        # LoFi preferred: download validation run artifact and parse report.json
        ############## NEEDED THIS CODE BEFORE BECAUSE ORACLE WAS CHANGED 
        #         # --- LOFI special handling ---
        # rows = _read_all_critical_testcases(path_name)

        # def fnum(row, key):
        #     v = row.get(key, "")
        #     try:
        #         return float(v)
        #     except ValueError:
        #         return None

        # fail = 0
        # for r in rows:
        #     obs = fnum(r, "Fitness_Min distance to obstacle")
        #     goal = fnum(r, "Fitness_Distance to goal")
        #     if obs is None or goal is None:
        #         continue

        #     # failure rule (AND). Change to "or" if needed.
        #     if (obs <= th_obstacle) and (goal <= th_goal):
        #         fail += 1

        # return num_tests_dupfree, int(fail)
        if wandb_project == "planer_final":
            # If report.json already exists locally, don't download again
            local_validation_folder = "lofi-validation"

            # First try: report already in the current folder (path_name)
            # Second try: report already in the cached validation folder (lofi-validation)
            report_path = None
            try:
                report_path = _find_report_json(path_name)
                validation_local_path = path_name
                print(f"[INFO] Found existing report.json in {path_name}; skipping download.")
            except FileNotFoundError:
                try:
                    report_path = _find_report_json(local_validation_folder)
                    validation_local_path = local_validation_folder
                    print(f"[INFO] Found existing report.json in {local_validation_folder}; skipping download.")
                except FileNotFoundError:
                    validation_local_path = _download_validation_run_for_lofi(
                        lofi_run_local_path=path_name,
                        entity=entity,
                        validation_project=validation_project,
                        validation_prefix="Validation_",
                        local_validation_folder=local_validation_folder,
                        artifact_base_names=("report", "results_folder"),
                    )
                    report_path = _find_report_json(validation_local_path)
        else:
            validation_local_path = path_name
            report_path = _find_report_json(validation_local_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        num_fail_dupfree = int(_extract_failures_from_report(report))
        return num_tests_dupfree, num_fail_dupfree