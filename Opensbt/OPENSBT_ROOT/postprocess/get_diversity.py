import os
import glob
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from mycoverage.filter import match_critical_simouts
from postprocess.wandb import download_run_artifacts_relative
from postprocess.retreive_predict_simouts import retrieve_simouts_predict
from postprocess.get_figures_here import _download_validation_run_for_lofi
from mycoverage.compute_coverage import coverage_pipeline

AlgoKind = Literal["predict", "lofi", "hifi"]


# =============================================================================
# Helpers (fs + kind)
# =============================================================================
def _glob_sorted(pattern: str) -> List[str]:
    return sorted(glob.glob(pattern, recursive=True))


def infer_algo_kind(run_dir: str) -> Optional[AlgoKind]:
    p = run_dir.lower()
    if "hifi" in p:
        return "hifi"
    if "lofi" in p:
        return "lofi"
    if "predict" in p:
        return "predict"
    return None


def _find_report_json(root: str) -> str:
    direct = os.path.join(root, "report.json")
    if os.path.isfile(direct):
        return direct
    candidates = _glob_sorted(os.path.join(root, "**", "report.json"))
    if not candidates:
        raise FileNotFoundError(f"No report.json found under {root}")
    return candidates[0]


def _find_rerun_simout_dir(root: str) -> Optional[str]:
    direct = os.path.join(root, "rerun_hifi", "simout")
    if os.path.isdir(direct):
        return direct
    candidates = [p for p in _glob_sorted(os.path.join(root, "**", "rerun_hifi", "simout")) if os.path.isdir(p)]
    return candidates[0] if candidates else None


# =============================================================================
# Precision-cut matching for simout_S<values...>.json (LoFi validation -> rerun simouts)
# =============================================================================
_SIMOUT_PREFIX_RE = re.compile(r"^simout_S")


def cut_precision(x: float, precision: int) -> float:
    f = 10 ** precision
    return float(np.trunc(float(x) * f) / f)


def cut_vector(vec: List[float], precision: int) -> Tuple[float, ...]:
    return tuple(cut_precision(v, precision) for v in vec)


def parse_simout_stem_to_vector(stem: str, expected_len: int) -> Optional[List[float]]:
    stem = _SIMOUT_PREFIX_RE.sub("", stem, count=1)
    if stem.startswith("_"):
        stem = stem[1:]
    parts = stem.split("_")
    if len(parts) < expected_len:
        return None
    try:
        return [float(p) for p in parts[:expected_len]]
    except ValueError:
        return None


def build_simout_vector_index(simout_folder: str, precision: int, expected_len: int) -> Dict[Tuple[float, ...], str]:
    idx: Dict[Tuple[float, ...], str] = {}
    for fn in os.listdir(simout_folder):
        if not fn.endswith(".json"):
            continue
        stem = os.path.splitext(fn)[0]
        vec = parse_simout_stem_to_vector(stem, expected_len=expected_len)
        if vec is None:
            continue
        idx.setdefault(cut_vector(vec, precision), os.path.join(simout_folder, fn))
    return idx


def _critical_xs_from_report_json(report_json_path: str, expected_len: int) -> List[List[float]]:
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    xs: List[List[float]] = []
    for t in rep.get("tests", []):
        if t.get("cb", False) is True and isinstance(t.get("x", None), list):
            x = [float(v) for v in t["x"][:expected_len]]
            if len(x) == expected_len:
                xs.append(x)
    return xs


# =============================================================================
# CRITICAL simout retrieval per ALGO
# =============================================================================
def retrieve_critical_simouts_hifi(run_dir: str, key_cols: List[str]) -> List[str]:
    csv_path = os.path.join(run_dir, "all_critical_testcases.csv")
    simout_dir = os.path.join(run_dir, "simout")
    print("csv_path:", csv_path)
    print("simout_dir:", simout_dir)
    if not (os.path.exists(csv_path) and os.path.isdir(simout_dir)):
        csv_candidates = _glob_sorted(os.path.join(run_dir, "**", "all_critical_testcases.csv"))
        print("len(csv_candidates):", len(csv_candidates))
        simout_candidates = [p for p in _glob_sorted(os.path.join(run_dir, "**", "simout")) if os.path.isdir(p)]
        print("len(simout_candidates):", len(simout_candidates))
        if csv_candidates:
            csv_path = csv_candidates[0]
        if simout_candidates:
            simout_dir = simout_candidates[0]

    if not (os.path.exists(csv_path) and os.path.isdir(simout_dir)):
        return []

    return sorted(set(match_critical_simouts(csv_path, simout_dir, key_cols=key_cols)))


def retrieve_critical_simouts_predict(run_dir: str, key_cols: List[str]) -> List[str]:
    _all_simouts, critical = retrieve_simouts_predict(run_dir, key_cols=key_cols)
    return sorted(set(critical))


def retrieve_critical_simouts_lofi(
    run_dir: str,
    *,
    entity: str,
    validation_project: str,
    validation_prefix: str = "Validation_",
    precision: int = 4,
    expected_len: int = 11,
) -> List[str]:
    # 0) If report.json is already in/under the LoFi run_dir, skip download
    try:
        report_json = _find_report_json(run_dir)
        validation_dir = run_dir
    except FileNotFoundError:
        validation_dir = _download_validation_run_for_lofi(
            lofi_run_local_path=run_dir,
            entity=entity,
            validation_project=validation_project,
            validation_prefix=validation_prefix,
            local_validation_folder="lofi-validation",
            artifact_base_names=("validation",),
        )
        report_json = _find_report_json(str(validation_dir))

    # 1) critical tests from report.json (cb==true)
    critical_xs = _critical_xs_from_report_json(report_json, expected_len=expected_len)
    if not critical_xs:
        return []

    # 2) simout dir: prefer original run_dir rerun_hifi/simout, fallback to validation_dir
    simout_dir = _find_rerun_simout_dir(run_dir) or _find_rerun_simout_dir(str(validation_dir))
    if simout_dir is None:
        return []

    # 3) precision-cut match x -> simout file
    simout_index = build_simout_vector_index(simout_dir, precision=precision, expected_len=expected_len)

    matched: List[str] = []
    for x in critical_xs:
        p = simout_index.get(cut_vector(x, precision))
        if p:
            matched.append(p)

    return sorted(set(matched))


# =============================================================================
# Report driver: 3 approaches = 3 algos; call YOUR coverage_pipeline
# =============================================================================
def diversity_report_from_simout(
    wb_project_path: str,
    *,
    entity: str,
    validation_project: str,
    validation_prefix: str = "Validation_",
    save_csv: str = "diversity_summary.csv",
    experiments_folder: str = "results_wandb",
    run_filter=None,
    one_per_name: bool = False,
    output_dir: str = "diversity",
    eps: float = 1.5,
    min_samples: int = 2,
    precision: int = 5,
) -> pd.DataFrame:
    """
    We treat the 3 approaches (for coverage/entropy) as the 3 ALGORITHMS:
      approach 0: predict critical simouts (union across runs)
      approach 1: lofi    critical simouts (union across runs)
      approach 2: hifi    critical simouts (union across runs)

    Then we call YOUR existing `coverage_pipeline(folder_list, critical_simout_paths, ...)`.
    """
    artifact_paths = download_run_artifacts_relative(
        wb_project_path=wb_project_path,
        local_root=experiments_folder,
        filter_runs=run_filter,
        one_per_name=one_per_name,
        artifact_base_names=("results_folder", "report", "rerun_folder"),
    )

    os.makedirs(output_dir, exist_ok=True)

    crit_predict: List[str] = []
    crit_lofi: List[str] = []
    crit_hifi: List[str] = []
    
    project_name = wb_project_path.split("/")[-1]
    if project_name == "autoware_final":
        key_cols = ["PedSpeed", "EgoSpeed", "PedDist"]
    else:
        key_cols = [
                    "goal_center_x", "goal_center_y", "goal_velocity_min", "goal_velocity_max",
                    "num_npc", "lane_count", "npc_x0", "npc_v_max", "npc_seed", "cutin_tx", "cutin_delay",
                ]


    # artifact_paths keys are typically "predict"/"lofi"/"hifi" in your download structure,
    # but we still infer from run_dir as a fallback.
    print("artifact_paths keys:", artifact_paths)
    for key, run_dirs in artifact_paths.items():
        for run_dir in run_dirs:
            algo_kind = infer_algo_kind(run_dir) or key  # fallback to artifact_paths key

            if algo_kind == "predict":
                crit_predict.extend(retrieve_critical_simouts_predict(run_dir, key_cols=key_cols))
                print("found predict crit simouts: len(crit_predict) =", len(crit_predict))
            elif algo_kind == "lofi":
                crit_lofi.extend(
                    retrieve_critical_simouts_lofi(
                        run_dir,
                        entity=entity,
                        validation_project=validation_project,
                        validation_prefix=validation_prefix,
                        precision=precision,
                        expected_len=11 if project_name == "planer_final" else 3,
                    )
                )
                print("found lofi crit simouts: len(crit_lofi) =", len(crit_lofi))
            elif algo_kind == "hifi":
                crit_hifi.extend(retrieve_critical_simouts_hifi(run_dir, key_cols=key_cols))
                print("found hifi crit simouts: len(crit_hifi) =", len(crit_hifi))
                
    crit_predict = sorted(set(crit_predict))
    crit_lofi = sorted(set(crit_lofi))
    crit_hifi = sorted(set(crit_hifi))

    sources: List[List[str]] = [crit_predict, crit_lofi, crit_hifi]
    critical_union = sorted(set(crit_predict + crit_lofi + crit_hifi))
    if not critical_union:
        raise RuntimeError("No critical simouts found across predict/lofi/hifi.")

    # ---- call your existing pipeline (must be in scope/imported in your script) ----
    vectors_std, labels, simout_mapping, cluster_to_scenarios, simout_paths, coverage_dict, entropy_dict = coverage_pipeline(
        folder_list=sources,
        critical_simout_paths=critical_union,
        eps=eps,
        min_samples=min_samples,
        save_folder=os.path.join(output_dir, "plots"),
        project_name=project_name,
    )

    rows: List[Dict[str, Any]] = []
    for idx, algo_name in enumerate(["predict", "lofi", "hifi"]):
        rows.append(
            {
                "Algorithm": algo_name,
                "n_simouts": int(np.sum(simout_mapping == idx)),
                "n_clusters_ex_noise": int(len(set(labels) - {-1})),
                "coverage": float(coverage_dict.get(idx, 0.0)),
                "entropy": float(entropy_dict.get(idx, 0.0)),
            }
        )

    df = pd.DataFrame(rows).sort_values(["Algorithm"]).reset_index(drop=True)

    out_csv = os.path.join(output_dir, save_csv)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df