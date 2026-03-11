import os
import glob
import json
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


KEY_COLS_DEFAULT = [
    "goal_center_x", "goal_center_y", "goal_velocity_min", "goal_velocity_max",
    "num_npc", "lane_count", "npc_x0", "npc_v_max", "npc_seed", "cutin_tx", "cutin_delay",
]


# =============================================================================
# Precision-cut matching (truncate, no rounding) + simout filename parsing
# =============================================================================
_SIMOUT_PREFIX_RE = re.compile(r"^simout_S")


def cut_precision(x: float, precision: int) -> float:
    """
    Truncate to `precision` decimals (no rounding).
    Example precision=2: 1.239 -> 1.23, -1.239 -> -1.23
    """
    f = 10 ** precision
    return float(np.trunc(float(x) * f) / f)


def cut_vector(vec: List[float], precision: int) -> Tuple[float, ...]:
    return tuple(cut_precision(v, precision) for v in vec)


def parse_simout_stem_to_vector(stem: str, expected_len: int) -> Optional[List[float]]:
    """
    stem example:
      simout_S157.896245_0.017177_10.707405_..._0.104958

    Returns list[float] length=expected_len, or None if unexpected.
    """
    stem = _SIMOUT_PREFIX_RE.sub("", stem, count=1)
    if stem.startswith("_"):
        stem = stem[1:]

    parts = stem.split("_")
    if len(parts) < expected_len:
        return None

    out: List[float] = []
    try:
        for p in parts[:expected_len]:
            out.append(float(p))
    except ValueError:
        return None
    return out


def build_simout_vector_index(simout_folder: str, precision: int, expected_len: int) -> Dict[Tuple[float, ...], str]:
    """
    Map: cut_vector(tuple) -> filepath
    Note: if multiple simouts collide after truncation, the first one wins.
    """
    idx: Dict[Tuple[float, ...], str] = {}
    for fn in os.listdir(simout_folder):
        if not fn.endswith(".json"):
            continue
        stem = os.path.splitext(fn)[0]
        vec = parse_simout_stem_to_vector(stem, expected_len=expected_len)
        if vec is None:
            continue
        key = cut_vector(vec, precision)
        idx.setdefault(key, os.path.join(simout_folder, fn))
    return idx


# =============================================================================
# Filesystem helpers
# =============================================================================
def _find_first_existing_dir(run_dir: str, dirname: str) -> Optional[str]:
    direct = os.path.join(run_dir, dirname)
    if os.path.isdir(direct):
        return direct
    candidates = [p for p in glob.glob(os.path.join(run_dir, "**", dirname), recursive=True) if os.path.isdir(p)]
    return sorted(candidates)[0] if candidates else None


def _find_first_existing_file(run_dir: str, filename: str) -> Optional[str]:
    direct = os.path.join(run_dir, filename)
    if os.path.isfile(direct):
        return direct
    candidates = glob.glob(os.path.join(run_dir, "**", filename), recursive=True)
    return sorted(candidates)[0] if candidates else None


def _find_preferred_report_json(run_dir: str) -> Optional[str]:
    """
    report.json can be either:
      - rerun_hifi/report.json  (preferred for rerun matching)
      - report/report.json      (fallback)

    If multiple exist, we pick in this order:
      1) any **/rerun_hifi/report.json
      2) any **/report/report.json
      3) any **/report.json
    """
    candidates_rerun = glob.glob(os.path.join(run_dir, "**", "rerun_hifi", "report.json"), recursive=True)
    if candidates_rerun:
        return sorted(candidates_rerun)[0]

    candidates_report_folder = glob.glob(os.path.join(run_dir, "**", "report", "report.json"), recursive=True)
    if candidates_report_folder:
        return sorted(candidates_report_folder)[0]

    candidates_any = glob.glob(os.path.join(run_dir, "**", "report.json"), recursive=True)
    return sorted(candidates_any)[0] if candidates_any else None


# =============================================================================
# Critical extraction
# =============================================================================
def _critical_values_from_all_testcases(csv_path: str, key_cols: List[str]) -> List[List[float]]:
    """
    Extract variable vectors from rows in all_testcases.csv, keeping only rows that are
    critical according to Critical_HiFi.

    - Only keep rows where Critical_HiFi == 1
    - Critical_HiFi may be NaN; treat NaN as 0
    - Adds backward-compatible column "Critical"
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    if "Critical_HiFi" in df.columns:
        crit = pd.to_numeric(df["Critical_HiFi"].fillna(0), errors="coerce").fillna(0).astype(int)
        df["Critical"] = crit
        df = df[df["Critical"] == 1].copy()

    return [df.loc[i, key_cols].astype(float).tolist() for i in df.index]


def _critical_values_from_rerun_report(report_json_path: str, expected_len: int) -> List[List[float]]:
    with open(report_json_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    critical_xs: List[List[float]] = []
    for t in rep.get("tests", []):
        if t.get("cb", False) is True:
            x = t.get("x", None)
            if isinstance(x, list) and len(x) >= expected_len:
                critical_xs.append([float(v) for v in x[:expected_len]])
    return critical_xs


# =============================================================================
# Updated main retrieval (predict)
# =============================================================================
def retrieve_simouts_predict(
    run_dir: str,
    *,
    precision: int = 4,
    key_cols: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Predict simout retrieval:

    Step A) From all_testcases.csv select only Critical_HiFi==1 rows and extract x.
    Step B) Match those x's to files in simout_hifi/ using precision CUT matching on filename vectors.
    Step C) Add rerun critical simouts:
            - open rerun_hifi/report.json OR report/report.json (whichever exists; rerun preferred)
            - take tests where cb==true (x)
            - match to rerun_hifi/simout/*.json using the same precision CUT matching
    """
    if key_cols is None:
        key_cols = KEY_COLS_DEFAULT
    expected_len = len(key_cols)

    all_testcases_csv = _find_first_existing_file(run_dir, "all_testcases.csv")
    simout_hifi_dir = _find_first_existing_dir(run_dir, "simout_hifi")

    rerun_hifi_dir = _find_first_existing_dir(run_dir, "rerun_hifi")
    rerun_simout_dir = os.path.join(rerun_hifi_dir, "simout") if rerun_hifi_dir else None

    report_json_path = _find_preferred_report_json(run_dir)

    critical_paths: List[str] = []

    # =========================
    # Step A + B: critical from all_testcases.csv -> simout_hifi
    # =========================
    if all_testcases_csv and simout_hifi_dir and os.path.isdir(simout_hifi_dir):
        critical_rows_x = _critical_values_from_all_testcases(all_testcases_csv, key_cols=key_cols)
        print("len(critical_rows_x):", len(critical_rows_x))

        simout_index = build_simout_vector_index(simout_hifi_dir, precision=precision, expected_len=expected_len)

        missing = 0
        for x in critical_rows_x:
            key = cut_vector(x[:expected_len], precision)
            p = simout_index.get(key)
            if p is None:
                missing += 1
                continue
            critical_paths.append(p)

        if missing:
            print(f"[WARNING] Missing {missing} simouts in simout_hifi for critical rows from {all_testcases_csv}")
        print(f"Found {len(critical_paths)} critical simouts from all_testcases.csv in simout_hifi")

    # =========================
    # Step C: add rerun critical simouts (cb==true) -> rerun_hifi/simout
    # =========================
    if (
        report_json_path
        and rerun_simout_dir
        and os.path.isfile(report_json_path)
        and os.path.isdir(rerun_simout_dir)
    ):
        rerun_critical_xs = _critical_values_from_rerun_report(report_json_path, expected_len=expected_len)
        rerun_index = build_simout_vector_index(rerun_simout_dir, precision=precision, expected_len=expected_len)

        print(f"report_json_path: {report_json_path}")
        print(f"Found {len(rerun_index)} simouts in rerun_hifi/simout index")
        print("len(rerun_critical_xs):", len(rerun_critical_xs))

        missing_rerun = 0
        for x in rerun_critical_xs:
            key = cut_vector(x[:expected_len], precision)
            p = rerun_index.get(key)
            if p is None:
                missing_rerun += 1
                continue
            critical_paths.append(p)

        if missing_rerun:
            print(f"[WARNING] Missing {missing_rerun} simouts in rerun_hifi/simout for cb==true tests from {report_json_path}")

    critical_paths = sorted(set(critical_paths))
    all_simout_paths = critical_paths[:]  # by your definition for predict
    return all_simout_paths, critical_paths