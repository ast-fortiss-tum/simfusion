import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

import csv
import os

import copy
from pathlib import Path
import numpy as np
from opensbt.problem.adas_problem import ADASProblem
from opensbt.simulation.simulator import SimulationOutput
from predictor.analyse_data import plot_design_space, plot_objective_space
from opensbt.visualization.visualizer import if_none_iterator

from opensbt.visualization.combined import read_pf_single
from opensbt.config import OUTPUT_PRECISION
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.utils.duplicates import duplicate_free
import dill
import sys
from typing import Tuple
import numpy as np
import math
from opensbt.utils import geometric
from simulations.simple_sim import *
from simulations.carla import *
from simulations.utils import *
import time
import argparse
from opensbt.visualization.visualizer import write_diversity
import dill
from opensbt.utils.wandb import wandb_log_metrics_df
import wandb
from collections import Counter
from datetime import datetime
import json
from opensbt.utils.encoder_utils import NumpyEncoder
from simulations.simulators import SIMULATOR_MAPPING


def move_ind_values(ind, run_lofi, direction=None):
    if run_lofi:
        ind.set("F_LOFI", ind.get("F"))
        ind.set("CB_LOFI", ind.get("CB"))
    else:
        ind.set("F_HIFI", ind.get("F"))
        ind.set("CB_HIFI", ind.get("CB"))
        
def reassign_critical_and_write(
    in_csv: str,
    out_csv: str = "critical_testscases_filtered.csv",
    th_obstacle: float = 2.0,
    th_goal: float = 0.0,
) -> pd.DataFrame:
    df = pd.read_csv(in_csv)

    # count "before" (handles missing column by treating as 0 criticals)
    if "Critical" in df.columns:
        n_critical_before = int((df["Critical"] == 1).sum())
    else:
        n_critical_before = 0

    col_obstacle = "Fitness_Min distance to obstacle"
    col_goal = "Fitness_Distance to goal"

    # Critical rule:
    # min_distance <= th_obstacle OR goal_distance > th_goal
    # (use <= if that's your intended rule; your snippet currently uses <)
    critical_mask = (df[col_obstacle] <= th_obstacle) | (df[col_goal] > th_goal)

    df = df.copy()
    df["Critical"] = critical_mask.astype(int)

    # count "after"
    n_critical_after = int((df["Critical"] == 1).sum())

    print(
        f"Critical count before: {n_critical_before}/{len(df)} | "
        f"after: {n_critical_after}/{len(df)}"
    )
    print("out_csv:", out_csv)
    # Always write all rows (no filtering)
    df.to_csv(out_csv, index=False)
    return df

def rerun_and_analyze(
    simulate_function,
    folder: str,
    sim_original: str,  # "lofi" or "hifi" - what the original results are
    sim_rerun: str,     # "lofi" or "hifi" - what to rerun with
    critical_only: bool,
    n_rerun: int = 1,
    only_if_no_hifi: bool = False,  # RENAMED
    ratio_sample: float = 1,    # how many tests to rerun
    filter_goal_threshold: str = None, # hack for cr case study (we need post filtering becaues of previously low goal threshold)
    filter_obstacle_distance: str = None,  
    seed = 123
):
    """
    Unified function for rerunning tests and analyzing results.

    Policy:
      - The "only_if_no_hifi" filter is ONLY applicable when sim_original == "lofi" and sim_rerun == "hifi".
        (Typical use: original LoFi results exist, and we want to run missing HiFi results.)

    Args:
        folder: Path to the folder containing test data
        sim_original: Original simulation type in the file ("lofi" or "hifi")
        sim_rerun: Simulation type for rerun ("lofi" or "hifi")
        critical_only: Whether to only rerun critical tests
        n_rerun: Number of times to rerun each test (1 = single rerun, >1 = flakiness analysis)
        only_if_no_hifi: If True (and only when sim_original=lofi & sim_rerun=hifi),
                         rerun only individuals with no existing HiFi execution (CB_HIFI is None).
    """
    run_lofi = (sim_rerun == "lofi")
    original_is_lofi = (sim_original == "lofi")
    is_flakiness_check = n_rerun >= 1
    
    print("SIMULATOR_MAPPING:", SIMULATOR_MAPPING)

    simulate_function = SIMULATOR_MAPPING[simulate_function]
    
    # Load problem configuration
    problem_path = os.path.join(folder, "backup/problem")
    
    if filter_goal_threshold is not None and filter_obstacle_distance is not None:
        print("Applying post-filtering with thresholds - obstacle:", filter_obstacle_distance, "goal:", filter_goal_threshold)
        filtered_csv_path = os.path.join(folder, "all_testcases_filtered.csv")
        reassign_critical_and_write(
            in_csv=os.path.join(folder, "all_testcases.csv"),
            out_csv=filtered_csv_path,
            th_obstacle=float(filter_obstacle_distance) if filter_obstacle_distance is not None else 2.0,
            th_goal=float(filter_goal_threshold) if filter_goal_threshold is not None else 0.0
        )
        test_cases_path = filtered_csv_path
    else:
        print("No post-filtering applied.")
        test_cases_path = os.path.join(folder, "all_testcases.csv")

    with open(problem_path, 'rb') as f:
        problem_read = dill.load(f)

    problem = ADASProblem(
        problem_name=f"Rerun_in_{sim_rerun.upper()}",
        scenario_path="./scenarios/CUT_IN_last.xml",
        xl=problem_read.xl,
        xu=problem_read.xu,
        simulation_variables=problem_read.simulation_variables,
        fitness_function=problem_read.fitness_function,
        critical_function=problem_read.critical_function,
        simulate_function=simulate_function,
        simulation_time=problem_read.simulation_time,
        sampling_time=problem_read.sampling_time
    )

    # Load test cases
    pop = read_pf_single(filename=test_cases_path, with_critical_column=True)
    pop_all = copy.deepcopy(duplicate_free(pop))

    # IMPORTANT: pop must be defined regardless of critical_only
    if critical_only:
        pop, _ = pop_all.divide_critical_non_critical()
    else:
        pop = pop_all
    
    n_candidates = len(pop)

    # Apply only-if-no-hifi policy ONLY when original is LoFi and we rerun in HiFi
    only_if_no_hifi_applies = bool(only_if_no_hifi and original_is_lofi and (sim_rerun == "hifi"))
    if only_if_no_hifi_applies:
        # pop = PopulationExtended(individuals=[
        #     ind for ind in pop
        # ])
        # If ratio_sample is an integer-like value, interpret as absolute count.
        if isinstance(ratio_sample, (int, np.integer)):
            n_sample = int(ratio_sample)
        elif isinstance(ratio_sample, float) and ratio_sample.is_integer():
            n_sample = int(ratio_sample)
        else:
            # Otherwise interpret as a ratio in (0, 1] (or allow >1 if you want to error).
            n_sample = math.ceil(float(ratio_sample) * n_candidates)

        # Clamp to valid range
        n_sample = max(0, min(n_sample, n_candidates))

        rng = np.random.default_rng(seed)
        idx = rng.choice(n_candidates, size=n_sample, replace=False)

        pop = PopulationExtended(individuals=[pop[i] for i in idx])
        print(f"Selected {len(pop)}/{n_candidates} samples.")
    else:
        n_sample = 0
        pop = PopulationExtended(individuals=[])
        
    save_folder = os.path.join(folder, f"rerun_{sim_rerun}/")
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    # Initialize metrics with default values
    valid_failure_rate = 0.0
    num_failures = 0
    time_extra = 0.0
    avg_valid_rate = None

    # Initialize report
    report = {
        "timestamp": datetime.now().isoformat(),
        "folder": folder,
        "sim_original": sim_original,
        "sim_rerun": sim_rerun,
        "n_reruns": n_rerun,
        "n_tests_rerun": len(pop),
        "n_tests_all": len(pop_all),
        "critical_only": critical_only,
        "only_if_no_hifi": only_if_no_hifi,
        "tests": []
    }
    n_tests_rerun = len(pop)
    if n_tests_rerun > 0:
        print(f"Doing rerun for {n_tests_rerun} tests.")

        # Store original values based on sim_original type (for the rerun subset)
        for ind in pop:
            if original_is_lofi:
                ind.set("F_LOFI", ind.get("F"))
                ind.set("CB_LOFI", ind.get("CB"))
            else:
                ind.set("F_HIFI", ind.get("F"))
                ind.set("CB_HIFI", ind.get("CB"))

        start_time = time.time()
        print(f"n_rerun set to: {n_rerun}")

        # Rerun logic
        for k, ind in enumerate(pop):
            print(f"Executing test {k+1}/{len(pop)}")
            if is_flakiness_check:
                fitness_list, cb_list, so_list = [], [], []
                for i in range(n_rerun):
                    print(f"Repeating {i+1}/{n_rerun} times")
                    evaluate_individuals(
                        Population(individuals=[ind]),
                        problem=problem,
                        backup_folder=save_folder
                    )
                    fitness_list.append(ind.get("F"))
                    cb_list.append(ind.get("CB"))
                    so_list.append(ind.get("SO"))

                majority_cb = Counter(cb_list).most_common(1)[0][0]
                chosen_index = next(i for i, cb in enumerate(cb_list) if cb == majority_cb)
                chosen_fitness, chosen_so = fitness_list[chosen_index], so_list[chosen_index]

                ind.set("F", chosen_fitness)
                ind.set("SO", chosen_so)
                ind.set("CB", majority_cb)

                valid_rate = sum(cb == 1 for cb in cb_list) / n_rerun
                report["tests"].append({
                    "x": ind.get("X"),
                    "fitness_all": fitness_list,
                    "cb_list": cb_list,
                    "fitness": chosen_fitness,
                    "cb": majority_cb,
                    "valid_rate": valid_rate,
                    "so_list": so_list,
                    "so": chosen_so
                })
            else:
                evaluate_individuals(
                    Population(individuals=[ind]),
                    problem=problem,
                    backup_folder=save_folder
                )
                report["tests"].append({
                    "x": ind.get("X"),
                    "fitness": ind.get("F"),
                    "cb": ind.get("CB"),
                    "so": ind.get("SO")
                })

        try:
            # Calculate statistics
            if is_flakiness_check and len(report["tests"]) > 0:
                avg_valid_rate = sum(test["valid_rate"] for test in report["tests"]) / len(report["tests"])
                report["avg_valid_rate"] = avg_valid_rate

            # "valid_*" metrics are based on this rerun batch (normalized by all tests)
            num_failures = sum(1 for test in report["tests"] if test["cb"] == 1)
            valid_failure_rate = num_failures / len(pop_all) if len(pop_all) > 0 else 0.0
            report["valid_failure_rate"] = valid_failure_rate
            report["valid_failures"] = num_failures

            time_extra = time.time() - start_time
            report["valid_execution_time"] = time_extra

            # Store rerun results based on sim_rerun type
            for ind in pop:
                if run_lofi:
                    ind.set("F_LOFI", ind.get("F"))
                    ind.set("CB_LOFI", ind.get("CB"))
                else:
                    ind.set("F_HIFI", ind.get("F"))
                    ind.set("CB_HIFI", ind.get("CB"))

            path_results = write_all_individuals_lohifi(
                problem=problem,
                all_individuals=pop,
                save_folder=save_folder,
                file_name="all_testcases.csv"
            )

            print("Rerun results stored in:", path_results)

            analyse_agree_disagree(
                file=path_results,
                save_folder=save_folder,
                file_name="agree_summary.csv",
                num_exec_extra=len(pop),
                extra_execution_time=time_extra
            )

            write_simulation_output(pop, save_folder=save_folder)

            # Generate plots
            # save_folder_plots = os.path.join(save_folder, "plots/")
            # Path(save_folder_plots).mkdir(exist_ok=True, parents=True)
            # plot_design_space(path_results, save_folder_plots)
            # plot_objective_space(path_results, save_folder_plots)
        except Exception as e:
            print("Exception happened:", e)
            
    else:
        print("No individuals for rerun.")
        report["valid_failure_rate"] = 0.0
        report["valid_failures"] = 0
        report["valid_execution_time"] = 0.0
        if is_flakiness_check:
            report["avg_valid_rate"] = None

    # ALWAYS save report (initial write)
    report_path = os.path.join(save_folder, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"Report saved at {report_path}")

    # Overall analysis - initialize all individuals with original values, then overlay rerun results
    for ind in pop_all:
        # set old values
        if original_is_lofi:
            ind.set("F_LOFI", ind.get("F"))
            ind.set("CB_LOFI", ind.get("CB"))
            ind.set("F_HIFI", [None] * problem.n_obj)
            ind.set("CB_HIFI", None)
        else:
            ind.set("F_HIFI", ind.get("F"))
            ind.set("CB_HIFI", ind.get("CB"))
            ind.set("F_LOFI", [None] * problem.n_obj)
            ind.set("CB_LOFI", None)

        # overlay rerun results where available
        for ind_rerun in pop:
            if np.array_equal(ind_rerun.get("X"), ind.get("X")):
                if run_lofi:
                    ind.set("F_LOFI", ind_rerun.get("F"))
                    ind.set("CB_LOFI", ind_rerun.get("CB"))
                else:
                    ind.set("F_HIFI", ind_rerun.get("F"))
                    ind.set("CB_HIFI", ind_rerun.get("CB"))
                break

    # Totals for the rerun sim across ALL individuals (after merge)
    total_cb_key = "CB_LOFI" if run_lofi else "CB_HIFI"
    total_rerun_failures = sum(
        1 for ind in pop_all
        if ind.get(total_cb_key) == 1
    )
    p = (n_candidates / n_sample) if n_sample > 0 else 0.0
    
    total_rerun_fail_rate = (total_rerun_failures / len(pop_all)) if len(pop_all) > 0 else 0.0
    
    known_failures = total_rerun_failures - num_failures

    estimated_failures = known_failures + min(num_failures * p, n_candidates) if p > 0 else float(total_rerun_failures)
    
    report["total_failures"] = total_rerun_failures
    report["total_fail_rate"] = total_rerun_fail_rate
    report["estimated_failures"] = estimated_failures

    path_results_combined = write_all_individuals_lohifi(
        problem=problem,
        all_individuals=pop_all,
        save_folder=save_folder,
        file_name="all_testcases_combined.csv"
    )

    print("Combined results stored in:", path_results_combined)

    agree_disagree_df = analyse_agree_disagree(
        file=path_results_combined,
        save_folder=save_folder,
        file_name="agree_summary_combined.csv",
        num_exec_extra=len(pop_all),
        extra_execution_time=time_extra
    )

    if wandb.run:
        print("Logging to wandb rerun.")
        wandb.log({
            "valid_failure_rate" : valid_failure_rate,
            "valid_failures" : num_failures,
            "avg_valid_rate": avg_valid_rate,
            "total_failures": total_rerun_failures,
            "total_fail_rate": total_rerun_fail_rate,
            "estim_failures": estimated_failures
        })
        wandb_log_metrics_df(agree_disagree_df)
    else:
        print("Wandb is not initialized.")

    write_diversity(pop_all, save_folder=save_folder, file_name="diversity_combined.csv")
    analyse_overall(path_results_combined, save_folder=save_folder)

    # save_folder_plots = os.path.join(save_folder, "plots_combined/")
    # Path(save_folder_plots).mkdir(exist_ok=True, parents=True)
    # plot_design_space(path_results_combined, save_folder_plots)
    # plot_objective_space(path_results_combined, save_folder_plots)

    # IMPORTANT: rewrite report so totals are persisted even when no rerun happened
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"Report updated at {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoFi or HiFi simulation rerun and analyze results.")
    parser.add_argument("--simulate_function", type=str, required=True, help="Simulator to use.")
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument("--sim_original", choices=["lofi", "hifi"], default="lofi",
                        help="Original simulation type in the file (lofi or hifi)")
    parser.add_argument("--sim_rerun", choices=["lofi", "hifi"], default="hifi",
                        help="Simulation type for rerun (lofi or hifi)")
    parser.add_argument("--critical_only", action="store_true", default=False,
                        help="If set, only critical results will be rerun")
    parser.add_argument("--n_rerun", type=int, default=1,
                        help="Number of reruns (1 = single rerun, >1 = flakiness analysis)")
    parser.add_argument("--ratio_sample", type=float, default=1,
                        help="Ratio of samples to rerun [0,1]")
    parser.add_argument("--filter_goal_threshold", type=float, default=None,
                        help="Post-filtering threshold for goal distance (optional)")
    parser.add_argument("--filter_obstacle_distance", type=float, default=None,
                        help="Post-filtering threshold for obstacle distance (optional)")   

    parser.add_argument(
        "--only_if_no_hifi",
        action="store_true",
        default=True,
        help="If set (and only when sim_original=lofi & sim_rerun=hifi), rerun only individuals with no existing HiFi execution (CB_HIFI is None).",
    )

    args = parser.parse_args()
    return args.folder, args.sim_original, args.sim_rerun, args.critical_only, args.n_rerun, args.ratio_sample, args.only_if_no_hifi, args.filter_goal_threshold, args.filter_obstacle_distance 


if __name__ == "__main__":
    folder, sim_original, sim_rerun, critical_only, n_rerun, ratio_sample, only_if_no_hifi, filter_goal_threshold, filter_obstacle_distance = parse_args()
    rerun_and_analyze(
        folder,
        sim_original,
        sim_rerun,
        critical_only,
        n_rerun,
        only_if_no_hifi=only_if_no_hifi,
        filter_goal_threshold=filter_goal_threshold,
        filter_obstacle_distance=filter_obstacle_distance,
        ratio_sample=ratio_sample
    )