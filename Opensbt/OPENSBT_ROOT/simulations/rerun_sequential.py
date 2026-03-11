import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from predictor.analyse_data import plot_design_space, plot_objective_space

import csv
import os

import copy
from pathlib import Path
import numpy as np

from opensbt.config import OUTPUT_PRECISION
from opensbt.problem.adas_problem import ADASProblem

from opensbt.utils.evaluation import evaluate_individuals
from opensbt.utils.duplicates import duplicate_free
from opensbt.visualization.combined import read_pf_single

import dill
import sys
from typing import Tuple
import numpy as np
import math
from opensbt.utils import geometric
from simulations.utils import *
import logging as log
import argparse
from opensbt.visualization.visualizer import write_diversity
import dill
from opensbt.utils.wandb import wandb_log_metrics_df
import wandb
from collections import Counter
from datetime import datetime
import json
from opensbt.utils.encoder_utils import NumpyEncoder
import time
from simulations.simulators import SIMULATOR_MAPPING

def rerun_and_analyze_sequential(
    simulate_function,
    folder: str,
    critical_only: bool = False,
    n_rerun: int = 1,
    only_if_no_hifi: bool = False):
    """
    Unified function for rerunning HiFi tests and analyzing results.

    Args:
        folder: Path to the folder containing test data
        critical_only: Whether to only rerun critical tests
        n_rerun: Number of times to rerun each test (1 = single rerun, >1 = flakiness analysis)
        only_if_no_hifi: If True, rerun ONLY individuals that don't already have HiFi results (CB_HIFI is None).
    """
    sim_rerun = "hifi"
    is_flakiness_check = n_rerun >= 1
    
    simulate_function = SIMULATOR_MAPPING[simulate_function]

    problem_path = os.path.join(folder,"backup/problem")
    test_cases_path = os.path.join(folder,"all_testcases.csv")

    with open(problem_path, 'rb') as f:
        problem_read = dill.load(f)

    problem = ADASProblem(
        problem_name=f"Rerun_in_{sim_rerun.upper()}",
        scenario_path=problem_read.scenario_path,
        xl=problem_read.xl,
        xu=problem_read.xu,
        simulation_variables=problem_read.simulation_variables,
        fitness_function=problem_read.fitness_function,
        critical_function=problem_read.critical_function,
        simulate_function=simulate_function,
        simulation_time=problem_read.simulation_time,
        sampling_time=problem_read.sampling_time
    )

    print("test_cases_path", test_cases_path)

    # read a file where lofi and hifi values are stored, and take over the hifi values from
    # there (if available) to the population
    pop_all: PopulationExtended = read_pf_single_sequential(filename=test_cases_path,
                                                        problem=problem,
                                                        with_critical_column=True,
                                                        skip_lofi_nan=False)
    print("Len pop read no dup free: ", len(pop_all))

    pop_all = copy.deepcopy(duplicate_free(pop_all))

    print("Len pop read: ", len(pop_all))

    if critical_only:
        pop, _ = pop_all.divide_critical_non_critical()
    else:
        pop = pop_all  # ensure pop is defined when critical_only is False

    # NEW: rerun only if no HiFi execution exists (CB_HIFI is None)
    # If critical_only is set, this becomes: only critical where no HiFi is set will be run.
    if only_if_no_hifi:
        pop = PopulationExtended(individuals=[
            ind for ind in pop
            if ind.get("CB_HIFI") is None
        ])

    # filtered out
    print("len pop to rerun: ", len(pop))

    save_folder = os.path.join(folder, f"rerun_{sim_rerun}/")
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    # Initialize metrics with default values
    valid_failure_rate = 0.0
    num_failures = 0
    time_extra = 0.0
    avg_valid_rate = None

    # Initialize report BEFORE the if statement
    report = {
        "timestamp": datetime.now().isoformat(),
        "folder": folder,
        "n_reruns": n_rerun,
        "n_tests_rerun": len(pop),
        "n_tests_all": len(pop_all),
        "only_if_no_hifi": only_if_no_hifi,
        "critical_only": critical_only,
        "tests": []
    }

    if len(pop) > 0:
        start_time = time.time()

        print(f"Doing rerun for {len(pop)} tests.")
        print(f"n_rerun set to: {n_rerun}")

        # Rerun logic
        for k, ind in enumerate(pop):
            print(f"Executing test {k+1}/{len(pop)}")
            if is_flakiness_check:
                # Flakiness check - multiple reruns
                fitness_list, cb_list, so_list = [], [], []
                for i in range(n_rerun):
                    print(f"Repeating {i+1}/{n_rerun} times")
                    evaluate_individuals(Population(individuals=[ind]), problem=problem, backup_folder=save_folder)
                    fitness_list.append(ind.get("F"))
                    cb_list.append(ind.get("CB"))
                    so_list.append(ind.get("SO"))

                majority_cb = Counter(cb_list).most_common(1)[0][0]
                chosen_index = next(i for i, cb in enumerate(cb_list) if cb == majority_cb)
                chosen_fitness, chosen_so = fitness_list[chosen_index], so_list[chosen_index]

                ind.set("F", chosen_fitness)
                ind.set("SO", chosen_so)
                ind.set("CB", majority_cb)
                ind.set("F_HIFI", chosen_fitness)
                ind.set("CB_HIFI", majority_cb)

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
                # Single rerun
                evaluate_individuals(Population(individuals=[ind]), problem=problem, backup_folder=save_folder)
                ind.set("F_HIFI", ind.get("F"))
                ind.set("CB_HIFI", ind.get("CB"))
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

            # "valid_*" metrics are based on *this rerun batch* (but normalized by total tests, as before)
            num_failures = sum(1 for test in report["tests"] if test["cb"] == 1)
            valid_failure_rate = num_failures / len(pop_all) if len(pop_all) > 0 else 0
            report["valid_failure_rate"] = valid_failure_rate
            report["valid_failures"] = num_failures

            time_extra = time.time() - start_time
            report["valid_execution_time"] = time_extra

            path_results = write_all_individuals_lohifi(problem=problem,
                                                        all_individuals=pop_all,
                                                        save_folder=save_folder,
                                                        file_name="all_testcases.csv")

            print("Rerun results stored in:", path_results)

            analyse_agree_disagree(file=path_results,
                                save_folder=save_folder,
                                file_name="agree_summary.csv",
                                num_exec_extra=len(pop),
                                extra_execution_time=time_extra)

            write_simulation_output(pop, save_folder=save_folder)

            ######## Plots

            # save_folder_plots = os.path.join(save_folder, "plots/")
            # Path(save_folder_plots).mkdir(exist_ok=True, parents=True)

            # Generate and save design space plots
            # plot_design_space(path_results, save_folder_plots)

            # Generate and save objective space plots
            # plot_objective_space(path_results, save_folder_plots)
        except Exception as e:
            print("Exception happened:", e)
            
    else:
        print("No tests to execute.")
        # Set default values for report when no tests
        report["valid_failure_rate"] = 0.0
        report["valid_failures"] = 0
        report["valid_execution_time"] = 0.0
        if is_flakiness_check:
            report["avg_valid_rate"] = None

    # ALWAYS save the report, even if no tests were run
    report_path = os.path.join(save_folder, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"Report saved at {report_path}")

    # Merge rerun results back into pop_all
    for ind in pop_all:
        # take over the values from the rerun
        for ind_rerun in pop:
            if np.array_equal(ind_rerun.get("X"), ind.get("X")):
                ind.set("F_HIFI",  ind_rerun.get("F_HIFI") )
                ind.set("CB_HIFI", ind_rerun.get("CB_HIFI"))
                break

    # NEW: total HiFi stats across ALL individuals after merge
    total_hifi_failures = sum(
        1 for ind in pop_all
        if ind.get("CB_HIFI") == 1
    )
    total_hifi_fail_rate = (total_hifi_failures / len(pop_all)) if len(pop_all) > 0 else 0.0
    report["total_failures"] = total_hifi_failures
    report["total_fail_rate"] = total_hifi_fail_rate

    path_results_combined = write_all_individuals_lohifi(problem=problem,
                                                all_individuals=pop_all,
                                                save_folder=save_folder,
                                                file_name="all_testcases_combined.csv")

    print("Combined results stored in:", path_results_combined)

    agree_disagree_df = analyse_agree_disagree(file=path_results_combined,
                        save_folder=save_folder,
                        file_name="agree_summary_combined.csv",
                        num_exec_extra = len(pop_all),
                        extra_execution_time=time_extra)

    if wandb.run:
        print("Logging to wandb rerun.")
        wandb.log({
            "valid_failure_rate": valid_failure_rate,
            "valid_failures": num_failures,
            "avg_valid_rate": avg_valid_rate,
            "total_failures": total_hifi_failures,
            "total_fail_rate": total_hifi_fail_rate,
        })
        wandb_log_metrics_df(agree_disagree_df)
    else:
        print("Wandb is not initialized.")

    # Plots
    write_diversity(pop_all, save_folder, file_name="diversity_combined.csv")
    analyse_overall(path_results_combined, save_folder=save_folder)

    # save_folder_plots_combined = os.path.join(save_folder, "plots_combined/")
    # Path(save_folder_plots_combined).mkdir(exist_ok=True, parents=True)

    # Generate and save design space plots
    # plot_design_space(path_results_combined, save_folder_plots_combined)

    # Generate and save objective space plots
    # plot_objective_space(path_results_combined, save_folder_plots_combined)

    # IMPORTANT: re-save report so the new total_hifi_* fields are actually written to disk
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4, cls=NumpyEncoder)
    print(f"Report updated at {report_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HiFi simulation rerun and analyze results.")
    parser.add_argument("--simulate_function", type=str, required=True, help="Simulator to use.")
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument("--critical_only", action="store_true", default=False, help="If set, only critical results will be processed")
    parser.add_argument("--n_rerun", type=int, default=2, help="Number of reruns for flakiness check")

    parser.add_argument(
        "--only_if_no_hifi",
        action="store_true",
        default=True,
        help="If set, rerun only individuals with no existing HiFi execution (CB_HIFI is None).",
    )

    args = parser.parse_args()
    return args.simulate_function, args.folder, args.critical_only, args.n_rerun, args.only_if_no_hifi


if __name__ == "__main__":
    simulate_fnc, folder, critical_only, n_rerun, only_if_no_hifi = parse_args()
    rerun_and_analyze_sequential(simulate_fnc, folder, critical_only, n_rerun=n_rerun, only_if_no_hifi=only_if_no_hifi)