import pymoo

from opensbt.model_ga.individual import IndividualSimulated

pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended

pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult

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
from predictor.analyse_data_cr_bng import plot_design_space, plot_objective_space
from opensbt.visualization.visualizer import if_none_iterator

from opensbt.visualization.combined import read_pf_single
from simulations.commonroad.commonroad_simulation import CommonRoadSimulator
from simulations.beamng.beamng_simulation import BeamNGSimulator
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
from opensbt.utils.wandb import wandb_log_metrics_df
import wandb
from collections import Counter
from datetime import datetime
import json
from opensbt.utils.encoder_utils import NumpyEncoder

def move_ind_values(ind, run_lofi, direction=None):
    if run_lofi:
        ind.set("F_LOFI", ind.get("F"))
        ind.set("CB_LOFI", ind.get("CB"))
    else:
        ind.set("F_HIFI", ind.get("F"))
        ind.set("CB_HIFI", ind.get("CB"))


def rerun_and_analyze(
    folder: str, 
    sim_original: str,  # NEW: "lofi" or "hifi" - what the original results are
    sim_rerun: str,     # "lofi" or "hifi" - what to rerun with
    critical_only: bool,
    n_rerun: int = 1
):
    run_lofi = (sim_rerun == "lofi")
    original_is_lofi = (sim_original == "lofi")
    is_flakiness_check = n_rerun >= 1

    problem_path = os.path.join(folder, "backup/problem")
    test_cases_path = os.path.join(folder, "all_testcases.csv")

    with open(problem_path, "rb") as f:
        problem_read = dill.load(f)

    simulate_function = CommonRoadSimulator.simulate if run_lofi else BeamNGSimulator.simulate

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
        sampling_time=problem_read.sampling_time,
    )

    pop = read_pf_single(filename=test_cases_path, with_critical_column=True)
    pop_all = copy.deepcopy(duplicate_free(pop))

    if critical_only:
        pop, _ = pop_all.divide_critical_non_critical()

    save_folder = os.path.join(folder, f"rerun_{sim_rerun}/")
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    valid_failure_rate = 0.0
    num_failures = 0
    time_extra = 0.0
    avg_valid_rate = None

    # Initialize report - MOVED BEFORE the if statement
    report = {
        "timestamp": datetime.now().isoformat(),
        "folder": folder,
        "sim_original": sim_original,
        "sim_rerun": sim_rerun,
        "n_reruns": n_rerun,
        "n_tests_rerun": len(pop),
        "n_tests_all": len(pop_all),
        "tests": []
    }

    if len(pop) > 0:
        print(f"Doing rerun for {len(pop)} tests.")

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
                report["tests"].append({
                    "x": ind.get("X"),
                    "fitness": ind.get("F"),
                    "cb": ind.get("CB"),
                    "so": ind.get("SO")
                })

        # Calculate statistics
        if is_flakiness_check and len(report["tests"]) > 0:
            avg_valid_rate = sum(test["valid_rate"] for test in report["tests"]) / len(report["tests"])
            report["avg_valid_rate"] = avg_valid_rate    

        # Calculate failure rate (tests with cb == 1 divided by total tests)
        num_failures = sum(1 for test in report["tests"] if test["cb"] == 1)
        valid_failure_rate = num_failures / len(pop_all) if len(pop_all) > 0 else 0
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
            file_name="all_testcases.csv",
        )

        print("Rerun results stored in:", path_results)

        analyse_agree_disagree(
            file=path_results,
            save_folder=save_folder,
            file_name="agree_summary.csv",
            num_exec_extra=len(pop),
            extra_execution_time=time_extra,
        )

        write_simulation_output(pop, save_folder=save_folder)

        ######## Plots
        save_folder_plots = os.path.join(save_folder, "plots/")
        Path(save_folder_plots).mkdir(exist_ok=True, parents=True)
        # Generate and save design space plots
        plot_design_space(path_results, save_folder_plots)
        # Generate and save objective space plots
        plot_objective_space(path_results, save_folder_plots)
    else:
        print("No individuals for rerun.")
        # Still set report values for consistency
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

    # Overall analysis - initialize all individuals with original values
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
       
        # Update with rerun results where available
        for ind_rerun in pop:
            if np.array_equal(ind_rerun.get("X"), ind.get("X")):
                if run_lofi:
                    ind.set("F_LOFI", ind_rerun.get("F"))
                    ind.set("CB_LOFI", ind_rerun.get("CB"))
                else:
                    ind.set("F_HIFI", ind_rerun.get("F"))
                    ind.set("CB_HIFI", ind_rerun.get("CB"))
                break

    path_results_combined = write_all_individuals_lohifi(
        problem=problem,
        all_individuals=pop_all,
        save_folder=save_folder,
        file_name="all_testcases_combined.csv",
    )

    print("Combined results stored in:", path_results_combined)

    agree_disagree_df = analyse_agree_disagree(
        file=path_results_combined,
        save_folder=save_folder,
        file_name="agree_summary_combined.csv",
        num_exec_extra=len(pop_all),
        extra_execution_time=time_extra,
    )

    if wandb.run:
        print("Logging to wandb rerun.")
        wandb.log({
            "valid_failure_rate" : valid_failure_rate,
            "valid_failures" : num_failures,
            "avg_valid_rate": avg_valid_rate
        })
        wandb_log_metrics_df(agree_disagree_df)
    else:
        print("Wandb is not initialized.")

    # Plots
    write_diversity(
        pop_all, save_folder=save_folder, file_name="diversity_combined.csv"
    )
    analyse_overall(path_results_combined, save_folder=save_folder)

    save_folder_plots = os.path.join(save_folder, "plots_combined/")
    Path(save_folder_plots).mkdir(exist_ok=True, parents=True)

    # Generate and save design space plots
    plot_design_space(path_results_combined, save_folder_plots)

    # Generate and save objective space plots
    plot_objective_space(path_results_combined, save_folder_plots)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LoFi or HiFi simulation rerun and analyze results."
    )
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument(
        "--sim_rerun",
        choices=["lofi", "hifi"],
        default="lofi",
        help="Select simulation type rerun",
    )
    parser.add_argument(
        "--critical_only",
        action="store_true",
        default=False,
        help="If set, only critical results will be processed",
    )

    args = parser.parse_args()
    return args.folder, args.sim_rerun, args.critical_only


if __name__ == "__main__":
    folder, sim_rerun, critical_only = parse_args()
    rerun_and_analyze(folder, sim_rerun, critical_only)
