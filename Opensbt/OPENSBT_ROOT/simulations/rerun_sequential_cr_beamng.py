from collections import Counter
from datetime import datetime
import json
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
from opensbt.utils.encoder_utils import NumpyEncoder
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

# Predictor plots for CR-BeamNG with correct variable names
from predictor.analyse_data_cr_beamng import plot_design_space, plot_objective_space

import os
from pathlib import Path
import numpy as np

from opensbt.config import OUTPUT_PRECISION
from opensbt.problem.adas_problem import ADASProblem
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.utils.duplicates import duplicate_free
from opensbt.visualization.combined import read_pf_single
from simulations.beamng.beamng_simulation import BeamNGSimulator

from typing import Tuple
import numpy as np
import math
from opensbt.utils import geometric
from simulations.utils import *
from simulations.utils import read_pf_single_sequential
import logging as log
import argparse
from opensbt.visualization.visualizer import write_diversity
from opensbt.utils.wandb import wandb_log_metrics_df
import wandb

def rerun_and_analyze_sequential(folder: str, critical_only=False, n_rerun: int = 1):
    sim_rerun = "hifi"
    is_flakiness_check = n_rerun >= 1
    
    problem_path = os.path.join(folder,"backup/problem")
    test_cases_path = os.path.join(folder,"all_testcases.csv")

    # Load the original problem configuration from backup
    import dill
    with open(problem_path, 'rb') as f:
        problem_read = dill.load(f)

    simulate_function = BeamNGSimulator.simulate

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

    pop_all: PopulationExtended = read_pf_single_sequential(filename=test_cases_path,
                                                        problem=problem, 
                                                        with_critical_column=True,
                                                        skip_lofi_nan=False)
    print("Len pop read no dup free: ", len(pop_all))
    pop_all = copy.deepcopy(duplicate_free(pop_all))

    # Filter to only critical test cases if requested
    if critical_only:
        pop, _ = pop_all.divide_critical_non_critical()

    # Display information about filtered test cases
    print("CB filtered out:", pop.get("CB"))
    # Display detailed information about each test case to be processed
    # for ind in pop:
    #     print("X:", ind.get("X"))
    #     print("F_LOFI:", ind.get("F_LOFI"))
    #     print("CB_LOFI:", ind.get("CB_LOFI"))
    
    import time

    # Create output folder for rerun results
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
        "tests": []
    }

    # Execute BeamNG simulations if there are test cases to process
    if len(pop) > 0:        
        start_time = time.time()
        print(f"Doing rerun for {len(pop)} tests.")
        print(f"n_rerun set to: {n_rerun}")
        # Process each individual test case
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

        # Save all individual results to CSV file
        path_results = write_all_individuals_lohifi(problem=problem,
                                                    all_individuals=pop,
                                                    save_folder=save_folder,
                                                    file_name="all_testcases.csv")

        print("BeamNG rerun results stored in:", path_results)

        # Analyze agreement/disagreement between LoFi and HiFi simulators
        analyse_agree_disagree(file=path_results,
                            save_folder=save_folder,
                            file_name="agree_summary.csv",
                            num_exec_extra=len(pop),
                            extra_execution_time=time_extra)

        # Write simulation output data for further analysis
        write_simulation_output(pop, save_folder=save_folder)

        # Generate visualization plots for rerun results
        ######## Plots
        
        save_folder_plots = os.path.join(save_folder, "plots/")
        Path(save_folder_plots).mkdir(exist_ok=True, parents=True)

        # Generate design space plots showing parameter distributions
        plot_design_space(path_results, save_folder_plots)

        # Generate objective space plots showing fitness landscapes
        plot_objective_space(path_results, save_folder_plots)

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

    # Perform overall analysis by combining original and rerun results
    ######### get the overall analysis
    
    # Read all original test cases (including those not rerun)
    # pop: PopulationExtended = read_pf_single_sequential(filename=test_cases_path, 
    #                                                     problem=problem,
    #                                                     with_critical_column=True,
    #                                                     skip_lofi_nan=False)
    # print("CB read:", pop.get("CB"))
    # pop_all = duplicate_free(pop)
    #pop_all = pop

    for ind in pop_all:
        # Find corresponding rerun result and copy HiFi values
        for ind_rerun in pop:
            # print("Comparing:",ind_rerun.get("X"), ind.get("X"), np.array_equal(ind_rerun.get("X"), ind.get("X")))
            if np.array_equal(ind_rerun.get("X"), ind.get("X")):
                ind.set("F_HIFI",  ind_rerun.get("F_HIFI") )    # Copy high-fidelity fitness
                ind.set("CB_HIFI", ind_rerun.get("CB_HIFI"))    # Copy high-fidelity critical behavior
                #ind.set("F_LOFI", [None] * problem.n_obj)
                break
                 
    # Save combined results (LoFi + HiFi) to CSV file             
    path_results_combined = write_all_individuals_lohifi(problem=problem,
                                                all_individuals=pop_all,
                                                save_folder=save_folder,
                                                file_name="all_testcases_combined.csv")
    
    # Analyze agreement/disagreement for combined dataset
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
            "avg_valid_rate": avg_valid_rate
        })
        wandb_log_metrics_df(agree_disagree_df)
    else:
        print("Wandb is not initialized.")
    
    # Generate diversity analysis and overall statistics
    write_diversity(pop_all, save_folder,file_name="diversity_combined.csv")
    analyse_overall(path_results_combined,save_folder=save_folder)

    # Generate comprehensive visualization plots for combined results
    # Combined plots now enabled for CR-BeamNG with correct variables
    save_folder_plots_combined = os.path.join(save_folder, "plots_combined/")
    Path(save_folder_plots_combined).mkdir(exist_ok=True, parents=True)

    # Generate design space plots for combined dataset
    plot_design_space(path_results_combined, save_folder_plots_combined)

    # Generate objective space plots for combined dataset
    plot_objective_space(path_results_combined, save_folder_plots_combined)


def parse_args():
    """
    Step 32: Parse command line arguments for script execution
    
    Returns:
        folder: Path to the results folder containing original simulation data
        critical_only: Boolean flag to process only critical test cases
    """
    parser = argparse.ArgumentParser(description="Run LoFi or HiFi simulation rerun and analyze results.")
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument("--critical_only", action="store_true", default=False, help="If set, only critical results will be processed")
    
    args = parser.parse_args()
    return args.folder, args.critical_only


if __name__ == "__main__":
    # Step 33: Main execution entry point
    # Parse command line arguments and execute rerun analysis
    folder, critical_only = parse_args()
    rerun_and_analyze_sequential(folder, critical_only)