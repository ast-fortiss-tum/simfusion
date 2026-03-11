import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

# Predictor plots for CR-BeamNG with correct variable names
from predictor.analyse_data_cr_beamng import plot_design_space, plot_objective_space

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
from simulations.beamng.beamng_simulation import BeamNGSimulator

import dill
import sys
from typing import Tuple
import numpy as np
import math
from opensbt.utils import geometric
from simulations.simple_sim import *
from simulations.carla import *
from simulations.utils import *
import logging as log
import argparse
from opensbt.visualization.visualizer import write_diversity

def rerun_and_analyze_sequential(folder: str, critical_only = False):
    sim_rerun = "hifi"
    
    problem_path = os.path.join(folder,"backup/problem")
    test_cases_path = os.path.join(folder,"all_testcases.csv")

    import dill
    with open(problem_path, 'rb') as f:
        problem_read = dill.load(f)

    simulate_function = BeamNGSimulator.simulate

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

    pop: PopulationExtended = read_pf_single_sequential(filename=test_cases_path,
                                                        problem=problem, 
                                                        with_critical_column=True,
                                                        skip_lofi_nan=True)
    print("CB read:", pop.get("CB"))
    pop = duplicate_free(pop)

    if critical_only:
        pop, _ = pop.divide_critical_non_critical()
    
    # filtered out
    print("CB filtered out:", pop.get("CB"))
    for ind in pop:
        print(ind.get("X"))
        print(ind.get("F_LOFI"))
        print(ind.get("CB_LOFI"))
    import time

    time_extra = 0
    
    save_folder = os.path.join(folder, f"rerun_{sim_rerun}/")
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    if len(pop) > 0:        
        start_time = time.time()
        for ind in pop:
            evaluate_individuals(Population(individuals=[ind]), problem=problem, 
                                    backup_folder=save_folder)
        time_extra = time.time() - start_time

        for ind in pop:
            ind.set("F_HIFI", ind.get("F"))
            ind.set("CB_HIFI", ind.get("CB"))

        print(f"Doing rerun for {len(pop)} tests.")

        path_results = write_all_individuals_lohifi(problem=problem,
                                                    all_individuals=pop,
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

        save_folder_plots = os.path.join(save_folder, "plots/")
        Path(save_folder_plots).mkdir(exist_ok=True, parents=True)

        # Generate and save design space plots
        plot_design_space(path_results, save_folder_plots)

        # Generate and save objective space plots
        plot_objective_space(path_results, save_folder_plots)

    else:
        print("No tests to execute.")

    ######### get the overall analysis
    pop: PopulationExtended = read_pf_single_sequential(filename=test_cases_path, 
                                                        problem=problem,
                                                        with_critical_column=True,
                                                        skip_lofi_nan=False)
    # print("CB read:", pop.get("CB"))
    pop_all = duplicate_free(pop)

    for ind in pop_all:
        # take over the values from the rerun
        for ind_rerun in pop:
            # print("Comparing:",ind_rerun.get("X"), ind.get("X"), np.array_equal(ind_rerun.get("X"), ind.get("X")))
            if np.array_equal(ind_rerun.get("X"), ind.get("X")):
                ind.set("F_HIFI",  ind_rerun.get("F_HIFI") )
                ind.set("CB_HIFI", ind_rerun.get("CB_HIFI"))    
                #ind.set("F_LOFI", [None] * problem.n_obj)
                break
                 
    path_results_combined = write_all_individuals_lohifi(problem=problem,
                                                all_individuals=pop_all,
                                                save_folder=save_folder,
                                                file_name="all_testcases_combined.csv")
    
    print("Combined results stored in:", path_results_combined)

    analyse_agree_disagree(file=path_results_combined,
                        save_folder=save_folder,
                        file_name="agree_summary_combined.csv",
                        num_exec_extra = len(pop_all),
                        extra_execution_time=-1)
    
    # Plots
    write_diversity(pop_all, save_folder,file_name="diversity_combined.csv")
    analyse_overall(path_results_combined,save_folder=save_folder)

    save_folder = os.path.join(save_folder, "plots_combined/")
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    # Generate and save design space plots
    plot_design_space(path_results_combined, save_folder)

    # Generate and save objective space plots
    plot_objective_space(path_results_combined, save_folder)


def parse_args():
    parser = argparse.ArgumentParser(description="Run LoFi or HiFi simulation rerun and analyze results.")
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument("--critical_only", action="store_true", default=False, help="If set, only critical results will be processed")
    
    args = parser.parse_args()
    return args.folder, args.critical_only


if __name__ == "__main__":
    folder, critical_only = parse_args()
    rerun_and_analyze_sequential(folder, critical_only)