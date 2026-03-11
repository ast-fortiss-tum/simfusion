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
from simulations.carla.carla_runner_sim import CarlaRunnerSimulator
from simulations.simple_sim.autoware_simulation import AutowareSimulation
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

def resume_rerun_and_analyze(folder: str, sim_rerun: str, critical_only: bool, evaluated_tests_path= "hifi_rerun/evaluated_tests.csv"):
    run_lofi = (sim_rerun == "lofi")

    problem_path = os.path.join(folder,"backup/problem")
    test_cases_path = os.path.join(folder,"all_testcases.csv")
    test_cases_eval_path = os.path.join(folder, evaluated_tests_path)

    with open(problem_path, 'rb') as f:
        problem_read = dill.load(f)

    if run_lofi:
        simulate_function = AutowareSimulation.simulate
    else:
        simulate_function = CarlaRunnerSimulator.simulate

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

    pop = read_pf_single(filename=test_cases_path, with_critical_column=True)
    pop = duplicate_free(pop)

    pop_evaluated = read_pf_single(filename=test_cases_eval_path, with_critical_column=True)
    
    if critical_only:
        pop, _ = pop.divide_critical_non_critical()
    
    if len(pop) > 0:
        print(f"Doing rerun for {len(pop)} tests.")
        
     
        save_folder = os.path.join(folder, f"rerun_{sim_rerun}/")
        Path(save_folder).mkdir(exist_ok=True, parents=True)

        def individual_already_evaluated(ind, archive):
            for ind_old in archive:
                print("Comparing:",ind_old.get("X"), ind.get("X"), np.array_equal(ind_old.get("X"), ind.get("X")))
                if ind_old is not None and np.array_equal(ind_old.get("X"), ind.get("X")):
                    # print("Invidual already evaluated. Skipping.")
                    return ind_old
            return None

        start_time = time.time()
        for ind in pop:
            # set values from first round execution
            if run_lofi:
                ind.set("F_HIFI", ind.get("F"))
                ind.set("CB_HIFI", ind.get("CB"))
            else:
                ind.set("F_LOFI", ind.get("F"))
                ind.set("CB_LOFI", ind.get("CB"))

            ind_old = individual_already_evaluated(ind, pop_evaluated)
            
            if ind_old == None:
                evaluate_individuals(Population(individuals=[ind]), problem=problem, backup_folder=save_folder)
                # set values from evaluated
                if run_lofi:
                    ind.set("F_LOFI", ind.get("F"))
                    ind.set("CB_LOFI", ind.get("CB"))
                else:
                    ind.set("F_HIFI", ind.get("F"))
                    ind.set("CB_HIFI", ind.get("CB"))
            else:
                # set values from evaluated ind
                if run_lofi:
                    ind.set("F_LOFI", ind_old.get("F"))
                    ind.set("CB_LOFI", ind_old.get("CB"))
                else:
                    ind.set("F_HIFI", ind_old.get("F"))
                    ind.set("CB_HIFI", ind_old.get("CB"))
            
        time_extra = time.time() - start_time

        path_results = write_all_individuals_lohifi(problem=problem,
                                                    all_individuals=pop,
                                                    save_folder=save_folder)
    
        print("Rerun results stored in:", path_results)

        analyse_agree_disagree(file=path_results,
                            save_folder=save_folder,
                            file_name="agree_summary.csv",
                            num_exec_extra = len(pop),
                            extra_execution_time=time_extra)

        write_simulation_output(pop, save_folder=save_folder)

        ######## Plots

        save_folder = os.path.join(save_folder, "plots/")
        Path(save_folder).mkdir(exist_ok=True, parents=True)

        # Generate and save design space plots
        plot_design_space(path_results, save_folder)

        # Generate and save objective space plots
        plot_objective_space(path_results, save_folder)
    else:
        print("No individuals for rerun.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LoFi or HiFi simulation rerun and analyze results.")
    parser.add_argument("--folder", type=str, required=True, help="Results folder path")
    parser.add_argument("--sim_rerun", choices=["lofi", "hifi"], default="lofi", help="Select simulation type rerun")
    parser.add_argument("--critical_only", action="store_true", default=False, help="If set, only critical results will be processed")
    parser.add_argument("--evaluated_tests_path", required=True, type=str, help="Path to already evaluated tests")

    args = parser.parse_args()
    return args.folder, args.sim_rerun, args.critical_only, args.evaluated_tests_path


if __name__ == "__main__":
    folder, sim_rerun, critical_only, evaluated_tests_path = parse_args()
    resume_rerun_and_analyze(folder, sim_rerun, critical_only, evaluated_tests_path)