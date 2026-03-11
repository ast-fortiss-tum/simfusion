import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

import os
import numpy as np
from typing import Tuple

from simulations.utils import generate_problem_name
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.simulation.simulator import SimulationOutput
from opensbt.evaluation.fitness import Fitness
from opensbt.evaluation.critical import Critical
from opensbt.utils import geometric
from opensbt.problem.adas_problem import ADASProblem
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps_rand_adaptive import PureSamplingAdaptiveRandom
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.pso_optimizer import *
from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.algorithm.nsga2dt_optimizer import NsgaIIDTOptimizer
from opensbt.config import LOG_FILE
from simulations.rerun_tests import rerun_and_analyze

import wandb
import glob
from opensbt.utils.wandb import logging_callback_archive, wandb_log_folder, wandb_log_artifact
from datetime import datetime

import logging as log
from opensbt.utils.log_utils import *
logger = log.getLogger(__name__)
setup_logging(LOG_FILE)
disable_pymoo_warnings()

from simulations.simple_sim.autoware_simulation import AutowareSimulation
from simulations.config import results, n_generations, \
    population_size, scenario_file, maximal_execution_time, simulation_variables,\
    xl, xu, sim_time, fitness_function, critical_function, rerun_only_critical, \
    seed
import argparse

from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser(description="Run ADAS optimization experiments.")
parser.add_argument("--rerun_only_critical", action='store_true', default=rerun_only_critical,
                    help="Rerun critical.")
parser.add_argument("--seed", type=int, default=seed, help="Random seed for reproducibility.")
parser.add_argument("--algo", type=str, default="ga", help="Which algorithm to run.")
parser.add_argument("--n_generations", type=int, default=n_generations,
                    help="Number of generations for the evolutionary algorithm.")
parser.add_argument("--population_size", type=int, default=population_size,
                    help="Population size used in each generation.")
parser.add_argument("--maximal_execution_time", type=str, default=maximal_execution_time,
                    help="Maximum allowed execution time in seconds.")
parser.add_argument("--name_prefix", type=str, default="",
                    help="Prefix for the name.")
parser.add_argument("--sim_rerun", choices=["lofi", "hifi"], default="hifi", 
                    help="Simulation type for rerun (lofi or hifi)")
parser.add_argument(
    "--n_rerun",
    type=int,
    default=1,
    help="Number of reruns for validation in hifi (only for critical tests)."
)
parser.add_argument(
    "--xu",
    type=float,
    nargs='+',
    default=xu,
    help="Upper bounds for each dimension (e.g., --xu 1.0 2.0 3.0)"
)

parser.add_argument(
    "--xl",
    type=float,
    nargs='+',
    default=xl,
    help="Lower bounds for each dimension (e.g., --xl 0.0 0.5 1.0)"
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="autoware",
    help="Project name to log.",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Turn off wanbd logging"
)
parser.add_argument(
    "--only_if_no_hifi",
    action="store_true",
    default=True,
    help="If set (and only when sim_original=lofi & sim_rerun=hifi), only rerun individuals that don't already have HiFi results (CB_HIFI is None).",
)

args = parser.parse_args()

algo = args.algo
seed = args.seed
xl = args.xl
xu = args.xu
name_prefix = args.name_prefix

n_generations = args.n_generations
population_size = args.population_size
maximal_execution_time = args.maximal_execution_time

class MyFitnessMinDistanceVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: SimulationOutput, car_length = 5, **kwargs) -> Tuple[float]:
        traceEgo = simout.location["ego"]
        traceAdv = simout.location["adversary"]
        ind_min_dist = np.argmin(geometric.distPair(traceEgo, traceAdv))
        distance = np.min(geometric.distPair(traceEgo, traceAdv)) - car_length / 2
        speed = simout.speed["ego"][ind_min_dist]
        return (distance, speed)

class MyFitnessMinDistanceLinearVelocity(Fitness):
    @property
    def min_or_max(self):
        return "min", "max"

    @property
    def name(self):
        return "Min distance", "Velocity at min distance"

    def eval(self, simout: SimulationOutput, car_length = 5) -> Tuple[float]:
        traceEgo = simout.location["ego"]
        traceAdv = simout.location["adversary"]
        dist_all_x = np.asarray([
            abs(traceEgo[i][0] - traceAdv[i][0]) - car_length
            for i in range(len(traceEgo))
        ])
        dist_min = np.min(dist_all_x)
        ind_min_dist = np.argmin(dist_all_x)
        speed = simout.speed["ego"][ind_min_dist]
        return (dist_min, speed)

class MyCriticalAdasDistanceVelocity(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None) -> bool:
        return True if(vector_fitness[0] < 3) and \
                 (abs(vector_fitness[1]) > 0) else False

problem_name = generate_problem_name(name_prefix=name_prefix,
                      base_name="LoFi",
                      seed=seed,
                      population_size=population_size,
                      n_generations=n_generations,
                      time=maximal_execution_time,
                      algo=args.algo,
)

tags = [f"{k}:{v}" for k, v in vars(args).items()]

if not args.no_wandb:
    # Initialize wandb normally
    wandb.init(
        entity="lofi-hifi",                  # team
        project=args.wandb_project,                   # project name
        name=problem_name,                    # run name
        group=datetime.now().strftime("%d-%m-%Y"),  # group by date
        tags=tags
    )
else:
    # Disable wandb logging
    wandb.init(mode="disabled")

# ADAS Problem Configuration
autoware_problem = ADASProblem(
    problem_name=problem_name,
    scenario_path=os.path.join(os.getcwd(), "scenarios", scenario_file),
    simulation_variables=simulation_variables,
    xl=xl,  # Lower bounds for the variables
    xu=xu,  # Upper bounds for the variables
    fitness_function=fitness_function,  
    critical_function=critical_function,
    simulate_function=AutowareSimulation.simulate,
    do_visualize=True,
    simulation_time=sim_time
)

my_config = DefaultSearchConfiguration()
my_config.population_size = population_size
my_config.n_generations = n_generations
my_config.maximal_execution_time = maximal_execution_time
my_config.seed = seed

if maximal_execution_time != None:
    my_config.maximal_execution_time = maximal_execution_time

# Optimization
if algo=="ga":
    optimizer = NsgaIIOptimizer(problem=autoware_problem, config=my_config,
                                callback=logging_callback_archive)
elif algo == "art":
    optimizer = PureSamplingAdaptiveRandom(problem=autoware_problem, 
                                            n_candidates = 10,
                                            config=my_config)
else:
    optimizer = PureSamplingRand(problem=autoware_problem, 
                config=my_config,
                callback=logging_callback_archive)

# Define two datetime objects
time1 = datetime.now()

# Run the optimization
res = optimizer.run()
time2 = datetime.now()

# Calculate the difference
time_gap = time2 - time1

# Output results
log.info(f"Time gap: {time_gap}")
log.info(f"Total seconds: {time_gap.total_seconds()}")

save_folder = res.write_results(results_folder=results,
                                params=optimizer.parameters,
                                save_folder=optimizer.save_folder)
log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")

# Rerun in hifi. Disable if you dont need reruns.
rerun_and_analyze(folder=save_folder,
                  sim_rerun=args.sim_rerun,
                  sim_original="lofi",
                  critical_only=args.rerun_only_critical,
                  n_rerun = args.n_rerun,
                  only_if_no_hifi=args.only_if_no_hifi
                  )

# we split logging in two steps, because report.json too large

wandb_log_folder(
    folder_path=save_folder,
    artifact_name="results_folder",
    artifact_type="output",
    exclude_patterns=["*report.json"]
)

report_file = glob.glob(os.path.join(save_folder, 
           "**", "*report.json"), recursive=True)[0]

wandb_log_artifact(
    file_path=report_file,
    artifact_name="report",
    artifact_type="validation"
)

print("uploaded results to wandb")
