import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from predictor.agreement_predictor import AgreementPredictor
from opensbt.problem.adas_multi_sim_predict_certain_problem import ADASMultiSimPredictCertainProblem

from opensbt.algorithm.nsga2_optimizer import *
from simulations.utils import generate_problem_name

import logging as log
import os

from opensbt.experiment.experiment_store import experiments_store
from default_experiments import *
from opensbt.utils.log_utils import *
from opensbt.config import RESULTS_FOLDER, LOG_FILE

logger = log.getLogger(__name__)

setup_logging(LOG_FILE)

disable_pymoo_warnings()

import os
from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.evaluation.fitness import *
from opensbt.problem.adas_multi_sim_problem import ADASMultiSimProblem
from opensbt.problem.pymoo_test_problem import PymooTestProblem
from opensbt.experiment.experiment_store import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import *

from simulations.carla.carla_runner_sim import CarlaRunnerSimulator
from simulations.simple_sim.autoware_simulation import AutowareSimulation
from simulations.config import results,n_generations, population_size, \
      maximal_execution_time, sim_time, xl, xu, critical_function, fitness_function, \
      simulation_variables, rerun_only_critical, seed, run_lofi_critical_in_hifi, \
      number_closest_k

import wandb
from opensbt.utils.wandb import logging_callback_archive
from datetime import datetime
from simulations.rerun_sequential import rerun_and_analyze_sequential
from opensbt.utils.wandb import logging_callback_archive, wandb_log_folder, wandb_log_artifact
import glob
import argparse

# Argument parsing
parser = argparse.ArgumentParser(description="Run ADAS optimization experiments.")
parser.add_argument("--rerun_only_critical", action='store_true', default=rerun_only_critical,
                    help="Rerun critical.")
parser.add_argument("--algo", type=str, default="ga", help="Which algorithm to run.")
parser.add_argument("--n_generations", type=int, default=n_generations,
                    help="Number of generations for the evolutionary algorithm.")
parser.add_argument("--population_size", type=int, default=population_size,
                    help="Population size used in each generation.")
parser.add_argument("--maximal_execution_time", type=str, default=maximal_execution_time,
                    help="Maximum allowed execution time in seconds.")
parser.add_argument("--seed", type=int, default=seed,
                    help="Maximum allowed execution time in seconds.")
parser.add_argument("--run_lofi_critical_in_hifi", action="store_true", default=run_lofi_critical_in_hifi, help="If set, lofi criticals will be passed to hifi")
parser.add_argument("--name_prefix", type=str, default="",
                    help="Prefix for the name.")
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
    "--th_certainty",
    type=float,
    default=0.8,
    help="Certainty threshold for predictions."
)
parser.add_argument(
    "--only_if_no_hifi",
    action="store_true",
    help="Only run lofi and rerun in hifi if no hifi results are available.",
)

args = parser.parse_args()

# # Constant Values
# CAR_LENGTH = 5.00

# class MyFitnessMinDistanceVelocity(Fitness):
#     @property
#     def min_or_max(self):
#         return "min", "max"

#     @property
#     def name(self):
#         return "Min distance", "Velocity at min distance"

#     def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float]:
#         traceEgo = simout.location["ego"]
#         traceAdv = simout.location["adversary"]
#         ind_min_dist = np.argmin(geometric.distPair(traceEgo, traceAdv))
#         distance = np.min(geometric.distPair(traceEgo, traceAdv)) - CAR_LENGTH / 2
#         speed = simout.speed["ego"][ind_min_dist]
#         return (distance, speed)

# class MyCriticalAdasDistanceVelocity(Critical):
#     def eval(self, vector_fitness, simout: SimulationOutput = None) -> bool:
#         return True if(vector_fitness[0] < 3) and \
#                  (abs(vector_fitness[1]) > 0) else False

seed = args.seed
n_generations = args.n_generations
population_size = args.population_size
maximal_execution_time = args.maximal_execution_time
xu = args.xu
xl = args.xl
run_lofi_critical_in_hifi = args.run_lofi_critical_in_hifi
name_prefix = args.name_prefix

problem_name = generate_problem_name(name_prefix=name_prefix,
                      base_name="PredictProb",
                      seed=seed,
                      population_size=population_size,
                      n_generations=n_generations,
                      time=maximal_execution_time,
                      k = number_closest_k,
                      algo=args.algo)

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

problem = ADASMultiSimPredictCertainProblem(
                        problem_name=problem_name,
                        scenario_path="scenarios/PedestrianCrossing.xosc",
                        xl=xl, 
                        xu=xu,
                        simulation_variables = simulation_variables,
                        fitness_function=fitness_function,  
                        critical_function=critical_function,
                        simulate_function=[AutowareSimulation.simulate, CarlaRunnerSimulator.simulate],
                        do_visualize = True,
                        simulation_time=sim_time,
                        predictor=AgreementPredictor("./predictor/data_final/", simulation_variables, k = number_closest_k),
                        certainty_threshold=args.th_certainty
                        )

my_config=DefaultSearchConfiguration()

my_config.population_size = population_size
my_config.n_generations = n_generations
my_config.maximal_execution_time = maximal_execution_time
my_config.seed = seed

optimizer = NsgaIIOptimizer(
            problem=problem,
            config=my_config,
            callback=logging_callback_archive)

res = optimizer.run()

save_folder = res.write_results(results_folder=results, 
                                params = optimizer.parameters,
                                save_folder=optimizer.save_folder)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")

###### Rerun in hifi all NaN values. Disable if you dont need reruns.
rerun_and_analyze_sequential(folder = save_folder,
                             critical_only = args.rerun_only_critical,
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