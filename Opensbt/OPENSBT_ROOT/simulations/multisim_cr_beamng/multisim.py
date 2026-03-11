import os
import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.problem.adas_multisim_cr_beamng import MultiSimProblem
from opensbt.evaluation.fitness import FitnessObstacleDistanceGoalDistance
from opensbt.evaluation.critical import CriticalObstacleGoalDistance
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from simulations.beamng.beamng_simulation import BeamNGSimulator
from simulations.commonroad.commonroad_simulation import CommonRoadSimulator

from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.ps import *
from opensbt.algorithm.ps_rand import *

import wandb
import glob
from datetime import datetime
from opensbt.utils.wandb import logging_callback_archive, wandb_log_folder, wandb_log_artifact
from simulations.utils import generate_problem_name

import logging as log
from opensbt.experiment.experiment_store import experiments_store
from default_experiments import *

from opensbt.utils.log_utils import *

import argparse

LOG_FILE = "log.txt"

logger = log.getLogger(__name__)

setup_logging(LOG_FILE)
disable_pymoo_warnings()

import os

def parse_args():
    p = argparse.ArgumentParser(description="Run OpenSBT Commonroad (LoFi) experiment")

    p.add_argument("--scenario", type=str, default="CUT_IN_last.xml")
    p.add_argument("--name_prefix", type=str, default="")
    p.add_argument("--seed", type=int, default=72)

    p.add_argument("--population_size", type=int, default=2)
    p.add_argument("--n_generations", type=int, default=None)
    p.add_argument("--maximal_execution_time", type=str, default=None)
    p.add_argument("--algo", type=str, default="ga")

    p.add_argument("--results_folder", type=str, default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--wandb_project", type=str, default="planer")
    p.add_argument("--entity", type=str, default="lofi-hifi")
    p.add_argument("--no_wandb", action="store_true")

    p.add_argument("--sim_rerun", choices=["lofi", "hifi"], default="hifi")
    p.add_argument("--n_rerun", type=int, default=1)
    p.add_argument("--rerun_only_critical", action="store_true")

    p.add_argument("--xl", type=float, nargs="+", default=[150.0, 0.0, 10, 20, 1, 1, 9, 15, 0, 0.5, 0])
    p.add_argument("--xu", type=float, nargs="+", default=[160.0, 0.5, 15, 25, 2, 3, 12, 20, 99, 1.0, 0.5])

    return p.parse_args()

args = parse_args()

algo = args.algo
seed = args.seed
xl = args.xl
xu = args.xu
name_prefix = args.name_prefix

n_generations = args.n_generations
population_size = args.population_size
maximal_execution_time = args.maximal_execution_time

scenario = args.scenario

RESULTS_FOLDER = os.path.join(os.getcwd(), "results", args.results_folder)
scenario_path = os.path.join(os.getcwd(), "scenarios", args.scenario)

problem_name = generate_problem_name(
    name_prefix=name_prefix,
    base_name="MultiSim",
    seed=seed,
    population_size=population_size,
    n_generations=n_generations,
    time=maximal_execution_time,
    algo=algo
)

tags = [f"{k}:{v}" for k, v in vars(args).items()]

if not args.no_wandb:
    wandb.init(
            entity=args.entity,                  # team
            project=args.wandb_project,                   # project name
            name=problem_name,                    # run name
            group=datetime.now().strftime("%d-%m-%Y"),  # group by date
            tags=tags,
        )
else:
    wandb.init(mode="disabled")

# ADAS Problem Configuration
problem = MultiSimProblem(
    problem_name=problem_name,
    scenario_path=scenario_path,
    simulation_variables=[
        "goal_center_x",
        "goal_center_y",
        "goal_velocity_min",
        "goal_velocity_max",
        "num_npc",
        "lane_count",
        "npc_x0",
        "npc_v_max",
        "npc_seed",
        "cutin_tx",
        "cutin_delay"
    ],
    xl=xl,
    xu=xu,
    fitness_function=FitnessObstacleDistanceGoalDistance(),
    critical_function=CriticalObstacleGoalDistance(min_obstacle_distance=0.5, max_goal_distance=0.0),
    simulate_function=[CommonRoadSimulator.simulate, BeamNGSimulator.simulate],
    do_visualize=True,
    simulation_time=2000.0,
    forwarding_policy="all_in_hifi",
)

fitness_function = problem.fitness_function
critical_function = problem.critical_function

my_config=DefaultSearchConfiguration()
my_config.population_size = 100
my_config.n_generations = 1
my_config.forwarding_policy = "all_in_hifi"
my_config.seed = 49

optimizer = PureSamplingRand(problem=problem, config=my_config)
res = optimizer.run()

save_folder = res.write_results(
    results_folder=RESULTS_FOLDER,
    params=optimizer.parameters,
    save_folder=optimizer.save_folder
)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
log.info(f"====== Results saved to: {save_folder}")

# ##### Rerun in hifi all NaN values. Disable if you dont need reruns.
# log.info("====== Starting CommonRoad vs BeamNG rerun analysis...")
# from simulations.rerun_analyse_cr_beamng import rerun_and_analyze_sequential
# rerun_and_analyze_sequential(folder = save_folder,
#                              critical_only = False
#                              )
# log.info("====== Rerun analysis completed!")
