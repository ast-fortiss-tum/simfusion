import os
import sys
import pymoo
sys.path.insert(0,"/home/user/testing/")
sys.path.insert(0,"/home/user/testing/MultiDrive/")
sys.path.insert(0,"/home/user/testing/MultiDrive/Frenetix-Motion-Planner")
sys.path.insert(0,"/home/user/testing/MultiDrive/venv/lib/python3.10/site-packages")

from opensbt.model_ga.individual import IndividualSimulated

pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended

pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult

pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem

pymoo.core.problem.Problem = SimulationProblem

import logging

from opensbt.problem.adas_problem import ADASProblem
from opensbt.evaluation.fitness import FitnessObstacleDistanceGoalDistance
from opensbt.evaluation.critical import CriticalObstacleGoalDistance
from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.ps import *
from opensbt.algorithm.ps_rand import *
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.algorithm.ps_rand_adaptive import PureSamplingAdaptiveRandom

import wandb
import glob
from datetime import datetime
from opensbt.utils.wandb import logging_callback_archive, wandb_log_folder, wandb_log_artifact
from simulations.utils import generate_problem_name

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from simulations.commonroad.commonroad_simulation import CommonRoadSimulator

logging.basicConfig(level=logging.INFO)

import argparse

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
    p.add_argument("--project", type=str, default="planer_final")
    p.add_argument("--entity", type=str, default="lofi-hifi")
    p.add_argument("--no_wandb", action="store_true")

    p.add_argument("--sim_rerun", choices=["lofi", "hifi"], default="hifi")
    p.add_argument("--n_rerun", type=int, default=1)
    p.add_argument("--rerun_only_critical", action="store_true")

    p.add_argument("--xl", type=float, nargs="+", default=[150.0, 0.0, 10, 20, 1, 1, 9, 15, 0, 0.5, 0])
    p.add_argument("--xu", type=float, nargs="+", default=[160.0, 0.5, 15, 25, 2, 3, 12, 20, 99, 1.0, 0.5])
    p.add_argument("--ratio_sample", type=float, default=0.25)
    p.add_argument(
        "--only_if_no_hifi",
        action="store_true",
        default=True,
        help="If set (and only when sim_original=lofi & sim_rerun=hifi), rerun only individuals with no existing HiFi execution (CB_HIFI is None).",
    )
    p.add_argument(
        "--no_validation",
        action="store_true",
        default=False,
        help="Disable rerun/validation in Hifi."
    )
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

RESULTS_FOLDER = args.results_folder
scenario_path = os.path.join(os.getcwd(), "scenarios", args.scenario)

problem_name = generate_problem_name(
    name_prefix=name_prefix,
    base_name="LoFi",
    seed=seed,
    population_size=population_size,
    n_generations=n_generations,
    time=maximal_execution_time,
    algo=algo
)

run_meta = {
    "algo": algo,
    "scenario": scenario,
    "population_size": population_size,
    "generation": n_generations,
    "maximal_execution_time": maximal_execution_time,
    "seed": seed,
    "xl": xl,
    "xu": xu,
    "simulator": "commonroad",
}

tags = [f"{k}:{v}" for k, v in run_meta.items()]

wandb.init(
        entity="lofi-hifi",                  # team
        project=args.project,                   # project name
        name=problem_name,                    # run name
        group=datetime.now().strftime("%d-%m-%Y"),  # group by date
        tags=tags,
    )

# wandb.init(mode="disabled")

problem = ADASProblem(
    problem_name=problem_name,
    scenario_path=os.path.join(os.getcwd(), "scenarios", scenario),
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
        "cutin_delay",
    ],
    xl=xl,
    xu=xu,
    fitness_function=FitnessObstacleDistanceGoalDistance(),
    critical_function=CriticalObstacleGoalDistance(
        min_obstacle_distance=0.5, max_goal_distance=2
    ),
    simulate_function=CommonRoadSimulator.simulate,
    do_visualize=True,
    simulation_time=600.0,
)

my_config = DefaultSearchConfiguration()
my_config.population_size = population_size
my_config.n_generations = n_generations
my_config.maximal_execution_time = maximal_execution_time
my_config.seed = seed

if algo=="ga":
    optimizer = NsgaIIOptimizer(
        problem=problem,
        config=my_config,
        callback=logging_callback_archive
    )
elif algo == "art":
    optimizer = PureSamplingAdaptiveRandom(problem=problem, 
                                            n_candidates = 10,
                                            config=my_config,
                                                   callback=logging_callback_archive)
else :
    optimizer = PureSamplingRand(problem=problem, config=my_config,
           callback=logging_callback_archive)

RUN_ID = wandb.run.id if wandb.run else datetime.now().strftime("%Y%m%d_%H%M%S")

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
save_folder = str(Path(RESULTS_FOLDER) / problem.problem_name / "NSGA-II" / timestamp)
Path(save_folder).mkdir(parents=True, exist_ok=True)

os.environ["OPENSBT_RUN_ROOT"] = save_folder
os.environ["OPENSBT_RUN_ID"]   = RUN_ID

res = optimizer.run()
print("\n Optimization completed.")

save_folder = res.write_results(
    results_folder=RESULTS_FOLDER,
    params=optimizer.parameters,
    save_folder=optimizer.save_folder,
)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
log.info(f"====== Results saved to: {save_folder}")


wandb_log_folder(
    folder_path=save_folder,
    artifact_name="results_folder",
    artifact_type="output",
    exclude_patterns=["*report.json", "executed-simulations-*"],
)

if not args.no_validation:
    #### Rerun in hifi all NaN values. Disable if you dont need reruns.
    log.info("====== Starting CommonRoad vs BeamNG rerun analysis...")
    from simulations.rerun_tests import rerun_and_analyze

    rerun_and_analyze(simulate_function="beamng",
                    folder=save_folder,
                    sim_original="lofi",
                    sim_rerun = args.sim_rerun,
                    critical_only=args.rerun_only_critical,
                    n_rerun=args.n_rerun,
                    only_if_no_hifi=args.only_if_no_hifi,
                    ratio_sample=args.ratio_sample
                    )
    log.info("====== Rerun analysis completed!")
    wandb_log_folder(
        folder_path=os.path.join(save_folder, "rerun_hifi") ,
        artifact_name="rerun_folder",
        artifact_type="validation",
        exclude_patterns=["*report.json", "executed-simulations-*"],
    )

    report_file = glob.glob(
        os.path.join(save_folder, "**", "*report.json"), recursive=True
    )[0]

    wandb_log_artifact(
        file_path=report_file, artifact_name="report", artifact_type="validation"
    )

    print("uploaded results to wandb")
