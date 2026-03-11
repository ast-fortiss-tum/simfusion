from datetime import datetime
import json
import os
import pymoo
import sys
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

from opensbt.problem.adas_problem_surrogate import ADASProblemSurrogate

#from predictor.agreement_predictor_cr_beamng import AgreementPredictor
from predictor.agreement_predictor import AgreementPredictor
from opensbt.problem.adas_multisim_predict_cr_beamng import MultiSimPredictProblem

from opensbt.evaluation.fitness import FitnessObstacleDistanceGoalDistance
from opensbt.evaluation.critical import CriticalObstacleGoalDistance
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from simulations.beamng.beamng_simulation import BeamNGSimulator
from simulations.commonroad.commonroad_simulation import CommonRoadSimulator
from opensbt.problem.adas_multi_sim_predict_certain_problem import ADASMultiSimPredictCertainProblem


from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.ps import *
from opensbt.algorithm.ps_rand import *

from opensbt.evaluation.fitness import *
from opensbt.evaluation.critical import *

import logging as log
from opensbt.experiment.experiment_store import experiments_store
from default_experiments import *

from opensbt.utils.log_utils import *

import wandb
import glob
from datetime import datetime
from opensbt.utils.wandb import (
    logging_callback_archive,
    wandb_log_folder,
    wandb_log_artifact,
)
from simulations.utils import generate_problem_name

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Run ADAS optimization experiments.")

    p.add_argument("--scenario", type=str, default="CUT_IN_last.xml")
    p.add_argument("--name_prefix", type=str, default="")
    p.add_argument("--seed", type=int, default=72)

    p.add_argument("--population_size", type=int, default=2)
    p.add_argument("--n_generations", type=int, default=None)
    p.add_argument("--maximal_execution_time", type=str, default=None)
    p.add_argument("--algo", type=str, default="ga")

    p.add_argument(
        "--results_folder", type=str, default=os.path.join(os.getcwd(), "results")
    )
    p.add_argument("--wandb_project", type=str, default="planer")
    p.add_argument("--entity", type=str, default="lofi-hifi")
    p.add_argument("--no_wandb", action="store_true")

    p.add_argument("--n_rerun", type=int, default=1)
    p.add_argument("--rerun_only_critical", action="store_true")
    p.add_argument("--run_lofi_critical_in_hifi", action="store_true", default=True)

    p.add_argument("--th_certainty", type=float, default=0.65)
    p.add_argument("--th_goal_distance", type=float, default=2)
    p.add_argument("--th_obstacle_distance", type=float, default=0.5)
    p.add_argument("--number_closest_k", type=int, default=3)

    p.add_argument(
        "--xl",
        type=float,
        nargs="+",
        default=[150.0, 0.0, 10, 20, 1, 1, 9, 15, 0, 0.5, 0],
    )
    p.add_argument(
        "--xu",
        type=float,
        nargs="+",
        default=[160.0, 0.5, 15, 25, 2, 3, 12, 20, 99, 1.0, 0.5],
    )
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
    p.add_argument(
        '--data_folder',
        type=str,
        required=True,
        help='Path to the data.'
    )
    p.add_argument(
        '--model_name',
        type=str,
        default="GL",
        help='Name of the model.'
    )
    p.add_argument(
        '--apply_smote',
        action="store_true",
        default=False,
        help='Apply SMOTE for training or not.'
    )
    p.add_argument(
        '--do_balance',
        action="store_true",
        default=False,
        help='Apply data balancing for training.'
    )
    p.add_argument(
        '--transform_cols',
        type=int,
        nargs='+',
        default=(),
        help='Columns to apply log1p transformation to.'
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

run_lofi_critical_in_hifi = args.run_lofi_critical_in_hifi
th_certainty = args.th_certainty
number_closest_k = args.number_closest_k
scenario = args.scenario

RESULTS_FOLDER = os.path.join(os.getcwd(), "results", args.results_folder)
scenario_path = os.path.join(os.getcwd(), "scenarios", scenario)

LOG_FILE = "log.txt"

logger = log.getLogger(__name__)

setup_logging(LOG_FILE)
disable_pymoo_warnings()

problem_name = generate_problem_name(
    name_prefix=name_prefix,
    base_name="Surrogate",
    seed=seed,
    population_size=population_size,
    n_generations=n_generations,
    time=maximal_execution_time,
    algo=algo,
    k = number_closest_k,
    thc = th_certainty,
    thg= args.th_goal_distance,
    tho = args.th_obstacle_distance
)

tags = [f"{k}:{v}" for k, v in vars(args).items()]

if not args.no_wandb:
    wandb.init(
        entity=args.entity,  # team
        project=args.wandb_project,  # project name
        name=problem_name,  # run name
        group=datetime.now().strftime("%d-%m-%Y"),  # group by date
        tags=tags,
    )
else:
    wandb.init(mode="disabled")

simulation_variables = [
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
]

# ADAS Problem Configuration
problem = ADASProblemSurrogate(
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
        min_obstacle_distance=args.th_obstacle_distance, max_goal_distance=args.th_goal_distance
    ),
    simulate_function=BeamNGSimulator.simulate,
    do_visualize=True,
    simulation_time=600.0,
    model_name=args.model_name,
    data_folder=args.data_folder,
    apply_smote=args.apply_smote,
    do_balance=args.do_balance,
    transform_cols=args.transform_cols
)

my_config = DefaultSearchConfiguration()

my_config.population_size = population_size
my_config.n_generations = n_generations
my_config.maximal_execution_time = maximal_execution_time
my_config.seed = seed

optimizer = NsgaIIOptimizer(
    problem=problem, config=my_config, callback=logging_callback_archive
)

RUN_ID = wandb.run.id if wandb.run else datetime.now().strftime("%Y%m%d_%H%M%S")

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
save_folder = str(Path(RESULTS_FOLDER) / problem.problem_name / "NSGA-II" / timestamp)
Path(save_folder).mkdir(parents=True, exist_ok=True)

os.environ["OPENSBT_RUN_ROOT"] = save_folder
os.environ["OPENSBT_RUN_ID"] = RUN_ID

res = optimizer.run()

save_folder = res.write_results(
    results_folder=RESULTS_FOLDER,
    params=optimizer.parameters,
    save_folder=optimizer.save_folder,
)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")
log.info(f"====== Results saved to: {save_folder}")

# metrics = problem.get_default_metrics()
# print(metrics)

# run_dir = Path(save_folder)
# run_dir.mkdir(parents=True, exist_ok=True)

# payload = {
#     **metrics,
#     "exec_time_sec": float(res.exec_time),
#     "seed": my_config.seed,
#     "population_size": my_config.population_size,
#     "timestamp": datetime.now().isoformat(timespec="seconds"),
# }

# # JSON
# (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

wandb_log_folder(
    folder_path=save_folder,
    artifact_name="results_folder",
    artifact_type="output",
    exclude_patterns=["*report.json", "executed-simulations-*"],
)

if not args.no_validation:
    from simulations.rerun_sequential import rerun_and_analyze_sequential

    rerun_and_analyze_sequential(
        simulate_function="beamng",
        folder=save_folder, 
        critical_only=args.rerun_only_critical, 
        n_rerun=args.n_rerun,
        only_if_no_hifi = args.only_if_no_hifi
    )
    
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
