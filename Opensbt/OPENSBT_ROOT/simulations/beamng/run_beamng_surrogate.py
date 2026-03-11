import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

#from surrogate.model import Model

from opensbt.algorithm.nsga2_optimizer import *

import logging as log
import os

from opensbt.experiment.experiment_store import experiments_store
from default_experiments import *
from opensbt.utils.log_utils import *
from opensbt.config import RESULTS_FOLDER, LOG_FILE


logger = log.getLogger(__name__)

setup_logging(LOG_FILE)

disable_pymoo_warnings()

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.evaluation.fitness import FitnessObstacleDistanceGoalDistance
from opensbt.experiment.experiment_store import *
from opensbt.algorithm.algorithm import *
from opensbt.evaluation.critical import CriticalObstacleGoalDistance
from opensbt.problem.adas_surrogate_problem import ADASProblemSurrogate

from simulations.beamng.beamng_simulation import BeamNGSimulator

problem = ADASProblemSurrogate(
    problem_name="beamng_surrogate",
    scenario_path=os.path.join(os.getcwd(), "scenarios", "CUT_IN_last.xml"),
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
    xl=[150.0, 0.0, 5, 11, 1, 1, 20, 10, 0, 0, 0],
    xu=[170.0, 0.5, 10, 15, 3, 4, 40, 12, 99, 3, 3],
    fitness_function=FitnessObstacleDistanceGoalDistance(),
    critical_function=CriticalObstacleGoalDistance(min_obstacle_distance=0.5, max_goal_distance=2.0),
    simulate_function=BeamNGSimulator.simulate,
    do_visualize = True,
    simulation_time=600.0,
    model_name = "RF",
    data_folder= "./surrogate/data_beamng/batch1/",
    apply_smote= False
    )

# my_config=DefaultSearchConfiguration()
# my_config.population_size = 10
# my_config.n_generations = None
# my_config.maximal_execution_time = "12:00:00"
# my_config.seed = 49


# my_config=DefaultSearchConfiguration()
# my_config.population_size = 20
# my_config.n_generations = None
# my_config.maximal_execution_time = "6:00:00"
# my_config.seed = 21

my_config=DefaultSearchConfiguration()
my_config.population_size = 20
my_config.n_generations = None
my_config.maximal_execution_time = "6:00:00"
my_config.seed = 70


optimizer = NsgaIIOptimizer(
            problem=problem,
            config=my_config)

res = optimizer.run()

save_folder = res.write_results(results_folder="/results/", 
                                params = optimizer.parameters,
                                save_folder=optimizer.save_folder)

log.info("====== Algorithm search time: " + str("%.2f" % res.exec_time) + " sec")

###### Rerun in hifi all NaN values. Disable if you dont need reruns.
from simulations.rerun_sequential_cr_beamng import rerun_and_analyze_sequential
rerun_and_analyze_sequential(folder = save_folder,
                             critical_only = True
                             )