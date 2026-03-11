from pathlib import Path
import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

import dill
import os
from opensbt.experiment.search_configuration import DefaultSearchConfiguration

path_result = "./results/Sequential_Testing_Predict_seed-303/NSGA-II/25-05-2025_04-57-04/backup/result"
path_problem = "./results/Sequential_Testing_Predict_seed-303/NSGA-II/25-05-2025_04-57-04/backup/problem"

with open(path_result, 'rb') as f:
    result = dill.load(f)
    print("Loaded Result:", result)

with open(path_problem, 'rb') as f:
    problem = dill.load(f)
    print("Loaded Problem:", problem)
    
# # extract algorithm 
# algo = result.history[-1]

# # write results
# algo =NSGAII(problem,
#                    config=DefaultSearchConfiguration())

# algo.res = result

path = Path(path_result)
results_folder = os.path.join(path.parent.parent, "generated/")

# check result objects
all_individuals = result.obtain_archive()

for ind in all_individuals:
    print("lofi f:", ind.get("F_LOFI"))
    print("hifi f:", ind.get("F_HIFI"))
    print("f:", ind.get("F"))

result.write_results(save_folder = results_folder,
                     params = {}
                     )