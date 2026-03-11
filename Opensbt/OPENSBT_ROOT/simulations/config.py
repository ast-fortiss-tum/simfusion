from opensbt.simulation.simulator import SimulationOutput
from opensbt.evaluation.fitness import FitnessMinDistanceVelocityFrontOnly
from opensbt.evaluation.critical import Critical, CriticalAdasDistanceVelocity
from opensbt.config import DEFAULT_CAR_LENGTH

class CriticalAdasDistanceVelocityAutoware(Critical):
    def eval(self, vector_fitness, simout: SimulationOutput = None):
        if simout is not None and "is_collision" in simout.otherParams:
            isCollision = simout.otherParams['isCollision']
        else:
            isCollision = None
        
        # dont use is_collision, because we can have also side collision
        if (abs(vector_fitness[0]) < 1) and (abs(vector_fitness[1]) > 0.8):
            return True
        else:
            return False
        
""" Config file for lofi, hifi and lofi-hifi experiments"""

launch = dict(
    path = '/home/latnino/praktikum/autoware_mini_ws/src/autoware_mini/launch/',
    file = 'start_sim.launch',
    args = [],
)

results = "/results/"

population_size = 2
n_generations = 2 # int or None
maximal_execution_time = None #"03:00:00" #"03:00:00" #or "None"

simout_messages_interval = 0.02

scenario_file = "PedestrianCrossing.xosc"

simulation_variables=[ "PedSpeed","EgoSpeed","PedDist"]

offset = 0.80 * float(DEFAULT_CAR_LENGTH) #0.75

# bounding box collision detection
offset_x = offset #0.65 # percent
offset_y = 0 # percent

# Main config 

xl = [0.3, 3,  5.0]     # PedSpeed, EgoSpeed in m/s, PedDist in m
xu = [4, 8.3, 20]

# xl = [0.98421070,8.02417918,15.70335527]
# xu = [2.8421070,8.02417918,20.70335527]

#####################

# xl = [2.53045987,6.02112804,17.00739165]
# xu = [2.53045987,6.02112804,17.00739165]

# xl = [4.5, 4., 7.43]   # 3.59, 4.43
# xu = [4.5, 4.,   7.43]   # observation: critical hifi

# xl = [0.30022255,	7.14249611,	6.26657076]
# xu = [0.30102255,	7.14349611,	6.26657076]
# xl = [3.95825835,3.43000000,7.43000000]
# xu = [3.95825835,3.43000000,7.43000000]

# xl_kmh = [3.6, 10.8, 5.0]   
# xu_kmh = [10.8, 28.8, 20.0]

# xl=[3, 8, 30.0]  # Lower bounds for the variables
# xu=[3, 8, 30.0]    # Upper bounds for the variables

# xl=[1, 5, 10]
# xu=[4, 20, 30]

# xl = [1, 20, 5.0]     # PedSpeed, EgoSpeed in m/s, PedDist in m
# xu = [3, 20, 30.0]

fitness_function = FitnessMinDistanceVelocityFrontOnly(offset_x = offset_x,
                                                      offset_y = offset_y) 
                                                      
critical_function = CriticalAdasDistanceVelocityAutoware()

sim_time = 40  # is not used in carla, TODO check how used for lofi
sampling_time = 50

goal = (97,-55, 0)

forwarding_policy = "critical_in_hifi" # "critical_in_hifi", "all_in_hifi", "non_critical_in_hifi"
rerun_only_critical = False

seed = 1
algo="rs"

# Sequential + Predictor
run_lofi_critical_in_hifi = True
number_closest_k = 5
# CARLA

# BASIC SETUP
max_substep_delta_time = 0.01 #0.01625
max_substeps = 15 #15
resolution = 0.15 #0.24

# # LAB SETUP
# max_substep_delta_time = 0.01625
# max_substeps = 15
# resolution = 0.24