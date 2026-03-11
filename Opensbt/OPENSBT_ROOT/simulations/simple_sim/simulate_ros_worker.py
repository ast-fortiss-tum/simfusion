import math
import sys

# Insert '/home/opensbt/' at the start of sys.path
sys.path.insert(0, '/home/opensbt/')

import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.utils.encoder_utils import NumpyEncoder

from opensbt.algorithm.ps_fps import PureSamplingFPS
from opensbt.algorithm.ps_grid import PureSamplingGrid
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.pso_optimizer import *
from opensbt.algorithm.algorithm import AlgorithmType
from opensbt.algorithm.nsga2dt_optimizer import NsgaIIDTOptimizer

import time
import json
import numpy as np
from typing import List
from opensbt.model_ga.individual import Individual
from opensbt.simulation.simulator import Simulator, SimulationOutput


from simulations.simple_sim.autoware_helper import AutowareHelper
from simulations.simple_sim.object_operations import ObjectType
from simulations.simple_sim.utility import Utility
import logging
from typing import List, Dict

# for restart
import traceback
import rospy
from simulations.simple_sim.restart_lofi import restart_lofi
import subprocess

DEBUG_MODE = True 
SCENARIO_DIR = "./tmpscenarios"
CAR_NAME = "ego_vehicle"

import json
import traceback
import sys
from simulations.simple_sim.autoware_simulation import AutowareSimulation, SCENARIO_DIR
from simulations.simple_sim.autoware_helper import AutowareHelper
from simulations.config import goal
import argparse
import time
from run_canceler import call_cancel_route_service

# WIP
parser = argparse.ArgumentParser()

# For variable_names, allow it to accept a list of strings
parser.add_argument(
    "--individual", 
    type=float, 
    nargs='+',  # This allows one or more values
    help="List of scenario variable values")

# For variable_names, allow it to accept a list of strings
parser.add_argument(
    "--variable_names", 
    type=str, 
    nargs='+',  # This allows one or more values
    help="List of variable names"
)

parser.add_argument("--xosc_path", type=str)
parser.add_argument("--sim_time", type=float)
parser.add_argument("--time_step", type=float)
args = parser.parse_args()

# Deserialize if necessary
individual = args.individual
variable_names = args.variable_names

class ROSController:
    """
    A utility class to manage ROS processes such as restarting or killing ROS nodes.
    """
    @staticmethod
    def kill_ros_processes(TIME_WAIT = 5):
        """
        Kill common ROS processes: roscore, roslaunch, and simulation nodes.
        """
        rospy.signal_shutdown("Resetting ROS node to reconnect")

        logging.warning("Killing ROS processes...")
        subprocess.run(["pkill", "-f", "roslaunch"])
        subprocess.run(["pkill", "-f", "roscore"])
        subprocess.run(["pkill", "-f", "scenario_simulation"])  # adjust to your simulation process name
        time.sleep(TIME_WAIT)  # wait for processes to terminate

    @staticmethod
    def reset_ros_node(reason="Resetting ROS node"):
        """
        Cleanly shut down current ROS node (if initialized).
        """
        try:
            rospy.signal_shutdown(reason)
            time.sleep(1)
        except Exception:
            logging.warning("ROS shutdown failed or not initialized.")
    
    @staticmethod
    def simulate_single(individual: List[float],  # individual implicitly passed via scenario file
                            variable_names: List[str], # variable names implicitly passed via scenario file
                            filepath: str, 
                            sim_time: int, 
                            time_step: float) -> SimulationOutput:
            """
            Simulate a single individual.

            Args:
                individual (List[float]): The individual to simulate.
                variable_names (List[str]): Variable names for simulation parameters.
                filepath (str): Path to the scenario file.
                sim_time (float): Total simulation time.
                time_step (float): Time step for the simulation updates.

            Returns:
                SimulationOutput: The simulation result.
            """
            
            max_retries = 3
            last_exception = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    if attempt > 1:
                        log.info(f"Retry attempt {attempt}/{max_retries}")
                        rospy.sleep(2.0)  # Wait between retries
                    
                    sim_time = int(math.ceil(sim_time))
                    print("configured sim_time: ", sim_time)
                    
                    global helper
                    helper = AutowareHelper()

                    scenario_params = Utility.extract_info_from_scenario_file(filepath)
                    print("[simulate_ros_worker] Scenario : {}".format(scenario_params))

                    helper.clear_state() # slow car

                    helper.create_object(scenario_params["adversary"]["position"]["x"], scenario_params["adversary"]["position"]["y"] * -1, 
                                            scenario_params["adversary"]["position"]["z"], ObjectType.DISTANCE, scenario_params["parameters"]["PedDist"], 
                                            heading= scenario_params["adversary"]["position"]["h"],
                                            speed=scenario_params["parameters"]["PedSpeed"]
                                            ) # scenario passthrough


                    helper.publish_initial_pose(scenario_params[CAR_NAME]["position"]["x"], 
                                                scenario_params[CAR_NAME]["position"]["y"] * -1, 
                                                scenario_params[CAR_NAME]["position"]["h"])
                    
                    # Set maximal speed based on scenario
                    rospy.set_param("/planning/lanelet2_global_planner/custom_speed_limit", scenario_params["parameters"]["EgoSpeed"] * 3.6)
                    # Set velocity to 0 first
                    helper.publish_velocity(0, 0)
                    helper.publish_goal(goal[0], goal[1])
                    
                    # helper.publish_velocity(scenario_params["parameters"]["EgoSpeed"], 0)

                    print("[simulate_ros_worker] sim_time is: ", sim_time)
                    start_time = time.time()

                    # clear all variables left from previous scenario
                    helper.start_object_publisher(sim_time)

                    # helper.publish_velocity(0, 0)

                    print("[simulate_ros_worker] Object Publisher started")

                    # Retrieve simulation results
                    ego_informations, obstacle_informations = helper.get_results() # target might differ from ego_goal
                
                    helper.clear_state()

                    for key in ego_informations:
                        print(f"Ego {key} entries: {len(ego_informations[key])}")


                    print("[simulate_ros_worker] Results received")
                    # We need to change later. For now, we are using one obstacle to generate simout.
                    # But the code supports to report more than one obstacles at the same time.
                    obstacle_information = obstacle_informations
                                    
                    result = {
                        "simTime": time.time() - start_time,
                        "times": ego_informations["timestamp"],
                        "timestamps" : {"ego":  ego_informations["timestamp"],
                                        "adversary":  obstacle_information["timestamp"]},
                        "location": {
                                    "ego": ego_informations["location"],
                                    "adversary": obstacle_information["location"]
                                    },

                        "velocity": {
                                    "ego": ego_informations["velocity"],
                                    "adversary": obstacle_information["velocity"]
                                    },

                        "speed": {
                                    "ego": ego_informations["speed"],
                                    "adversary": obstacle_information["speed"]
                                },

                        "acceleration": {
                                    "ego": ego_informations["acceleration"],
                                    "adversary": obstacle_information["acceleration"]
                                },

                        "yaw": {
                                    "ego": ego_informations["yaw"],
                                    "adversary": obstacle_information["yaw"]
                                        },

                        "collisions": [],
                        "actors" : {
                            "ego": "ego",
                            "adversary": "adversary",
                            "vehicles" : [],
                            "pedestrians" : []
                            },
                        "otherParams": {
                        }
                    }
                    
                    return SimulationOutput.from_json(json.dumps(result))
                
                except Exception as e:
                    last_exception = e
                    log.info(f"[simulate_ros_worker] Exception occurred in ros worker (attempt {attempt}/{max_retries}).")
                    log.info(e)
                    traceback.print_exc(sys.stdout)
                    
                    # Cleanup after failed attempt
                    try:
                        call_cancel_route_service()
                        helper.object_operations.stop_object_publisher()
                        helper.clear_state()
                    except Exception as cleanup_error:
                        log.info("Could not stop object publisher during cleanup.")
                    
                    # If last attempt, exit
                    if attempt == max_retries:
                        log.error(f"All {max_retries} attempts failed. Exiting.")
                        sys.exit(1)
                
                finally:
                    try:
                        call_cancel_route_service()
                        print("Simulation finished. Stopping object publisher and canceling route.")
                        helper.object_operations.stop_object_publisher()
                    except Exception as e:
                        log.info("Could not stop object publisher.")


if __name__ == "__main__":  
    try:
        # Use JSON for safe parsing
        individual = args.individual
        variable_names = args.variable_names

        # Run simulation
        res = ROSController.simulate_single(
            individual,
            variable_names,
            args.xosc_path,
            args.sim_time,
            args.time_step
        )
        print("[simulate_ros_worker] result ready")

        # # Save results
        with open("/home/opensbt/simulations/simple_sim/simulation_results.json", "w") as f:
            json.dump(res.to_dict(), f, indent=4, allow_nan=True, cls=NumpyEncoder)
        
        sys.exit(0)

    except Exception:
        print("[simulate_ros_worker] Exception ocurred.")
        traceback.print_exc()
        # Optional cleanup on failure
        ROSController.kill_ros_processes()
        sys.exit(1)
