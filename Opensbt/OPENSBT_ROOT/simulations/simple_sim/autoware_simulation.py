import pymoo

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.algorithm.nsga2_optimizer import *
from opensbt.algorithm.pso_optimizer import *

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

DEBUG_MODE = False 
SCENARIO_DIR = "./tmpscenarios"
CAR_NAME = "ego_vehicle"

TIME_WAIT_RETRY_AFTER_RESTART = 4

class AutowareSimulation(Simulator):
    """
    A simulation class for running scenarios using the Autoware.
    
    Attributes:
        TIME_STEP (float): The simulation time step interval.
        SCENARIO_TIME_LIMIT (int): Maximum duration for a scenario (in seconds).
        helper (AutowareHelper): Helper object for interfacing with Autoware.
        obstacle_duration (float): Duration for which an obstacle is active in the simulation.
    """

    TIME_STEP = 1
    helper = None

    @staticmethod
    def simulate(list_individuals: List[Individual], variable_names: List[str], scenario_path: str, sim_time: float, 
                 time_step: float = TIME_STEP, do_visualize: bool = False) -> List[SimulationOutput]:
        """
        Simulate a batch of individuals using the specified scenario.

        Args:
            list_individuals (List[Individual]): List of individuals to simulate.
            variable_names (List[str]): Variable names for simulation parameters.
            scenario_path (str): Path to the scenario file.
            sim_time (float): Total simulation time for each scenario.
            time_step (float, optional): Time step for the simulation. Defaults to TIME_STEP.
            do_visualize (bool, optional): Whether to visualize the simulation. Defaults to False.

        Returns:
            List[SimulationOutput]: List of simulation results for each individual scenario.
        """
        #sim_time = 7 # in sec; for now we override
        time_step = 1 # for now we override 

        MAX_RETRIES = 5
        attempt = 0
        results = []
        while attempt < MAX_RETRIES:
            try:
                attempt = attempt + 1
                global ros_proc
                   
                if DEBUG_MODE:
                    print("The program is running in debug mode.")

                for individual in list_individuals:
                    instance_values = [v for v in zip(variable_names, individual)]
                    logging.info(instance_values)
                    AutowareSimulation.create_scenario_instance_xosc(
                        scenario_path,
                        dict(instance_values),
                        outfolder=SCENARIO_DIR
                    )

                with os.scandir(SCENARIO_DIR) as entries:
                    for entry in entries:
                        if entry.name.endswith('.xosc') and entry.is_file():
                            print(f"Running scenario {entry.name}")
                            # Prepare the command to run `simulate_ros_worker.py` with required arguments
                            command = [
                                "python3", "/home/opensbt/simulations/simple_sim/simulate_ros_worker.py",
                                "--individual"] + [str(v) for v in individual.tolist()] + [
                                "--variable_names"] + variable_names + [
                                "--xosc_path", entry.path,
                                "--sim_time", str(sim_time),
                                "--time_step", str(time_step)
                            ]
                            # Start the subprocess to run the ROS simulation worker script
                            ros_proc = subprocess.Popen(
                                command,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )

                            # Wait for the subprocess to complete and capture output
                            stdout, stderr = ros_proc.communicate()

                            # Check the return code and handle errors if any
                            if ros_proc.returncode != 0:
                                print("Subprocess failed with error:")
                                print("STDERR:\n", stderr)      
                                print("STDOUT:\n", stdout)
                                raise subprocess.CalledProcessError(
                                    returncode=ros_proc.returncode,
                                    cmd=command,
                                    output=stdout,
                                    stderr=stderr
                                )
                            else:
                                print("Subprocess completed successfully:")
                                print("STDOUT:\n", stdout)

                            # time.sleep(5)  # Wait for the ROS node to start
                            ros_proc.wait()  # This waits for the process to finish before continuing
        
                            result_file = "/home/opensbt/simulations/simple_sim/simulation_results.json"  
                            with open(result_file, "r") as f:
                                json_dict = json.load(f)  # Load the file content as a Python dictionary
                                results.append(SimulationOutput(**json_dict))  # Directly pass the dictionary to the class constructor

                            os.remove(result_file)

                            print(f"[autoware_simulation] Scenario {entry.name} is done")
                # If everything succeeded, break the loop
                break

            except Exception as e:
                traceback.print_exc()

                logging.warning(f"An error occurred. {attempt+1} attempt of simulation..")
                
                restart_lofi()

                # Optional delay to prevent tight retry loop
                time.sleep(TIME_WAIT_RETRY_AFTER_RESTART)
            finally:
                logging.info("++ Removing temporary scenarios ++")
                file_list = [f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc")]
                for f in file_list:
                    os.remove(os.path.join(SCENARIO_DIR, f))


        return results


    ''' Replace parameter values in parameter declaration section by provided parameters '''
    @staticmethod
    def create_scenario_instance_xosc(filename: str, values_dict: Dict, outfolder=None):
        import xml.etree.ElementTree as ET
        import os
        from pathlib import Path
        
        xml_tree = ET.parse(filename)

        # Update parameter values
        parameters = xml_tree.find('ParameterDeclarations')
        if parameters is not None:
            for name, value in values_dict.items():
                print(values_dict)
                for parameter in parameters:
                    if parameter.attrib.get("name") == name:
                        parameter.attrib["value"] = str(value)
        
        # Handle output folder
        if outfolder is not None:
            Path(outfolder).mkdir(parents=True, exist_ok=True)
            filename = os.path.join(outfolder, os.path.basename(filename))
        
        split_filename = os.path.splitext(filename)
        new_path_prefix = split_filename[0]
        ending = split_filename[1]

        suffix = ""
        for k, v in values_dict.items():
            suffix += f"_{v}"
        
        # Extract description from FileHeader
        file_header = xml_tree.find('FileHeader')
        if file_header is not None:
            description = file_header.attrib.get("description", "")
            if description.startswith("CARLA:"):
                scenario_name = description.split("CARLA:")[1].strip()
                file_header.attrib["description"] = f"CARLA:{scenario_name}{suffix}"
            else:
                scenario_name = "UnknownScenario"
        else:
            scenario_name = "UnknownScenario"

        new_file_name = f"{new_path_prefix}{suffix}{ending}"
        xml_tree.write(new_file_name)

        return new_file_name
