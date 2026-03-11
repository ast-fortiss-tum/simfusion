# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from pathlib import Path
import sys
import os
import subprocess
from run_canceler import call_cancel_route_service
import carla
import rospy

from srunner.scenarios.open_scenario import OpenScenario
from srunner.scenarioconfigs.openscenario_configuration import OpenScenarioConfiguration
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenario_manager import ScenarioManager
from simulations.multi_sim.set_goal_script import CarlaHelper
import time
import xml.etree.ElementTree as ET
from scenario_runner import ScenarioRunner
import threading

CAR_NAME = "ego_vehicle"

class Scenario:

    _xosc = None

    def __init__(self, xosc):
        self._xosc = xosc
        self.carla_helper = CarlaHelper()

    def check_carla_server_alive(self, simulator, timeout_seconds=2.0, host="localhost", port=2000):
        """Raise an exception if CARLA server is unresponsive."""
        client = simulator.get_client()
        client.set_timeout(timeout_seconds)
        try:
            version = client.get_server_version()
            rospy.loginfo(f"CARLA server is alive. Version: {version}")
        except RuntimeError as e:
            raise RuntimeError(f"CARLA server is not responding at {host}:{port} (timeout {timeout_seconds}s): {e}")

    def simulate(self, simulator, agent, recorder):
        client = simulator.get_client()

        world = simulator.get_world()

        CarlaDataProvider.set_client(client)
        CarlaDataProvider.set_world(world)

        config = OpenScenarioConfiguration(
            self._xosc,
            client,
            {}
        )

        CarlaDataProvider.set_traffic_manager_port(
            simulator.get_traffic_manager_port()
        )

        vehicles = self.get_existing_ego_vehicles(CAR_NAME, config)

        #controller = agent(simulator)

        scenario = OpenScenario(
            world,
            vehicles,
            config,
            self._xosc
        )

        recording = recorder.add_recording(
            Path(self._xosc.path).stem
        )

        scenario_manager = ScenarioManager()
        scenario_manager.load_scenario(scenario)   
        
        # surpress side influence
        self.set_traffic_lights_green(simulator)

        max_speed = self.extract_initial_speed_from_xosc(self._xosc, "EgoSpeed")
        self.carla_helper.set_maximal_speed(float(max_speed) * 3.6)
        
        self.run_carla_helper_script(0)
        
        time.sleep(1)

        # Run CARLA monitor in background
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self.check_carla_server_alive, args=(simulator, 1.0, stop_event))
        monitor_thread.start()

        try:
            client.start_recorder(
                recording,
                True
            )
        
            scenario_manager.run_scenario()
        except Exception as e:

            print("[CarlaRunner] Exception during scenario execution ocurred.")
            raise e
        finally:
            stop_event.set()
            monitor_thread.join()
            
     
        client.stop_recorder()
        #simulator.clean_up()

        for vehicle in vehicles:
            role_name = vehicle.attributes.get('role_name', '')
            rospy.loginfo(f"Checking vehicle ID {vehicle.id}, role_name: '{role_name}'")

            if role_name == CAR_NAME:
                rospy.loginfo(f"Applying hand brake and destroying vehicle ID {vehicle.id}")
                vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                time.sleep(1)
            if role_name != CAR_NAME and vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
            
                
        scenario_manager.stop_scenario()
        scenario_manager.cleanup()
        CarlaDataProvider.cleanup()


        # reset msgs

        call_cancel_route_service()

        
    def get_existing_ego_vehicles(self, expected_role_name, scenario_config):
        world = CarlaDataProvider.get_world()
        ego_vehicles = []

        while not ego_vehicles:
            vehicles = world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if vehicle.attributes['role_name'] == expected_role_name:
                    ego_vehicles.append(vehicle)
            
            if not ego_vehicles:
                time.sleep(1)

        for vehicle_config in scenario_config.ego_vehicles:
            if vehicle_config.rolename == expected_role_name:
                transform = vehicle_config.transform
                ego_vehicles[0].set_transform(transform)


        for vehicle in ego_vehicles:
            CarlaDataProvider.register_actor(vehicle)

        return ego_vehicles
    
    def extract_initial_speed_from_xosc(self, xosc_path, role_name):
        """
        Extracts initial speed defined in the OpenScenario file for a given role_name.
        """

        print("./tmpscenarios/" + xosc_path.name)
        xml_tree = ET.parse("./tmpscenarios/" + xosc_path.name)
        value = 0

        # Update parameter values
        parameters = xml_tree.find('ParameterDeclarations')
        if parameters is not None:
            for parameter in parameters:
                if parameter.attrib.get("name") == role_name:
                    value = parameter.attrib["value"]

        return value
    
    def run_carla_helper_script(self, velocity=5.0):
        script_path = os.path.abspath("./simulations/multi_sim/set_goal_script.py")
        process = subprocess.Popen(
            ["python3", script_path, str(velocity)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print(stdout.decode("utf-8"))
        else:
            print(stderr.decode("utf-8"))

    def set_traffic_lights_green(self, simulator):
        world = simulator.get_world()

        traffic_lights = world.get_actors().filter('traffic.traffic_light')

        for tl in traffic_lights:
            tl.set_state(carla.TrafficLightState.Green)
            tl.freeze(True)

        print(f"All traffic lights set to GREEN.")