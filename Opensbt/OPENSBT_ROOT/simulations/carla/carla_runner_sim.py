from pathlib import Path
import sys
from typing import List, Dict
from examples.carla_runner.src.carla_simulation import balancer
from opensbt.simulation.simulator import Simulator, SimulationOutput
from simulations.carla.restart_hifi import restart_hifi
import logging
import json
import os
import traceback
import time
import docker
from simulations.config import max_substep_delta_time, resolution, max_substeps

SCENARIO_DIR = "./tmpscenarios"

TIME_WAIT_RETRY_AFTER_RESTART = 10
TIME_WAIT_RETRY_FOR_CARLA = 10

class CarlaRunnerSimulator(Simulator):

    simulation_counter = 0

    ''' Simulates a set of scenarios and returns the output '''
    @staticmethod
    def simulate(
        list_individuals,
        variable_names,
        scenario_path: str,
        sim_time: float,
        time_step:float,
        do_visualize:bool = False
    ) -> List[SimulationOutput]:

        xosc = scenario_path

        MAX_RETRIES = 5
        attempt = 0
        results = []
        
        print("Simulations completed" + str(CarlaRunnerSimulator.simulation_counter))

        while attempt < MAX_RETRIES:
            try:
                attempt = attempt + 1

                for ind in list_individuals:
                    logging.info("provided following values:")
                    instance_values = [v for v in zip(variable_names,ind)]
                    logging.info(instance_values)
                    CarlaRunnerSimulator.create_scenario_instance_xosc(xosc, dict(instance_values), outfolder=SCENARIO_DIR)
                
                logging.info("++ running scenarios with carla ++ ")
                logging.info("++ CARLA Config Parameters: ++")
                logging.info(f"  max_substep_delta_time = {max_substep_delta_time}")
                logging.info(f"  max_substeps           = {max_substeps}")
                logging.info(f"  resolution             = {resolution}")


                outs = balancer.run_scenarios(scenario_dir=SCENARIO_DIR, 
                                              visualization_flag=do_visualize,
                                              max_substep_delta_time=max_substep_delta_time,
                                              max_substeps=max_substeps,
                                              resolution=resolution)
                results = []
                for out in outs:
                    last_out = out
                    simout = SimulationOutput.from_json(json.dumps(out))
                    simout.otherParams["isCollision"] = (len(simout.collisions) != 0)
                    results.append(simout)
                
                CarlaRunnerSimulator.simulation_counter += len(list_individuals)
                
                break
            except KeyboardInterrupt:
                traceback.print_exc()

                logging.warning(f"An error occurred. {attempt+1} attempt of simulation..")
                
                CarlaRunnerSimulator.restart_carla()
                
                time.sleep(TIME_WAIT_RETRY_FOR_CARLA)
        
                restart_hifi()

                # Optional delay to prevent tight retry loop
                time.sleep(TIME_WAIT_RETRY_AFTER_RESTART)
            except Exception as e:
                traceback.print_exc()

                logging.warning(f"An error occurred. {attempt+1} attempt of simulation..")
                
                CarlaRunnerSimulator.restart_carla()
                
                time.sleep(TIME_WAIT_RETRY_FOR_CARLA)
        
                restart_hifi()

                # Optional delay to prevent tight retry loop
                time.sleep(TIME_WAIT_RETRY_AFTER_RESTART)
            finally:
                logging.info("++ removing temporary scenarios ++")
                file_list = [ f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xosc") ]
                for f in file_list:
                    os.remove(os.path.join(SCENARIO_DIR, f))

        return results
    @staticmethod
    def restart_carla():
        container_name = "carla_sim"
        image_name = "prathap/carla_sim:v0.9.13"
        command = ["bash", "./CarlaUE4.sh", "-RenderOffScreen", "--world-port=2000"]

        volumes = {
            "/tmp/.X11-unix": {'bind': '/tmp/.X11-unix', 'mode': 'rw'},
            os.path.expanduser("~/.Xauthority"): {'bind': '/root/.Xauthority', 'mode': 'rw'}
        }

        environment = {
            "DISPLAY": os.getenv('DISPLAY', ':0'),  # Fixed: properly get DISPLAY variable
            "QT_X11_NO_MITSHM": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "XAUTHORITY": "/root/.Xauthority"
        }

        # GPU support - matching your docker-compose
        device_requests = [
            docker.types.DeviceRequest(
                count=1,
                capabilities=[['gpu']]
            )
        ]

        client = docker.from_env()

        # Stop and remove existing container
        try:
            container = client.containers.get(container_name)
            container.remove(force=True)
            print(f"Removed existing container '{container_name}'.")
        except docker.errors.NotFound:
            print(f"No existing container named '{container_name}' found.")
        except Exception as e:
            print(f"Error stopping/removing container '{container_name}': {e}")

        time.sleep(6)  # wait for cleanup

        # Start new container
        try:
            container = client.containers.run(
                image=image_name,
                command=command,
                name=container_name,
                volumes=volumes,
                environment=environment,
                network_mode="host",
                privileged=True,
                device_requests=device_requests,  # Added: GPU support
                detach=True,
                stdin_open=True,
                tty=True  # Added: helps with interactive containers
            )
            print(f"Started container '{container_name}'.")
            
            # Give container time to initialize before checking logs
            time.sleep(2)
            
            logs = container.logs(stdout=True, stderr=True, tail=50).decode()
            print(f"\n---- Container '{container_name}' Logs ----\n{logs}")
        except Exception as e:
            print(f"Error starting container '{container_name}': {e}")
            traceback.print_exc()  # Better error reporting
            
    # @staticmethod
    # def restart_carla():
    #     container_name = "carla_sim"
    #     image_name = "prathap/carla_sim:v0.9.13"
    #     command = ["bash", "./CarlaUE4.sh", "-RenderOffScreen", "--world-port=2000"]

    #     # ports = {
    #     #     '2000/tcp': 2000,
    #     #     '2001/tcp': 2001,
    #     #     '2002/tcp': 2002
    #     # }

    #     volumes = {
    #         "/tmp/.X11-unix": {'bind': '/tmp/.X11-unix', 'mode': 'rw'},
    #         "/home/user/.Xauthority": {'bind': '/root/.Xauthority', 'mode': 'rw'}
    #     }

    #     environment = {
    #         "DISPLAY": "unix$DISPLAY",
    #         "QT_X11_NO_MITSHM": "1",
    #         "NVIDIA_DRIVER_CAPABILITIES": "all",
    #         "XAUTHORITY": "/root/.Xauthority"
    #     }

    #     client = docker.from_env()

    #     # try:
    #     #     container = client.containers.get(container_name)
    #     #     container.stop(timeout=5)
    #     #     print(f"Stopped and removed existing container '{container_name}'.")
    #     # except docker.errors.NotFound:
    #     #     print(f"No existing container named '{container_name}' found.")
    #     # except Exception as e:
    #     #     print(f"Error stopping/removing container '{container_name}': {e}")

    #     try:
    #         container = client.containers.get(container_name)
    #         container.remove(force=True)
    #         print(f"Removed existing container '{container_name}'.")
    #     except docker.errors.NotFound:
    #         print(f"No existing container named '{container_name}' found.")
    #     except Exception as e:
    #         print(f"Error stopping/removing container '{container_name}': {e}")

    #     time.sleep(6)  # wait for cleanup

    #     try:
    #         container = client.containers.run(
    #             image=image_name,
    #             command=command,
    #             name=container_name,
    #             # ports=ports,
    #             volumes=volumes,
    #             environment=environment,
    #             network_mode="host",
    #             privileged=True,
    #             detach=True,
    #             stdin_open=True
    #         )
    #         print(f"Started container '{container_name}'.")
    #         logs = container.logs(stdout=True, stderr=True, tail=50).decode()
    #         print(f"\n---- Container '{container_name}' Logs ----\n{logs}")
    #     except Exception as e:
    #         print(f"Error starting container '{container_name}': {e}")
    #         input()

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