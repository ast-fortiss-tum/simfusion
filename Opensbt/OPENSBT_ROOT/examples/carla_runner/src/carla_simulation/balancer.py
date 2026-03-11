# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import time
import multiprocessing as mp
import logging

import carla
import matplotlib.pyplot as plt
from examples.carla_runner.src.carla_simulation.runner import Runner
from examples.carla_runner.src.carla_simulation.metrics.raw import RawData
from examples.carla_runner.src.carla_simulation.controllers.npc import NpcAgent

NETWORK_NAME = 'carla-network'
SCENARIO_TIMEOUT_SINGLE = int(os.environ.get('CARLA_SCENARIO_TIMEOUT_SINGLE', 80)) # TIMEOUT FOR A SINGLE SCENARIO, TO PREVENT HANGS IN CASE OF BUGS IN THE SIMULATION OR UNEXPECTED SCENARIO BEHAVIOR. SHOULD BE LESS THAN SCENARIO_TIMEOUT TO ALLOW FOR RETRIES.

logger = logging.getLogger(__name__)

def run_scenarios(scenario_dir, 
                  visualization_flag=False,
                  max_substep_delta_time=0.01,
                  max_substeps=10,
                  resolution=0.1):
    
    # Create a list of servers of docker containers
    #client = docker.from_env()
    #network = client.networks.get(NETWORK_NAME)
    #servers = [
    #    container.attrs[
    #        'NetworkSettings'
    #    ][
    #        'Networks'
    #    ][
    #        NETWORK_NAME
    #    ][
    #        'IPAddress'
    #    ]
    #    for container in network.containers
    #]

    servers = ['127.0.0.1']

    scenario_map = 'Town01'
    for server in servers:
        client = carla.Client(server, 2000)
        client.set_timeout(10.0)  # Set client timeout
        server_map = client.get_world().get_map().name.split('/')[-1]
        if server_map != scenario_map:
            logger.info(f"Loading world {scenario_map}...")
            client.load_world(scenario_map)

    scenarios = mp.JoinableQueue()
    scenario_count = 0
    with os.scandir(scenario_dir) as entries:
        for entry in entries:
            if entry.name.endswith('.xosc') and entry.is_file():
                scenarios.put(entry.name)
                scenario_count += 1
    
    if scenario_count == 0:
        logger.warning(f"No scenarios found in {scenario_dir}")
        return []

    logger.info(f"Found {scenario_count} scenario(s) to execute")

    with mp.Manager() as manager:
        start_time = time.time()

        evaluations = manager.list()
        processes = []
        
        try:
            for server in servers:
                runner = Runner(
                    server,
                    NpcAgent,
                    RawData,
                    rendering=visualization_flag,
                    max_substep_delta_time=max_substep_delta_time,
                    max_substeps=max_substeps,
                    resolution=resolution
                )
                process = mp.Process(
                    target=runner.run,
                    args=(scenario_dir, scenarios, evaluations),
                    daemon=True
                )
                process.start()
                processes.append(process)
                logger.info(f"Started runner process (PID: {process.pid}) for server {server}")

            SCENARIO_TIMEOUT = SCENARIO_TIMEOUT_SINGLE * scenario_count
            # Wait for scenarios with timeout using polling
            logger.info(f"Waiting for scenarios to complete (timeout: {SCENARIO_TIMEOUT}s)...")
            
            poll_interval = 5  # Check every second
            elapsed = 0
            
            while elapsed < SCENARIO_TIMEOUT:
                # Check if all processes have finished
                alive_processes = [p for p in processes if p.is_alive()]
                
                if not alive_processes:
                    # All processes finished successfully
                    logger.info("All processes completed successfully")
                    break
                
                time.sleep(poll_interval)
                elapsed += poll_interval
                
                # Optional: Log progress every 10 seconds
                if elapsed % 10 == 0:
                    logger.info(f"Still running... ({elapsed}s elapsed, {len(alive_processes)} process(es) active)")
            
            else:
                # Timeout occurred (while loop completed without break)
                alive_processes = [p for p in processes if p.is_alive()]
                
                if alive_processes:
                    logger.warning(f"Scenario execution timed out after {SCENARIO_TIMEOUT}s")
                    logger.warning(f"{len(alive_processes)} process(es) still running, terminating...")
                    
                    # Terminate processes gracefully
                    for process in alive_processes:
                        logger.info(f"Terminating process PID {process.pid}")
                        process.terminate()
                    
                    # Wait up to 5 seconds for graceful termination
                    time.sleep(5)
                    
                    # Force kill if still alive
                    still_alive = [p for p in alive_processes if p.is_alive()]
                    if still_alive:
                        logger.warning(f"{len(still_alive)} process(es) did not terminate, force killing...")
                        for process in still_alive:
                            logger.warning(f"Force killing process PID {process.pid}")
                            process.kill()
                            process.join(timeout=2)
                    
                    # Raise exception to trigger retry mechanism
                    raise TimeoutError(f"Scenario execution timed out after {SCENARIO_TIMEOUT} seconds")

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, cleaning up processes...")
            
            # Terminate all running processes
            alive_processes = [p for p in processes if p.is_alive()]
            for process in alive_processes:
                logger.info(f"Terminating process PID {process.pid}")
                process.terminate()
            
            # Wait up to 5 seconds for graceful termination
            time.sleep(5)
            
            # Force kill if still alive
            still_alive = [p for p in alive_processes if p.is_alive()]
            if still_alive:
                logger.warning(f"{len(still_alive)} process(es) did not terminate, force killing...")
                for process in still_alive:
                    logger.warning(f"Force killing process PID {process.pid}")
                    process.kill()
                    process.join(timeout=2)
            
            # Re-raise to propagate to the retry mechanism
            raise

        stop_time = time.time()
        elapsed_time = stop_time - start_time

        logger.info(f'Scenario execution completed in {elapsed_time:.2f}s')
        
        # Ensure all processes have finished
        for process in processes:
            if process.is_alive():
                process.join(timeout=1)

        results = list(evaluations)
        logger.info(f"Collected {len(results)} evaluation result(s)")
        
        return results