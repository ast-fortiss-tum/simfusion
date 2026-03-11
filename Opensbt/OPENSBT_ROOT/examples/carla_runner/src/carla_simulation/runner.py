# Copyright (c) 2022 fortiss GmbH
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os

from examples.carla_runner.src.carla_simulation.simulator import Simulator
from examples.carla_runner.src.carla_simulation.scenario import Scenario
from examples.carla_runner.src.carla_simulation.recorder import Recorder


class Runner:

    _host_carla = None
    _port_carla = 2000
    _timeout_carla = 10
    _rendering_carla = False

    _recording_dir = '/tmp'
    _metrics_dir = '/tmp'

    _agent_class = None
    _metric_class = None

    def __init__(self, host, 
                agent, 
                metric, 
                rendering=False,
                resolution = 0.1,
                max_substep_delta_time = 0.01,
                max_substeps = 10):
        self._host_carla = host
        self._agent_class = agent
        self._metric_class = metric
        self._rendering_carla = rendering
        self._resolution_carla = resolution
        self._max_substep_delta_time = max_substep_delta_time
        self._max_substeps = max_substeps

    def run(self, directory, queue, evaluations):
        while not queue.empty():
            pattern = queue.get()

            simulator = self.get_simulator(
                self._host_carla,
                self._port_carla,
                self._timeout_carla,
                self._rendering_carla,
                self._resolution_carla,
                self._max_substep_delta_time,
                self._max_substeps
            )
            scenarios = self.get_scenarios(directory, pattern)
            recorder = self.get_recorder(self._recording_dir)
            evaluator = self.get_evaluator()
            agent = self.get_agent()

            for scenario in scenarios:
                scenario.simulate(simulator, agent, recorder)

            recordings = recorder.get_recordings()
                            
            for recording in recordings:
                evaluations.append(
                    evaluator.evaluate(
                        simulator,
                        recording
                    )
                )
                #os.remove(recording)


            queue.task_done()

    def get_simulator(self, 
                      host, 
                      port, 
                      timeout, 
                      rendering = True, 
                      resolution = 0.1,
                      max_substep_delta_time = 0.01,
                      max_substeps = 10):
        return Simulator(
            host = host,
            port = port,
            timeout = timeout,
            rendering = rendering,
            resolution = resolution,
            max_substep_delta_time = max_substep_delta_time,
            max_substeps= max_substeps
        )

    def get_scenarios(self, directory, pattern):
        scenarios = None
        with os.scandir(directory) as entries:
            scenarios = [
                Scenario(entry)
                    for entry in entries
                        if entry.name.endswith(pattern) and entry.is_file()
            ]
        return scenarios

    def get_evaluator(self):
        return self._metric_class()

    def get_agent(self):
        return self._agent_class

    def get_recorder(self, directory):
        return Recorder(directory)

    def get_recording_files(self, _recording_dir):
        import os
        from pathlib import Path
        
        if not Path(_recording_dir).exists():
            raise FileNotFoundError(f"The directory {_recording_dir} does not exist.")
        
        return [str(file) for file in Path(_recording_dir).rglob('*') if file.is_file()]
