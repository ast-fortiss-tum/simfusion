from dataclasses import dataclass, is_dataclass, asdict
from typing import Dict, List
from pymoo.core.individual import Individual
from abc import ABC, abstractstaticmethod
from opensbt.utils.encoder_utils import NumpyEncoder

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

def recursive_to_dict(obj):
    if is_dataclass(obj):
        return {k: recursive_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, dict):
        return {k: recursive_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_dict(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@dataclass
class SimulationOutput(object):
    """
    Class represents data output after execution of a simulation. Example JSON representation:
    {
        "simTime": 10,
        "times": [0.0, 0.1, ..., 10.0],
        "location": {"ego": [[x, y], ...], "adversary": [[x, y], ...]},
        ...
        "otherParams": {"car_width": 3, "car_length": 5}
    }
    """
    simTime: float
    times: List
    timestamps: Dict
    location: Dict
    velocity: Dict
    speed: Dict
    acceleration: Dict
    yaw: Dict
    collisions: List
    actors: Dict
    otherParams: Dict

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict:
        return recursive_to_dict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=True, indent=4, cls=NumpyEncoder)

    @classmethod
    def from_json(cls, json_str: str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

class Simulator(ABC):
    """Base class to be inherited and implemented by a concrete simulator in OpenSBT."""

    @abstractstaticmethod
    def simulate(list_individuals: List[Individual],
                 variable_names: List[str],
                 scenario_path: str,
                 sim_time: float = 10,
                 time_step: float = 0.01,
                 do_visualize: bool = True) -> List[SimulationOutput]:
        """
        Simulates scenarios using the scenario path, variable names, and individual values.

        :param list_individuals: List of individuals representing scenarios.
        :param variable_names: The scenario variables.
        :param scenario_path: Path to the abstract/logical scenario.
        :param sim_time: Total simulation time.
        :param time_step: Simulation time step.
        :param do_visualize: Whether to visualize the simulation.
        :return: List of simulation outputs.
        """
        pass