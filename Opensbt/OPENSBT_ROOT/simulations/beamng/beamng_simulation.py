import json
from platform import processor
import shutil
import sys
import os
import time
import re
import logging
from datetime import datetime
from typing import List, Dict
import pandas as pd
import numpy as np
from pathlib import Path

from opensbt.model_ga import individual
from opensbt.simulation.simulator import Simulator, SimulationOutput
from opensbt.model_ga.individual import Individual
from opensbt.utils import evaluation
import pickle
import warnings

import simulations.commonroad.cr_beamng_utils as cr_utils

# sys.path.append("../")

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MultiDrive"))
)

from MultiDrive.cr_beamng_cosimulation.commands import simulate_with_beamng
from MultiDrive.cr_beamng_cosimulation.simulation_visualization.commands import *
from MultiDrive.cli import apply_monkey_path
from MultiDrive.cr_beamng_cosimulation.utils import LOGS_DIR
            
import os
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# LOGS_DIR = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
#     "executed-simulations-beamng",
# )
SCENARIO_DIR = "./scenarios/tmpscenarios"
OUTPUT_DIR = "./scenarios/output"

import time
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class StepProfiler:
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def now(self) -> float:
        return time.perf_counter()

    def mark(self, name: str, t0: float, t1: float, **meta):
        self.steps.append({"name": name, "sec": float(t1 - t0), **meta})


class BeamNGSimulator(Simulator):
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float = 10.0,
        do_visualize: bool = False,
        time_step: float = 0.1,
    ) -> List[SimulationOutput]:

        results = []
        run_root = Path(os.environ["OPENSBT_RUN_ROOT"]).resolve()
        run_id   = os.environ.get("OPENSBT_RUN_ID", "noid")
        LOGS_DIR = str(run_root / f"executed-simulations-beamng-{run_id}")
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Create scenario instances for each individual
        for ind in list_individuals:
            prof = StepProfiler()
            t_total0 = prof.now()

            param_dict = dict(zip(variable_names, ind))
            logger.info(f"Simulating individual: {param_dict}")
            
            t0 = prof.now()
            scenario_file = cr_utils.create_scenario_instance(
                scenario_path, param_dict, outfolder=SCENARIO_DIR
            )
            t1 = prof.now()
            prof.mark("create_scenario_instance", t0, t1)

            scenario_name = Path(scenario_file).stem

            scenario_folder = os.path.join(LOGS_DIR, scenario_name)
            
            print("scenario foler:", scenario_folder)
  
            ctx = dict()
            ctx["output_folder"] = LOGS_DIR
            ctx["verbose"] = False
            ctx["start_time"] = time.time()
            ctx["scenario_name"] = f"{scenario_name}"
            ctx["scenario_folder"] = scenario_folder
            ctx["goal_check"] = {
                "x": float(param_dict["goal_center_x"]),
                "y": float(param_dict["goal_center_y"]),
                "threshold": 1.0,
                "ego_id": 3001
            }

            t0 = prof.now()
            apply_monkey_path()
            

            max_retries = 5
            for attempt in range(1, max_retries + 1):
                try:
                    logging.info(f"Main simulation loop retry {attempt}/{max_retries}")
                    
                    start_time = time.time()  # Start timing
                    _, evaluation = simulate_with_beamng(ctx, scenario_file, "car")
                    
                    if evaluation is None:
                        raise Exception("Error happened during simulation.")
                    
                    end_time = time.time()    # End timing
                    logging.info(f"Simulation time: {end_time - start_time}")

                    # If successful, exit the loop
                    break
                except Exception as e:
                    # Delete existing folder if it exists (because of reruns)
                    if os.path.exists(scenario_folder) and os.path.isdir(scenario_folder):
                        shutil.rmtree(scenario_folder)
                        print("removing scenario foler")
                        
                    from kill_beamng import kill_beamng
                    kill_beamng()
                    
                    print(f"Attempt {attempt} failed: {e}")
                    time.sleep(1)  # optional delay between retries
            else:
                print(f"simulate_with_beamng failed after {max_retries} attempts.")     
                       
            t1 = prof.now()
            prof.mark("simulate_with_beamng", t0, t1)

            t0 = prof.now()
            # generate path of the file to be opened
            file_path = f"{ctx['scenario_folder']}/cosimulated_scenario_with_bng.xml"
            pp_src = os.path.join(ctx["scenario_folder"], "scenario_to_cosimulate_with_bng.xml")
            # plot_simulate(ctx, pp_src, file_path, "BeamNG Simulation", None, None, 10, "png")
            # animate_simulate(ctx, os.path.join(ctx["scenario_folder"], "plots"))
            t1 = prof.now()
            prof.mark("plot_and_animate", t0, t1)

            t0 = prof.now()
            plots_dir = os.path.join(ctx["scenario_folder"], "plots")

            try:
                shutil.rmtree(plots_dir)
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.warning(f"Could not delete plots folder {plots_dir}: {e}")
            t1 = prof.now()
            prof.mark("cleanup_plots", t0, t1)

            t0 = prof.now()
            # Read simulation results
            logs_csv_path = os.path.join(
                ctx["scenario_folder"], "logs", "3001", "logs.csv"
            )
            df = pd.read_csv(logs_csv_path, sep=";")
            t1 = prof.now()
            prof.mark("read_ego_logs_csv", t0, t1)

            t0 = prof.now()
            trajectory_data = {}
            for _, row in df.iterrows():
                trajectory_number = row["trajectory_number"]
                trajectory_data[trajectory_number] = {
                    "x_positions": cr_utils.parse_to_array(
                        row["x_position_vehicle_m"]
                    ),
                    "y_positions": cr_utils.parse_to_array(
                        row["y_position_vehicle_m"]
                    ),
                    "velocities": cr_utils.parse_to_array(
                        row["velocities_mps"]
                    ),
                    "accelerations": cr_utils.parse_to_array(
                        row["accelerations_mps2"]
                    ),
                    "yaw_orientations_rad": cr_utils.parse_to_array(
                        row["theta_orientations_rad"]
                    ),
                    "trajectory_ids": cr_utils.parse_to_array(
                        row["trajectory_number"]
                    ),
                }


            # Collect first values from each trajectory to create time series
            vel = []
            acc = []
            yaw = []
            trajectory_steps = []
            
            # Sort trajectory numbers to ensure consistent order
            sorted_trajectory_numbers = sorted(trajectory_data.keys())
            
            for traj_num in sorted_trajectory_numbers:
                data = trajectory_data[traj_num]
                # Take first value from each trajectory's arrays
                if len(data["velocities"]) > 0:
                    vel.append(data["velocities"][0])
                if len(data["accelerations"]) > 0:
                    acc.append(data["accelerations"][0])
                if len(data["yaw_orientations_rad"]) > 0:
                    yaw.append(data["yaw_orientations_rad"][0])
                trajectory_steps.append(traj_num)

            steps = trajectory_steps  # Use actual timestep values from CSV

            # Coordinates
            all_trajectories_3d = []
            for _, data in trajectory_data.items():
                x_vals = data["x_positions"]
                y_vals = data["y_positions"]
                if len(x_vals) == len(y_vals):
                    coords = [(x, y, 0) for x, y in zip(x_vals, y_vals)]
                    all_trajectories_3d.extend(coords)

            global_max_len = max(len(vel), len(acc), len(yaw))
            
            logs_root = os.path.join(ctx["scenario_folder"], "logs")
            per_adv = {}

            for oid in sorted(os.listdir(logs_root)):
                if oid == "3001":
                    continue
                oid_path = os.path.join(logs_root, oid)
                csv_path = os.path.join(oid_path, "logs.csv")
                if not (os.path.isdir(oid_path) and os.path.exists(csv_path)):
                    continue

                df_dynamic = pd.read_csv(csv_path, sep=";")

                dynamic_trajectory_data = {}
                for _, row in df_dynamic.iterrows():
                    tnum = row["trajectory_number"]
                    dynamic_trajectory_data[tnum] = {
                        "x_positions": cr_utils.parse_to_array(row["x_position_vehicle_m"]),
                        "y_positions": cr_utils.parse_to_array(row["y_position_vehicle_m"]),
                        "velocities":  cr_utils.parse_to_array(row["velocities_mps"]),
                        "accelerations": cr_utils.parse_to_array(row["accelerations_mps2"]),
                        "yaw_orientations_rad": cr_utils.parse_to_array(row["theta_orientations_rad"]),
                        "trajectory_ids": cr_utils.parse_to_array(row["trajectory_number"]),
                    }

                # Initialize dynamic obstacle data
                d_vel, d_acc, d_yaw, d_steps = [], [], [], []
                for tnum in sorted(dynamic_trajectory_data.keys()):
                    dat = dynamic_trajectory_data[tnum]
                    if len(dat["velocities"]) > 0:
                        d_vel.append(dat["velocities"][0])
                    if len(dat["accelerations"]) > 0:
                        d_acc.append(dat["accelerations"][0])
                    if len(dat["yaw_orientations_rad"]) > 0:
                        d_yaw.append(dat["yaw_orientations_rad"][0])
                    d_steps.append(tnum)

                d_traj3d = []
                for dat in dynamic_trajectory_data.values():
                    xs, ys = dat["x_positions"], dat["y_positions"]
                    if len(xs) == len(ys):
                        d_traj3d.extend([(x, y, 0) for x, y in zip(xs, ys)])

                per_adv[oid] = {
                    "steps": d_steps,
                    "vel":   d_vel,
                    "acc":   d_acc,
                    "yaw":   d_yaw,
                    "traj3d": d_traj3d,
                }

            timestamps_dict = {"ego": steps}
            location_dict   = {"ego": all_trajectories_3d}
            velocity_dict   = {"ego": vel}
            speed_dict      = {"ego": vel}
            acceleration_dict = {"ego": acc}
            yaw_dict        = {"ego": yaw}

            for oid, d in per_adv.items():
                key = f"adversary_{oid}"
                d_steps = cr_utils.pad_array(d["steps"], global_max_len)
                d_vel   = cr_utils.pad_array(d["vel"],   global_max_len)
                d_acc   = cr_utils.pad_array(d["acc"],   global_max_len)
                d_yaw   = cr_utils.pad_array(d["yaw"],   global_max_len)
                d_traj3d= cr_utils.pad_trajectory_3d(d["traj3d"], global_max_len)

                timestamps_dict[key]   = d_steps
                location_dict[key]     = d_traj3d
                velocity_dict[key]     = d_vel
                speed_dict[key]        = d_vel
                acceleration_dict[key] = d_acc
                yaw_dict[key]          = d_yaw

            # Add goal position to otherParams if available
            other_params = {"simulator": "BeamNGSimulator"}
            if "goal_center_x" in param_dict:
                other_params["goal_center_x"] = param_dict["goal_center_x"]
            if "goal_center_y" in param_dict:
                other_params["goal_center_y"] = param_dict["goal_center_y"]
            if "num_npc" in param_dict:
                other_params["num_npc"] = param_dict["num_npc"]
            if "lane_count" in param_dict:
                other_params["lane_count"] = param_dict["lane_count"]

            t1 = prof.now()
            prof.mark("parse_logs_and_build_dicts", t0, t1)

            t0 = prof.now()
            simout = SimulationOutput(
                simTime=time.time() - ctx["start_time"],
                times=steps,
                timestamps=timestamps_dict,
                location=location_dict,
                velocity=velocity_dict,
                speed=speed_dict,
                acceleration=acceleration_dict,
                yaw=yaw_dict,
                collisions=[],
                actors={
                    "ego": "ego",
                    "vehicles": ["ego"],
                    "pedestrians": [],
                },
                otherParams=other_params,
            )
            t1 = prof.now()
            prof.mark("build_simulation_output", t0, t1)

            results.append(simout)
            logger.info(f"Simulated individual: {param_dict}")

            t_total1 = prof.now()
            prof.mark("TOTAL_wall_time", t_total0, t_total1)

            report = {
                "scenario_name": scenario_name,
                "params": param_dict,
                "steps": prof.steps,
            }
            with open(os.path.join(ctx["scenario_folder"], "timings.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            print(f"Simulation time for individual: {simout.simTime:.2f} sec")

        logging.info("++ Removing temporary scenarios ++")
        file_list = [f for f in os.listdir(SCENARIO_DIR) if f.endswith(".xml")]
        for f in file_list:
            os.remove(os.path.join(SCENARIO_DIR, f))

        return results

