import datetime
import random
import shutil
import time
import argparse
import os
import sys
import math
import copy
import csv
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pymoo
import dill

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from flaky.utils import plot_boxplot, plot_velocities

from opensbt.problem.adas_problem import ADASProblem
from opensbt.simulation.simulator import SimulationOutput
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.utils.duplicates import duplicate_free
from opensbt.utils import geometric
from opensbt.config import OUTPUT_PRECISION

from opensbt.visualization.combined import read_pf_single
from opensbt.visualization.visualizer import (
    if_none_iterator,
    write_diversity,
)

from flaky.utils import write_simout_flaky, get_incremented_filename

from predictor.analyse_data import plot_design_space, plot_objective_space

from simulations.carla.carla_runner_sim import CarlaRunnerSimulator
from simulations.simple_sim.autoware_simulation import AutowareSimulation
from simulations.simple_sim import *
from simulations.carla import *
from simulations.utils import *
from simulations.config import *
import wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Flakiness analysis for CARLA-based ADAS simulations."
    )

    parser.add_argument("--n_repeat", type=int, default=1)
    parser.add_argument("--n_samples_corner", type=int, default=1)

    parser.add_argument("--test_cases_path", type=str, required=True)
    parser.add_argument("--results_folder", type=str, default="./results_flaky/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--scenario_file", type=str, default=os.path.join(os.getcwd(), "scenarios",scenario_file))

    parser.add_argument(
        "--corner_ids",
        type=int,
        nargs="+",
        default=None,
        help="Corner IDs to sample (0–7). If omitted or empty, sampling is done from the full population."
    )

    parser.add_argument(
        "--critical_only",
        action="store_true",
        help="If set, sample only from the critical population."
    )
    parser.add_argument(
        "--simulator",
        type=str,
        help="Simulator to use",
        default="hifi"
    )

    parser.add_argument("--no_wandb", action="store_true")

    return parser.parse_args()


def write_metadata(save_folder: str, args: argparse.Namespace) -> None:
    """
    Write experiment metadata (command-line arguments) to metadata.json.
    """
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
    }

    metadata_path = os.path.join(save_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_repeat = args.n_repeat
    n_samples_corner = args.n_samples_corner
    test_cases_path = args.test_cases_path
    results_folder = args.results_folder

    if args.simulator  == "hifi":
        simulate = CarlaRunnerSimulator.simulate
    elif args.simulator  == "lofi":
        simulate = AutowareSimulation.simulate
    else:
        raise ValueError("Not available sim.")

    problem = ADASProblem(
        problem_name=f"Flakiness_Carla_r-{n_repeat}_n-{n_samples_corner}",
        scenario_path=args.scenario_file,
        xl=xl,
        xu=xu,
        simulation_variables=simulation_variables,
        fitness_function=fitness_function,
        critical_function=critical_function,
        simulate_function=simulate,
        simulation_time=sim_time,
        sampling_time=sampling_time,
        approx_eval_time=10,
        do_visualize=True,
    )

    tags = [
        f"{k}:{v[:min(len(v), 44)] if isinstance(v, str) else v}"
        for k, v in vars(args).items()
    ]

    if not args.no_wandb:
        wandb.init(
            entity="lofi-hifi",
            project="autoware_flakiness",
            name=problem.problem_name,
            group=datetime.datetime.now().strftime("%d-%m-%Y"),
            tags=tags
        )
    else:
        wandb.init(mode="disabled")

    pop = read_pf_single(filename=test_cases_path, with_critical_column=True)
    pop = duplicate_free(pop)

    # ------------------------------------------------------------------
    # Corner definitions
    # ------------------------------------------------------------------
    corner_cubes = [
        {'x': (0.30, 1.41), 'y': (3.00, 4.59), 'z': (5.00, 9.50)},     # 0
        {'x': (0.30, 1.41), 'y': (3.00, 4.59), 'z': (15.50, 20.00)},  # 1
        {'x': (0.30, 1.41), 'y': (6.71, 8.30), 'z': (5.00, 9.50)},    # 2
        {'x': (0.30, 1.41), 'y': (6.71, 8.30), 'z': (15.50, 20.00)},  # 3
        {'x': (2.89, 4.00), 'y': (3.00, 4.59), 'z': (5.00, 9.50)},    # 4
        {'x': (2.89, 4.00), 'y': (3.00, 4.59), 'z': (15.50, 20.00)},  # 5
        {'x': (2.89, 4.00), 'y': (6.71, 8.30), 'z': (5.00, 9.50)},    # 6
        {'x': (2.89, 4.00), 'y': (6.71, 8.30), 'z': (15.50, 20.00)},  # 7
    ]

    # ------------------------------------------------------------------
    # Population selection
    # ------------------------------------------------------------------
    all_population = pop

    if args.critical_only:
        all_population, _ = all_population.divide_critical_non_critical()

    print("Total test size:", len(all_population))

    if not args.corner_ids:
        pops_corners = [list(all_population)]
        selected_corner_ids = ["all"]
    else:
        invalid = [cid for cid in args.corner_ids if cid < 0 or cid >= len(corner_cubes)]
        if invalid:
            raise ValueError(f"Invalid corner IDs: {invalid}")

        pops_corners = []
        selected_corner_ids = args.corner_ids

        for cid in selected_corner_ids:
            cube = corner_cubes[cid]
            inds_corner = []

            for ind in all_population:
                x, y, z = ind.get("X")[:3]
                if (
                    cube["x"][0] <= x <= cube["x"][1]
                    and cube["y"][0] <= y <= cube["y"][1]
                    and cube["z"][0] <= z <= cube["z"][1]
                ):
                    inds_corner.append(ind)

            pops_corners.append(inds_corner)

    # ------------------------------------------------------------------
    # Output folders
    # ------------------------------------------------------------------
    current_datetime_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_folder = os.path.join(results_folder, problem.problem_name, current_datetime_str)
    Path(save_folder).mkdir(exist_ok=True, parents=True)

    shutil.copy2(
        test_cases_path,
        os.path.join(save_folder, os.path.basename(test_cases_path)),
    )

    # Write metadata.json
    write_metadata(save_folder, args)

    F_all_total, CB_all_total = [], []

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    for batch_idx, pop_batch in enumerate(pops_corners):
        start_time = time.time()

        save_folder_batch = os.path.join(save_folder, f"batch_{batch_idx}")
        Path(save_folder_batch).mkdir(exist_ok=True, parents=True)

        n_sample = min(n_samples_corner, len(pop_batch))
        sampled_inds = random.sample(pop_batch, n_sample)

        F_all, CB_all, SO_all = [], [], []
        print(f"Starting batch {selected_corner_ids[batch_idx]}")

        inds_final = []

        for e, ind in enumerate(sampled_inds):
            print(f"Running {e + 1}/{len(sampled_inds)}")
            F_ind, CB_ind, SO_ind = [], [], []

            for k in range(n_repeat):
                print(f"  Repeat {k + 1}/{n_repeat}")
                ind = copy.deepcopy(ind)
                evaluate_individuals(
                    Population(individuals=[ind]),
                    problem=problem,
                    backup_folder=save_folder,
                )
                F_ind.append(ind.get("F"))
                CB_ind.append(ind.get("CB"))
                SO_ind.append(ind.get("SO"))
                print("fitness computed:", ind.get("F"))
                print("is critical:", ind.get("CB"))
                inds_final.append(ind)

            F_all.append(F_ind)
            CB_all.append(CB_ind)
            SO_all.append(SO_ind)

        F_all_total.extend(F_all)
        CB_all_total.extend(CB_all)
        
        write_simout_flaky(
            path=os.path.join(save_folder_batch,"simout"),
            pop=PopulationExtended(individuals=inds_final)
        )

        save_folder_plots = os.path.join(save_folder_batch, "plots")
        Path(save_folder_plots).mkdir(exist_ok=True, parents=True)

    for field in ["V","X","Y"]:
        from flaky.utils import plot_timeseries_from_pop
        plot_timeseries_from_pop(pop=PopulationExtended(individuals=inds_final),
                                    save_folder=save_folder_plots,
                                    field=field,
                                    accessor = "SO",
                                    name_folder="trajectories")
    # for i, SO in enumerate(SO_all)
    #     # Speeds
    #     V_all_ego = [simout.speed["ego"] for simout in SO]
    #     ego_file = get_incremented_filename(os.path.join(save_folder_plots, f"velocity_traces_{i}.png"))
    #     plot_velocities(V_all_ego, save_folder_plots, os.path.basename(ego_file))

    #     V_all_adv = [simout.speed["adversary"] for simout in SO]
    #     adv_file = get_incremented_filename(os.path.join(save_folder_plots, f"velocity_traces_adv_{i}.png"))
    #     plot_velocities(V_all_adv, save_folder_plots, os.path.basename(adv_file))

    #     # Locations
    #     X_all_ego = [[pos[0] for pos in simout.location["ego"]] for simout in SO]
    #     Y_all_ego = [[pos[1] for pos in simout.location["ego"]] for simout in SO]

    #     X_file = get_incremented_filename(os.path.join(save_folder_plots, f"location_x_ego_{i}.png"))
    #     plot_velocities(X_all_ego, save_folder_plots, os.path.basename(X_file))

    #     Y_file = get_incremented_filename(os.path.join(save_folder_plots, f"location_y_ego_{i}.png"))
    #     plot_velocities(Y_all_ego, save_folder_plots, os.path.basename(Y_file))

    #     X_all_adv = [simout.location["adversary"][0] for simout in SO]
    #     Y_all_adv = [simout.location["adversary"][1] for simout in SO]

    #     X_file_adv = get_incremented_filename(os.path.join(save_folder_plots, f"location_x_adv_{i}.png"))
    #     plot_velocities(X_all_adv, save_folder_plots, os.path.basename(X_file_adv))

    #     Y_file_adv = get_incremented_filename(os.path.join(save_folder_plots, f"location_y_adv_{i}.png"))
    #     plot_velocities(Y_all_adv, save_folder_plots, os.path.basename(Y_file_adv)) 
    
    # ------------------------------------------------------------------
    # Global plots
    # ------------------------------------------------------------------
    save_folder_plots_total = os.path.join(save_folder, "plots")
    Path(save_folder_plots_total).mkdir(exist_ok=True, parents=True)

    F_plot_total = np.asarray(F_all_total, dtype=float)
    F_plot_total[F_plot_total == 1000] = 2.0

    plot_boxplot(
        F_plot_total,
        problem.fitness_function.name,
        save_folder_plots_total,
        "fitness_boxplot.png",
    )

    plot_boxplot(
        np.asarray(CB_all_total).astype(int),
        "Critical",
        save_folder_plots_total,
        "critical_boxplot.png",
    )

    try:
        artifact = wandb.Artifact("results_folder", "output")
        artifact.add_dir(save_folder)
        wandb.log_artifact(artifact)
        print("uploaded results to wandb")
    except Exception:
        print("upload to wandb failed")


if __name__ == "__main__":
    main()
