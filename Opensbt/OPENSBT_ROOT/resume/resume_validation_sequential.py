from datetime import datetime

import wandb
import glob
import os
import argparse
import sys

sys.path.insert(0,"/home/user/testing/")
sys.path.insert(0,"/home/user/testing/MultiDrive/")
sys.path.insert(0,"/home/user/testing/MultiDrive/Frenetix-Motion-Planner")
sys.path.insert(0,"/home/user/testing/MultiDrive/venv/lib/python3.10/site-packages")

import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from opensbt.utils.wandb import *
from simulations.simulators import *
from simulations.rerun_sequential import rerun_and_analyze_sequential

def rerun_sequential_and_upload_to_wandb(
    simulate_function: str,
    save_folder: str,
    project: str,
    entity: str,
    run_id: str,
    rerun_only_critical: bool = True,
    n_rerun: int = 1,
    only_if_no_hifi: bool = True,
    only_rerun_folder: bool = True,
):
    """ save folder should exist and contain "problem" file in backup folder, and "all_testcases.csv" 
    """

    wandb.init(
        project=project,
        entity=entity,
        id=run_id
    )

    os.environ["OPENSBT_RUN_ROOT"] = save_folder
    os.environ["OPENSBT_RUN_ID"]   = run_id

    ###### Rerun in hifi all NaN values. Disable if you dont need reruns.
    rerun_and_analyze_sequential(
        simulate_function=simulate_function,
        folder=save_folder,
        critical_only=rerun_only_critical,
        n_rerun=n_rerun,
        only_if_no_hifi=only_if_no_hifi,
    )

    # we split logging in two steps, because report.json too large
    if only_rerun_folder:
        rerun_folders = glob.glob(os.path.join(save_folder, "rerun_*"))
        for folder in rerun_folders:
            wandb_log_folder(
                folder_path=folder,
                artifact_name="results_folder_rerun",
                artifact_type="output",
                exclude_patterns=["*report.json"],
            )
    else:
        wandb_log_folder(
            folder_path=save_folder,
            artifact_name="results_folder",
            artifact_type="output",
            exclude_patterns=["*report.json"],
        )

    # always upload validation artifact
    report_file = glob.glob(
        os.path.join(save_folder, "**", "*report.json"), recursive=True
    )[0]

    wandb_log_artifact(
        file_path=report_file,
        artifact_name="report",
        artifact_type="validation",
    )

    print("uploaded results to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate_function", type = str, required=True)
    parser.add_argument("--save_folder",          type=str, required=True)
    parser.add_argument("--project",              type=str, required=True)
    parser.add_argument("--entity",               type=str, required=True)
    parser.add_argument("--run_id",               type=str, required=True)
    parser.add_argument("--rerun_only_critical",  action="store_true", default=True)
    parser.add_argument("--n_rerun",              type=int, default=1)
    parser.add_argument("--only_if_no_hifi",      action="store_true", default=True)
    parser.add_argument("--only_rerun_folder",    action="store_true", default=True,
                        help="If set, only upload rerun_* subfolders instead of the full save_folder")
    args = parser.parse_args()

    rerun_sequential_and_upload_to_wandb(
        simulate_function=args.simulate_function,
        save_folder=args.save_folder,
        project=args.project,
        entity=args.entity,
        run_id=args.run_id,
        rerun_only_critical=args.rerun_only_critical,
        n_rerun=args.n_rerun,
        only_if_no_hifi=args.only_if_no_hifi,
        only_rerun_folder=args.only_rerun_folder,
    )