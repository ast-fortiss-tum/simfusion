from datetime import datetime

import os
import sys
import glob
import argparse
import wandb

sys.path.insert(0, "/home/user/testing/")
sys.path.insert(0, "/home/user/testing/MultiDrive/")
sys.path.insert(0, "/home/user/testing/MultiDrive/Frenetix-Motion-Planner")
sys.path.insert(0, "/home/user/testing/MultiDrive/venv/lib/python3.10/site-packages")

import pymoo
from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from simulations.rerun_tests import rerun_and_analyze
from simulations.simulators import *  # noqa: F403

from opensbt.utils.wandb import *  # noqa: F403
from simulations.simulators import SIMULATOR_MAPPING

def _validation_run_name_from_save_folder(save_folder: str, prefix: str = "Validation") -> str:
    norm = os.path.normpath(save_folder)
    parent_dir = os.path.dirname(norm)
    parent_name = os.path.basename(parent_dir) or "unknown"
    return f"{prefix}_{parent_name}"


def _validation_run_name(problem_name: str | None, save_folder: str, prefix: str = "Validation") -> str:
    if problem_name and problem_name.strip():
        return f"{prefix}_{problem_name.strip()}"
    return _validation_run_name_from_save_folder(save_folder, prefix=prefix)


def rerun_and_upload_to_wandb(
    simulate_function: str,
    save_folder: str,
    project: str,
    entity: str,
    sim_rerun: str,
    sim_original: str = "lofi",
    rerun_only_critical: bool = True,
    n_rerun: int = 1,
    only_if_no_hifi: bool = True,
    only_rerun_folder: bool = False,
    ratio: float = 1.0,
    run_name_prefix: str = "Validation",
    problem_name: str | None = None,
    tags: list[str] | None = None,  # <-- NEW
    filter_goal_threshold: float = None,  # optional post-filtering (hack for CR case study)
    filter_obstacle_distance: float = None,  # optional post-filtering
):
    run_name = _validation_run_name(problem_name, save_folder=save_folder, prefix=run_name_prefix)

    # If caller didn't pass tags, create defaults (similar to tags = [f"{k}:{v}" for ...])
    if tags is None:
        tags = [
            f"simulate_function:{simulate_function}",
            f"sim_original:{sim_original}",
            f"sim_rerun:{sim_rerun}",
            f"rerun_only_critical:{rerun_only_critical}",
            f"n_rerun:{n_rerun}",
            f"only_if_no_hifi:{only_if_no_hifi}",
            f"only_rerun_folder:{only_rerun_folder}",
            f"ratio:{ratio}",
            f"problem_name:{problem_name}" if problem_name else "problem_name:(derived)",
            f"run_name_prefix:{run_name_prefix}",
        ]

    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,  # <-- HERE (this is what you meant)
    )

    run_id = wandb.run.id

    os.environ["OPENSBT_RUN_ROOT"] = save_folder
    os.environ["OPENSBT_RUN_ID"] = run_id

    rerun_and_analyze(
        simulate_function=simulate_function,
        folder=save_folder,
        sim_original=sim_original,
        sim_rerun=sim_rerun,
        critical_only=rerun_only_critical,
        n_rerun=n_rerun,
        only_if_no_hifi=only_if_no_hifi,
        ratio_sample=ratio,
        filter_goal_threshold=filter_goal_threshold,
        filter_obstacle_distance=filter_obstacle_distance,
    )

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
            exclude_patterns=["executed-simulations-*","*report.json"],
        )

    report_files = glob.glob(os.path.join(save_folder, "**", "*report.json"), recursive=True)
    if not report_files:
        raise FileNotFoundError(f"No *report.json found under save_folder={save_folder}")
    report_file = report_files[0]

    wandb_log_artifact(
        file_path=report_file,
        artifact_name="report",
        artifact_type="validation",
    )

    print(f"uploaded results to wandb (run_name={run_name}, run_id={run_id})")

    wandb.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--simulate_function", type=str, required=True)
    parser.add_argument("--sim_rerun", type=str, required=True)

    parser.add_argument("--problem_name", type=str, default=None)

    parser.add_argument("--sim_original", type=str, default="lofi")
    parser.add_argument("--rerun_only_critical", action="store_true", default=True)
    parser.add_argument("--n_rerun", type=int, default=1)
    parser.add_argument("--only_if_no_hifi", action="store_true", default=True)
    parser.add_argument("--only_rerun_folder", action="store_true", default=True)
    parser.add_argument("--ratio", type=float, default=1.0)

    parser.add_argument("--run_name_prefix", type=str, default="Validation")

    # Optional: add extra tags manually; repeatable: --tag foo --tag bar:baz
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--filter_goal_threshold", type=float, default=None,
                        help="Post-filtering threshold for goal distance (optional)")
    parser.add_argument("--filter_obstacle_distance", type=float, default=None,
                        help="Post-filtering threshold for obstacle distance (optional)")

    args = parser.parse_args()

    # Auto tags derived from args (your idea), plus any explicit --tag
    auto_tags = [f"{k}:{v}" for k, v in vars(args).items() if k != "tag"]
    tags = auto_tags + list(args.tag)

    rerun_and_upload_to_wandb(
        simulate_function=args.simulate_function,
        save_folder=args.save_folder,
        project=args.project,
        entity=args.entity,
        sim_rerun=args.sim_rerun,
        problem_name=args.problem_name,
        sim_original=args.sim_original,
        rerun_only_critical=args.rerun_only_critical,
        n_rerun=args.n_rerun,
        only_if_no_hifi=args.only_if_no_hifi,
        only_rerun_folder=args.only_rerun_folder,
        ratio=args.ratio,
        run_name_prefix=args.run_name_prefix,
        tags=tags,
        filter_goal_threshold=args.filter_goal_threshold,
        filter_obstacle_distance=args.filter_obstacle_distance,
    )