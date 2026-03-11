import argparse
import re
from pathlib import Path
import sys

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
from simulations.simulators import *

from opensbt.utils.wandb import *

from resume.resume_validation_sample import rerun_and_upload_to_wandb


def parse_seeds(seeds_str: str) -> set[int]:
    return {int(s.strip()) for s in seeds_str.split(",") if s.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()

    # Option A: discover runs by searching for all_testcases.csv under --root
    parser.add_argument("--root", type=Path, default=None)

    # Option B: provide explicit save folders (repeatable)
    # Example: --path /a/b/c/run1 --path /a/b/c/run2
    parser.add_argument("--path", action="append", default=[])

    # Option C (NEW): provide a single save folder (non-repeatable convenience)
    parser.add_argument(
        "--save_folder",
        type=Path,
        default=None,
        help="Single save folder path (folder containing all_testcases.csv).",
    )

    # NEW: override problem name (otherwise inferred from save_folder.parent.name)
    parser.add_argument(
        "--problem_name",
        type=str,
        default=None,
        help="Override inferred problem name (e.g., MyProblem_seed10).",
    )

    parser.add_argument("--seeds", type=str, default="10,11,12")
    parser.add_argument("--simulate_function", type=str, default="beamng")
    parser.add_argument("--project", type=str, default="lofi-validation")
    parser.add_argument("--entity", type=str, default="lofi-hifi")
    parser.add_argument("--sim_rerun", type=str, default="hifi")
    parser.add_argument("--ratio", type=float, default=0.25)
    parser.add_argument("--exclude_string", type=str, default=None)
    parser.add_argument("--only_if_no_hifi", action="store_true", default=True)
    parser.add_argument("--rerun_only_critical", action="store_true", default=False)
    parser.add_argument("--n_rerun", type=int, default=1)
    parser.add_argument("--sim_original", type=str, default="lofi")
    parser.add_argument("--th_goal", type=float, default=None)
    parser.add_argument("--th_obstacle_distance", type=float, default=None)
    parser.add_argument(
        "--filter_goal_threshold",
        type=float,
        default=None,
        help="Post-filtering threshold for goal distance (optional)",
    )
    parser.add_argument(
        "--filter_obstacle_distance",
        type=float,
        default=None,
        help="Post-filtering threshold for obstacle distance (optional)",
    )

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    seed_re = re.compile(r"_seed(\d+)$")

    # Build list of save_folders to validate
    save_folders: list[Path] = []

    # Prefer explicit folder specifications in this order:
    # 1) --save-folder
    # 2) --path (repeatable)
    # 3) discovery under --root
    if args.save_folder is not None:
        save_folders.append(args.save_folder.expanduser().resolve())
    elif args.path:
        save_folders.extend(Path(p).expanduser().resolve() for p in args.path)
    else:
        root = (args.root or Path("./results_wandb")).expanduser().resolve()

        for csv_path in root.rglob("all_testcases.csv"):
            save_folder = csv_path.parent  # parent folder of all_testcases.csv
            # ensure backup is in the same folder and contains "LoFi"
            if (save_folder / "backup").is_dir() and "LoFi" in str(save_folder):
                if args.exclude_string and args.exclude_string in str(save_folder):
                    continue
                save_folders.append(save_folder)

    # Deduplicate while preserving order
    seen = set()
    save_folders = [p for p in save_folders if not (str(p) in seen or seen.add(str(p)))]
    print("save_folder:", save_folders)

    for save_folder in save_folders:
        # If user provided --problem-name, use it; otherwise infer from folder structure
        problem_name = args.problem_name or save_folder.parent.name

        # Keep the old seed filtering behavior unless the user overrides the problem name.
        # If overridden, we still apply seed filtering IF the override includes _seedNNN.
        if args.problem_name is None:
            m = seed_re.search(problem_name)
            if not m:
                continue
            seed = int(m.group(1))
            if seed not in seeds:
                continue
        else:
            m = seed_re.search(problem_name)
            if m and int(m.group(1)) not in seeds:
                continue

        print(f"Validating: {save_folder}  (problem={problem_name})")

        try:
            rerun_and_upload_to_wandb(
                simulate_function=args.simulate_function,
                save_folder=str(save_folder),
                project=args.project,
                entity=args.entity,
                sim_rerun=args.sim_rerun,
                problem_name=problem_name,
                ratio=args.ratio,
                only_if_no_hifi=args.only_if_no_hifi,
                run_name_prefix="Validation",
                rerun_only_critical=args.rerun_only_critical,
                n_rerun=args.n_rerun,
                sim_original=args.sim_original,
                filter_goal_threshold=args.filter_goal_threshold,
                filter_obstacle_distance=args.filter_obstacle_distance,
            )

            print(f"Successfully validated: {save_folder}")
        except Exception as e:
            print(f"Failed to validate: {save_folder}  (error={e})")


if __name__ == "__main__":
    main()