import os
import csv
import re
import numpy as np
import logging as log
from dataclasses import dataclass
from typing import Dict, List, Tuple
import copy

from pymoo.core.problem import Problem
from pandas.core.frame import DataFrame
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput
import wandb

from surrogate.scaler import MinMaxScaler
from surrogate.plot_mae import plot_mae_over_time
from surrogate.model import Model
from surrogate_log.test_surrogate_log import (
    compute_criticality_points,
    train_model,
    train_model_from_points,
    transform_targets,
    inverse_transform_targets,
)
from simulations.config import xl, xu


def get_surrogate_simout():
    return SimulationOutput(
        simTime=0.0,
        times=[],
        timestamps=[],
        location={},
        velocity={},
        speed={},
        acceleration={},
        yaw={},
        collisions=[],
        actors={},
        otherParams={}
    )


@dataclass
class ADASProblemSurrogate(Problem):
    """Basic problem class for ADAS problems backed by a surrogate model."""

    def __init__(
        self,
        xl: List[float],
        xu: List[float],
        scenario_path: str,
        fitness_function: Fitness,
        simulate_function: object,
        critical_function: Critical,
        simulation_variables: List[str],
        simulation_time: float = 10,
        sampling_time: float = 100,
        design_names: List[str] = None,
        objective_names: List[str] = None,
        problem_name: str = None,
        other_parameters: Dict = None,
        approx_eval_time: float = None,
        do_visualize: bool = False,
        model_name: str = None,
        data_folder: str = None,
        apply_smote: bool = False,
        do_balance: bool = False,
        transform_cols: Tuple[int, ...] = (0,),  # log1p applied to these fitness columns
                                                  # (0,)  → F1 only      (default)
                                                  # (0,1) → F1 and F2
                                                  # ()    → no transform
    ):
        super().__init__(
            n_var=len(xl),
            n_obj=len(fitness_function.name),
            xl=xl,
            xu=xu,
        )

        assert xl is not None
        assert xu is not None
        assert scenario_path is not None
        assert fitness_function is not None
        assert simulation_time is not None
        assert sampling_time is not None
        assert np.equal(len(xl), len(xu))
        assert np.less_equal(xl, xu).all()
        assert len(fitness_function.min_or_max) == len(fitness_function.name)

        self.fitness_function = fitness_function
        self.simulate_function = simulate_function
        self.critical_function = critical_function
        self.simulation_time = simulation_time
        self.sampling_time = sampling_time
        self.simulation_variables = simulation_variables
        self.do_visualize = do_visualize

        self.model_name = model_name
        self.data_folder = data_folder
        self.dataset_all = None
        self.apply_smote = apply_smote
        self.do_balance = do_balance
        self.transform_cols = transform_cols

        self.design_names = design_names if design_names is not None else simulation_variables
        self.objective_names = objective_names if objective_names is not None else fitness_function.name

        self.scenario_path = scenario_path
        self.problem_name = problem_name

        if other_parameters is not None:
            self.other_parameters = other_parameters

        if approx_eval_time is not None:
            self.approx_eval_time = approx_eval_time

        self.signs = []
        for value in self.fitness_function.min_or_max:
            if value == 'max':
                self.signs.append(-1)
            elif value == 'min':
                self.signs.append(1)
            else:
                raise ValueError(
                    f"Error: The optimization property '{value}' is not supported."
                )

        self.counter = 0
        self.initialized = False

        log.info(f"[ADASProblemSurrogate] transform_cols={self.transform_cols} "
                 f"({'log1p on cols ' + str(transform_cols) if transform_cols else 'no transform'})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fitness_in_confidence_int(
        self,
        critical_function,
        fitness: np.ndarray,
        error: np.ndarray,
    ) -> bool:
        """Return True when the confidence interval is entirely critical or non-critical."""
        fitness_bottom = fitness - error
        fitness_up = fitness + error

        log.info(f"Conf interval: {fitness_bottom, fitness_up}")
        log.info(f"Error: {error}")
        log.info(f"Predicted: {fitness}")

        def is_critical(f):
            return critical_function.eval(f)

        both_critical = is_critical(fitness_bottom) and is_critical(fitness_up)
        both_non_critical = not is_critical(fitness_bottom) and not is_critical(fitness_up)

        if both_critical or both_non_critical:
            log.info("Prediction is in interval.")
            return True
        else:
            log.info("Prediction is not in interval.")
            return False

    def predict_with_surrogate(self, model, x: np.ndarray) -> np.ndarray:
        """
        Predict fitness for input x using the surrogate model.

        Feature scaling : MinMaxScaler over [xl, xu]  (matches training)
        Target transform: log1p applied to self.transform_cols during training
                          → inverse expm1 applied here to the same columns.
                          Pass transform_cols=() to skip entirely.

        Returns un-transformed predictions in the original fitness space.
        """
        x_scaler = MinMaxScaler(
            min_val=np.array(self.xl),
            max_val=np.array(self.xu),
            clip=False,
        )
        x_scaled = x_scaler.transform(x)

        log.info(f"[predict] x_raw:    {x}")
        log.info(f"[predict] x_scaled: {x_scaled}")

        y_pred_scaled = model.predict(x_scaled)

        log.info(f"[predict] y_pred (transformed space): {y_pred_scaled}")

        # Inverse transform — only columns in transform_cols need inverting
        # NOTE: surrogate was trained on fitness values that already have
        # self.signs applied (stored via out["F"][i] = fitness_hifi),
        # so NO sign correction is needed here.
        y_pred = inverse_transform_targets(
            y_pred_scaled,
            f1_shift=0.0,
            transform_cols=self.transform_cols,
        )

        log.info(f"predicted y (original scale): {y_pred}")
        return y_pred

    def predict_scaled(self, model, x, x_min, x_max, y_min, y_max):
        """Deprecated — use predict_with_surrogate() instead."""
        log.warning("predict_scaled() is deprecated. Use predict_with_surrogate() instead.")
        return self.predict_with_surrogate(model, x)

    def _initialize_surrogate(
        self,
        apply_smote: bool = True,
        model_output_path: str = "./model_out/",
    ):
        """Train the surrogate on the initial data_folder dataset."""
        model, mae, X_train, X_val, y_train, y_val = train_model(
            model_name=self.model_name,
            data_folder=self.data_folder,
            apply_smote=apply_smote,
            model_output_path=model_output_path,
            critical_function=self.critical_function,
            balance=self.do_balance,
            transform_cols=self.transform_cols,
        )

        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        print(f"Surrogate model initialised on dataset size: {X_full.shape}")
        print(f"Initialised surrogate model MAE: {mae}")
        return model, mae, X_full, y_full, X_val, y_val

    def append_mae_to_csv(self, filepath: str, step: int, mae, pos: int, neg: int):
        """Append a single MAE row to a CSV log file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"filepath is: {filepath}")
        print(f"counter is: {step}")
        print(f"mae is: {mae}")
        file_exists = os.path.isfile(filepath)
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['step', 'mae', 'pos', 'neg'])
            writer.writerow([step, mae, pos, neg])

    def log_mae(self, error):
        """Log per-objective MAE to wandb."""
        def normalize(name: str) -> str:
            return re.sub(r"\s+", "_", name.strip().lower())

        metrics = {
            f"mae_{normalize(name)}": err
            for name, err in zip(self.fitness_function.name, error)
        }
        wandb.log(metrics)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, x, out, *args, **kwargs):
        hifi_sim_simulate = self.simulate_function

        n = len(x)
        out["SO"]       = [None] * n
        out["SO_LOFI"]  = [None] * n
        out["SO_HIFI"]  = [None] * n
        out["F_LOFI"]   = [[None] * self.n_obj for _ in range(n)]
        out["F_HIFI"]   = [[None] * self.n_obj for _ in range(n)]
        out["F"]        = [None] * n
        out["CB_LOFI"]  = [None] * n
        out["CB_HIFI"]  = [None] * n
        out["CB"]       = [None] * n
        out["P"]        = [None] * n

        save_folder = kwargs["algorithm"].save_folder

        # ── Initialise surrogate on first call ──────────────────────────
        if not self.initialized:
            model, mae, X_data, y_data, X_val, y_val = self._initialize_surrogate(
                apply_smote=self.apply_smote,
                model_output_path=save_folder + "/model_out/",
            )
            self.data_points     = [X_data, y_data]
            self.validation_data = [X_val, y_val]
            self.model           = model
            self.model_error     = mae
            self.initialized     = True

            labels = compute_criticality_points(
                F=self.data_points[1],
                critical_function=self.critical_function,
            )
            self.append_mae_to_csv(
                save_folder + "/model_out/mae.csv",
                step=0,
                mae=self.model_error,
                pos=int(np.sum(labels == True)),
                neg=int(len(self.data_points[1]) - np.sum(labels == True)),
            )
            self.log_mae(mae)

        # ── Per-individual evaluation ────────────────────────────────────
        for i, xi in enumerate(x):
            print(f"xi is {xi}")
            self.counter += 1
            log.info(f"Running evaluation number {self.counter}")

            # --- Lo-fi: surrogate prediction ----------------------------
            try:
                fitness_lofi = self.predict_with_surrogate(
                    self.model, np.asarray([xi])
                )[0]

                log.info(f"input: {np.asarray([xi])}")
                log.info(f"fitness_lofi (surrogate): {fitness_lofi}")
                fitness_lofi = np.asarray(fitness_lofi)
                simout_lofi  = None
            except Exception as e:
                log.error("Exception in surrogate predictor:")
                raise e

            out["SO_LOFI"][i]  = simout_lofi
            out["F_LOFI"][i]   = fitness_lofi
            is_critical_lofi   = self.critical_function.eval(fitness_lofi)
            out["CB_LOFI"][i]  = is_critical_lofi

            # --- Decide: trust surrogate or run hi-fi? ------------------
            if self._fitness_in_confidence_int(
                self.critical_function,
                fitness=fitness_lofi,
                error=self.model_error,
            ):
                log.info(f"########## SURROGATE VALUE USED ##########")

                # Surrogate is reliable — use lo-fi result
                out["SO"][i]  = simout_lofi
                out["F"][i]   = fitness_lofi
                out["CB"][i]  = is_critical_lofi

            else:
                log.info("Surrogate prediction is uncertain. Running hi-fi simulation.")
                try:
                    simout_hifi = hifi_sim_simulate(
                        [xi],
                        self.simulation_variables,
                        self.scenario_path,
                        sim_time=self.simulation_time,
                        time_step=self.sampling_time,
                        do_visualize=self.do_visualize,
                    )
                except Exception as e:
                    log.error("Exception during HiFi simulation:")
                    raise e

                simout_hifi  = simout_hifi[0]
                fitness_hifi = np.asarray(self.signs) * np.array(
                    self.fitness_function.eval(simout_hifi, **kwargs)
                )
                fitness_hifi = np.asarray(fitness_hifi)

                out["SO_HIFI"][i]  = simout_hifi
                out["F_HIFI"][i]   = fitness_hifi
                is_critical_hifi   = self.critical_function.eval(fitness_hifi)
                out["CB_HIFI"][i]  = is_critical_hifi

                out["SO"][i]  = simout_hifi
                out["F"][i]   = fitness_hifi
                out["CB"][i]  = is_critical_hifi

                # Add new hi-fi point to dataset and retrain unconditionally
                xi_array      = np.asarray([xi])           # (1, n_features)
                fitness_array = np.asarray([out["F"][i]])  # (1, n_targets)

                print(f"dataset old size: {self.data_points[0].shape}")
                self.data_points[0] = np.concatenate(
                    [self.data_points[0], xi_array], axis=0
                )
                self.data_points[1] = np.concatenate(
                    [self.data_points[1], fitness_array], axis=0
                )
                print(f"dataset updated, new size: {self.data_points[0].shape}")

                model, meta = train_model_from_points(
                    model_name=self.model_name,
                    X=self.data_points[0],
                    y=self.data_points[1],
                    model_output_path=save_folder + "/model_out/",
                    name_suffix=f"_eval-{self.counter}",
                    apply_smote=self.apply_smote,
                    balance=False,
                    validation_data=self.validation_data,
                    critical_function=self.critical_function,
                    transform_cols=self.transform_cols,
                    xl = self.xl,
                    xu = self.xu
                )
                self.model       = model
                self.model_error = meta[0]  # meta = (mae, X_train, X_val, y_train, y_val)
                print("model error after retraining: ", meta[0])

                labels = compute_criticality_points(
                    F=self.data_points[1],
                    critical_function=self.critical_function,
                )
                path_mae = save_folder + "/model_out/mae.csv"
                self.append_mae_to_csv(
                    path_mae,
                    step=self.counter,
                    mae=self.model_error,
                    pos=int(np.sum(labels)),
                    neg=int(len(self.data_points[1]) - np.sum(labels)),
                )
                self.log_mae(self.model_error)
                plot_mae_over_time(path_mae)

                print(f"new model error: {self.model_error}")
                print("new surrogate model created")

        out["F"] = np.asarray(out["F"])
        log.info(f"type F: {type(out['F'])}")
        log.info(f"F: {out['F']}")

    def is_simulation(self):
        return True

    def update_name(self, attribute, value):
        """Patch a numeric suffix on an attribute name inside problem_name."""
        pattern = fr"{attribute}\d+"
        replacement = f"{attribute}{value}"
        self.problem_name = re.sub(pattern, replacement, self.problem_name)