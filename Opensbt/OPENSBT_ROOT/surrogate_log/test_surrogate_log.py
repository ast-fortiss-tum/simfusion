import sys
sys.path.insert(0,"/home/user/testing/")
sys.path.insert(0,"/home/user/testing/MultiDrive/")
sys.path.insert(0,"/home/user/testing/MultiDrive/Frenetix-Motion-Planner")
sys.path.insert(0,"/home/user/testing/MultiDrive/venv/lib/python3.10/site-packages")

from cr_beamng_cosimulation.beamng_simulation import BeamNGSimulation
import numpy as np
import pandas as pd
from glob import glob
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from imblearn.over_sampling import SMOTE
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner import HyperModel
from kerastuner.tuners import BayesianOptimization
from math import floor
from skopt import BayesSearchCV
import logging as log
from typing import Tuple
import shutil

from cr_beamng_cosimulation.beamng_simulation import BeamNGSimulation as BNG_Simulation
from simulations.config import critical_function,fitness_function, xl, xu
from surrogate.visualize_data import visualize_data
from surrogate.scaler import MinMaxScaler
from pymoo.core.population import Population
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.problem.adas_problem import ADASProblem
from simulations.config import scenario_file, simulation_variables, fitness_function, sim_time, sampling_time
from pathlib import Path

# === CONFIGURATION ===
output_folder     = "./surrogate_log/plots/"
model_output_path = "./surrogate_log/out/model_"
save_folder       = "./surrogate_log/out/"


# =========================
# === TARGET TRANSFORMS ===
# =========================

def transform_targets(y, f1_shift=0.0, transform_cols: Tuple[int, ...] = (0,)):
    """
    Apply log1p to the columns listed in transform_cols.
    All other columns are left in their original scale.

    :param transform_cols: column indices to apply log1p to.
                           (0,)   → F1 only          (default, original behaviour)
                           (0, 1) → F1 and F2
                           ()     → no transform at all
    """
    y_out = y.copy().astype(float)
    for col in transform_cols:
        vals = y[:, col] + f1_shift
        if np.any(vals <= -1):
            print(f"[warn] transform_cols col {col} has {np.sum(vals <= -1)} values <= -1 "
                  f"(min={vals.min():.4f}) — clipping to -1+eps before log1p")
            vals = np.clip(vals, -1 + 1e-6, None)
        y_out[:, col] = np.log1p(vals)
    return y_out


def inverse_transform_targets(y_scaled, f1_shift=0.0, transform_cols: Tuple[int, ...] = (0,)):
    """
    Inverse of transform_targets — applies expm1 to transform_cols only.
    Must receive the same transform_cols that was passed to transform_targets.
    """
    y_out = y_scaled.copy().astype(float)
    for col in transform_cols:
        y_out[:, col] = np.expm1(y_scaled[:, col]) - f1_shift
    return y_out


# =========================
# === FORMAT DETECTION ===
# =========================

def detect_format(df: pd.DataFrame) -> dict:
    """
    Auto-detect input and target columns from CSV column names.

    Convention:
      - Excluded  : 'Index', any column starting with 'Critical'
      - Targets   : any column starting with 'Fitness_'
      - Inputs    : everything else
    """
    all_cols = list(df.columns)

    exclude     = {c for c in all_cols if c == "Index" or c.startswith("Critical")}
    target_cols = [c for c in all_cols if c.startswith("Fitness_")]
    input_cols  = [c for c in all_cols if c not in exclude and not c.startswith("Fitness_")]

    if not target_cols:
        raise ValueError(f"No Fitness_* columns found.\nColumns found: {all_cols}")
    if not input_cols:
        raise ValueError(f"No input columns found.\nColumns found: {all_cols}")

    print(f"[load] Input columns  ({len(input_cols)})  : {input_cols}")
    print(f"[load] Target columns ({len(target_cols)}) : {target_cols}")

    return {
        "input_columns": input_cols,
        "target_cols":   target_cols,
    }


# =========================
# === DATA LOADING ===
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=["GL", "GNL", "RF", "MLP"],
                        help="List of surrogate models to evaluate: GL, GNL, RF, MLP")
    parser.add_argument("--sample", type=float, nargs="+",
                        help="Input sample as floats matching the detected format's input columns")
    parser.add_argument("--data_folder", type=str, default="./surrogate_log/data/batch0/",
                        help="Path to the folder containing training CSV files")
    parser.add_argument("--apply_smote", action="store_true", default=False,
                        help="Apply SMOTE to balance classes (default: False)")
    parser.add_argument("--balance", action="store_true", default=False,
                        help="Balance classes by taking equal samples from each (default: False)")
    parser.add_argument("--transform_cols", type=int, nargs="*", default=[0],
                        help="Fitness column indices to apply log1p transform to. "
                             "Default: 0 (F1 only). Pass no values to disable: --transform_cols")
    return parser.parse_args()


def load_and_preprocess_data(data_folder: str):
    csv_files = glob(f"{data_folder}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f"[load] Files loaded        : {csv_files}")
    print(f"[load] Rows loaded         : {len(df)}")

    fmt         = detect_format(df)
    input_cols  = fmt["input_columns"]
    target_cols = fmt["target_cols"]

    # Filter sentinel rows
    for col in target_cols:
        df = df[df[col] != 1000].reset_index(drop=True)
    print(f"[load] Rows after filtering: {len(df)}")

    xl_fmt = np.array(df[input_cols].min().values, dtype=float)
    xu_fmt = np.array(df[input_cols].max().values, dtype=float)
    print(f"[load] xl (data min)       : {xl_fmt}")
    print(f"[load] xu (data max)       : {xu_fmt}")

    return df, input_cols, target_cols, xl_fmt, xu_fmt


# =========================
# === CRITICALITY ===
# =========================

def compute_criticality(df, target_cols, critical_function):
    fitness_tuples = list(df[target_cols].itertuples(index=False, name=None))
    label_values = [critical_function.eval(np.asarray(t)) for t in fitness_tuples]
    return np.asarray(label_values)


def compute_criticality_points(F, critical_function):
    label_values = [critical_function.eval(p) for p in F]
    return np.asarray(label_values)


# =========================
# === NaN GUARD ===
# =========================

def assert_no_nan(y, label="y"):
    """Raise a clear error if y contains NaN, showing which columns and values."""
    nan_counts = np.isnan(y).sum(axis=0)
    if nan_counts.any():
        nan_rows = np.where(np.isnan(y).any(axis=1))[0]
        raise ValueError(
            f"[NaN guard] NaN detected in {label}.\n"
            f"  NaNs per column : {nan_counts}\n"
            f"  Offending rows  : {nan_rows}\n"
            f"  Original values : {y[nan_rows]}\n"
            f"Hint: a Fitness column may have values <= -1 in transform_cols, "
            f"or the dataset contains missing values."
        )


# =========================
# === SMOTE FUNCTIONS ===
# =========================

def apply_smote_with_fit_no_sidecol(X, y_reg, y_class, apply_smote, data_folder_name,
                                    gt_fit=True, save_folder=None,
                                    desired_valid_samples=100, max_iterations=100):
    if not apply_smote:
        return X, y_reg

    gt_suffix = "_gt" if gt_fit else ""
    valid_synthetic_X_all = []
    valid_F_all = []
    valid_CB_all = []

    current_X       = X.copy()
    current_y_class = y_class.copy()

    n_valid_collected = 0
    iteration         = 0

    print(f"[SMOTE] desired_valid_samples: {desired_valid_samples}")
    print(f"[SMOTE] class balance — critical: {np.sum(y_class==True)}  "
          f"non-critical: {np.sum(y_class==False)}")

    while n_valid_collected < desired_valid_samples and iteration < max_iterations:
        print(f"[INFO] SMOTE iteration {iteration+1}  "
              f"(collected {n_valid_collected}/{desired_valid_samples})")

        import random
        smote = SMOTE(random_state=random.randint(0, 10**7))
        X_resampled, y_class_resampled = smote.fit_resample(current_X, current_y_class)

        n_orig      = len(current_X)
        synthetic_X = X_resampled[n_orig:]

        if not gt_fit:
            raise ValueError("Incremental SMOTE with fitness evaluation requires gt_fit=True")

        problem = ADASProblem(
            problem_name="dummy",
            scenario_path=os.path.join(os.getcwd(), "scenarios", scenario_file),
            xl=xl,
            xu=xu,
            simulation_variables=simulation_variables,
            fitness_function=fitness_function,
            critical_function=critical_function,
            simulate_function=BeamNGSimulation.simulate,
            simulation_time=sim_time,
            sampling_time=sampling_time,
            approx_eval_time=10,
            do_visualize=True
        )

        population = Population.new("X", synthetic_X)
        for ind in population:
            evaluate_individuals(
                Population(individuals=[ind]),
                problem=problem,
                backup_folder=save_folder if save_folder else "./model/init/"
            )

        F  = population.get("F")
        CB = population.get("CB").flatten()

        valid_mask        = F[:, 0] != 1000
        valid_synthetic_X = synthetic_X[valid_mask]
        valid_F           = F[valid_mask]
        valid_CB          = CB[valid_mask]

        critical_mask     = valid_CB == True
        valid_synthetic_X = valid_synthetic_X[critical_mask]
        valid_F           = valid_F[critical_mask]
        valid_CB          = valid_CB[critical_mask]

        n_found = len(valid_synthetic_X)
        log.info(f"[SMOTE] iter {iteration+1}: {n_found} valid critical samples "
                 f"({np.sum(valid_mask)} valid total, {len(synthetic_X)} synthetic generated)")

        if n_found > 0:
            valid_synthetic_X_all.append(valid_synthetic_X)
            valid_F_all.append(valid_F)
            valid_CB_all.append(valid_CB)
            n_valid_collected += n_found

            current_X       = np.vstack([current_X, valid_synthetic_X])
            current_y_class = np.hstack([current_y_class, valid_CB])

        iteration += 1

    if valid_synthetic_X_all:
        valid_synthetic_X = np.vstack(valid_synthetic_X_all)
        valid_F           = np.vstack(valid_F_all)
        valid_CB          = np.hstack(valid_CB_all)

        valid_synthetic_X = valid_synthetic_X[:desired_valid_samples]
        valid_F           = valid_F[:desired_valid_samples]
        valid_CB          = valid_CB[:desired_valid_samples]

        X_final       = np.vstack([X, valid_synthetic_X])
        y_reg_final   = np.vstack([y_reg, valid_F])
        y_class_final = np.hstack([y_class, valid_CB])

        print(f"[SMOTE] Collected {len(valid_synthetic_X)} critical synthetic samples "
              f"in {iteration} iterations")
    else:
        print("[WARN] No valid synthetic samples found. Returning original data.")
        X_final       = X
        y_reg_final   = y_reg
        y_class_final = y_class

    visualize_data(
        X_final, y_class_final,
        filename=f"data_{data_folder_name}_smote{gt_suffix}_plot.png",
        output_folder=output_folder
    )
    visualize_data(
        X_final, y_class_final,
        filename="data_with_smote_plot.png",
        output_folder=output_folder
    )

    print(f"[INFO] Original samples:    {len(X)}")
    print(f"[INFO] Final dataset size:  {len(X_final)} "
          f"(+{len(X_final) - len(X)} valid critical synthetic samples)")
    print(f"[INFO] Final class balance — "
          f"critical: {np.sum(y_class_final==True)}  "
          f"non-critical: {np.sum(y_class_final==False)}")

    return X_final, y_reg_final


# =========================
# === MODEL SELECTION ===
# =========================

def get_model(name, X_train_scaled, y_train_scaled):
    if name == "GL":
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)) \
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1.0))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=0,
            n_restarts_optimizer=5,
        )
        model.fit(X_train_scaled, y_train_scaled)
        return model

    elif name == "RF":
        param_grid = {
            'n_estimators': (50, 200),
            'max_depth': (3, 20),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 4)
        }
        base_model = RandomForestRegressor(random_state=0)
        search = BayesSearchCV(base_model, param_grid, n_iter=10, cv=3, random_state=42)
        search.fit(X_train_scaled, y_train_scaled)
        return search.best_estimator_

    elif name == "MLP":
        tune_hyperparameter = True

        if tune_hyperparameter:
            # Scope tuner cache to input/output shape — prevents stale cache conflicts
            n_inputs      = X_train_scaled.shape[1]
            n_outputs     = y_train_scaled.shape[1]
            tuner_project = f"mlp_regression_in{n_inputs}_out{n_outputs}"
            tuner_dir     = "./surrogate_log/tuner_dir"

            class MLPHyperModel(HyperModel):
                def __init__(self, input_shape, output_shape):
                    self.input_shape  = input_shape
                    self.output_shape = output_shape

                def build(self, hp):
                    model = Sequential()
                    model.add(Dense(
                        units=hp.Int('units_1', min_value=10,
                                     max_value=floor((2 / 3) * len(X_train_scaled)), step=10),
                        activation=hp.Choice('activation_1', ['relu', 'tanh', 'sigmoid']),
                        kernel_initializer=hp.Choice('init_1', ['he_uniform', 'glorot_uniform']),
                        input_shape=self.input_shape
                    ))
                    model.add(Dense(
                        units=hp.Int('units_2', min_value=10,
                                     max_value=floor((2 / 3) * len(X_train_scaled)), step=10),
                        activation=hp.Choice('activation_2', ['relu', 'tanh', 'sigmoid']),
                        kernel_initializer=hp.Choice('init_2', ['he_uniform', 'glorot_uniform']),
                    ))
                    model.add(Dense(self.output_shape, activation='linear'))
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    return model

            print(f"[MLP] input shape: {X_train_scaled.shape}  output shape: {y_train_scaled.shape}")
            print(f"[MLP] tuner project: {tuner_project}  dir: {tuner_dir}")

            hypermodel = MLPHyperModel(
                input_shape=(X_train_scaled.shape[1],),
                output_shape=y_train_scaled.shape[1]
            )
            tuner = BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=10,
                seed=42,
                directory=tuner_dir,
                project_name=tuner_project,
                overwrite=False,   # set True to force retrain and ignore cache
            )
            tuner.search(X_train_scaled, y_train_scaled,
                         validation_split=0.2, epochs=50, verbose=0)
            return tuner.get_best_models(num_models=1)[0]
        else:
            model = Sequential([
                Dense(100, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                Dense(100, activation='relu'),
                Dense(y_train_scaled.shape[1], activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model

    else:
        raise ValueError(f"Unknown model: {name}")


# =========================
# === TRAIN FROM POINTS ===
# =========================

def train_model_from_points(model_name,
                             X, y,
                             model_output_path="./model_out/",
                             apply_smote=True,
                             name_suffix="",
                             balance=False,
                             validation_data=None,
                             critical_function=critical_function,
                             transform_cols: Tuple[int, ...] = (0,),
                             xl=xl,
                             xu=xu):

    Path(model_output_path).mkdir(exist_ok=True, parents=True)

    y_reg   = y
    y_class = compute_criticality_points(y_reg, critical_function)

    gt_fit           = True
    gt_suffix        = "_gt" if gt_fit else ""
    data_folder_name = f"points_{datetime.now().strftime('%Y%m%d')}"

    if balance:
        print("=== Balancing classes ===")
        critical_mask     = y_class == True
        non_critical_mask = y_class == False

        n_critical     = int(np.sum(critical_mask))
        n_non_critical = int(np.sum(non_critical_mask))
        print(f"Original — Critical: {n_critical}, Non-critical: {n_non_critical}")

        n_samples_per_class  = min(n_critical, n_non_critical)
        critical_indices     = np.where(critical_mask)[0]
        non_critical_indices = np.where(non_critical_mask)[0]

        np.random.seed(42)
        selected_critical     = np.random.choice(critical_indices,     n_samples_per_class, replace=False)
        selected_non_critical = np.random.choice(non_critical_indices, n_samples_per_class, replace=False)

        balanced_indices = np.concatenate([selected_critical, selected_non_critical])
        np.random.shuffle(balanced_indices)

        X       = X[balanced_indices]
        y_reg   = y_reg[balanced_indices]
        y_class = y_class[balanced_indices]

        print(f"Balanced — Critical: {np.sum(y_class==True)}, Non-critical: {np.sum(y_class==False)}")
        print(f"Total balanced samples: {len(X)}")

    X_resampled, y_reg_resampled = apply_smote_with_fit_no_sidecol(
        X, y_reg, y_class,
        apply_smote,
        data_folder_name,
        gt_fit=gt_fit,
        save_folder=model_output_path,
        desired_valid_samples=max(0, int(np.sum(y_class == False)) - int(np.sum(y_class == True))),
    )

    if apply_smote:
        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled, y_reg_resampled, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

    if validation_data is not None:
        X_val, y_val = validation_data[0], validation_data[1]

    y_all    = np.concatenate([y_train, y_val], axis=0)
    y_labels = compute_criticality_points(y_all, critical_function)
    print("=== Data before training ===")
    total_samples = len(y_labels)
    print(f"Total number of train samples: {len(y_train)}")
    print(f"Total number of samples:       {total_samples}")
    print(f"Number critical samples:       {np.sum(y_labels)}")
    print(f"Number non-critical samples:   {total_samples - np.sum(y_labels)}")

    x_scaler       = MinMaxScaler(min_val=np.array(xl), max_val=np.array(xu), clip=False)
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled   = x_scaler.transform(X_val)

    f1_shift = 0.0
    print(f"f1_shift: {f1_shift}  |  transform_cols: {transform_cols}")
    y_train_scaled = transform_targets(y_train, f1_shift, transform_cols=transform_cols)
    y_val_scaled   = transform_targets(y_val,   f1_shift, transform_cols=transform_cols)

    # ── NaN guard ────────────────────────────────────────────────────
    assert_no_nan(y_train_scaled, label="y_train_scaled")
    assert_no_nan(y_val_scaled,   label="y_val_scaled")
    # ─────────────────────────────────────────────────────────────────

    print(f"\n=== Training and Evaluating Model: {model_name} ===")
    model = get_model(model_name, X_train_scaled, y_train_scaled)

    if model_name == "MLP":
        model.fit(X_train_scaled, y_train_scaled,
                  validation_data=(X_val_scaled, y_val_scaled),
                  epochs=1000, batch_size=64, verbose=1,
                  callbacks=[EarlyStopping(patience=50, restore_best_weights=True)])
        model.save(f"{model_output_path}{model_name}_latest.h5")
        y_pred_scaled = model.predict(X_val_scaled)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        joblib.dump(model, f"{model_output_path}{model_name}_latest.pkl")
        y_pred_scaled = model.predict(X_val_scaled)

    y_pred     = inverse_transform_targets(y_pred_scaled, f1_shift, transform_cols=transform_cols)
    y_val_orig = inverse_transform_targets(y_val_scaled,  f1_shift, transform_cols=transform_cols)

    mae = np.mean(np.abs(y_pred - y_val_orig), axis=0)
    mae_str = "  |  ".join([f"F{i+1}: {v:.4f}" for i, v in enumerate(mae)])
    print(f"MAE — {mae_str}")

    return model, (mae, X_train, X_val, y_train, y_val)


# =========================
# === TRAINING FUNCTION ===
# =========================

def train_model(model_name, data_folder,
                model_output_path="./model_out/",
                apply_smote=True,
                name_suffix="",
                balance=False,
                critical_function=critical_function,
                transform_cols: Tuple[int, ...] = (0,)):

    data_folder_name = os.path.basename(os.path.normpath(data_folder))

    print("=== Load and Preprocess ===")
    df, input_cols, target_cols, xl_fmt, xu_fmt = load_and_preprocess_data(data_folder)

    print("=== Compute criticality ===")
    label_values = compute_criticality(df, target_cols, critical_function=critical_function)

    X       = df[input_cols].values
    y_reg   = df[target_cols].values
    y_class = np.asarray(label_values)

    gt_fit    = True
    gt_suffix = "_gt" if gt_fit else ""

    Path(model_output_path).mkdir(exist_ok=True, parents=True)

    if balance:
        print("=== Balancing classes ===")
        critical_mask     = y_class == True
        non_critical_mask = y_class == False

        n_critical     = np.sum(critical_mask)
        n_non_critical = np.sum(non_critical_mask)
        print(f"Original — Critical: {n_critical}, Non-critical: {n_non_critical}")

        n_samples_per_class  = min(n_critical, n_non_critical)
        critical_indices     = np.where(critical_mask)[0]
        non_critical_indices = np.where(non_critical_mask)[0]

        np.random.seed(42)
        selected_critical     = np.random.choice(critical_indices,     n_samples_per_class, replace=False)
        selected_non_critical = np.random.choice(non_critical_indices, n_samples_per_class, replace=False)

        balanced_indices = np.concatenate([selected_critical, selected_non_critical])
        np.random.shuffle(balanced_indices)

        X       = X[balanced_indices]
        y_reg   = y_reg[balanced_indices]
        y_class = y_class[balanced_indices]

        print(f"Balanced — Critical: {np.sum(y_class==True)}, Non-critical: {np.sum(y_class==False)}")
        print(f"Total balanced samples: {len(X)}")

    X_resampled, y_reg_resampled = apply_smote_with_fit_no_sidecol(
        X, y_reg, y_class,
        apply_smote,
        data_folder_name,
        gt_fit=gt_fit,
        save_folder=model_output_path,
        desired_valid_samples=np.sum(y_class == False) - np.sum(y_class == True)
    )

    if apply_smote:
        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled, y_reg_resampled, test_size=0.2, random_state=42
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )

    y_all    = np.concatenate([y_train, y_val], axis=0)
    y_labels = compute_criticality_points(y_all, critical_function)
    print("=== Data before training ===")
    total_samples = len(y_labels)
    print(f"Total number of train samples: {len(y_train)}")
    print(f"Total number of samples:       {total_samples}")
    print(f"Number critical samples:       {np.sum(y_labels)}")
    print(f"Number non-critical samples:   {total_samples - np.sum(y_labels)}")

    # Use format-specific bounds from data
    x_scaler       = MinMaxScaler(min_val=xl_fmt, max_val=xu_fmt, clip=False)
    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled   = x_scaler.transform(X_val)

    f1_shift = 0.0
    print(f"f1_shift: {f1_shift}  |  transform_cols: {transform_cols}")
    print(f"[debug] y_train min per col : {y_train.min(axis=0)}")
    print(f"[debug] y_train max per col : {y_train.max(axis=0)}")

    y_train_scaled = transform_targets(y_train, f1_shift, transform_cols=transform_cols)
    y_val_scaled   = transform_targets(y_val,   f1_shift, transform_cols=transform_cols)

    # ── NaN guard ────────────────────────────────────────────────────
    assert_no_nan(y_train_scaled, label="y_train_scaled")
    assert_no_nan(y_val_scaled,   label="y_val_scaled")
    # ─────────────────────────────────────────────────────────────────

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    suffix    = f"{data_folder_name}{'_smote' if apply_smote else ''}{gt_suffix}"
    print(f"\n=== Training and Evaluating Model: {model_name} ===")

    model = get_model(model_name, X_train_scaled, y_train_scaled)

    if model_name == "MLP":
        model.fit(X_train_scaled, y_train_scaled,
                  validation_data=(X_val_scaled, y_val_scaled),
                  epochs=1000, batch_size=64, verbose=1,
                  callbacks=[EarlyStopping(patience=50, restore_best_weights=True)])
        model.save(f"{model_output_path}{model_name}_{suffix}_{timestamp}{name_suffix}.h5")
        y_pred_scaled = model.predict(X_val_scaled)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        joblib.dump(model, f"{model_output_path}{model_name}_{suffix}_{timestamp}.pkl")
        y_pred_scaled = model.predict(X_val_scaled)

    y_pred     = inverse_transform_targets(y_pred_scaled, f1_shift, transform_cols=transform_cols)
    y_val_orig = inverse_transform_targets(y_val_scaled,  f1_shift, transform_cols=transform_cols)

    mae = np.mean(np.abs(y_pred - y_val_orig), axis=0)
    mae_str = "  |  ".join([f"F{i+1}: {v:.4f}" for i, v in enumerate(mae)])
    print(f"MAE — {mae_str}")

    return model, mae, X_train, X_val, y_train, y_val


# =========================
# === MAIN ===
# =========================

def main():
    args = parse_args()
    data_folder    = args.data_folder
    apply_smote    = args.apply_smote
    balance        = args.balance
    transform_cols = tuple(args.transform_cols)
    data_folder_name = os.path.basename(os.path.normpath(data_folder))

    gt_suffix = "_balanced" if balance else ""
    print("data folder used:", data_folder)
    print(f"Apply SMOTE:     {apply_smote}")
    print(f"Balance classes: {balance}")
    print(f"Transform cols:  {transform_cols}")
    results = []

    suffix = f"{data_folder_name}{'_smote' if apply_smote else ''}{gt_suffix}"

    for model_name in args.models:
        print(f"\n=== Training and Evaluating Model: {model_name} ===")
        model, mae, X_train, X_val, y_train, y_val = train_model(
            model_name=model_name,
            data_folder=data_folder,
            apply_smote=apply_smote,
            model_output_path="./surrogate_log/out/",
            name_suffix="",
            balance=balance,
            critical_function=critical_function,
            transform_cols=transform_cols,
        )
        results.append((model_name, *mae))

    # Dynamic column names — works for any number of objectives
    n_obj       = len(results[0]) - 1
    target_cols = [f"Fitness{i+1}" for i in range(n_obj)]
    results_df  = pd.DataFrame(results, columns=["Model"] + target_cols)
    print("\n=== MAE Results ===")
    print(results_df)

    mae_output_file = os.path.join(save_folder, f"mae_results_log_{suffix}.csv")
    results_df.to_csv(mae_output_file, index=False)
    print(f"\nResults saved to: {mae_output_file}")


if __name__ == "__main__":
    main()