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

from simulations.config import critical_function, xl, xu
from surrogate.visualize_data import visualize_data
from surrogate.scaler import MinMaxScaler
from pymoo.core.population import Population
from opensbt.utils.evaluation import evaluate_individuals
from opensbt.problem.adas_problem import ADASProblem
from simulations.carla.carla_runner_sim import CarlaRunnerSimulator
from simulations.config import scenario_file, simulation_variables, fitness_function, sim_time, sampling_time
from pathlib import Path

# === CONFIGURATION ===
input_columns = ["PedSpeed", "EgoSpeed", "PedDist"]
optional_columns =  ["Fitness_Min distance", "Fitness_Velocity at min distance"]
desired_columns = ["Fitness_Min distance_HiFi", "Fitness_Velocity at min distance_HiFi"]
output_folder = "./surrogate/plots/"
model_output_path = "./surrogate/out/model_"
save_folder = "./surrogate/out/"

def parse_args():
    parser = argparse.ArgumentParser(description="Train surrogate model")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=["GL", "GNL", "RF", "MLP"],
                        help="List of surrogate models to evaluate: GL, GNL, RF, MLP")
    parser.add_argument("--sample", type=float, nargs="+",
                        help=f"Input sample as {len(input_columns)} floats: {' '.join(input_columns)}")
    parser.add_argument("--data_folder", type=str, default="./surrogate/data/batch0/",
                        help="Path to the folder containing training CSV files")
    parser.add_argument("--apply_smote", action="store_true", default=False,
                        help="Apply SMOTE to balance classes (default: False)")
    return parser.parse_args()

def load_and_preprocess_data(data_folder):
    csv_files = glob(f"{data_folder}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Determine which target columns to use
    if optional_columns[0] in df.columns:
        target_cols = optional_columns
    else:
        target_cols = desired_columns

    df = df[input_columns + target_cols].dropna()
    # Remove rows with dummy values 1000 in target columns
    for col in target_cols:
        df = df[df[col] != 1000].reset_index(drop=True)

    return df, target_cols

def compute_criticality(df, target_cols, critical_function):
    fitness_tuples = list(df[target_cols].itertuples(index=False, name=None))
    label_values = [critical_function.eval(np.asarray(t)) for t in fitness_tuples]
    # critical_tests = [t for t in fitness_tuples if critical_function.eval(np.asarray(t))] 
    # print("=== Compute criticality points ===")
    # print(f"num tests (training): {len(df)}")
    # print(f"num critical: {len(critical_tests)}")
    # print(f"non critical tests: {len(df) - len(critical_tests)}")
    return np.asarray(label_values)

def compute_criticality_points(F, critical_function):
    label_values = [critical_function.eval(p) for p in F]
    # print("=== Compute criticality points ===")
    # print(f"num tests (training): {len(label_values)}")
    # print(f"num critical: {np.sum(label_values)}")
    # print(f"non critical tests: {len(label_values) - np.sum(label_values)}")
    return np.asarray(label_values)

def apply_smote_with_fit(X, y_reg, y_class, apply_smote, data_folder_name,
                         gt_fit = True, save_folder= None):
    if not apply_smote:
        return X, y_reg

    gt_suffix = "_gt" if gt_fit else ""

    smote = SMOTE(random_state=42)
    X_resampled, y_class_resampled = smote.fit_resample(X, y_class)

    if gt_fit:
        problem = ADASProblem(
            problem_name="dummy",
            scenario_path=os.path.join(os.getcwd(), "scenarios", scenario_file),
            xl=xl,
            xu=xu,
            simulation_variables=simulation_variables,
            fitness_function=fitness_function,
            critical_function=critical_function,
            simulate_function=CarlaRunnerSimulator.simulate,
            simulation_time=sim_time,
            sampling_time=sampling_time,
            approx_eval_time=10,
            do_visualize=True
        )

        n_orig = len(X)
        n_resampled = len(X_resampled)
        synthetic_indices = np.arange(n_orig, n_resampled)

        synthetic_X = X_resampled[synthetic_indices]
        population = Population.new("X", synthetic_X)
        print(f"total tests: {n_resampled}")
        print(f"Running HiFi on {len(synthetic_indices)} samples for SMOTE.")

        for ind in population:
            evaluate_individuals(Population(individuals = [ind]), 
                                problem=problem, 
                                backup_folder=save_folder if save_folder is not None else "./model/init/")

        F = population.get("F")
        CB = population.get("CB").flatten()

        valid_mask = F[:, 0] != 1000
        valid_synthetic_X = synthetic_X[valid_mask]
        valid_F = F[valid_mask]
        valid_CB = CB[valid_mask]

        X_resampled = np.vstack([X, valid_synthetic_X])
        y_reg_resampled = np.vstack([y_reg, valid_F])
        y_class_resampled = np.hstack([y_class, valid_CB])

        print(f"[INFO] Kept {len(valid_synthetic_X)} / {len(synthetic_X)} synthetic samples (F != 1000)")

    else:
        from sklearn.neighbors import NearestNeighbors

        n_orig = len(X)
        n_resampled = len(X_resampled)
        synthetic_indices = np.arange(n_orig, n_resampled)

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X)

        synthetic_X = X_resampled[synthetic_indices]
        distances, neighbors_idx = nn.kneighbors(synthetic_X)

        y_reg_resampled = np.vstack([
            y_reg,
            y_reg[neighbors_idx.flatten()]
        ])

    visualize_data(
        X_resampled, y_class_resampled,
        filename=f"data_{data_folder_name}_smote{gt_suffix}_plot.png",
        output_folder=output_folder
    )
    print(f"Original samples: {len(X)}")
    print(f"After SMOTE samples: {len(X_resampled)}")
    visualize_data(X_resampled, y_class_resampled, filename="data_with_smote_plot.png", output_folder=output_folder)

    return X_resampled, y_reg_resampled

def apply_smote_with_fit_no_sidecol(X, y_reg, y_class, apply_smote, data_folder_name,
                         gt_fit=True, save_folder=None, desired_valid_samples=100, max_iterations=100):
    if not apply_smote:
        return X, y_reg

    gt_suffix = "_gt" if gt_fit else ""

    # Store accumulated valid synthetic data
    valid_synthetic_X_all = []
    valid_F_all = []
    valid_CB_all = []

    current_X = X.copy()
    current_y_class = y_class.copy()
    iteration = 0

    # desired_valid_samples = 5

    print("[SMOTE] desired_valid_samples is:", desired_valid_samples)
    
    while len(valid_synthetic_X_all) < desired_valid_samples and iteration < max_iterations:
        print(f"[INFO] SMOTE iteration {iteration+1}")
        import random
        smote = SMOTE(random_state=random.randint(0,10**7))
        X_resampled, y_class_resampled = smote.fit_resample(current_X, current_y_class)

        n_orig = len(current_X)
        n_resampled = len(X_resampled)
        synthetic_indices = np.arange(n_orig, n_resampled)
        
        print("synthetic indices:", synthetic_indices)
        # print("Selected X:", X_resampled[synthetic_indices])
        print("type selected X shape:", X_resampled[synthetic_indices[0]:synthetic_indices[0] + 1].shape)
        print("synthetic x:", X_resampled[synthetic_indices[0]:synthetic_indices[0] + 1])
        synthetic_X = X_resampled[synthetic_indices[0]:synthetic_indices[0] + 1]  # select only one (we repeat SMOTE anyways)

        if gt_fit:
            problem = ADASProblem(
                problem_name="dummy",
                scenario_path=os.path.join(os.getcwd(), "scenarios", scenario_file),
                xl=xl,
                xu=xu,
                simulation_variables=simulation_variables,
                fitness_function=fitness_function,
                critical_function=critical_function,
                simulate_function=CarlaRunnerSimulator.simulate,
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

            F = population.get("F")
            CB = population.get("CB").flatten()

            valid_mask = F[:, 0] != 1000
            valid_synthetic_X = synthetic_X[valid_mask]
            valid_F = F[valid_mask]
            valid_CB = CB[valid_mask]

            valid_synthetic_X_all.append(valid_synthetic_X)
            valid_F_all.append(valid_F)
            valid_CB_all.append(valid_CB)

            log.info(f"SMOTE valid x: {len(valid_synthetic_X_all)}")

            log.info(f"[INFO] Kept {len(valid_synthetic_X)} / {len(synthetic_X)} synthetic samples (F != 1000)")
        else:
            raise ValueError("Incremental SMOTE with fitness evaluation requires gt_fit=True")

        # Update for next iteration to avoid repeatedly sampling the same inputs
        # current_X = X_resampled
        # current_y_class = y_class_resampled
        iteration += 1

    # Combine all valid synthetic data
    if valid_synthetic_X_all:
        valid_synthetic_X = np.vstack(valid_synthetic_X_all)
        valid_F = np.vstack(valid_F_all)
        valid_CB = np.hstack(valid_CB_all)

        X_final = np.vstack([X, valid_synthetic_X])
        y_reg_final = np.vstack([y_reg, valid_F])
        y_class_final = np.hstack([y_class, valid_CB])
    else:
        print("[WARN] No valid synthetic samples found. Returning original data.")
        X_final = X
        y_reg_final = y_reg
        y_class_final = y_class

    # Visualization
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

    print(f"[INFO] Original samples: {len(X)}")
    print(f"[INFO] Final dataset size: {len(X_final)} (including {len(X_final) - len(X)} valid synthetic samples)")

    return X_final, y_reg_final
 
def get_model(name, X_train_scaled, y_train_scaled):
    if name == "GL":
        kernel = DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-2, 10)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1))
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=0)
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
            class MLPHyperModel(HyperModel):
                def __init__(self, input_shape, output_shape):
                    self.input_shape = input_shape
                    self.output_shape = output_shape

                def build(self, hp):
                    model = Sequential()
                    model.add(Dense(
                        units=hp.Int('units_1', min_value=10, max_value=floor((2 / 3) * len(X_train_scaled)), step=10),
                        activation=hp.Choice('activation_1', ['relu', 'tanh', 'sigmoid']),
                        kernel_initializer=hp.Choice('init_1', ['he_uniform', 'glorot_uniform']),
                        input_shape=self.input_shape
                    ))
                    model.add(Dense(
                        units=hp.Int('units_2', min_value=10, max_value=floor((2/ 3) * len(X_train_scaled)), step=10),
                        activation=hp.Choice('activation_2', ['relu', 'tanh', 'sigmoid']),
                        kernel_initializer=hp.Choice('init_2', ['he_uniform', 'glorot_uniform']),
                    ))
                    model.add(Dense(self.output_shape, activation='linear'))
                    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    return model

            hypermodel = MLPHyperModel(input_shape=(X_train_scaled.shape[1],), output_shape=y_train_scaled.shape[1])

            tuner = BayesianOptimization(
                hypermodel,
                objective='val_loss',
                max_trials=10,
                seed=42,
                directory='tuner_dir',
                project_name='mlp_regression'
            )

            tuner.search(X_train_scaled, y_train_scaled, validation_split=0.2, epochs=50, verbose=0)
            model = tuner.get_best_models(num_models=1)[0]
            return model
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
    
def train_model_from_points(model_name, 
                            X, y,
                            model_output_path = "./model_out/",
                            name_suffix = "",
                            validation_data = None): # validation on static data
    
    Path(model_output_path).mkdir(exist_ok=True, parents=True)

    # === Filter: exclude samples where any y == 1000 ===
    mask = ~(np.any(y == 1000, axis=1))  # Keep only rows where no element is 1000
    X = X[mask]
    y = y[mask]

    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size=0.2, random_state=42)

    # Scale features and targets
    x_min = np.array(xl)
    x_max = np.array(xu)
    y_min = np.array([0, -10])
    y_max = np.array([10, 0])

    x_scaler = MinMaxScaler(min_val=x_min, max_val=x_max, clip=False)
    y_scaler = MinMaxScaler(min_val=y_min, max_val=y_max, clip=False)

    X_train_scaled = x_scaler.transform(X_train)
    if validation_data is not None:
        X_val_scaled = x_scaler.transform(validation_data[0])
    else:
        X_val_scaled = x_scaler.transform(X_val)

    y_train_scaled = y_scaler.transform(y_train)
    if validation_data is not None:
        y_val_scaled = y_scaler.transform(validation_data[1])
    else:
        y_val_scaled = y_scaler.transform(Y_val)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"\n=== Training and Evaluating Model: {model_name} ===")
    model = get_model(model_name, X_train_scaled, y_train_scaled)

    Path(model_output_path).mkdir(exist_ok=True, parents=True)

    if model_name == "MLP":
        model.fit(X_train_scaled, y_train_scaled,
                    validation_data=(X_val_scaled, y_val_scaled),
                    epochs=1000, batch_size=64, verbose=1,
                    callbacks=[EarlyStopping(patience=50, restore_best_weights=True)])
        model.save(f"{model_output_path}{model_name}_{timestamp}{name_suffix}.h5")
        y_pred_scaled = model.predict(X_val_scaled)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        joblib.dump(model,f"{model_output_path}{model_name}_{timestamp}{name_suffix}.pkl")
        y_pred_scaled = model.predict(X_val_scaled)

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_val_orig = y_scaler.inverse_transform(y_val_scaled)
    mae = np.mean(np.abs(y_pred - y_val_orig), axis=0)

    #joblib.dump(x_scaler, f"{model_output_path}{model_name}_{'fromp'}_{timestamp}_xscaler.pkl")
    #joblib.dump(y_scaler, f"{model_output_path}{model_name}_{'fromp'}_{timestamp}_yscaler.pkl")
    print("MAE is: ", mae)
    # results.append((model_name, *mae))
    return model, (mae, X_train, X_val, y_train, y_val)

def train_model(model_name, data_folder, 
                model_output_path = "./model_out/", apply_smote = True,
                name_suffix = "",
                balance = False,
                critical_function = critical_function):

    data_folder_name = os.path.basename(os.path.normpath(data_folder))

    print("=== Load and Preprocess ===")
    df, target_cols = load_and_preprocess_data(data_folder)
    
    print("=== Compute criticality ===")
    label_values = compute_criticality(df, target_cols,
                                       critical_function=critical_function)
    # print("labels: ", label_values)
    # print("critical labels:", np.sum(label_values))
    # print("target cols: ", target_cols)
    # Use a slice of data for initial visualization & SMOTE
    X = df[input_columns].values
    y_reg = df[target_cols].values
    y_class = np.asarray(label_values)
    y_class_calculated = compute_criticality_points(y_reg,
                                       critical_function=critical_function)
    
    assert np.array_equal(y_class, y_class_calculated)

    gt_fit = True # use ground truth scores with smote
    gt_suffix = "_gt" if gt_fit else ""
    
    Path(model_output_path).mkdir(exist_ok=True, parents=True)

    X_resampled, y_reg_resampled = \
        apply_smote_with_fit_no_sidecol(
        X, y_reg, y_class, 
        apply_smote, 
        data_folder_name,
        gt_fit=gt_fit,
        save_folder=model_output_path,
        desired_valid_samples= np.sum(y_class==False)  - np.sum(y_class==True) 
    )

    if apply_smote:
        # Use full data after SMOTE for training
        X_full = df[input_columns].values
        y_full = df[target_cols].values

        X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_reg_resampled, test_size=0.2, random_state=42)
    else:
        X_full = df[input_columns].values
        y_full = df[target_cols].values

        X_train, X_val, y_train, y_val = train_test_split(X_full, y_reg, test_size=0.2, random_state=42)

        assert np.array_equal(X_full, X)
        if balance:
            print("==== Balancing datasets. ===")

            # verify
            y_class_compute = compute_criticality_points(y_full, 
                                                         critical_function=critical_function)
            y_class = np.asarray(y_class_compute)
            # Check if arrays are equal
            if not np.array_equal(y_class, y_class_compute):
                print(f"Shape y_class: {len(y_class)}, Shape y_class_compute: {len(y_class_compute)}")

                # Find indices where they differ
                mismatches = np.where(y_class != y_class_compute)[0]
                print(f"Number of mismatches: {len(mismatches)}")

                # Print first few mismatches
                for idx in mismatches[:10]:
                    print(f"Index {idx}: y_class = {y_class[idx]}, y_class_compute = {y_class_compute[idx]}")

                # Optional: assert to stop execution
                assert False, "Arrays differ. See printed mismatch information."     

            # Select indices for balancing
            crit_indices = np.where(y_class == True)[0]
            noncrit_indices = np.where(y_class == False)[0]
          
            n_crit = len(crit_indices)
            np.random.seed(42)  # for reproducibility
            selected_noncrit_indices = np.random.choice(noncrit_indices, size=n_crit, replace=False)
            assert len(selected_noncrit_indices) == n_crit

            balanced_indices = np.concatenate([crit_indices, selected_noncrit_indices])
            # print("crit indices:", crit_indices)
            # print("non crit indices:", noncrit_indices)

            # print("num crit indices:", len(crit_indices))
            # print("num non crit indices:", len(selected_noncrit_indices))
            assert len(crit_indices) == len(selected_noncrit_indices)
            assert len(np.concatenate([crit_indices, selected_noncrit_indices])) == (len(crit_indices) + len(selected_noncrit_indices))            # np.random.shuffle(balanced_indices)

            X_balanced = X[balanced_indices]
            y_balanced = y_reg[balanced_indices]

            X_train, X_val, y_train, y_val = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    y_all = np.concatenate([y_train, y_val], axis=0)
    y_labels = compute_criticality_points(y_all, critical_function)

    assert len(y_labels) == y_all.shape[0]

    print("=== Data before training ===")
    total_samples = len(y_labels)
    print(f"Total number of samples: {total_samples}")
    # print(f"Number training samples: {len(X_train)}")
    # print(f"Number validation samples: {len(X_val)}")
    print(f"Number critical samples: {np.sum(y_labels)}")
    print(f"Number non-critical samples: {total_samples - np.sum(y_labels)}")
    # Scale features and targets
    x_min = np.array(xl)
    x_max = np.array(xu)
    y_min = np.array([0, -10])
    y_max = np.array([10, 0])

    x_scaler = MinMaxScaler(min_val=x_min, max_val=x_max, clip=False)
    y_scaler = MinMaxScaler(min_val=y_min, max_val=y_max, clip=False)

    X_train_scaled = x_scaler.transform(X_train)
    X_val_scaled = x_scaler.transform(X_val)

    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    suffix = f"{data_folder_name}{'_smote' if apply_smote else ''}{gt_suffix}"
    print(f"\n=== Training and Evaluating Model: {model_name} ===")
    model = get_model(model_name, X_train_scaled, y_train_scaled)

    if model_name == "MLP":
        model.fit(X_train_scaled, y_train_scaled,
                    validation_data=(X_val_scaled, y_val_scaled),
                    epochs=1000, batch_size=64, verbose=1,
                    callbacks=[EarlyStopping(patience=50, restore_best_weights=True)])
        model.save(f"{model_output_path}{model_name}_{suffix}{timestamp}{name_suffix}.h5")
        y_pred_scaled = model.predict(X_val_scaled)
    else:
        model.fit(X_train_scaled, y_train_scaled)
        joblib.dump(model, f"{model_output_path}{model_name}_{suffix}_{timestamp}.pkl")
        y_pred_scaled = model.predict(X_val_scaled)

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_val_orig = y_scaler.inverse_transform(y_val_scaled)
    mae = np.mean(np.abs(y_pred - y_val_orig), axis=0)

    # joblib.dump(x_scaler, f"{model_output_path}{model_name}_{suffix}_{timestamp}_xscaler.pkl")
    # joblib.dump(y_scaler, f"{model_output_path}{model_name}_{suffix}_{timestamp}_yscaler.pkl")

    # results.append((model_name, *mae))
    return model, mae, X_train, X_val, y_train, y_val 

def main():
    args = parse_args()
    data_folder = args.data_folder
    apply_smote = args.apply_smote
    data_folder_name = os.path.basename(os.path.normpath(data_folder))

    gt_suffix = "_balanced"
    # visualize_data(
    #     X, y_class,
    #     filename=f"data_{data_folder_name}_nosmote_plot.png",
    #     output_folder=output_folder
    # )
    print("data folder used:", data_folder)
    results = []
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    suffix = f"{data_folder_name}{'_smote' if apply_smote else ''}{gt_suffix}"

    for model_name in args.models:
        print(f"\n=== Training and Evaluating Model: {model_name} ===")
        model, mae, X_train, X_val, y_train, y_val  = train_model(model_name=model_name,
                    data_folder=data_folder,
                    apply_smote=apply_smote,
                    model_output_path="./surrogate/out/",
                    name_suffix="",
                    balance=True,
                    critical_function=critical_function
                    )
        results.append((model_name, *mae))

    target_cols = ["Fitness1", "Fitness2"]
    results_df = pd.DataFrame(results, columns=["Model"] + target_cols)
    print("\n=== MAE Results ===")
    print(results_df.to_string(index=False))

    mae_output_file = os.path.join(save_folder, f"mae_results_{suffix}.csv")    
    results_df.to_csv(mae_output_file, index=False)

if __name__ == "__main__":
    main()