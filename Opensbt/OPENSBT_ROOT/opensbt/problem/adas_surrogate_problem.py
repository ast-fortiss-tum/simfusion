from dataclasses import dataclass
from typing import Dict, List
from surrogate.scaler import MinMaxScaler
from surrogate.test_surrogate_cr_bng import compute_criticality_points, train_model, train_model_from_points
#from surrogate.model import Model
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log
import copy
from opensbt.simulation.simulator import SimulationOutput
from pandas.core.frame import DataFrame
import os, csv, json
from pathlib import Path


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
    """ Basic problem class for ADAS problems """
    
    def __init__(self,
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
        apply_smote: bool = False):

        """Generates a simulation-based testing problem.

        :param xl: Lower bound for the search domain.
        :type xl: List[float]
        :param xu: Upper bound for the search domain.
        :type xu: List[float]
        :param scenario_path: The path to the scenario to be simulated.
        :type scenario_path: str
        :param fitness_function: The instance of the fitness function to evaluate simulations.
        :type fitness_function: Fitness
        :param simulate_function: The pointer to the simulate function of the simulator.
        :type simulate_function: _type_
        :param critical_function: The instance of the oracle function to assign pass fail verdicts.
        :type critical_function: Critical
        :param simulation_variables: The name of the simulation variables to alter for test generation.
        :type simulation_variables: List[float]
        :param simulation_time: The simulation time for the test execution, defaults to 10
        :type simulation_time: float, optional
        :param sampling_time: The sampling time for the simulator, defaults to 100
        :type sampling_time: float, optional
        :param design_names: The name for the search variable to use in the output. If none is said, simulation_variables is used, defaults to None
        :type design_names: List[str], optional
        :param objective_names: The name of the objectives to use in the output, defaults to None
        :type objective_names: List[str], optional
        :param problem_name: The name of the problem, defaults to None
        :type problem_name: str, optional
        :param other_parameters: Other necessary parameters, defaults to None
        :type other_parameters: Dict, optional
        :param approx_eval_time: The approximate evaluation time for one test execution, defaults to None
        :type approx_eval_time: float, optional
        :param do_visualize: Visualize test execution or not, defaults to False
        :type do_visualize: bool, optional
        :raises ValueError: Raises ValueError if min_or_max of fitness function is not correctly configured.
        """

        super().__init__(n_var=len(xl),
                         n_obj=len(fitness_function.name),
                         xl=xl,
                         xu=xu)

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
        
        #################

        if design_names is not None:
            self.design_names = design_names
        else:
            self.design_names = simulation_variables

        if objective_names is not None:
            self.objective_names = objective_names
        else:
            self.objective_names = fitness_function.name

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
                    "Error: The optimization property " + str(value) + " is not supported.")

        self.counter = 0
        self.initialized = False

    def _init_decision_log(self, save_folder: str):
        outdir = Path(save_folder) / "surrogate_logs"
        outdir.mkdir(parents=True, exist_ok=True)

        self._decision_csv = str(outdir / "decisions.csv")

        if not os.path.isfile(self._decision_csv):
            with open(self._decision_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "eval", 
                    "cap_d", "cap_g",        # cap values
                    "decision",              # SKIP_HIFI / RUN_HIFI
                    "pred_d", "pred_g", 
                    "err_d", "err_g", 
                    "cb_pred",               # critical(pred)
                    "cb_ci",
                    "hifi_d", "hifi_g", 
                    "cb_hifi",               # critical(hifi)
                ])

    def _append_decision_row(self, row: list):
        with open(self._decision_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _compute_cap(self, y_full: np.ndarray, q: float = 95.0, min_cap=(1e-6, 1e-6)):
        # compute cap values for each objective based on q-th percentile of training data
        y_full = np.asarray(y_full, dtype=float)
        cap = np.nanpercentile(y_full, q=q, axis=0)   # shape (2,)
        cap = np.maximum(cap, np.array(min_cap, dtype=float))
        return cap

    def _fitness_in_confidence_int(self,
                                   critical_function, 
                                   fitness: np.ndarray, 
                                   error: np.ndarray):
        fitness = np.asarray(fitness)
        print("fitness is:", fitness)
        error = np.asarray(error)
        print("error is:", error)

        low  = np.clip(fitness - error, [0.0, 0.0], self.cap)
        high = np.clip(fitness + error, [0.0, 0.0], self.cap)

        safest = np.array([high[0], low[1]])

        return critical_function.eval(safest, simout=None)
        
    def predict_scaled(self, model, x, 
                       x_min, 
                       x_max, 
                       y_min, 
                       y_max):
        x_scaler = MinMaxScaler(min_val=x_min, max_val=x_max, clip=False)
        y_scaler = MinMaxScaler(min_val=y_min, max_val=y_max, clip=False)
        
        x_scaled = x_scaler.transform(x)
        y_pred = model.predict(x_scaled)
        y = y_scaler.inverse_transform(y_pred)

        print("predicted y:", y)

        return y

    def _initialize_surrogate(self, apply_smote = True, model_output_path = "./model_out/"):
        model, error, X_train, X_val, y_train, y_val = train_model(
                                model_name = self.model_name,
                                data_folder = self.data_folder,
                                apply_smote=apply_smote,
                                model_output_path=model_output_path
                            )
        
        X_full = np.concatenate([X_train, X_val], axis=0)
        y_full = np.concatenate([y_train, y_val], axis=0)

        self.cap = self._compute_cap(y_full, q=95.0)

        print("Surrogate model initialized on dataset size: ", X_full.shape)
        print("CAP:", self.cap)
        return model, error, X_full, y_full
    
        # trains the model on initial data

    def append_mae_to_csv(self, filepath, step, mae, pos, neg):
        import os, csv
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        file_exists = os.path.isfile(filepath)
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['step', 'mae', 'pos', 'neg'])
            writer.writerow([step, mae, pos, neg])

    def _evaluate(self, x, out, *args, **kwargs):
        hifi_sim_simulate = self.simulate_function

        # Initialize all outputs to None
        n = len(x)
        out["SO"] = [None] * n
        out["SO_LOFI"] = [None] * n
        out["SO_HIFI"] = [None] * n
        out["F_LOFI"] =  [[None] * self.n_obj for _ in range(n)]
        out["F_HIFI"] = [[None] * self.n_obj for _ in range(n)]
        out["F"] = [None] * n
        out["CB_LOFI"] = [None] * n
        out["CB_HIFI"] = [None] * n
        out["CB"] = [None] * n
        out["P"] = [None] * n  # Prediction output

        ################
        save_folder = kwargs["algorithm"].save_folder

        if not self.initialized:
            model, error, X_data, y_data  = self._initialize_surrogate(apply_smote=self.apply_smote,
                                                                    model_output_path= save_folder + "/model_out/")
            self.data_points = [X_data, y_data]
            self.model = model
            self.model_error = error 
            self.initialized = True

            labels = compute_criticality_points(F = self.data_points[1], 
                                                    critical_function=self.critical_function)

            self.append_mae_to_csv(save_folder  + "/model_out/mae.csv", 0, 
                                   self.model_error,
                                    pos =  np.sum(labels),
                                    neg = len(self.data_points[1]) - np.sum(labels)
                                   )
            print("labels are:", labels)
            self._init_decision_log(save_folder)

        # the surrogate takes over the role of the lofi
        # Scale features and targets
        x_min = np.array(self.xl)
        x_max = np.array(self.xu)
        y_min = np.array([0, 0])
        y_max = np.array([20, 40])

        for i, xi in enumerate(x):
            print("xi is ", xi)
            self.counter += 1
            log.info(f"Running evaluation number {self.counter}")

            # Surrogates gives fitness value
            try:
                fitness_predict = self.predict_scaled(self.model,
                                                      np.asarray([xi]),
                                                      x_min=x_min,
                                                      x_max=x_max,
                                                      y_min=y_min,
                                                      y_max=y_max
                )[0]
                print("input:",np.asarray([xi]) )
                print("fitness_predict:", fitness_predict)

                # clamp to physical domain (distances cannot be negative)
                fitness_predict = np.asarray(fitness_predict, dtype=float)
                # fitness_predict[0] = max(0.0, fitness_predict[0])  # d_pred
                # fitness_predict[1] = max(0.0, fitness_predict[1])  # g_pred
                fitness_lofi = np.clip(fitness_predict, [0.0, 0.0], self.cap)
                
                print("fitness_lofi:", fitness_lofi)

                # fitness_lofi = np.asarray(
                #     self.signs) * np.array(fitness_predict)
                simout_lofi = None# get_surrogate_simout()
            except Exception as e:
                log.info("Exception in predictor occurred:")
                raise e
            
            out["SO_LOFI"][i] = simout_lofi
            out["F_LOFI"][i] = fitness_lofi
            is_critical_lofi = self.critical_function.eval(fitness_lofi, 
                                                               simout=simout_lofi)
            out["CB_LOFI"][i] = is_critical_lofi

            cb_ci = self._fitness_in_confidence_int(
                self.critical_function,
                fitness=fitness_lofi,
                error=self.model_error
            )
            if cb_ci:
            # if self._fitness_in_confidence_int(self.critical_function,
            #                              fitness = fitness_lofi,
            #                              error = self.model_error):
                print("Prediction: Critical, skipping HiFi simulation")
                # Populate unified output
                out["SO"][i] = simout_lofi
                out["F"][i] = fitness_lofi
                out["CB"][i] = is_critical_lofi

                self._append_decision_row([
                    self.counter,
                    float(self.cap[0]), float(self.cap[1]),
                    "SKIP_HIFI",
                    float(fitness_lofi[0]), float(fitness_lofi[1]),
                    float(self.model_error[0]), float(self.model_error[1]),
                    bool(is_critical_lofi),
                    bool(cb_ci), 
                    "", "", ""
                ])
           
            else:
                try:
                    print("Prediction: Not Confident Critical, running HiFi simulation")
                    simout_hifi = hifi_sim_simulate(
                        [xi], self.simulation_variables, self.scenario_path,
                        sim_time=self.simulation_time, time_step=0.1,
                        do_visualize=self.do_visualize
                    )
                except Exception as e:
                    log.info("Exception during HiFi simulation occurred:")
                    raise e
                
                simout_hifi = simout_hifi[0]
                out["SO_HIFI"][i] = simout_hifi
                fitness_hifi = np.asarray(self.signs) * np.array(
                    self.fitness_function.eval(simout_hifi, **kwargs)
                )
                fitness_hifi = np.asarray(fitness_hifi)

                out["F_HIFI"][i] = fitness_hifi
                is_critical_hifi = self.critical_function.eval(fitness_hifi, simout=simout_hifi)
                out["CB_HIFI"][i] = is_critical_hifi

                # Populate unified output
                out["SO"][i] = simout_hifi
                out["F"][i] = fitness_hifi
                out["CB"][i] = is_critical_hifi

                print(f"--- CHECK ---")
                print(f"Input: {xi}")
                print(f"Model Prediction (LoFi): {fitness_lofi}")
                print(f"Real Simulation (HiFi): {fitness_hifi}")
                print(f"----------------")

                self._append_decision_row([
                    self.counter,
                    float(self.cap[0]), float(self.cap[1]),
                    "RUN_HIFI",
                    float(fitness_lofi[0]), float(fitness_lofi[1]),
                    float(self.model_error[0]), float(self.model_error[1]),
                    bool(is_critical_lofi),
                    bool(cb_ci), 
                    float(fitness_hifi[0]), float(fitness_hifi[1]),
                    bool(is_critical_hifi)
                ])

                # if abs(fitness_hifi[0]) == 1000:
                #     # ignore side collisions
                #     log.info("[Problem] Test produces side collision. Ignoring.")
                #     pass
                #else:
                # include sample in dataset 
                # X repreents xi
                # y represents    out["F"][i] 
                xi_array = np.asarray([xi])  # shape (1, n_features)
                fitness_array = np.asarray([out["F"][i]])  # shape (1, n_targets) or (1,) if scalar

                print("dataset old size: ", self.data_points[0].shape)

                self.data_points[0] = np.concatenate([self.data_points[0], xi_array], axis=0)
                self.data_points[1] = np.concatenate([self.data_points[1], fitness_array], axis=0)

                print("dataset updated, new size: ", self.data_points[0].shape)
                # Recompute cap with the updated dataset
                self.cap = self._compute_cap(self.data_points[1], q=95.0)
                print("CAP updated:", self.cap)

                model, meta = train_model_from_points(model_name=self.model_name,
                                                        X = self.data_points[0],
                                                        y = self.data_points[1],
                                                        model_output_path=save_folder + "/model_out/",
                                                        name_suffix=f"_eval-{self.counter}") 
                self.model = model
                self.model_error = meta[0]
                
                labels = compute_criticality_points(F = self.data_points[1], 
                                                    critical_function=self.critical_function)
                print("labels are:", labels)
                
                self.append_mae_to_csv(save_folder  + "/model_out/mae.csv",
                                    self.counter, 
                                    self.model_error,
                                    pos =  np.sum(labels),
                                    neg = len(self.data_points[1]) - np.sum(labels)
                                    )

                print("new surrogate model created")

    def is_simulation(self):
        return True
    
    # to be used if problem already initalized
    def update_name(self, attribute, value):
        import re
        pattern = fr"{attribute}\d+"
        replacement = f"{attribute}{value}"
        updated_name = re.sub(pattern, replacement, self.problem_name)
        self.problem_name = updated_name