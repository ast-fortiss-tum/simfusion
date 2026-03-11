from dataclasses import dataclass
from typing import Dict, List
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log
import copy
import time
from predictor.agreement_predictor_cr_beamng import AgreementPredictor


@dataclass
class MultiSimPredictProblem(Problem):
    """ Basic problem class for ADAS problems """
    
    def __init__(self,
        xl: List[float],
        xu: List[float],
        scenario_path: str,
        fitness_function: Fitness,
        simulate_function: List,
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
        predictor: AgreementPredictor = None,
        run_lofi_critical_in_hifi: bool = True
        ):

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
        assert len(simulate_function) > 0
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

        self.predictor = predictor
        self.run_lofi_critical_in_hifi = run_lofi_critical_in_hifi

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

        self.metrics_start_time = None

        self.metrics_n_total = 0
        self.metrics_time_to_first_flagged_failure = None

        self.metrics_time_to_first_hifi_failure = None
        self.metrics_n_hifi_eval = 0
        self.metrics_n_hifi_fail = 0

        self.metrics_n_lofi_only = 0          # HiFi skipped
        self.metrics_n_lofi_skip_safe = 0     # Agreement + LoFi safe -> skip
        self.metrics_n_lofi_to_hifi = 0
        self.metrics_n_flagged_fail = 0
       # Agreement + LoFi critical -> HiFi

        self.metrics_n_pred_agreement = 0
        self.metrics_n_pred_disagreement = 0

        self.metrics_n_flagged_pre_hifi = 0
        self.metrics_time_to_first_flagged_pre_hifi = None


    def _evaluate(self, x, out, *args, **kwargs):
        if self.metrics_start_time is None:
            self.metrics_start_time = time.time()

        self.counter += 1
        log.info(f"Running evaluation number {self.counter}")

        lofi_sim_simulate = self.simulate_function[0]
        hifi_sim_simulate = self.simulate_function[1]

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

        for i, xi in enumerate(x):
            print(f"Evaluating individual {i}, xi = {xi}")
            # Predictor decides which fidelity to execute
            try:
                prediction = self.predictor.predict(np.asarray(xi, dtype=float))
            except Exception as e:
                log.info("Exception in predictor occurred:")
                raise e

            out["P"][i] = prediction  # Save prediction
            
            if prediction == 1:
                print(">> DECISION: Agreement Predicted -> Running LoFi (CommonRoad) First")
                try:
                    simout_lofi = lofi_sim_simulate(
                        [xi],
                        self.simulation_variables,
                        self.scenario_path,
                        sim_time=self.simulation_time,
                        time_step=0.1,
                        do_visualize=self.do_visualize,
                    )

                except Exception as e:
                    log.info("Exception during LoFi simulation occurred:")
                    raise e
                simout_lofi = simout_lofi[0]

                out["SO_LOFI"][i] = simout_lofi

                fitness_lofi = np.asarray(self.signs) * np.array(
                    self.fitness_function.eval(simout_lofi, **kwargs)
                )
                fitness_lofi = np.asarray(fitness_lofi)
                out["F_LOFI"][i] = fitness_lofi
                is_critical_lofi = self.critical_function.eval(fitness_lofi, 
                                                               simout=simout_lofi)
                
                out["CB_LOFI"][i] = is_critical_lofi

                if bool(is_critical_lofi):
                    self.metrics_n_flagged_pre_hifi += 1
                    if self.metrics_time_to_first_flagged_pre_hifi is None:
                        self.metrics_time_to_first_flagged_pre_hifi = time.time() - self.metrics_start_time


                ##################
                
                if is_critical_lofi and self.run_lofi_critical_in_hifi:
                    self.metrics_n_lofi_to_hifi += 1
                    log.info(">> DECISION: Agreement + LoFi Critical -> Running HiFi (BeamNG) for Verification")
                    try:
                        simout_hifi = hifi_sim_simulate(
                            [xi],
                            self.simulation_variables,
                            self.scenario_path,
                            sim_time=self.simulation_time,
                            time_step=0.1,
                            do_visualize=self.do_visualize,
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
                else:
                    log.info(">> DECISION: Agreement + LoFi Safe -> SKIPPING HiFi")  
                    # Populate unified output
                    out["SO"][i] = simout_lofi
                    out["F"][i] = fitness_lofi
                    out["CB"][i] = is_critical_lofi 
            else:
                log.info(">> DECISION: Disagreement Predicted -> Running HiFi (BeamNG) Directly")
                # disagreement
                try:
                    simout_hifi = hifi_sim_simulate(
                        [xi],
                        self.simulation_variables,
                        self.scenario_path,
                        sim_time=self.simulation_time,
                        time_step=0.1,
                        do_visualize=self.do_visualize,
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
            
            # --- METRICS UPDATE (per individual) ---
            self.metrics_n_total += 1

            if prediction == 1:
                self.metrics_n_pred_agreement += 1
            else:
                self.metrics_n_pred_disagreement += 1

            # flagged = final decision (LoFi-only or HiFi) -> out["CB"][i]
            if bool(out["CB"][i]):
                self.metrics_n_flagged_fail += 1
                if self.metrics_time_to_first_flagged_failure is None:
                    self.metrics_time_to_first_flagged_failure = time.time() - self.metrics_start_time

            did_hifi = out["SO_HIFI"][i] is not None
            if did_hifi:
                self.metrics_n_hifi_eval += 1
                if bool(out["CB_HIFI"][i]):
                    self.metrics_n_hifi_fail += 1
                    if self.metrics_time_to_first_hifi_failure is None:
                        self.metrics_time_to_first_hifi_failure = time.time() - self.metrics_start_time
            else:
                self.metrics_n_lofi_only += 1
                if prediction == 1 and not bool(out["CB_LOFI"][i]):
                    self.metrics_n_lofi_skip_safe += 1

        # input("Input to continue...")
        out["F"] = np.array(out["F"])
        out["F_LOFI"] = np.array(out["F_LOFI"])
        out["F_HIFI"] = np.array(out["F_HIFI"])

        
    def is_simulation(self):
        return True
    
    # to be used if problem already initalized
    def update_name(self, attribute, value):
        import re
        pattern = fr"{attribute}\d+"
        replacement = f"{attribute}{value}"
        updated_name = re.sub(pattern, replacement, self.problem_name)
        self.problem_name = updated_name

    def get_default_metrics(self) -> Dict[str, float]:
        """
            Returns summary metrics for the run.

            Definitions:
            - "candidate" / "test case": one evaluated individual produced by the search algorithm.
            - "HiFi-evaluated": the candidate was executed in the HiFi simulator (BeamNG).
            - "HiFi failure": candidate is critical according to the oracle when evaluated in HiFi.
            - "flagged failure": candidate is marked critical by the *final decision used by the algorithm*
            (can be LoFi-only or HiFi, depending on the routing policy).
        """

        # Ratio among HiFi-evaluated tests only (precision of HiFi path).
        hifi_failure_ratio = (self.metrics_n_hifi_fail / self.metrics_n_hifi_eval) if self.metrics_n_hifi_eval > 0 else 0.0

        # HiFi-confirmed failures normalized by total number of candidates generated/evaluated.
        overall_failure_ratio = (self.metrics_n_hifi_fail / self.metrics_n_total) if self.metrics_n_total > 0 else 0.0

        # Flagged failures normalized by total number of candidates.
        flagged_failure_ratio = (self.metrics_n_flagged_fail / self.metrics_n_total) if self.metrics_n_total > 0 else 0.0

        return {
            # Total number of candidates processed by the search (LoFi-only + HiFi).
            "n_total_candidates": self.metrics_n_total,

            # Number of candidates that were actually executed in HiFi.
            "n_hifi_evaluated": self.metrics_n_hifi_eval,

            # Number of HiFi-confirmed critical cases.
            "n_hifi_failures": self.metrics_n_hifi_fail,

            # time (seconds) from start until the first HiFi-confirmed failure is found.
            "time_to_first_hifi_failure_sec": self.metrics_time_to_first_hifi_failure,

            # Fraction of HiFi-evaluated tests that are failures: n_hifi_failures / n_hifi_evaluated.
            "hifi_failure_ratio": hifi_failure_ratio,

            # Fraction of total candidates that are flagged as failures by the algorithm's final decision
            # (can include LoFi-only decisions): n_flagged_failures / n_total_candidates.
            "flagged_failure_ratio": flagged_failure_ratio,

            # Fraction of total candidates that are HiFi-confirmed failures:
            # n_hifi_failures / n_total_candidates.
            "overall_hifi_failure_ratio": overall_failure_ratio,

            # time (seconds) from start until the first "flagged failure" occurs
            # (based on out["CB"], i.e., the final decision used by the algorithm).
            "time_to_first_flagged_failure_sec": self.metrics_time_to_first_flagged_failure,

            # Number of candidates that never ran HiFi (LoFi-only path).
            "n_lofi_only": self.metrics_n_lofi_only,

            # Number of candidates skipped after LoFi because LoFi predicted safe (Agreement -> LoFi safe -> skip HiFi).
            "n_lofi_skip_safe": self.metrics_n_lofi_skip_safe,

            # Number of candidates forwarded from LoFi to HiFi due to LoFi being critical
            # (Agreement -> LoFi critical -> HiFi verification).
            "n_lofi_to_hifi": self.metrics_n_lofi_to_hifi,

            # Predictor decision counts (routing distribution).
            "n_pred_agreement": self.metrics_n_pred_agreement,
            "n_pred_disagreement": self.metrics_n_pred_disagreement,

            "n_flagged_pre_hifi": self.metrics_n_flagged_pre_hifi,
            "time_to_first_flagged_pre_hifi_sec": self.metrics_time_to_first_flagged_pre_hifi,

        }


