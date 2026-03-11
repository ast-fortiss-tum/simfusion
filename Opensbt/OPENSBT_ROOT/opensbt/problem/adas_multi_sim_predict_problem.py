from dataclasses import dataclass
from typing import Dict, List
from pymoo.core.problem import Problem
import numpy as np
from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
import logging as log
import copy
from predictor.agreement_predictor import AgreementPredictor


@dataclass
class ADASMultiSimPredictProblem(Problem):
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

    def _evaluate(self, x, out, *args, **kwargs):
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
            print("xi is ", xi)
            # Predictor decides which fidelity to execute
            try:
                prediction = self.predictor.predict(np.asarray([xi]))
            except Exception as e:
                log.info("Exception in predictor occurred:")
                raise e

            out["P"][i] = int(prediction)  # Save prediction

            assert prediction in [0,1]
            
            if prediction == 1: 
                # critical or non-critical agreement
                try:
                    simout_lofi = lofi_sim_simulate(
                        [xi], self.simulation_variables, self.scenario_path,
                        sim_time=self.simulation_time, time_step=self.sampling_time,
                        do_visualize=self.do_visualize
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

                ##################
                
                if is_critical_lofi and self.run_lofi_critical_in_hifi:
                    try:
                        simout_hifi = hifi_sim_simulate(
                            [xi], self.simulation_variables, self.scenario_path,
                            sim_time=self.simulation_time, time_step=self.sampling_time,
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
                else:     
                    # Populate unified output
                    out["SO"][i] = simout_lofi
                    out["F"][i] = fitness_lofi
                    out["CB"][i] = is_critical_lofi
                

            else: 
                # disagreement
                try:
                    simout_hifi = hifi_sim_simulate(
                        [xi], self.simulation_variables, self.scenario_path,
                        sim_time=self.simulation_time, time_step=self.sampling_time,
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
            
            # input("Input to continue...")
        out["F"] = np.array(out["F"])
        out["F_LOFI"] = np.array(out["F_LOFI"])
        out["F_HIFI"] = np.array(out["F_HIFI"])

        log.info(f"F: {out['F']}")

        
    def is_simulation(self):
        return True
    
    # to be used if problem already initalized
    def update_name(self, attribute, value):
        import re
        pattern = fr"{attribute}\d+"
        replacement = f"{attribute}{value}"
        updated_name = re.sub(pattern, replacement, self.problem_name)
        self.problem_name = updated_name