import csv
from itertools import combinations
import os

import copy
from pathlib import Path
import numpy as np
from opensbt.model_ga.individual import IndividualSimulated as Individual
from opensbt.model_ga.population import PopulationExtended as Population

from opensbt.config import OUTPUT_PRECISION
from opensbt.visualization.visualizer import if_none_iterator, write_simout
from opensbt.utils.duplicates import duplicate_free

from opensbt.visualization.combined import read_pf_single
import dill
import sys
from typing import Tuple
import numpy as np
import math

from opensbt.problem.adas_problem import ADASProblem
from opensbt.simulation.simulator import SimulationOutput

from opensbt.utils import geometric
from simulations.simple_sim import *
from simulations.carla import *
import logging as log
import pandas as pd

def generate_problem_name(name_prefix,
                          base_name,
                          seed,
                          population_size,
                          n_generations=None,
                          time=None,
                          algo=None,
                          **kwargs):
    # def format_time(t_str):
    #     """Convert 'HH:MM:SS' string to compact format like '1h30m15s'."""
    #     h, m, s = map(int, t_str.split(":"))
    #     parts = []
    #     if h > 0:
    #         parts.append(f"{h}h")
    #     if m > 0:
    #         parts.append(f"{m}m")
    #     if s > 0:
    #         parts.append(f"{s}s")
    #     return "".join(parts) if parts else "0s"

    if name_prefix != None and name_prefix is not "":
        parts = [name_prefix, algo.upper(), base_name, f"pop{population_size}"]
    else:
        parts = [base_name, algo.upper(), f"pop{population_size}"]

    if n_generations is not None and not "":
        parts.append(f"gen{n_generations}")
    if time is not None and not "":
        # formatted_time = format_time(time)
        parts.append(f"t{time}")
    
    # Handle additional keyword arguments
    for key, value in kwargs.items():
        if value is None or "":
            continue
        parts.append(f"{key}{value}")
    
    parts.append(f"seed{seed}")

    return "_".join(parts)

def analyse_overall(file, save_folder, 
                           file_name = "overall_summary.csv"):
    # Load the data
    data = pd.read_csv(file)  # Replace with your actual file name or use pd.read_csv(io.StringIO(...)) if using raw string
    
    # Critical in LoFi but not in HiFi
    lofi_critical = ((data['Critical_LoFi'] == 1)).sum()
    # Critical in HiFi but not in LoFi
    hifi_critical = ((data['Critical_HiFi'] == 1)).sum()
    n_tests = len(data)

    # Print result
    ###################
    # Prepare results as DataFrame
    results = pd.DataFrame({
        'Metric': [
            'Number Critical LoFi',
            'Number Critical HiFi',
            'Number tests'
        ],
        'Count': [
            lofi_critical,
            hifi_critical,
            n_tests
        ]
    })

    # Write to CSV
    results.to_csv(f'{save_folder}/{file_name}', index=False)

def analyse_agree_disagree(file, save_folder,
                           file_name="agree_summary.csv",
                           num_exec_extra=0,
                           extra_execution_time="0"):
    import pandas as pd

    # Load the data
    data = pd.read_csv(file)
    
    # Map numeric values to descriptive names and preserve NaNs
    data_filled = data[['Critical_LoFi', 'Critical_HiFi']].copy()
    data_filled['Critical_LoFi'] = data_filled['Critical_LoFi'].map({1: 'Critical', 0: 'NonCritical'})
    data_filled['Critical_HiFi'] = data_filled['Critical_HiFi'].map({1: 'Critical', 0: 'NonCritical'})
    
    # Agreements: only rows where neither is None
    agreements = ((data_filled['Critical_LoFi'] == data_filled['Critical_HiFi']) &
                  data_filled['Critical_LoFi'].notna() &
                  data_filled['Critical_HiFi'].notna()).sum()
    
    # Disagreements: only rows where neither is None and values differ
    disagreements = ((data_filled['Critical_LoFi'] != data_filled['Critical_HiFi']) &
                     data_filled['Critical_LoFi'].notna() &
                     data_filled['Critical_HiFi'].notna()).sum()
    
    # Critical in LoFi but not in HiFi
    lofi_only = ((data_filled['Critical_LoFi'] == 'Critical') & (data_filled['Critical_HiFi'] == 'NonCritical')).sum()
    
    # Critical in HiFi but not in LoFi
    hifi_only = ((data_filled['Critical_HiFi'] == 'Critical') & (data_filled['Critical_LoFi'] == 'NonCritical')).sum()
    
    # Independent counts
    lofi_critical = (data_filled['Critical_LoFi'] == 'Critical').sum()
    hifi_critical = (data_filled['Critical_HiFi'] == 'Critical').sum()
    lofi_noncritical = (data_filled['Critical_LoFi'] == 'NonCritical').sum()
    hifi_noncritical = (data_filled['Critical_HiFi'] == 'NonCritical').sum()
    lofi_none = data_filled['Critical_LoFi'].isna().sum()
    hifi_none = data_filled['Critical_HiFi'].isna().sum()
    
    # Print results
    # Print results
    print(f"Number agreements: {agreements}")
    print(f"Number disagreements: {disagreements}")
    print(f"Critical LoFi, Not Critical HiFi: {lofi_only}")
    print(f"Critical HiFi, Not Critical LoFi: {hifi_only}")
    print(f"Critical LoFi: {lofi_critical}")
    print(f"Critical HiFi: {hifi_critical}")
    print(f"NonCritical LoFi: {lofi_noncritical}")
    print(f"NonCritical HiFi: {hifi_noncritical}")
    print(f"LoFi None: {lofi_none}")
    print(f"HiFi None: {hifi_none}")
    print(f"Extra execution time: {extra_execution_time}")
    
    # Prepare results as DataFrame with Python-style metric names
    results = pd.DataFrame({
        'metric': [
            'agreements',
            'disagreements',
            'critical_lofi_noncritical_hifi',
            'critical_hifi_noncritical_lofi',
            'critical_lofi',
            'critical_hifi',
            'noncritical_lofi',
            'noncritical_hifi',
            'lofi_none',
            'hifi_none',
            'extra_execution_time',
            'number_executions_extra'
        ],
        'count': [
            agreements,
            disagreements,
            lofi_only,
            hifi_only,
            lofi_critical,
            hifi_critical,
            lofi_noncritical,
            hifi_noncritical,
            lofi_none,
            hifi_none,
            extra_execution_time,
            num_exec_extra
        ]
    })
    
    # Save CSV
    results.to_csv(f'{save_folder}/{file_name}', index=False)
    
    return results

def write_all_individuals_lohifi(problem, all_individuals, save_folder, file_name = "all_testcases.csv"):
    
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + file_name, 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_LoFi")
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_HiFi")
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical_LoFi")
        header.append(f"Critical_HiFi")
        # header.append(f"Prediction")

        write_to.writerow(header)

        index = 0
        for index, ind in enumerate(all_individuals):
                row = [index]
                # print("ind get F lofi", ind.get("F_LOFI"))
                # print("ind get F hifi", ind.get("F_HIFI"))

                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
                
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_LOFI", None, problem.n_obj)
                ])
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_HIFI", None, problem.n_obj)
                ])

                cb_val = ind.get("CB_LOFI")
                row.extend(["%i" % int(cb_val) if cb_val is not None and not np.isnan(cb_val) else "nan"
                ])                
                cb_val = ind.get("CB_HIFI")
                row.extend(["%i" % int(cb_val) if cb_val is not None and not np.isnan(cb_val) else "nan"
                ])                
                # cb_val = ind.get("P")
                # row.extend(["%i" % cb_val if cb_val is not None else "nan"])
                write_to.writerow(row)
        f.close()
    return save_folder + file_name

def write_simulation_output(inds, save_folder: str,
                            accessor = "SO", folder_name = "simout"):

    save_folder_simout = save_folder + os.sep + folder_name + os.sep
    Path(save_folder_simout).mkdir(parents=True, exist_ok=True)

    write_simout(save_folder_simout, 
                inds,
                accessor=accessor)
    
def read_pf_single_sequential(filename, problem,
                              with_critical_column=False,
                              skip_lofi_nan=False):

    import pandas as pd

    table = pd.read_csv(filename).replace("nan", pd.NA)
    individuals = []

    decision_columns = problem.simulation_variables

    fitness_base = ["Fitness_" + n for n in problem.fitness_function.name]
    f_lofi = [f + "_LoFi" for f in fitness_base]
    f_hifi = [f + "_HiFi" for f in fitness_base]

    has_lofi_cb = "Critical_LoFi" in table.columns
    has_hifi_cb = "Critical_HiFi" in table.columns

    for _, row in table.iterrows():
        X = row[decision_columns].to_numpy(float)

        F_lofi = row[f_lofi].to_numpy(float)
        F_hifi = row[f_hifi].to_numpy(float)

        CB_lofi = row["Critical_LoFi"] if with_critical_column and has_lofi_cb else None
        CB_hifi = row["Critical_HiFi"] if with_critical_column and has_hifi_cb else None

        CB_lofi = None if pd.isna(CB_lofi) else CB_lofi
        CB_hifi = None if pd.isna(CB_hifi) else CB_hifi

        if skip_lofi_nan and has_lofi_cb and CB_lofi is None:
            continue

        # HiFi priority, LoFi fallback
        main_cb = CB_hifi if CB_hifi is not None else CB_lofi

        ind = Individual()
        ind.set("X", X)
        ind.set("F_LOFI", F_lofi)
        ind.set("F_HIFI", F_hifi)
        ind.set("CB_LOFI", CB_lofi)
        ind.set("CB_HIFI", CB_hifi)
        ind.set("CB", main_cb)

        individuals.append(ind)

    return Population(individuals=individuals)
# def read_pf_single_sequential(filename, with_critical_column=False, skip_lofi_nan = False):
#     individuals = []
#     table = pd.read_csv(filename)

#     # Ensure 'nan' strings are treated as actual NaN values
#     table.replace("nan", pd.NA, inplace=True)

#     # Filter: only rows that contain at least one NaN
#     table = table[table.isna().any(axis=1)]
#     # Identify number of decision variables (columns before first 'Fitness_')
#     n_var = -1
#     for idx, col in enumerate(table.columns[1:], start=1):  # skip index column
#         if col.startswith("Fitness_"):
#             n_var = idx - 1
#             break

#     for i in range(len(table)):
#         row = table.iloc[i]

#         # X = decision variables (1 to n_var)
#         X = row[1:n_var + 1].to_numpy(dtype=float)

#         # F = objective values
#         if with_critical_column:
#             F = row[n_var + 1: -1 - n_var].to_numpy(dtype=float)
#             CB = row.iloc[-2]
#         else:
#             F = row[n_var + 1:].to_numpy(dtype=float)
#             CB = None
#             print("CB set to None")

#         # skip the lofi NaN values
#         if skip_lofi_nan and pd.isna(CB):
#             print("Skipped a lofi nan.")
#             continue

#         ind = Individual()
#         ind.set("X", X)
#         ind.set("F_LOFI", F)
#         ind.set("CB_LOFI", CB if not pd.isna(CB) else None)
#         ind.set("CB", CB if not pd.isna(CB) else None)
#         individuals.append(ind)
#         # print("Individual x:", ind.get("X"))
#         # print("Individual f:", ind.get("F_LOFI"))
#     return Population(individuals=individuals)