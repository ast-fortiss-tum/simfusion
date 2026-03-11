pip install docker

python -m simulations.multi_sim.test_multi_sim_predict \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --only_if_no_hifi \
           --rerun_only_critical \
           --wandb_project "autoware_final" \
           --th_certainty 0.6

python -m simulations.carla.test_carla_sim \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --rerun_only_critical \
           --wandb_project "autoware_final" \
           --only_if_no_hifi

python -m simulations.simple_sim.opensbt_start \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --rerun_only_critical \
           --wandb_project "autoware_final" \
           --only_if_no_hifi

########################

python -m simulations.multi_sim.test_multi_sim_predict \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --only_if_no_hifi \
           --rerun_only_critical \
           --th_certainty 0.6

python -m simulations.carla.test_carla_sim \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --rerun_only_critical \
           --only_if_no_hifi

python -m simulations.simple_sim.opensbt_start \
           --seed 1 \
           --population_size 20 \
           --n_generations 30 \
           --maximal_execution_time "03:00:00" \
           --n_rerun 1 \
           --rerun_only_critical \
           --only_if_no_hifi

# python -m simulations.multi_sim.test_multi_sim_predict \
#            --seed 2 \
#            --population_size 20 \
#            --n_generations 30 \
#            --maximal_execution_time "03:00:00" \
#            --n_rerun 1 \
#            --rerun_only_critical \
#            --only_if_no_hifi

# python -m simulations.carla.test_carla_sim \
#            --seed 2 \
#            --population_size 20 \
#            --n_generations 30 \
#            --maximal_execution_time "03:00:00" \
#            --n_rerun 1 \
#            --rerun_only_critical

# python -m simulations.simple_sim.opensbt_start \
#            --seed 2 \
#            --population_size 20 \
#            --n_generations 30 \
#            --maximal_execution_time "03:00:00" \
#            --n_rerun 1 \
#            --rerun_only_critical

# python -m simulations.multi_sim.test_multi_sim_predict \
#            --seed 3 \
#            --population_size 20 \
#            --n_generations 30 \
#            --maximal_execution_time "03:00:00" \
#            --n_rerun 1 \
#            --rerun_only_critical

# python -m simulations.carla.test_carla_sim \
#            --seed 3 \
#            --population_size 20 \
#            --n_generations 30 \
#            --maximal_execution_time "03:00:00" \
#            --n_rerun 1 \
#            --rerun_only_critical


# python -m simulations.carla.test_carla_sim \
#            --seed 1 \
#            --population_size 5 \
#            --n_generations 100 \
#            --maximal_execution_time "00:08:00" \
#             --n_rerun 3 \
#            --rerun_only_critical

# python -m simulations.carla.test_carla_sim_surrogate \
#             --seed 310 \
#             --population_size 3 \
#             --n_generations 20 \
#             --rerun_only_critical \
#             --n_rerun 3 \
#             --model "RF" \
#             --data_folder "./surrogate/data/batch6/" \
#             --maximal_execution_time "00:08:00"



# python teardown_containers.py

##############################################

# python -m simulations.multi_sim.test_multi_sim_predict \
#            --seed 311 \
#            --population_size 20 \
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical

# python -m simulations.simple_sim.opensbt_start \
#            --seed 311 \
#            --population_size 20\
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical

# python -m simulations.carla.test_carla_sim \
#            --seed 311 \
#            --population_size 20\
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical

# ########################################

# python -m simulations.multi_sim.test_multi_sim_predict \
#            --seed 312 \
#            --population_size 20 \
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical


# python -m simulations.simple_sim.opensbt_start \
#            --seed 312 \
#            --population_size 20\
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical

# python -m simulations.carla.test_carla_sim \
#            --seed 312 \
#            --population_size 20\
#            --n_generations 40 \
#            --maximal_execution_time "03:00:00" \
#            --rerun_only_critical