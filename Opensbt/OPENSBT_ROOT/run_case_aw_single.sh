pip install docker

########### Tests all core experiment searches ###############

# python -m simulations.multi_sim.test_multi_sim_predict \
#     --seed 0 \
#     --population_size 2 \
#     --n_generations 2 \
#     --maximal_execution_time "00:00:30" \
#     --xl 2.43045987 8.0011280 10.00739165 \
#     --xu 2.53045987 10.0211280 17.00739165 \
#     --rerun_only_critical

# python -m simulations.simple_sim.opensbt_start  \
#     --seed 0 \
#     --population_size 2 \
#     --n_generations 2 \
#     --maximal_execution_time "00:00:20" \
#     --xl 2.43045987 8.0011280 10.00739165 \
#     --xu 2.53045987 10.0211280 17.00739165 \
#     --rerun_only_critical

python -m simulations.carla.test_carla_sim \
            --seed 0 \
            --population_size 4 \
            --n_generations 4 \
            --maximal_execution_time "00:05:00" \
            --xl 2.43045987 8.0011280 10.00739165 \
            --xu 2.53045987 10.0211280 17.00739165