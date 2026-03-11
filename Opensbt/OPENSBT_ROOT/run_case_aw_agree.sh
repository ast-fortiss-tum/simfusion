pip install docker

python -m simulations.carla.test_carla_sim \
           --seed 1 \
           --algo "rs" \
           --population_size 300 \
           --n_generations 1 \
           --sim_rerun "lofi" \
           --n_rerun 1

python -m simulations.simple_sim.opensbt_start  \
           --seed 1 \
           --algo "ga" \
           --population_size 15 \
           --n_generations 20 \
           --sim_rerun "lofi" \
           --n_rerun 1

python -m simulations.carla.test_carla_sim \
           --seed 1 \
           --algo "ga" \
           --population_size 15 \
           --n_generations 20 \
           --sim_rerun "lofi" \
           --n_rerun 1;

python teardown_containers.py
