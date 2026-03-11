pip install docker

########### Tests all core experiment searches ###############
python -m simulations.carla.test_carla_sim \
           --seed 2 \
           --algo "ga" \
           --population_size 3 \
           --n_generations 2 \
           --sim_rerun "hifi" \
           --wandb_project "test_autoware" \
           --only_if_no_hifi \
           --n_rerun 1

python -m simulations.simple_sim.opensbt_start  \
           --seed 1 \
           --algo "ga" \
           --population_size 2 \
           --n_generations 2 \
           --sim_rerun "hifi" \
           --n_rerun 1 \
           --rerun_only_critical \
           --only_if_no_hifi \
           --wandb_project "test_autoware"
           
python -m simulations.multi_sim.test_multi_sim_predict \
           --seed 1 \
           --population_size 2 \
           --n_generations 2 \
           --maximal_execution_time "00:01:00" \
           --rerun_only_critical \
           --n_rerun 1 \
           --only_if_no_hifi \
           --th_certainty 0.6 \
           --wandb_project "test_autoware"

python -m simulations.carla.test_carla_sim \
           --seed 1 \
           --algo "ga" \
           --population_size 5 \
           --n_generations 2 \
           --sim_rerun "hifi" \
           --n_rerun 3 \
           --rerun_only_critical \
           --wandb_project "test_autoware"

python -m simulations.multi_sim.test_multi_sim_predict \
           --seed 1 \
           --population_size 3 \
           --n_generations 2 \
           --rerun_only_critical \
           --n_rerun 1 \
           --wandb_project "test_autoware"
