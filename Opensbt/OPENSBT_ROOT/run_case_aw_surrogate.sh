pip install docker

########### Tests all core experiment searches ###############

# PREDICT
python -m simulations.carla.test_carla_sim_surrogate \
           --seed 310 \
           --population_size 3 \
           --n_generations 20 \
           --rerun_only_critical \
           --model "RF" \
           --data_folder "./surrogate/data/batch6/" \
           --do_balance \
           --maximal_execution_time "00:01:00";

