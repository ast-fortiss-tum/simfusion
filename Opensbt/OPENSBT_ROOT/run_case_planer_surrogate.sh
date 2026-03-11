pip install docker

########### Tests all core experiment searches ###############

# PREDICT
for seed in {30..35}; do
  python -m simulations.multisim_cr_beamng.multisim_surrogate \
    --seed $seed \
    --population_size 10 \
    --n_generations 20000 \
    --rerun_only_critical \
    --model_name "RF" \
    --data_folder "./surrogate_log/data_cr/batch_final" \
    --maximal_execution_time "06:00:00" \
    --wandb_project "planer_final" \
    --only_if_no_hifi
done

# python -m simulations.carla.test_cala_sim_surrogate \
#            --seed 310 \
#            --population_size 20 \
#            --n_generations 20 \
#            --rerun_only_critical \
#            --model "RF" \
#            --data_folder "./surrogate/data/batch0/" \
#            --apply_smote \
#            --maximal_execution_time "03:00:00" 
