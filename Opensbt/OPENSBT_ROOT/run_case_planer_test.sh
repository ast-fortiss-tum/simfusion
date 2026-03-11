# python -m simulations.multisim_cr_beamng.multisim_predict \
#   --seed 10 \
#   --population_size 2 \
#   --maximal_execution_time "00:10:00" \
#   --wandb_project "planer_final" \
#   --n_rerun 1 \
#   --rerun_only_critical \
#   --th_certainty 0.65 \
#   --th_goal_distance 2.0 \
#   --only_if_no_hifi

# python -m simulations.beamng.run_beamng \
#   --seed 1 \
#   --population_size 1 \
#   --n_generations 1 \
#   --name_prefix hifi_test \
#   --wandb_project "planer_test" \
#   --sim_rerun hifi \
#   --n_rerun 1 \
#   --rerun_only_critical 
  
# python -m simulations.multisim_cr_beamng.multisim_predict \
#   --seed 1 \
#   --population_size 1 \
#   --n_generations 1 \
#   --name_prefix predict_test \
#   --wandb_project "planer_test" \
#   --n_rerun 1 \
#   --rerun_only_critical 

python -m simulations.commonroad.run_cr \
  --seed 1 \
  --population_size 1 \
  --n_generations 1 \
  --name_prefix lofi_test \
  --project "planer_test" \
  --sim_rerun hifi \
  --n_rerun 1 \
  --rerun_only_critical
# python -m simulations.multisim_cr_beamng.multisim_predict \
#   --seed 1 \
#   --population_size 1 \
#   --n_generations 1 \
#   --name_prefix predict_test \
#   --wandb_project "planer_test" \
#   --n_rerun 1 \
#   --rerun_only_critical \
#   --no_validation

# python -m simulations.commonroad.run_cr \
#   --seed 1 \
#   --population_size 2 \
#   --n_generations 1 \
#   --name_prefix lofi_test \
#   --project "planer_test" \
#   --sim_rerun hifi \
#   --n_rerun 1 \
#   --rerun_only_critical \
#   --no_validation