

python -m surrogate.test_surrogate \
        --sample 3.33045987 8.0211280 17.00739165 \
        --model "RF" "GL" \
        --data_folder "./surrogate_log/data/batch0/"

############################
# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch0/"

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch1/"

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch2/"

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch3/"   
########################

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch0/" \
#         --apply_smote

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch1/" \
#         --apply_smote

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch2/" \
#         --apply_smote

# python -m surrogate.test_surrogate \
#         --sample 3.33045987 8.0211280 17.00739165 \
#         --model "MLP" "RF" "GL" \
#         --data_folder "./surrogate/data/batch4/" \
#         --apply_smote
# python -m simulations.carla.test_carla_sim \
#             --seed 0 \
#             --population_size 2 \
#             --n_generations 2 \
#             --maximal_execution_time "00:02:00" \
#             --xl 3.33045987 8.0211280 17.00739165 \
#             --xu 3.33045987 8.0211280 17.00739165 \
#             --rerun_only_critical