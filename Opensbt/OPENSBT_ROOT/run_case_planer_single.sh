# python -m simulations.beamng.run_beamng \
#   --seed 1 \
#   --population_size 10 \
#   --maximal_execution_time "06:00:00" \
#   --rerun_only_critical \

# python -m simulations.commonroad.run_cr \
#   --seed 1 \
#   --population_size 10 \
#   --maximal_execution_time "06:00:00" \
#   --rerun_only_critical \

#!/bin/bash

for SEED in 16 17
do
python -m simulations.commonroad.run_cr\
    --seed ${SEED} \
    --population_size 10 \
    --maximal_execution_time "06:00:00" \
    --project "planer_final" \
    --n_rerun 1 \
    --rerun_only_critical \
    --only_if_no_hifi
done