#!/bin/bash

seeds=(11 12 13 15)
tags=(b1frm3n5 sji3plny jxhix0h9 xnrjawkh)

for i in "${!seeds[@]}"; do
    seed=${seeds[$i]}
    tag=${tags[$i]}

    python -m resume.resume_validation \
        --simulate_function beamng \
        --save_folder "./results_wandb/LoFi_GA_pop10_t06_00_00_seed${seed}/${tag}" \
        --project planer_final \
        --entity lofi-hifi \
        --run_id "${tag}" \
        --sim_rerun hifi \
        --sim_original lofi \
        --rerun_only_critical \
        --n_rerun 1 \
        --only_if_no_hifi \
        --only_rerun_folder
done