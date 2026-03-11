# python -m flaky.evaluate_flakiness \
#   --test_cases_path ./results_old/ART_Hifi_Testing_seed-2_pop-500_gen-1/ARS/16-06-2025_22-18-53/all_testcases.csv \
#   --corner_ids 0 2 5 7 \
#   --n_repeat 3 \
#   --n_samples_corner 3 \
#   --seed 42

python -m flaky.evaluate_flakiness \
  --test_cases_path ./debugging/all_testcases_sut2.csv \
  --n_repeat 1 \
  --n_samples_corner 100 \
  --scenario_file ./scenarios/PedestrianCrossing.xosc \
  --simulator "hifi" \
  --seed 1000

#bash run_teardown.sh