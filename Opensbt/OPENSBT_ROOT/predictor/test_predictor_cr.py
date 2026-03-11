from predictor.agreement_predictor import AgreementPredictor
import numpy as np
import time

config_path = "./predictor/models/cr/best_model_config_rf.json"

predictor = AgreementPredictor(
    csv_folder="./predictor/data_cr_beamng/data_final/",
    feature_columns=[
        "goal_center_x",
        "goal_center_y",
        "goal_velocity_min",
        "goal_velocity_max",
        "num_npc",
        "lane_count",
        "npc_x0",
        "npc_v_max",
        "npc_seed",
        "cutin_tx",
        "cutin_delay",
    ],
    config_path=config_path
)

# Example inference
x_new = np.array([[150.0, 0.0, 10, 20, 1, 1, 9, 15, 0, 0.5, 0]])
start_time = time.perf_counter()
label, certainty = predictor.predict_with_label_and_certainty(x_new)
end_time = time.perf_counter()
print(label, certainty)

print(f"Inference took {end_time - start_time:.4f} seconds")