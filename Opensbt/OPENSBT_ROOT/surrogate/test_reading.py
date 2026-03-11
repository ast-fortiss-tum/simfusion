import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from surrogate.model import Model
import argparse
import joblib
import os
from tensorflow.keras.models import load_model

######## test reading
sample = np.asarray([[20, 3, 25]])
# model: Model = Model.load(filepath_scaler="./surrogate/rbf_model_2025-06-09_15-15-40_scaler.pkl",
#                    filepath_model="./surrogate/rbf_model_2025-06-09_15-15-40.keras")
                   
# val = model.predict(sample)
# print("max:", model.y_max)
# print("min:", model.y_min)
# print(val)


# === LOAD MODEL AND SCALERS ===
print("\nLoading saved model and scalers for verification...")
model_output_path = "./surrogate/out_test/"
model = "model_MLP"
timestamp = "2025-06-11_22-38-40"
model_loaded = load_model(f"{model_output_path}{model}_{timestamp}.h5")

x_scaler_loaded = joblib.load(f"{model_output_path}{model}_{timestamp}_xscaler.pkl")
y_scaler_loaded = joblib.load(f"{model_output_path}{model}_{timestamp}_yscaler.pkl")

# === EVALUATE LOADED MODEL ===
y_pred_scaled_loaded = model_loaded.predict(sample)
print(y_pred_scaled_loaded)