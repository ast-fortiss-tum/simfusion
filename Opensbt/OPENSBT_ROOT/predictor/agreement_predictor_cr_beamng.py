import numpy as np
import pandas as pd
from glob import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class AgreementPredictor:
    """
    - Label = 1  : Critical_LoFi == Critical_HiFi
    - Label = 0  : Critical_LoFi != Critical_HiFi
    """
    def __init__(self, csv_folder: str, feature_columns: list[str], k: int = 1):
        """
        Initialize the classifier from multiple CSV files.

        Parameters:
        - csv_folder: str, path to folder containing CSV files
        - feature_columns: list[str], names of feature columns
        - k: int, number of nearest neighbors
        """
        self.feature_columns = feature_columns
        self.csv_folder = csv_folder
        self.k = k

        # Load and concatenate CSV files
        csv_files = glob(f"{csv_folder}/*.csv")
        if not csv_files:
            raise ValueError(f"No CSV files found in folder: {csv_folder}")

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

        if "Critical_LoFi" not in df.columns or "Critical_HiFi" not in df.columns:
            raise KeyError("Input CSVs must contain 'Critical_LoFi' and 'Critical_HiFi' columns.")

        df["AgreementLabel"] = np.where(
            df["Critical_LoFi"] == df["Critical_HiFi"],
            1, 
            0
        )

        # Extract features and labels
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_features:
            raise KeyError(f"Missing feature columns in CSVs: {missing_features}")

        X = df[feature_columns].to_numpy()
        y = df["AgreementLabel"].to_numpy()

        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Fit k-NN model
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X_scaled, y)

    def predict(self, x_new: np.ndarray) -> int:
        """
        Predict Agreement/Disagreement for a new sample.

        Returns:
        - 1 or 0
        """
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)

        x_new_scaled = self.scaler.transform(x_new)
        return int(self.model.predict(x_new_scaled)[0])

    def predict_prob(self, x_new: np.ndarray) -> dict[str, float]:
        """
        Returns Agreement/Disagreement probabilities.

        Returns:
        - {"Agreement": p_agreement, "Disagreement": p_disagreement}
        """
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new.reshape(1, -1)

        x_new_scaled = self.scaler.transform(x_new)
        probs = self.model.predict_proba(x_new_scaled)[0]
        classes = self.model.classes_
        return {cls: float(p) for cls, p in zip(classes, probs)}

# if __name__ == "__main__":
#     csv_folder = "./predictor/data_cr_beamng/"
#     features = [
#         "goal_center_x",
#         "goal_center_y",
#         "goal_velocity_min",
#         "goal_velocity_max",
#         "num_npc",
#         "lane_count",
#         "npc_x0_min",
#         "npc_x0_max",
#         "npc_v_max",
#         "npc_seed",
#         "cutin_tx",
#         "cutin_delay",
#     ]

#     # Create and train the classifier
#     predictor = AgreementPredictor(csv_folder, features, k=3)

#     # New test example (e.g., single scenario parameter set)
#     x_new = np.array([
#         10.0,   # goal_center_x
#         0.0,    # goal_center_y
#         5.0,    # goal_velocity_min
#         10.0,   # goal_velocity_max
#         3,      # num_npc
#         2,      # lane_count
#         -5.0,   # npc_x0_min
#         15.0,   # npc_x0_max
#         12.0,   # npc_v_max
#         42,     # npc_seed
#         3.0,    # cutin_tx
#         1.0,    # cutin_delay
#     ])

#     label = predictor.predict(x_new)
#     proba = predictor.predict_prob(x_new)

#     print(f"Predicted label: {label}")
#     print(f"Probabilities: {proba}")
