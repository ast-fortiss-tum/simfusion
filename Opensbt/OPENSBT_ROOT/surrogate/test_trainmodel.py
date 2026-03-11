import numpy as np
import os
from surrogate.test_surrogate import train_model_from_points

# === Generate synthetic 3D input data and 2D targets ===

# Parameters
num_samples = 100  # Number of training points

# 3D input space: X in ℝ³
X = np.random.uniform(low=-1.0, high=1.0, size=(num_samples, 3))

# 2D output space: y in ℝ² (nonlinear mapping for demonstration)
y = np.stack([
    np.sin(X[:, 0]) + X[:, 1]**2,      # Output dimension 1
    np.cos(X[:, 1]) - X[:, 2]**3       # Output dimension 2
], axis=1)

# === Train model from points ===
train_model_from_points("RF", X, y, model_output_path="")
