import numpy as np

class MinMaxScaler:
    def __init__(self, min_val, max_val, clip=True):
        """
        Parameters:
        - min_val: array-like of shape (n_features,) or scalar
        - max_val: array-like of shape (n_features,) or scalar
        - clip: whether to clip values outside [0, 1] range
        """
        self.min_val = np.asarray(min_val)
        self.max_val = np.asarray(max_val)
        self.clip = clip
        
        if np.any(self.max_val <= self.min_val):
            raise ValueError("max_val must be greater than min_val for all features")

    def transform(self, X):
        """
        Scale X to [0, 1] using predefined min and max.
        X: array-like of shape (n_samples, n_features) or (n_samples,)
        Returns scaled array of same shape.
        """
        X = np.asarray(X)
        scaled = (X - self.min_val) / (self.max_val - self.min_val)
        # if self.clip:
        #     scaled = np.clip(scaled, 0.0, 1.0)
        return scaled

    def inverse_transform(self, X_scaled):
        """
        Inverse of transform: map [0, 1] back to original scale.
        X_scaled: array-like of same shape as X
        """
        X_scaled = np.asarray(X_scaled)
        if self.clip:
            X_scaled = np.clip(X_scaled, 0.0, 1.0)
        return X_scaled * (self.max_val - self.min_val) + self.min_val
