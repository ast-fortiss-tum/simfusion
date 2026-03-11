import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from keras.initializers import RandomUniform, Initializer, Constant
from keras.layers import Activation, Dense
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import LambdaCallback, EarlyStopping
from tensorflow.keras.utils import register_keras_serializable
import joblib
from tensorflow.keras.models import load_model

class InitCentersRandom(Initializer):
    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]

@register_keras_serializable()
class RBFLayer(Layer):
    def __init__(self, output_dim, initializer=None, betas=1.0, **kwargs):
        self.output_dim = output_dim
        self.init_betas = betas
        self.initializer = initializer if initializer else RandomUniform(0.0, 1.0)
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centers = self.add_weight(
            name='centers',
            shape=(self.output_dim, input_shape[1]),
            initializer=self.initializer,
            trainable=True
        )
        self.betas = self.add_weight(
            name='betas',
            shape=(self.output_dim,),
            initializer=Constant(value=self.init_betas),
            trainable=True
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C - K.transpose(x))
        return K.exp(-self.betas * K.sum(H ** 2, axis=1))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim}
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Model:
    def __init__(self, no_of_neurons, data, n_epochs = 1000):
        self.ss = StandardScaler()
        self.y_min = None
        self.y_max = None
        self.n_epochs = n_epochs
        self.train(no_of_neurons, data, n_epochs)

    def train(self, no_of_neurons, data, n_epochs = 1000):
        X = data[:, :-2]  # all columns except last two are features
        y = data[:, -2:]  # last two columns are targets

        # Store min and max per output dimension for inverse normalization
        self.y_min = y.min(axis=0)
        self.y_max = y.max(axis=0)

        # Normalize targets to [0,1] independently per fitness dimension
        y_norm = (y - self.y_min) / (self.y_max - self.y_min + 1e-8)  # add epsilon for stability

        X_scaled = self.ss.fit_transform(X)
        input_dim = X.shape[1]

        self.model = Sequential()
        rbflayer = RBFLayer(
            no_of_neurons,
            initializer=InitCentersRandom(X_scaled),
            betas=3.0,
            input_shape=(input_dim,)
        )
        self.model.add(rbflayer)
        self.model.add(Dense(2))  # two output units
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_absolute_error', optimizer='adam')

            # === EarlyStopping callback ===
        early_stop = EarlyStopping(
            monitor='loss',
            patience=300,
            restore_best_weights=True,
            verbose=1
        )
        # Callback to print loss every 100 epochs
        print_callback = LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
                print(f"Epoch {epoch+1}, loss: {logs['loss']:.4f}") if (epoch + 1) % 100 == 0 else None
        )

        self.model.fit(X_scaled, y_norm, epochs=n_epochs, batch_size=8, verbose=0, callbacks=[print_callback, early_stop])

    def predict(self, val):
        val = np.array([val])
        val_scaled = self.ss.transform(val)

        # print("val scaled:", val_scaled)
        # print("type val", type(val))
        y_pred_norm = self.model.predict(val_scaled, verbose=0)[0]

        # Inverse normalization for both outputs
        y_pred = y_pred_norm * (self.y_max - self.y_min) + self.y_min
        return y_pred
    
    def evaluate_model(model, val_data, y_min, y_max):
        X_val = val_data[:, :-2]
        y_val = val_data[:, -2:]

        # Scale input
        X_val_scaled = model.ss.transform(X_val)

        # Predict
        y_pred_scaled = model.model.predict(X_val_scaled, verbose=0)

        # Denormalize predictions
        y_pred = y_pred_scaled * (y_max - y_min) + y_min

        # Compute MAE or other metrics
        mae = np.mean(np.abs(y_val - y_pred), axis=0)
        return mae
    
    def save(self, filepath):
        joblib.dump({
            'scaler': self.ss,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'n_epochs' : self.n_epochs
        }, filepath +"_scaler.pkl")

        self.model.save(filepath + ".keras")

    @staticmethod
    def load(filepath_model, filepath_scaler):
        model = load_model(filepath_model, custom_objects={"RBFLayer": RBFLayer})

        meta = joblib.load(filepath_scaler)
        model.ss = meta['scaler']
        model.y_min = meta['y_min']
        model.y_max = meta['y_max']
        model.n_epochs = meta['n_epochs']

        return model


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Parameters
    n_samples = 100
    n_features = 4
    n_targets = 2

    # Generate synthetic data
    X = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples, n_targets) * 10  # scale targets
    data = np.hstack([X, y])

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = np.hstack([X_train, y_train])
    val_data = np.hstack([X_val, y_val])

    # Train model
    model = Model(no_of_neurons=10, data=train_data)

    # Predict a new random input
    sample = np.random.rand(n_features)
    print("Predicted fitness:", model.predict(sample))

    # Evaluate
    preds = np.array([model.predict(x) for x in X_val])
    mae = np.mean(np.abs(preds - y_val), axis=0)
    print("Validation MAE:", mae)

            