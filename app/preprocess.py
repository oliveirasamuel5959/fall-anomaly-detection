import numpy as np

WINDOW_SIZE = 100

def preprocess(window):
    X = np.array(window)

    # Example normalization (replace with your scaler)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    return X.reshape(1, WINDOW_SIZE, -1)