"""
This script is used to train a model to predict future positions of a path.

The output is a model and a scaler that can be used to predict future positions of a path.
"""

import joblib
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Reshape
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from functions.conversions import cartesian_to_relative
from functions.preprocessing import normalize


def filter_condition(path):
    return (
        np.mean(np.linalg.norm(path, axis=1)) > 1
        and np.max(np.linalg.norm(path, axis=1)) < 5
    )


def load_data(path_length):
    paths = np.load(f"data/paths/paths_{path_length}.npy")
    paths = np.array([cartesian_to_relative(path) for path in paths])

    indices = []
    for i, path in enumerate(paths):
        if filter_condition(path):
            indices.append(i)

    paths = paths[indices]
    print("Paths shape:", paths.shape)

    return paths


def prepare_data(paths, prediction_length):
    paths, scaler = normalize(paths)
    X = paths[:, :-prediction_length]
    y = paths[:, -prediction_length:]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    joblib.dump(scaler, "scaler.pkl")
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test, length, prediction_length):
    model = Sequential()
    model.add(
        LSTM(
            64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True
        )
    )
    model.add(Dropout(0.05))
    model.add(LSTM(32))
    model.add(Dropout(0.05))
    model.add(Dense(60))
    model.add(Reshape((30, 2)))
    model.compile(loss="mean_squared_error", optimizer="adam")

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train,
        epochs=1,
        batch_size=32,
        callbacks=[early_stopping],
        validation_data=(X_test, y_test),
    )

    model.save(f"data/models/model_{length}_{prediction_length}.h5")


def main():
    path_length = 80
    prediction_length = 30

    paths = load_data(path_length)
    X_train, X_test, y_train, y_test = prepare_data(paths, prediction_length)
    train(X_train, X_test, y_train, y_test, path_length, prediction_length)


if __name__ == "__main__":
    main()
