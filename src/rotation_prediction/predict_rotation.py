import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import (
    LSTM,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
)
from keras.models import Sequential


def filter_data(X, y, limit):
    stay = y < limit
    X = X[stay]
    y = y[stay]
    stay = -limit < y
    X = X[stay]
    y = y[stay]

    return X, y


def load_data_raw(n_bins, size, step):
    X_train = (np.load(f"data/video_dataset_{size}_{step}.npy") / 255).astype(np.uint8)
    y_train = np.load("data/segment_rotations.npy")

    X_test = (np.load(f"data/test/video_dataset_{size}_{step}.npy") / 255).astype(
        np.uint8
    )
    y_test = np.load("data/test/segment_rotations.npy")

    limit = np.pi / 3
    X_train = X_train[np.abs(y_train) < limit]
    y_train = y_train[np.abs(y_train) < limit]

    indices_shuffle = np.random.permutation(X_train.shape[0])
    X_train = X_train[indices_shuffle]
    y_train = y_train[indices_shuffle]

    X_test = X_test[np.abs(y_test) < limit]
    y_test = y_test[np.abs(y_test) < limit]

    y_train, y_test = digitize_data(y_train, y_test, n_bins, limit)
    return X_train, y_train, X_test, y_test


def digitize_data(y_train, y_test, n_bins, limit):
    bins = np.linspace(-limit, limit, n_bins + 1)
    y_train = np.digitize(y_train, bins) - 1
    y_test = np.digitize(y_test, bins) - 1

    return y_train, y_test


def colorize(X):
    X_colored = np.zeros((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 3))
    for i in range(X.shape[0]):
        print(i, X.shape[0], end="\r")
        for j in range(X.shape[1]):
            X_colored[i, j] = cv2.cvtColor(X[i, j], cv2.COLOR_GRAY2RGB)

    print()
    return X_colored


def train(n_bins, size, step):
    X_train, y_train, X_test, y_test = load_data_raw(n_bins, size, step)

    X_train_colored = colorize(X_train)
    X_test_colored = colorize(X_test)
    X_train, X_test = X_train_colored, X_test_colored

    model = Sequential()
    model.add(
        TimeDistributed(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_normal",
                input_shape=(X_train.shape[2], X_train.shape[3], X_train.shape[4]),
            )
        )
    )
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(128))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(n_bins, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    print(model.summary())

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

    model.save(f"data/models/model_{size}_{step}.h5")
    model.evaluate(X_test, y_test)


def main():
    for size in [32, 64]:
        for step in [5, 10, 20]:
            if size == 32 and step == 5:
                continue
            print(size, step)
            train(7, size, step)


if __name__ == "__main__":
    main()
