import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from functions.angles import angle_difference


def distance(y_true, y_pred):
    dist = np.abs(y_true - y_pred)
    return min(dist, 12 - dist)


def angle_difference_(angle1, angle2):
    diff = angle1 - angle2
    diff = tf.atan2(tf.sin(diff), tf.cos(diff))
    return diff


def custom_loss(y_true, y_pred, n_bins=12):
    y_true = tf.cast(y_true, tf.float32)

    diff = tf.abs(y_true[:, None] - tf.range(n_bins, dtype=tf.float32))
    diff = tf.minimum(diff, n_bins - diff)
    diff = diff[:, 0, :]
    diff = diff * y_pred
    loss = tf.norm(diff, axis=-1)
    loss = tf.reduce_sum(loss)

    return loss


def custom_loss_regression(y_true, y_pred):
    return tf.reduce_sum(tf.square(angle_difference_(y_true, y_pred)))


def train(X_train, y_train, X_test, y_test):
    """Train a classification model to predict the angle of the line in the image."""
    bins = np.linspace(-np.pi, np.pi, 13)
    y_train = np.digitize(y_train, bins) - 1
    y_test = np.digitize(y_test, bins) - 1

    shuffle = np.random.permutation(len(X_train))
    X_train = X_train[shuffle]
    print(X_train.shape)
    y_train = y_train[shuffle]

    model = Sequential()
    model.add(
        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(X_train.shape[1], X_train.shape[2], 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(12, activation="softmax"))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
    )
    model.save("data/model.h5")


def main():
    X_train = np.load("data/frame_dataset.npy")
    X_train = X_train / 255
    y_train = np.load("data/angle_dataset.npy")
    X_test = np.load("data/test_frame_dataset.npy")
    X_test = X_test / 255
    y_test = np.load("data/test_angle_dataset.npy")

    print(X_train.shape, y_train.shape)

    train(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
