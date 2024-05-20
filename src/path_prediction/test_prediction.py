import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from functions.conversions import cartesian_to_relative, relative_to_cartesian


def load_test_paths():
    test_paths = []
    for i in range(1, 11):
        path = np.load(f"data/test_paths/path{i}.npy")
        test_paths.append(path)
    return test_paths


def predict(model, path, prediction_length=1, to_predict=10):
    predicted = path.copy()

    for i in range(0, to_predict, prediction_length):
        curr_path = predicted[i : i + len(path), :]

        if len(curr_path.shape) < len(path.shape):
            curr_path = np.expand_dims(curr_path, axis=0)

        point = model.predict(curr_path, verbose=0)
        predicted = np.append(predicted, point, axis=0)

    return predicted[-to_predict:]


def plot_predicted(
    test_paths, model, scaler, path_length=25, prediction_length=1, to_predict=10
):
    _, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, ax in enumerate(axes.flat):
        test_path = test_paths[i]

        maximum = len(test_path) - (path_length + prediction_length)
        if maximum < 0:
            continue

        index = np.random.randint(0, maximum)

        path = test_path[index : index + path_length + to_predict]
        path_transformed = cartesian_to_relative(path[: path_length - 0])
        path_transformed = scaler.transform(path_transformed)
        predicted = predict(model, path_transformed, prediction_length, to_predict)
        predicted = scaler.inverse_transform(predicted)[: prediction_length - 1]
        predicted = relative_to_cartesian(predicted) + path[path_length - 1]

        ax.plot(
            path[: path_length - 1, 0],
            path[: path_length - 1, 1],
            marker="o",
            markersize=2,
            color="blue",
            label="True Path",
        )
        ax.plot(
            path[path_length - 1 :, 0],
            path[path_length - 1 :, 1],
            marker="o",
            markersize=2,
            color="green",
            label="True Future",
        )
        ax.plot(
            predicted[:, 0],
            predicted[:, 1],
            marker="o",
            markersize=2,
            color="red",
            label="Predicted Future",
        )
        ax.axis("equal")
        ax.legend()


def main():
    model = load_model("data/models/model_80_30.h5")
    scaler = joblib.load("data/scaler.pkl")
    test_paths = load_test_paths()

    plot_predicted(test_paths, model, scaler)


if __name__ == "__main__":
    main()
