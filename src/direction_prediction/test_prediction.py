import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score

from functions.angles import angle_difference


def draw_line(ax, angle, color):
    x = np.array([32, 32 + 30 * np.cos(-angle)])
    y = np.array([32, 32 + 30 * np.sin(-angle)])
    ax.plot(x, y, color=color, linewidth=2)


def draw_classification(ax, y_true, y_pred, correct):
    angle_start = y_pred * np.pi / 6 - np.pi
    angle_end = y_pred * np.pi / 6 - np.pi + np.pi / 6

    draw_line(ax, y_true, "blue")

    color = "green" if correct else "red"
    draw_line(ax, angle_start, color)
    draw_line(ax, angle_end, color)


def main():
    X = np.load("data/test_frame_dataset.npy") / 255
    y = np.load("data/test_angle_dataset.npy")

    model = load_model("data/models/model.h5")

    y_pred = model.predict(X).argmax(axis=1)
    y = (y + np.pi) / (2 * np.pi)
    y = y * 2 * np.pi - np.pi

    bins = np.linspace(-np.pi, np.pi, 13)
    y_bins = np.digitize(y, bins) - 1

    accuracy = (y_bins == y_pred).mean()
    print(f"Accuracy: {accuracy:.2f}")

    _, axes = plt.subplots(3, 3, figsize=(15, 15))

    for _, ax in enumerate(axes.flat):
        idx = np.random.randint(0, len(X))
        ax.imshow(X[idx])
        ax.axis("off")

        y_true = y[idx]
        predicted = y_pred[idx]

        correct = y_bins[idx] == predicted
        draw_classification(ax, y_true, predicted, correct)

    plt.show()


if __name__ == "__main__":
    main()
