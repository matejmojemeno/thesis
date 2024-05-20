import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score


def digitize_data(y, n_bins, limit):
    bins = np.linspace(-limit, limit, n_bins + 1)
    y = np.digitize(y, bins) - 1

    return y


def main():
    n_bins = 7
    y_test = np.load(f"data/test/segment_rotations.npy")
    limit = np.pi / 3
    indices_stay = np.abs(y_test) < limit
    y_test = y_test[indices_stay]
    y_test = digitize_data(y_test, n_bins, limit)

    print("y_test", y_test.shape)

    for size in [32, 64]:
        for step in [5, 10, 20]:
            X_test = np.load(f"data/test/video_dataset_{size}_{step}.npy")
            X_test = X_test[indices_stay]

            X_test_colored = np.zeros(
                (X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3], 3)
            )

            for i in range(X_test.shape[0]):
                print(i, X_test.shape[0], end="\r")
                for j in range(X_test.shape[1]):
                    X_test_colored[i, j] = cv2.cvtColor(
                        X_test[i, j], cv2.COLOR_GRAY2RGB
                    )

            X_test = X_test_colored

            model = load_model(f"data/model_{size}_{step}.h5")
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=1)
            correct = np.abs(y_test - y_pred) < 2
            print(np.sum(correct) / len(correct))
            correct = y_test == y_pred

            print("Size:", size, "Step:", step)
            print("F1 Score:", f1_score(y_test, y_pred, average=None))
            print("Accuracy:", np.sum(correct) / len(correct))

            if size == 32 and step == 5:
                conf_matrix = confusion_matrix(y_test, y_pred)
                sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="g", cbar=False)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                plt.savefig("confusion_matrix.svg", format="svg")
                plt.show()
                plt.clf()


if __name__ == "__main__":
    main()
