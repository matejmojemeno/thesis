# NOTE: https://keras.io/examples/vision/grad_cam/

import cv2
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model


def draw_line(ax, angle, color):
    x = np.array([32, 32 + 25 * np.cos(-angle)])
    y = np.array([32, 32 + 25 * np.sin(-angle)])
    ax.plot(x, y, color=color, linewidth=2)


def draw_classification(ax, y_true, y_pred, correct):
    angle_start = y_pred * np.pi / 6 - np.pi
    angle_end = y_pred * np.pi / 6 - np.pi + np.pi / 6

    color = "green" if correct else "red"
    draw_line(ax, angle_start, color)
    draw_line(ax, angle_end, color)

    draw_line(ax, y_true, "blue")


def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def generate_cam(
    img_array, model, last_conv_layer_name, pred_index=None, respond=False
):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    last_conv_layer_output = last_conv_layer_output[0]

    if respond:
        respond_weights = np.sum(last_conv_layer_output * grads, axis=(0, 1, 2)) / (
            np.sum(last_conv_layer_output + 1e-10, axis=(0, 1, 2))
        )

        heatmap = last_conv_layer_output * respond_weights
    else:
        heatmap = last_conv_layer_output * grads
    heatmap = np.sum(heatmap, axis=-1)
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.8):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    print(superimposed_img[0][0])
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def make_hirescam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    return generate_cam(
        img_array, model, last_conv_layer_name, pred_index, respond=False
    )


def make_respondcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    return generate_cam(
        img_array, model, last_conv_layer_name, pred_index, respond=True
    )


img_size = (64, 64)
last_conv_layer_name = "conv2d"
img_path = "img.jpg"


def main():
    data = np.load("data/test_frame_dataset.npy")
    y_data = np.load("data/test_angle_dataset.npy")
    bins = np.linspace(-np.pi, np.pi, 13)
    y_data_bins = np.digitize(y_data, bins) - 1

    _, axes = plt.subplots(4, 5, figsize=(10, 8))
    colorbar_ax = plt.gcf().add_axes([0.90, 0.1, 0.02, 0.8])

    model = load_model("data/models/model.h5")
    model.layers[-1].activation = None

    for i in range(4):
        index = np.random.randint(0, data.shape[0])
        img = np.expand_dims(data[index], axis=0)
        y = y_data[index]
        y_bin = y_data_bins[index]
        print(index)

        preds = model.predict(img)
        pred = preds[0].argmax()

        ax = axes[i, 0]
        ax.matshow(img[0])
        ax.axis("off")
        draw_classification(ax, y, pred, y_bin == pred)

        for j, method in enumerate(
            (make_gradcam_heatmap, make_hirescam_heatmap, make_respondcam_heatmap)
        ):
            heatmap = method(img, model, last_conv_layer_name)
            ax = axes[i, j + 1]
            ax.matshow(heatmap)
            ax.axis("off")

    for i, title in enumerate(["Original", "GradCAM", "HiResCAM", "RespondCAM"]):
        axes[0, i].set_title(title, fontsize=20)  # Adjusted index to start from 1

    plt.colorbar(ax.matshow(heatmap), cax=colorbar_ax)
    colorbar_ax.yaxis.set_ticks_position("left")
    colorbar_ax.tick_params(axis="y", which="both", length=5, width=1, labelsize=20)

    for i in range(4):
        ax = axes[i, 4]
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("explain.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    main()
