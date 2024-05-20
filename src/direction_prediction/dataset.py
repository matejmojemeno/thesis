import cv2
import matplotlib.pyplot as plt
import numpy as np

from functions.angles import path_angles
from functions.video_paths import (
    get_test_video_paths,
    get_test_video_paths_arrays,
    get_video_paths,
    get_video_paths_arrays,
)


def adjust_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi

    return -angle


def rotate(frame, angle):
    rotation_angle = np.random.randint(0, 360)
    rows, cols = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    frame = cv2.warpAffine(frame, rotation_matrix, (cols, rows))

    angle -= np.radians(rotation_angle)

    return frame, adjust_angle(angle)


def adjust_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[30:-30, 30:-30]
    frame = cv2.resize(frame, (64, 64))
    frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)[1]
    return frame


def create_dataset(video_paths, paths):
    angle_dataset = []
    frame_dataset = []

    for video_path, path in zip(video_paths, paths):
        cap = cv2.VideoCapture(video_path)
        angles = path_angles(path)

        assert len(angles) == len(path)
        for i in range(len(angles)):
            _, frame = cap.read()

            frame, angle = rotate(frame, angles[i])
            frame = adjust_frame(frame)

            angle_dataset.append(angle)
            frame_dataset.append(frame)

    return (
        np.array(frame_dataset),
        np.array(angle_dataset),
    )


def main():
    video_paths = get_video_paths()
    test_video_paths = get_test_video_paths()

    paths = get_video_paths_arrays()
    test_paths = get_test_video_paths_arrays()

    frames, angles = create_dataset(video_paths, paths)
    np.save("data/frame_dataset.npy", frames)
    np.save("data/angle_dataset.npy", angles)

    test_frames, test_angles = create_dataset(test_video_paths, test_paths)

    np.save("data/test_frame_dataset.npy", test_frames)
    np.save("data/test_angle_dataset.npy", test_angles)


if __name__ == "__main__":
    main()
