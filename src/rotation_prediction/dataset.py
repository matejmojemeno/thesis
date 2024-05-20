import cv2
import numpy as np

from functions.angles import segment_rotation
from functions.video_paths import (
    get_test_video_paths,
    get_test_video_paths_arrays,
    get_video_paths,
    get_video_paths_arrays,
)


def get_video_dataset(video_paths, directory="", size=64, step=4):
    video_array = []

    print(video_paths[29])

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(5, length - 105):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            video = []
            angle = np.random.randint(0, 360)
            rotation_matrix = cv2.getRotationMatrix2D((120, 120), angle, 1.0)

            for _ in range(100 // step):
                _, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame[40:-40, 40:-40]
                frame = cv2.warpAffine(frame, rotation_matrix, (240, 240))
                frame = cv2.resize(frame, (size, size))
                frame = cv2.threshold(frame, 10, 255, cv2.THRESH_BINARY)[1]
                frame = cv2.morphologyEx(
                    frame, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=3
                )
                video.append(frame)

                for i in range(step - 1):
                    cap.read()

            video_array.append(video)

    video_array = np.array(video_array)
    print("Video array shape: ", video_array.shape)
    np.save(
        "data/" + directory + f"video_dataset_{size}_{step}.npy",
        video_array,
    )


def get_rotation_segments(paths, directory=""):
    segment_rotations = []

    for path in paths:
        path = np.array(path)
        rotation_array = segment_rotation(path)

        count = 0

        for i in range(5, len(path) - 105):
            segment_rotations.append(rotation_array[i + 100] - rotation_array[i])

            if (
                -np.pi / 3 + np.pi / 9
                < rotation_array[i + 100] - rotation_array[i]
                < -np.pi / 3 + 2 * (np.pi / 9)
            ):
                count += 1

    segment_rotations = np.array(segment_rotations)
    print("Segment rotation shape:", segment_rotations.shape)
    np.save("data/" + directory + "segment_rotations.npy", segment_rotations)


def main():
    video_paths = get_video_paths()
    paths = get_video_paths_arrays()
    test_video_paths = get_test_video_paths()
    test_paths = get_test_video_paths_arrays()

    get_rotation_segments(paths)
    get_rotation_segments(test_paths, "test/")

    for size in [32, 64]:
        for step in [5, 10, 20]:
            get_video_dataset(video_paths, "", size, step)
            get_video_dataset(test_video_paths, "test/", size, step)


if __name__ == "__main__":
    main()
