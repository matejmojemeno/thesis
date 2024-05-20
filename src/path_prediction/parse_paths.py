"""
Parse paths from json file into uniform sequences and save them as numpy arrays.
"""

import json

import numpy as np


def create_test_paths(paths):
    cnt = 0

    while cnt < 10:
        index = np.random.randint(0, len(paths))
        path = np.array(paths[index])[:, :2]

        max_norm = np.max(np.linalg.norm(path, axis=1))
        mean_norm = np.mean(np.linalg.norm(path, axis=1))

        norms = []
        for i in range(len(path) - 1):
            vector = path[i + 1] - path[i]
            norms.append(np.linalg.norm(vector))

        max_norm = np.max(norms)
        mean_norm = np.mean(norms)

        if len(path) > 100 and max_norm < 5 and mean_norm > 2:
            cnt += 1

            paths.pop(index)
            np.save("data/test_paths/test_path_" + str(cnt) + ".npy", path)


def make_array(paths, path_length):
    path_array = []

    print(len(paths))

    for i, path in enumerate(paths):
        path = np.array(path)[:, :2]

        for i in range(len(path) - path_length):
            angle = np.random.randint(0, 360)
            rotation_matrix = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            curr_path = path[i : i + path_length]
            curr_path = np.dot(curr_path, rotation_matrix)
            path_array.append(curr_path)

    path_array = np.array(path_array)
    print(path_array.shape)
    np.save(f"data/paths/paths_{path_length}.npy", path_array)


def main():
    with open("data/paths.json", "r") as f:
        paths = json.load(f)

    length = 80
    create_test_paths(paths)
    make_array(paths, length)


if __name__ == "__main__":
    main()
