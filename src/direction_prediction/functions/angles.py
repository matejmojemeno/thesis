import numpy as np
from scipy.interpolate import splev, splprep


def angle_difference(angle1, angle2):
    diff = angle1 - angle2
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return diff


def get_tangent(tck, point_u):
    """Get tangent at a point on the spline"""
    dydx = splev(point_u, tck, der=1)

    return dydx


def prepare_path(path):
    """Prepare path for spline generation"""
    path_unique = np.unique(path, axis=0)

    x = np.zeros_like(path_unique[:, 0])
    y = np.zeros_like(path_unique[:, 1])
    index = 0

    for point in path:
        if np.any(np.all(path_unique == point, axis=1)):
            x[index] = point[0]
            y[index] = point[1]
            index += 1
            path_unique = np.delete(
                path_unique, np.where((path_unique == point).all(axis=1)), axis=0
            )
    return x, y


def path_angles(path):
    x, y = prepare_path(path)
    tck, _ = splprep([x, y], s=3 * len(path), k=3)
    angles = []

    for i in range(len(path)):
        dydx = get_tangent(tck, i / len(path))
        tangent = dydx / np.linalg.norm(dydx)
        angle = np.arctan2(tangent[1], tangent[0])
        angles.append(angle)

    return angles
