import numpy as np


def cartesian_to_relative(path):
    """
    Convert a path from cartesian to relative coordinates.

    Parameters
    ----------
    path : np.ndarray
        A 2D array of shape (N, 2) representing the path in cartesian coordinates.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 2) representing the path in relative coordinates.
    """
    relative_path = np.zeros((len(path) - 1, 2))
    for i in range(len(relative_path)):
        relative_path[i] = path[i + 1] - path[i]
    return relative_path


def relative_to_cartesian(path):
    """
    Convert a path from relative to cartesian coordinates.

    Parameters
    ----------
    path : np.ndarray
        A 2D array of shape (N, 2) representing the path in relative coordinates.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 2) representing the path in cartesian coordinates.
    """
    cartesian_path = np.zeros((len(path) + 1, 2))
    for i in range(1, len(cartesian_path)):
        cartesian_path[i] = cartesian_path[i - 1] + path[i - 1]
    return cartesian_path
