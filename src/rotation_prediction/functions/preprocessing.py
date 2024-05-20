import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize(paths):
    """
    Normalize a path using MinMaxScaler.

    Parameters
    ----------
    paths : np.ndarray
        A 2D array of shape (N, 2) representing the path in some type of coordinates.

    Returns
    -------
    np.ndarray
        A 2D array of shape (N, 2) representing the path in normalized coordinates.
    MinMaxScaler
        The scaler used to normalize the paths.
    """

    scaler = MinMaxScaler()

    paths_shape = paths.shape
    paths = np.reshape(paths, (paths_shape[0] * paths_shape[1], 2))
    paths = scaler.fit_transform(paths)
    paths = np.reshape(paths, paths_shape)

    return paths, scaler
