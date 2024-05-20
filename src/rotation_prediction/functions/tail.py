import glob
import json

import cv2
import numpy as np


def find_largest_section(array):
    max_section_len = 0
    section = (None, None)

    for i, index1 in enumerate(array):
        for j, index2 in enumerate(array[i:], start=i):
            if index2 - index1 == j - i:
                if j - i > max_section_len:
                    max_section_len = j - i
                    section = (index1, index2)

    return section


def contour_polyfit(contour, degree=6):
    x = contour[:, 0]
    y = contour[:, 1]

    poly = np.polyfit(x, y, degree)
    poly_x = np.linspace(np.min(x), np.max(x), 50)
    poly_y = np.polyval(poly, poly_x)

    return poly_x, poly_y


def find_farthest_points(contour):
    farthest_point1 = None
    farthest_point2 = None
    max_distance = 0

    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            distance = np.linalg.norm(contour[i] - contour[j])
            if distance > max_distance:
                max_distance = distance
                farthest_point1 = contour[i]
                farthest_point2 = contour[j]

    return farthest_point1, farthest_point2


def find_tail(contour):
    contour = contour[:, 0, :]

    furthest_point1, furthest_point2 = find_farthest_points(contour)
    contour_vector = furthest_point2 - furthest_point1
    contour_angle = np.arctan2(contour_vector[1], contour_vector[0])

    rotation_matrix = np.array(
        [
            [np.cos(contour_angle), -np.sin(contour_angle)],
            [np.sin(contour_angle), np.cos(contour_angle)],
        ]
    )
    contour = np.dot(contour, rotation_matrix)

    sorted_indices = np.argsort(contour[:, 0])
    contour = contour[sorted_indices]

    n = 20
    contour_split = np.array_split(contour, n)

    tail = []
    for i in range(n):
        y = contour_split[i][:, 1]
        if np.max(y) - np.min(y) < 10:
            tail.append(i)

    if tail:
        start, end = find_largest_section(tail)
        tail = np.concatenate(contour_split[start:end])
        return tail, rotation_matrix
    else:
        return contour


def find_curve(contour):
    tail, rotation_matrix = find_tail(contour)
    curve_x, curve_y = contour_polyfit(tail, degree=3)
    curve = np.stack([curve_x, curve_y], axis=1)
    curve = np.dot(curve, np.linalg.inv(rotation_matrix)).astype(int)

    return curve


def get_tail(frame):
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    return find_curve(contour)
