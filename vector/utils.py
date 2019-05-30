import cv2
import numpy as np


def morph_close(img, num_iter=1):
    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for _ in range(num_iter):
        img = cv2.dilate(img, strel, iterations=1)
        img = cv2.erode(img, strel, iterations=1)

    return img


def morph_open(img, num_iter=1):
    strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for _ in range(num_iter):
        img = cv2.erode(img, strel, iterations=1)
        img = cv2.dilate(img, strel, iterations=1)

    return img


def get_guidance_points_and_weights(w, h, num=6, bottom_to_top=False):
    m = w // 2
    p = 0.3
    spacing = (0.9 - p) * h / (num - 1)

    # top -> bottom
    weights = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1])

    points = np.concatenate((
        np.array([[m, int(p * h + i * spacing)] for i in range(num - 1)]),
        [[m, int(0.9 * h)]]
    )).astype(np.int32)

    if bottom_to_top:
        weights = weights[::-1]
        points = points[::-1]

    return spacing, points, weights


def get_left_right_markers(row):
    left = (row == 255).argmax()
    right = len(row) - (row[::-1] == 255).argmax()

    return left, right


def get_point_on_lane(row, point):
    px, py = point
    left, right = row[:px], row[px:]

    ll, lr = get_left_right_markers(left)
    if ll < lr:
        return int(0.5 * (ll + lr))

    rl, rr = get_left_right_markers(right)
    if rl < rr:
        return int(0.5 * (rl + rr))

    return 0, 0


def compute_speed_delta(diffs, weights, div_factor=8):
    x = np.sum(diffs * weights)

    s = np.round(x / div_factor)
    # s = 1.6 * x / (1 + abs(x))

    return s


def get_region_of_interest(img, sx=0.23, sy=0.15, delta=200, return_vertices=False):
    """
    :param img: image to extract ROI from
    :param sx: X-axis factor for ROI bottom base
    :param sy: Y-axis factor for ROI top base
    :param delta: ROI top base length
    :param return_vertices: whether to return the ROI vertices
    :return: ROI (optional: vertices)
    """
    assert len(img.shape) == 2

    h, w = img.shape

    mask = np.zeros(img.shape)
    fill_color = 255

    vertices = np.array([
        [0.5 * (w - delta), sy * h],
        [0.5 * (w + delta), sy * h],
        [(1 - sx) * w, h - 1],
        [sx * w, h - 1],
    ])

    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), fill_color)

    roi = mask.astype(np.uint8) & img.astype(np.uint8)

    if return_vertices:
        return roi, vertices
    else:
        return roi


def draw_roi(img, roi_vertices) -> None:
    n = len(roi_vertices)

    for i in range(n):
        p1 = tuple(roi_vertices[i % n])
        p2 = tuple(roi_vertices[(i + 1) % n])
        cv2.line(img, p1, p2, color=(0, 255, 0), thickness=1)
