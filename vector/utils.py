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


def get_center_points_and_weights(w, h, num=6, bottom_to_top=False):
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

    return 0, 0  # TODO


def compute_speed_delta(diffs, weights, div_factor=8):
    x = np.sum(diffs * weights)

    s = np.round(x / div_factor)
    # s = 1.6 * x / (1 + abs(x))

    return s


def region_of_interest(img):
    assert len(img.shape) == 2

    h, w = img.shape
    sx, sy = 0.23, 0.15
    delta = 200

    mask = np.zeros(img.shape)
    fill_color = 255

    vertices = np.array([
        [0.5 * (w - delta), sy * h],
        [0.5 * (w + delta), sy * h],
        [(1 - sx) * w, h - 1],
        [sx * w, h - 1],
    ])

    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), fill_color)

    return mask.astype(np.uint8) & img.astype(np.uint8)  # cv2.bitwise_and(mask, img)

# def get_middle_lane(mask):
#     h, w = mask.shape
#     lane_mask = np.zeros_like(mask)
#
#     for i in range(50, h):
#         left = 0
#         right = w
#         for j in range(1, w):
#             if mask[i, j - 1] == 0 and mask[i, j] == 255:
#                 left = j
#             if mask[i, j - 1] == 255 and mask[i, j] == 0:
#                 right = j
#
#         middle = int((left + right) / 2)
#         lane_mask[i, middle] = lane_mask[i, middle - 1] = lane_mask[i, middle + 1] = 255
#
#     return lane_mask
