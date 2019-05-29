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


def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with

    vertices = np.array([[160, 50], [320, 50], [479, 639], [0, 639]],
                        dtype=np.int32)

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        _, _, c = img.shape
        ignore_mask_color = (255,) * c
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_or(img, ~mask)

    return masked_image, mask


def get_middle_lane(mask):
    h, w = mask.shape
    lane_mask = np.zeros_like(mask)

    for i in range(50, h):
        left = 0
        right = w
        for j in range(1, w):
            if mask[i, j - 1] == 0 and mask[i, j] == 255:
                left = j
            if mask[i, j - 1] == 255 and mask[i, j] == 0:
                right = j

        middle = int((left + right) / 2)
        lane_mask[i, middle] = lane_mask[i, middle - 1] = lane_mask[i, middle + 1] = 255

    return lane_mask
