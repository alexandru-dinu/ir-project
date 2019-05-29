import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def open_img(img_path, gray=False):
    if gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)

    return img


def show(arr, from_bgr=False, cmap=None):
    if from_bgr:
        b = arr[:, :, 0]
        g = arr[:, :, 1]
        r = arr[:, :, 2]
        arr = np.dstack((r, g, b))

    plt.imshow(arr, cmap=cmap)
    plt.tight_layout()

    plt.show()


def save_img(arr, name, from_bgr=False):
    if from_bgr:
        b = arr[:, :, 0]
        g = arr[:, :, 1]
        r = arr[:, :, 2]
        arr = np.dstack((r, g, b))

    scipy.misc.imsave(name, arr)
