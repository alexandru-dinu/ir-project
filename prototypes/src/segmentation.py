import sys

from utils import *


def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with

    vertices = np.array([[160, 50], [320, 50], [479, 639], [0, 639]], dtype=np.int32)

    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_or(img, ~mask)
    return masked_image, mask


def resize_image(img, height=640, width=480):
    if len(img.shape) > 2:
        h, w, _ = img.shape
    else:
        h, w = img.shape

    sf1 = h / float(height)
    sf2 = w / float(width)

    img = cv2.resize(img, (int(w / sf2), int(h / sf1)))

    return img


def get_lane_mask(img):
    mask = np.copy(img)

    h, w = mask.shape

    for i in range(h):
        left1 = left2 = -1
        right1 = right2 = -1
        for j in range(1, w):
            if mask[i, j - 1] != 255 and mask[i, j] == 255:
                left1 = j
            if mask[i, j - 1] == 255 and mask[i, j] != 255:
                left2 = j

            if left1 and left2:
                break

        for j in range(w - 2, -1, -1):
            if mask[i, j + 1] != 255 and mask[i, j] == 255:
                right1 = j
            if mask[i, j + 1] == 255 and mask[i, j] != 255:
                right2 = j

            if right1 and right2:
                break

        if left1 == -1:
            mask[i, :] = 255
            continue

        if left2 != right1 and left1 != right2:
            mask[i, :left1] = 255
            mask[i, right1:] = 255

    return mask


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


def main(image_name):
    img = open_img(image_name, False)
    h, w, _ = img.shape

    # resize image
    img = resize_image(img)

    # filter after white color
    mask = cv2.inRange(
        img, lowerb=np.array([100, 100, 150]), upperb=np.array([255, 255, 255])
    )

    # remove noise
    cv2.imshow("Image0", mask)
    mask = morph_open(mask, 3)

    masked_roi, m = get_region_of_interest(mask)
    masked_roi = ~masked_roi

    # get the middle line
    middle_lane_mask = get_middle_lane(masked_roi)
    color_img = open_img(image_name, False)
    color_img = resize_image(color_img)

    color_mask = np.zeros_like(color_img)
    color_mask[masked_roi != 0] = [0, 255, 0]
    color_img = cv2.addWeighted(color_mask, 0.3, color_img, 1 - 0.3, 0)

    color_img[middle_lane_mask != 0] = [255, 0, 0]

    # cv2.line(color_img, first_point, end_point, (255, 0, 0), 2)

    cv2.imshow("Image1", mask)
    cv2.imshow("Image2", masked_roi)
    cv2.imshow("Imagefinal", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_name = sys.argv[1]
    main(image_name)
