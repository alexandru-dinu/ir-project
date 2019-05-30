import argparse
import warnings

from scipy.interpolate import splev, splprep

from utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--interp', type=int, default=20)
parser.add_argument('--save', action='store_true')
args = parser.parse_args()


def save_or_show(img, suffix=""):
    if args.save:
        name = args.file.split('/')[-1].split('.')[0]
        save_img(img, name=f'../out/out-{name}{suffix}.png', from_bgr=True)
    else:
        show(img, from_bgr=True)


def smooth_contour(cnt, num=25):
    x, y = cnt.T
    x = x.tolist()[0]
    y = y.tolist()[0]

    # find the B-spline representation of the contour
    tck, u = splprep([x, y], u=None, s=1.0, per=1)
    u_new = np.linspace(u.min(), u.max(), num)

    # evaluate spline given points and knots
    x_new, y_new = splev(u_new, tck, der=0)

    s_cnt = np.array(list(zip(x_new, y_new))).astype(np.int)

    return s_cnt


def separate_contours(contours):
    assert len(contours) >= 2

    contours = sorted(contours, key=lambda x: len(x), reverse=True)
    contours = contours[:2]

    w0s = np.median([e[0][0] for e in contours[0]])
    w1s = np.median([e[0][0] for e in contours[1]])

    idx_left = 0 if w0s < w1s else 1
    idx_right = 1 - idx_left

    cnt_l = np.transpose(contours[idx_left], (0, 2, 1)).squeeze()  # size = N x 2
    cnt_r = np.transpose(contours[idx_right], (0, 2, 1)).squeeze()  # size = N x 2

    # cnt_l = smooth_contour(contours[idx_left], num=args.interp)
    # cnt_r = smooth_contour(contours[idx_right], num=args.interp)

    return cnt_l, cnt_r


def get_middle_line(img):
    assert img[img == 1].size + img[img == 0].size == img.size

    # show(img)

    h, w = img.shape
    out_l = np.zeros((h, w), dtype=np.uint8)
    out_r = np.zeros((h, w), dtype=np.uint8)
    out = np.zeros((h, w), dtype=np.uint8)

    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(f"Found {len(contours)} contours")

    cnt_l, cnt_r = separate_contours(contours)

    cv2.drawContours(out_l, [cnt_l], 0, 1, thickness=1)
    cv2.drawContours(out_r, [cnt_r], 0, 1, thickness=1)

    for i in range(h):
        ll = np.argwhere(out_l[i, :] == 1)
        rr = np.argwhere(out_r[i, :] == 1)

        # FIXME
        # if ll.size == 0 and rr.size > 0:
        #     out[i, rr[0] // 2] = 1
        # if ll.size > 0 and rr.size == 0:
        #     out[i, ll[-1] * 2] = 1
        # --

        if ll.size > 0 and rr.size > 0:
            pl, pr = ll[-1], rr[0]
            m = abs(pr - pl) // 2
            out[i, pl + m] = 1

    out = morph_close(out, num_iter=1)

    # show(out + img)

    return out


def on_video():
    capture = cv2.VideoCapture(args.file)
    assert capture.isOpened()

    fps = capture.get(cv2.CAP_PROP_FPS)
    __plot = None

    count = 0

    while capture.isOpened():
        ret, img = capture.read()
        count += 1

        if count == 180:
            return

        # if ret and count % 2 != 1:
        #     continue

        sf = 4
        h, w, c = img.shape
        img = cv2.resize(img, (int(h / sf), int(w / sf)))

        # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, lowerb=np.array([150, 150, 150]), upperb=np.array([255, 255, 255]))
        # show(mask)

        mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
        edges = cv2.Canny(mask, threshold1=10, threshold2=30)
        edges = morph_close(edges, num_iter=1)
        edges = np.divide(edges, 255).astype(np.uint8)
        # show(edges)

        mid_line = get_middle_line(edges)
        ps = np.argwhere(mid_line == 1)
        for (x, y) in ps:
            cv2.circle(img, (y, x), 1, (0, 255, 0), thickness=-1)

        cv2.imshow('out', img)
        cv2.waitKey(1)
        continue

        # play
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        img = np.dstack((r, g, b))
        if __plot is None:
            __plot = plt.imshow(img)
        else:
            __plot.set_data(img)
        plt.pause(1 / fps)
        plt.draw()


def with_hough(img, edges):
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180.0, threshold=70)

    thr = 30.0

    for i, line in enumerate(lines):
        rho, theta = line[0]

        if (thr / 180) * np.pi < theta < ((180.0 - thr) / 180) * np.pi:
            print(f"[{i}]SKIP theta: {theta * 180.0 / np.pi}")
            continue

        print(f"[{i}]OK theta: {theta * 180.0 / np.pi}")

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    save_or_show(img, suffix='-hough')


def on_image(img=None):
    img = open_img(args.file, gray=False)
    h, w, c = img.shape

    sf = 6.3  # scale factor: scale original image to 640x480
    img = cv2.resize(img, (int(h / sf), int(w / sf)))

    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img, lowerb=np.array([150, 150, 150]), upperb=np.array([255, 255, 255]))
    # show(mask)

    mask = cv2.GaussianBlur(mask, ksize=(5, 5), sigmaX=2)
    edges = cv2.Canny(mask, threshold1=80, threshold2=120)
    edges = morph_close(edges, num_iter=1)
    # show(edges); return

    if True:
        with_hough(img, edges)
        return

    edges = np.divide(edges, 255).astype(np.uint8)
    # show(edges)

    mid_line = get_middle_line(edges)
    ps = np.argwhere(mid_line == 1)
    for (x, y) in ps:
        cv2.circle(img, (y, x), 1, (0, 255, 0), thickness=-1)

    # show(img, from_bgr=True)
    save_or_show(img)


def main():
    if 'jpg' in args.file:
        on_image()
    elif 'mp4' in args.file:
        on_video()


if __name__ == '__main__':
    main()
