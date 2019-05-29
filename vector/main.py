import threading

import anki_vector
from anki_vector import behavior
from anki_vector import events
import matplotlib.pyplot as plt

import numpy as np

from utils import *

c = 0


def on_new_raw_camera_image(robot, event_type, event, done):
    global c

    print(f"GOT HERE {c}")

    # img = cv2.cvtColor(np.array(event.image), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(np.array(event.image), cv2.COLOR_BGR2HSV)

    h_hist = np.histogram(img[:, :, 0], bins=180)
    s_hist = np.histogram(img[:, :, 1], bins=255)
    v_hist = np.histogram(img[:, :, 2], bins=255)


    #plt.plot(h_hist)
    #plt.plot(s_hist)
    #plt.plot(v_hist)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.medianBlur(img, 5)
    # img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)



    # 1. color thresholding
    # mask = cv2.inRange(img, lowerb=200, upperb=255)
    mask = cv2.inRange(img, lowerb=np.array([255, 180, 0]), upperb=np.array([255, 255, 170]))
    # mask = cv2.inRange(img,
    #                    lowerb=np.array([20, 30, 40], dtype=np.uint8),
    #                    upperb=np.array([40, 60, 70], dtype=np.uint8)
    #                    )

    # 2. noise removal
    # mask = morph_close(mask, num_iter=3)

    # # 3. ROI
    # masked_roi, _ = region_of_interest(mask)
    # masked_roi = ~masked_roi
    #
    # # 4. get middle lane
    # middle_lane_mask = get_middle_lane(masked_roi)
    # color_mask = np.zeros(img.shape)
    # color_mask[masked_roi != 0] = [0, 255, 0]
    # color_img = cv2.addWeighted(color_mask, 0.3, img, 1 - 0.3, 0)
    # color_img[middle_lane_mask != 0] = [255, 0, 0]

    # out_img = np.zeros(img.shape)
    # for i in range(3):
    #     out_img[:, :, i] = cv2.bitwise_and(img[:, :, i], mask)

    # show stream
    cv2.imshow("stream", mask)
    cv2.waitKey(1)

    if c == 5000:
        done.set()
    c += 1


def main():
    args = anki_vector.util.parse_command_args()

    with behavior.ReserveBehaviorControl():

        with anki_vector.Robot(args.serial, enable_face_detection=False) as robot:
            robot.camera.init_camera_feed()
            robot.motors.set_lift_motor(5)
            done = threading.Event()
            robot.events.subscribe(on_new_raw_camera_image, events.Events.new_raw_camera_image, done)

            print("------ waiting for imgs, press ctrl+c to exit early ------")

            try:
                done.wait()
            except KeyboardInterrupt:
                pass

        robot.events.unsubscribe(on_new_raw_camera_image, events.Events.new_raw_camera_image)


if __name__ == '__main__':
    main()
