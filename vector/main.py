import threading

import anki_vector
from anki_vector import behavior
from anki_vector import events

from utils import *

FRAME_COUNT = 0

# mm per second
INITIAL_SPEED_L, INITIAL_SPEED_R = 30, 30


def on_new_raw_camera_image(robot, event_type, event, done):
    global FRAME_COUNT

    print(f"FRAME COUNT {FRAME_COUNT}")

    raw_img = np.array(event.image)

    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

    h, w, c = rgb_img.shape

    # 0. noise removal
    # img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(gray_img, ksize=(5, 5), sigmaX=0)

    # 1. color thresholding
    mask = cv2.inRange(img, lowerb=0, upperb=130)
    # mask = cv2.inRange(img, lowerb=np.array([0, 0, 0]), upperb=np.array([50, 50, 50]))

    # 2. noise removal
    mask = morph_close(mask, num_iter=5)

    # 3. get region of interest
    roi_masked_image, roi_vertices = get_region_of_interest(mask, sx=0.23, sy=0.15, delta=200, return_vertices=True)
    draw_roi(rgb_img, roi_vertices)

    # 4. get and draw middle guidance points
    spacing, center_points, weights = get_guidance_points_and_weights(w, h, num=6, bottom_to_top=True)
    for p in center_points:
        cv2.circle(rgb_img, tuple(p), 5, (0, 0, 255), -1)

    # 5. find middle points on track
    diffs = np.zeros((len(center_points),))
    for i in range(len(center_points)):
        ctr_x, ctr_y = center_points[i]

        left, right = get_left_right_markers(roi_masked_image[ctr_y])

        # if there's no mid point on track corresponding to current center point,
        # use previous information to compute the diff
        if left == right == 0:
            neg = len(diffs[:i][diffs[:i] < 0])
            pos = len(diffs[:i][diffs[:i] >= 0])
            diffs[i] = -ctr_x if neg > pos else ctr_x
            mid = ctr_x

        # there is a mid point on track, corresponding to the current center point,
        # compute the difference between that point and current center point
        else:
            mid = int(0.5 * (left + right))
            diffs[i] = mid - ctr_x

        # draw point
        cv2.circle(rgb_img, (mid, ctr_y), 5, (255, 0, 0), -1)

    # 6. compute speed factor based on X-difference
    speed_delta = compute_speed_delta(diffs, weights, div_factor=8)
    print("DIFFS", diffs, "SPEED_DELTA:", speed_delta)

    # wait 100 frames before moving the robot
    if FRAME_COUNT >= 100:
        robot.motors.set_wheel_motors(INITIAL_SPEED_L + speed_delta, INITIAL_SPEED_R - speed_delta)

    # show stream
    cv2.imshow("roi", roi_masked_image)
    cv2.imshow("stream", rgb_img)
    cv2.waitKey(1)

    FRAME_COUNT += 1
    if FRAME_COUNT == 5000:
        done.set()


def main():
    args = anki_vector.util.parse_command_args()

    # keep robot still
    with behavior.ReserveBehaviorControl():
        # get camera feed
        with anki_vector.Robot(args.serial, enable_face_detection=False) as robot:
            robot.camera.init_camera_feed()

            robot.motors.set_lift_motor(5)
            robot.motors.set_head_motor(-5)

            done = threading.Event()
            robot.events.subscribe(on_new_raw_camera_image, events.Events.new_raw_camera_image, done)

            print("> waiting for imgs, press CTRL+C to exit early")

            try:
                done.wait()
            except KeyboardInterrupt:
                pass

        robot.events.unsubscribe(on_new_raw_camera_image, events.Events.new_raw_camera_image)


if __name__ == '__main__':
    main()
