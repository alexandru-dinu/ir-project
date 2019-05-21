package com.example.app;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class SegmentationBased {

    public static Mat get_lane_mask(Mat src) {
        // src is a binary image, 1 channel

        if (src.channels() != 1) {
            throw new AssertionError("src must have 1 channel");
        }

        Mat mask = src.clone();

        int h = src.height();
        int w = src.width();

        // TODO: might be slow

        for (int i = 0; i < h; i++) {
            int left1 = -1, left2 = -1;
            int right1 = -1, right2 = -1;

            for (int j = 1; j < w; j++) {
                int[] data1 = new int[1];
                int[] data2 = new int[1];

                mask.get(i, j - 1, data1);
                mask.get(i, j, data2);

                if ((data1[0] != 255) && (data2[0] == 255))
                    left1 = j;
                if ((data1[0] == 255) && (data2[0] != 255))
                    left2 = j;

                if ((left1 > 0) && (left2 > 0))
                    break;
            }

            for (int j = w - 2; j >= 0; j--) {
                int[] data1 = new int[1];
                int[] data2 = new int[1];

                mask.get(i, j + 1, data1);
                mask.get(i, j, data2);

                if ((data1[0] != 255) && (data2[0] == 255))
                    right1 = j;
                if ((data1[0] == 255) && (data2[0] != 255))
                    right2 = j;

                if ((right1 > 0) && (right2 > 0))
                    break;
            }

            if (left1 == -1) {
                Utils.fill_row(mask, i, 255);
                continue;
            }

            if ((left2 != right1) && (left1 != right2)) {
                Utils.fill_row_up_to(mask, i, left1, 255);
                Utils.fill_row_from(mask, i, right1, 255);
            }
        }

        return mask;
    }

    public static Mat get_middle_lane(Mat mask) {
        // mask is a 1-channel image

        if (mask.channels() != 1) {
            throw new AssertionError("src must have 1 channel");
        }

        if (mask.type() != CvType.CV_8UC1) {
            throw new AssertionError("src must have type CV_8UC1");
        }

        int h = mask.height(), w = mask.width();

        Mat lane_mask = new Mat(h, w, mask.type(), new Scalar(0));

        for (int i = 100; i < h; i++) {
            int left = 0, right = w;

            for (int j = (int) (0.3 * w); j < (int) (0.7 * w); j++) {
                double[] d1 = mask.get(i, j - 1);
                double[] d2 = mask.get(i, j);
                int data1 = (int) Math.round(d1[0]);
                int data2 = (int) Math.round(d2[0]);


                if ((data1 == 0) && (data2 == 255))
                    left = j;

                if ((data1 == 255) && (data2 == 0))
                    right = j;
            }

            int mid = (int) Math.round(0.5 * (left + right));

            double[] val = {255};

            lane_mask.put(i, mid, val);
            lane_mask.put(i, mid - 1, val);
            lane_mask.put(i, mid + 1, val);
        }

        return lane_mask;
    }
}
