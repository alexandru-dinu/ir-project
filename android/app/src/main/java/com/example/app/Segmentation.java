package com.example.app;

import android.util.Log;

import org.apache.commons.lang3.ArrayUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Segmentation {

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

    public static void draw_middle_lane(Mat mask, List<Point> roi_vertices, Mat dst) {
        // mask is a 1-channel image

        if (mask.channels() != 1)
            throw new AssertionError("src must have 1 channel");

        if (mask.type() != CvType.CV_8UC1)
            throw new AssertionError("src must have type CV_8UC1");

        int h = mask.height(), w = mask.width();
        int thickness = 7;
        double[] val = new double[]{0, 255, 0};

        Point p1 = roi_vertices.get(0);
        Point p2 = roi_vertices.get(1);
        Point p3 = roi_vertices.get(2);
        Point p4 = roi_vertices.get(3);

        Imgproc.line(dst, p1, p2, new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, p2, p3, new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, p3, p4, new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, p4, p1, new Scalar(255, 0, 0), 3);

        int h_low = (int) p1.y;
        int h_hi = (int) p3.y;
        int w_low = (int) p1.x;
        int w_hi = (int) p2.x;

        Imgproc.line(dst, new Point(w_low, h_low), new Point(w_hi, h_low), new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, new Point(w_hi, h_low), new Point(w_hi, h_hi), new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, new Point(w_hi, h_hi), new Point(w_low, h_hi), new Scalar(255, 0, 0), 3);
        Imgproc.line(dst, new Point(w_low, h_hi), new Point(w_low, h_low), new Scalar(255, 0, 0), 3);


        for (int i = h_low; i < h_hi; i++) {
//            int left = Utils.argmax(i, w_low, w_hi, mask);
//            int right = Utils.reverse_argmax(i, w_hi, w_low, mask);

            int left = -1, right = -1;
            int j;


            j = w_low;
            while (j < w_hi) {
                double[] d1 = mask.get(i, j - 1);
                double[] d2 = mask.get(i, j);
                int data1 = (int) Math.round(d1[0]);
                int data2 = (int) Math.round(d2[0]);

                if ((data1 == 0) && (data2 == 255)) {
                    left = j;
                    break;
                }

                j++;
            }

            j = w_hi;
            while (j > 0) {
                double[] d1 = mask.get(i, j - 1);
                double[] d2 = mask.get(i, j);
                int data1 = (int) Math.round(d1[0]);
                int data2 = (int) Math.round(d2[0]);

                if ((data1 == 255) && (data2 == 0)) {
                    right = j;
                    break;
                }

                j--;
            }


            if (left != -1 && right != -1) {
                int mid = (int) Math.round(0.5 * (left + right));

                Log.i("MIDD=====", "" + mid);

                for (int t = -(thickness / 2); t < thickness / 2; t++) {
                    dst.put(i, mid + t, val);
                }
            }

        }
    }
}
