package com.example.app;

import android.util.Log;
import android.util.Pair;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Utils {
    public static Mat resize_image(Mat src, int width, int height) {
        Mat dst = new Mat();

        Imgproc.resize(src, dst, new Size(width, height));

        return dst;
    }

    public static Mat morph_close(Mat src, int num_iter) {
        Mat strel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

        for (int i = 0; i < num_iter; i++) {
            Imgproc.dilate(src, src, strel);
            Imgproc.erode(src, src, strel);
        }

        return src;
    }

    public static Mat morph_open(Mat src, int num_iter) {
        Mat strel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

        for (int i = 0; i < num_iter; i++) {
            Imgproc.erode(src, src, strel);
            Imgproc.dilate(src, src, strel);
        }

        return src;
    }

    public static Pair<Mat, List<Point>> region_of_interest(Mat src) {

        if (src.type() != CvType.CV_8UC1)
            throw new AssertionError("Invalid src type, must be CV_8UC1");

        int w = src.width();
        int h = src.height();
        double sx = 0.2;
        double sy = 0.15;
        int delta = 250;

        MatOfPoint vertices = new MatOfPoint(
                new Point(0.5 * (w - delta), sy * h),
                new Point(0.5 * (w + delta), sy * h),
                new Point((1 - sx) * w, h - 1),
                new Point(sx * w, h - 1)
        );

        Scalar ignore_mask_color, zero_scalar;

        if (src.channels() == 1) {
            ignore_mask_color = new Scalar(255);
            zero_scalar = new Scalar(0);
        }
        else {
            ignore_mask_color = new Scalar(255, 255, 255);
            zero_scalar = new Scalar(0, 0, 0);
        }

        Mat dst = new Mat(src.rows(), src.cols(), src.type());
        Mat mask = new Mat(src.rows(), src.cols(), src.type(), zero_scalar);

        Imgproc.fillPoly(mask, new ArrayList<>(Arrays.asList(vertices)), ignore_mask_color);

        Core.bitwise_and(src, mask, dst);

        return new Pair<>(dst, vertices.toList());
    }

    public static void fill_row(Mat mat, int row, int value) {
        for (int i = 0; i < mat.cols(); i++) {
            mat.put(row, i, new int[]{value});
        }
    }

    public static void fill_row_from(Mat mat, int row, int lim, int value) {
        for (int i = lim; i < mat.cols(); i++) {
            mat.put(row, i, new int[]{value});
        }
    }

    public static void fill_row_up_to(Mat mat, int row, int lim, int value) {
        for (int i = 0; i < lim; i++) {
            mat.put(row, i, new int[]{value});
        }
    }

    public static int argmax(int row, int low, int hi, Mat data) {
        int idx = -1;
        double max = -1;

        for (int i = low; i < hi; i++) {
            Log.i("LIM", "" + low + ";" + hi + ";" + i);
            double[] x = data.get(row, i);

            if (x[0] > max) {
                idx = i;
                max = x[0];
            }
        }

        return idx;
    }

    public static int reverse_argmax(int row, int hi, int low, Mat data) {
        int idx = -1;
        double max = -1;

        for (int i = hi; i >= low; i--) {
            double[] x = data.get(row, i);

            if (x[0] > max) {
                idx = i;
                max = x[0];
            }
        }

        return idx;
    }


    public static String mat_to_string(Mat mat) {
        StringBuilder s = new StringBuilder();
        for (int x = 0; x < mat.height(); x++) {
            for (int y = 0; y < mat.width(); y++) {
                if (mat.get(x, y)[0] > 0)
                    s.append(String.valueOf(mat.get(x, y)[0])).append(";");
            }
            s.append("\n");
        }
        return s.toString();
    }
}
