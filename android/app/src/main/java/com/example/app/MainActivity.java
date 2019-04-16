package com.example.app;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private JavaCameraView javaCameraView = null;

    private BaseLoaderCallback mLoader = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == BaseLoaderCallback.SUCCESS) {
                if (javaCameraView != null)
                    javaCameraView.enableView();
            }
            else {
                super.onManagerConnected(status);
            }
        }
    };

    private Mat matImg = null;
    private Mat matRot90 = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        javaCameraView = findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d("[main-activity]", "Problem while loading OpenCV...");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoader);
        }
        else {
            Log.d("[main-activity]", "Successfully loaded OpenCV");
            mLoader.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        // 1056(w) x 704(h)
        this.matImg = new Mat(height, width, CvType.CV_8UC3);
        this.matRot90 = Imgproc.getRotationMatrix2D(new Point(height / 2.0F, width / 2.0F), -90, 1);
    }

    @Override
    public void onCameraViewStopped() {
        if (this.matImg != null)
            this.matImg.release();
    }


    private Mat morphClose(Mat img, int numIter) {
        Mat strel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));

        for (int i = 0; i < numIter; i++) {
            Imgproc.dilate(img, img, strel);
            Imgproc.erode(img, img, strel);
        }

        return img;
    }

    private Pair<MatOfPoint, MatOfPoint> separateContours(List<MatOfPoint> contours) {
        /**
         * MatOfPoint M size = 1(width/cols) x N(height/rows)
         * M[0, i] = double[2] = Point(x, y)
         */

        if (contours.size() < 2)
            return null;

        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint o1, MatOfPoint o2) {
                return o2.rows() - o1.rows();
            }
        });

        Pair<MatOfPoint, MatOfPoint> pair = new Pair<>(contours.get(0), contours.get(1));

        // TODO: use median / mean to find left / right

        return pair;
    }

    private List<Point> getMiddleLane(Mat matEdgeBin) {
        // TODO: don't draw contours, directly construct `points` from contourPair

        Mat matOut = new Mat(matEdgeBin.size(), CvType.CV_8UC1);
        List<Point> points = new ArrayList<>();

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(matEdgeBin, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // get left/right contours
        Pair<MatOfPoint, MatOfPoint> contourPair = separateContours(contours);

        // no sufficient contours
        if (contourPair == null)
            return new ArrayList<>();


        return points;
    }

    private String matToString(Mat mat) {
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


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // TODO: USE HSL color space
        // TODO: use separate thread for processing

        String mode = "rgb";

        // fix orientation
        Imgproc.warpAffine(inputFrame.rgba(), this.matImg, this.matRot90, this.matImg.size());

        // get input image as RGB
        Imgproc.cvtColor(this.matImg, this.matImg, Imgproc.COLOR_RGBA2RGB);

        // thresholding for white
        Mat mask = new Mat(this.matImg.size(), CvType.CV_8UC1);

        if (mode == "hsv") {
            Imgproc.cvtColor(this.matImg, this.matImg, Imgproc.COLOR_RGB2HSV);
            Core.inRange(this.matImg, new Scalar(0, 0, 200), new Scalar(360, 10, 255), mask);
        }
        if (mode == "rgb") {
            Core.inRange(this.matImg, new Scalar(150, 150, 150), new Scalar(255, 255, 255), mask);
        }
        if (mode == "gray") {
            Mat matGray = new Mat(this.matImg.size(), CvType.CV_8UC1);
            Imgproc.cvtColor(this.matImg, matGray, Imgproc.COLOR_RGB2GRAY);
            Core.inRange(matGray, new Scalar(200), new Scalar(255), mask);
            matGray.release();
        }

        // Canny
        Imgproc.GaussianBlur(mask, mask, new Size(5, 5), 2);
        morphClose(mask, 2);

        Mat matEdgeBin = new Mat(mask.size(), CvType.CV_8UC1);
        Imgproc.Canny(mask, matEdgeBin, 140, 160);

        mask.release();

        morphClose(matEdgeBin, 1);
//        Core.divide(matEdgeBin, new Scalar(255.0), matEdgeBin);

        // HOUGH
        Mat matLines = new Mat(this.matImg.size(), CvType.CV_8UC1);

        Imgproc.HoughLines(matEdgeBin, matLines, 4, Math.PI / 180.0, 250);

        for (int i = 0; i < matLines.rows(); i++) {
            double[] data = matLines.get(i, 0);

            if (data == null || data.length < 2)
                continue;

            double rho1 = data[0];
            double theta1 = data[1];
            double cosTheta = Math.cos(theta1);
            double sinTheta = Math.sin(theta1);

            Log.i("HOUGH theta", "lineIdx[" + i + "] theta:" + theta1 * 180.0 / Math.PI);

            // theta thresholding
//            if (theta1 < -Math.PI / 3 || theta1 > Math.PI / 3)
//                continue;


            double x0 = cosTheta * rho1;
            double y0 = sinTheta * rho1;
            Point pt1 = new Point(x0 + 5000 * (-sinTheta), y0 + 5000 * cosTheta);
            Point pt2 = new Point(x0 - 5000 * (-sinTheta), y0 - 5000 * cosTheta);

            Imgproc.line(this.matImg, pt1, pt2, new Scalar(0, 255, 0), 5);
        }

//        return matEdgeBin;
        return this.matImg;
    }
}

