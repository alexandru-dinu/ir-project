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

    private Mat matInput = null;
    private Mat matGray = null;
    private Mat matOutput = null;
    private Mat matRot90 = null;

    /**
     * PARAMS
     */
    private Scalar lowerBound = new Scalar(120, 120, 120);
    private Scalar upperBound = new Scalar(255, 255, 255);
    private int thr1 = 80;
    private int thr2 = 120;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
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
        this.matInput = new Mat(height, width, CvType.CV_8UC4);
        this.matGray = new Mat(height, width, CvType.CV_8UC1);
        this.matOutput = new Mat(height, width, CvType.CV_8UC1);

        this.matRot90 = Imgproc.getRotationMatrix2D(new Point(height / 2.0F, width / 2.0F), -90, 1);
    }

    @Override
    public void onCameraViewStopped() {
        if (this.matInput != null)
            this.matInput.release();
    }


    private Mat doCanny(Mat matRawFrame) {
        Imgproc.cvtColor(matRawFrame, this.matGray, Imgproc.COLOR_RGBA2GRAY);
        Mat mEdges = new Mat(this.matGray.size(), CvType.CV_8UC1);
        Imgproc.Canny(this.matGray, mEdges, 50.0, 100.0);

        // fix orientation
        Imgproc.warpAffine(mEdges, this.matOutput, this.matRot90, mEdges.size());

        return this.matOutput;
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
        if (contours.size() < 2)
            return null;


        // TODO: implement
        return new Pair<>(contours.get(0), contours.get(1));
    }

    private List<Point> getMiddleLine(Mat img) {
//        Mat outLeft = new Mat(img.size(), CvType.CV_8UC1);
//        Mat outRight = new Mat(img.size(), CvType.CV_8UC1);
        Mat out = new Mat(img.size(), CvType.CV_8UC1);

        List<MatOfPoint> contours = new ArrayList<>();
        Imgproc.findContours(img, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Pair<MatOfPoint, MatOfPoint> cnts = separateContours(contours);
        if (cnts == null)
            return new ArrayList<>();

//        Imgproc.drawContours(outLeft, Arrays.asList(cnts.first), 0, new Scalar(1), 1);
//        Imgproc.drawContours(outRight, Arrays.asList(cnts.second), 0, new Scalar(1), 1);

//        for (int i = 0; i < img.height(); i++) {
//            int ll = -1, rr = -1;
//
//            // FIXME - optimize
//            for (int j = 0; j < img.width(); j++) {
//                double[] c_ll = outLeft.get(i, j);
//                double[] c_rr = outLeft.get(i, j);
//                if (c_ll[0] > 0)
//                    ll = j;
//                if (c_rr[0] > 0)
//                    rr = j;
//            }
//
//            if (ll > -1 && rr > -1) {
//                double[] data = new double[]{1};
//                out.put(i, ll + (rr - ll) / 2, data);
//            }
//        }

//        outLeft.release();
//        outRight.release();
//        out = morphClose(out, 1);

        List<Point> points = new ArrayList<>();

        for (int i = 0; i < out.height(); i++) {
            for (int j = 0; j < out.width(); j++) {
                if (out.get(i, j)[0] > 0) {
                    points.add(new Point(i, j));
                }
            }
        }


        return points;
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // TODO: USE HSL color space
        // fix orientation
        Imgproc.warpAffine(inputFrame.rgba(), this.matInput, this.matRot90, this.matInput.size());

        // get input image as RGB
        Imgproc.cvtColor(this.matInput, this.matInput, Imgproc.COLOR_RGBA2RGB);

        // thresholding for white
//        Imgproc.cvtColor(this.matInput, this.matInput, Imgproc.COLOR_BGR2HSV);
        Mat mask = new Mat(this.matInput.size(), CvType.CV_8UC1);
        Core.inRange(this.matInput, new Scalar(120, 120, 120), new Scalar(255, 255, 255), mask);

        // Canny
        Imgproc.GaussianBlur(mask, mask, new Size(5, 5), 2);
        Mat edges = new Mat(mask.size(), CvType.CV_8UC1);
        Imgproc.Canny(mask, edges, this.thr1, this.thr2);
        mask.release();
        edges = morphClose(edges, 1);
//        Core.divide(edges, new Scalar(255.0), edges);

//        List<Point> points = getMiddleLine(edges);
//
//        this.matOutput = this.matInput;
//
//        for (Point p : points) {
//            Imgproc.circle(this.matOutput, p, 1, new Scalar(0, 255, 0), -1);
//        }

        return edges;
    }
}

