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
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

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
    private Queue<Pair<Point, Point>> leftLines = new LinkedList<>();
    private Queue<Pair<Point, Point>> rightLines = new LinkedList<>();
    private int windowSize = 7;


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


    private Queue<Pair<Point, Point>> windowedAdd(Queue<Pair<Point, Point>> pts, Pair<Point, Point> pt) {
        if (pts.size() < this.windowSize) {
            pts.add(pt);
        }

        if (pts.size() == this.windowSize) {
            pts.remove();
            pts.add(pt);
        }

        return pts;
    }

    private Pair<Point, Point> averageLine(Queue<Pair<Point, Point>> lines) {
        double fx = 0.0, fy = 0.0, sx = 0.0, sy = 0.0;

        for (Pair<Point, Point> line : lines) {
            fx += line.first.x / windowSize;
            fy += line.first.y / windowSize;
            sx += line.second.x / windowSize;
            sy += line.second.y / windowSize;
        }

        return new Pair<>(new Point(fx, fy), new Point(sx, sy));
    }

    private Pair<Point, Point> swap(Pair<Point, Point> p) {
        return new Pair<>(p.second, p.first);
    }

    private Mat drawLines() {
        Pair<Point, Point> leftLine = averageLine(this.leftLines);
        Pair<Point, Point> rightLine = averageLine(this.rightLines);
        List<Point> midPoints = new ArrayList<>();
        int numPoints = 100;

        Imgproc.line(this.matImg, leftLine.first, leftLine.second, new Scalar(255, 0, 0), 5);
        Imgproc.line(this.matImg, rightLine.first, rightLine.second, new Scalar(0, 0, 255), 5);


        leftLine = leftLine.first.y > leftLine.second.y ? swap(leftLine) : leftLine;
        rightLine = rightLine.first.y > rightLine.second.y ? swap(rightLine) : rightLine;

        // middle line
        for (int i = 0; i < numPoints; i++) {
            double a = 1.0 * i / (numPoints - 1);
            double xLeft, yLeft, xRight, yRight;

            xLeft = (1 - a) * leftLine.first.x + a * leftLine.second.x;
            yLeft = (1 - a) * leftLine.first.y + a * leftLine.second.y;
            xRight = (1 - a) * rightLine.first.x + a * rightLine.second.x;
            yRight = (1 - a) * rightLine.first.y + a * rightLine.second.y;

            int x = (int) Math.round(0.5 * (xLeft + xRight));
            int y = (int) Math.round(0.5 * (yLeft + yRight));

            midPoints.add(new Point(x, y));
        }


        for (Point p : midPoints) {
            Log.i("POINT", p.toString());
            Imgproc.circle(this.matImg, p, 5, new Scalar(0, 255, 0), -1);
        }


        return this.matImg;
    }


    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        // TODO: USE HSL color space
        // TODO: use separate thread for processing

        int width = inputFrame.gray().width();

        String mode = "rgb";

        // fix orientation
        Imgproc.warpAffine(inputFrame.rgba(), this.matImg, this.matRot90, this.matImg.size());

        // get input image as RGB
        Imgproc.cvtColor(this.matImg, this.matImg, Imgproc.COLOR_RGBA2RGB);
        // matImg shape = 1056(w) x 704(h)

        // 1. white thresholding
        Mat mask = new Mat(this.matImg.size(), CvType.CV_8UC1);
        Core.inRange(this.matImg, new Scalar(150, 150, 150), new Scalar(255, 255, 255), mask);

        // 2. remove noise
        Imgproc.GaussianBlur(mask, mask, new Size(5, 5), 2);
        // Utils.morph_close(mask, 3); TODO
        Utils.morph_open(mask, 3);

        // 3. get masked roi
        Mat masked_roi = Utils.region_of_interest(mask);
        Core.bitwise_not(masked_roi, masked_roi);

        // 4. get middle lane
        Mat middle_lane_mask = SegmentationBased.get_middle_lane(masked_roi);
        Mat color_mask = new Mat(this.matImg.rows(), this.matImg.cols(), this.matImg.type(), new Scalar(0, 0, 0));

        // color_mask[masked_roi != 0] = [0, 255, 0]
        for (int i = 0; i < masked_roi.rows(); i++) {
            for (int j = 0; j < masked_roi.cols(); j++) {
                double[] data = masked_roi.get(i, j);

                if ((int) Math.round(data[0]) != 0)
                    color_mask.put(i, j, new double[]{0, 255, 0});
            }
        }

        Core.addWeighted(color_mask, 0.3, this.matImg, 0.7, 0, this.matImg);

        // color_img[middle_lane_mask != 0] = [255, 0, 0]
        for (int i = 0; i < middle_lane_mask.rows(); i++) {
            for (int j = 0; j < middle_lane_mask.cols(); j++) {
                double[] data = middle_lane_mask.get(i, j);

                if ((int) Math.round(data[0]) != 0)
                    this.matImg.put(i, j, new double[]{255, 0, 0});
            }
        }

        return this.matImg;
    }
}

