package com.example.app;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

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

    private Mat mRGBA = null;
    private Mat mImgGray = null;
    private Mat mOut = null;
    private Mat mRot90 = null;


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
        this.mRGBA = new Mat(height, width, CvType.CV_8UC4);
        this.mImgGray = new Mat(height, width, CvType.CV_8UC1);
        this.mOut = new Mat(height, width, CvType.CV_8UC1);

        this.mRot90 = Imgproc.getRotationMatrix2D(new Point(height / 2.0F, width / 2.0F), -90, 1);
    }

    @Override
    public void onCameraViewStopped() {
        if (this.mRGBA != null)
            this.mRGBA.release();
    }

    private Mat doWork(Mat frame) {
        // TODO
        return null;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        this.mRGBA = inputFrame.rgba();

        Imgproc.cvtColor(this.mRGBA, this.mImgGray, Imgproc.COLOR_RGBA2GRAY);

        Mat mEdges = new Mat(this.mImgGray.size(), CvType.CV_8UC1);

        Imgproc.Canny(this.mImgGray, mEdges, 50.0, 100.0);

        // fix orientation
        Imgproc.warpAffine(mEdges, this.mOut, this.mRot90, mEdges.size());

        return this.mOut;
    }
}

