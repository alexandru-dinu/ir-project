package com.example.app;

public class ContourBased {
    //        // thresholding for white
//        Mat mask = new Mat(this.matImg.size(), CvType.CV_8UC1);
//
//        if (mode == "hsv") {
//            Imgproc.cvtColor(this.matImg, this.matImg, Imgproc.COLOR_RGB2HSV);
//            Core.inRange(this.matImg, new Scalar(0, 0, 200), new Scalar(360, 10, 255), mask);
//        }
//        if (mode == "rgb") {
//            Core.inRange(this.matImg, new Scalar(150, 150, 150), new Scalar(255, 255, 255), mask);
//        }
//        if (mode == "gray") {
//            Mat matGray = new Mat(this.matImg.size(), CvType.CV_8UC1);
//            Imgproc.cvtColor(this.matImg, matGray, Imgproc.COLOR_RGB2GRAY);
//            Core.inRange(matGray, new Scalar(200), new Scalar(255), mask);
//            matGray.release();
//        }

//        // Canny
//        Imgproc.GaussianBlur(mask, mask, new Size(5, 5), 2);
//        morphClose(mask, 2);
//
//        Mat matEdgeBin = new Mat(mask.size(), CvType.CV_8UC1);
//        Imgproc.Canny(mask, matEdgeBin, 140, 160);
//
//        mask.release();
//
//        morphClose(matEdgeBin, 1);
////        Core.divide(matEdgeBin, new Scalar(255.0), matEdgeBin);
//
//        // HOUGH
//        Mat matLines = new Mat(this.matImg.size(), CvType.CV_8UC1);
//
//        Imgproc.HoughLines(matEdgeBin, matLines, 4, Math.PI / 180.0, 250);
//
//        int d = 30;
//        double f = 0.3;
//
//        for (int i = 0; i < matLines.rows(); i++) {
//            double[] data = matLines.get(i, 0);
//
//            if (data == null || data.length < 2)
//                continue;
//
//            double rho1 = data[0];
//            double theta1 = data[1]; // 0 -> pi
//            double thetaDeg = 180.0 * theta1 / Math.PI; // 0 -> 180
//            double cosTheta = Math.cos(theta1);
//            double sinTheta = Math.sin(theta1);
//            double x0 = cosTheta * rho1;
//            double y0 = sinTheta * rho1;
//
//            // angle thresholding
//            if (d < thetaDeg && thetaDeg < (180 - d))
//                continue;
//
//            // position thresholding
//            if (x0 < f * width || x0 > (1 - f) * width)
//                continue;
//
////            Log.i("HOUGH theta(deg)", "lineIdx[" + i + "] theta:" + thetaDeg);
//
//            Point pt1 = new Point(x0 + 1000 * (-sinTheta), y0 + 1000 * cosTheta);
//            Point pt2 = new Point(x0 - 1000 * (-sinTheta), y0 - 1000 * cosTheta);
//
//            if (x0 < 0.5 * width)
//                windowedAdd(this.leftLines, new Pair<>(pt1, pt2));
//            else
//                windowedAdd(this.rightLines, new Pair<>(pt1, pt2));
//        }

//        return matEdgeBin;
//        return drawLines();
}
