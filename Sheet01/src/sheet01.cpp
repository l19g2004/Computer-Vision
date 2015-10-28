#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    
//	==================== Load image =========================================
  	// Read the image file - bonn.png is of type CV_8UC3
	const Mat bonn = imread(argv[1], IMREAD_COLOR);

	// Create gray version of bonn.png
	Mat bonn_gray;
	cvtColor(bonn, bonn_gray, CV_BGR2GRAY);
//	=========================================================================
//	==================== Solution of task 1 =================================
//	=========================================================================
	cout << "Task 1:" << endl;
	//====(a)==== draw random rectangles ====
	
	// Create a copy of the input image to draw the rectangles into
	Mat img_task1;
	bonn_gray.copyTo(img_task1);
	
	// Create a random number generator
	RNG rng(getTickCount());
	Rect rects[100];
	for (auto& r : rects) {
		// Random coordinates of the rectangle
		const int x1 = rng.uniform(0, bonn_gray.cols);
		const int x2 = rng.uniform(0, bonn_gray.cols);
		const int y1 = rng.uniform(0, bonn_gray.rows);
		const int y2 = rng.uniform(0, bonn_gray.rows);
		
        // set points to the top-left corner
        r = Rect(Point(x1, y1), Point(x2,y2));

        // Draw the rectangle into the image "img_task1"
        rectangle(img_task1, r.tl(), r.br(), Scalar(255,255,255));

	}
    // Display the image "img_task1" with the rectangles
    namedWindow("Task1: (a) draw random rectangles", WINDOW_AUTOSIZE);
    imshow("Task1: (a) draw random rectangles", img_task1);

	//====(b)==== summing up method ====
	// Variables for time measurements
	int64_t tick, tock;
	double sumOfPixels;
    double sumOfPixelsPerRect;
	// Measure the time
	tick = getTickCount();
	
	// Repeat 10 times
	for (size_t n = 0; n < 10; ++n) {
		sumOfPixels = 0.0;
        sumOfPixelsPerRect = 0.0;
		for (auto& r : rects) {
			for (int y = r.y; y < r.y + r.height; ++y) {
				for (int x = r.x; x < r.x+r.width; ++x) {
				  	//TODO: Sum up all pixels at "bonn_gray.at<uchar>(y,x)" inside the rectangle and store it in "sumOfPixels" 
					// ...
                    sumOfPixels += bonn_gray.at<uchar>(y,x);
                    sumOfPixelsPerRect++;
				}
			} 
		}
	}
    // normalize
    sumOfPixels /= sumOfPixelsPerRect;

	tock = getTickCount();
	cout << "Summing up each pixel gives " << sumOfPixels << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	// //====(c)==== integral image method - using OpenCV function "integral" ==== 

    // Measure the time
    tick = getTickCount();

    Mat rectIntegral;
    integral(bonn_gray, rectIntegral, bonn_gray.depth());
    double meanIntegrals;
    for (size_t n = 0; n < 10; ++n) {
        meanIntegrals = 0.0;
        for (auto& r : rects) {
            //cout << "Point " << r.tl() << " - " << r.br() << " : " << r.width << " , "<< r.height << endl;


            //      .       .
            //      .       .
            // .....D-------C
            //      |       |
            // .....B-------A
/*
            cout << "Values: " << rectIntegral.at<uchar>(r.br());
            cout << "        " << rectIntegral.at<uchar>(Point(r.tl().x,r.tl().y + r.height));
            cout << "        " << rectIntegral.at<uchar>(Point(r.tl().x + r.width, r.tl().y));
            cout << "        " << rectIntegral.at<uchar>(r.tl());
            cout << endl;
*/

            meanIntegrals +=  0.25*( rectIntegral.at<uchar>(r.br())                           // A
                                - rectIntegral.at<uchar>(Point(r.tl().x,r.tl().y + r.height)) // B
                                - rectIntegral.at<uchar>(Point(r.tl().x + r.width, r.tl().y)) // C
                                + rectIntegral.at<uchar>(r.tl())                              // D
                                );

//            cout << "meanIntegrals " << meanIntegrals << endl;


        }
    }
    // normalize
    meanIntegrals /= (sizeof(rects)/sizeof(*rects));

    cout << "<-------- fix is needed" << endl;


    tock = getTickCount();
    cout << "computing an integral image gives " << meanIntegrals << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	
	//====(d)==== integral image method - custom implementation====
	//TODO: implement your solution here
	// ...

    waitKey(0); // waits until the user presses a button and then continues with task 2 -> uncomment this
    destroyAllWindows(); // closes all open windows -> uncomment this
	
//	=========================================================================	
//	==================== Solution of task 2 =================================
//	=========================================================================
	cout << "Task 2:" << endl;
	
	
	//====(a)==== Histogram equalization - using opencv "equalizeHist" ====
	Mat ocvHistEqualization;
    equalizeHist(bonn_gray, ocvHistEqualization);

	//====(b)==== Histogram equalization - custom implementation ====
	Mat myHistEqualization(bonn_gray.size(), bonn_gray.type()); 

	// Count the frequency of intensities
	Vec<double,256> hist(0.0);
	for (int y=0; y < bonn_gray.rows; ++y) {
		for (int x=0;  x < bonn_gray.cols; ++x) {
			++hist[bonn_gray.at<uchar>(y,x)];
		}
	}
	// Normalize vector of new intensity values to sum up to a sum of 255
	hist *= 255. / sum(hist)[0];
	
	// Compute integral histogram - representing the new intensity values
	for (size_t i=1; i < 256; ++i) {
		hist[i] += hist[i-1];
	}
	
    // Fill the matrix "myHistEqualization" -> replace old intensity values with new intensities taken from the integral histogram
    for (int y=0; y < bonn_gray.rows; ++y) {
        for (int x=0;  x < bonn_gray.cols; ++x) {
            myHistEqualization.at<uchar>(y,x) = hist[bonn_gray.at<uchar>(y,x)];
        }
    }
	
    // Show the results of (a) and (b)
    namedWindow("Task2: (a) with equalizeHist", WINDOW_AUTOSIZE);
    imshow("Task2: (a) with equalizeHist", ocvHistEqualization);
    namedWindow("Task2: (b) without equalizeHist", WINDOW_AUTOSIZE);
    imshow("Task2: (b) without equalizeHist", ocvHistEqualization);
	
	//====(c)==== Difference between openCV implementation and custom implementation ====
    // Compute absolute differences between pixel intensities of (a) and (b) using "absdiff"

    Mat diff;
    absdiff(myHistEqualization, ocvHistEqualization, diff);
    double minVal, maxVal;
    minMaxLoc(diff, &minVal, &maxVal);
    cout << "maximum pixel error: " << maxVal << endl;

    waitKey(0);
    destroyAllWindows();
	
//	=========================================================================	
//	==================== Solution of task 4 =================================
//	=========================================================================
	cout << "Task 4:" << endl;
	Mat img_gb;
	Mat img_f2D;
	Mat img_sepF2D;
	const double sigma = 2. * sqrt(2.);

    //  ====(a)==== 2D Filtering - using opencv "GaussianBlur()" ====
	tick = getTickCount();

    GaussianBlur(bonn_gray, img_gb, Size(0,0), sigma);

	tock = getTickCount();
	cout << "OpenCV GaussianBlur() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;




    //  ====(b)==== 2D Filtering - using opencv "filter2D()" ==== 
	// Compute gaussian kernel manually
	const int k_width = 3.5 * sigma;
	tick = getTickCount();
	Matx<float, 2*k_width+1, 2*k_width+1> kernel2D;
	for (int y = 0; y < kernel2D.rows; ++y) {
		const int dy = abs(k_width - y);
		for (int x = 0; x < kernel2D.cols; ++x) {
            const int dx = abs(k_width - x);
			//TODO: Fill kernel2D matrix with values of a gaussian
            kernel2D(y,x) = (1/(2*M_PI*sigma*sigma))*exp(-((x*x+y*y)/(2*sigma*sigma)));
        }
	}
    //cout << "kernel2D: "<< kernel2D << endl;

	kernel2D *= 1. / sum(kernel2D)[0];

	// TODO: implement your solution here - use "filter2D"
    filter2D(bonn_gray, img_f2D, bonn_gray.depth(), kernel2D);

	tock = getTickCount();
	cout << "OpenCV filter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

    //  ====(c)==== 2D Filtering - using opencv "sepFilter2D()" ====
	tick = getTickCount();
	// TODO: implement your solution here
  //  sepFilter2D(bonn_gray, img_sepF2D, bonn_gray.depth(), kernel2D, kernel2D);
        filter2D(bonn_gray, img_sepF2D, bonn_gray.depth(), kernel2D);

    cout << "<-------- fix is needed" << endl;

    tock = getTickCount();
	cout << "OpenCV sepFilter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

    // Show result images
    namedWindow("Task4: (a) grayimage", WINDOW_AUTOSIZE);
    imshow("Task4: (a) grayimage", bonn_gray);

    namedWindow("Task4: (a) GaussianBlur", WINDOW_AUTOSIZE);
    imshow("Task4: (a) GaussianBlur", img_gb);
    namedWindow("Task4: (b) filter2D", WINDOW_AUTOSIZE);
    imshow("Task4: (b) filter2D", img_f2D);
    namedWindow("Task4: (c) sepFilter2D", WINDOW_AUTOSIZE);
    imshow("Task4: (c) sepFilter2D", img_sepF2D);

	// compare blurring methods
    // Compute absolute differences between pixel intensities of (a), (b) and (c) using "absdiff"
    Mat diff0;
    Mat diff1;
    Mat diff2;
    Mat diff3;

    absdiff(img_gb, img_f2D, diff1);
    absdiff(img_gb, img_sepF2D, diff2);
    absdiff(img_f2D, img_sepF2D, diff3);

    diff0 = 0.333*(diff1 + diff2 + diff3);

    // TODO: Find the maximum pixel error using "minMaxLoc"
    double minVal2, maxVal2;
    minMaxLoc(diff0, &minVal2, &maxVal2);
    cout << "maximum pixel error: " << maxVal2 << endl;


    waitKey(0);
    destroyAllWindows();
	
//	=========================================================================	
//	==================== Solution of task 6 =================================
//	=========================================================================
	cout << "Task 6:" << endl;
    //  ====(a)==================================================================
    // twice with a Gaussian kernel with σ = 2
    Mat twiceGaussianKernel;
    GaussianBlur(bonn_gray, twiceGaussianKernel, Size(0,0), 2);
    GaussianBlur(twiceGaussianKernel, twiceGaussianKernel, Size(0,0), 2);

    //  ====(b)==================================================================	
    // once with a Gaussian kernel with σ = 2*sqrt(2)
    Mat onceGaussianKernel;
    GaussianBlur(bonn_gray, onceGaussianKernel, Size(0,0), (2*sqrt(2)));
	


    //compute the absolute pixel-wise difference between the
    // results, and print the maximum pixel error.
    Mat diff6;
    absdiff(twiceGaussianKernel, onceGaussianKernel, diff6);
    double minVal6, maxVal6;
    minMaxLoc(diff6, &minVal6, &maxVal6);
    cout << "maximum pixel error: " << maxVal6 << endl;

	
    // Show result images
    namedWindow("Task6: (a) twiceGaussian", WINDOW_AUTOSIZE);
    imshow("Task6: (a) twiceGaussian", twiceGaussianKernel);
    namedWindow("Task6: (b) onceGaussian", WINDOW_AUTOSIZE);
    imshow("Task6: (b) onceGaussian", onceGaussianKernel);

    waitKey(0);
    destroyAllWindows();

//	=========================================================================	
//	==================== Solution of task 7 =================================
//	=========================================================================	
	cout << "Task 7:" << endl;
	// Create an image with salt and pepper noise
	Mat bonn_salt_pepper(bonn_gray.size(), bonn_gray.type());
	randu(bonn_salt_pepper, 0, 100); // Generates an array of random numbers
	for (int y = 0; y < bonn_gray.rows; ++y) {
		for (int x=0; x < bonn_gray.cols; ++x) {
			uchar& pix = bonn_salt_pepper.at<uchar>(y,x);
			if (pix < 15) {
              // Set set pixel "pix" to black
                bonn_salt_pepper.at<uchar>(y,x) = 0;
			}else if (pix >= 85) { 
              // Set set pixel "pix" to white
                bonn_salt_pepper.at<uchar>(y,x) = 255;
			}else { 
              // Set set pixel "pix" to its corresponding intensity value in bonn_gray
                bonn_salt_pepper.at<uchar>(y,x) = bonn_gray.at<uchar>(y,x);
			}
		}
	}

    imshow("Task7: bonn.png with salt and pepper", bonn_salt_pepper);

    //  ====(a)==================================================================
    // Gaussian kernel
    Mat imageGaussianFilter;
    GaussianBlur(bonn_salt_pepper, imageGaussianFilter, Size(0,0), 2.2);

    //  ====(b)==================================================================	
    // Median filter medianBlur
    Mat imageMedianBlurFilter;
    medianBlur(bonn_salt_pepper, imageMedianBlurFilter, 5);

    //  ====(c)==================================================================	
    // Bilateral filter
    Mat imageBilateralFilter;
    bilateralFilter(bonn_salt_pepper, imageBilateralFilter, 30, 250, 250);
	
    imshow("Task7: Gaussian kernel", imageGaussianFilter);
    imshow("Task7: Median filter", imageMedianBlurFilter);
    imshow("Task7: Bilateral", imageBilateralFilter);


    waitKey(0);
    destroyAllWindows();
//	=========================================================================	
//	==================== Solution of task 8 =================================
//	=========================================================================	
	cout << "Task 8:" << endl;
	// Declare Kernels
	Mat kernel1 = (Mat_<float>(3,3) << 0.0113, 0.0838, 0.0113, 0.0838, 0.6193, 0.0838, 0.0113, 0.0838, 0.0113);	
	Mat kernel2 = (Mat_<float>(3,3) << -0.8984, 0.1472, 1.1410, -1.9075, 0.1566, 2.1359, -0.8659, 0.0573, 1.0337);
	
    //  ====(a)==================================================================
	// TODO: implement your solution here
	// ...
    //  ====(b)==================================================================	
	// TODO: implement your solution here
	// ...		
    //  ====(c)==================================================================	
	// TODO: implement your solution here
	// ...			
	
	
	
	
	
  cout << "Program finished successfully" << endl;
  destroyAllWindows(); 
  return 0;
}
