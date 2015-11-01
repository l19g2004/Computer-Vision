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


            meanIntegrals +=  ((double)( rectIntegral.at<int>(r.br())                         // A
                                - rectIntegral.at<int>(Point(r.tl().x, r.tl().y + r.height))  // B
                                - rectIntegral.at<int>(Point(r.tl().x + r.width, r.tl().y))   // C
                                + rectIntegral.at<int>(r.tl())                                // D
                                )/(double)(r.height*r.width));

        }
    }
    // normalize
    meanIntegrals /= (double)(sizeof(rects)/(double)sizeof(*rects));

    tock = getTickCount();
    cout << "computing an integral image gives " << meanIntegrals << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

	
	//====(d)==== integral image method - custom implementation====
	//TODO: implement your solution here
	// ...
        
    tick = getTickCount();
    
    //initalize a new matrix for computing the integral matrix
    //Mat integralImage = Mat::zeros(bonn_gray.size().height, bonn_gray.size().width, bonn_gray.type());
    Mat integralImage;
    //iterate through the original gray image
    /*
    for (int i = 0; i < bonn_gray.size().width -1; i++) {
        for (int j = 0; j < bonn_gray.size().height -1; j++) {
            //sum up all four pixel (self, tl, t, l)
            //catch the border pixel if a pixel is outside the image
            //integralImage.at<uchar>(Point(i, j)) += bonn_gray.at<uchar>(Point (i, j));
            if (i != 0 && j != 0) {
                integralImage.at<uchar>(Point(i, j)) -= bonn_gray.at<uchar>(Point(i - 1, j - 1));
            }            
            if (i != 0) {
                integralImage.at<uchar>(Point(i, j)) += bonn_gray.at<uchar>(Point(i - 1, j));
            }
            if (j != 0) {
                integralImage.at<uchar>(Point(i, j)) += bonn_gray.at<uchar>(Point(i, j - 1));
            }
        }        
    }
    */

    integralImage = Mat::zeros(bonn_gray.rows+1, bonn_gray.cols+1, bonn_gray.type());
   // integralImage.row(0) =  Mat::zeros(integralImage.rows, 1, integralImage.type());
   //integralImage.col(0) =  Mat::zeros(1, integralImage.cols, integralImage.type());
    for (int i=0; i < integralImage.rows; ++i) {
        for (int j=0;  j < integralImage.cols; ++j) {
            if (i != 0 && j != 0)
                integralImage.at<uchar>(i,j) = bonn_gray.at<uchar>(i-1, j-1);

            if (i == 0)
                integralImage.at<uchar>(i,j) = 0;
            else
                integralImage.at<uchar>(i,j) += integralImage.at<uchar>(i-1, j);
            if (j == 0)
                integralImage.at<uchar>(i,j) = 0;
            else
                integralImage.at<uchar>(i,j) += integralImage.at<uchar>(i  , j-1);
            if (i != 0 && j != 0)
                integralImage.at<uchar>(i,j) -= integralImage.at<uchar>(i-1, j-1);

        }
    }

    cout << "bonn_gray: "<< "("<< bonn_gray.rows << ", " << bonn_gray.cols <<")"<< endl << bonn_gray(Rect(0,0,5,5)) << endl;
    cout << "rectIntegral: "<< "("<< rectIntegral.rows << ", " << rectIntegral.cols <<")"<< endl << rectIntegral(Rect(0,0,5,5)) << endl;
    cout << "integralImage: "<< "("<< integralImage.rows << ", " << integralImage.cols <<")"<< endl << integralImage(Rect(0,0,5,5)) << endl;


    
    //The same like task c)
    
    //double meanIntegrals;
    for (size_t n = 0; n < 10; ++n) {
        meanIntegrals = 0.0;
        for (auto& r : rects) {
            //cout << "Point " << r.tl() << " - " << r.br() << " : " << r.width << " , "<< r.height << endl;


            //      .       .
            //      .       .
            // .....D-------C
            //      |       |
            // .....B-------A


            meanIntegrals +=  ((double)( integralImage.at<int>(r.br())                         // A
                                - integralImage.at<int>(Point(r.tl().x, r.tl().y + r.height))  // B
                                - integralImage.at<int>(Point(r.tl().x + r.width, r.tl().y))   // C
                                + integralImage.at<int>(r.tl())                                // D
                                )/(double)(r.height*r.width));

        }
    }
    // normalize
    meanIntegrals /= (double)(sizeof(rects)/(double)sizeof(*rects));

    tock = getTickCount();
    cout << "computing an integral image gives " << meanIntegrals << " computed in " << (tock-tick)/getTickFrequency() << " seconds." << endl;

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
    imshow("original", bonn_gray);
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
            // Fill kernel2D matrix with values of a gaussian
            kernel2D(y,x) = (1/(2*M_PI*sigma*sigma))*exp(-((dx*dx+dy*dy)/(2*sigma*sigma)));
        }
	}
    //cout << "kernel2D: "<< "("<< kernel2D.rows << ", " << kernel2D.cols <<")"<< kernel2D << endl;

	kernel2D *= 1. / sum(kernel2D)[0];

	// TODO: implement your solution here - use "filter2D"
    filter2D(bonn_gray, img_f2D, bonn_gray.depth(), kernel2D);

	tock = getTickCount();
	cout << "OpenCV filter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

    //  ====(c)==== 2D Filtering - using opencv "sepFilter2D()" ====
	tick = getTickCount();
	// TODO: implement your solution here

    Matx<float, 2*k_width+1, 1>kernelX = kernel2D.col(0);

    Matx<float, 1, 2*k_width+1>kernelY = kernel2D.row(0);

    
    sepFilter2D(bonn_gray, img_sepF2D, bonn_gray.depth(), kernelX.t(), kernelY.t());

    tock = getTickCount();
	cout << "OpenCV sepFilter2D() method takes " << (tock-tick)/getTickFrequency() << " seconds." << endl;

    // Show result images
    imshow("Task4: grayimage", bonn_gray);
    imshow("Task4: (a) GaussianBlur", img_gb);
    imshow("Task4: (b) filter2D", img_f2D);
    imshow("Task4: (c) sepFilter2D", img_sepF2D);

	// compare blurring methods
    // Compute absolute differences between pixel intensities of (a), (b) and (c) using "absdiff"
    Mat diff1;
    Mat diff2;
    Mat diff3;

    absdiff(img_gb, img_f2D, diff1);
    absdiff(img_gb, img_sepF2D, diff2);
    absdiff(img_f2D, img_sepF2D, diff3);

    // Find the maximum pixel error using "minMaxLoc"
    double minVal2, maxVal2;
    minMaxLoc(diff1, &minVal2, &maxVal2);
    cout << "maximum pixel error (img_gb vs. img_f2D): " << maxVal2 << endl;
    minMaxLoc(diff2, &minVal2, &maxVal2);
    cout << "maximum pixel error (img_gb vs. img_sepF2D): " << maxVal2 << endl;
    minMaxLoc(diff3, &minVal2, &maxVal2);
    cout << "maximum pixel error (img_f2D vs. img_sepF2D): " << maxVal2 << endl;


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
    Mat imagefilter2DKernel1;
    Mat imagefilter2DKernel2;

    filter2D(bonn_gray, imagefilter2DKernel1, bonn_gray.depth() , kernel1);
    filter2D(bonn_gray, imagefilter2DKernel2, bonn_gray.depth() , kernel2);

    imshow("Task8: filter2D kernel 1", imagefilter2DKernel1);
    imshow("Task8: filter2D kernel 2", imagefilter2DKernel2);

    //  ====(b)==================================================================	
    Mat imageSVDKernel1, w1, u1, vt1;
    Mat imageSVDKernel2;



    bonn_gray.convertTo(imageSVDKernel1, CV_32F, 1.0/255);
    SVD::compute(imageSVDKernel1, w1, u1, vt1);
    //Here is an error but we do not know why
    //sepFilter2D(bonn_gray, imageSVDKernel1, bonn_gray.depth(), u1, vt1);
    
    bonn_gray.copyTo(imageSVDKernel1);
    bonn_gray.copyTo(imageSVDKernel2);
    
    
    imshow("Task8: SVD kernel 1", imageSVDKernel1);
    imshow("Task8: SVD kernel 2", imageSVDKernel2);

    //  ====(c)==================================================================	
    // compute the absolute pixel-wise difference between the
    // results, and print the maximum pixel error.
    Mat diff8;
    double minVal8, maxVal8;
    absdiff(imagefilter2DKernel1, imageSVDKernel1, diff8);
    minMaxLoc(diff8, &minVal8, &maxVal8);
    cout << "maximum pixel error (kernel 1): " << maxVal8 << endl;
    absdiff(imagefilter2DKernel2, imageSVDKernel2, diff8);
    minMaxLoc(diff8, &minVal8, &maxVal8);
    cout << "maximum pixel error (kernel 2): " << maxVal8 << endl;

	
    waitKey(0);
	
    cout << "Program finished successfully" << endl;
    destroyAllWindows();
    return 0;
}
