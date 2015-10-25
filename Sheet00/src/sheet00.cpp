#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void readImage(const char* file, Mat& result) {
    //TODO: implement your solution here
    cout << "(a) read image... " << endl;
    result = imread(file);
    if(!result.data ) // Check for invalid input
    {
        cout <<  "Could not open or find the image: " << file << std::endl ;
    }
}

void display(const char* windowTitle, Mat& img) {
    //TODO: implement your solution here
    cout << "(a) show image... " << endl;
    imshow(windowTitle, img);

    //hold the window up to a key event
    waitKey(0);
}

void convertToGrayImg(Mat& img, Mat& result) {
    //TODO: implement your solution here
    cout << "(b) convert image to intensity image... " << endl;
    cvtColor(img, result, CV_BGR2GRAY );
}

void subtractIntensityImage(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    cout << "(c) subtract the intensity and from each color channel " << endl;

    // convert result to a grayscaled image with 3 channels
    cvtColor(grayImg, result, CV_GRAY2BGR);

    // bgrImg - 0.5*grayImg
    subtract(bgrImg, 0.5*result, result);

    // max(bgrImg − 0.5*grayImg, 0)
    cv::max(result, 0.0);

}

void pixelwiseSubtraction(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    cout << "(d) subtract the intensity and from each color channel (pixelwise operations)" << endl;

    // convert result to a grayscaled image with 3 channels
    cvtColor(grayImg, result, CV_GRAY2BGR);

    int nRows = result.rows;
    int nCols = result.cols * result.channels();

    if (bgrImg.isContinuous() && result.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* result_j;
    uchar* bgrImg_j;

    for( i = 0; i < nRows; ++i)
    {
        result_j = result.ptr<uchar>(i);
        bgrImg_j = bgrImg.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            result_j[j] = std::max( (bgrImg_j[j] - 0.5*result_j[j]), 0.);
        }
    }

}

void extractPatch(Mat& img, Mat& result) {
    //TODO: implement your solution here
    cout << "(e) random patch" << endl;

    //get the start coordinates to begin extracting the patch
    int startX = (img.size().width / 2) - 8;
    int startY = (img.size().height / 2) - 8;


    img(Rect(startX,startY,16,16)).copyTo(result);
}

void copyPatchToRandomLocation(Mat& img, Mat& patch, Mat& result) {
    //TODO: implement your solution here

    cout << "(e) afterRandomCopying" << endl;

    //initialize a random object
    RNG rng;

    //copy the original image matrix to the result matrix
    img.copyTo(result);

    //get a random position in the original image to place the patch
    int rand_x = rng.uniform(0, (result.size().width - 16));
    int rand_y = rng.uniform(0, (result.size().height - 16));

    // copy patch to the result image at the uniform random position
    patch.copyTo(result(Rect(rand_x, rand_y, patch.cols, patch.rows)));

}

void drawRandomRectanglesAndEllipses(Mat& img) {
    //TODO: implement your solution here

    cout << "(f) drawing random rectanges and ellipses on the image" << endl;

    //get the size of the image
    int width = img.size().width;
    int height = img.size().height;

    //initialize a random object
    RNG rng;

    //generate ten rectangles and eclipse with a random position and size
    //in the range of the image size and place them randomly
    //in the image
    for (int i = 0; i < 10; i++) {

        // draw rectangles
        rectangle(  img,
                    Point(rng.uniform(0, width), rng.uniform(0, height)),
                    Point(rng.uniform(0, width), rng.uniform(0, height)),
                    Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))
                 );

        // draw ellipses
        ellipse(    img,
                    Point(rng.uniform(0, width), rng.uniform(0, height) ),
                    Size( rng.uniform(0, width), rng.uniform(0, height) ),
                    rng.uniform(0, 360),
                    rng.uniform(0, 360),
                    rng.uniform(0, 360),
                    Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))
                );


    }

}

int main(int argc, char* argv[])
{
    // check input arguments
    if(argc!=2) {
        cout << "usage: " << argv[0] << " <path_to_image>" << endl;
        return -1;
    }

    // read and display the input image
    Mat bgrImg;
    readImage(argv[1], bgrImg);
    display("(a) inputImage", bgrImg);

    // (b) convert bgrImg to grayImg
    Mat grayImg;
    convertToGrayImg(bgrImg, grayImg);
    display("(b) grayImage", grayImg);

    // (c) bgrImg - 0.5*grayImg
    Mat imgC;
    subtractIntensityImage(bgrImg, grayImg, imgC);
    display("(c) subtractedImage", imgC);

    // (d) pixelwise operations
    Mat imgD;
    pixelwiseSubtraction(bgrImg, grayImg, imgD);
    display("(d) subtractedImagePixelwise", imgD);

    // (e) copy 16x16 patch from center to a random location
    Mat patch;
    extractPatch(bgrImg, patch);
    display("(e) random patch", patch);
    Mat imgE;
    copyPatchToRandomLocation(bgrImg, patch, imgE);
    display("(e) afterRandomCopying", imgE);


    // (f) drawing random rectanges on the image
    drawRandomRectanglesAndEllipses(bgrImg);
    display("(f) random elements", bgrImg);



    return 0;
}
