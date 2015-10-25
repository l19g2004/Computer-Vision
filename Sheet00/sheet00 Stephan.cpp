#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void readImage(const char* file, Mat& result) {
    //TODO: implement your solution here
    //read in the image of the parameter  file and save 
    //the matrix in result
    result = imread(file, 1);
}

void display(const char* windowTitle, Mat& img) {
    //TODO: implement your solution here
    //show the image in a window
    imshow(windowTitle, img);
    //hold the window up to a key event
    waitKey(0);
}

void convertToGrayImg(Mat& img, Mat& result) {
    //TODO: implement your solution here
    //convert the color image img to a gray image
    //and save it in result
    cvtColor(img, result, CV_BGR2GRAY);
}

void subtractIntensityImage(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    //gernerate three new matrices 
    Mat channel[3];
    //split the image into his three channels rgb
    split(bgrImg, channel);
    //subtract of each channel the product of intensity image
    // and the factor 0.5
    subtract(channel[0], 0.5 * grayImg, result);
    subtract(channel[1], 0.5 * grayImg, result);
    subtract(channel[2], 0.5 * grayImg, result);
}

void pixelwiseSubtraction(Mat& bgrImg, Mat& grayImg, Mat& result) {
    //TODO: implement your solution here
    result = bgrImg;
}

void extractPatch(Mat& img, Mat& result) {
    //TODO: implement your solution here
    //get the start coordinates to begin extracting the patch
    int startX = img.size().width / 2 - 8;
    int startY = img.size().height / 2 - 8;
    //initialize the result matrix with zeros 
    result = Mat::zeros(16,16, img.type());
    //iterate over the 16x16 field and save every pixel
    //in the result matrix
    for (int i = startX; i < startX + 16; i++) {
        for (int j = startY; j < startY + 16; j++) {
            result.at<Vec3b>(i - startX, j - startY) = img.at<Vec3b>(i, j);
        }
    }
}

void copyPatchToRandomLocation(Mat& img, Mat& patch, Mat& result) {
    //TODO: implement your solution here
    //initialize a random object 
    RNG rng;
    //copy the original image matrix to the result matrix
    img.copyTo(result);
    //get a random position in the original image to place the patch
    int x = rng.uniform(0, result.size().width - 16);
    int y = rng.uniform(0, result.size().height - 16);
    //copy every pixel of the patch image to the specific position 
    //in the original image matrix 
    for (int i = x; i < x + 16; i++) {
        for (int j = y; j < y + 16; j++) {
            result.at<Vec3b>(x, y) = patch.at<Vec3b>(i - x, j - x);
            
        }
    }
}

void drawRandomRectanglesAndEllipses(Mat& img) {
    //TODO: implement your solution here
    //get the size of the image
    int width = img.size().width;
    int height = img.size().height;
    //generate ten rectangles and eclipse with a random position and size
    //in the range of the image size and place them randomly
    //in the image
    for (int i = 0; i < 10; i++) {
        rectangle(img, 
                    Point(rand() % width, rand() % height), 
                    Point(rand() % width, rand() % height),
                    Scalar(255, 0, 0));
        //eclipse();
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
