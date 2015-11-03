#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

void part1();
void part2();
void part3();
void part4();
void part5();
void drawArrow(cv::Mat &image, cv::Point p, cv::Scalar color, double scaleMagnitude, double scaleArrowHead, double magnitube, double orientationDegrees);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char* argv[])
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.

    part1();
    //part2();
    //part3();
    //part4();
    //part5();

    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    //////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                            std::endl;

}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // read the image file
    cv::Mat im_Traffic_BGR = cv::imread("../images/traffic.jpg", cv::IMREAD_COLOR);
    // gray version of bonn.png
    cv::Mat                      im_Traffic_Gray;
    cv::cvtColor(im_Traffic_BGR, im_Traffic_Gray, CV_BGR2GRAY);

    // construct Gaussian pyramids
    std::vector<cv::Mat>   gpyr;    // this will hold the Gaussian Pyramid created with OpenCV
    std::vector<cv::Mat> myGpyr;    // this will hold the Gaussian Pyramid created with your custom way

    std::vector<cv::Mat> myLapyr;    // this will hold the Laplacian Pyramid


    imshow("Task1: traffic.jpg", im_Traffic_BGR);

    // Perform the computations asked in the exercise sheet and show them using **std::cout**

    // Show every layer of the pyramid
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    Mat tmp;

    // Gaussian Pyramid created with OpenCV
    im_Traffic_BGR.copyTo(tmp);
    while( (tmp.cols*0.5 > 10) && (tmp.rows*0.5 > 10) ){
        cv::pyrDown( tmp, tmp, Size( (tmp.cols+1)*0.5, (tmp.rows+1)*0.5 ) );
        gpyr.push_back(tmp);
    }

    // Gaussian Pyramid created with your custom way
    im_Traffic_BGR.copyTo(tmp);    
    while( (tmp.cols*0.5 > 10) && (tmp.rows*0.5 > 10) ){
        GaussianBlur(tmp, tmp, Size(5,5), 1.5);
        resize(tmp, tmp, Size((tmp.cols+1)*0.5, (tmp.rows+1)*0.5));
        myGpyr.push_back(tmp);
    }



    if(gpyr.size() == myGpyr.size()){
        for(std::vector<Mat>::size_type i = 0; i != gpyr.size(); i++) {

            //equalizeHist( src, dst );
            std::cout << "<---------------------- normalization here " << std::endl;

            myLapyr.push_back(gpyr[i] - myGpyr[i]);

            imshow("Task1: Laplacian Pyramid", myLapyr[i]);

            imshow("Task1: Gaussian Pyramid (OpenCV)", gpyr[i]);
            std::cout << "Image size (OpenCV): " << gpyr[i].rows << ", " << gpyr[i].cols << std::endl;

            imshow("Task1: Gaussian Pyramid (custom Way)", myGpyr[i]);
            std::cout << "Image size (custom Way): " << myGpyr[i].rows << ", " << myGpyr[i].cols << std::endl;

            cv::waitKey(0);
        }
    } else {
        std::cout << "OpenCV Pyramid and custom way are different" << std::endl;
    }


    std::cout << "<---------------------- difference here " << std::endl;



    // For the laplacian pyramid you should define your own container.
    // If needed perform normalization of the image to be displayed



   // cv::waitKey(0); // waits until the user presses a button
    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    // apple and orange are CV_32FC3
    cv::Mat im_Apple, im_Orange;
    cv::imread("./images/apple.jpg",  cv::IMREAD_COLOR).convertTo(im_Apple,  CV_32FC3, (1./255.));
    cv::imread("./images/orange.jpg", cv::IMREAD_COLOR).convertTo(im_Orange, CV_32FC3, (1./255.));
    cv::imshow("orange", im_Orange);
    cv::imshow("apple",  im_Apple );
    std::cout << "\n" << "Input images" << "   \t\t\t\t\t\t\t" << "Press any key..." << std::endl;
    cv::waitKey(0);

    // Perform the blending using a Laplacian Pyramid

    // Show the blending results @ several layers
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part3()
{
    std::cout <<                                                            std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 3    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    cv::Mat           im_Traffic_BGR = cv::imread("./images/traffic.jpg"); // traffic.jpg // circles.png
    // Blur to denoise
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the computations asked in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    // Use the function **drawArrow** provided at the end of this file in order to
    // draw Vectors showing the Gradient Magnitude and Orientation
    // (to avoid clutter, draw every 10nth gradient,
    // only if the magnitude is above a threshold)

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part4()
{
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 4    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;

    cv::Mat im_Traffic_BGR = cv::imread("./images/traffic.jpg");
    // BGR to Gray
    cv::Mat                       im_Traffic_Gray;
    cv::cvtColor( im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );

    // Perform edge detection as described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part5()
{
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 5    ///////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout << "/////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                            std::endl;

    // Read image, Blur to denoise
    cv::Mat                           im_Traffic_BGR = cv::imread("./images/traffic.jpg");
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR,  cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );

    // Read Template
    cv::Mat im_Sign_BGR = cv::imread("./images/sign.png");
    cv::Mat im_Sign_Gray;
    // BGR to Gray
    cv::cvtColor( im_Sign_BGR, im_Sign_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    cv::destroyAllWindows();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// Use this function for visualizations in part3

void drawArrow(cv::Mat &image, cv::Point p, cv::Scalar color, double scaleMagnitude, double scaleArrowHead, double magnitube, double orientationDegrees)
{
    int arrowHeadAngleCoeff = 10;

    magnitube *= scaleMagnitude;

    double theta = orientationDegrees * M_PI / 180; // rad
    cv::Point q;
    q.x = p.x + magnitube * cos(theta);
    q.y = p.y + magnitube * sin(theta);

    //Draw the principle line
    cv::line(image, p, q, color);
    //compute the angle alpha
    double angle = atan2((double)p.y-q.y, (double)p.x-q.x);
    //compute the coordinates of the first segment
    p.x = (int) ( q.x + static_cast<int>(  round( scaleArrowHead * cos(angle + M_PI/arrowHeadAngleCoeff)) )  );
    p.y = (int) ( q.y + static_cast<int>(  round( scaleArrowHead * sin(angle + M_PI/arrowHeadAngleCoeff)) )  );
    //Draw the first segment
    cv::line(image, p, q, color);
    //compute the coordinates of the second segment
    p.x = (int) ( q.x + static_cast<int>(  round( scaleArrowHead * cos(angle - M_PI/arrowHeadAngleCoeff)) )  );
    p.y = (int) ( q.y + static_cast<int>(  round( scaleArrowHead * sin(angle - M_PI/arrowHeadAngleCoeff)) )  );
    //Draw the second segment
    cv::line(image, p, q, color);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
