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
    part2();
    part3();
    part4();
    part5();

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
        GaussianBlur(tmp, tmp, Size(3,3), 0.55);
        resize(tmp, tmp, Size((tmp.cols+1)*0.5, (tmp.rows+1)*0.5));
        myGpyr.push_back(tmp);
    }

    Mat diff;
    double minVal, maxVal;


    if(gpyr.size() == myGpyr.size()){
        for(std::vector<Mat>::size_type i = 0; i != gpyr.size(); i++) {

            myLapyr.push_back(gpyr[i] - myGpyr[i]);

            imshow("Task1: Laplacian Pyramid", myLapyr[i]);

            imshow("Task1: Gaussian Pyramid (OpenCV)", gpyr[i]);
            std::cout << "Image size (OpenCV): " << gpyr[i].rows << ", " << gpyr[i].cols << std::endl;

            imshow("Task1: Gaussian Pyramid (custom Way)", myGpyr[i]);
            std::cout << "Image size (custom Way): " << myGpyr[i].rows << ", " << myGpyr[i].cols << std::endl;

            diff.zeros(diff.rows,diff.cols,diff.type());
            absdiff(gpyr[i], myGpyr[i], diff);
            minMaxLoc(diff, &minVal, &maxVal);
            cout << "maximum pixel error: " << maxVal << endl;
            minVal = 0;
            maxVal = 0;
            diff.zeros(diff.rows,diff.cols,diff.type());

            cv::waitKey(0);
        }
    } else {
        std::cout << "OpenCV Pyramid and custom way are different" << std::endl;
    }



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
    cv::imread("../images/apple.jpg",  cv::IMREAD_COLOR).convertTo(im_Apple,  CV_32FC3, (1./255.));
    cv::imread("../images/orange.jpg", cv::IMREAD_COLOR).convertTo(im_Orange, CV_32FC3, (1./255.));
    cv::imshow("orange", im_Orange);
    cv::imshow("apple",  im_Apple );
    std::cout << "\n" << "Input images" << "   \t\t\t\t\t\t\t" << "Press any key..." << std::endl;
    cv::waitKey(0);

    // Perform the blending using a Laplacian Pyramid
    
    GaussianBlur(im_Apple, im_Apple, Size(5,5), 1.5);
    GaussianBlur(im_Orange, im_Orange, Size(5,5), 1.5);
    //Generate Gaussian pyramid for orange and apply image
    std::vector<cv::Mat>   applepyr;
    std::vector<cv::Mat>   orangepyr;
    
    Mat tmp;
    im_Apple.copyTo(tmp);
    applepyr.push_back(tmp);
    while( (tmp.cols*0.5 > 10) && (tmp.rows*0.5 > 10) ){
        cv::pyrDown( tmp, tmp, Size( (tmp.cols+1)*0.5, (tmp.rows+1)*0.5 ) );
        applepyr.push_back(tmp);
    }
    im_Orange.copyTo(tmp);
    orangepyr.push_back(tmp);
    while( (tmp.cols*0.5 > 10) && (tmp.rows*0.5 > 10) ){
        cv::pyrDown( tmp, tmp, Size( (tmp.cols+1)*0.5, (tmp.rows+1)*0.5 ) );
        orangepyr.push_back(tmp);
    }
    
    //Generate Laplacian pyramid for orange and apple gaussian pyramid
    std::vector<cv::Mat> appleLapyr;
    std::vector<cv::Mat> orangeLapyr;
    Mat scaleUp;
    Mat subLap;
    for(std::vector<Mat>::size_type i = applepyr.size() - 1; i > 0; i--) {
            Size picsize;
            if (Size( (applepyr[i].cols)*2 - 1, (applepyr[i].rows)*2 - 1) == applepyr[i - 1].size()) {
                picsize = Size( (applepyr[i].cols)*2 - 1, (applepyr[i].rows)*2 - 1);
            } else {
                picsize = Size( (applepyr[i].cols)*2, (applepyr[i].rows)*2);
            }
            cv::pyrUp(applepyr[i], scaleUp, picsize);
            cv::subtract(applepyr[i - 1],scaleUp, subLap);
            appleLapyr.push_back(subLap);
    }
    
    for(std::vector<Mat>::size_type i = orangepyr.size() - 1; i > 0; i--) {
            Size picsize;
            if (Size( (orangepyr[i].cols)*2 - 1, (orangepyr[i].rows)*2 - 1) == orangepyr[i - 1].size()) {
                picsize = Size( (orangepyr[i].cols)*2 - 1, (orangepyr[i].rows)*2 - 1);
            } else {
                picsize = Size( (orangepyr[i].cols)*2, (orangepyr[i].rows)*2);
            }
            cv::pyrUp(orangepyr[i], scaleUp, picsize);
            cv::subtract(orangepyr[i - 1],scaleUp, subLap);
            orangeLapyr.push_back(subLap);
    }
    
    //Combine the half of the apple image with the half of the orange image 
    std::vector<cv::Mat> combipyr;
    std::vector<cv::Mat> combiLapyr;
    Mat combi;
    for(std::vector<Mat>::size_type i = 0; i != orangeLapyr.size(); i++) {
        combi = Mat(orangeLapyr[i].size(), orangeLapyr[i].type());
        for (int j = 0; j < orangeLapyr[i].cols; j++) {
            if (j < orangeLapyr[i].cols / 2) {
                appleLapyr[i].col(j).copyTo(combi.col(j));
            } else {
                orangeLapyr[i].col(j).copyTo(combi.col(j));
            }
        }
        combiLapyr.push_back(combi);
    }
    
    for(std::vector<Mat>::size_type i = 0; i != orangepyr.size(); i++) {
        combi = Mat(orangepyr[i].size(), orangepyr[i].type());
        for (int j = 0; j < orangepyr[i].cols; j++) {
            if (j < orangepyr[i].cols / 2) {
                applepyr[i].col(j).copyTo(combi.col(j));
            } else {
                orangepyr[i].col(j).copyTo(combi.col(j));
            }
        }
        combipyr.push_back(combi);
    }
    
    //Reconstruct the image of each level
    std::vector<cv::Mat> reconstrutpyr;
    Mat reconstruct;
    reconstrutpyr.push_back(combipyr[combiLapyr.size()]);
    for (std::vector<Mat>::size_type i = 0; i != combiLapyr.size(); i++) {
        
        cv::add(combipyr[combiLapyr.size() - i - 1], combiLapyr[i], reconstruct);
        reconstrutpyr.push_back(reconstruct);
    }
    
    //Display the reconstructed image for each level 
    for(std::vector<Mat>::size_type i = 0; i != reconstrutpyr.size(); i++) {
        imshow("Task2: Image combination", reconstrutpyr[i]);
        cv::waitKey(0);
    }
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

    cv::Mat           im_Traffic_BGR = cv::imread("../images/traffic.jpg"); // traffic.jpg // circles.png
    // Blur to denoise
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the computations asked in the exercise sheet
    cv::Mat im_Traffic_Sobel_x;
    cv::Mat im_Traffic_Sobel_y;
    /// Total Gradient (approximate)
    Mat abs_grad_x, abs_grad_y;
    Mat grad;

    cv::Sobel(im_Traffic_Gray, im_Traffic_Sobel_x, im_Traffic_Gray.depth(), 1, 0, 3);
    cv::Sobel(im_Traffic_Gray, im_Traffic_Sobel_y, im_Traffic_Gray.depth(), 0, 1, 3);

    Mat im_Traffic_Orientations;
    im_Traffic_Gray.copyTo(im_Traffic_Orientations);

    double gradient;
    double edgeStrength;

    for(int i = 0; i < im_Traffic_Gray.rows; i++){
       for(int j = 0; j < im_Traffic_Gray.cols; j++){

           // Store in orientation matrix element
            gradient = fastAtan2(im_Traffic_Sobel_x.at<int>(i,j), im_Traffic_Sobel_y.at<int>(i,j)) * 180 / M_PI;

            edgeStrength = sqrt( (im_Traffic_Sobel_x.at<int>(i,j)*im_Traffic_Sobel_x.at<int>(i,j) + im_Traffic_Sobel_y.at<int>(i,j)*im_Traffic_Sobel_y.at<int>(i,j)) );

            edgeStrength /= 45000;

            if(edgeStrength > 1)
                drawArrow(im_Traffic_Orientations, Point(j,i), cv::Scalar(255,255,255), 10., 5., edgeStrength, gradient);
       }
    }


    cout << "im_Traffic_Sobel_x " << im_Traffic_Sobel_x.rows << ", " << im_Traffic_Sobel_x .cols << " - " << im_Traffic_Sobel_x.depth() << endl;
    cout << "im_Traffic_Sobel_y " << im_Traffic_Sobel_y.rows << ", " << im_Traffic_Sobel_y .cols << " - " << im_Traffic_Sobel_y.depth()<< endl;
    cout << "im_Traffic_Orientations " << im_Traffic_Orientations.rows << ", " << im_Traffic_Orientations .cols << " - " << im_Traffic_Orientations.depth()<< endl;

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    imshow("Task3: gray image", im_Traffic_Orientations);

    imshow("Task3: sobel image x", im_Traffic_Sobel_x);
    imshow("Task3: sobel image y", im_Traffic_Sobel_y);





    // Use the function **drawArrow** provided at the end of this file in order to
    // drawArrow(cv::Mat &image, cv::Point p, cv::Scalar color, double scaleMagnitude, double scaleArrowHead, double magnitube, double orientationDegrees);



    // draw Vectors showing the Gradient Magnitude and Orientation
    // (to avoid clutter, draw every 10nth gradient,
    // only if the magnitude is above a threshold)


    cv::waitKey(0);
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

    cv::Mat im_Traffic_BGR = cv::imread("../images/traffic.jpg");
    // BGR to Gray
    cv::Mat                       im_Traffic_Gray;
    cv::cvtColor( im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );
    cv::Mat im_Traffic_Edges;

    blur(im_Traffic_Gray, im_Traffic_Gray, Size(3,3));

    // Perform edge detection as described in the exercise sheet
    cv::Canny(im_Traffic_Gray, im_Traffic_Edges, 120.0, 200.0, 3);

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    imshow("Task 4: gray image", im_Traffic_Gray);
    imshow("Task 4: edge image", im_Traffic_Edges);

    cv::waitKey(0); // waits until the user presses a button
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
    cv::Mat                           im_Traffic_BGR = cv::imread("../images/traffic.jpg");
    cv::GaussianBlur( im_Traffic_BGR, im_Traffic_BGR,  cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    // BGR to Gray
    cv::Mat                           im_Traffic_Gray;
    cv::cvtColor(     im_Traffic_BGR, im_Traffic_Gray, cv::COLOR_BGR2GRAY );

    // Read Template
    cv::Mat im_Sign_BGR = cv::imread("../images/sign.png");
    // Rescale the template to the size of the two largest traffic signs in the image.
    resize(im_Sign_BGR, im_Sign_BGR, Size(80,80));

    cv::Mat im_Sign_Gray;
    // BGR to Gray
    cv::cvtColor( im_Sign_BGR, im_Sign_Gray, cv::COLOR_BGR2GRAY ); // cv::COLOR_BGR2GRAY // CV_BGR2GRAY

    // Perform the steps described in the exercise sheet
    cv::Mat im_Traffic_Edges;
    cv::Mat im_Sign_Edges;

    // canny edge detector
    cv::Canny(im_Traffic_Gray, im_Traffic_Edges, 120.0, 200.0, 5);
    cv::Canny(im_Sign_Gray, im_Sign_Edges, 120.0, 200.0, 5);


    // invert edges to get right distance transform
    bitwise_not ( im_Traffic_Edges, im_Traffic_Edges );

    // distance Transform
    cv::Mat im_Traffic_Distances;
    cv::distanceTransform(im_Traffic_Edges, im_Traffic_Distances, CV_DIST_L2 , 3);
    normalize(im_Traffic_Distances, im_Traffic_Distances, 0, 1., NORM_MINMAX);

    // output image with boundingbox
    Mat detectionSign;
    im_Traffic_Gray.copyTo(detectionSign);

    // voting space
    Mat distancesToSign = Mat(im_Traffic_Gray.rows, im_Traffic_Gray.cols, CV_32FC1);
    distancesToSign = 50.;

    normalize(im_Sign_Edges, im_Sign_Edges, 0, 1., NORM_MINMAX);
    im_Sign_Edges.convertTo(im_Sign_Edges, CV_32FC1);

    float templateSum;

    for (int i=0; i < im_Traffic_Distances.rows; ++i) {
        for (int j=0;  j < im_Traffic_Distances.cols; ++j) {

                templateSum = 0;

                if( ((i + im_Sign_Edges.rows) < im_Traffic_Gray.rows) && ((j + im_Sign_Edges.cols) < im_Traffic_Gray.cols) ) {


                for (int t=0; t < im_Sign_Edges.rows; ++t) {
                    for (int k=0;  k < im_Sign_Edges.cols; ++k) {

                            if(im_Sign_Edges.at<uchar>(t,k) > 0)
                                templateSum += im_Traffic_Distances.at<float>(i+t,j+k);

                    }
                }

                // vote for template
                distancesToSign.at<float>(i,j) = templateSum;

                }

        }
    }
    // normalize
    normalize(distancesToSign, distancesToSign, 0, 255, NORM_MINMAX);

    // to view distances/voting space as grayscaled image
    distancesToSign.convertTo(distancesToSign, CV_8UC1);

    // calc position of minimum distance => detection
    double minValSign, maxValSign;
    Point minLocSign, maxLocSign;
    minMaxLoc(distancesToSign, &minValSign, &maxValSign, &minLocSign, &maxLocSign);


    // draw detected sign
    rectangle(detectionSign, Rect(minLocSign.x-(0.5*im_Sign_BGR.cols),minLocSign.y-(0.5*im_Sign_BGR.rows),im_Sign_BGR.cols,im_Sign_BGR.rows), Scalar(255,255,255), 2);

    imshow("Task 5: Detection", distancesToSign);


    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    imshow("Task 5: detection Sign", detectionSign);


    cv::waitKey(0); // waits until the user presses a button

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
