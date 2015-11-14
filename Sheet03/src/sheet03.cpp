#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void customHoughCircles(  const cv::Mat                &ImageGRAY,
                                std::vector<cv::Vec3f> &detectedCircles,      // OUT
                          const int              numberOfBestCirclesReturned, // 1 - numberOfBestCirclesReturned (sorted)
                          const int              accumDivider,                // 2 - accumulator resolution (sizeImage/XXX)
                          const double           cannyHighThresh,             // 3 - Canny high threshold (low = 50%)
                          const double           minRadius,                   // 4 - min radius
                          const double           maxRadius,                   // 5 - max radius
                          const int              radiiNumb,                   // 6 - number of radii to check
                          const double           thetaStep_Degrees            // 7 - resolution of theta (degrees!)
                       );

void part1_1();
void part1_2();
void part1_3();
void part2_1();
void part2_2();
void part2_3();
void part3();

std::string  PATH_Circles = "./images/circles.png";
std::string  PATH_Face    = "./images/face2.png";
std::string  PATH_Flower  = "./images/flower.png";


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main()
{

    // Uncomment the part of the exercise that you wish to implement.
    // For the final submission all implemented parts should be uncommented.

    //part1_1();
    //part1_2();
    //part1_3();
    part2_1();
    part2_2();
    part2_3();
    //part3();

    std::cout <<                                                                                                   std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    END    /////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout <<                                                                                                   std::endl;

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1_1()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 1    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    cv::Mat       im_Circles_BGR = cv::imread( PATH_Circles );
    // BGR to Gray
    cv::Mat                       im_Circles_Gray;
    cv::cvtColor( im_Circles_BGR, im_Circles_Gray, cv::COLOR_BGR2GRAY );
    
    // Synthetic image - No Blurring necessary for denoising !!!
    
    // Perform the steps described in the exercise sheet
    
    ///////////Our Solution/////////////////////
    //First of all blur the image
    cv::GaussianBlur(im_Circles_Gray, im_Circles_Gray, cv::Size(3,3), 2,2);
    
    //initalize a vector to store the detected circles 
    std::vector<cv::Vec3f> circles;
    
    //apply the hough transform for circles
    cv::HoughCircles(im_Circles_Gray, circles, CV_HOUGH_GRADIENT, 1, 2, 1, 15, 0, 0);
    
    //iterate through the vector of detected circles
    for (std::size_t i = 0; i < circles.size(); i++) {
        //draw for each detected a red circle into the original image
        cv::Point center = cv::Point(cvRound(circles[i][0]), cvRound(circles[i][1]));
        cv::circle(im_Circles_BGR, center, cvRound(circles[i][2]), cv::Scalar(0, 0, 255));                
    }

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::imshow("Part 1 1", im_Circles_BGR);
    cv::waitKey(0); 
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1_2()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 2    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    cv::Mat im_Circles_BGR = cv::imread( PATH_Circles );
    // BGR to Gray
    cv::Mat                       im_Circles_Gray;
    cv::cvtColor( im_Circles_BGR, im_Circles_Gray, cv::COLOR_BGR2GRAY );
    // Synthetic image - No Blurring necessary for denoising !!!

    // Perform the steps described in the exercise sheet
    
    //////////Our solution//////////
    //First of all blur the image
    cv::GaussianBlur(im_Circles_Gray, im_Circles_Gray, cv::Size(3,3), 2,2);
    
    cv::Mat edges;
    cv::Canny(im_Circles_Gray, edges, 1, 15);
    
    cv::Mat hough = cv::Mat::zeros(edges.size(), CV_32FC3);
    
  //  const int sizes[] = { 100, 100, 100 };
  //  cv::Mat hough2 = cv::Mat::zeros(3, sizes, CV_32F);


    //std::cout << edges(cv::Range(0, 100), cv::Range(0, 100)) << std::endl;




    for (int x = 0; x < edges.rows - 2; x++) {
        for (int y = 0; y < edges.cols - 2; y++) {

                // for each edge point
                if(edges.at<uchar>(y,x) == 255) {

                    for (int r = 0; r < 50; r++) {

                        for (int theta = 0; theta < 180; theta++) {

                            //int d = x * cos(theta) + y * sin(theta);

                            int a = x - r * cos(theta);
                            int b = x - r * sin(theta);

                            // std::cout << r << std::endl;
                          //  hough.at<float>(d, theta) ++;

                        }

                    }

                }


            /*
                for (int theta = 1; theta <= 180; theta++) {
                    int r = y * cos(theta) + x * sin(theta);
                    if (1 <= r && r <= 15) {
                       // std::cout << r << std::endl;
                        hough.at<int>(r, theta) ++;
                    }   
                }
            */
                
                /*std::cout << "x: " << x << std::endl;
                std::cout << "y: " << y << std::endl;
                std::cout << edges.size() << std::endl;*/
                //hough.at<cv::Vec3b>(x, y) = edges.at<cv::Vec3b>(x, y);
                
                /*
                
                double d = sqrt (pow(y - 0, 2) + pow(x - 0, 2));
                double theta =  (edges.cols * y + 0 * x) / (sqrt(pow(edges.cols, 2) + pow(0, 2)) * sqrt(pow(x, 2) + pow(y, 2)));
                int a = y - d * cos(theta);
                int b = x - d * sin(theta);
                
                for (int i = 0; i < edges.cols - 2; i++) {
                    int j = a * i + b;
                    std::cout << "Berechnung: " << (a * tan(theta) - i * tan(theta) + j) << std::endl;
                    std::cout << "b: " << b << std::endl;
                    if (b == (a * tan(theta) - i * tan(theta) + j)) {
                        std::cout << "test" << std::endl;
                        hough.at<int>(i, j) = 255;
                    }
                                        
                }*/
                
            }
            

        
    }

   // std::cout << hough(cv::Range(0, 100), cv::Range(0, 100)) << std::endl;


   // std::cout << "counter " << counter << std::endl;

     
    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    cv::imshow("Part 1 2", edges);
    cv::imshow("test", hough);
    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part1_3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 1 - 3    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << std::endl << "Please Wait... CustomHoughCircles @ face..." << std::endl;

    cv::Mat im_Face_BGR = cv::imread( PATH_Face );
    cv::Mat im_Face_Gray;

    // BGR to Gray
    cv::cvtColor( im_Face_BGR, im_Face_Gray, cv::COLOR_BGR2GRAY );
    // Blur for denoising
    cv::GaussianBlur( im_Face_Gray, im_Face_Gray, cv::Size(0,0), 1,3 );
    // Erosion/Dilation to eliminate some noise
    int erDil = 1;
    cv::erode(  im_Face_Gray, im_Face_Gray, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(erDil,erDil) ) );
    cv::dilate( im_Face_Gray, im_Face_Gray, cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(erDil,erDil) ) );

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    // If needed perform normalization of the image to be displayed

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_1()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 1    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    cv::imshow("original image",                         flower);

    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("gray image", flower_gray);
    // Perform the steps described in the exercise sheet
    

    cv::Mat map(flower_gray.rows * flower_gray.cols, 3, flower_gray.type());
    for (int x = 0; x < flower_gray.rows; x++) {
        for (int y = 0; y < flower_gray.cols; y++) {
            for(int z = 0; z < 3; z++) {
                map.at<float>(x + y * flower_gray.rows, z) = flower_gray.at<cv::Vec3f>(x, y)[z];
            }           
        }
    }
    


    cv::Mat bestLabels;
    cv::Mat center;
    cv::Mat result(flower_gray.size(), flower_gray.type());
    for (int k = 2; k <= 10; k = k + 2) {
        cv::kmeans(map, k, bestLabels, cv::TermCriteria(), 5, cv::KMEANS_PP_CENTERS, center);

        for (int x = 0; x < flower_gray.rows; x++) {
            for (int y = 0; y < flower_gray.cols; y++) {
                
                int index = bestLabels.at<int>(x + y * flower_gray.rows, 0);
                result.at<cv::Vec3f>(x, y) [0] = center.at<float>(index, 0);
                result.at<cv::Vec3f>(x, y) [1] = center.at<float>(index, 1);
                result.at<cv::Vec3f>(x, y) [2] = center.at<float>(index, 2);
            }
        }
        cv::imshow("Intensity " + std::to_string(k), result);
    }



    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_2()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 2    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with color" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet

    cv::Mat map(flower.rows * flower.cols, 3, flower.type());
    for (int x = 0; x < flower.rows; x++) {
        for (int y = 0; y < flower.cols; y++) {
            for(int z = 0; z < 3; z++) {
                map.at<float>(x + y * flower.rows, z) = flower.at<cv::Vec3f>(x, y)[z];
            }           
        }
    }
    
    cv::Mat bestLabels;
    cv::Mat center;
    cv::Mat result(flower.size(), flower.type());
    for (int k = 2; k <= 10; k = k + 2) {
        cv::kmeans(map, k, bestLabels, cv::TermCriteria(), 5, cv::KMEANS_PP_CENTERS, center);
        for (int x = 0; x < flower.rows; x++) {
            for (int y = 0; y < flower.cols; y++) {
                
                int index = bestLabels.at<int>(x + y * flower.rows, 0);
                result.at<cv::Vec3f>(x, y) [0] = center.at<float>(index, 0);
                result.at<cv::Vec3f>(x, y) [1] = center.at<float>(index, 1);
                result.at<cv::Vec3f>(x, y) [2] = center.at<float>(index, 2);
            }
        }
        cv::imshow("Color " + std::to_string(k), result);
    }

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**
    cv::waitKey(0);
    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part2_3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 2 - 3    //////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    std::cout << "\n" << "kmeans with gray and pixel coordinates" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    // gray version of flower
    cv::Mat              flower_gray;
    cv::cvtColor(flower, flower_gray, CV_BGR2GRAY);
    cv::imshow("original image", flower);

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void part3()
{
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////    Part 3    //////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;
    std::cout << "////////////////////////////////////////////////////////////////////////////////////////////" << std::endl;

    // read the image file flower
    cv::Mat                                              flower;
    cv::imread( PATH_Flower, cv::IMREAD_COLOR).convertTo(flower, CV_32FC3, (1./255.)); // image normalized [0,255] -> [0,1]
    cv::imshow("original image",                         flower);
    // BGR -> LUV
    cv::Mat              flower_luv;
    cv::cvtColor(flower, flower_luv, CV_BGR2Luv);

    // Perform the steps described in the exercise sheet

    // Show results
    // using **cv::imshow and cv::waitKey()** and when necessary **std::cout**
    // In the end, after the last cv::waitKey(), use **cv::destroyAllWindows()**

    cv::destroyAllWindows();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
