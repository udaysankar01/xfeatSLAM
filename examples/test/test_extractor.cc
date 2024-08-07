#include <iostream>
#include <opencv2/opencv.hpp>
#include "XFextractor.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Could not load image " << argv[1] << std::endl;
        return -1;
    }

    // Initialize ORB extractor
    int nfeatures = 1000; 
    float scaleFactor = 1.2f; 
    int nlevels = 8;
    int iniThFAST = 20; 
    int minThFAST = 7;
    ORB_SLAM3::XFextractor extractor(nfeatures, scaleFactor, nlevels, iniThFAST, minThFAST);
    
    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<int> vLappingArea = {0, image.cols}; // TODO: Implement Overlapping Area
    
    extractor(image, cv::noArray(), keypoints, descriptors, vLappingArea);

    // Draw keypoints on the image
    cv::Mat outputImage;
    cv::drawKeypoints(image, keypoints, outputImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    // Display the result
    cv::imshow("Keypoints", outputImage);
    cv::waitKey(0);

    // // Save the result
    // cv::imwrite("output_keypoints.jpg", outputImage);

    return 0;
}