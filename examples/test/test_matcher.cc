#include "XFextractor.h"
#include "ORBmatcher.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: ./test_stereo path_to_left_image path_to_right_image" << endl;
        return -1;
    }

    // Load images
    Mat imLeft = imread(argv[1], IMREAD_GRAYSCALE);
    Mat imRight = imread(argv[2], IMREAD_GRAYSCALE);

    if (imLeft.empty() || imRight.empty())
    {
        cerr << "Cannot load images!" << endl;
        return -1;
    }

    // Initialize XFextractor
    int nFeatures = 1000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

    XFextractor XFextractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    // Define vLapping area for stereo
    std::vector<int> vLappingArea = {imLeft.cols / 4, 3 * imLeft.cols / 4};

    // Extract keypoints and descriptors for left image
    vector<KeyPoint> keypointsLeft;
    Mat descriptorsLeft;
    int monoIndexLeft = XFextractor(imLeft, Mat(), keypointsLeft, descriptorsLeft, vLappingArea);

    // Extract keypoints and descriptors for right image
    vector<KeyPoint> keypointsRight;
    Mat descriptorsRight;
    int monoIndexRight = XFextractor(imRight, Mat(), keypointsRight, descriptorsRight, vLappingArea);

    // Initialize ORBmatcher and match descriptors
    ORBmatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptorsLeft, descriptorsRight, matches);

    // Draw matches
    Mat imMatches;
    drawMatches(imLeft, keypointsLeft, imRight, keypointsRight, matches, imMatches);

    // Display the matches
    imshow("Matches", imMatches);
    waitKey(0);

    // Print keypoints and descriptor information
    cout << "Left Image Keypoints: " << keypointsLeft.size() << endl;
    cout << "Right Image Keypoints: " << keypointsRight.size() << endl;
    cout << "Descriptor Sizes:" << endl;
    cout << "Left Descriptors: " << descriptorsLeft.rows << " x " << descriptorsLeft.cols << endl;
    cout << "Right Descriptors: " << descriptorsRight.rows << " x " << descriptorsRight.cols << endl;

    // Print sections of descriptor values
    if (!descriptorsLeft.empty() && !descriptorsRight.empty())
    {
        cout << "Sample Descriptors (Left Image):" << endl;
        for (int i = 0; i < min(5, descriptorsLeft.rows); ++i) {
            cout << "Descriptor " << i + 1 << ": ";
            for (int j = 0; j < min(5, descriptorsLeft.cols); ++j) {
                cout << descriptorsLeft.at<float>(i, j) << " ";
            }
            cout << "..." << endl;
        }

        cout << "Sample Descriptors (Right Image):" << endl;
        for (int i = 0; i < min(5, descriptorsRight.rows); ++i) {
            cout << "Descriptor " << i + 1 << ": ";
            for (int j = 0; j < min(5, descriptorsRight.cols); {
                cout << descriptorsRight.at<float>(i, j) << " ";
            }
            cout << "..." << endl;
        }
    }

    return 0;
}