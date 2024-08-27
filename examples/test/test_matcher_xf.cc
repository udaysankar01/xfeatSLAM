#include "XFextractor.h"
#include "ORBmatcher.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>  // For std::sqrt

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

// Function to compute Euclidean distance between two descriptors
float computeDescriptorDistance(const Mat& desc1, const Mat& desc2)
{
    float dist = 0.0;
    for (int i = 0; i < desc1.cols; i++)
    {
        float diff = desc1.at<float>(0, i) - desc2.at<float>(0, i);
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

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

    // Brute force matching
    vector<DMatch> matches;
    for (int i = 0; i < descriptorsLeft.rows; i++)
    {
        int bestMatchIndex = -1;
        float bestMatchDistance = std::numeric_limits<float>::max();

        for (int j = 0; j < descriptorsRight.rows; j++)
        {
            // Compute distance between descriptors
            float dist = computeDescriptorDistance(descriptorsLeft.row(i), descriptorsRight.row(j));

            // Check if this is the best match so far
            if (dist < bestMatchDistance)
            {
                bestMatchDistance = dist;
                bestMatchIndex = j;
            }
        }

        // Store the best match
        if (bestMatchIndex >= 0)
        {
            matches.push_back(DMatch(i, bestMatchIndex, bestMatchDistance));
        }
    }

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

    return 0;
}
