#include "ORBextractor.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;
using namespace ORB_SLAM3;

int DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

int computeHammingDistance(const Mat& desc1, const Mat& desc2)
{
    // return norm(desc1, desc2, NORM_HAMMING);
    return DescriptorDistance(desc1, desc2);
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

    // Initialize ORBextractor
    int nFeatures = 1000;
    float scaleFactor = 1.2f;
    int nLevels = 8;
    int iniThFAST = 20;
    int minThFAST = 7;

    ORBextractor orbExtractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    // Extract keypoints and descriptors for left image
    vector<KeyPoint> keypointsLeft;
    Mat descriptorsLeft;
    vector<int> vLappingAreaLeft;
    int monoIndexLeft = orbExtractor(imLeft, Mat(), keypointsLeft, descriptorsLeft, vLappingAreaLeft);

    // Extract keypoints and descriptors for right image
    vector<KeyPoint> keypointsRight;
    Mat descriptorsRight;
    vector<int> vLappingAreaRight;
    int monoIndexRight = orbExtractor(imRight, Mat(), keypointsRight, descriptorsRight, vLappingAreaRight);

    // Brute force matching with ratio test
    vector<DMatch> matches;
    for (int i = monoIndexLeft; i < descriptorsLeft.rows; i++)
    {
        int bestMatchIndex = -1;
        int secondBestMatchIndex = -1;
        int bestMatchDistance = std::numeric_limits<int>::max();
        int secondBestMatchDistance = std::numeric_limits<int>::max();

        for (int j = monoIndexRight; j < descriptorsRight.rows; j++)
        {
            // Compute Hamming distance between descriptors
            int dist = computeHammingDistance(descriptorsLeft.row(i), descriptorsRight.row(j));

            if (dist < bestMatchDistance)
            {
                secondBestMatchDistance = bestMatchDistance;
                secondBestMatchIndex = bestMatchIndex;
                bestMatchDistance = dist;
                bestMatchIndex = j;
            }
            else if (dist < secondBestMatchDistance)
            {
                secondBestMatchDistance = dist;
                secondBestMatchIndex = j;
            }
        }

        // Apply Lowe's ratio test to ensure the best match is sufficiently better than the second-best
        if (bestMatchDistance < 0.75 * secondBestMatchDistance)
        {
            matches.push_back(DMatch(i, bestMatchIndex, static_cast<float>(bestMatchDistance)));
        }
    }

    // Draw matches
    Mat imMatches;
    drawMatches(imLeft, keypointsLeft, imRight, keypointsRight, matches, imMatches);

    // Display the matches
    imshow("Stereo Keypoints Matches", imMatches);
    waitKey(0);

    // Print keypoints and descriptor information
    cout << "Left Image Keypoints: " << keypointsLeft.size() << endl;
    cout << "Right Image Keypoints: " << keypointsRight.size() << endl;
    cout << "Number of Matches: " << matches.size() << endl;

    return 0;
}
