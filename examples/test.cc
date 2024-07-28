#include "XFeat.h"

// forward declarations
std::string parsePath(std::string relativePath);
void warp_corners_and_draw_matches(cv::Mat& mkpts_0, cv::Mat& mkpts_1, cv::Mat& img1, cv::Mat& img2);


int main(int argc, char** argv) {

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <weights> <image1> <image2>\n";
        return -1;
    }

    // instantiate XFDetector
    int top_k = 4096;
    float detection_threshold = 0.05;
    bool use_cuda = true; 
    ORB_SLAM3::XFDetector detector(top_k, detection_threshold, use_cuda);

    // load the image and convert to tensor
    cv::Mat image1 = cv::imread(argv[1]);
    cv::Mat image2 = cv::imread(argv[2]);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }

    // Perform feature matching on the same image
    cv::Mat mkpts_0, mkpts_1;
    std::tie(mkpts_0, mkpts_1) = detector.match_xfeat(image1, image2);

    warp_corners_and_draw_matches(mkpts_0, mkpts_1, image1, image2);

    return 0;
}

void warp_corners_and_draw_matches(cv::Mat& ref_points, cv::Mat& dst_points, cv::Mat& img1, cv::Mat& img2)
{      
    // Check if there are enough points to find a homography
    if (ref_points.rows < 4 || dst_points.rows < 4) {
        std::cerr << "Not enough points to compute homography" << std::endl;
        return;
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0, mask, 1000, 0.994);
    if (H.empty()) {
        std::cerr << "Homography matrix is empty" << std::endl;
        return;
    }
    mask = mask.reshape(1);

    float h = img1.rows;
    float w = img1.cols;
    std::vector<cv::Point2f> corners_img1 = {cv::Point2f(    0,     0), 
                                                cv::Point2f(w - 1,     0), 
                                                cv::Point2f(w - 1, h - 1), 
                                                cv::Point2f(    0, h - 1)};
    std::vector<cv::Point2f> warped_corners;
    cv::perspectiveTransform(corners_img1, warped_corners, H);

    cv::Mat img2_with_corners = img2.clone();
    for (size_t i = 0; i < warped_corners.size(); ++i) {
        cv::line(img2_with_corners, warped_corners[i], warped_corners[(i+1) % warped_corners.size()], cv::Scalar(0, 255, 0), 4);
    }

    // prepare keypoints and matches for drawMatches function
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < mask.rows; ++i) {
        keypoints1.emplace_back(ref_points.at<cv::Point2f>(i, 0), 5);
        keypoints2.emplace_back(dst_points.at<cv::Point2f>(i, 0), 5);
        if (mask.at<uchar>(i, 0))
            matches.emplace_back(i, i, 0);
    }
    
    // Draw inlier matches
    cv::Mat img_matches;
    if (!keypoints1.empty() && !keypoints2.empty() && !matches.empty()) {
        cv::drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::Mat resized_frame;
        cv::resize(img_matches, resized_frame, cv::Size(1366, 1440));
        cv::imshow("Matches", img_matches);
        cv::waitKey(0); // Wait for a key press
                    
        // // Uncomment to save the matched image
        // std::string output_path = "doc/image_matches.png";
        // if (cv::imwrite(output_path, img_matches)) {
        //     std::cout << "Saved image matches to " << output_path << std::endl;
        // } else {
        //     std::cerr << "Failed to save image matches to " << output_path << std::endl;
        // }

    } else {
        std::cerr << "Keypoints or matches are empty, cannot draw matches" << std::endl;
    }
}

std::string parsePath(std::string relativePath)
{
    std::filesystem::path current_file = __FILE__;
    std::filesystem::path parent_dir = current_file.parent_path();
    std::filesystem::path full_path = parent_dir / ".." / relativePath;
    full_path = std::filesystem::absolute(full_path);

    return static_cast<std::string>(full_path);   
}