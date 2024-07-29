#include "XFeat.h"

void warp_corners_and_draw_matches(cv::Mat& mkpts_0, cv::Mat& mkpts_1, cv::Mat& img1, cv::Mat& img2);

std::tuple<torch::Tensor, torch::Tensor> match(const torch::Tensor& feats1, const torch::Tensor& feats2, double min_cossim)
{   
    // Compute cosine similarity between feats1 and feats2
    torch::Tensor cossim = torch::matmul(feats1, feats2.t());
    torch::Tensor cossim_t = torch::matmul(feats2, feats1.t());

    torch::Tensor match12, match21;
    std::tie(std::ignore, match12) = cossim.max(1);
    std::tie(std::ignore, match21) = cossim_t.max(1);

    // Index tensor
    torch::Tensor idx0 = torch::arange(match12.size(0), match12.options());
    torch::Tensor mutual = match21.index({match12}) == idx0;

    torch::Tensor idx1;

    if (min_cossim > 0)
    {
        std::tie(cossim, std::ignore) = cossim.max(1);
        torch::Tensor good = cossim > min_cossim;
        idx0 = idx0.index({mutual & good});
        idx1 = match12.index({mutual & good});
    }
    else
    {
        idx0 = idx0.index({mutual});
        idx1 = match12.index({mutual});
    }

    return std::make_tuple(idx0, idx1);
}

int main(int argc, char** argv) 
{

    if (argc < 4) 
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> <min_cossim>\n";
        return -1;
    }

    try {
        ORB_SLAM3::XFDetector detector(4096, 0.05, true);

        // Load the images
        cv::Mat img1 = cv::imread(argv[1]);
        cv::Mat img2 = cv::imread(argv[2]);
        if (img1.empty() || img2.empty()) 
        {
            std::cerr << "Could not open or find the images!" << std::endl;
            return -1;
        }

        cv::Mat mkpts_0, mkpts_1;

        // Detect keypoints and compute descriptors for the first image
        detector.detect(img1);
        std::vector<cv::KeyPoint> keypoints1;
        detector.getKeyPoints(0.05, 0, img1.cols, 0, img1.rows, keypoints1);
        cv::Mat descriptors1;
        detector.computeDescriptors(keypoints1, descriptors1);

        // Detect keypoints and compute descriptors for the second image
        detector.detect(img2);
        std::vector<cv::KeyPoint> keypoints2;
        detector.getKeyPoints(0.05, 0, img2.cols, 0, img2.rows, keypoints2);
        cv::Mat descriptors2;
        detector.computeDescriptors(keypoints2, descriptors2);

        // Convert descriptors to torch tensors
        torch::Tensor feats1 = torch::from_blob(descriptors1.data, {descriptors1.rows, descriptors1.cols}, torch::kFloat).to(torch::kCUDA) / 255.0;
        torch::Tensor feats2 = torch::from_blob(descriptors2.data, {descriptors2.rows, descriptors2.cols}, torch::kFloat).to(torch::kCUDA) / 255.0;

        // Match keypoints using custom matcher
        torch::Tensor idx0, idx1;
        double min_cossim = std::stod(argv[3]);
        std::tie(idx0, idx1) = match(feats1, feats2, min_cossim);

        // Prepare points for homography
        cv::Mat ref_points(idx0.size(0), 2, CV_32F);
        cv::Mat dst_points(idx0.size(0), 2, CV_32F);
        for (int i = 0; i < idx0.size(0); ++i) {
            ref_points.at<cv::Point2f>(i) = keypoints1[idx0[i].item<int>()].pt;
            dst_points.at<cv::Point2f>(i) = keypoints2[idx1[i].item<int>()].pt;
        }

        // Draw matches and warp corners
        warp_corners_and_draw_matches(ref_points, dst_points, img1, img2);

    } 
    catch (const std::exception& ex) 
    {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return -1;
    }

    return 0;
}

void warp_corners_and_draw_matches(cv::Mat& ref_points, cv::Mat& dst_points, cv::Mat& img1, cv::Mat& img2)
{   
    // Check if there are enough points to find a homography
    if (ref_points.rows < 4 || dst_points.rows < 4) 
    {
        std::cerr << "Not enough points to compute homography" << std::endl;
        return;
    }

    cv::Mat mask;
    cv::Mat H = cv::findHomography(ref_points, dst_points, cv::USAC_MAGSAC, 10.0, mask, 1000, 0.994);
    if (H.empty()) 
    {
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
    for (size_t i = 0; i < warped_corners.size(); ++i) 
    {
        cv::line(img2_with_corners, warped_corners[i], warped_corners[(i+1) % warped_corners.size()], cv::Scalar(0, 255, 0), 4);
    }

    // prepare keypoints and matches for drawMatches function
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < mask.rows; ++i) 
    {
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

    } 
    else 
    {
        std::cerr << "Keypoints or matches are empty, cannot draw matches" << std::endl;
    }
}