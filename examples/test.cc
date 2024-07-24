#include "XFeat.h"

// cv::Mat processImage(const cv::Mat& image)
// {
//     // Convert the image to grayscale if it is not already
//     cv::Mat grayImage;
//     if (image.channels() == 3) {
//         cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
//     } else {
//         grayImage = image;
//     }

//     // Resize the image to 224x224
//     cv::Mat resizedImage;
//     cv::resize(grayImage, resizedImage, cv::Size(448, 448));

//     return resizedImage;
// }

int main(int argc, char** argv) {

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <weights> <image1>\n";
        return -1;
    }

    // create a new xfeat model
    auto xfeat = std::make_shared<ORB_SLAM3::XFeatModel>();

    // load the model parameters
    torch::serialize::InputArchive archive;
    archive.load_from(argv[1]);
    xfeat->load(archive);
    std::cout << "Model weights loaded successfully." << std::endl;

    // load the image and convert to tensor
    cv::Mat image = cv::imread(argv[2]);
    if (image.empty()) {
        std::cerr << "Could not open or find the image!" << std::endl;
        return -1;
    }

    // instantiate XFDetector
    ORB_SLAM3::XFDetector detector(xfeat);

    // Detect and compute features in the image
    int top_k = 1000;  // Example value for top_k
    float detection_threshold = 0.5;  // Example threshold
    bool use_cuda = true;  // Use CUDA if available

    // extract features using the detector
    auto result = detector.detectAndCompute(image, top_k, detection_threshold, use_cuda);

    // Convert keypoints to OpenCV format and draw them
    std::vector<cv::KeyPoint> cv_keypoints;
    for (const auto& item : result) {
        auto keypoints_tensor = item.at("keypoints");
        auto scores_tensor = item.at("scores");
        for (int i = 0; i < keypoints_tensor.size(0); ++i) {
            float x = keypoints_tensor[i][0].item<float>();
            float y = keypoints_tensor[i][1].item<float>();
            float score = scores_tensor[i].item<float>();
            cv::KeyPoint kp(x, y, 8, -1, score);
            cv_keypoints.push_back(kp);
        }
    }

    // Draw the keypoints on the image
    cv::Mat outImage;
    cv::drawKeypoints(image, cv_keypoints, outImage);

    // Display the image with keypoints
    cv::imshow("Keypoints", outImage);
    cv::waitKey(0); // Wait for a key press

    return 0;
}