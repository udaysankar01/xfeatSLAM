#include "XFeat.h"

int main(int argc, char** argv) {

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <weights> <image1> <image2>\n";
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
    cv::Mat image1 = cv::imread(argv[2]);
    cv::Mat image2 = cv::imread(argv[3]);
    if (image1.empty() || image2.empty()) {
        std::cerr << "Could not open or find the images!" << std::endl;
        return -1;
    }

    // instantiate XFDetector
    ORB_SLAM3::XFDetector detector(xfeat);

    // Detect and match features in the image
    int top_k = 4096;  // Example value for top_k
    float min_cossim = -1;  // Example value for minimum cosine similarity
    bool use_cuda = true;  // Use CUDA if available

    // Perform feature matching on the same image
    cv::Mat mkpts_0, mkpts_1;
    std::tie(mkpts_0, mkpts_1) = detector.match_xfeat(image1, image2, top_k, min_cossim);

    detector.warp_corners_and_draw_matches(mkpts_0, mkpts_1, image1, image2);

    return 0;
}