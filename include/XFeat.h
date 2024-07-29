#ifndef XFEAT_H
#define XFEAT_H

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <tuple>

namespace ORB_SLAM3
{
    struct BasicLayerImpl : torch::nn::Module
    {   
        /*
            Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
        */
        torch::nn::Sequential layer;

        BasicLayerImpl(int in_channels, 
                    int out_channels, 
                    int kernel_size,
                    int stride,
                    int padding);
        torch::Tensor forward(torch::Tensor x);
    };

    TORCH_MODULE(BasicLayer);

    struct XFeatModel : torch::nn::Module
    {
        /* 
            C++ implementation (declaration) of architecture described in
            "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
        */
        torch::nn::InstanceNorm2d norm{nullptr};
        torch::nn::Sequential skip1{nullptr}; 
        torch::nn::Sequential block1{nullptr}, 
                            block2{nullptr}, 
                            block3{nullptr}, 
                            block4{nullptr}, 
                            block5{nullptr};
        torch::nn::Sequential block_fusion{nullptr}, 
                            heatmap_head{nullptr}, 
                            keypoint_head{nullptr};
        torch::nn::Sequential fine_matcher{nullptr};

        XFeatModel();
        torch::Tensor unfold2d(torch::Tensor x, int ws=2);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x);
    };

    class InterpolateSparse2d : public torch::nn::Module
    {
    public:
        InterpolateSparse2d(const std::string& mode = "bilinear", bool align_corners = false);
        torch::Tensor forward(torch::Tensor x, torch::Tensor pos, int H, int W);

    private:
        torch::Tensor normgrid(torch::Tensor x, int H, int W);

        std::string mode;
        bool align_corners;
    };

    class XFDetector
    {
    public:
        XFDetector(int _top_k=4096, float _detection_threshold=0.05, bool use_cuda=true);
        void detect(cv::Mat &img);
        void getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint>& _keypoints);
        void computeDescriptors(const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
        
    private:
        std::string getWeightsPath(std::string weights);
        void parseInput(cv::Mat& img, torch::Tensor& tensor);
        std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor& x);
        torch::Tensor getKptsHeatmap(torch::Tensor& kpts, float softmax_temp=1.0);
        torch::Tensor NMS(torch::Tensor& x, int kernel_size = 5);

        int top_k;
        float detection_threshold;
        torch::DeviceType device_type;
        std::shared_ptr<XFeatModel> model;
        std::shared_ptr<InterpolateSparse2d> bilinear, nearest;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

}  // namepsace ORB_SLAM3


#endif // XFEAT_H