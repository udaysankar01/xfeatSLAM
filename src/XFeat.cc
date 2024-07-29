#include "XFeat.h"
#include <tuple>

namespace ORB_SLAM3
{
    /////////////////////////////  MODEL IMPLEMENTATION  ////////////////////////////////////
    BasicLayerImpl::BasicLayerImpl(int in_channels, 
                    int out_channels, 
                    int kernel_size,
                    int stride,
                    int padding)
    {
        layer = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .padding(padding)
                .stride(stride)
                .dilation(1)
                .bias(false)),
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(out_channels).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
        );
        register_module("layer", layer);
    }

    torch::Tensor BasicLayerImpl::forward(torch::Tensor x) 
    {
        return layer->forward(x);
    }

    XFeatModel::XFeatModel()
    {
        norm = torch::nn::InstanceNorm2d(1);

        // CNN Backbone and Heads

        skip1 = torch::nn::Sequential(
            torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 24, 1).stride(1).padding(0))
        );
        
        block1 = torch::nn::Sequential(
            BasicLayer(1,  4, 3, 1, 1),
            BasicLayer(4,  8, 3, 2, 1),
            BasicLayer(8,  8, 3, 1, 1),
            BasicLayer(8, 24, 3, 2, 1)
        );

        block2 = torch::nn::Sequential(
            BasicLayer(24, 24, 3, 1, 1),
            BasicLayer(24, 24, 3, 1, 1)
        );

        block3 = torch::nn::Sequential(
            BasicLayer(24, 64, 3, 2, 1),
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 1, 1, 0)
        );

        block4 = torch::nn::Sequential(
            BasicLayer(64, 64, 3, 2, 1),
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 3, 1, 1)
        );

        block5 = torch::nn::Sequential(
            BasicLayer( 64, 128, 3, 2, 1),
            BasicLayer(128, 128, 3, 1, 1),
            BasicLayer(128, 128, 3, 1, 1),
            BasicLayer(128,  64, 1, 1, 0)
        );

        block_fusion = torch::nn::Sequential(
            BasicLayer(64, 64, 3, 1, 1),
            BasicLayer(64, 64, 3, 1, 1),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 1).padding(0))
        );

        heatmap_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 1, 1)),
            torch::nn::Sigmoid()
        );

        keypoint_head = torch::nn::Sequential(
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            BasicLayer(64, 64, 1, 1, 0),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 65, 1))
        );

        // Fine Matcher MLP

        fine_matcher = torch::nn::Sequential(
            torch::nn::Linear(128, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 512),
            torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512).affine(false)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Linear(512, 64)
        );


        register_module("norm", norm);
        register_module("skip1", skip1);
        register_module("block1", block1);
        register_module("block2", block2);
        register_module("block3", block3);
        register_module("block4", block4);
        register_module("block5", block5);
        register_module("block_fusion", block_fusion);
        register_module("heatmap_head", heatmap_head);
        register_module("keypoint_head", keypoint_head);
        register_module("fine_matcher", fine_matcher);
    }

    torch::Tensor XFeatModel::unfold2d(torch::Tensor x, int ws)
    {   
        /*
           Unfolds tensor in 2D with desired ws (window size) and concat the channels
        */
        auto shape = x.sizes();
        int B = shape[0], C = shape[1], H = shape[2], W = shape[3];
        x = x.unfold(2, ws, ws).unfold(3, ws, ws).reshape({B, C, H/ws, W/ws, ws*ws});
        return x.permute({0, 1, 4, 2, 3}).reshape({B, -1, H/ws, W/ws});
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> XFeatModel::forward(torch::Tensor x) 
    {   
        /* 
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats      -> torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints  -> torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap    -> torch.Tensor(B,  1, H/8, W/8) reliability map
        */

        // don't backprop through normalization
        torch::NoGradGuard no_grad;
        x = x.mean(1, true);
        x = norm->forward(x);

        // main backbone
        torch::Tensor x1 = block1->forward(x);
        torch::Tensor x2 = block2->forward(x1 + skip1->forward(x));
        torch::Tensor x3 = block3->forward(x2);
        torch::Tensor x4 = block4->forward(x3);
        torch::Tensor x5 = block5->forward(x4);

        // pyramid fusion
        std::vector<int64_t> size_array = {x3.size(2), x3.size(3)};
        x4 = torch::nn::functional::interpolate(x4, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        x5 = torch::nn::functional::interpolate(x5, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                .mode(torch::kBilinear)
                                                                                                .align_corners(false));
        torch::Tensor feats = block_fusion->forward(x3 + x4 + x5);

        // heads
        torch::Tensor heatmap = heatmap_head->forward(feats);
        torch::Tensor keypoints = keypoint_head->forward(unfold2d(x, 8));

        return std::make_tuple(feats, keypoints, heatmap);
    }

    //////////////////////////////// InterpolateSparse2d /////////////////////////////////
    InterpolateSparse2d::InterpolateSparse2d(const std::string& mode, bool align_corners)
    : mode(mode), align_corners(align_corners)
    {
    }

    torch::Tensor InterpolateSparse2d::normgrid(torch::Tensor x, int H, int W)
    {
        // normalize coordinates to [-1, 1]
        torch::Tensor size_tensor = torch::tensor({W - 1, H - 1}, x.options());
        return 2.0 * (x / size_tensor) - 1.0;
    }

    torch::Tensor InterpolateSparse2d::forward(torch::Tensor x, torch::Tensor pos, int H, int W)
    {
        // normalize the positions
        torch::Tensor grid = normgrid(pos, H, W).unsqueeze(-2).to(x.dtype());

        // grid sampling  ---- EDIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if (mode == "bilinear")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).align_corners(align_corners));
        }
        else if (mode == "nearest")
        {
            x = torch::nn::functional::grid_sample(x, grid, torch::nn::functional::GridSampleFuncOptions().mode(torch::kNearest).align_corners(align_corners));
        }   
        else
        {
            std::cerr << "Choose either 'bilinear' or 'nearest'." << std::endl;
            exit(EXIT_FAILURE);
        }

        //reshape output to [B, N, C]
        return x.permute({0, 2, 3, 1}).squeeze(-2);
    }


    ////////////////////////////////  POST PROCESSING  /////////////////////////////////////
    XFDetector::XFDetector(int _top_k, float _detection_threshold, bool use_cuda) 
        : top_k(_top_k), 
          detection_threshold(_detection_threshold)
    {   
        // load the model
        std::string weights = "weights/xfeat.pt";
        model = std::make_shared<XFeatModel>();
        torch::serialize::InputArchive archive;
        archive.load_from(getWeightsPath(weights));
        model->load(archive);
        std::cout << "Model weights loaded successfully." << std::endl;

        // move the model to device
        device_type = (use_cuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU;
        torch::Device device(device_type);
        std::cout << "Device: " << device << '\n';
        model->to(device);

        // load the bilinear interpolator
        bilinear = std::make_shared<InterpolateSparse2d>("bilinear");     
        nearest  = std::make_shared<InterpolateSparse2d>("nearest"); 
    }

    void XFDetector::detect(cv::Mat& img)
    {   
        torch::Device device(device_type);

        torch::Tensor x;
        parseInput(img, x);
        x = x.to(device);

        float rh1, rw1;
        std::tie(x, rh1, rw1) = preprocessTensor(x);

        auto   B = x.size(0);
        auto _H1 = x.size(2);
        auto _W1 = x.size(3);

        // forward pass
        auto out = model->forward(x);
        torch::Tensor M1, K1, H1;
        std::tie(M1, K1, H1) = out;
        M1 = torch::nn::functional::normalize(M1, torch::nn::functional::NormalizeFuncOptions().dim(1));

        // convert logits to heatmap and extract keypoints
        torch::Tensor K1h = getKptsHeatmap(K1);
        torch::Tensor mkpts = NMS(K1h, 5);

        // compute reliability scores
        auto scores = (nearest->forward(K1h, mkpts, _H1, _W1) * bilinear->forward(H1, mkpts, _H1, _W1)).squeeze(-1);
        auto mask = torch::all(mkpts == 0, -1);
        scores.masked_fill_(mask, -1);

        // Select top-k features
        torch::Tensor idxs = scores.neg().argsort(-1, false);
        auto mkpts_x = mkpts.index({torch::indexing::Ellipsis, 0})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});
        auto mkpts_y = mkpts.index({torch::indexing::Ellipsis, 1})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});
        mkpts_x = mkpts_x.unsqueeze(-1);
        mkpts_y = mkpts_y.unsqueeze(-1);
        mkpts = torch::cat({mkpts_x, mkpts_y}, -1);
        scores = scores.gather(-1, idxs).index({torch::indexing::Slice(), torch::indexing::Slice(0, top_k)});

        // Interpolate descriptors at kpts positions
        torch::Tensor feats = bilinear->forward(M1, mkpts, _H1, _W1);

        // L2-Normalize
        feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));
        auto min_val = feats.min();
        auto max_val = feats.max();

        // correct kpt scale
        torch::Tensor scaling_factors = torch::tensor({rw1, rh1}, mkpts.options()).view({1, 1, -1});
        mkpts = mkpts * scaling_factors;

        keypoints.clear();
        descriptors.release();

        for (int b = 0; b < B; b++)
        {
            auto valid = scores[b] > 0;
            auto valid_keypoints = mkpts[b].index({valid});
            auto valid_scores = scores[b].index({valid});
            auto valid_descriptors = feats[b].index({valid}).to(torch::kCPU);

            for (int i = 0; i < valid_keypoints.size(0); i++)
            {
                float x = valid_keypoints[i][0].item<float>();
                float y = valid_keypoints[i][1].item<float>();
                float score = valid_scores[i].item<float>();
                keypoints.emplace_back(x, y, 1, -1, score);
            }

            int num_keypoints = valid_descriptors.size(0);
            cv::Mat desc_mat(cv::Size(64, num_keypoints), CV_32F);
            std::memcpy(desc_mat.data, valid_descriptors.data_ptr(), num_keypoints * 64 * sizeof(float));
            descriptors.push_back(desc_mat);
        }
    }   

    void XFDetector::getKeyPoints(float threshold, int iniX, int maxX, int iniY, int maxY, std::vector<cv::KeyPoint>& _keypoints)
    {
        _keypoints.clear();
        std::vector<float> desc_vec;
        
        for (int i = 0; i < keypoints.size(); ++i)
        {
            const auto& kp = keypoints[i];
            // Check if the keypoint's response is above the threshold and within the specified rectangle
            if (kp.response > threshold && kp.pt.x >= iniX && kp.pt.x <= maxX && kp.pt.y >= iniY && kp.pt.y <= maxY)
            {
                _keypoints.push_back(kp);
                auto desc_ptr = descriptors.ptr<float>(i); // Use ptr for accessing the descriptor data
                desc_vec.insert(desc_vec.end(), desc_ptr, desc_ptr + 64);
            }
        }

        if (!_keypoints.empty()) 
        {
            descriptors = cv::Mat(_keypoints.size(), 64, CV_32F, desc_vec.data()).clone();
        }
        else
        {
            descriptors.release();
        }
    }

    void XFDetector::computeDescriptors(const std::vector<cv::KeyPoint>& _keypoints, cv::Mat& _descriptors)
    {   
        if (!descriptors.empty())
        {
            _descriptors = descriptors.clone();
        }
        else
        {
            _descriptors.release();
        }
    }

    void XFDetector::parseInput(cv::Mat& img, torch::Tensor& tensor)
    {   
        torch::Device device(device_type);

        // if the image is grayscale
        if (img.channels() == 1)
        {
            tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 1}, torch::kByte);
            tensor = tensor.to(device);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return;
        }
        // if image is in RGB format
        else if (img.channels() == 3) 
        {   
            tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
            tensor = tensor.to(device);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return;
        }
        else
        {
            // If the image has an unsupported number of channels, throw an error
            std::cout << "Image channels: " << img.channels() << std::endl;
            throw std::invalid_argument("Unsupported number of channels in the input image."); 
        }
         
    }

    std::tuple<torch::Tensor, double, double> XFDetector::preprocessTensor(torch::Tensor& x)
    {   
        // ensure the tensor has the correct type
        x = x.to(torch::kFloat);

        // calculate new size divisible by 32
        int H = x.size(-2);
        int W = x.size(-1);
        int64_t _H = (H / 32) * 32;
        int64_t _W = (W / 32) * 32;

        // calculate resize ratios
        double rh = static_cast<double>(H) / _H;
        double rw = static_cast<double>(W) / _W;

        std::vector<int64_t> size_array = {_H, _W};
        x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions().size(size_array)
                                                                                                 .mode(torch::kBilinear)
                                                                                                 .align_corners(false));
        return std::make_tuple(x, rh, rw);
    }

    torch::Tensor XFDetector::getKptsHeatmap(torch::Tensor& kpts, float softmax_temp)
    {   
        torch::Tensor scores = torch::nn::functional::softmax(kpts * softmax_temp, torch::nn::functional::SoftmaxFuncOptions(1));
        scores = scores.index({torch::indexing::Slice(), torch::indexing::Slice(0, 64), torch::indexing::Slice(), torch::indexing::Slice()});

        int B = scores.size(0);
        int H = scores.size(2);
        int W = scores.size(3);

        // reshape and permute the tensor to form heatmap
        torch::Tensor heatmap = scores.permute({0, 2, 3, 1}).reshape({B, H, W, 8, 8});
        heatmap = heatmap.permute({0, 1, 3, 2, 4}).reshape({B, 1, H*8, W*8});
        return heatmap;
    }

    torch::Tensor XFDetector::NMS(torch::Tensor& x, int kernel_size)
    {   
        int B = x.size(0);
        int H = x.size(2);
        int W = x.size(3);
        int pad = kernel_size / 2;

        auto local_max = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1)
                                                                                                                      .padding(pad));
        auto pos = (x == local_max) & (x > detection_threshold);
        std::vector<torch::Tensor> pos_batched;
        for (int b = 0; b < pos.size(0); ++b) 
        {
            auto k = pos[b].nonzero();
            k = k.index({torch::indexing::Ellipsis, torch::indexing::Slice(1, torch::indexing::None)}).flip(-1);
            pos_batched.push_back(k);
        }

        int pad_val = 0;
        for (const auto& p : pos_batched) {
            pad_val = std::max(pad_val, static_cast<int>(p.size(0)));
        }
        
        torch::Tensor pos_tensor = torch::zeros({B, pad_val, 2}, torch::TensorOptions().dtype(torch::kLong).device(x.device()));
        for (int b = 0; b < B; ++b) {
            if (pos_batched[b].size(0) > 0) {
                pos_tensor[b].narrow(0, 0, pos_batched[b].size(0)) = pos_batched[b];
            }
        }

        return pos_tensor;
    }

    std::string XFDetector::getWeightsPath(std::string weights)
    {
        std::filesystem::path current_file = __FILE__;
        std::filesystem::path parent_dir = current_file.parent_path();
        std::filesystem::path full_path = parent_dir / ".." / weights;
        full_path = std::filesystem::absolute(full_path);

        return static_cast<std::string>(full_path);   
    }

} // namespace ORB_SLAM3 