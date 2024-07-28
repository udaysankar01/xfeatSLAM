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


    ////////////////////////////////  POST PROCESSING  /////////////////////////////////////
    XFDetector::XFDetector(int _top_k, float _detection_threshold, bool use_cuda) 
        : top_k(_top_k), 
          detection_threshold(_detection_threshold)
    {   
        // load the model
        weights = "weights/xfeat.pt";
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

        // cosine similarity matching threshold
        min_cossim = -1;
    }

    std::vector<std::unordered_map<std::string, torch::Tensor>> XFDetector::detectAndCompute(torch::Tensor& x)
    {   
        torch::Device device(device_type);
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
        torch::Tensor mkpts = NMS(K1h, detection_threshold, 5);

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

        std::vector<std::unordered_map<std::string, torch::Tensor>> result;
        for (int b = 0; b < B; b++) {
            auto valid = scores[b] > 0;
            std::unordered_map<std::string, torch::Tensor> item;
            item["keypoints"] = mkpts[b].index({valid});
            item["scores"] = scores[b].index({valid});
            item["descriptors"] = feats[b].index({valid});
            result.push_back(item);
        }
        return result;
    }   

    std::tuple<torch::Tensor, torch::Tensor> XFDetector::match(torch::Tensor& feats1, torch::Tensor& feats2, float _min_cossim)
    {   
        // set the min_cossim
        min_cossim = _min_cossim;

        // compute cossine similarity between feats1 and feats2
        torch::Tensor cossim = torch::matmul(feats1, feats2.t());
        torch::Tensor cossim_t = torch::matmul(feats2, feats1.t());

        torch::Tensor match12, match21;
        std::tie(std::ignore, match12) = cossim.max(1);
        std::tie(std::ignore, match21) = cossim_t.max(1);

        // index tensor
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

    std::pair<cv::Mat, cv::Mat> XFDetector::match_xfeat(cv::Mat& img1, cv::Mat& img2)
    {   
        torch::Tensor tensor_img1 = parseInput(img1);
        torch::Tensor tensor_img2 = parseInput(img2);

        auto out1 = detectAndCompute(tensor_img1)[0]; // no batches
        auto out2 = detectAndCompute(tensor_img2)[0];

        torch::Tensor idxs0, idxs1;
        std::tie(idxs0, idxs1) = match(out1["descriptors"], out2["descriptors"], -1.0);

        torch::Tensor mkpts_0 = out1["keypoints"].index({idxs0});
        torch::Tensor mkpts_1 = out2["keypoints"].index({idxs1});
        cv::Mat mkpts_0_cv = tensorToMat(mkpts_0);
        cv::Mat mkpts_1_cv = tensorToMat(mkpts_1);

        return std::make_pair(mkpts_0_cv, mkpts_1_cv);
    }

    torch::Tensor XFDetector::parseInput(cv::Mat &img)
    {   
        // if the image is grayscale
        if (img.channels() == 1)
        {
            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 1}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return tensor;
        }

        // if image is in RGB format
        if (img.channels() == 3) {
            torch::Tensor tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
            tensor = tensor.permute({0, 3, 1, 2}).to(torch::kFloat) / 255.0;
            return tensor;
        }

        // If the image has an unsupported number of channels, throw an error
        throw std::invalid_argument("Unsupported number of channels in the input image.");  
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

    torch::Tensor XFDetector::NMS(torch::Tensor& x, float threshold, int kernel_size)
    {   
        int B = x.size(0);
        int H = x.size(2);
        int W = x.size(3);
        int pad = kernel_size / 2;

        auto local_max = torch::nn::functional::max_pool2d(x, torch::nn::functional::MaxPool2dFuncOptions(kernel_size).stride(1)
                                                                                                                      .padding(pad));
        auto pos = (x == local_max) & (x > threshold);
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

    cv::Mat XFDetector::tensorToMat(const torch::Tensor& tensor)
    {   
        // ensure tesnor is on CPU and convert to float
        torch::Tensor cpu_tensor = tensor.to(torch::kCPU).to(torch::kFloat);
        cv::Mat mat(cpu_tensor.size(0), 2, CV_32F);
        std::memcpy(mat.data, cpu_tensor.data_ptr<float>(), cpu_tensor.numel() * sizeof(float));
        return mat;
    }

    std::string XFDetector::getWeightsPath(std::string weights)
    {
        std::filesystem::path current_file = __FILE__;
        std::filesystem::path parent_dir = current_file.parent_path();
        std::filesystem::path full_path = parent_dir / ".." / weights;
        full_path = std::filesystem::absolute(full_path);

        return static_cast<std::string>(full_path);   
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


} // namespace ORB_SLAM3 