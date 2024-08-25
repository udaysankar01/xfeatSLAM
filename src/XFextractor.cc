/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

#include "XFextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;

    XFextractor::XFextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }
        
        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
        
        // load the xfeat model
        std::string weights = "weights/xfeat.pt";
        model = std::make_shared<XFeatModel>();
        torch::serialize::InputArchive archive;
        archive.load_from(getModelWeightsPath(weights));
        model->load(archive);
        std::cout << "XFeat model weights loaded successfully!" << std::endl;

        // move the model to device
        device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        torch::Device device(device_type);
        std::cout << "Device: " << device << std::endl;
        model->to(device);

        // load the interpolators
        bilinear = std::make_shared<InterpolateSparse2d>("bilinear");     
        nearest  = std::make_shared<InterpolateSparse2d>("nearest"); 
    }

    std::string XFextractor::getModelWeightsPath(std::string weights)
    {
        std::filesystem::path current_file = __FILE__;
        std::filesystem::path parent_dir = current_file.parent_path();
        std::filesystem::path full_path = parent_dir / ".." / weights;
        full_path = std::filesystem::absolute(full_path);

        return static_cast<std::string>(full_path);   
    }

    torch::Tensor XFextractor::parseInput(cv::Mat &img)
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

    std::tuple<torch::Tensor, double, double> XFextractor::preprocessTensor(torch::Tensor& x)
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

    torch::Tensor XFextractor::getKptsHeatmap(torch::Tensor& kpts, float softmax_temp)
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

    torch::Tensor XFextractor::NMS(torch::Tensor& x, float threshold, int kernel_size)
    {   
        int B = x.size(0);
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

    int XFextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        if(_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );
        torch::Tensor x = parseInput(image);

        torch::Device device(device_type);
        x = x.to(device);

        float rh1, rw1;
        std::tie(x, rh1, rw1) = preprocessTensor(x);

        auto _H1 = x.size(2);
        auto _W1 = x.size(3);

        // forward pass
        auto out = model->forward(x);
        torch::Tensor M1, K1, H1;
        std::tie(M1, K1, H1) = out;
        M1 = torch::nn::functional::normalize(M1, torch::nn::functional::NormalizeFuncOptions().dim(1));

        // convert logits to heatmap and extract keypoints
        torch::Tensor K1h = getKptsHeatmap(K1);
        torch::Tensor mkpts = NMS(K1h, 0.05, 5);

        // compute reliability scores
        auto scores = (nearest->forward(K1h, mkpts, _H1, _W1) * bilinear->forward(H1, mkpts, _H1, _W1)).squeeze(-1);
        auto mask = torch::all(mkpts == 0, -1);
        scores.masked_fill_(mask, -1);

        // Select top-k features
        torch::Tensor idxs = scores.neg().argsort(-1, false);
        auto mkpts_x = mkpts.index({torch::indexing::Ellipsis, 0})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, nfeatures)});
        auto mkpts_y = mkpts.index({torch::indexing::Ellipsis, 1})
                            .gather(-1, idxs)
                            .index({torch::indexing::Slice(), torch::indexing::Slice(0, nfeatures)});
        mkpts_x = mkpts_x.unsqueeze(-1);
        mkpts_y = mkpts_y.unsqueeze(-1);
        mkpts = torch::cat({mkpts_x, mkpts_y}, -1);
        scores = scores.gather(-1, idxs).index({torch::indexing::Slice(), torch::indexing::Slice(0, nfeatures)});

        // Interpolate descriptors at kpts positions
        torch::Tensor feats = bilinear->forward(M1, mkpts, _H1, _W1);

        // L2-Normalize
        feats = torch::nn::functional::normalize(feats, torch::nn::functional::NormalizeFuncOptions().dim(-1));

        // correct kpt scale
        torch::Tensor scaling_factors = torch::tensor({rw1, rh1}, mkpts.options()).view({1, 1, -1});
        mkpts = mkpts * scaling_factors;

        // Prepare keypoints and descriptors
        // _keypoints.clear();
        // _keypoints.reserve(nfeatures);  // Reserve space for keypoints
        _keypoints = vector<cv::KeyPoint>(nfeatures);
        Mat desc_mat(cv::Size(64, nfeatures), CV_32F, cv::Scalar(0));  // Initialize descriptor matrix

        auto valid = scores[0] > 0;
        auto valid_keypoints = mkpts[0].index({valid});
        auto valid_scores = scores[0].index({valid});
        auto valid_descriptors = feats[0].index({valid}).to(torch::kCPU);

        // testing
        int monoIndex = 0, stereoIndex = nfeatures - 1;
        Mat desc = cv::Mat(nfeatures, 64, CV_32F);
        for (int i = 0; i < valid_keypoints.size(0); i++) 
        {
            float x = valid_keypoints[i][0].item<float>();
            float y = valid_keypoints[i][1].item<float>();
            float score = valid_scores[i].item<float>();
            KeyPoint keypoint(x, y, 1, -1, score);

            cv::Mat desc_row(cv::Size(64, 1), CV_32F);       
            std::memcpy(desc_row.data, valid_descriptors[i].data_ptr(), 64 * sizeof(float));     

            if (x >= vLappingArea[0] && x <= vLappingArea[1])
            {
                _keypoints.at(stereoIndex) = keypoint;
                desc_row.copyTo(desc_mat.row(stereoIndex));
                stereoIndex--;
            }
            else
            {
                _keypoints.at(monoIndex) = keypoint;
                desc_row.copyTo(desc_mat.row(monoIndex));
                monoIndex++;
            }            
        }

        int num_keypoints = valid_descriptors.size(0);
        
        if (num_keypoints > 0) 
        {   
            desc_mat.rowRange(cv::Range(0, _keypoints.size())).copyTo(_descriptors);
        } 
        else 
        {
            _descriptors.release();
        }

        return monoIndex;
        
        // // NO FISHEYE STEREO VERSION
        // for (int i = 0; i < valid_keypoints.size(0); i++) 
        // {
        //     float x = valid_keypoints[i][0].item<float>();
        //     float y = valid_keypoints[i][1].item<float>();
        //     float score = valid_scores[i].item<float>();
        //     _keypoints.emplace_back(x, y, 1, -1, score);
        // }
        // int num_keypoints = valid_descriptors.size(0);
        // if (num_keypoints > 0) 
        // {   
        //     cv::Mat desc_mat(cv::Size(num_keypoints, 64), CV_32F);
        //     std::memcpy(desc_mat.data, valid_descriptors.data_ptr(), num_keypoints * 64 * sizeof(float));
        //     desc_mat.copyTo(_descriptors);
        // } 
        // else 
        // {
        //     _descriptors.release();
        // }
        // return 0;
    }

} //namespace ORB_SLAM
