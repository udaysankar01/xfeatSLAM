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

#ifndef XFEXTRACTOR_H
#define XFEXTRACTOR_H

#include "XFeat.h"
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include "torch/torch.h"


namespace ORB_SLAM3
{

class XFextractor
{
public:
    
    XFextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~XFextractor(){}

    int operator()( cv::InputArray _image, cv::InputArray _mask,
                    std::vector<cv::KeyPoint>& _keypoints,
                    cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

protected:
    std::string getModelWeightsPath(std::string weights);
    torch::Tensor parseInput(cv::Mat& img);
    std::tuple<torch::Tensor, double, double> preprocessTensor(torch::Tensor& x);
    torch::Tensor getKptsHeatmap(torch::Tensor& kpts, float softmax_temp=1.0);
    torch::Tensor NMS(torch::Tensor& x, float threshold = 0.05, int kernel_size = 5);

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    torch::DeviceType device_type;
    std::shared_ptr<XFeatModel> model;
    std::shared_ptr<InterpolateSparse2d> bilinear, nearest;
};

} //namespace ORB_SLAM

#endif

