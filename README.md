# xfeatSLAM

## Introduction

xfeatSLAM is a Visual SLAM system that integrates the lightweight and efficient [XFeat](https://github.com/verlab/accelerated_features) architecture into the [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) pipeline. This project is an experimental combination of ORB-SLAM3 with the XFeat model to create a SLAM system utilizing deep learning-based image descriptors.

Typically, models used to obtain deep learning-based local features provide accurate descriptions but are highly resource-intensive. The lightweight nature of XFeat makes it particularly well-suited for environments with limited processing power, such as mobile robots and embedded systems, where real-time performance is critical.

First, the entire XFeat model is implemented in C++ using PyTorch C++ API. I have separated the code for this implementation into [this repo](https://github.com/udaysankar01/xfeat_cpp). This implementation is then integrated into the ORB-SLAM3 pipeline, following an approach similar to that used in [GCNv2 SLAM](https://github.com/jiexiong2016/GCNv2_SLAM).

![xfeatSLAM in action](doc/xfeatSLAM_compressed.gif)

## Dependencies

### C++17 Compiler

xfeatSLAM uses libtorch and requires a compiler that supports at least C++17. It has been tested with GCC and G++ compilers v11.4.0.

### PyTorch

xfeatSLAM uses [PyTorch](https://github.com/pytorch/pytorch) C++ API (libtorch) for running the C++ Implementation of XFeat model.

Please avoid using the pre-built version of libtorch as it may cause linking issues ([CXX11 ABI issue](https://github.com/pytorch/pytorch/issues/13541))

### Pangolin

[Pangolin](https://github.com/stevenlovegrove/Pangolin) is used for visualization and UI.

### OpenCV

[OpenCV](http://opencv.org/) is used for image and feature processing.

The version of OpenCV used for testing is v4.5.4.

### Eigen3

[Eigen3](http://eigen.tuxfamily.org/) is requried for matrix operations.

Project was tested using Eigen3 v3.4.0.

### DBoW2, Sophus and g2o (included in _thirdparty_ folder)

The project use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition, [Sophus](https://github.com/strasdat/Sophus) for Lie groups, and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. All three modified libraries (which are BSD) are included in the _thirdparty_ folder.

## Getting Started

To download xfeatSLAM:

```bash
git clone https://github.com/udaysankar01/xfeatSLAM
cd xfeatSLAM
```

To install all the necessary packages:

```bash
chmod +x project_setup.sh
./project_setup.sh
```

To build the project:

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
```

## RGB-D Example

To run deep RGB-D SLAM:

```bash
./examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt \
                          examples/RGB-D/TUMX.yaml \
                          PATH_TO_SEQUENCE_FOLDER \
                          ASSOCIATIONS_FILE
```

To use ORB feature descriptor instead of XFeat, set the environment variable `USE_ORB` as 1.

```bash
USE_ORB=1 ./examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt \
                                    examples/RGB-D/TUMX.yaml \
                                    PATH_TO_SEQUENCE_FOLDER \
                                    ASSOCIATIONS_FILE
```

## Monocular Example

To run deep Monocular SLAM:

```bash
./examples/Monocular/mono_tum Vocabulary/ORBvoc.txt \
                              examples/Monocular/TUMX.yaml \
                              PATH_TO_SEQUENCE_FOLDER
```

Support for additional sensors will be added soon.

## Bibtex Citation

```tex
@misc{potje2024xfeatacceleratedfeatureslightweight,
      title={XFeat: Accelerated Features for Lightweight Image Matching},
      author={Guilherme Potje and Felipe Cadar and Andre Araujo and Renato Martins and Erickson R. Nascimento},
      year={2024},
      eprint={2404.19174},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.19174},
}
```

```tex
@article{ORBSLAM3_TRO,
  title={{ORB-SLAM3}: An Accurate Open-Source Library for Visual, Visual-Inertial
           and Multi-Map {SLAM}},
  author={Campos, Carlos AND Elvira, Richard AND G\´omez, Juan J. AND Montiel,
          Jos\'e M. M. AND Tard\'os, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={37},
  number={6},
  pages={1874-1890},
  year={2021}
 }
```

```tex
@unknown{unknown,
author = {Tang, Jiexiong and Ericson, Ludvig and Folkesson, John and Jensfelt, Patric},
year = {2019},
month = {02},
pages = {},
title = {GCNv2: Efficient Correspondence Prediction for Real-Time SLAM}
}
```
