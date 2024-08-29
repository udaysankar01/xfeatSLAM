# xfeatSLAM

## Introduction

xfeatSLAM is a Visual SLAM system that integrates the lightweight and efficient XFeat architecture into the ORB-SLAM3 pipeline. This project is an experimental combination of ORB-SLAM3 with the XFeat model to create a SLAM system utilizing deep learning-based image descriptors.

Typically, models used to obtain deep learning-based local features provide accurate descriptions but are highly resource-intensive. The lightweight nature of XFeat makes it particularly well-suited for environments with limited processing power, such as mobile robots and embedded systems, where real-time performance is critical.

## Getting Started

To download xfeatSLAM:

```bash
git clone https://github.com/udaysankar01/xfeatSLAM
cd xfeatSLAM
```

To install the necessary packages:

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

Right now, xfeatSLAM supports RGB-D SLAM. To run:

```bash
./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt \
                          examples/RGB-D/TUMX.yaml \
                          PATH_TO_SEQUENCE_FOLDER \
                          ASSOCIATIONS_FILE
```
