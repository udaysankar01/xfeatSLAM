#!/usr/bin/env bash

# install all required dependencies
sudo apt-get update
grep -v '^\s*#' dependencies.txt | grep -v '^\s*$' | cut -d'#' -f1 | xargs sudo apt-get install -y

# setup libtorch
cd thirdparty
git clone https://github.com/pytorch/pytorch
cd pytorch
git submodule update --init --recursive
git submodule sync
python tools/build_libtorch.py
cd ../..

# setup g2o
cd thirdparty/g2o
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../../..

# setup DBoW2
cd thirdparty/DBoW2
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../../..

# setup Sophus
cd thirdparty/Sophus
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../../..


# setup Pangolin
cd thirdparty
git clone https://github.com/stevenlovegrove/Pangolin
cd Pangolin
./scripts/install_prerequisites.sh recommended
mkdir -p build
cd build
cmake ..
make -j8
# prompt user for 'sudo make install'
read -p "Do you want to proceed with 'sudo make install' for Pangolin? [Y/n]: " choice
if [[ "$choice" == "Y" || "$choice" == "y" ]]; then
    sudo make install
elif [[ "$choice" == "N" || "$choice" == "n" ]]; then
    echo "Project CMake configuration might not work without 'sudo make install'."
else
    echo "Choose either of these: [Y, y, N, n]"
fi
cd ../../..