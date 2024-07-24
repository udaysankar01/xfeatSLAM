cmake_minimum_required(VERSION 3.22.1)
project(xfeatSLAM)

# Build type:
# - Set to RELEASE to benchmark the code
# - Set to DEBUG to debug the code
set(CMAKE_BUILD_TYPE Debug)

# enable folder organization in IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# for libtorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/thirdparty/pytorch/torch")

# find libraries
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)

# libtorch config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${EIGEN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}

    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/

    # ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
    # ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/
)

set(THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}

    # ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o${CMAKE_SHARED_LIBRARY_SUFFIX}
    # ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/build/src/libDBoW3${CMAKE_SHARED_LIBRARY_SUFFIX}
)

add_subdirectory(src)
add_subdirectory(examples)