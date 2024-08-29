cmake_minimum_required(VERSION 3.22.1)
project(xfeatSLAM)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -w  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

# Check C++17 support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)

if(COMPILER_SUPPORTS_CXX17)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++17.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# enable folder organization in IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# for libtorch
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/thirdparty/pytorch/torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message("-- Torch Version: " ${Torch_VERSION})

# find libraries
find_package(OpenCV REQUIRED)
message("-- OpenCV Version: " ${OpenCV_VERSION})
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# find_package(realsense2)
include_directories(
    ${EIGEN_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}

    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels

    ${PROJECT_SOURCE_DIR}/thirdparty/
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/
    ${PROJECT_SOURCE_DIR}/thirdparty/Sophus/
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
    src/System.cc
    src/Tracking.cc
    src/LocalMapping.cc
    src/LoopClosing.cc
    src/ORBextractor.cc
    src/ORBmatcher.cc
    src/FrameDrawer.cc
    src/Converter.cc
    src/MapPoint.cc
    src/KeyFrame.cc
    src/Atlas.cc
    src/Map.cc
    src/MapDrawer.cc
    src/Optimizer.cc
    src/Frame.cc
    src/KeyFrameDatabase.cc
    src/Sim3Solver.cc
    src/Viewer.cc
    src/ImuTypes.cc
    src/G2oTypes.cc
    src/CameraModels/Pinhole.cpp
    src/CameraModels/KannalaBrandt8.cpp
    src/OptimizableTypes.cpp
    src/MLPnPsolver.cpp
    src/GeometricTools.cc
    src/TwoViewReconstruction.cc
    src/Config.cc
    src/Settings.cc
    src/XFeat.cc
    src/XFextractor.cc

    include/System.h
    include/Tracking.h
    include/LocalMapping.h
    include/LoopClosing.h
    include/ORBextractor.h
    include/ORBmatcher.h
    include/FrameDrawer.h
    include/Converter.h
    include/MapPoint.h
    include/KeyFrame.h
    include/Atlas.h
    include/Map.h
    include/MapDrawer.h
    include/Optimizer.h
    include/Frame.h
    include/KeyFrameDatabase.h
    include/Sim3Solver.h
    include/Viewer.h
    include/ImuTypes.h
    include/G2oTypes.h
    include/CameraModels/GeometricCamera.h
    include/CameraModels/Pinhole.h
    include/CameraModels/KannalaBrandt8.h
    include/OptimizableTypes.h
    include/MLPnPsolver.h
    include/GeometricTools.h
    include/TwoViewReconstruction.h
    include/SerializationUtils.h
    include/Config.h
    include/Settings.h
    include/XFeat.h
    include/XFextractor.h
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${TORCH_LIBRARIES}

    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib/libDBoW2${CMAKE_SHARED_LIBRARY_SUFFIX}
    -lboost_serialization
    -lcrypto
)

# ########## Build Examples ###########

# RGB-D example
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/RGB-D)

add_executable(rgbd_tum examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

# Stereo Inertial example
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/Stereo-Inertial)

add_executable(stereo_inertial_tum_vi examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

# test examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/test)

add_executable(test_extractor examples/test/test_extractor.cc)
target_link_libraries(test_extractor ${PROJECT_NAME})

add_executable(test_stereo_xf examples/test/test_stereo_xf.cc)
target_link_libraries(test_stereo_xf ${PROJECT_NAME})

add_executable(test_stereo_orb examples/test/test_stereo_orb.cc)
target_link_libraries(test_stereo_orb ${PROJECT_NAME})

add_executable(test_matcher_xf examples/test/test_matcher_xf.cc)
target_link_libraries(test_matcher_xf ${PROJECT_NAME})

add_executable(test_matcher_orb examples/test/test_matcher_orb.cc)
target_link_libraries(test_matcher_orb ${PROJECT_NAME})